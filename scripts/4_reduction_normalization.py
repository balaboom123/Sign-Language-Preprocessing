#!/usr/bin/env python3
"""
Apply landmark reduction and normalization following paper methodology.

This script processes raw landmark data from Step 3 and applies:
1. Visibility masking: set (x, y, z) to -999 if visibility < threshold
2. Whole-clip isotropic unit bounding box normalization: scale to fit in 1×1×1 cube
3. Remove visibility channel from output

Paper Reference (YouTube-ASL):
    "We normalize the landmarks by scaling them to fit in a unit bounding box
    across the duration of the clip. We represent landmarks that are not present
    in a frame with a large negative value. MediaPipe also predicts visibility
    (self-occlusion) of landmarks within the frame, which we ignore."

Implementation Details:
    - Isotropic scaling: Single scale factor for x, y, z to preserve aspect ratios
    - Clip-wise normalization: Compute bounding box across entire video clip
    - Sentinel values: Missing landmarks represented as -999.0 (large negative value)

Usage:
    python scripts/4_reduction_normalization.py

Configuration:
    Edit configs/reduction_normalization.py to change:
    - INPUT_DIR: Directory with raw landmarks from Step 3
    - OUTPUT_DIR: Directory to save normalized landmarks
    - VISIBILITY_THRESHOLD: Threshold for masking invisible landmarks
    - UNVISIBLE_FRAME / UNVISIBLE_LANDMARK: Sentinel values (-999.0)
    - REMOVE_Z: Whether to drop z-coordinate (default: False)

Input Format:
    NumPy arrays (.npy) with shape (T, 85, 4) where:
    - T: Number of frames
    - 85: Total keypoints
    - 4: [x, y, z, visibility] per keypoint

Output Format:
    NumPy arrays (.npy) with shape (T, 255) or (T, 170) where:
    - T: Number of frames (unchanged)
    - 255: Flattened normalized landmarks (85 keypoints × 3 coords)
    - 170: If REMOVE_Z=True (85 keypoints × 2 coords)
    - Valid values in [0, 1], masked points = -999.0
"""
import os
import sys
import glob
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configs.reduction_normalization as cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_clip(path: str) -> np.ndarray:
    """
    Load raw landmark clip from .npy file.

    Accepts two input formats:
    - (T, 85, 4): Standard format with x, y, z, visibility
    - (T, 340): Flattened format (85 × 4)

    Args:
        path: Path to .npy file

    Returns:
        Array with shape (T, 85, 4) where:
        - T: Number of frames
        - 85: Number of keypoints
        - 4: [x, y, z, visibility] per keypoint

    Raises:
        ValueError: If array shape is not compatible
    """
    arr = np.load(path).astype(np.float32)

    # Handle different input formats
    if arr.ndim == 2:
        T, F = arr.shape
        if F == 340:  # Flattened (T, 85*4)
            arr = arr.reshape(T, 85, 4)
        elif F == 255:  # Old format without visibility
            raise ValueError(
                f"Input file {path} has shape {arr.shape} (255 features). "
                "This script requires raw landmarks with visibility (shape: T, 85, 4)."
            )
        else:
            raise ValueError(
                f"Unsupported shape {arr.shape} in {path}. "
                "Expected (T, 85, 4) or (T, 340)."
            )
    elif arr.ndim == 3:
        T, K, C = arr.shape
        if K != 85 or C != 4:
            raise ValueError(
                f"Expected shape (T, 85, 4), got {arr.shape} in {path}"
            )
    else:
        raise ValueError(
            f"Unexpected ndim={arr.ndim} for {path}. Expected 2 or 3."
        )

    return arr


def apply_visibility_mask(clip_xyzv: np.ndarray) -> np.ndarray:
    """
    Apply two-level visibility masking to landmarks.

    This function implements the paper's "large negative value" approach for
    missing landmarks by using sentinel values (-999.0).

    Frame-level masking (if MASK_FRAME_LEVEL enabled):
    - Detects placeholder frames where all values are exactly 0.0
    - Sets entire frame coordinates to UNVISIBLE_FRAME

    Landmark-level masking (if MASK_LANDMARK_LEVEL enabled):
    - Sets individual landmark coordinates to UNVISIBLE_LANDMARK when:
      - visibility < VISIBILITY_THRESHOLD
      - OR all coordinates (x, y, z, vis) are zero

    Args:
        clip_xyzv: Array with shape (T, K, 4) = [x, y, z, visibility]
                   T = number of frames, K = 85 keypoints

    Returns:
        Array with shape (T, K, 3) containing only [x, y, z]
        Masked coordinates set to sentinel values (UNVISIBLE_FRAME or UNVISIBLE_LANDMARK)
    """
    T, K, _ = clip_xyzv.shape
    xyz = clip_xyzv[..., :3].copy()  # (T, K, 3)
    vis = clip_xyzv[..., 3]          # (T, K)

    # Frame-level masking: detect and mask entire placeholder frames
    if cfg.MASK_FRAME_LEVEL:
        # A frame is a placeholder if ALL values (xyz + visibility) are zero
        frame_all_zero = np.all(clip_xyzv == 0.0, axis=(1, 2))  # (T,)

        # Set all landmarks in placeholder frames to UNVISIBLE_FRAME
        for t in range(T):
            if frame_all_zero[t]:
                xyz[t, :, :] = cfg.UNVISIBLE_FRAME

    # Landmark-level masking: mask individual low-visibility landmarks
    if cfg.MASK_LANDMARK_LEVEL:
        # Condition 1: visibility below threshold
        low_vis_mask = vis < cfg.VISIBILITY_THRESHOLD

        # Condition 2: all coordinates are zero for this landmark (detector failed)
        all_zero_mask = np.all(clip_xyzv == 0.0, axis=-1)  # (T, K)

        # Combine conditions
        missing_mask = np.logical_or(low_vis_mask, all_zero_mask)

        # Apply masking: set (x,y,z) to UNVISIBLE_LANDMARK
        xyz[missing_mask] = cfg.UNVISIBLE_LANDMARK

    return xyz  # (T, K, 3)


def normalize_clip_xyz(xyz_masked: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks using whole-clip isotropic unit bounding box scaling.

    Paper methodology:
    "We normalize the landmarks by scaling them to fit in a unit bounding box
    across the duration of the clip."

    Implementation:
    This function treats each (x, y, z) triplet as a 3D point and computes
    a single 3D bounding box across the entire video clip. Unlike per-axis
    normalization, this approach uses ONE scale factor for all dimensions,
    preserving aspect ratios of the signer's motion.

    Steps:
    1. Collect all valid 3D points (exclude sentinel values -999.0)
    2. Compute 3D bounding box:
       coord_min = [min_x, min_y, min_z]
       coord_max = [max_x, max_y, max_z]
    3. Find maximum range: max_range = max(x_range, y_range, z_range)
    4. Apply isotropic scaling: p_norm = (p - coord_min) / max_range
    5. Result fits in 1×1×1 cube with preserved aspect ratios

    Sentinel values (UNVISIBLE_FRAME, UNVISIBLE_LANDMARK) remain at -999.0
    to match the paper's "large negative value" representation.

    Args:
        xyz_masked: Array with shape (T, K, 3) containing [x, y, z]
                    May include sentinel values -999.0 for missing data

    Returns:
        Normalized array with shape (T, K, 3)
        Valid coordinates in [0, 1], sentinel values unchanged at -999.0

    Raises:
        ValueError: If input shape is not (T, K, 3)
    """
    if xyz_masked.ndim != 3 or xyz_masked.shape[-1] != 3:
        raise ValueError(
            f"Expected xyz_masked with shape (T, K, 3), got {xyz_masked.shape}"
        )

    out = xyz_masked.copy()

    # Identify invalid coordinates (either sentinel value)
    # A coordinate is invalid if it equals UNVISIBLE_FRAME or UNVISIBLE_LANDMARK
    invalid_coord = np.logical_or(
        xyz_masked == cfg.UNVISIBLE_FRAME,
        xyz_masked == cfg.UNVISIBLE_LANDMARK,
    )  # shape (T, K, 3)

    # A 3D point is valid only if ALL three coordinates (x, y, z) are valid
    valid_points_mask = ~np.any(invalid_coord, axis=-1)  # shape (T, K)

    # Edge case: clip has no valid landmarks at all
    if not np.any(valid_points_mask):
        logger.warning(
            "All landmarks are invalid in this clip, skipping normalization."
        )
        return out

    # Collect all valid 3D points as an (N, 3) array
    valid_coords = xyz_masked[valid_points_mask]  # shape (N, 3)

    # Compute 3D bounding box across entire clip
    coord_min = valid_coords.min(axis=0)  # shape (3,) -> [min_x, min_y, min_z]
    coord_max = valid_coords.max(axis=0)  # shape (3,) -> [max_x, max_y, max_z]
    coord_range = coord_max - coord_min   # shape (3,) -> [range_x, range_y, range_z]

    # Find maximum range across all dimensions for isotropic scaling
    max_range = float(np.max(coord_range))

    # Edge case: all valid points are at the same location (zero extent)
    if np.isclose(max_range, 0.0):
        logger.warning(
            "Bounding box has zero extent; setting all valid points to 0.0."
        )
        scaled_valid = np.zeros_like(valid_coords, dtype=np.float32)
    else:
        # Apply isotropic scaling: uses same scale factor for x, y, z
        # This preserves aspect ratios of the signer's motion
        scaled_valid = (valid_coords - coord_min) / max_range
        scaled_valid = scaled_valid.astype(np.float32)

    # Write normalized values back to output array
    # Sentinel values remain unchanged at -999.0
    out[valid_points_mask] = scaled_valid

    return out  # shape (T, K, 3)


def maybe_drop_z(xyz_norm: np.ndarray, remove_z: bool) -> np.ndarray:
    """
    Optionally remove z-coordinate from landmarks.

    Useful for 2D-only models or when depth information from monocular
    video is unreliable. Reduces storage by 33%.

    Args:
        xyz_norm: Array with shape (T, K, 3) containing [x, y, z]
        remove_z: If True, drop z-coordinate

    Returns:
        Array with shape (T, K, 2) if remove_z=True (only [x, y])
        Array with shape (T, K, 3) if remove_z=False (unchanged)
    """
    if remove_z:
        return xyz_norm[..., :2]  # Keep only (x, y)
    return xyz_norm


def flatten_per_frame(xyz_final: np.ndarray) -> np.ndarray:
    """
    Flatten landmarks from (T, K, C) to (T, K×C) for model input.

    Converts 3D landmark array to 2D array where each frame's landmarks
    are flattened into a single feature vector.

    Args:
        xyz_final: Array with shape (T, K, C) where:
                   - T: Number of frames
                   - K: Number of keypoints (85)
                   - C: Coordinates per keypoint (2 or 3)

    Returns:
        Flattened array with shape (T, K×C):
        - (T, 255) if C=3 (x, y, z)
        - (T, 170) if C=2 (x, y)
    """
    T, K, C_out = xyz_final.shape
    return xyz_final.reshape(T, K * C_out).astype(np.float32)


def process_file(input_path: str, output_path: str) -> Tuple[str, bool, str]:
    """
    Process a single landmark file through the complete normalization pipeline.

    Pipeline stages:
    1. Load raw landmarks (T, 85, 4)
    2. Apply visibility masking → (T, 85, 3) with sentinel values
    3. Normalize using isotropic unit bounding box → (T, 85, 3) in [0, 1]
    4. Optionally drop z-coordinate → (T, 85, 2) or (T, 85, 3)
    5. Flatten per frame → (T, 170) or (T, 255)
    6. Save normalized landmarks

    Args:
        input_path: Path to input .npy file with shape (T, 85, 4)
        output_path: Path to output .npy file with shape (T, 255) or (T, 170)

    Returns:
        Tuple of (filename, success, error_message):
        - filename: Base name of the processed file
        - success: True if processed successfully, False otherwise
        - error_message: Error description or "skipped (exists)" or empty string
    """
    filename = os.path.basename(input_path)

    try:
        # Skip if output already exists and SKIP_EXISTING is enabled
        if cfg.SKIP_EXISTING and os.path.exists(output_path):
            return filename, True, "skipped (exists)"

        # Step 1: Load raw landmarks with visibility
        clip_raw = load_clip(input_path)  # Shape: (T, 85, 4)

        # Step 2: Apply visibility masking
        # Convert low-visibility or missing landmarks to sentinel values (-999.0)
        xyz_masked = apply_visibility_mask(clip_raw)  # Shape: (T, 85, 3)

        # Step 3: Whole-clip isotropic unit bounding box normalization
        # Scale all valid coordinates to fit in 1×1×1 cube while preserving aspect ratios
        xyz_norm = normalize_clip_xyz(xyz_masked)  # Shape: (T, 85, 3), values in [0,1]

        # Step 4: Optionally remove z-coordinate (for 2D-only models)
        xyz_final = maybe_drop_z(xyz_norm, cfg.REMOVE_Z)  # Shape: (T, 85, 2) or (T, 85, 3)

        # Step 5: Flatten per frame for model input
        flattened = flatten_per_frame(xyz_final)  # Shape: (T, 170) or (T, 255)

        # Step 6: Save normalized landmarks to output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, flattened)

        return filename, True, ""

    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return filename, False, str(e)


def main():
    """
    Main function to orchestrate landmark normalization pipeline.

    Processes all .npy files in INPUT_DIR using parallel workers,
    applying visibility masking and isotropic unit bounding box normalization.
    """
    logger.info("=" * 80)
    logger.info("ASL Dataset Preprocessor - Step 4: Landmark Reduction & Normalization")
    logger.info("=" * 80)

    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Find all .npy files in input directory
    pattern = os.path.join(cfg.INPUT_DIR, "**", "*.npy")
    npy_files = glob.glob(pattern, recursive=True)

    if not npy_files:
        logger.error(f"No .npy files found in {cfg.INPUT_DIR}")
        return

    logger.info(f"\nConfiguration:")
    logger.info(f"  - Input directory: {cfg.INPUT_DIR}")
    logger.info(f"  - Output directory: {cfg.OUTPUT_DIR}")
    logger.info(f"  - Frame-level masking: {cfg.MASK_FRAME_LEVEL}")
    logger.info(f"  - Landmark-level masking: {cfg.MASK_LANDMARK_LEVEL}")
    logger.info(f"  - Visibility threshold: {cfg.VISIBILITY_THRESHOLD}")
    logger.info(f"  - Unvisible frame value: {cfg.UNVISIBLE_FRAME}")
    logger.info(f"  - Unvisible landmark value: {cfg.UNVISIBLE_LANDMARK}")
    logger.info(f"  - Remove z-coordinate: {cfg.REMOVE_Z}")
    logger.info(f"  - Normalization method: Whole-clip isotropic unit bounding box")
    logger.info(f"  - Max workers: {cfg.MAX_WORKERS}")
    logger.info(f"  - Skip existing: {cfg.SKIP_EXISTING}\n")

    logger.info(f"Found {len(npy_files)} files to process\n")

    # Build task list: (input_path, output_path)
    tasks = []
    for input_path in npy_files:
        # Preserve directory structure in output
        rel_path = os.path.relpath(input_path, cfg.INPUT_DIR)
        output_path = os.path.join(cfg.OUTPUT_DIR, rel_path)
        tasks.append((input_path, output_path))

    # Process files in parallel
    success_count = 0
    skip_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_file, inp, out): (inp, out)
            for inp, out in tasks
        }

        # Process results with progress bar
        with tqdm(total=len(tasks), desc="Normalizing landmarks") as pbar:
            for future in as_completed(futures):
                filename, success, message = future.result()

                if success:
                    if message == "skipped (exists)":
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    error_count += 1
                    logger.error(f"Failed: {filename} - {message}")

                pbar.update(1)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Processing Complete")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(tasks)}")
    logger.info(f"  - Successfully processed: {success_count}")
    logger.info(f"  - Skipped (already exist): {skip_count}")
    logger.info(f"  - Errors: {error_count}")
    logger.info(f"\nOutput saved to: {cfg.OUTPUT_DIR}")

    if error_count > 0:
        logger.warning(f"\n{error_count} files failed to process. Check logs above.")


if __name__ == "__main__":
    main()
