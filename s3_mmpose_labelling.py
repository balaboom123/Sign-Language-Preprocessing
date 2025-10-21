import gc
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from typing import List, Optional, Tuple

import cv2
import multiprocessing
import numpy as np
import pandas as pd
import psutil

from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from models.rtmpose3d import *  # noqa: F401, F403
import conf as c

"""
MMPose-based 3D Pose Keypoint Extraction Pipeline

This module extracts 3D pose keypoints from video segments using MMPose RTMPose3D models.
It processes video segments defined in a CSV file and outputs landmark arrays.

Architecture:
    Main Process
        ├─> Read CSV with video segments
        ├─> Validate videos and build task list
        └─> ProcessPoolExecutor (MAX_WORKERS workers)
                ├─> Worker 1: init_worker() [loads models ONCE]
                │       └─> process_batch() → process_video_segment() × N
                ├─> Worker 2: init_worker() [loads models ONCE]
                │       └─> process_batch() → process_video_segment() × N
                └─> Worker N: init_worker() [loads models ONCE]
                        └─> process_batch() → process_video_segment() × N

Key Optimization:
    Models are loaded ONCE per worker process via init_worker() initializer.
    ProcessPoolExecutor is created OUTSIDE the batch loop to persist workers.
    This avoids repeated checkpoint loading (200+ MB per load) for each video segment.

    Without optimization: 100 segments → 100 model loads
    With per-batch optimization: 100 segments / 10 batches → 10 × MAX_WORKERS loads
    With full optimization: 100 segments → MAX_WORKERS loads (regardless of batch count)

Output Format:
    - Shape: (T, 340) if ADD_VISIBLE=True → 85 keypoints × (x, y, z, visible)
    - Shape: (T, 255) if ADD_VISIBLE=False → 85 keypoints × (x, y, z)
    - Keypoints: 6 pose + 37 face + 42 hands from COCO-WholeBody format
    - Coordinates: x,y normalized to [0,1], z rebased to shoulder reference
"""

# ==== Configuration ====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Global Model Cache for Worker Processes ====
# These are initialized once per worker process to avoid repeated checkpoint loading
_detector = None
_pose_estimator = None


def init_worker():
    """
    Initializes MMDet detector and MMPose estimator once per worker process.
    This function is called by ProcessPoolExecutor when each worker starts.
    Models are cached in global variables to avoid repeated checkpoint loading.
    """
    global _detector, _pose_estimator

    try:
        # Initialize MMDet detector
        _detector = init_detector(
            c.DET_MODEL_CONFIG,
            c.DET_MODEL_CHECKPOINT,
            device='cuda:0'
        )
        _detector.cfg = adapt_mmdet_pipeline(_detector.cfg)

        # Initialize MMPose 3D pose estimator
        _pose_estimator = init_model(
            c.POSE_MODEL_CONFIG,
            c.POSE_MODEL_CHECKPOINT,
            device='cuda:0'
        )
        # Set to 3D mode
        _pose_estimator.cfg.model.test_cfg.mode = '3d'

        logger.info(f"Worker process {os.getpid()} initialized with models")

    except Exception as e:
        logger.error(f"Failed to initialize models in worker {os.getpid()}: {str(e)}")
        raise


def get_video_filenames(directory: str, pattern: str = "*.mp4") -> List[str]:
    """
    Retrieves video filenames from specified directory without extensions.
    """
    return [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(directory, pattern))
    ]


def validate_video_file(video_path: str) -> bool:
    """
    Validates if a video file exists and can be opened by OpenCV.
    Returns True if valid, False otherwise.
    """
    if not os.path.exists(video_path):
        return False

    try:
        cap = cv2.VideoCapture(video_path)
        is_valid = cap.isOpened()
        cap.release()
        return is_valid
    except Exception:
        return False


def _get_video_fps(video_path: str) -> float:
    """
    Return video FPS (float). Returns 0.0 if FPS cannot be obtained.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        return float(fps)
    except Exception:
        return 0.0


class FPSSampler:
    """
    Two sampling strategies:
      1) reduce mode (priority): Downsample source fps to target fps
         (uses accumulation error method for non-integer ratios, solves 30->24 etc.)
      2) skip mode: Sample every Nth frame.
    """
    def __init__(self, src_fps: float, reduce_to: Optional[float], frame_skip_by: int):
        self.mode = 'reduce' if (reduce_to is not None and src_fps > 0) else 'skip'
        if self.mode == 'reduce':
            # Only downsample: if target >= src, sample every frame (equivalent to no reduction)
            self.target = min(reduce_to, src_fps)
            # Accumulation error method (Bresenham-like): accumulate r=target/src per frame,
            # when acc>=1, sample and acc-=1
            self.r = self.target / max(src_fps, 1e-6)
            self.acc = 0.0
        else:
            self.n = max(int(frame_skip_by), 1)
            self.count = 0

    def take(self) -> bool:
        if self.mode == 'reduce':
            self.acc += self.r
            if self.acc >= 1.0:
                self.acc -= 1.0
                return True
            return False
        else:
            take_now = (self.count % self.n) == 0
            self.count += 1
            return take_now


def _to_numpy(x):
    """
    Converts torch.Tensor or other array-like objects to numpy arrays.
    """
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    elif hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    return np.asarray(x)


def _squeeze_kpts(arr, expect_last: int = 2):
    """
    Ensure shapes:
      - 2D transformed keypoints -> (N, K, 2)
      - 3D keypoints             -> (N, K, 3)
    Removes common singleton dims like (N,1,K,2) -> (N,K,2)
    """
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr


def pack_xy_transformed_and_z_keypoints(
    pred_3d_instances,
    img_w: int,
    img_h: int,
    instance_index: int = 0
) -> Optional[np.ndarray]:
    """
    Extracts and packs keypoints from 3D pose estimation results.

    Returns flattened array where:
      - First person only (instance_index=0)
      - K=NUM_KEYPOINTS keypoints (filtered by COCO_WHOLEBODY_IDX)
      - If ADD_VISIBLE=False: Each keypoint has [x_norm, y_norm, z_rebased],
        output shape: (NUM_KEYPOINTS * 3,)
      - If ADD_VISIBLE=True: Each keypoint has [x_norm, y_norm, z_rebased, visible],
        output shape: (NUM_KEYPOINTS * 4,)

    Args:
        pred_3d_instances: Predicted 3D pose instances from MMPose
        img_w: Original image width for normalization
        img_h: Original image height for normalization
        instance_index: Which person instance to extract (default: 0 for first person)

    Returns:
        Flattened numpy array of keypoints or None if extraction fails
    """
    if pred_3d_instances is None:
        return None

    # Get arrays
    tk = getattr(pred_3d_instances, 'transformed_keypoints', None)
    k3d = getattr(pred_3d_instances, 'keypoints', None)
    if tk is None or k3d is None:
        return None

    tk = _to_numpy(tk)
    k3d = _to_numpy(k3d)
    tk = _squeeze_kpts(tk)   # (N, K, 2)
    k3d = _squeeze_kpts(k3d) # (N, K, 3)

    # Guard: need at least one instance
    if tk.ndim != 3 or k3d.ndim != 3 or tk.shape[0] == 0 or k3d.shape[0] == 0:
        return None

    # Select instance (default: first person)
    xy = tk[instance_index]      # (K, 2) in original image coords
    xyz = k3d[instance_index]    # (K, 3) in model-input coords

    # Filter by COCO_WHOLEBODY_IDX to get only the keypoints we need
    xy = xy[c.COCO_WHOLEBODY_IDX]          # (NUM_KEYPOINTS, 2)
    xyz = xyz[c.COCO_WHOLEBODY_IDX]        # (NUM_KEYPOINTS, 3)

    # Normalize x,y by original image size
    x_norm = xy[..., 0] / float(img_w)
    y_norm = xy[..., 1] / float(img_h)

    # z rebase using average of keypoints 6 and 7 (shoulders in COCO-WholeBody)
    z = xyz[..., 2]
    if z.shape[0] > 7:  # make sure idx 6 & 7 exist
        z_ref = 0.5 * (z[6] + z[7])
        z = z - z_ref

    if c.ADD_VISIBLE:
        # Get visibility scores
        kpt_scores = getattr(pred_3d_instances, 'keypoint_scores', None)
        if kpt_scores is not None:
            kpt_scores = _to_numpy(kpt_scores)
            # Handle different dimensions
            if kpt_scores.ndim == 2:  # (N, K)
                visible = kpt_scores[instance_index]
            elif kpt_scores.ndim == 3:  # (N, K, 1)
                visible = kpt_scores[instance_index, :, 0]
            else:
                visible = np.ones(len(c.COCO_WHOLEBODY_IDX), dtype=np.float32)

            # Filter by COCO_WHOLEBODY_IDX
            visible = visible[c.COCO_WHOLEBODY_IDX]  # (NUM_KEYPOINTS,)
        else:
            # Default to all visible if scores not available
            visible = np.ones(len(c.COCO_WHOLEBODY_IDX), dtype=np.float32)

        # Stack -> (NUM_KEYPOINTS, 4), then flatten to (NUM_KEYPOINTS * 4,)
        out = np.stack([x_norm, y_norm, z, visible], axis=-1).flatten().astype(np.float32)
    else:
        # Stack -> (NUM_KEYPOINTS, 3), then flatten to (NUM_KEYPOINTS * 3,)
        out = np.stack([x_norm, y_norm, z], axis=-1).flatten().astype(np.float32)

    return out


def process_frame_with_mmpose(
    frame: np.ndarray,
    detector,
    pose_estimator
) -> Optional[np.ndarray]:
    """
    Processes a single frame through MMPose detection and pose estimation.

    Pipeline:
        frame -> detector -> pose_estimator -> keypoint extraction

    Args:
        frame: Input video frame (BGR format)
        detector: MMDet person detector model
        pose_estimator: MMPose 3D pose estimator model

    Returns:
        Packed keypoint array or None if detection/estimation fails
    """
    # Person detection
    det_result = inference_detector(detector, frame)
    pred_instance = det_result.pred_instances.cpu().numpy()

    # Filter person instances by category and bbox threshold
    bboxes = pred_instance.bboxes
    bboxes = bboxes[np.logical_and(
        pred_instance.labels == c.DET_CAT_ID,
        pred_instance.scores > c.BBOX_THR
    )]

    # No person detected
    if len(bboxes) == 0:
        return None

    # 3D Pose estimation
    pose_est_results = inference_topdown(pose_estimator, frame, bboxes)

    # Post-processing: squeeze dimensions and sort by track_id
    for idx, pose_est_result in enumerate(pose_est_results):
        pose_est_result.track_id = pose_est_results[idx].get('track_id', 1e4)

        pred_instances = pose_est_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores

        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_est_results[idx].pred_instances.keypoint_scores = keypoint_scores

        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        pose_est_results[idx].pred_instances.keypoints = keypoints

    # Sort by track_id and merge
    pose_est_results = sorted(
        pose_est_results, key=lambda x: x.get('track_id', 1e4))

    pred_3d_data_samples = merge_data_samples(pose_est_results)
    pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

    if pred_3d_instances is None:
        return None

    # Extract and pack keypoints
    H, W = frame.shape[:2]
    packed = pack_xy_transformed_and_z_keypoints(
        pred_3d_instances, W, H, instance_index=0
    )

    return packed


def process_video_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    output_file: str
):
    """
    Processes a video segment to extract 3D pose keypoints using MMPose.
    Uses pre-loaded models from global cache (initialized once per worker).

    Args:
        video_path: Path to the input video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_file: Path to save the output .npy file
    """
    global _detector, _pose_estimator

    cap = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video: {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        # Create FPS sampler: prioritize REDUCE_FPS_TO, otherwise use FRAME_SKIP
        target_fps = None if getattr(c, "REDUCE_FPS_TO", None) is None else float(c.REDUCE_FPS_TO)
        sampler = FPSSampler(src_fps=fps, reduce_to=target_fps, frame_skip_by=c.FRAME_SKIP)

        # Calculate frame ranges
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Use pre-loaded global models (initialized in init_worker)
        landmark_sequences = []
        current_frame = start_frame

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Use sampler to decide whether to process this frame
            if sampler.take():
                packed = process_frame_with_mmpose(frame, _detector, _pose_estimator)
                if packed is not None:
                    landmark_sequences.append(packed)

            current_frame += 1

        # Save landmarks if valid data exists
        if landmark_sequences:
            landmark_array = np.array(landmark_sequences)
            if landmark_array.size > 0 and np.any(landmark_array):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                np.save(output_file, landmark_array)
                logger.info(
                    f"Saved {len(landmark_sequences)} frames of landmarks to {output_file}"
                )
            else:
                logger.info(f"No valid landmarks for segment {video_path}, not saving.")
        else:
            logger.info(f"No landmarks detected for segment {video_path}")

    except Exception as e:
        logger.error(f"Error processing {video_path}: {str(e)}")

    finally:
        # Resource cleanup
        if cap is not None:
            cap.release()

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache to free GPU memory between segments
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Log memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.debug(
            f"Memory usage after processing: {memory_info.rss / 1024 / 1024:.2f} MB"
        )


def process_batch(task_batch: List[Tuple[str, float, float, str]]):
    """
    Process a batch of video segment tasks for bulk processing.

    Args:
        task_batch: List of tuples (video_path, start_time, end_time, output_path)
    """
    for video_path, start, end, output_path in task_batch:
        try:
            process_video_segment(video_path, start, end, output_path)
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")

        # Add small delay after each task to avoid resource contention
        time.sleep(0.1)


def main():
    """
    Main function to orchestrate video processing and landmark extraction using MMPose.
    """
    assert has_mmdet, 'Please install mmdet to run this script.'

    # Read CSV and detect column format
    timestamp_data_full = pd.read_csv(c.CSV_FILE, delimiter="\t", on_bad_lines="skip")
    columns = timestamp_data_full.columns.tolist()

    # Detect which timestamp columns are available
    if "START" in columns and "END" in columns:
        start_col, end_col = "START", "END"
        logger.info("Using START/END columns for timestamps")
    elif "START_REALIGNED" in columns and "END_REALIGNED" in columns:
        start_col, end_col = "START_REALIGNED", "END_REALIGNED"
        logger.info("Using START_REALIGNED/END_REALIGNED columns for timestamps")
    else:
        raise ValueError(
            "Neither START/END nor START_REALIGNED/END_REALIGNED columns found in CSV"
        )

    # Select required columns and drop missing values
    timestamp_data = timestamp_data_full[
        ["VIDEO_NAME", "SENTENCE_NAME", start_col, end_col]
    ].dropna()

    video_files = get_video_filenames(c.VIDEO_DIR, pattern="*.mp4")
    processed_files = get_video_filenames(c.NPY_DIR, pattern="*.npy")

    logger.info(f"Found {len(video_files)} video files")

    # Create video validation cache
    video_validation_cache = {}
    invalid_videos = set()
    skipped_due_to_invalid_video = 0
    skipped_due_to_existing_file = 0
    skipped_due_to_duration = 0
    skipped_due_to_fps_range = 0
    skipped_due_to_too_short = 0
    video_fps_cache = {}

    processing_tasks = []
    for _, row in timestamp_data.iterrows():
        video_name = row.VIDEO_NAME
        sentence_name = row.SENTENCE_NAME
        start, end = row[start_col], row[end_col]

        video_path = os.path.join(c.VIDEO_DIR, f"{video_name}.mp4")
        output_path = os.path.join(c.NPY_DIR, f"{sentence_name}.npy")

        # Skip if output file already exists
        if sentence_name in processed_files:
            skipped_due_to_existing_file += 1
            continue

        # Segment duration limits: 200ms <= duration <= 60 seconds
        seg_dur = float(end - start)
        if seg_dur < 0.2:
            skipped_due_to_too_short += 1
            continue
        if seg_dur > 60.0:
            skipped_due_to_duration += 1
            continue

        # Validate video file (use cache to avoid repeated checks)
        if video_path not in video_validation_cache:
            video_validation_cache[video_path] = validate_video_file(video_path)
            if not video_validation_cache[video_path]:
                invalid_videos.add(video_name)
                logger.warning(f"Invalid or missing video file: {video_path}")

        if not video_validation_cache[video_path]:
            skipped_due_to_invalid_video += 1
            continue

        # Video FPS filtering (skip if FPS not within ACCEPT_VIDEO_FPS_WITHIN range)
        if video_path not in video_fps_cache:
            video_fps_cache[video_path] = _get_video_fps(video_path)
        vfps = video_fps_cache[video_path]
        min_fps, max_fps = c.ACCEPT_VIDEO_FPS_WITHIN
        if vfps <= 0.0 or vfps < float(min_fps) or vfps > float(max_fps):
            skipped_due_to_fps_range += 1
            continue

        processing_tasks.append((video_path, start, end, output_path))

    # Log summary of skipped tasks
    logger.info(f"Task summary:")
    logger.info(f"  - Tasks to process: {len(processing_tasks)}")
    logger.info(f"  - Skipped (existing files): {skipped_due_to_existing_file}")
    logger.info(f"  - Skipped (duration > 60s): {skipped_due_to_duration}")
    logger.info(f"  - Skipped (duration < 0.2s): {skipped_due_to_too_short}")
    logger.info(f"  - Skipped (fps out of {c.ACCEPT_VIDEO_FPS_WITHIN}): {skipped_due_to_fps_range}")
    logger.info(f"  - Skipped (invalid videos): {skipped_due_to_invalid_video}")
    if invalid_videos:
        logger.warning(
            f"Invalid video files found: {', '.join(sorted(invalid_videos))}"
        )

    # Process in batches to avoid submitting too many tasks at once
    BATCH_SIZE = 100  # Process 100 tasks per batch
    MAX_WORKERS = min(c.MAX_WORKERS, multiprocessing.cpu_count() - 1)  # Reserve one CPU core

    logger.info(
        f"Starting parallel processing with {MAX_WORKERS} workers. "
        f"Models will be loaded ONCE for the entire duration (not per batch)."
    )

    # Move ProcessPoolExecutor outside the loop to avoid repeated model loading
    # This ensures workers are initialized only once for the entire execution
    # Models will be loaded MAX_WORKERS times total (not MAX_WORKERS * number_of_batches)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:

        for i in range(0, len(processing_tasks), BATCH_SIZE):
            batch = processing_tasks[i:i + BATCH_SIZE]
            logger.info(
                f"Processing batch {i // BATCH_SIZE + 1}, "
                f"tasks {i + 1} to {min(i + BATCH_SIZE, len(processing_tasks))}"
            )

            # Further subdivide batches for different processes
            tasks_per_worker = len(batch) // MAX_WORKERS + 1
            worker_batches = [
                batch[j:j + tasks_per_worker]
                for j in range(0, len(batch), tasks_per_worker)
            ]

            # Submit tasks to already-initialized workers
            # Workers reuse the same loaded models across all batches
            futures = []
            for worker_batch in worker_batches:
                if worker_batch:  # Ensure batch is not empty
                    future = executor.submit(process_batch, worker_batch)
                    futures.append(future)

            # Wait for all tasks in this batch to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in worker process: {str(e)}")

            # Add delay between batches to allow system time to release resources
            time.sleep(0.5)

            # Force garbage collection
            gc.collect()

            # Log progress and memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(
                f"Batch {i // BATCH_SIZE + 1} completed. "
                f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB"
            )

    # All batches completed, workers will be destroyed here
    logger.info("All batches completed. Workers destroyed.")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
