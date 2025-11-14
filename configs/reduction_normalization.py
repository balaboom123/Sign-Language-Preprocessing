"""Configuration for landmark reduction and normalization (scripts/4_reduction_normalization.py)"""
import os

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input directory: raw landmarks from Step 3 (shape: seq_length, num_keypoint, 4)
INPUT_DIR = os.path.join(ROOT, "dataset", "npy")

# Output directory: normalized landmarks (shape: T, num_keypoint * num_coordinate)
OUTPUT_DIR = os.path.join(ROOT, "dataset", "npy_normalized")

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

# Skip files that already exist in OUTPUT_DIR
SKIP_EXISTING = True

# Maximum number of worker processes for parallel processing
# Set to 1 for debugging, increase based on available CPU cores
MAX_WORKERS = 4

# =============================================================================
# VISIBILITY MASKING CONFIGURATION
# =============================================================================

# Enable/disable frame-level masking
# When True: placeholder frames (all zeros) are set to UNVISIBLE_FRAME
# Frame-level masking does not use VISIBILITY_THRESHOLD
MASK_FRAME_LEVEL = True
UNVISIBLE_FRAME = -999.0

# Enable/disable landmark-level masking
# When True: individual landmarks with low visibility are set to UNVISIBLE_LANDMARK
# Requires VISIBILITY_THRESHOLD
MASK_LANDMARK_LEVEL = True
UNVISIBLE_LANDMARK = -999.0

# Visibility threshold: landmarks with visibility < this value are masked
# Only used when MASK_LANDMARK_LEVEL = True
VISIBILITY_THRESHOLD = 0.3

# =============================================================================
# NORMALIZATION CONFIGURATION
# =============================================================================

# Whether to remove z-coordinate (reduces to 2D landmarks)
# True: Output shape (T, 170) with only x, y coordinates
# False: Output shape (T, 255) with x, y, z coordinates
REMOVE_Z = False

# Normalization method: Isotropic unit bounding box scaling
# Computes single scale factor across x, y, z to preserve aspect ratios
NORMALIZATION_METHOD = 'minmax'

# =============================================================================
# NOTES
# =============================================================================

# Paper methodology (YouTube-ASL):
# 1. Use MediaPipe Holistic to extract 532 landmarks
# 2. Reduce to 85 selected landmarks (done in Step 3)
# 3. Normalize by scaling to fit in unit bounding box across clip duration
# 4. Represent missing landmarks with large negative value (-999)
# 5. Ignore visibility in final output (remove 4th dimension)
# 6. Final output: (T, 255) where 255 = 85 keypoints � 3 coords

# Normalization implementation:
# - Clip-wise: Compute bounding box across entire video clip, not per-frame
# - Isotropic: Single scale factor for x, y, z (preserves aspect ratios)
# - Algorithm:
#   1. Collect all valid 3D points (x, y, z) from entire clip
#   2. Compute 3D bounding box: [min_x, min_y, min_z] to [max_x, max_y, max_z]
#   3. Find max_range = max(x_range, y_range, z_range)
#   4. Scale: p_norm = (p - coord_min) / max_range
#   5. Result: Landmarks fit in 1×1×1 unit cube
# - Sentinel values: UNVISIBLE_FRAME and UNVISIBLE_LANDMARK remain at -999.0
#
# Two-level masking system:
# 1. Frame-level (MASK_FRAME_LEVEL): Entire frames with no detection → UNVISIBLE_FRAME
# 2. Landmark-level (MASK_LANDMARK_LEVEL): Individual landmarks with low visibility → UNVISIBLE_LANDMARK
# Both levels are independent and can be enabled/disabled separately
