"""Configuration for MMPose 3D landmark extraction (scripts/3b_extract_mmpose.py)"""
import os

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "dataset", "origin")
NPY_DIR = os.path.join(ROOT, "dataset", "npy")

# Manifest CSV file (contains video segment timestamps and metadata)
CSV_FILE = os.path.join(ROOT, "dataset", "how2sign", "how2sign_realigned_val.csv")

# =============================================================================
# FRAME SAMPLING CONFIGURATION
# =============================================================================

# Option to downsample frames to a fixed FPS (takes priority over FRAME_SKIP)
REDUCE_FPS_TO = 24.0  # Target FPS (set to None to disable FPS reduction)

# Frame sampling when NOT using REDUCE_FPS_TO (sample every Nth frame)
FRAME_SKIP = 2  # e.g., 2 means sampling rate is 1/2

# Accepted video FPS range (videos outside this range will be skipped)
ACCEPT_VIDEO_FPS_WITHIN = (24.0, 60.0)  # (min_fps, max_fps)

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

# Maximum number of worker processes for parallel video processing
# Note: Each worker loads models into GPU memory, adjust based on available GPU memory
MAX_WORKERS = 4

# =============================================================================
# COCO-WHOLEBODY KEYPOINT SELECTION (85 keypoints total)
# =============================================================================

# Selected keypoints from COCO-WholeBody format (133 total keypoints)
# We filter to 85 keypoints optimized for ASL recognition
COCO_WHOLEBODY_IDX = [
    # Body keypoints (6 points): shoulders, elbows, hips
    5, 6, 7, 8, 11, 12,

    # Face shape keypoints (9 points): face contour
    23, 25, 27, 29, 31, 33, 35, 37, 39,

    # Eyes and eyebrows keypoints (10 points): expressions
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,

    # Nose keypoints (4 points): facial orientation
    52, 54, 56, 58,

    # Mouth keypoints (14 points): mouth shapes for ASL phonemes
    71, 73, 75, 77, 79, 81, 83, 84, 85, 86, 87, 88, 89, 90,
] + list(range(91, 133))  # All hand landmarks (42 points): 21 left + 21 right

# Total: 6 body + 37 face + 42 hands = 85 keypoints

# =============================================================================
# MODEL PATHS (RTMPose3D + RTMDet)
# =============================================================================

# RTMPose3D: 3D whole-body pose estimator (detects 133 keypoints in 3D)
POSE_MODEL_CHECKPOINT = os.path.join(
    ROOT, "models", "checkpoints",
    "rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
)
POSE_MODEL_CONFIG = os.path.join(
    ROOT, "models", "configs",
    "rtmw3d-l_8xb64_cocktail14-384x288.py"
)

# RTMDet: Person detector (prerequisite for pose estimation)
DET_MODEL_CHECKPOINT = os.path.join(
    ROOT, "models", "checkpoints",
    "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"
)
DET_MODEL_CONFIG = os.path.join(
    ROOT, "models", "configs",
    "rtmdet_nano_320-8xb32_coco-person.py"
)

# Download URLs for model checkpoints (if needed)
POSE_MODEL_CHECKPOINT_LINK = (
    "https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/"
    "rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
)
DET_MODEL_CHECKPOINT_LINK = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
    "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"
)

# =============================================================================
# OUTPUT FORMAT CONFIGURATION
# =============================================================================

# Include visibility scores in output
# If True:  output shape is (T, 85*4) with (x, y, z, visible) per keypoint
# If False: output shape is (T, 85*3) with (x, y, z) per keypoint
ADD_VISIBLE = True

# =============================================================================
# INFERENCE PARAMETERS
# =============================================================================

# Bounding box score threshold for person detection
# Higher values = fewer but more confident person detections
BBOX_THR = 0.5  # Range: 0.0 to 1.0

# Keypoint score threshold for pose estimation
# Keypoints with scores below this threshold will have lower visibility scores
KPT_THR = 0.3  # Range: 0.0 to 1.0

# Category ID for person detection in COCO dataset
# 0 = person category
DET_CAT_ID = 0
