import os

# =============================================================================
# PROJECT PATHS AND CONFIGURATION
# =============================================================================

# Base paths
ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = f"{ROOT}/dataset/origin/"
NPY_DIR = f"{ROOT}/dataset/npy/"
TRANSCRIPT_DIR = f"{ROOT}/dataset/transcript/"

# Dataset files
ID = "resource/youtube-asl_youtube_asl_video_ids.txt"
CSV_FILE = f"dataset/how2sign/how2sign_realigned_val.csv" # "resource/youtube_asl.csv"

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

# Option to downsample frames to a fixed FPS (takes priority over FRAME_SKIP)
# Note: Only downsamples, does not upsample (if source fps < REDUCE_FPS_TO, keeps every frame)
REDUCE_FPS_TO = 14.0  # default = 24; set to None to disable FPS reduction

# Frame sampling when NOT using REDUCE_FPS_TO (sample every Nth frame)
FRAME_SKIP = 2  # e.g., 2 means sampling rate is 1/2

# Accepted video FPS range (videos outside this range will be skipped)
ACCEPT_VIDEO_FPS_WITHIN = (24.0, 60.0)  # default: (24, 60)

# Threading
MAX_WORKERS = 4

# FPS reduction (legacy setting for s4_fps_reduce.py, can be ignored if not used)
TARGET_FPS = 8.0  # Target FPS for reduced landmark data

# Supported languages
LANGUAGE = [
    "en",
    "ase",
    "en-US",
    "en-CA",
    "en-GB",
    "en-AU",
    "en-NZ",
    "en-IN",
    "en-ZA",
    "en-IE",
    "en-SG",
    "en-PH",
    "en-NG",
    "en-PK",
    "en-JM",
]

# =============================================================================
# YOUTUBE DOWNLOADER CONFIGURATION
# =============================================================================

YT_CONFIG = {
    # Video quality and format
    "format": "worstvideo[height>=720]/bestvideo[height<=480]",
    "writesubtitles": False,
    "outtmpl": os.path.join(VIDEO_DIR, "%(id)s.%(ext)s"),
    
    # Connection and security
    "nocheckcertificate": True,
    "geo-bypass": True,
    "limit_rate": "5M",
    "http-chunk-size": 10485760,  # 10MB chunks
    
    # Playlist and metadata
    "noplaylist": True,
    "no-metadata-json": True,
    "no-metadata": True,
    
    # Performance optimization
    "concurrent-fragments": 5,
    "hls-prefer-ffmpeg": True,
    "sleep-interval": 0,
}

# =============================================================================
# MEDIAPIPE LANDMARK INDICES
# =============================================================================
# BlazePose format (Mediapipe)
HAND_IDX = list(range(21))
POSE_IDX = [11, 12, 13, 14, 23, 24]
FACE_IDX = [
    0, 4, 13, 14, 17, 33, 37, 39, 46, 52, 55, 61, 64, 81, 82, 93,
    133, 151, 152, 159, 172, 178, 181, 263, 269, 276, 282, 285, 291,
    294, 311, 323, 362, 386, 397, 468, 473
]

# =============================================================================
# MMPOSE PROCESSING CONFIGURATION
# =============================================================================
# ==== COCO-WholeBody indices (133 total, 0-indexed) ====
COCO_WHOLEBODY_IDX = [
    5, 6, 7, 8, 11, 12,  # shoulders, elbows, hips - 6 points
    23, 25, 27, 29, 31, 33, 35, 37, 39,  # face shape - 9 points
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,  # eyes brows - 10 points
    52, 54, 56, 58,  # nose - 4 points
    71, 73, 75, 77, 79, 81, 83, 84, 85, 86, 87, 88, 89, 90  # mouth - 14 points
    ] + list(range(91, 133))  # all 21*2 hand landmarks - 42 points

# 3D Pose estimator configuration
POSE_MODEL_CHECKPOINT = 'models/checkpoints/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth'
POSE_MODEL_CHECKPOINT_LINK = 'https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth'
POSE_MODEL_CONFIG = 'models/configs/rtmw3d-l_8xb64_cocktail14-384x288.py'

# Detector configuration (for person detection)
DET_MODEL_CHECKPOINT = 'models/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
DET_MODEL_CHECKPOINT_LINK = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
DET_MODEL_CONFIG = 'models/configs/rtmdet_m_640-8xb32_coco-person.py'

# Output format configuration
# If True, output shape is (T, NUM_KEYPOINTS*4) with (x, y, z, visible)
# If False, output shape is (T, NUM_KEYPOINTS*3) with (x, y, z)
ADD_VISIBLE = True

# Detection and inference parameters
BBOX_THR = 0.5  # Bounding box score threshold for person detection
KPT_THR = 0.3   # Keypoint score threshold for pose estimation
DET_CAT_ID = 0  # Category ID for person detection in COCO dataset


