# ASL Dataset Preprocessing Pipeline

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-modular-black.svg)](https://github.com/psf/black)

A professional, modular pipeline for preprocessing **American Sign Language (ASL)** datasets, supporting both **YouTube-ASL** and **How2Sign** datasets. This project implements the methodology from ["YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus" (Uthus et al., 2023)](https://arxiv.org/abs/2306.15162).

The pipeline handles the complete workflow from video acquisition to landmark extraction, preparing data for ASL translation tasks using **MediaPipe Holistic** and **MMPose RTMPose3D**.

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
  - [YouTube-ASL Pipeline](#youtube-asl-pipeline)
  - [How2Sign Pipeline](#how2sign-pipeline)
- [Configuration](#-configuration)
- [Pipeline Stages](#-pipeline-stages)
- [Dataset Information](#-dataset-information)
- [Advanced Usage](#-advanced-usage)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Features

- 🎯 **Modular Architecture** - Clean separation of concerns with reusable components
- 🔄 **Two Landmark Extractors** - MediaPipe Holistic and MMPose RTMPose3D support
- 📊 **Dual Dataset Support** - Works with YouTube-ASL and How2Sign datasets
- ⚡ **Parallel Processing** - Multi-worker support for efficient video processing
- 🎬 **Smart Frame Sampling** - Configurable FPS reduction and frame skipping
- 📝 **Comprehensive Logging** - Detailed progress tracking and error reporting
- 🔧 **Flexible Configuration** - Script-specific config files for easy customization
- 📦 **Production Ready** - Type hints, docstrings, and error handling throughout

---

## 📁 Project Structure

```
ASL-Dataset-Preprocess/
├── assets/                          # Demo files and dataset metadata
│   ├── demo.png                     # Example visualization
│   ├── demo_video.mp4               # Sample video
│   ├── youtube-asl_youtube_asl_video_ids.txt  # Video ID list
│   └── youtube_asl.csv              # Processed manifest (generated)
│
├── configs/                         # Script-specific configurations
│   ├── download.py                  # YouTube download settings
│   ├── build_manifest.py            # Transcript processing settings
│   ├── extract_mediapipe.py         # MediaPipe extraction config
│   └── extract_mmpose.py            # MMPose extraction config
│
├── src/asl_prep/                    # Core library modules
│   ├── common/                      # Shared utilities
│   │   ├── files.py                 # File operations
│   │   └── video.py                 # Video processing (FPSSampler, etc.)
│   ├── download/                    # YouTube download logic
│   │   └── youtube.py               # Video & transcript downloading
│   ├── transcripts/                 # Transcript preprocessing
│   │   └── preprocess.py            # Text normalization & segmentation
│   ├── pipeline/                    # Pipeline orchestration
│   │   └── processor.py             # Task building & validation
│   └── extractors/                  # Landmark extraction
│       ├── base.py                  # Abstract extractor interface
│       ├── mediapipe.py             # MediaPipe holistic extractor
│       └── mmpose.py                # MMPose 3D extractor
│
├── scripts/                         # Executable pipeline scripts
│   ├── 1_download_data.py           # Download videos & transcripts
│   ├── 2_build_manifest.py          # Process transcripts to CSV
│   ├── 3a_extract_mediapipe.py      # Extract MediaPipe landmarks
│   └── 3b_extract_mmpose.py         # Extract MMPose 3D landmarks
│
├── dataset/                         # Processing data (not in git)
│   ├── origin/                      # Downloaded videos
│   ├── transcript/                  # Transcript JSON files
│   ├── npy/                         # Extracted landmark arrays
│   └── how2sign/                    # How2Sign dataset files
│
├── models/                          # MMPose model files (not in git)
│   ├── configs/                     # Model configuration files
│   └── checkpoints/                 # Model weights (.pth files)
│
└── README.md                        # This file
```

---

## 🔧 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows (WSL recommended for Windows)
- **GPU**: CUDA-compatible GPU recommended for MMPose (optional for MediaPipe)
- **Storage**: ~100GB+ for datasets and models

### Core Dependencies

- **MediaPipe** - Holistic body landmark detection
- **MMPose** - Advanced 3D pose estimation (optional)
- **OpenCV** - Video processing
- **NumPy** - Numerical operations
- **Pandas** - Data manipulation
- **yt-dlp** - YouTube video downloading

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ASL-Dataset-Preprocess.git
cd ASL-Dataset-Preprocess
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download MMPose Model Checkpoints (If Using MMPose)

```bash
# Create checkpoint directory
mkdir -p models/checkpoints

# Download RTMPose3D model (whole-body 3D pose)
wget https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth \
  -O models/checkpoints/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth

# Download RTMDet model (person detection)
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  -O models/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
```

---

## 🚀 Quick Start

### YouTube-ASL Pipeline

Complete workflow to process YouTube-ASL dataset:

```bash
# Step 1: Download videos and transcripts
python scripts/1_download_data.py

# Step 2: Process transcripts into manifest CSV
python scripts/2_build_manifest.py

# Step 3a: Extract landmarks using MediaPipe (recommended for CPU)
python scripts/3a_extract_mediapipe.py

# Step 3b: OR extract landmarks using MMPose (recommended for GPU)
python scripts/3b_extract_mmpose.py
```

### How2Sign Pipeline

For How2Sign dataset (videos already downloaded):

```bash
# 1. Download How2Sign dataset manually from https://how2sign.github.io/
# 2. Place videos in dataset/origin/
# 3. Place how2sign_realigned_val.csv in dataset/how2sign/

# Extract landmarks (skip steps 1-2)
python scripts/3a_extract_mediapipe.py  # MediaPipe
# OR
python scripts/3b_extract_mmpose.py     # MMPose
```

---

## ⚙️ Configuration

Each pipeline script has its own configuration file in `configs/`:

### `configs/download.py` - YouTube Download Settings

```python
# Video ID source
VIDEO_ID_FILE = "assets/youtube-asl_youtube_asl_video_ids.txt"

# Download directories
VIDEO_DIR = "dataset/origin/"
TRANSCRIPT_DIR = "dataset/transcript/"

# YouTube download settings
YT_CONFIG = {
    "format": "worstvideo[height>=720]/bestvideo[height<=480]",
    "limit_rate": "5M",  # Limit to 5 MB/s
    # ... more settings
}

# Supported languages for transcripts
LANGUAGE = ["en", "ase", "en-US", ...]
```

### `configs/build_manifest.py` - Transcript Processing

```python
# Input/Output paths
VIDEO_ID_FILE = "assets/youtube-asl_youtube_asl_video_ids.txt"
TRANSCRIPT_DIR = "dataset/transcript/"
OUTPUT_CSV = "assets/youtube_asl.csv"

# Filtering constraints
MAX_TEXT_LENGTH = 300  # characters
MIN_DURATION = 0.2     # seconds
MAX_DURATION = 60.0    # seconds
```

### `configs/extract_mediapipe.py` - MediaPipe Extraction

```python
# Data paths
CSV_FILE = "dataset/how2sign/how2sign_realigned_val.csv"
VIDEO_DIR = "dataset/origin/"
NPY_DIR = "dataset/npy/"

# Frame sampling
REDUCE_FPS_TO = 24.0  # Target FPS (None to disable)
FRAME_SKIP = 2        # Skip every Nth frame (when not using REDUCE_FPS_TO)
ACCEPT_VIDEO_FPS_WITHIN = (24.0, 60.0)  # Valid FPS range

# Processing
MAX_WORKERS = 4  # Parallel workers

# Landmark selection (from YouTube-ASL paper)
POSE_IDX = [11, 12, 13, 14, 23, 24]  # Shoulders, elbows, hips
FACE_IDX = [0, 4, 13, 14, 17, ...]   # 37 facial landmarks
HAND_IDX = list(range(21))           # All hand landmarks
```

### `configs/extract_mmpose.py` - MMPose 3D Extraction

```python
# Data paths
CSV_FILE = "dataset/how2sign/how2sign_realigned_val.csv"
VIDEO_DIR = "dataset/origin/"
NPY_DIR = "dataset/npy/"

# Frame sampling
REDUCE_FPS_TO = 24.0
FRAME_SKIP = 2
ACCEPT_VIDEO_FPS_WITHIN = (24.0, 60.0)
MAX_WORKERS = 4

# Keypoint selection (85 keypoints from COCO-WholeBody)
COCO_WHOLEBODY_IDX = [5, 6, 7, 8, 11, 12, ...]

# Model paths
POSE_MODEL_CHECKPOINT = "models/checkpoints/rtmw3d-l_..."
DET_MODEL_CHECKPOINT = "models/checkpoints/rtmdet_m_..."

# Output format
ADD_VISIBLE = True  # Include visibility scores

# Inference parameters
BBOX_THR = 0.5  # Person detection threshold
KPT_THR = 0.3   # Keypoint confidence threshold
```

---

## 🔄 Pipeline Stages

### Stage 1: Data Acquisition (`1_download_data.py`)

Downloads YouTube videos and transcripts based on video ID list.

**Features:**
- Rate limiting to prevent API throttling
- Resume capability (skips already downloaded files)
- Progress tracking with tqdm
- Automatic retry on transient errors

**Output:**
- Videos: `dataset/origin/{video_id}.mp4`
- Transcripts: `dataset/transcript/{video_id}.json`

### Stage 2: Manifest Building (`2_build_manifest.py`)

Processes raw transcripts into a structured manifest CSV.

**Processing Steps:**
1. Unicode normalization (fixes mojibake)
2. Text cleaning (removes brackets, non-ASCII)
3. Duration filtering (0.2s - 60s)
4. Length filtering (max 300 characters)
5. Segment creation with timestamps

**Output:**
- Manifest CSV: `assets/youtube_asl.csv`
- Format: Tab-separated with columns:
  - `VIDEO_NAME`: Source video ID
  - `SENTENCE_NAME`: Segment ID (video_id-XXX)
  - `START_REALIGNED`: Start time (seconds)
  - `END_REALIGNED`: End time (seconds)
  - `SENTENCE`: Normalized transcript text

### Stage 3a: MediaPipe Extraction (`3a_extract_mediapipe.py`)

Extracts holistic body landmarks using MediaPipe.

**Features:**
- Detects 255 features per frame:
  - 6 pose landmarks (shoulders, elbows, hips)
  - 37 face landmarks (expressions, mouth shapes)
  - 21 left hand landmarks
  - 21 right hand landmarks
- Adaptive FPS sampling
- Parallel processing
- Automatic retry and error handling

**Output:**
- Landmark arrays: `dataset/npy/{sentence_name}.npy`
- Shape: `(T, 255)` where T is number of frames
- Format: Float32 numpy arrays

### Stage 3b: MMPose Extraction (`3b_extract_mmpose.py`)

Extracts 3D pose landmarks using MMPose RTMPose3D.

**Features:**
- Two-stage pipeline: RTMDet → RTMPose3D
- 85 keypoints with 3D coordinates (x, y, z)
- Optional visibility scores
- GPU-accelerated (CUDA)
- Model caching per worker process

**Output:**
- Landmark arrays: `dataset/npy/{sentence_name}.npy`
- Shape: `(T, 340)` if ADD_VISIBLE=True (85 × 4)
- Shape: `(T, 255)` if ADD_VISIBLE=False (85 × 3)
- Format: Float32 numpy arrays with normalized coordinates

---

## 📚 Dataset Information

### YouTube-ASL Dataset

- **Size**: 11,000+ videos, 73,000+ segments
- **Domain**: Open-domain, diverse topics
- **Source**: YouTube user-uploaded content
- **Paper**: [Uthus et al., 2023](https://arxiv.org/abs/2306.15162)
- **Video List**: [google-research/youtube_asl](https://github.com/google-research/google-research/blob/master/youtube_asl/README.md)

**Citation:**
```bibtex
@misc{uthus2023youtubeasl,
  author = {Uthus, David and Tanzer, Garrett and Georg, Manfred},
  title = {YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus},
  year = {2023},
  eprint = {2306.15162},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2306.15162},
}
```

### How2Sign Dataset

- **Size**: 80+ hours, 16,000+ sentences
- **Domain**: Instructional videos ("how-to" content)
- **Source**: Professional signers, controlled environment
- **Paper**: [Duarte et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html)
- **Website**: [how2sign.github.io](https://how2sign.github.io/)

**Citation:**
```bibtex
@inproceedings{Duarte_CVPR2021,
    title={{How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language}},
    author={Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deepti and DeHaan, Kenneth and
                   Metze, Florian and Torres, Jordi and Giro-i-Nieto, Xavier},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}
```

---

## 🔬 Advanced Usage

### Custom Landmark Selection

Edit landmark indices in config files to extract different keypoints:

```python
# configs/extract_mediapipe.py

# Example: Extract only hands (no pose, no face)
POSE_IDX = []                # Empty - skip pose
FACE_IDX = []                # Empty - skip face
HAND_IDX = list(range(21))   # All hand landmarks

# Output will be: 21 left + 21 right = 42 landmarks × 3 coords = 126 features
```

### Adjust Frame Sampling

Control processing speed vs. temporal resolution:

```python
# configs/extract_mediapipe.py

# Option 1: Fixed target FPS (recommended)
REDUCE_FPS_TO = 15.0  # Downsample all videos to 15 FPS
FRAME_SKIP = 1        # Not used when REDUCE_FPS_TO is set

# Option 2: Skip every Nth frame
REDUCE_FPS_TO = None  # Disable FPS reduction
FRAME_SKIP = 3        # Sample every 3rd frame (1/3 rate)
```

### Parallel Processing Tuning

Adjust worker count based on your hardware:

```python
# configs/extract_mediapipe.py or extract_mmpose.py

# CPU-bound (MediaPipe)
MAX_WORKERS = 4  # Typically CPU cores - 1

# GPU-bound (MMPose)
MAX_WORKERS = 2  # Fewer workers due to GPU memory constraints
```

### Filter Videos by FPS

Skip videos with unusual frame rates:

```python
# configs/extract_mediapipe.py

# Only process videos between 24-60 FPS
ACCEPT_VIDEO_FPS_WITHIN = (24.0, 60.0)

# Accept all frame rates
ACCEPT_VIDEO_FPS_WITHIN = (1.0, 120.0)
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. Import Error: `cannot import name 'TooManyRequests'`**

Update youtube-transcript-api:
```bash
pip install --upgrade youtube-transcript-api
```

**2. MMPose Model Not Found**

Download model checkpoints (see Installation section) or update paths in `configs/extract_mmpose.py`.

**3. CUDA Out of Memory (MMPose)**

Reduce `MAX_WORKERS` in `configs/extract_mmpose.py`:
```python
MAX_WORKERS = 1  # Process one video at a time
```

**4. Video Download Fails**

Check if video is still available on YouTube. Update yt-dlp:
```bash
pip install --upgrade yt-dlp
```

**5. Slow Processing**

- Enable FPS reduction: Set `REDUCE_FPS_TO = 15.0`
- Increase `FRAME_SKIP` to sample fewer frames
- Reduce `MAX_WORKERS` if system is overloaded

### Debug Mode

Enable detailed logging:

```bash
# Add to scripts before running
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validation

Check output landmark arrays:

```python
import numpy as np

# Load landmark array
landmarks = np.load("dataset/npy/video_id-001.npy")

print(f"Shape: {landmarks.shape}")        # (T, 255) or (T, 340)
print(f"Min: {landmarks.min():.3f}")      # Should be ~-1 to 0
print(f"Max: {landmarks.max():.3f}")      # Should be ~1 to 2
print(f"Mean: {landmarks.mean():.3f}")    # Should be ~0 to 1
print(f"Has NaN: {np.isnan(landmarks).any()}")  # Should be False
```

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **YouTube-ASL Team** - For the dataset and methodology
- **How2Sign Team** - For the How2Sign dataset
- **MediaPipe Team** - For holistic body landmark detection
- **MMPose Team** - For advanced 3D pose estimation
- **OpenMMLab** - For the excellent computer vision framework

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ASL-Dataset-Preprocess/issues)
- **Documentation**: See `REORGANIZATION_SUMMARY.md` for architecture details
- **Contributing**: Pull requests welcome!

---

**Happy ASL Preprocessing! 🤟**
