# ASL Dataset Preprocessing Pipeline

A professional, modular pipeline for preprocessing **American Sign Language (ASL)** datasets, supporting both **YouTube-ASL** and **How2Sign** datasets. This project implements the methodology from ["YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus" (Uthus et al., 2023)](https://arxiv.org/abs/2306.15162).

The pipeline handles the complete workflow from video acquisition to landmark extraction, preparing data for ASL translation tasks using **MediaPipe Holistic** and **MMPose RTMPose3D**.

---

## Table of Contents

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

## Features

- **Modular Architecture** - Clean separation of concerns with reusable components
- **Two Landmark Extractors** - MediaPipe Holistic and MMPose RTMPose3D support
- **Dual Dataset Support** - Works with YouTube-ASL and How2Sign datasets
- **Parallel Processing** - Multi-worker support for efficient video processing
- **Smart Frame Sampling** - Configurable FPS reduction and frame skipping
- **Comprehensive Logging** - Detailed progress tracking and error reporting
- **Flexible Configuration** - Script-specific config files for easy customization
- **Production Ready** - Type hints, docstrings, and error handling throughout

---

## Project Structure

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

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows (WSL recommended for Windows)
- **GPU**: CUDA-compatible GPU recommended for MMPose (optional for MediaPipe)
- **Storage**: ~100GB+ for datasets and models

---

## Installation

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

### 3. Download MMPose and the checkpoints (If Using MMPose)

```bash
pip install -U openmim
mim install mmcv==2.0.1 mmengine==0.10.7 mmdet==3.1.0
cd ..
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .

# add mmpose to the pythonpath
echo 'export PYTHONPATH="/your/path/to/mmpose:$PYTHONPATH"' >> ~/.bashrc

# Create checkpoint directory
cd path/of/this/project/models/checkpoints

# Download RTMPose3D model (whole-body 3D pose)
wget https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth

# Download RTMDet model (person detection)
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth
```

---

## Quick Start

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

## Pipeline Stages

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

## Dataset Information

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

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---


