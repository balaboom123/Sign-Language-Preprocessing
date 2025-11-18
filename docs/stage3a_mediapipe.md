# Stage 3a: Extract MediaPipe Holistic Landmarks

## Overview

Extracts 3D pose, face, and hand landmarks using MediaPipe Holistic for each manifest segment. CPU-friendly extractor suitable for both YouTube-ASL and How2Sign datasets.

---

## Files

- **Config**: `configs/extract_mediapipe.py`
- **Script**: `scripts/3a_extract_mediapipe.py`

---

## Configuration Reference

### Path Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `ROOT` | Path | Project root directory | Auto-detected |
| `VIDEO_DIR` | Path | Input video directory | `{ROOT}/dataset/origin` |
| `NPY_DIR` | Path | Output landmark directory | `{ROOT}/dataset/npy` |
| `CSV_FILE` | Path | Manifest CSV file | `{ROOT}/assets/youtube_asl.csv` |

### Frame Sampling Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `REDUCE_FPS_TO` | float | Target FPS for extraction (None to disable) | `24.0` |
| `FRAME_SKIP` | int | Skip every Nth frame (when not using FPS reduction) | `1` |
| `ACCEPT_VIDEO_FPS_WITHIN` | Tuple[float, float] | Acceptable original video FPS range | `(24.0, 60.0)` |

### Parallelism Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `MAX_WORKERS` | int | Number of worker processes | `os.cpu_count()` |

### Landmark Selection

| Parameter | Description | Count |
|-----------|-------------|-------|
| `HAND_IDX` | Hand keypoint indices (per hand) | 21 × 2 = 42 |
| `POSE_IDX` | Body keypoint indices | 6 |
| `FACE_IDX` | Face keypoint indices | 37 |
| **Total** | All selected keypoints | **85** |

**Landmark Layout:**
- **Pose (6)**: Left/right shoulders, elbows, hips
- **Face (37)**: Selected facial features for expressions
- **Hands (42)**: 21 landmarks per hand (left + right)

---

## Workflow

### 1. Build Processing Tasks

**Process:**
1. **Read Manifest CSV**:
   - Load `CSV_FILE` using `read_manifest_csv`
   - Automatically detect timestamp columns (`START_REALIGNED`, `END_REALIGNED`, etc.)

2. **Match Videos**:
   - For each segment in manifest:
   - Find video file: `VIDEO_DIR/{VIDEO_NAME}.mp4`
   - Skip if video file missing

3. **Filter Segments**:
   - **Duration**: `0.2 <= duration <= 60.0` seconds
   - **Video FPS**: Within `ACCEPT_VIDEO_FPS_WITHIN` range
   - **Existing output**: Skip if `.npy` already exists in `NPY_DIR`

4. **Build Task List**:
   ```python
   task = {
       "video_path": Path to video file,
       "start": Start time (seconds),
       "end": End time (seconds),
       "output_npy": Path to output .npy file
   }
   ```

### 2. Extract Landmarks (Parallel)

**For each task (in parallel with `MAX_WORKERS`):**

1. **Open Video**:
   ```python
   cap = cv2.VideoCapture(video_path)
   fps = cap.get(cv2.CAP_PROP_FPS)
   total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
   ```

2. **Initialize FPS Sampler**:
   ```python
   sampler = FPSSampler(
       original_fps=fps,
       target_fps=REDUCE_FPS_TO  # e.g., 24.0
   )
   ```

3. **Process Frames**:
   - Seek to start time
   - For each frame in [start, end]:
     - Check if frame should be sampled (via `FPSSampler`)
     - If yes:
       - Read frame with OpenCV
       - Convert BGR → RGB
       - Pass to MediaPipe Holistic
       - Extract landmarks:
         - `pose_landmarks`: 33 keypoints → select 6 via `POSE_IDX`
         - `face_landmarks`: 468 keypoints → select 37 via `FACE_IDX`
         - `left_hand_landmarks`: 21 keypoints
         - `right_hand_landmarks`: 21 keypoints
       - Concatenate: `(pose, face, left_hand, right_hand)` → shape `(85, 4)`
       - Missing parts → zeros `(0, 0, 0, 0)`

4. **Stack and Save**:
   ```python
   # Stack frames: (T, 85, 4)
   landmarks_array = np.stack(frame_landmarks, axis=0)

   # Save as .npy
   np.save(output_npy, landmarks_array)
   ```

---

## Usage

### Basic Execution

```bash
python scripts/3a_extract_mediapipe.py
```

### Expected Output

```
=== Stage 3a: Extract MediaPipe Landmarks ===

Configuration:
  VIDEO_DIR: dataset/origin
  NPY_DIR: dataset/npy
  CSV_FILE: assets/youtube_asl.csv
  REDUCE_FPS_TO: 24.0
  MAX_WORKERS: 8

Building processing tasks...
Total segments: 1000
Valid tasks: 850
Skipped (existing): 100
Skipped (missing video): 30
Skipped (invalid FPS): 20

Processing segments...
[1/850] abc123-000: 72 frames → dataset/npy/abc123-000.npy
[2/850] abc123-001: 58 frames → dataset/npy/abc123-001.npy
...
[850/850] xyz789-042: 94 frames → dataset/npy/xyz789-042.npy

Extraction complete!
Processed: 850 segments
Total frames: 61,245
Output directory: dataset/npy
```

---

## Output Format

### Landmark Array Structure

**File**: `{NPY_DIR}/{SENTENCE_NAME}.npy`

**Shape**: `(T, num_landmarks, 4)` where:
- `T`: Number of frames in segment
- `num_landmarks`: 85 (pose + face + hands)
- `4`: Channels `(x, y, z, visibility)`

**Coordinate System**:
- `x, y`: Image-normalized coordinates (0.0 - 1.0)
- `z`: Depth relative to hips (approximate, in image-scale units)
- `visibility`: Confidence score (0.0 - 1.0)

**Example**:
```python
import numpy as np

# Load landmark array
landmarks = np.load("dataset/npy/abc123-000.npy")
print(landmarks.shape)  # (72, 85, 4)

# Access specific keypoint in frame 0
left_shoulder = landmarks[0, 0, :]  # (x, y, z, vis)
print(left_shoulder)  # [0.45, 0.32, -0.15, 0.98]
```

### Missing Landmarks

**Behavior**: When MediaPipe cannot detect a body part, that section is filled with zeros.

**Example**:
```python
# Face not detected in frame 10
landmarks[10, 6:43, :] = 0.0  # Face landmarks (indices 6-42)

# Left hand not detected in frame 15
landmarks[15, 43:64, :] = 0.0  # Left hand landmarks
```

---

## Landmark Index Reference

### Pose Landmarks (Indices 0-5)

| Index | Landmark | Description |
|-------|----------|-------------|
| 0 | Left Shoulder | Upper left torso |
| 1 | Right Shoulder | Upper right torso |
| 2 | Left Elbow | Left arm joint |
| 3 | Right Elbow | Right arm joint |
| 4 | Left Hip | Lower left torso |
| 5 | Right Hip | Lower right torso |

### Face Landmarks (Indices 6-42)

37 selected facial keypoints for expression detection:
- Eyebrows, eyes, nose, lips, jaw contours
- Full mapping in `configs/extract_mediapipe.py::FACE_IDX`

### Hand Landmarks (Indices 43-84)

**Left Hand (43-63)**: 21 keypoints
**Right Hand (64-84)**: 21 keypoints

Standard MediaPipe hand layout:
- Wrist (index 0)
- Thumb (indices 1-4)
- Index finger (indices 5-8)
- Middle finger (indices 9-12)
- Ring finger (indices 13-16)
- Pinky (indices 17-20)

---

## Performance

### Processing Speed

**CPU-based** (typical performance on modern CPU):
- **Speed**: 2-5 segments per second per worker
- **Scaling**: Linear with `MAX_WORKERS` up to CPU core count
- **Bottleneck**: MediaPipe inference and video decoding

**Example Timing**:
```
1000 segments × 3 seconds avg duration = 3000 seconds of video
Processing time: ~30-60 minutes with 8 workers
```

### Storage Requirements

**Output Size**:
```
Size per frame = 85 landmarks × 4 channels × 4 bytes (float32) = 1,360 bytes
Size per second (24 fps) = 1,360 × 24 = 32.6 KB

Typical segment (3 seconds, 72 frames):
72 frames × 1,360 bytes = 98 KB per .npy file

1000 segments × 98 KB ≈ 98 MB total
```

---

## Troubleshooting

### Issue: Video FPS Out of Range

**Symptoms:**
- Warning: "Video FPS {fps} outside acceptable range"
- Segments skipped

**Solutions:**
1. Adjust `ACCEPT_VIDEO_FPS_WITHIN` in config:
   ```python
   ACCEPT_VIDEO_FPS_WITHIN = (15.0, 120.0)  # Wider range
   ```
2. Or disable FPS check entirely (not recommended)

### Issue: Missing Landmarks

**Symptoms:**
- Large sections of `.npy` files filled with zeros
- Poor detection quality

**Solutions:**
1. Check video quality (resolution, lighting)
2. Verify people are visible and in frame
3. MediaPipe works best with frontal views
4. Consider using Stage 3b (MMPose) for difficult cases

### Issue: Out of Memory

**Symptoms:**
- Process killed during extraction
- Memory errors

**Solutions:**
1. Reduce `MAX_WORKERS`:
   ```python
   MAX_WORKERS = 4  # Lower value
   ```
2. Process in batches (split manifest CSV)
3. Close other applications

### Issue: Corrupted Output Files

**Symptoms:**
- Cannot load `.npy` files
- Shape errors when loading

**Solutions:**
1. Delete corrupted files: `rm dataset/npy/*.npy`
2. Re-run extraction
3. Check disk space availability

### Issue: Slow Processing

**Symptoms:**
- Extraction takes very long
- Low CPU utilization

**Solutions:**
1. Increase `MAX_WORKERS` to match CPU cores
2. Increase `FRAME_SKIP` to process fewer frames:
   ```python
   FRAME_SKIP = 2  # Process every other frame
   ```
3. Reduce `REDUCE_FPS_TO`:
   ```python
   REDUCE_FPS_TO = 15.0  # Lower FPS
   ```

---

## Advanced Configuration

### Disable FPS Reduction

```python
# Process at original video FPS
REDUCE_FPS_TO = None
FRAME_SKIP = 1
```

### Custom Frame Skip Pattern

```python
# Process every 3rd frame
REDUCE_FPS_TO = None
FRAME_SKIP = 3

# Effective FPS = original_fps / 3
```

### Custom Landmark Selection

Edit `configs/extract_mediapipe.py`:

```python
# Include full pose (33 keypoints)
POSE_IDX = list(range(33))

# Fewer face landmarks (20 instead of 37)
FACE_IDX = [1, 2, 3, 4, 5, ...]  # Custom selection

# Hands only (no pose/face)
# Comment out pose and face sections in extractor
```

### Process Specific Videos

```python
# Create subset manifest
import pandas as pd
df = pd.read_csv("assets/youtube_asl.csv", sep="\t")
subset = df[df["VIDEO_NAME"].isin(["abc123", "def456"])]
subset.to_csv("assets/subset.csv", sep="\t", index=False)

# Update config
CSV_FILE = ROOT / "assets" / "subset.csv"
```

---

## Validation

### Check Output Integrity

```python
import numpy as np
from pathlib import Path

npy_dir = Path("dataset/npy")
files = list(npy_dir.glob("*.npy"))

print(f"Total files: {len(files)}")

# Check each file
for npy_file in files:
    arr = np.load(npy_file)
    assert arr.ndim == 3, f"Wrong dims: {npy_file}"
    assert arr.shape[1] == 85, f"Wrong landmarks: {npy_file}"
    assert arr.shape[2] == 4, f"Wrong channels: {npy_file}"

print("✓ All files valid")
```

### Statistics

```python
import numpy as np
from pathlib import Path

npy_dir = Path("dataset/npy")
files = list(npy_dir.glob("*.npy"))

total_frames = 0
for npy_file in files:
    arr = np.load(npy_file)
    total_frames += arr.shape[0]

print(f"Total segments: {len(files)}")
print(f"Total frames: {total_frames}")
print(f"Avg frames per segment: {total_frames / len(files):.1f}")
```

---

## Next Steps

After Stage 3a completes successfully:

1. **Verify Outputs**: Check `.npy` file counts and shapes
2. **Proceed to Stage 4**: Normalize landmarks
   ```bash
   python scripts/4_reduction_normalization.py
   ```

---

## Dependencies

- `mediapipe`: Holistic pose estimation
- `opencv-python` (`cv2`): Video processing
- `numpy`: Array operations
- Python 3.8+

Install:
```bash
pip install mediapipe opencv-python numpy
```
