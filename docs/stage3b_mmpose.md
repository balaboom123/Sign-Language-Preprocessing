# Stage 3b: Extract MMPose RTMPose3D Landmarks

## Overview

Extracts 3D whole-body landmarks using MMPose (RTMDet + RTMPose3D) for each manifest segment. GPU-accelerated extractor with depth estimation, suitable for high-quality landmark extraction.

---

## Files

- **Config**: `configs/extract_mmpose.py`
- **Script**: `scripts/3b_extract_mmpose.py`

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
| `REDUCE_FPS_TO` | float | Target FPS for extraction | `24.0` |
| `FRAME_SKIP` | int | Skip every Nth frame | `1` |
| `ACCEPT_VIDEO_FPS_WITHIN` | Tuple[float, float] | Acceptable video FPS range | `(24.0, 60.0)` |
| `MAX_WORKERS` | int | Number of worker processes | `1` (GPU) |

### Model Configuration

**Detection Model (RTMDet):**
| Parameter | Description |
|-----------|-------------|
| `DET_MODEL_CONFIG` | RTMDet config file path |
| `DET_MODEL_CHECKPOINT` | RTMDet weights file path |
| `DET_MODEL_CHECKPOINT_LINK` | Download URL for weights |

**Pose Model (RTMPose3D):**
| Parameter | Description |
|-----------|-------------|
| `POSE_MODEL_CONFIG` | RTMPose3D config file path |
| `POSE_MODEL_CHECKPOINT` | RTMPose3D weights file path |
| `POSE_MODEL_CHECKPOINT_LINK` | Download URL for weights |

### Inference Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `ADD_VISIBLE` | bool | Include visibility/confidence scores | `True` |
| `BBOX_THR` | float | Min score for person detection | `0.3` |
| `KPT_THR` | float | Min score for keypoints | `0.3` |
| `DET_CAT_ID` | int | Class ID for "person" in COCO | `0` |

### Landmark Selection

| Parameter | Description | Count |
|-----------|-------------|-------|
| `COCO_WHOLEBODY_IDX` | Selected keypoints from COCO WholeBody layout | **85** |

**Layout**:
- **Body**: Upper body and torso keypoints
- **Face**: Selected facial features
- **Hands**: Full hand keypoints (left + right)

---

## Workflow

### 1. Setup and Initialization

**Process:**
1. **Verify Dependencies**:
   - Check `mmdet` is installed (required for RTMDet)
   - Check `mmpose` is available

2. **Download Models** (if needed):
   ```bash
   # Models should be in models/checkpoints/
   # Download from DET_MODEL_CHECKPOINT_LINK and POSE_MODEL_CHECKPOINT_LINK
   ```

3. **Initialize Workers**:
   - Each worker process loads models **once**:
     - RTMDet detector
     - RTMPose3D pose estimator
   - Models loaded on `cuda:0` (GPU)

### 2. Build Processing Tasks

Same as Stage 3a:
1. Read manifest CSV
2. Match video files
3. Filter by duration, FPS, existing outputs
4. Build task list

### 3. Extract Landmarks (Parallel)

**For each task:**

1. **Open Video and Sample Frames**:
   ```python
   cap = cv2.VideoCapture(video_path)
   sampler = FPSSampler(original_fps, target_fps=REDUCE_FPS_TO)
   ```

2. **Process Each Frame**:

   **a) Detect Person (RTMDet)**:
   ```python
   detections = inference_detector(det_model, frame)
   person_boxes = detections[DET_CAT_ID]  # Class 0 = person
   high_score_boxes = person_boxes[scores > BBOX_THR]
   ```

   **Multi-person check**:
   - If 0 persons: Skip frame (no landmarks)
   - If 1 person: Continue
   - If >1 person: Raise `MultiPersonDetected` → **skip entire segment**

   **b) Estimate 2D Pose (RTMPose)**:
   ```python
   pose_results = inference_topdown(pose_model, frame, bboxes)
   keypoints_2d = pose_results[0].pred_instances.keypoints  # (133, 2)
   scores = pose_results[0].pred_instances.keypoint_scores  # (133,)
   ```

   **c) Lift to 3D**:
   ```python
   # Use RTMPose3D lifting module
   keypoints_3d = lift_to_3d(keypoints_2d)  # (133, 3)
   ```

   **d) Normalize Coordinates**:
   ```python
   x_norm = x / image_width
   y_norm = y / image_height
   z_norm = z  # Already in normalized space
   visibility = scores  # Confidence from RTMPose

   # Shape: (133, 4) = (x, y, z, vis)
   ```

   **e) Select Keypoints**:
   ```python
   # Use COCO_WHOLEBODY_IDX to select 85 keypoints
   selected = keypoints_3d[COCO_WHOLEBODY_IDX, :]  # (85, 4)
   ```

3. **Stack and Save**:
   ```python
   # Stack all frames: (T, 85, 4)
   landmarks_array = np.stack(frame_landmarks, axis=0)

   # Save as .npy
   np.save(output_npy, landmarks_array)
   ```

---

## Usage

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install mmdet mmpose mmcv opencv-python numpy torch
   ```

2. **Download Models**:
   ```bash
   # Create checkpoint directory
   mkdir -p models/checkpoints

   # Download RTMDet weights
   wget {DET_MODEL_CHECKPOINT_LINK} -O models/checkpoints/rtmdet_m.pth

   # Download RTMPose3D weights
   wget {POSE_MODEL_CHECKPOINT_LINK} -O models/checkpoints/rtmw3d-l.pth
   ```

3. **Check CUDA**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   # Should output: True
   ```

### Basic Execution

```bash
python scripts/3b_extract_mmpose.py
```

### Expected Output

```
=== Stage 3b: Extract MMPose Landmarks ===

Configuration:
  VIDEO_DIR: dataset/origin
  NPY_DIR: dataset/npy
  CSV_FILE: assets/youtube_asl.csv
  REDUCE_FPS_TO: 24.0
  MAX_WORKERS: 1 (GPU)

Loading models...
✓ RTMDet loaded: models/checkpoints/rtmdet_m.pth
✓ RTMPose3D loaded: models/checkpoints/rtmw3d-l.pth

Building processing tasks...
Total segments: 1000
Valid tasks: 850

Processing segments...
[1/850] abc123-000: 72 frames → dataset/npy/abc123-000.npy
[2/850] abc123-001: 58 frames → dataset/npy/abc123-001.npy
⊗ [3/850] abc123-002: Skipped (multiple persons detected)
...
[850/850] xyz789-042: 94 frames → dataset/npy/xyz789-042.npy

Extraction complete!
Processed: 847 segments
Skipped (multi-person): 3
Total frames: 60,984
Output directory: dataset/npy
```

---

## Output Format

### Landmark Array Structure

**File**: `{NPY_DIR}/{SENTENCE_NAME}.npy`

**Shape**: `(T, 85, 4)` where:
- `T`: Number of frames
- `85`: Selected COCO WholeBody keypoints
- `4`: Channels `(x, y, z, visibility)`

**Coordinate System**:
- `x`: Normalized by image width (0.0 - 1.0)
- `y`: Normalized by image height (0.0 - 1.0)
- `z`: Depth in normalized camera space (can be negative)
- `visibility`: Confidence score from RTMPose (0.0 - 1.0)

**Example**:
```python
import numpy as np

landmarks = np.load("dataset/npy/abc123-000.npy")
print(landmarks.shape)  # (72, 85, 4)

# Frame 0, keypoint 0
keypoint = landmarks[0, 0, :]
x, y, z, vis = keypoint
print(f"Position: ({x:.3f}, {y:.3f}, {z:.3f}), Confidence: {vis:.3f}")
```

---

## Multi-Person Detection Handling

### Why Skip Multi-Person Segments?

ASL translation models typically expect **single-person monologues**. Segments with multiple people:
- Introduce ambiguity (which person is signing?)
- Mix different signers' styles
- Complicate landmark tracking

### Detection Logic

```python
high_score_persons = detections[scores > BBOX_THR]

if len(high_score_persons) == 0:
    # No person detected → skip frame (return zeros or None)
    pass
elif len(high_score_persons) == 1:
    # Single person → process normally
    pass
else:
    # Multiple persons → skip ENTIRE segment
    raise MultiPersonDetected
```

### Adjusting Sensitivity

```python
# More strict (fewer false positives, may miss some people)
BBOX_THR = 0.5

# More lenient (catch more people, more false positives)
BBOX_THR = 0.2
```

---

## Performance

### Processing Speed

**GPU-based** (typical performance with RTX 3090):
- **Speed**: 10-20 frames per second
- **Bottleneck**: GPU inference (RTMDet + RTMPose3D)
- **Scaling**: Limited (single GPU, `MAX_WORKERS=1` recommended)

**Example Timing**:
```
1000 segments × 3 seconds × 24 fps = 72,000 frames
Processing time: ~60-120 minutes on single GPU
```

### Storage Requirements

Same as Stage 3a:
```
Size per .npy file (3 seconds, 72 frames): ~98 KB
1000 segments: ~98 MB total
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms:**
- Error: "CUDA out of memory"
- Process crashes during extraction

**Solutions:**
1. Ensure `MAX_WORKERS = 1` (single GPU worker)
2. Reduce batch size (process fewer frames at once)
3. Use smaller models:
   ```python
   # Use RTMDet-nano instead of RTMDet-m
   DET_MODEL_CONFIG = "rtmdet_nano_320-8xb32_coco-person.py"
   ```
4. Close other GPU applications

### Issue: Models Not Found

**Symptoms:**
- Error: "FileNotFoundError: checkpoint file not found"

**Solutions:**
1. Check model paths in config
2. Download models manually:
   ```bash
   wget {DET_MODEL_CHECKPOINT_LINK}
   wget {POSE_MODEL_CHECKPOINT_LINK}
   ```
3. Verify file permissions

### Issue: mmdet Not Installed

**Symptoms:**
- Error: "ModuleNotFoundError: No module named 'mmdet'"

**Solutions:**
```bash
pip install mmdet mmpose mmcv
# Or with conda
conda install -c conda-forge mmdet mmpose
```

### Issue: Too Many Segments Skipped (Multi-Person)

**Symptoms:**
- Many segments skipped due to multiple persons
- Expected single-person videos

**Solutions:**
1. Increase `BBOX_THR` to reduce false detections:
   ```python
   BBOX_THR = 0.5  # Higher threshold
   ```
2. Check source videos for actual multi-person content
3. Filter manifest to exclude multi-person videos

### Issue: Poor Depth (Z) Quality

**Symptoms:**
- Z-coordinates seem inaccurate or noisy
- Depth variations don't match video

**Solutions:**
1. RTMPose3D depth is approximate (monocular estimation)
2. Consider removing Z in Stage 4 if not needed:
   ```python
   # In configs/reduction_normalization.py
   REMOVE_Z = True
   ```
3. For accurate depth, consider stereo cameras or depth sensors

---

## Advanced Configuration

### Use Different Models

**Smaller Models (faster, less accurate)**:
```python
# RTMDet Nano (lightweight)
DET_MODEL_CONFIG = ROOT / "models/configs/rtmdet_nano_320-8xb32_coco-person.py"
DET_MODEL_CHECKPOINT = ROOT / "models/checkpoints/rtmdet_nano.pth"

# RTMPose3D Medium
POSE_MODEL_CONFIG = ROOT / "models/configs/rtmw3d-m_8xb64_cocktail14-384x288.py"
POSE_MODEL_CHECKPOINT = ROOT / "models/checkpoints/rtmw3d-m.pth"
```

**Larger Models (slower, more accurate)**:
```python
# RTMPose3D X-Large
POSE_MODEL_CONFIG = ROOT / "models/configs/rtmw3d-x_8xb32_cocktail14-384x288.py"
POSE_MODEL_CHECKPOINT = ROOT / "models/checkpoints/rtmw3d-x.pth"
```

### Custom Keypoint Selection

Edit `COCO_WHOLEBODY_IDX` in config:
```python
# Full COCO WholeBody (133 keypoints)
COCO_WHOLEBODY_IDX = list(range(133))

# Body + hands only (no face)
COCO_WHOLEBODY_IDX = list(range(17)) + list(range(91, 133))

# Upper body only
COCO_WHOLEBODY_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### Disable Visibility Channel

```python
# In config
ADD_VISIBLE = False

# Output will be (T, 85, 3) instead of (T, 85, 4)
# Channels: (x, y, z) only
```

---

## Model Downloads

### RTMDet (Person Detector)

```bash
# RTMDet-M (medium, recommended)
wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth

# RTMDet-Nano (lightweight)
wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_nano_8xb32-300e_coco/rtmdet_nano_8xb32-300e_coco_20220801_234417-89ae0f26.pth
```

### RTMPose3D (Pose Estimator)

```bash
# RTMPose3D-L (large, recommended)
wget https://download.openmmlab.com/mmpose/v1/projects/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288.pth

# RTMPose3D-X (extra large)
wget https://download.openmmlab.com/mmpose/v1/projects/rtmw3d/rtmw3d-x_8xb32_cocktail14-384x288.pth
```

---

## Validation

### Check Output Integrity

```python
import numpy as np
from pathlib import Path

npy_dir = Path("dataset/npy")
files = list(npy_dir.glob("*.npy"))

for npy_file in files:
    arr = np.load(npy_file)
    assert arr.shape[1] == 85, f"Wrong keypoints: {npy_file}"
    assert arr.shape[2] == 4, f"Wrong channels: {npy_file}"

    # Check coordinate ranges
    x = arr[:, :, 0]
    y = arr[:, :, 1]
    vis = arr[:, :, 3]

    assert x.min() >= 0 and x.max() <= 1, f"X out of range: {npy_file}"
    assert y.min() >= 0 and y.max() <= 1, f"Y out of range: {npy_file}"
    assert vis.min() >= 0 and vis.max() <= 1, f"Vis out of range: {npy_file}"

print(f"✓ All {len(files)} files valid")
```

---

## Next Steps

After Stage 3b completes successfully:

1. **Verify Outputs**: Check `.npy` files
2. **Proceed to Stage 4**: Normalize landmarks
   ```bash
   python scripts/4_reduction_normalization.py
   ```

---

## Dependencies

- `mmdet`: Object detection framework
- `mmpose`: Pose estimation framework
- `mmcv`: OpenMMLab computer vision library
- `torch`: PyTorch (with CUDA)
- `opencv-python`: Video processing
- `numpy`: Array operations

Install:
```bash
pip install mmdet mmpose mmcv torch opencv-python numpy
```
