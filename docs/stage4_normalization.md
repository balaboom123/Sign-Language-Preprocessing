# Stage 4: Landmark Reduction & Normalization

## Overview

Applies visibility-based masking, performs whole-clip unit bounding box normalization, optionally drops z-coordinate, and flattens arrays to model-ready format. This stage ensures landmarks are properly scaled and consistent across all segments.

---

## Files

- **Config**: `configs/reduction_normalization.py`
- **Script**: `scripts/4_reduction_normalization.py`

---

## Configuration Reference

### Path Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `ROOT` | Path | Project root directory | Auto-detected |
| `INPUT_DIR` | Path | Raw landmark directory | `{ROOT}/dataset/npy` |
| `OUTPUT_DIR` | Path | Normalized landmark directory | `{ROOT}/dataset/npy_normalized` |

### Execution Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `SKIP_EXISTING` | bool | Don't overwrite existing files | `True` |
| `MAX_WORKERS` | int | Number of worker processes | `os.cpu_count()` |

### Masking Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `MASK_FRAME_LEVEL` | bool | Mask frames with all zeros | `True` |
| `UNVISIBLE_FRAME` | float | Sentinel value for missing frames | `-999.0` |
| `MASK_LANDMARK_LEVEL` | bool | Mask low-visibility landmarks | `True` |
| `UNVISIBLE_LANDMARK` | float | Sentinel for missing landmarks | `-999.0` |
| `VISIBILITY_THRESHOLD` | float | Min visibility to treat as valid | `0.5` |

### Geometry Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `REMOVE_Z` | bool | Drop z-coordinate, keep only (x, y) | `False` |
| `NORMALIZATION_METHOD` | str | Normalization strategy | `'minmax'` |

---

## Workflow

### Processing Pipeline

For each `.npy` file in `INPUT_DIR`:

```
Load → Visibility Mask → Normalize → Drop Z (optional) → Flatten → Save
```

### 1. Load Array

**Input Formats Accepted**:
- `(T, 85, 4)`: Standard format from Stage 3
- `(T, 340)`: Pre-flattened format (85 × 4)

**Process**:
```python
arr = np.load(input_file)
if arr.ndim == 2:
    # Reshape (T, 340) → (T, 85, 4)
    T = arr.shape[0]
    arr = arr.reshape(T, 85, 4)
```

### 2. Apply Visibility Masking

**a) Frame-Level Masking** (if `MASK_FRAME_LEVEL = True`):

Detects frames where all keypoints are zero (missing detection).

```python
# Check if frame has any valid data
all_zero = np.all(arr[frame, :, :3] == 0)

if all_zero:
    # Mark entire frame as invisible
    arr[frame, :, :3] = UNVISIBLE_FRAME  # -999.0
```

**b) Landmark-Level Masking** (if `MASK_LANDMARK_LEVEL = True`):

Detects individual keypoints with low visibility or zero coordinates.

```python
for each landmark in frame:
    x, y, z, vis = landmark

    # Check conditions
    is_zero = (x == 0 and y == 0 and z == 0)
    low_visibility = (vis < VISIBILITY_THRESHOLD)

    if is_zero or low_visibility:
        # Mark landmark as invisible
        x, y, z = UNVISIBLE_LANDMARK, UNVISIBLE_LANDMARK, UNVISIBLE_LANDMARK
```

**Purpose**: Distinguish between "zero" as a valid coordinate vs "missing" data.

### 3. Whole-Clip Normalization

**Method**: Isotropic scaling to unit bounding box

**Process**:
```python
# Collect all VALID (x, y, z) points
valid_points = []
for frame in arr:
    for landmark in frame:
        if landmark not in [UNVISIBLE_FRAME, UNVISIBLE_LANDMARK]:
            valid_points.append(landmark[:3])  # (x, y, z)

# Compute 3D bounding box
min_x, min_y, min_z = np.min(valid_points, axis=0)
max_x, max_y, max_z = np.max(valid_points, axis=0)

# Find maximum range across all dimensions
range_x = max_x - min_x
range_y = max_y - min_y
range_z = max_z - min_z
max_range = max(range_x, range_y, range_z)

# Apply isotropic scaling
for each valid point:
    x_norm = (x - min_x) / max_range
    y_norm = (y - min_y) / max_range
    z_norm = (z - min_z) / max_range

# Sentinel values remain unchanged (-999.0)
```

**Key Properties**:
- **Isotropic**: Same scale factor for x, y, z (preserves aspect ratio)
- **Unit cube**: Normalized landmarks fit in 1×1×1 cube
- **Clip-level**: Single normalization per segment (not per-frame)
- **Preserves missing data**: Sentinel values stay at -999.0

### 4. Optional Z-Coordinate Removal

If `REMOVE_Z = True`:

```python
# Drop z-coordinate
arr = arr[:, :, :2]  # Keep only (x, y)

# Shape: (T, 85, 2)
```

**Use Cases**:
- 2D-only models
- Reduce storage by 33%
- Ignore unreliable depth from monocular video

### 5. Flatten Per Frame

**Process**:
```python
# Reshape (T, K, C) → (T, K × C)
T, K, C = arr.shape  # e.g., (72, 85, 3)
arr_flat = arr.reshape(T, K * C)  # (72, 255)
```

**Result**:
- 3D: `(T, 85 × 3) = (T, 255)`
- 2D: `(T, 85 × 2) = (T, 170)`

### 6. Save Normalized Array

```python
# Preserve relative directory structure
relative_path = input_file.relative_to(INPUT_DIR)
output_file = OUTPUT_DIR / relative_path

# Create parent directories
output_file.parent.mkdir(parents=True, exist_ok=True)

# Save
np.save(output_file, arr_flat)
```

---

## Usage

### Basic Execution

```bash
python scripts/4_reduction_normalization.py
```

### Expected Output

```
=== Stage 4: Landmark Normalization ===

Configuration:
  INPUT_DIR: dataset/npy
  OUTPUT_DIR: dataset/npy_normalized
  REMOVE_Z: False
  MAX_WORKERS: 8

Processing landmarks...
[1/850] abc123-000.npy → (72, 255) ✓
[2/850] abc123-001.npy → (58, 255) ✓
[3/850] abc123-002.npy → Skipped (already exists)
...
[850/850] xyz789-042.npy → (94, 255) ✓

Normalization complete!
Processed: 847 files
Skipped: 3 files (already exist)
Output directory: dataset/npy_normalized
```

---

## Output Format

### Normalized Landmark Arrays

**File**: `{OUTPUT_DIR}/{SENTENCE_NAME}.npy`

**Shape**:
- With Z: `(T, 255)` = `(T, 85 × 3)`
- Without Z: `(T, 170)` = `(T, 85 × 2)`

**Value Ranges**:
- **Valid landmarks**: `[0.0, 1.0]` (unit cube normalization)
- **Missing landmarks**: `-999.0` (sentinel value)

**Example**:
```python
import numpy as np

# Load normalized array
arr = np.load("dataset/npy_normalized/abc123-000.npy")
print(arr.shape)  # (72, 255) for 3D

# Reshape back to (T, 85, 3)
arr_3d = arr.reshape(72, 85, 3)

# Access specific keypoint in frame 0
left_shoulder = arr_3d[0, 0, :]  # (x, y, z)
print(left_shoulder)  # [0.45, 0.32, 0.18] or [-999.0, -999.0, -999.0]
```

---

## Sentinel Values Explained

### Why Use Sentinel Values?

**Problem**: How to distinguish between:
1. Landmark at position (0, 0, 0) - a valid coordinate
2. Missing/undetected landmark - invalid data

**Solution**: Use sentinel value `-999.0` for missing data.

### Sentinel Types

**UNVISIBLE_FRAME**: Applied to entire frames with no detections
```python
# All landmarks in frame become -999.0
frame[:, :3] = -999.0
```

**UNVISIBLE_LANDMARK**: Applied to individual low-confidence keypoints
```python
# Single landmark becomes -999.0
landmark[:3] = -999.0
```

### Handling in Models

**Approach 1: Mask in loss function**
```python
# PyTorch example
mask = (landmarks != -999.0)
loss = ((pred - target) * mask).mean()
```

**Approach 2: Replace with zeros**
```python
landmarks[landmarks == -999.0] = 0.0
```

**Approach 3: Interpolate**
```python
# Fill missing landmarks with temporal interpolation
for i in range(len(landmarks)):
    if landmarks[i] == -999.0:
        landmarks[i] = (landmarks[i-1] + landmarks[i+1]) / 2
```

---

## Performance

### Processing Speed

**CPU-based** (typical performance):
- **Speed**: 50-200 files per second
- **Bottleneck**: Disk I/O
- **Scaling**: Linear with `MAX_WORKERS`

**Example Timing**:
```
1000 files @ ~10-100 KB each
Processing time: ~5-30 seconds with 8 workers
```

### Storage Requirements

**Output Size** (same as input):
```
3D: 255 values per frame
2D: 170 values per frame

Typical segment (72 frames):
- 3D: 72 × 255 × 4 bytes = ~73 KB
- 2D: 72 × 170 × 4 bytes = ~49 KB

Storage savings with REMOVE_Z: ~33%
```

---

## Troubleshooting

### Issue: All Landmarks Masked as -999.0

**Symptoms:**
- Output files contain only -999.0 values
- No valid landmarks after normalization

**Solutions:**
1. Check `VISIBILITY_THRESHOLD`:
   ```python
   VISIBILITY_THRESHOLD = 0.3  # Lower threshold
   ```
2. Verify input files have valid visibility scores (channel 3)
3. Inspect raw landmark quality from Stage 3

### Issue: Invalid Shape Errors

**Symptoms:**
- Error: "Cannot reshape array"
- Shape mismatch

**Solutions:**
1. Verify input files are from Stage 3 (shape `(T, 85, 4)`)
2. Check for corrupted `.npy` files
3. Re-run Stage 3 for problematic segments

### Issue: Normalization Range Issues

**Symptoms:**
- Normalized values outside [0, 1]
- Unexpected coordinate distributions

**Solutions:**
1. Check for outliers in raw data
2. Verify sentinel values are correctly applied
3. Inspect normalization logic for edge cases

### Issue: Out of Memory

**Symptoms:**
- Process killed during normalization
- Memory errors

**Solutions:**
1. Reduce `MAX_WORKERS`:
   ```python
   MAX_WORKERS = 2
   ```
2. Process in batches (split `INPUT_DIR`)

---

## Advanced Configuration

### Disable Masking

```python
# Process raw landmarks without masking
MASK_FRAME_LEVEL = False
MASK_LANDMARK_LEVEL = False
```

**Note**: May result in zero coordinates mixed with valid data.

### Adjust Visibility Threshold

```python
# More strict (fewer valid landmarks)
VISIBILITY_THRESHOLD = 0.7

# More lenient (more valid landmarks)
VISIBILITY_THRESHOLD = 0.3

# Accept all non-zero landmarks
VISIBILITY_THRESHOLD = 0.0
```

### Per-Axis Normalization (Not Isotropic)

Edit `scripts/4_reduction_normalization.py`:

```python
# Instead of max_range, use per-axis ranges
x_norm = (x - min_x) / range_x
y_norm = (y - min_y) / range_y
z_norm = (z - min_z) / range_z
```

**Trade-off**: Distorts aspect ratio, but uses full [0, 1] range per dimension.

### Custom Sentinel Values

```python
# Use NaN instead of -999.0
UNVISIBLE_FRAME = float('nan')
UNVISIBLE_LANDMARK = float('nan')

# Use large negative value
UNVISIBLE_FRAME = -1e6
UNVISIBLE_LANDMARK = -1e6
```

---

## Validation

### Check Output Integrity

```python
import numpy as np
from pathlib import Path

output_dir = Path("dataset/npy_normalized")
files = list(output_dir.glob("*.npy"))

for npy_file in files:
    arr = np.load(npy_file)

    # Check shape
    assert arr.ndim == 2, f"Wrong dimensions: {npy_file}"
    assert arr.shape[1] in [170, 255], f"Wrong feature count: {npy_file}"

    # Check value ranges (excluding sentinel)
    valid_mask = arr != -999.0
    valid_values = arr[valid_mask]

    if len(valid_values) > 0:
        assert valid_values.min() >= 0, f"Invalid min: {npy_file}"
        assert valid_values.max() <= 1, f"Invalid max: {npy_file}"

print(f"✓ All {len(files)} files valid")
```

### Statistics

```python
import numpy as np
from pathlib import Path

output_dir = Path("dataset/npy_normalized")
files = list(output_dir.glob("*.npy"))

total_frames = 0
total_valid = 0
total_masked = 0

for npy_file in files:
    arr = np.load(npy_file)
    total_frames += arr.shape[0]
    total_valid += np.sum(arr != -999.0)
    total_masked += np.sum(arr == -999.0)

print(f"Total files: {len(files)}")
print(f"Total frames: {total_frames}")
print(f"Valid values: {total_valid} ({100*total_valid/(total_valid+total_masked):.1f}%)")
print(f"Masked values: {total_masked} ({100*total_masked/(total_valid+total_masked):.1f}%)")
```

### Compare Input vs Output

```python
import numpy as np

# Load raw and normalized
raw = np.load("dataset/npy/abc123-000.npy")  # (T, 85, 4)
norm = np.load("dataset/npy_normalized/abc123-000.npy")  # (T, 255)

# Reshape normalized
T = norm.shape[0]
norm_3d = norm.reshape(T, 85, 3)

print(f"Raw shape: {raw.shape}")
print(f"Normalized shape: {norm.shape}")
print(f"Raw value range: [{raw[:,:,:3].min():.3f}, {raw[:,:,:3].max():.3f}]")
print(f"Norm value range: [{norm[norm != -999.0].min():.3f}, {norm[norm != -999.0].max():.3f}]")
```

---

## Model Integration

### Loading Normalized Data

```python
import numpy as np
import torch

def load_segment(npy_file, device='cuda'):
    """Load normalized landmark segment."""
    arr = np.load(npy_file)  # (T, 255) or (T, 170)

    # Convert to tensor
    tensor = torch.from_numpy(arr).float()

    # Handle sentinel values (optional)
    tensor[tensor == -999.0] = 0.0  # or use masking

    return tensor.to(device)

# Example
segment = load_segment("dataset/npy_normalized/abc123-000.npy")
print(segment.shape)  # torch.Size([72, 255])
```

### Batch Processing

```python
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_segments(npy_files):
    """Collate variable-length segments into batch."""
    segments = []
    masks = []

    for npy_file in npy_files:
        arr = np.load(npy_file)
        tensor = torch.from_numpy(arr).float()

        # Create mask for valid data
        mask = (tensor != -999.0).all(dim=-1)

        segments.append(tensor)
        masks.append(mask)

    # Pad to same length
    segments_padded = pad_sequence(segments, batch_first=True)
    masks_padded = pad_sequence(masks, batch_first=True)

    return segments_padded, masks_padded

# Example
files = ["abc123-000.npy", "abc123-001.npy", "def456-000.npy"]
batch, masks = collate_segments(files)
print(batch.shape)  # (3, max_T, 255)
print(masks.shape)  # (3, max_T)
```

---

## Next Steps

After Stage 4 completes successfully:

1. **Verify Outputs**: Check normalized files
2. **Model Training**: Use normalized landmarks for ASL translation
3. **Data Analysis**: Compute statistics and visualizations

---

## Dependencies

- `numpy`: Array operations
- Python 3.8+

Install:
```bash
pip install numpy
```

---

## Summary

**Stage 4 transforms raw landmarks into model-ready format:**
- ✓ Visibility-based masking
- ✓ Whole-clip unit bounding box normalization
- ✓ Optional z-coordinate removal
- ✓ Flattened per-frame representation
- ✓ Sentinel values for missing data

**Ready for downstream tasks**: ASL translation, recognition, retrieval, etc.
