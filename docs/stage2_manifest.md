# Stage 2: Build Manifest from Transcripts

## Overview

Converts raw YouTube transcript JSON files into a tab-separated manifest CSV containing clean caption segments with start/end times, ready for landmark extraction.

---

## Files

- **Config**: `configs/build_manifest.py`
- **Script**: `scripts/2_build_manifest.py`

---

## Configuration Reference

### Path Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `ROOT` | Path | Project root directory | Auto-detected |
| `TRANSCRIPT_DIR` | Path | Directory with transcript `.json` files | `{ROOT}/dataset/transcript` |
| `VIDEO_ID_FILE` | Path | List of video IDs to process | `{ROOT}/assets/youtube-asl_youtube_asl_video_ids.txt` |
| `OUTPUT_CSV` | Path | Output manifest file path | `{ROOT}/assets/youtube_asl.csv` |

### Filtering Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `MAX_TEXT_LENGTH` | int | Maximum caption length (characters) | `300` |
| `MIN_DURATION` | float | Minimum caption duration (seconds) | `0.2` |
| `MAX_DURATION` | float | Maximum caption duration (seconds) | `60.0` |

---

## Workflow

### 1. Load Video IDs

**Process:**
- Read `VIDEO_ID_FILE` line by line
- Strip whitespace and empty lines
- Build list of target video IDs

### 2. Process Each Transcript

**For each video ID:**

1. **Load Transcript JSON**:
   ```python
   # Load from: {TRANSCRIPT_DIR}/{video_id}.json
   [
     {"text": "Hello", "start": 0.0, "duration": 2.5},
     {"text": "World", "start": 2.5, "duration": 1.8}
   ]
   ```

2. **Normalize Text**:
   - Fix unicode characters (NFKC normalization)
   - Collapse multiple whitespaces to single space
   - Strip leading/trailing whitespace

3. **Filter Segments**:
   - **Text length**: `len(text) <= MAX_TEXT_LENGTH`
   - **Duration**: `MIN_DURATION <= duration <= MAX_DURATION`
   - Discard segments that don't meet criteria

4. **Generate Segment Names**:
   ```python
   # Format: {video_id}-{index:03d}
   # Examples:
   #   abc123-000
   #   abc123-001
   #   def456-000
   ```

5. **Build Row**:
   ```python
   {
     "VIDEO_NAME": video_id,
     "SENTENCE_NAME": f"{video_id}-{index:03d}",
     "START_REALIGNED": segment["start"],
     "END_REALIGNED": segment["start"] + segment["duration"],
     "SENTENCE": normalized_text
   }
   ```

### 3. Write Manifest CSV

**Process:**
- Append all rows to `OUTPUT_CSV`
- Use tab separator (`\t`)
- Write header: `VIDEO_NAME`, `SENTENCE_NAME`, `START_REALIGNED`, `END_REALIGNED`, `SENTENCE`

---

## Usage

### Basic Execution

```bash
python scripts/2_build_manifest.py
```

### Expected Output

```
=== Stage 2: Build Manifest ===

Configuration:
  TRANSCRIPT_DIR: dataset/transcript
  VIDEO_ID_FILE: assets/youtube-asl_youtube_asl_video_ids.txt
  OUTPUT_CSV: assets/youtube_asl.csv
  MAX_TEXT_LENGTH: 300
  MIN_DURATION: 0.2
  MAX_DURATION: 60.0

Processing transcripts...
✓ abc123: 45 segments
✓ def456: 38 segments
⊘ xyz789: Transcript not found
✓ ghi012: 52 segments

Manifest complete!
Total videos processed: 3
Total segments: 135
Output: assets/youtube_asl.csv
```

---

## Output Format

### Manifest CSV Structure

**Format**: Tab-separated values (TSV)

**Columns**:
1. `VIDEO_NAME`: Original YouTube video ID
2. `SENTENCE_NAME`: Unique segment identifier
3. `START_REALIGNED`: Start time in seconds (float)
4. `END_REALIGNED`: End time in seconds (float)
5. `SENTENCE`: Normalized caption text

**Example**:
```tsv
VIDEO_NAME	SENTENCE_NAME	START_REALIGNED	END_REALIGNED	SENTENCE
abc123	abc123-000	0.0	2.5	Hello everyone
abc123	abc123-001	2.5	5.3	Welcome to my channel
def456	def456-000	1.2	4.8	Today we will learn
def456	def456-001	4.8	8.6	About sign language
```

### Segment Naming Convention

```
{video_id}-{segment_index:03d}

Examples:
  abc123xyz-000  # First segment of video abc123xyz
  abc123xyz-001  # Second segment
  abc123xyz-042  # 43rd segment
```

---

## Filtering Logic

### Text Length Filter

**Rule**: `len(text) <= MAX_TEXT_LENGTH`

**Purpose**: Remove excessively long captions that may span multiple sentences or contain errors

**Example**:
```python
MAX_TEXT_LENGTH = 300

# Accepted
"This is a short caption."  # 26 characters ✓

# Rejected
"This is an extremely long caption that goes on and on..."  # 350 characters ✗
```

### Duration Filter

**Rule**: `MIN_DURATION <= duration <= MAX_DURATION`

**Purpose**:
- Remove too-short segments (likely errors)
- Remove too-long segments (not suitable for ASL translation)

**Example**:
```python
MIN_DURATION = 0.2  # 200ms
MAX_DURATION = 60.0  # 1 minute

# Accepted
duration = 2.5   # ✓
duration = 45.0  # ✓

# Rejected
duration = 0.1   # ✗ Too short
duration = 75.0  # ✗ Too long
```

---

## Troubleshooting

### Issue: Missing Transcripts

**Symptoms:**
- Warning: "Transcript not found: {video_id}"
- Fewer segments than expected

**Solutions:**
1. Verify Stage 1 completed successfully
2. Check `TRANSCRIPT_DIR` for missing `.json` files
3. Re-run Stage 1 for missing videos

### Issue: No Segments Generated

**Symptoms:**
- Video processed but 0 segments output
- All segments filtered out

**Solutions:**
1. Check transcript JSON is not empty
2. Review filtering parameters:
   - Increase `MAX_TEXT_LENGTH` if captions are long
   - Decrease `MIN_DURATION` if segments are short
   - Increase `MAX_DURATION` if segments are long
3. Inspect raw transcript file for issues

### Issue: Empty Manifest File

**Symptoms:**
- `OUTPUT_CSV` contains only header
- No data rows

**Solutions:**
1. Verify `VIDEO_ID_FILE` contains valid IDs
2. Check transcripts exist in `TRANSCRIPT_DIR`
3. Review filtering criteria (may be too restrictive)

### Issue: Unicode/Encoding Errors

**Symptoms:**
- Garbled characters in manifest
- Encoding exceptions

**Solutions:**
1. Script uses NFKC normalization automatically
2. Check transcript JSON encoding (should be UTF-8)
3. Review `SENTENCE` column for corrupted text

---

## Performance Tips

### Processing Speed

- **Typical**: 100-200 videos per minute
- **Bottleneck**: Disk I/O for reading JSON files
- **Optimization**: Use SSD for faster file access

### Storage Requirements

- **Input**: Transcript JSONs (~10-50 KB each)
- **Output**: Manifest CSV (~1-10 MB for 1000 videos)
- **Minimal storage impact**

---

## Advanced Configuration

### Custom Duration Filters

```python
# Very short segments only (0.1 - 5 seconds)
MIN_DURATION = 0.1
MAX_DURATION = 5.0

# Long-form content (5 - 120 seconds)
MIN_DURATION = 5.0
MAX_DURATION = 120.0

# No upper limit
MIN_DURATION = 0.2
MAX_DURATION = float('inf')
```

### Custom Text Length

```python
# Short captions only
MAX_TEXT_LENGTH = 100

# Long-form captions
MAX_TEXT_LENGTH = 1000

# No limit
MAX_TEXT_LENGTH = float('inf')
```

### Process Subset of Videos

**Option 1**: Edit `VIDEO_ID_FILE`
```bash
# Create subset file
head -n 100 assets/youtube-asl_youtube_asl_video_ids.txt > assets/subset.txt

# Update config
VIDEO_ID_FILE = ROOT / "assets" / "subset.txt"
```

**Option 2**: Filter in script
```python
# In scripts/2_build_manifest.py
video_ids = video_ids[:100]  # First 100 only
```

---

## Validation

### Verify Manifest Integrity

```python
import pandas as pd

# Load manifest
df = pd.read_csv("assets/youtube_asl.csv", sep="\t")

# Check structure
assert list(df.columns) == [
    "VIDEO_NAME", "SENTENCE_NAME",
    "START_REALIGNED", "END_REALIGNED", "SENTENCE"
]

# Check timestamps
assert (df["END_REALIGNED"] > df["START_REALIGNED"]).all()

# Check segment names
assert df["SENTENCE_NAME"].str.match(r"^.+-\d{3}$").all()

print(f"✓ Manifest valid: {len(df)} segments")
```

### Statistics

```python
import pandas as pd

df = pd.read_csv("assets/youtube_asl.csv", sep="\t")

print(f"Total segments: {len(df)}")
print(f"Unique videos: {df['VIDEO_NAME'].nunique()}")
print(f"Avg segments per video: {len(df) / df['VIDEO_NAME'].nunique():.1f}")
print(f"Duration range: {df['END_REALIGNED'] - df['START_REALIGNED']}")
print(f"Text length range: {df['SENTENCE'].str.len().describe()}")
```

---

## Next Steps

After Stage 2 completes successfully:

1. **Verify Manifest**: Check segment counts and content
2. **Proceed to Stage 3a/3b**: Extract landmarks
   ```bash
   # MediaPipe (CPU)
   python scripts/3a_extract_mediapipe.py

   # OR MMPose (GPU)
   python scripts/3b_extract_mmpose.py
   ```

---

## Dependencies

- Python 3.8+
- Standard library only (no external packages)

**Optional for validation:**
- `pandas`: CSV analysis and validation
