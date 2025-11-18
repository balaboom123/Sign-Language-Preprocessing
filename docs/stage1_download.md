# Stage 1: Download YouTube Videos & Transcripts

## Overview

Downloads YouTube-ASL videos and user-generated transcripts from a list of video IDs. Handles rate-limiting and automatically resumes by skipping already downloaded files.

---

## Files

- **Config**: `configs/download.py`
- **Script**: `scripts/1_download_data.py`

---

## Configuration Reference

### Path Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `ROOT` | Path | Project root directory | Auto-detected |
| `VIDEO_DIR` | Path | Output directory for `.mp4` videos | `{ROOT}/dataset/origin` |
| `TRANSCRIPT_DIR` | Path | Output directory for transcript `.json` files | `{ROOT}/dataset/transcript` |
| `VIDEO_ID_FILE` | Path | Text file with one YouTube video ID per line | `{ROOT}/assets/youtube-asl_youtube_asl_video_ids.txt` |

### Download Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `LANGUAGE` | List[str] | Language codes for transcript requests | `["en", "ase", "en-US", ...]` |
| `YT_CONFIG` | Dict | `yt-dlp` download options | See below |

### YT_CONFIG Options

```python
YT_CONFIG = {
    'format': 'worstvideo[height>=720]/bestvideo[height<=480]',  # Video quality
    'outtmpl': str(VIDEO_DIR / '%(id)s.%(ext)s'),               # Output template
    'skip_unavailable_fragments': True,                          # Skip broken chunks
    'nocheckcertificate': True,                                  # Bypass SSL checks
    'no_warnings': True,                                         # Suppress warnings
    'noplaylist': True,                                          # Download single videos only
    'ignoreerrors': True,                                        # Continue on errors
    'quiet': False,                                              # Show progress
    'no_progress': False,                                        # Show progress bar
    'limit_rate': '5M',                                          # Bandwidth limit (5 MB/s)
    'http_chunk_size': 10485760,                                # Chunk size (10 MB)
    'writesubtitles': True,                                      # Download subtitles
    'writeautomaticsub': True,                                   # Download auto-generated subs
}
```

---

## Workflow

### 1. Download Transcripts

**Process:**
1. Read video IDs from `VIDEO_ID_FILE`
2. For each video ID:
   - Check if `{video_id}.json` exists in `TRANSCRIPT_DIR`
   - If exists: Skip
   - If missing: Request transcript using `youtube_transcript_api`
   - Try languages in order from `LANGUAGE` list
   - Save first successful transcript as JSON

**Output Format:**
```json
[
  {
    "text": "Hello world",
    "start": 0.0,
    "duration": 2.5
  },
  {
    "text": "This is a transcript",
    "start": 2.5,
    "duration": 3.0
  }
]
```

### 2. Download Videos

**Process:**
1. Read video IDs from `VIDEO_ID_FILE`
2. For each video ID:
   - Check if `{video_id}.mp4` exists in `VIDEO_DIR`
   - If exists: Skip
   - If missing: Download using `yt-dlp` with `YT_CONFIG` settings
   - Save as `{video_id}.mp4`

**Rate Limiting:**
- Bandwidth limited to `5M` (5 MB/s) by default
- Adjust `limit_rate` in config for faster/slower downloads
- Automatic retry on transient failures

---

## Usage

### Basic Execution

```bash
python scripts/1_download_data.py
```

### Expected Output

```
=== Stage 1: Download Videos & Transcripts ===

Downloading transcripts...
Processing video ID: abc123
✓ Transcript saved: dataset/transcript/abc123.json
Processing video ID: def456
⊘ Transcript already exists: dataset/transcript/def456.json

Downloading videos...
Processing video ID: abc123
[download] Downloading video...
[download] 100% of 15.3MiB in 00:03
✓ Video saved: dataset/origin/abc123.mp4
Processing video ID: def456
⊘ Video already exists: dataset/origin/def456.mp4

Download complete!
Total transcripts: 2
Total videos: 2
```

---

## Output Files

### Directory Structure

```
dataset/
├── origin/
│   ├── {video_id_1}.mp4
│   ├── {video_id_2}.mp4
│   └── ...
└── transcript/
    ├── {video_id_1}.json
    ├── {video_id_2}.json
    └── ...
```

### File Naming

- **Videos**: `{video_id}.mp4` (e.g., `abc123xyz.mp4`)
- **Transcripts**: `{video_id}.json` (e.g., `abc123xyz.json`)

---

## Troubleshooting

### Issue: Video Unavailable

**Symptoms:**
- Error: "Video unavailable"
- Download skipped for certain IDs

**Solutions:**
1. Check video is public and not deleted
2. Verify video ID is correct in `VIDEO_ID_FILE`
3. Try accessing video in browser

### Issue: Transcript Not Found

**Symptoms:**
- Error: "No transcripts available"
- Missing transcript files

**Solutions:**
1. Verify video has user-generated captions
2. Check `LANGUAGE` list includes target languages
3. Some videos only have auto-generated captions (requires `writeautomaticsub`)

### Issue: Rate Limiting / Too Many Requests

**Symptoms:**
- HTTP 429 errors
- Downloads failing after many successful ones

**Solutions:**
1. Increase `limit_rate` delay in `YT_CONFIG`
2. Add delays between download batches
3. Split `VIDEO_ID_FILE` into smaller chunks

### Issue: SSL Certificate Errors

**Symptoms:**
- SSL verification failures
- Certificate errors during download

**Solutions:**
1. Ensure `nocheckcertificate: True` in `YT_CONFIG`
2. Update `yt-dlp`: `pip install -U yt-dlp`
3. Check system certificates

### Issue: Corrupted Downloads

**Symptoms:**
- Incomplete video files
- Playback errors

**Solutions:**
1. Delete partial files and re-run script
2. Adjust `http_chunk_size` in `YT_CONFIG`
3. Check disk space availability

---

## Performance Tips

### Faster Downloads
- Increase `limit_rate` (e.g., `"10M"` or `"50M"`)
- Use wired internet connection
- Download during off-peak hours

### Storage Management
- YouTube-ASL videos: ~50-200 MB each
- Transcripts: ~10-50 KB each
- Estimate: `num_videos × 100 MB` for total storage

### Parallel Processing
- Script runs sequentially by default
- For parallel downloads, split `VIDEO_ID_FILE` and run multiple instances
- Be careful with rate limiting

---

## Advanced Configuration

### Custom Video Quality

```python
# High quality (720p+)
'format': 'bestvideo[height>=720]+bestaudio/best[height>=720]'

# Low quality (360p)
'format': 'worstvideo[height>=360]/bestvideo[height<=360]'

# Audio only
'format': 'bestaudio/best'
```

### Custom Language Priority

```python
# Prioritize ASL, then English
LANGUAGE = ["ase", "en", "en-US"]

# English only
LANGUAGE = ["en"]
```

### Skip Transcripts or Videos

Edit `scripts/1_download_data.py`:

```python
# Skip transcripts
# download_transcripts(...)  # Comment out

# Skip videos
# download_videos(...)  # Comment out
```

---

## Next Steps

After Stage 1 completes successfully:

1. **Verify Downloads**: Check file counts and sizes
2. **Proceed to Stage 2**: Build manifest from transcripts
   ```bash
   python scripts/2_build_manifest.py
   ```

---

## Dependencies

- `yt-dlp`: YouTube video downloader
- `youtube-transcript-api`: Transcript fetcher
- Python 3.8+

Install:
```bash
pip install yt-dlp youtube-transcript-api
```
