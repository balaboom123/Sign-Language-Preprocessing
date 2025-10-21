"""Configuration for YouTube video and transcript download (scripts/1_download_data.py)"""
import os

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base paths (ROOT is project root directory)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "dataset", "origin")
TRANSCRIPT_DIR = os.path.join(ROOT, "dataset", "transcript")

# Dataset source files (video IDs to download)
VIDEO_ID_FILE = os.path.join(ROOT, "assets", "youtube-asl_youtube_asl_video_ids.txt")

# =============================================================================
# LANGUAGE SETTINGS
# =============================================================================

# Supported languages for transcript download
LANGUAGE = [
    "en",       # English
    "ase",      # American Sign Language
    "en-US",    # English (United States)
    "en-CA",    # English (Canada)
    "en-GB",    # English (United Kingdom)
    "en-AU",    # English (Australia)
    "en-NZ",    # English (New Zealand)
    "en-IN",    # English (India)
    "en-ZA",    # English (South Africa)
    "en-IE",    # English (Ireland)
    "en-SG",    # English (Singapore)
    "en-PH",    # English (Philippines)
    "en-NG",    # English (Nigeria)
    "en-PK",    # English (Pakistan)
    "en-JM",    # English (Jamaica)
]

# =============================================================================
# YOUTUBE DOWNLOAD CONFIGURATION
# =============================================================================

YT_CONFIG = {
    # Video quality and format
    # Use worst quality video with height >= 720p, or best video with height <= 480p
    "format": "worstvideo[height>=720]/bestvideo[height<=480]",

    # Subtitle settings
    "writesubtitles": False,

    # Output template (saves as: {video_id}.{ext} in VIDEO_DIR)
    "outtmpl": os.path.join(VIDEO_DIR, "%(id)s.%(ext)s"),

    # Connection and security settings
    "nocheckcertificate": True,  # Skip SSL certificate verification
    "geo-bypass": True,           # Bypass geographic restrictions
    "limit_rate": "5M",           # Limit download rate to 5 MB/s
    "http-chunk-size": 10485760,  # 10MB chunks for downloads

    # Playlist and metadata settings
    "noplaylist": True,           # Don't download playlists, only single videos
    "no-metadata-json": True,     # Don't save metadata JSON files
    "no-metadata": True,          # Don't embed metadata in video files

    # Performance optimization
    "concurrent-fragments": 5,    # Download 5 fragments concurrently
    "hls-prefer-ffmpeg": True,    # Use ffmpeg for HLS streams
    "sleep-interval": 0,          # No sleep between downloads (rate limiting handled externally)
}
