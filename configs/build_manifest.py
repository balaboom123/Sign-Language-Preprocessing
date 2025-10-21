"""Configuration for transcript preprocessing (scripts/2_build_manifest.py)"""
import os

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPT_DIR = os.path.join(ROOT, "dataset", "transcript")

# Dataset source files
VIDEO_ID_FILE = os.path.join(ROOT, "assets", "youtube-asl_youtube_asl_video_ids.txt")

# Output manifest CSV file (contains processed transcript segments)
OUTPUT_CSV = os.path.join(ROOT, "assets", "youtube_asl.csv")

# =============================================================================
# TEXT FILTERING CONSTRAINTS
# =============================================================================

# Maximum text length in characters for a single segment
# Longer segments will be filtered out to ensure manageable sizes
MAX_TEXT_LENGTH = 300  # characters

# Duration constraints for transcript segments
MIN_DURATION = 0.2   # seconds (200ms minimum)
MAX_DURATION = 60.0  # seconds (1 minute maximum)

# Segments outside these duration bounds will be filtered out
# This ensures video segments are neither too short (unstable) nor too long (memory issues)
