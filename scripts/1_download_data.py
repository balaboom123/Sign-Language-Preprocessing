#!/usr/bin/env python3
"""
Download YouTube videos and transcripts for ASL dataset.

This script downloads YouTube transcripts and videos for video IDs specified
in the configuration file. Transcripts are downloaded first (if not already saved)
and then videos are downloaded (if not already present).

Usage:
    python scripts/1_download_data.py

Configuration:
    Edit configs/download.py to change:
    - VIDEO_ID_FILE: Path to file containing video IDs
    - VIDEO_DIR: Directory to save downloaded videos
    - TRANSCRIPT_DIR: Directory to save transcript JSON files
    - YT_CONFIG: YouTube download settings (quality, rate limiting, etc.)
    - LANGUAGE: Supported languages for transcript download
"""
import logging
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configs.download as cfg
from src.asl_prep.download.youtube import download_transcripts, download_videos

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function to orchestrate transcript and video downloads."""
    logger.info("=" * 80)
    logger.info("ASL Dataset Downloader - Step 1: Download Data")
    logger.info("=" * 80)

    # Download transcripts
    logger.info("\nStarting transcript download...")
    download_transcripts(
        video_id_file=cfg.VIDEO_ID_FILE,
        transcript_dir=cfg.TRANSCRIPT_DIR,
        languages=cfg.LANGUAGE
    )
    logger.info("Transcript download completed.\n")

    # Download videos
    logger.info("Starting video download...")
    download_videos(
        video_id_file=cfg.VIDEO_ID_FILE,
        video_dir=cfg.VIDEO_DIR,
        npy_dir=os.path.join(cfg.ROOT, "dataset", "npy"),
        download_options=cfg.YT_CONFIG
    )
    logger.info("Video download completed.")

    logger.info("\n" + "=" * 80)
    logger.info("Download process finished successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
