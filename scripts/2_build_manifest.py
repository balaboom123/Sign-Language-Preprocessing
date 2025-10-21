#!/usr/bin/env python3
"""
Build manifest CSV from downloaded transcripts.

This script processes video transcripts into a segmented CSV manifest.
Each transcript is normalized, filtered based on duration and text length,
and saved as individual segments in a tab-separated CSV file.

Usage:
    python scripts/2_build_manifest.py

Configuration:
    Edit configs/build_manifest.py to change:
    - VIDEO_ID_FILE: Path to file containing video IDs
    - TRANSCRIPT_DIR: Directory containing transcript JSON files
    - OUTPUT_CSV: Path to output manifest CSV file
    - MAX_TEXT_LENGTH: Maximum text length for segments
    - MIN_DURATION, MAX_DURATION: Duration constraints for segments

Output Format:
    Tab-separated CSV with columns:
    - VIDEO_NAME: Source video ID
    - SENTENCE_NAME: Unique segment identifier (video_id-XXX)
    - START_REALIGNED: Segment start time in seconds
    - END_REALIGNED: Segment end time in seconds
    - SENTENCE: Normalized transcript text
"""
import os
import sys
import logging

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configs.build_manifest as cfg
from src.asl_prep.transcripts.preprocess import (
    read_transcript_file,
    process_transcript_segments,
    save_segments_to_csv
)
from src.asl_prep.download.youtube import load_video_ids

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function to process video transcripts into segmented CSV data."""
    logger.info("=" * 80)
    logger.info("ASL Dataset Preprocessor - Step 2: Build Manifest")
    logger.info("=" * 80)

    # Load video IDs
    video_ids = load_video_ids(cfg.VIDEO_ID_FILE)
    logger.info(f"\nProcessing {len(video_ids)} videos...")
    logger.info(f"Transcript directory: {cfg.TRANSCRIPT_DIR}")
    logger.info(f"Output CSV: {cfg.OUTPUT_CSV}")
    logger.info(f"Filtering constraints:")
    logger.info(f"  - Max text length: {cfg.MAX_TEXT_LENGTH} characters")
    logger.info(f"  - Duration range: {cfg.MIN_DURATION}s - {cfg.MAX_DURATION}s\n")

    processed_count = 0
    skipped_count = 0
    total_segments = 0

    for video_id in video_ids:
        try:
            json_file = os.path.join(cfg.TRANSCRIPT_DIR, f"{video_id}.json")

            # Skip if transcript file doesn't exist
            if not os.path.exists(json_file):
                skipped_count += 1
                continue

            # Read and process transcript
            transcript_data = read_transcript_file(json_file)
            if not transcript_data:
                skipped_count += 1
                continue

            # Process transcript segments with filtering
            processed_segments = process_transcript_segments(
                transcripts=transcript_data,
                video_id=video_id,
                max_text_length=cfg.MAX_TEXT_LENGTH,
                min_duration=cfg.MIN_DURATION,
                max_duration=cfg.MAX_DURATION
            )

            # Save segments to CSV if any valid segments exist
            if processed_segments:
                save_segments_to_csv(processed_segments, cfg.OUTPUT_CSV)
                processed_count += 1
                total_segments += len(processed_segments)
                logger.info(
                    f"Processed {video_id}: {len(processed_segments)} segments"
                )
            else:
                skipped_count += 1

        except Exception as e:
            logger.error(f"Error processing {video_id}: {e}")
            skipped_count += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Processing Summary:")
    logger.info(f"  - Videos processed: {processed_count}")
    logger.info(f"  - Videos skipped: {skipped_count}")
    logger.info(f"  - Total segments created: {total_segments}")
    logger.info(f"  - Manifest saved to: {cfg.OUTPUT_CSV}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
