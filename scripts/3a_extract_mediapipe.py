#!/usr/bin/env python3
"""
Extract MediaPipe holistic landmarks from video segments.

This script processes video segments defined in a manifest CSV file and extracts
holistic body landmarks (pose, face, hands) using MediaPipe. The landmarks are
saved as NumPy arrays for each segment.

Usage:
    python scripts/3a_extract_mediapipe.py

Configuration:
    Edit configs/extract_mediapipe.py to change:
    - CSV_FILE: Path to manifest CSV with video segments
    - VIDEO_DIR: Directory containing source videos
    - NPY_DIR: Directory to save landmark arrays
    - REDUCE_FPS_TO: Target FPS for downsampling (None to disable)
    - FRAME_SKIP: Skip every Nth frame (when not using REDUCE_FPS_TO)
    - MAX_WORKERS: Number of parallel worker processes
    - POSE_IDX, FACE_IDX, HAND_IDX: Landmark indices to extract

Output Format:
    NumPy arrays (.npy) with shape (T, 255) where:
    - T: Number of frames
    - 255: Flattened landmarks (6 pose + 37 face + 21 left_hand + 21 right_hand) × 3 coords
"""
import os
import sys
import logging
import multiprocessing
import gc
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import psutil

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configs.extract_mediapipe as cfg
from src.asl_prep.pipeline.processor import read_manifest_csv, build_processing_tasks
from src.asl_prep.common.files import get_video_filenames
from src.asl_prep.common.video import FPSSampler
from src.asl_prep.extractors.mediapipe import MediaPipeExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_video_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    output_file: str
) -> None:
    """
    Process a video segment to extract holistic keypoints using MediaPipe.

    Args:
        video_path: Path to the input video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_file: Path to save the output .npy file
    """
    cap = None
    extractor = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video: {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        # Create FPS sampler: prioritize REDUCE_FPS_TO, otherwise use FRAME_SKIP
        target_fps = None if cfg.REDUCE_FPS_TO is None else float(cfg.REDUCE_FPS_TO)
        sampler = FPSSampler(src_fps=fps, reduce_to=target_fps, frame_skip_by=cfg.FRAME_SKIP)

        # Calculate frame ranges
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Create MediaPipe extractor
        extractor = MediaPipeExtractor(
            pose_idx=cfg.POSE_IDX,
            face_idx=cfg.FACE_IDX,
            hand_idx=cfg.HAND_IDX
        )

        landmark_sequences = []
        current_frame = start_frame

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Use sampler to decide whether to process this frame
            if sampler.take():
                landmarks = extractor.process_frame(frame)
                if landmarks is not None:
                    landmark_sequences.append(landmarks)

            current_frame += 1

        # Save landmarks if valid data exists
        if landmark_sequences:
            landmark_array = np.array(landmark_sequences)
            if landmark_array.size > 0 and np.any(landmark_array):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                np.save(output_file, landmark_array)
                logger.info(
                    f"Saved {len(landmark_sequences)} frames to {output_file}"
                )
            else:
                logger.info(f"No valid landmarks for {video_path}, not saving.")
        else:
            logger.info(f"No landmarks detected for {video_path}")

    except Exception as e:
        logger.error(f"Error processing {video_path}: {str(e)}")

    finally:
        # Resource cleanup
        if cap is not None:
            cap.release()
        if extractor is not None:
            extractor.close()

        # Force garbage collection
        gc.collect()

        # Log memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.debug(
            f"Memory usage after processing: {memory_info.rss / 1024 / 1024:.2f} MB"
        )


def process_batch(task_batch):
    """
    Process a batch of video segment tasks.

    Args:
        task_batch: List of tuples (video_path, start_time, end_time, output_path)
    """
    for video_path, start, end, output_path in task_batch:
        try:
            process_video_segment(video_path, start, end, output_path)
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")

        # Small delay to avoid resource contention
        time.sleep(0.1)


def main():
    """Main function to orchestrate video processing and landmark extraction."""
    logger.info("=" * 80)
    logger.info("ASL Dataset Preprocessor - Step 3a: Extract MediaPipe Landmarks")
    logger.info("=" * 80)

    # Read manifest CSV
    logger.info(f"\nReading manifest: {cfg.CSV_FILE}")
    timestamp_data_full, start_col, end_col = read_manifest_csv(cfg.CSV_FILE)

    # Select required columns and drop missing values
    timestamp_data = timestamp_data_full[
        ["VIDEO_NAME", "SENTENCE_NAME", start_col, end_col]
    ].dropna()

    # Get existing files
    video_files = get_video_filenames(cfg.VIDEO_DIR, pattern="*.mp4")
    processed_files = get_video_filenames(cfg.NPY_DIR, pattern="*.npy")

    logger.info(f"Found {len(video_files)} video files")
    logger.info(f"Configuration:")
    logger.info(f"  - FPS reduction: {cfg.REDUCE_FPS_TO}")
    logger.info(f"  - Frame skip: {cfg.FRAME_SKIP}")
    logger.info(f"  - FPS range filter: {cfg.ACCEPT_VIDEO_FPS_WITHIN}")
    logger.info(f"  - Max workers: {cfg.MAX_WORKERS}\n")

    # Build processing tasks
    processing_tasks, stats = build_processing_tasks(
        timestamp_data=timestamp_data,
        video_dir=cfg.VIDEO_DIR,
        output_dir=cfg.NPY_DIR,
        start_col=start_col,
        end_col=end_col,
        existing_files=processed_files,
        min_duration=0.2,
        max_duration=60.0,
        fps_range=cfg.ACCEPT_VIDEO_FPS_WITHIN,
    )

    if not processing_tasks:
        logger.info("\nNo tasks to process. Exiting.")
        return

    # Process in batches
    BATCH_SIZE = 100
    MAX_WORKERS = min(cfg.MAX_WORKERS, multiprocessing.cpu_count() - 1)

    logger.info(f"\nStarting parallel processing with {MAX_WORKERS} workers...\n")

    for i in range(0, len(processing_tasks), BATCH_SIZE):
        batch = processing_tasks[i:i + BATCH_SIZE]
        logger.info(
            f"Processing batch {i // BATCH_SIZE + 1}, "
            f"tasks {i + 1} to {min(i + BATCH_SIZE, len(processing_tasks))}"
        )

        # Subdivide batches for different processes
        tasks_per_worker = len(batch) // MAX_WORKERS + 1
        worker_batches = [
            batch[j:j + tasks_per_worker]
            for j in range(0, len(batch), tasks_per_worker)
        ]

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for worker_batch in worker_batches:
                if worker_batch:
                    future = executor.submit(process_batch, worker_batch)
                    futures.append(future)

            # Wait for completion
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in worker process: {str(e)}")

        # Delay between batches
        time.sleep(0.5)
        gc.collect()

        # Log progress
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(
            f"Batch {i // BATCH_SIZE + 1} completed. "
            f"Memory: {memory_info.rss / 1024 / 1024:.2f} MB\n"
        )

    logger.info("=" * 80)
    logger.info("MediaPipe landmark extraction completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
