"""Common pipeline utilities for video segment processing."""
import os
import logging
from typing import List, Tuple, Dict, Optional, Set

import pandas as pd

from ..common.files import get_video_filenames
from ..common.video import validate_video_file, get_video_fps

logger = logging.getLogger(__name__)


def read_manifest_csv(csv_file: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Read manifest CSV and detect timestamp column format.

    The function automatically detects which timestamp columns are available:
    - START/END (standard format)
    - START_REALIGNED/END_REALIGNED (realigned format)

    Args:
        csv_file: Path to the manifest CSV file

    Returns:
        Tuple of (dataframe, start_column_name, end_column_name)

    Raises:
        ValueError: If neither timestamp column format is found

    Examples:
        >>> df, start_col, end_col = read_manifest_csv("manifest.csv")
        >>> print(start_col, end_col)
        START_REALIGNED END_REALIGNED
    """
    data_full = pd.read_csv(csv_file, delimiter="\t", on_bad_lines="skip")
    columns = data_full.columns.tolist()

    if "START" in columns and "END" in columns:
        start_col, end_col = "START", "END"
        logger.info("Using START/END columns for timestamps")
    elif "START_REALIGNED" in columns and "END_REALIGNED" in columns:
        start_col, end_col = "START_REALIGNED", "END_REALIGNED"
        logger.info("Using START_REALIGNED/END_REALIGNED columns for timestamps")
    else:
        raise ValueError(
            "Neither START/END nor START_REALIGNED/END_REALIGNED columns found in CSV"
        )

    return data_full, start_col, end_col


def build_processing_tasks(
    timestamp_data: pd.DataFrame,
    video_dir: str,
    output_dir: str,
    start_col: str,
    end_col: str,
    existing_files: Optional[List[str]] = None,
    min_duration: float = 0.2,
    max_duration: float = 60.0,
    fps_range: Optional[Tuple[float, float]] = None,
) -> Tuple[List[Tuple[str, float, float, str]], Dict[str, int]]:
    """
    Build task list from manifest with validation and filtering.

    Args:
        timestamp_data: DataFrame with segment metadata
        video_dir: Source video directory
        output_dir: Output directory for landmarks
        start_col: Name of start timestamp column
        end_col: Name of end timestamp column
        existing_files: List of already-processed files to skip (optional)
        min_duration: Minimum segment duration in seconds (default: 0.2)
        max_duration: Maximum segment duration in seconds (default: 60.0)
        fps_range: Tuple of (min_fps, max_fps) to filter videos (optional)

    Returns:
        Tuple of:
        - List of (video_path, start_time, end_time, output_path) tuples
        - Dictionary with skip statistics

    Examples:
        >>> df = pd.read_csv("manifest.csv", sep="\\t")
        >>> tasks, stats = build_processing_tasks(
        ...     df, "/videos", "/output", "START", "END",
        ...     fps_range=(24.0, 60.0)
        ... )
        >>> print(f"Tasks to process: {len(tasks)}")
        >>> print(f"Skipped existing: {stats['existing_files']}")
    """
    if existing_files is None:
        existing_files = []

    existing_set = set(existing_files)

    # Create video validation cache
    video_validation_cache = {}
    invalid_videos: Set[str] = set()
    video_fps_cache = {}

    # Statistics
    stats = {
        "existing_files": 0,
        "too_short": 0,
        "too_long": 0,
        "invalid_video": 0,
        "fps_out_of_range": 0,
    }

    processing_tasks = []

    for _, row in timestamp_data.iterrows():
        video_name = row.VIDEO_NAME
        sentence_name = row.SENTENCE_NAME
        start, end = row[start_col], row[end_col]

        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        output_path = os.path.join(output_dir, f"{sentence_name}.npy")

        # Skip if output file already exists
        if sentence_name in existing_set:
            stats["existing_files"] += 1
            continue

        # Segment duration limits
        seg_dur = float(end - start)
        if seg_dur < min_duration:
            stats["too_short"] += 1
            continue
        if seg_dur > max_duration:
            stats["too_long"] += 1
            continue

        # Validate video file (use cache to avoid repeated checks)
        if video_path not in video_validation_cache:
            video_validation_cache[video_path] = validate_video_file(video_path)
            if not video_validation_cache[video_path]:
                invalid_videos.add(video_name)
                logger.warning(f"Invalid or missing video file: {video_path}")

        if not video_validation_cache[video_path]:
            stats["invalid_video"] += 1
            continue

        # Video FPS filtering (if fps_range is specified)
        if fps_range is not None:
            if video_path not in video_fps_cache:
                video_fps_cache[video_path] = get_video_fps(video_path)
            vfps = video_fps_cache[video_path]
            min_fps, max_fps = fps_range
            if vfps <= 0.0 or vfps < float(min_fps) or vfps > float(max_fps):
                stats["fps_out_of_range"] += 1
                continue

        processing_tasks.append((video_path, start, end, output_path))

    # Log summary
    logger.info(f"Task summary:")
    logger.info(f"  - Tasks to process: {len(processing_tasks)}")
    logger.info(f"  - Skipped (existing files): {stats['existing_files']}")
    logger.info(f"  - Skipped (duration < {min_duration}s): {stats['too_short']}")
    logger.info(f"  - Skipped (duration > {max_duration}s): {stats['too_long']}")
    if fps_range:
        logger.info(f"  - Skipped (fps out of {fps_range}): {stats['fps_out_of_range']}")
    logger.info(f"  - Skipped (invalid videos): {stats['invalid_video']}")

    if invalid_videos:
        logger.warning(
            f"Invalid video files found: {', '.join(sorted(invalid_videos))}"
        )

    return processing_tasks, stats
