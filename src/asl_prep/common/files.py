"""Common file and directory utilities for ASL preprocessing pipeline."""
import os
from glob import glob
from typing import List


def get_video_filenames(directory: str, pattern: str = "*.mp4") -> List[str]:
    """
    Retrieves video filenames from specified directory without extensions.

    Args:
        directory: Path to directory containing files
        pattern: File pattern to match (default: "*.mp4")

    Returns:
        List of filenames without extensions

    Examples:
        >>> get_video_filenames("/videos", "*.mp4")
        ['video1', 'video2', 'video3']
    """
    return [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(directory, pattern))
    ]


def get_filenames(directory: str, pattern: str, extension: str) -> List[str]:
    """
    Generic version - retrieves filenames with any pattern/extension.

    Args:
        directory: Path to directory containing files
        pattern: File pattern to match (e.g., "*", "video_*")
        extension: File extension without dot (e.g., "npy", "json")

    Returns:
        List of filenames without extensions

    Examples:
        >>> get_filenames("/data", "*", "npy")
        ['landmarks1', 'landmarks2']
    """
    search_pattern = f"{pattern}.{extension}"
    return [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(directory, search_pattern))
    ]
