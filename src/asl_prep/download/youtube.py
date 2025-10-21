"""YouTube video and transcript download utilities."""
import os
import time
import logging
from glob import glob
from typing import Set, Tuple, Dict

from yt_dlp import YoutubeDL
from yt_dlp.utils import (
    DownloadError,
    ExtractorError,
    PostProcessingError,
    UnavailableVideoError,
)
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from youtube_transcript_api.formatters import JSONFormatter
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_existing_ids(directory: str, ext: str) -> Set[str]:
    """
    Return a set of IDs from files with the specified extension in the directory.

    Args:
        directory: Directory to search for files
        ext: File extension without dot (e.g., 'json', 'mp4')

    Returns:
        Set of file IDs (filenames without extensions)

    Examples:
        >>> get_existing_ids("/transcripts", "json")
        {'video1', 'video2', 'video3'}
    """
    files = glob(os.path.join(directory, f"*.{ext}"))
    return {os.path.splitext(os.path.basename(f))[0] for f in files}


def load_video_ids(file_path: str) -> Set[str]:
    """
    Load video IDs from a text file.

    Args:
        file_path: Path to text file containing video IDs (one per line)

    Returns:
        Set of video ID strings

    Examples:
        >>> load_video_ids("video_ids.txt")
        {'dQw4w9WgXcQ', 'jNQXAC9IVRw', ...}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def download_single_transcript(
    video_id: str,
    transcript_dir: str,
    languages: list,
    formatter: JSONFormatter,
    sleep_time: float
) -> Tuple[bool, float]:
    """
    Download a single transcript for a video ID.

    Args:
        video_id: YouTube video ID
        transcript_dir: Directory to save transcript JSON files
        languages: List of language codes to try
        formatter: JSON formatter for transcript output
        sleep_time: Current sleep time for rate limiting

    Returns:
        Tuple of (success: bool, updated_sleep_time: float)

    Examples:
        >>> formatter = JSONFormatter()
        >>> success, new_sleep = download_single_transcript(
        ...     "dQw4w9WgXcQ", "/transcripts", ["en"], formatter, 0.2
        ... )
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        json_transcript = formatter.format_transcript(transcript)
        transcript_path = os.path.join(transcript_dir, f"{video_id}.json")
        with open(transcript_path, "w", encoding="utf-8") as out_file:
            out_file.write(json_transcript)
        logger.info("SUCCESS: Transcript for %s saved.", video_id)
        return True, sleep_time
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        logger.warning("Transcript unavailable for %s: %s", video_id, e)
        return False, sleep_time
    except Exception as e:
        # Catch all other errors including rate limiting
        sleep_time += 0.1  # Slightly increase delay on error
        logger.error("Error downloading transcript for %s. Error: %s", video_id, e)
        return False, sleep_time


def download_transcripts(
    video_id_file: str,
    transcript_dir: str,
    languages: list
) -> None:
    """
    Download transcripts for video IDs if not already saved.

    Args:
        video_id_file: Path to file containing video IDs
        transcript_dir: Directory to save transcript JSON files
        languages: List of language codes to try

    Examples:
        >>> download_transcripts(
        ...     "video_ids.txt",
        ...     "/transcripts",
        ...     ["en", "en-US"]
        ... )
    """
    os.makedirs(transcript_dir, exist_ok=True)
    existing_ids = get_existing_ids(transcript_dir, "json")

    all_ids = load_video_ids(video_id_file)
    ids = list(all_ids - existing_ids)

    if not ids:
        logger.info("All transcripts are already downloaded.")
        return

    formatter = JSONFormatter()
    sleep_time = 0.2
    error_count = 0

    # Use progress bar to show download progress
    with tqdm(ids, desc="Downloading transcripts") as pbar:
        for video_id in pbar:
            sleep_time = min(sleep_time, 2)  # Cap sleep time at 2 seconds
            time.sleep(sleep_time)  # Rate limiting pause
            success, sleep_time = download_single_transcript(
                video_id, transcript_dir, languages, formatter, sleep_time
            )

            if not success:
                error_count += 1

            pbar.set_postfix(errors=error_count)


def download_single_video(
    video_id: str,
    download_options: Dict
) -> bool:
    """
    Download a YouTube video using specified options.

    Args:
        video_id: YouTube video ID
        download_options: yt-dlp configuration dictionary

    Returns:
        True if download succeeded, False otherwise

    Examples:
        >>> options = {"format": "best", "outtmpl": "/videos/%(id)s.%(ext)s"}
        >>> download_single_video("dQw4w9WgXcQ", options)
        True
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with YoutubeDL(download_options) as yt:
            yt.extract_info(video_url)
        logger.info("SUCCESS: Video %s downloaded.", video_id)
        return True
    except (
        DownloadError,
        ExtractorError,
        PostProcessingError,
        UnavailableVideoError,
    ) as e:
        logger.error("Error downloading video %s. Error: %s", video_id, e)
        return False
    except Exception as e:
        logger.error("Unexpected error for %s. Error: %s", video_id, e)
        return False


def download_videos(
    video_id_file: str,
    video_dir: str,
    npy_dir: str,
    download_options: Dict
) -> None:
    """
    Download videos for video IDs if not already downloaded.

    Args:
        video_id_file: Path to file containing video IDs
        video_dir: Directory to save downloaded videos
        npy_dir: Directory where processed files will be saved (used to check completion)
        download_options: yt-dlp configuration dictionary

    Examples:
        >>> options = {"format": "best", "outtmpl": "/videos/%(id)s.%(ext)s"}
        >>> download_videos("video_ids.txt", "/videos", "/npy", options)
    """
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    existing_ids = get_existing_ids(video_dir, "mp4")

    all_ids = load_video_ids(video_id_file)
    ids = list(all_ids - existing_ids)

    if not ids:
        logger.info("All videos have already been downloaded.")
        return

    error_count = 0
    # Use tqdm progress bar to show progress
    with tqdm(ids, desc="Downloading videos", unit="video") as pbar:
        for video_id in pbar:
            time.sleep(0.2)  # Rate limiting pause
            success = download_single_video(video_id, download_options)
            if not success:
                error_count += 1
            pbar.set_postfix(errors=error_count)

    logger.info("Video download completed. Total errors: %d", error_count)
