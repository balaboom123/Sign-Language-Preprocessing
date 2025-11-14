"""Transcript text normalization and segmentation."""
import os
import csv
import json
import re
from typing import List, Dict

import ftfy
import pandas as pd


def normalize_text(text: str) -> str:
    """
    Keep semantic content; only fix mojibake and whitespace.
    Do NOT remove non-ASCII characters or bracketed content.

    This function:
    1. Corrects mojibake and other Unicode encoding errors (ftfy)
    2. Normalizes whitespace (newlines to spaces, collapse multiple spaces)

    It explicitly PRESERVES:
    - Original case (no lowercasing)
    - Punctuation
    - Non-ASCII characters (for multilingual content)
    - Bracketed content (may contain semantic information)

    Args:
        text: Input text to be normalized

    Returns:
        Text with fixed encoding and normalized whitespace

    Examples:
        >>> normalize_text("Hello  World\\n")
        'Hello World'
        >>> normalize_text("CafÃ©")  # mojibake
        'Café'
    """
    # Fix mojibake and ASCII issues
    text = ftfy.fix_text(text)

    # Normalize newlines to spaces
    text = text.replace("\n", " ").replace("\r", " ")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def read_transcript_file(json_file: str) -> List[Dict]:
    """
    Read and parse a JSON transcript file.

    Args:
        json_file: Path to JSON transcript file

    Returns:
        Transcript data as a list of dictionaries, each containing:
        - text: Transcript text
        - start: Start time in seconds
        - duration: Duration in seconds

    Examples:
        >>> read_transcript_file("/transcripts/video123.json")
        [{'text': 'Hello', 'start': 0.0, 'duration': 1.5}, ...]
    """
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)


def process_transcript_segments(
    transcripts: List[Dict],
    video_id: str,
    max_text_length: int = 300,
    min_duration: float = 0.2,
    max_duration: float = 60.0
) -> List[Dict]:
    """
    Process individual transcript captions with filtering based on constraints.

    Applies filtering criteria:
    - Text length <= max_text_length characters
    - Duration between min_duration and max_duration seconds
    - Non-empty text after normalization

    Args:
        transcripts: List of transcript dictionaries
        video_id: Video identifier for naming segments
        max_text_length: Maximum allowed text length in characters
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds

    Returns:
        List of processed transcript dictionaries meeting the criteria

    Examples:
        >>> transcripts = [
        ...     {'text': 'Hello world', 'start': 0.0, 'duration': 2.0},
        ...     {'text': 'Too short', 'start': 2.0, 'duration': 0.1},
        ... ]
        >>> process_transcript_segments(transcripts, "video123")
        [{'VIDEO_NAME': 'video123', 'SENTENCE_NAME': 'video123-000', ...}]
    """
    processed_segments = []
    segment_index = 0

    # Filter valid transcript entries
    valid_entries = [
        t for t in transcripts
        if "text" in t and "start" in t and "duration" in t
    ]

    if not valid_entries:
        return processed_segments

    for entry in valid_entries:
        # Normalize the text
        processed_text = normalize_text(entry["text"])

        # Apply filtering criteria
        if (len(processed_text) <= max_text_length and
                min_duration <= entry["duration"] <= max_duration and
                processed_text):  # Ensure non-empty text

            segment_data = {
                "VIDEO_NAME": video_id,
                "SENTENCE_NAME": f"{video_id}-{segment_index:03d}",
                "START_REALIGNED": entry["start"],
                "END_REALIGNED": entry["start"] + entry["duration"],
                "SENTENCE": processed_text,
            }
            processed_segments.append(segment_data)
            segment_index += 1

    return processed_segments


def save_segments_to_csv(segment_data: List[Dict], csv_path: str) -> None:
    """
    Save processed transcript segments to CSV file, appending if file exists.

    Args:
        segment_data: List of segment dictionaries to save
        csv_path: Path to target CSV file

    Examples:
        >>> segments = [
        ...     {
        ...         "VIDEO_NAME": "video123",
        ...         "SENTENCE_NAME": "video123-000",
        ...         "START_REALIGNED": 0.0,
        ...         "END_REALIGNED": 2.0,
        ...         "SENTENCE": "Hello world"
        ...     }
        ... ]
        >>> save_segments_to_csv(segments, "/data/manifest.csv")
    """
    df = pd.DataFrame(segment_data)
    mode = "a" if os.path.exists(csv_path) else "w"
    header = not os.path.exists(csv_path)

    df.to_csv(
        csv_path,
        sep="\t",
        mode=mode,
        header=header,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )
