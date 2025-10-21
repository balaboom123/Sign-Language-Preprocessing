import os
import csv
import json
import re
import numpy as np
import pandas as pd
import ftfy
import conf as c  # Keeping original conf import name


def normalize_text(text):
    """
    Normalizes text by primarily fixing Unicode issues with ftfy and then
    performing minimal, non-semantic cleaning to align with YouTube-ASL paper's spirit.

    This function aims to:
    1. Correct mojibake and other Unicode encoding errors (ftfy).
    2. Standardize whitespace (remove extra spaces, newlines).
    3. Remove bracketed descriptive content (e.g., [Laughter]).
    4. Remove non-ASCII characters that are not part of standard English text,
       assuming the target is English captions for ASL.

    It explicitly AVOIDS:
    - Lowercasing (as per the paper's statement).
    - Removing punctuation (beyond bracketed content).
    - Any other semantic normalization (stemming, lemmatization, stop words).

    Args:
        text (str): Input text to be normalized.

    Returns:
        str: Cleaned and corrected text.
    """
    text = ftfy.fix_text(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_transcript_file(json_file):
    """
    Reads and parses a JSON transcript file.

    Args:
        json_file (str): Path to JSON transcript file

    Returns:
        list: Transcript data as a list of dictionaries
    """
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)


def process_transcript_segments(transcripts, video_id):
    """
    Processes individual transcript captions, filtering based on length and duration constraints.

    Args:
        transcripts (list): List of transcript dictionaries
        video_id (str): Video identifier for naming segments

    Returns:
        list: List of processed transcript dictionaries meeting the criteria
    """
    processed_segments = []
    segment_index = 0

    # Filter valid transcript entries
    valid_entries = [t for t in transcripts if "text" in t and "start" in t and "duration" in t]
    if not valid_entries:
        print(f"No valid transcripts for video {video_id}")
        return processed_segments

    for entry in valid_entries:
        # Get the normalized text
        processed_text = normalize_text(entry["text"])

        # Apply filtering criteria:
        # - Text length <= 300 characters
        # - Duration between 0.2s and 60s
        if (len(processed_text) <= 300 and
                0.2 <= entry["duration"] <= 60.0 and
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


def save_segments_to_csv(segment_data, csv_path):
    """
    Saves processed transcript segments to CSV file, appending if file exists.

    Args:
        segment_data (list): List of segment dictionaries to save
        csv_path (str): Path to target CSV file
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


def main():
    """
    Main function to process video transcripts into segmented CSV data.
    Reads video IDs, processes their transcripts, and saves the results.
    """
    with open(c.ID, "r", encoding="utf-8") as file:
        video_ids = [line.strip() for line in file if line.strip()]

    print(f"Processing {len(video_ids)} videos.")

    for video_id in video_ids:
        try:
            json_file = os.path.join(c.TRANSCRIPT_DIR, f"{video_id}.json")
            if not os.path.exists(json_file):
                continue

            transcript_data = read_transcript_file(json_file)
            if not transcript_data:
                continue

            processed_segments = process_transcript_segments(transcript_data, video_id)
            if processed_segments:
                save_segments_to_csv(processed_segments, c.CSV_FILE)

        except Exception as e:
            print(f"Error processing {video_id}: {e}")


if __name__ == "__main__":
    main()
