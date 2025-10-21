import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from glob import glob
from typing import Dict, List
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import gc
import psutil
import time
import conf as c

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_video_filenames(directory: str, pattern="*.mp4") -> List[str]:
	"""
    Retrieves video filenames from specified directory without extensions.
    """
	return [
		os.path.splitext(os.path.basename(f))[0]
		for f in glob(os.path.join(directory, pattern))
	]


def validate_video_file(video_path: str) -> bool:
	"""
    Validates if a video file exists and can be opened by OpenCV.
    Returns True if valid, False otherwise.
    """
	if not os.path.exists(video_path):
		return False

	try:
		cap = cv2.VideoCapture(video_path)
		is_valid = cap.isOpened()
		cap.release()
		return is_valid
	except Exception:
		return False


def _get_video_fps(video_path: str) -> float:
	"""
    Return video FPS (float). Returns 0.0 if FPS cannot be obtained.
    """
	try:
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			return 0.0
		fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
		cap.release()
		return float(fps)
	except Exception:
		return 0.0


class FPSSampler:
	"""
    Two sampling strategies:
      1) reduce mode (priority): Downsample source fps to target fps
         (uses accumulation error method for non-integer ratios, solves 30->24 etc.)
      2) skip mode: Sample every Nth frame.
    """
	def __init__(self, src_fps: float, reduce_to: float | None, frame_skip_by: int):
		self.mode = 'reduce' if (reduce_to is not None and src_fps > 0) else 'skip'
		if self.mode == 'reduce':
			# Only downsample: if target >= src, sample every frame (equivalent to no reduction)
			self.target = min(reduce_to, src_fps)
			# Accumulation error method (Bresenham-like): accumulate r=target/src per frame,
			# when acc>=1, sample and acc-=1
			self.r = self.target / max(src_fps, 1e-6)
			self.acc = 0.0
		else:
			self.n = max(int(frame_skip_by), 1)
			self.count = 0

	def take(self) -> bool:
		if self.mode == 'reduce':
			self.acc += self.r
			if self.acc >= 1.0:
				self.acc -= 1.0
				return True
			return False
		take_now = (self.count % self.n) == 0
		self.count += 1
		return take_now


def process_mediapipe_detection(image, model):
	"""
    Processes an image through MediaPipe detection model.
    """
	return model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def extract_landmark_coordinates(results):
	"""
    Extracts landmark coordinates from MediaPipe detection results.
    """

	def convert_landmarks_to_array(landmarks, indices):
		return (
			np.array(
				[[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices]
			)
			if landmarks
			else np.zeros((len(indices), 3))
		)

	# Extract landmarks for different body parts
	pose_landmarks = convert_landmarks_to_array(
		getattr(results.pose_landmarks, "landmark", None), c.POSE_IDX
	)
	left_hand_landmarks = convert_landmarks_to_array(
		getattr(results.left_hand_landmarks, "landmark", None), c.HAND_IDX
	)
	right_hand_landmarks = convert_landmarks_to_array(
		getattr(results.right_hand_landmarks, "landmark", None), c.HAND_IDX
	)
	face_landmarks = convert_landmarks_to_array(
		getattr(results.face_landmarks, "landmark", None), c.FACE_IDX
	)

	return np.concatenate([
		pose_landmarks.flatten(),
		face_landmarks.flatten(),
		left_hand_landmarks.flatten(),
		right_hand_landmarks.flatten(),
	])


def process_video_segment(video_path: str, start_time: float, end_time: float, output_file: str):
	"""
    Processes a video segment to extract holistic keypoints and save them.
    """
	cap = None
	holistic = None

	try:
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			logger.error(f"Error opening video: {video_path}")
			return

		# FPS & create sampler
		fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
		target_fps = None if getattr(c, "REDUCE_FPS_TO", None) is None else float(c.REDUCE_FPS_TO)
		sampler = FPSSampler(src_fps=fps, reduce_to=target_fps, frame_skip_by=c.FRAME_SKIP)

		# Calculate frame ranges
		start_frame, end_frame = int(start_time * fps), int(end_time * fps)
		cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

		landmark_sequences = []

		# Create MediaPipe model
		holistic = mp.solutions.holistic.Holistic(
			model_complexity=1,
			refine_face_landmarks=True,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
		)

		current_frame = start_frame
		while current_frame <= end_frame:
			ret, frame = cap.read()
			if not ret:
				break

			if sampler.take():
				results = process_mediapipe_detection(frame, holistic)
				landmark_sequences.append(extract_landmark_coordinates(results))

			current_frame += 1

		# Save landmarks if valid data exists
		landmark_array = np.array(landmark_sequences)
		if landmark_array.size > 0 and np.any(landmark_array):
			os.makedirs(os.path.dirname(output_file), exist_ok=True)
			np.save(output_file, landmark_array)
			logger.info(f"Saved landmarks to {output_file}")
		else:
			logger.info(f"No valid landmarks for segment {video_path}, not saving.")

	except Exception as e:
		logger.error(f"Error processing {video_path}: {str(e)}")

	finally:
		# Ensure resource cleanup
		if cap is not None:
			cap.release()
		if holistic is not None:
			holistic.close()

		# Force garbage collection
		gc.collect()

		# Log memory usage
		process = psutil.Process(os.getpid())
		memory_info = process.memory_info()
		logger.debug(f"Memory usage after processing: {memory_info.rss / 1024 / 1024:.2f} MB")


def process_batch(task_batch):
	"""
    Process a batch of tasks for bulk processing
    """
	for video_path, start, end, output_path in task_batch:
		try:
			process_video_segment(video_path, start, end, output_path)
		except Exception as e:
			logger.error(f"Error in batch processing: {str(e)}")

		# Add small delay after each task to avoid resource contention
		time.sleep(0.1)


def main():
	"""
    Main function to orchestrate video processing and landmark extraction.
    """
	# Read CSV and detect column format
	timestamp_data_full = pd.read_csv(c.CSV_FILE, delimiter="\t", on_bad_lines="skip")
	columns = timestamp_data_full.columns.tolist()
	
	# Detect which timestamp columns are available
	if "START" in columns and "END" in columns:
		start_col, end_col = "START", "END"
		logger.info("Using START/END columns for timestamps")
	elif "START_REALIGNED" in columns and "END_REALIGNED" in columns:
		start_col, end_col = "START_REALIGNED", "END_REALIGNED"
		logger.info("Using START_REALIGNED/END_REALIGNED columns for timestamps")
	else:
		raise ValueError("Neither START/END nor START_REALIGNED/END_REALIGNED columns found in CSV")
	
	# Select required columns and drop missing values
	timestamp_data = timestamp_data_full[
		["VIDEO_NAME", "SENTENCE_NAME", start_col, end_col]
	].dropna()

	video_files = get_video_filenames(c.VIDEO_DIR, pattern="*.mp4")
	processed_files = get_video_filenames(c.NPY_DIR, pattern="*.npy")

	logger.info(f"Found {len(video_files)} video files")

	# Create video validation cache
	video_validation_cache = {}
	invalid_videos = set()
	skipped_due_to_invalid_video = 0
	skipped_due_to_existing_file = 0
	skipped_due_to_duration = 0
	skipped_due_to_fps_range = 0
	skipped_due_to_too_short = 0
	video_fps_cache = {}

	processing_tasks = []
	for _, row in timestamp_data.iterrows():
		video_name = row.VIDEO_NAME
		sentence_name = row.SENTENCE_NAME
		start, end = row[start_col], row[end_col]

		video_path = os.path.join(c.VIDEO_DIR, f"{video_name}.mp4")
		output_path = os.path.join(c.NPY_DIR, f"{sentence_name}.npy")

		# Skip if output file already exists
		if sentence_name in processed_files:
			skipped_due_to_existing_file += 1
			continue

		# Segment duration limits: 200ms <= duration <= 60 seconds
		seg_dur = float(end - start)
		if seg_dur < 0.2:
			skipped_due_to_too_short += 1
			continue
		if seg_dur > 60.0:
			skipped_due_to_duration += 1
			continue

		# Validate video file (use cache to avoid repeated checks)
		if video_path not in video_validation_cache:
			video_validation_cache[video_path] = validate_video_file(video_path)
			if not video_validation_cache[video_path]:
				invalid_videos.add(video_name)
				logger.warning(f"Invalid or missing video file: {video_path}")

		if not video_validation_cache[video_path]:
			skipped_due_to_invalid_video += 1
			continue

		# Video FPS filtering
		if video_path not in video_fps_cache:
			video_fps_cache[video_path] = _get_video_fps(video_path)
		vfps = video_fps_cache[video_path]
		min_fps, max_fps = c.ACCEPT_VIDEO_FPS_WITHIN
		if vfps <= 0.0 or vfps < float(min_fps) or vfps > float(max_fps):
			skipped_due_to_fps_range += 1
			continue

		processing_tasks.append((video_path, start, end, output_path))

	# Log summary of skipped tasks
	logger.info(f"Task summary:")
	logger.info(f"  - Tasks to process: {len(processing_tasks)}")
	logger.info(f"  - Skipped (existing files): {skipped_due_to_existing_file}")
	logger.info(f"  - Skipped (duration > 60s): {skipped_due_to_duration}")
	logger.info(f"  - Skipped (duration < 0.2s): {skipped_due_to_too_short}")
	logger.info(f"  - Skipped (fps out of {c.ACCEPT_VIDEO_FPS_WITHIN}): {skipped_due_to_fps_range}")
	logger.info(f"  - Skipped (invalid videos): {skipped_due_to_invalid_video}")
	if invalid_videos:
		logger.warning(f"Invalid video files found: {', '.join(sorted(invalid_videos))}")

	# Process in batches to avoid submitting too many tasks at once
	BATCH_SIZE = 100  # Process 100 tasks per batch
	MAX_WORKERS = min(c.MAX_WORKERS, multiprocessing.cpu_count() - 1)  # Reserve one CPU core

	for i in range(0, len(processing_tasks), BATCH_SIZE):
		batch = processing_tasks[i:i + BATCH_SIZE]
		logger.info(
			f"Processing batch {i // BATCH_SIZE + 1}, tasks {i + 1} to {min(i + BATCH_SIZE, len(processing_tasks))}")

		# Further subdivide batches for different processes
		tasks_per_worker = len(batch) // MAX_WORKERS + 1
		worker_batches = [batch[j:j + tasks_per_worker] for j in range(0, len(batch), tasks_per_worker)]

		with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
			futures = []
			for worker_batch in worker_batches:
				if worker_batch:  # Ensure batch is not empty
					future = executor.submit(process_batch, worker_batch)
					futures.append(future)

			# Wait for all tasks to complete
			for future in futures:
				try:
					future.result()
				except Exception as e:
					logger.error(f"Error in worker process: {str(e)}")

		# Add delay between batches to allow system time to release resources
		time.sleep(0.5)

		# Force garbage collection
		gc.collect()

		# Log progress and memory usage
		process = psutil.Process(os.getpid())
		memory_info = process.memory_info()
		logger.info(f"Batch {i // BATCH_SIZE + 1} completed. Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
	main()