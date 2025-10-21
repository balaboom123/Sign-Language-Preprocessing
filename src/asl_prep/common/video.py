"""Video processing utilities for ASL preprocessing pipeline."""
import os
import cv2
from typing import Optional


def validate_video_file(video_path: str) -> bool:
    """
    Validates if a video file exists and can be opened by OpenCV.

    Args:
        video_path: Path to the video file

    Returns:
        True if video file is valid and can be opened, False otherwise

    Examples:
        >>> validate_video_file("/videos/sample.mp4")
        True
        >>> validate_video_file("/videos/missing.mp4")
        False
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


def get_video_fps(video_path: str) -> float:
    """
    Return video FPS (frames per second) as float.

    Args:
        video_path: Path to the video file

    Returns:
        FPS as float, or 0.0 if FPS cannot be obtained

    Examples:
        >>> get_video_fps("/videos/sample.mp4")
        30.0
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
    Frame sampling strategies for video processing.

    Supports two sampling modes:
      1) reduce mode (priority): Downsample source fps to target fps
         Uses accumulation error method (Bresenham-like) for non-integer ratios
         Handles cases like 30fps -> 24fps smoothly
      2) skip mode: Sample every Nth frame

    Args:
        src_fps: Source video FPS
        reduce_to: Target FPS for downsampling (None to disable reduce mode)
        frame_skip_by: Skip every Nth frame (used when reduce_to is None)

    Examples:
        >>> # Downsample 30fps video to 15fps
        >>> sampler = FPSSampler(src_fps=30.0, reduce_to=15.0, frame_skip_by=2)
        >>> sampler.mode
        'reduce'

        >>> # Skip every 2nd frame
        >>> sampler = FPSSampler(src_fps=30.0, reduce_to=None, frame_skip_by=2)
        >>> sampler.mode
        'skip'
    """

    def __init__(self, src_fps: float, reduce_to: Optional[float], frame_skip_by: int):
        """Initialize FPS sampler with source FPS and target configuration."""
        self.mode = 'reduce' if (reduce_to is not None and src_fps > 0) else 'skip'

        if self.mode == 'reduce':
            # Only downsample: if target >= src, sample every frame (no reduction)
            self.target = min(reduce_to, src_fps)
            # Accumulation error method (Bresenham-like):
            # Accumulate r=target/src per frame, when acc>=1, sample and acc-=1
            self.r = self.target / max(src_fps, 1e-6)
            self.acc = 0.0
        else:
            # Skip mode: sample every Nth frame
            self.n = max(int(frame_skip_by), 1)
            self.count = 0

    def take(self) -> bool:
        """
        Returns True if current frame should be sampled.

        In reduce mode:
            Uses accumulation to determine sampling points for smooth downsampling

        In skip mode:
            Returns True every Nth frame

        Returns:
            True if frame should be processed, False otherwise

        Examples:
            >>> sampler = FPSSampler(30.0, 15.0, 2)
            >>> [sampler.take() for _ in range(4)]
            [True, False, True, False]
        """
        if self.mode == 'reduce':
            self.acc += self.r
            if self.acc >= 1.0:
                self.acc -= 1.0
                return True
            return False
        else:
            take_now = (self.count % self.n) == 0
            self.count += 1
            return take_now
