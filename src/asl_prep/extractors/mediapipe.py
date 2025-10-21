"""MediaPipe-based holistic landmark extraction."""
from typing import Optional, List

import cv2
import mediapipe as mp
import numpy as np

from .base import LandmarkExtractor


class MediaPipeExtractor(LandmarkExtractor):
    """
    Extracts holistic landmarks using MediaPipe.

    Extracts landmarks for:
    - Pose: Upper body keypoints (shoulders, elbows, hips)
    - Face: Facial landmarks for expressions
    - Hands: Left and right hand landmarks

    Args:
        pose_idx: List of pose landmark indices to extract
        face_idx: List of face landmark indices to extract
        hand_idx: List of hand landmark indices to extract
        model_complexity: MediaPipe model complexity (0, 1, or 2)
        refine_face_landmarks: Whether to refine face landmarks
        min_detection_confidence: Minimum confidence for detection
        min_tracking_confidence: Minimum confidence for tracking

    Examples:
        >>> extractor = MediaPipeExtractor(
        ...     pose_idx=[11, 12, 13, 14, 23, 24],
        ...     face_idx=list(range(37)),
        ...     hand_idx=list(range(21))
        ... )
        >>> frame = cv2.imread("image.jpg")
        >>> landmarks = extractor.process_frame(frame)
        >>> extractor.close()
    """

    def __init__(
        self,
        pose_idx: List[int],
        face_idx: List[int],
        hand_idx: List[int],
        model_complexity: int = 1,
        refine_face_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize MediaPipe holistic model with configuration."""
        self.pose_idx = pose_idx
        self.face_idx = face_idx
        self.hand_idx = hand_idx

        self.holistic = mp.solutions.holistic.Holistic(
            model_complexity=model_complexity,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract holistic landmarks from a single frame.

        Args:
            frame: Input video frame (BGR format)

        Returns:
            Flattened array of landmarks: [pose, face, left_hand, right_hand]
            Shape: (num_pose*3 + num_face*3 + num_hand*3 + num_hand*3,)
            Returns None if MediaPipe fails to process the frame

        Examples:
            >>> frame = cv2.imread("image.jpg")
            >>> landmarks = extractor.process_frame(frame)
            >>> landmarks.shape
            (255,)  # 6*3 + 37*3 + 21*3 + 21*3 = 255
        """
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.holistic.process(image_rgb)

        # Extract landmarks
        pose_landmarks = self._convert_landmarks_to_array(
            getattr(results.pose_landmarks, "landmark", None), self.pose_idx
        )
        face_landmarks = self._convert_landmarks_to_array(
            getattr(results.face_landmarks, "landmark", None), self.face_idx
        )
        left_hand_landmarks = self._convert_landmarks_to_array(
            getattr(results.left_hand_landmarks, "landmark", None), self.hand_idx
        )
        right_hand_landmarks = self._convert_landmarks_to_array(
            getattr(results.right_hand_landmarks, "landmark", None), self.hand_idx
        )

        # Concatenate all landmarks into a single flattened array
        landmark_array = np.concatenate([
            pose_landmarks.flatten(),
            face_landmarks.flatten(),
            left_hand_landmarks.flatten(),
            right_hand_landmarks.flatten(),
        ])

        return landmark_array

    def _convert_landmarks_to_array(
        self,
        landmarks: Optional[List],
        indices: List[int]
    ) -> np.ndarray:
        """
        Convert MediaPipe landmarks to numpy array.

        Args:
            landmarks: MediaPipe landmark list (or None if not detected)
            indices: Indices of landmarks to extract

        Returns:
            Numpy array of shape (len(indices), 3) with [x, y, z] coordinates
            Returns zeros if landmarks is None

        Examples:
            >>> landmarks = results.pose_landmarks.landmark
            >>> pose_array = self._convert_landmarks_to_array(landmarks, [11, 12])
            >>> pose_array.shape
            (2, 3)
        """
        if landmarks:
            return np.array(
                [[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices]
            )
        else:
            return np.zeros((len(indices), 3))

    def close(self):
        """
        Release MediaPipe holistic model resources.

        Examples:
            >>> extractor = MediaPipeExtractor(pose_idx, face_idx, hand_idx)
            >>> # ... use extractor ...
            >>> extractor.close()
        """
        if self.holistic is not None:
            self.holistic.close()
