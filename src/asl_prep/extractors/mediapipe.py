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
        apply_reduction: Whether to apply keypoint reduction (True) or keep all landmarks (False)

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
        apply_reduction: bool = True,
    ):
        """Initialize MediaPipe holistic model with configuration."""
        self.pose_idx = pose_idx
        self.face_idx = face_idx
        self.hand_idx = hand_idx
        self.apply_reduction = apply_reduction

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
            Array of landmarks with shape (num_keypoints, 4):
            - Concatenated as [pose, face, left_hand, right_hand]
            - Each keypoint: [x, y, z, visibility]
            When apply_reduction=True:
                Shape: (85, 4) = 6 pose + 41 face + 21 left_hand + 21 right_hand
            When apply_reduction=False:
                Shape: (543, 4) = 33 pose + 468 face + 21 left_hand + 21 right_hand
            Returns None if MediaPipe fails to process the frame

        Examples:
            >>> frame = cv2.imread("image.jpg")
            >>> landmarks = extractor.process_frame(frame)
            >>> landmarks.shape
            (85, 4)  # With reduction: 85 keypoints × [x, y, z, visibility]
            >>> # or (543, 4) without reduction
        """
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.holistic.process(image_rgb)

        # Extract landmarks with visibility
        if self.apply_reduction:
            # Apply keypoint reduction using specified indices
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
        else:
            # Keep all landmarks without reduction
            pose_landmarks = self._convert_all_landmarks_to_array(
                getattr(results.pose_landmarks, "landmark", None)
            )
            face_landmarks = self._convert_all_landmarks_to_array(
                getattr(results.face_landmarks, "landmark", None)
            )
            left_hand_landmarks = self._convert_all_landmarks_to_array(
                getattr(results.left_hand_landmarks, "landmark", None)
            )
            right_hand_landmarks = self._convert_all_landmarks_to_array(
                getattr(results.right_hand_landmarks, "landmark", None)
            )

        # Concatenate all landmarks: shape (num_keypoints, 4)
        landmark_array = np.concatenate([
            pose_landmarks,
            face_landmarks,
            left_hand_landmarks,
            right_hand_landmarks,
        ], axis=0)

        return landmark_array.astype(np.float32)

    def _convert_landmarks_to_array(
        self,
        landmarks: Optional[List],
        indices: List[int]
    ) -> np.ndarray:
        """
        Convert MediaPipe landmarks to numpy array with visibility.

        Args:
            landmarks: MediaPipe landmark list (or None if not detected)
            indices: Indices of landmarks to extract

        Returns:
            Numpy array of shape (len(indices), 4) with [x, y, z, visibility]
            Returns zeros if landmarks is None

        Note:
            - Pose landmarks have visibility attribute
            - Hand/face landmarks may not have visibility, defaulting to 1.0

        Examples:
            >>> landmarks = results.pose_landmarks.landmark
            >>> pose_array = self._convert_landmarks_to_array(landmarks, [11, 12])
            >>> pose_array.shape
            (2, 4)  # [[x, y, z, visibility], ...]
        """
        if landmarks:
            out = []
            for i in indices:
                lm = landmarks[i]
                # MediaPipe has lm.visibility for pose; hands/face may not have it
                vis = getattr(lm, "visibility", 1.0)  # Default visible if no attribute
                out.append([lm.x, lm.y, lm.z, vis])
            return np.array(out, dtype=np.float32)
        else:
            # No detection: fill with zeros (will be handled in stage 4)
            return np.zeros((len(indices), 4), dtype=np.float32)

    def _convert_all_landmarks_to_array(
        self,
        landmarks: Optional[List]
    ) -> np.ndarray:
        """
        Convert all MediaPipe landmarks to numpy array without filtering.

        Args:
            landmarks: MediaPipe landmark list (or None if not detected)

        Returns:
            Numpy array of shape (num_landmarks, 4) with [x, y, z, visibility]
            Returns zeros if landmarks is None
            - Pose: (33, 4)
            - Face: (468, 4)
            - Hands: (21, 4) each

        Note:
            Used when apply_reduction=False to keep all original landmarks.

        Examples:
            >>> landmarks = results.face_landmarks.landmark
            >>> face_array = self._convert_all_landmarks_to_array(landmarks)
            >>> face_array.shape
            (468, 4)  # All face landmarks with visibility
        """
        if landmarks:
            out = []
            for lm in landmarks:
                # MediaPipe has lm.visibility for pose; hands/face may not have it
                vis = getattr(lm, "visibility", 1.0)  # Default visible if no attribute
                out.append([lm.x, lm.y, lm.z, vis])
            return np.array(out, dtype=np.float32)
        else:
            # No detection: return empty array (will be concatenated with others)
            # Note: This will result in (0, 4) shape which is fine for concatenation
            return np.zeros((0, 4), dtype=np.float32)

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
