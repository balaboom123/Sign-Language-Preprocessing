"""MMPose-based 3D pose landmark extraction."""
from typing import Optional, List

import numpy as np

from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector

from .base import LandmarkExtractor


class MultiPersonDetected(Exception):
    """Raised when more than one person is detected in a frame."""
    pass


class MMPoseExtractor(LandmarkExtractor):
    """
    Extracts 3D pose landmarks using MMPose RTMPose3D.

    This extractor uses a two-stage pipeline:
    1. RTMDet for person detection (bounding boxes)
    2. RTMPose3D for 3D pose estimation

    Args:
        detector: Initialized MMDet detector model
        pose_estimator: Initialized MMPose 3D pose estimator model
        keypoint_indices: List of COCO-WholeBody keypoint indices to extract
        bbox_threshold: Confidence threshold for person detection
        det_cat_id: Detection category ID for person (default: 0)
        add_visible: Whether to include visibility scores in output
        apply_reduction: Whether to apply keypoint reduction (True) or keep all 133 keypoints (False)

    Examples:
        >>> from mmdet.apis import init_detector
        >>> from mmpose.apis import init_model
        >>> detector = init_detector(det_config, det_checkpoint, device='cuda:0')
        >>> pose_model = init_model(pose_config, pose_checkpoint, device='cuda:0')
        >>> extractor = MMPoseExtractor(
        ...     detector, pose_model,
        ...     keypoint_indices=list(range(85)),
        ...     bbox_threshold=0.5
        ... )
        >>> landmarks = extractor.process_frame(frame)
    """

    def __init__(
        self,
        detector,
        pose_estimator,
        keypoint_indices: List[int],
        bbox_threshold: float = 0.5,
        det_cat_id: int = 0,
        add_visible: bool = True,
        apply_reduction: bool = True,
    ):
        """Initialize MMPose extractor with pre-loaded models."""
        self.detector = detector
        self.pose_estimator = pose_estimator
        # If apply_reduction=False, use all 133 keypoints; otherwise use specified indices
        self.keypoint_indices = keypoint_indices if apply_reduction else list(range(133))
        self.bbox_threshold = bbox_threshold
        self.det_cat_id = det_cat_id
        self.add_visible = add_visible
        self.apply_reduction = apply_reduction

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect person and extract 3D landmarks from a single frame.

        Pipeline:
        1. Detect person bounding box using RTMDet
        2. Extract 3D pose using RTMPose3D
        3. Pack keypoints: [x_norm, y_norm, z_raw, visible]

        Args:
            frame: Input video frame (BGR format)

        Returns:
            Array of landmarks with shape (num_keypoints, 4):
            - Each keypoint: [x, y, z, visibility]
            - x, y normalized to [0, 1]
            - z uses raw values from pose estimator
            When apply_reduction=True:
                Shape: (85, 4) - ASL-optimized keypoint subset
            When apply_reduction=False:
                Shape: (133, 4) - All COCO-WholeBody keypoints
            Returns None if no person detected or extraction fails
            Raises MultiPersonDetected if multiple persons detected

        Examples:
            >>> frame = cv2.imread("image.jpg")
            >>> landmarks = extractor.process_frame(frame)
            >>> landmarks.shape
            (85, 4)  # With reduction: 85 keypoints × [x, y, z, visibility]
            >>> # or (133, 4) without reduction
        """
        # Person detection
        det_result = inference_detector(self.detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()

        # Filter person instances by category and bbox threshold
        bboxes = pred_instance.bboxes
        bboxes = bboxes[np.logical_and(
            pred_instance.labels == self.det_cat_id,
            pred_instance.scores > self.bbox_threshold
        )]

        # No person detected
        if len(bboxes) == 0:
            return None

        # More than one person detected: mark this segment as invalid
        if len(bboxes) > 1:
            raise MultiPersonDetected(
                f"Detected {len(bboxes)} persons with score > {self.bbox_threshold}"
            )

        # 3D Pose estimation
        pose_est_results = inference_topdown(self.pose_estimator, frame, bboxes)

        # Post-processing: squeeze dimensions and sort by track_id
        for idx, pose_est_result in enumerate(pose_est_results):
            pose_est_result.track_id = pose_est_results[idx].get('track_id', 1e4)

            pred_instances = pose_est_result.pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores

            # Squeeze extra dimensions
            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pose_est_results[idx].pred_instances.keypoint_scores = keypoint_scores

            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)

            pose_est_results[idx].pred_instances.keypoints = keypoints

        # Sort by track_id and merge
        pose_est_results = sorted(
            pose_est_results, key=lambda x: x.get('track_id', 1e4))

        pred_3d_data_samples = merge_data_samples(pose_est_results)
        pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

        if pred_3d_instances is None:
            return None

        # Extract and pack keypoints
        H, W = frame.shape[:2]
        packed = self._pack_keypoints(pred_3d_instances, W, H)

        return packed

    def _pack_keypoints(
        self,
        pred_3d_instances,
        img_w: int,
        img_h: int,
        instance_index: int = 0
    ) -> Optional[np.ndarray]:
        """
        Extract and pack keypoints from 3D pose estimation results.

        Returns array with shape (num_keypoints, 4) where:
        - First person only (instance_index=0)
        - Filtered by self.keypoint_indices
        - x,y normalized to [0,1], z uses raw values from pose estimator
        - visibility scores included

        Args:
            pred_3d_instances: Predicted 3D pose instances from MMPose
            img_w: Original image width for normalization
            img_h: Original image height for normalization
            instance_index: Which person instance to extract (default: 0)

        Returns:
            Numpy array of shape (num_keypoints, 4) or None if extraction fails
            Each row: [x_norm, y_norm, z_raw, visibility]
        """
        if pred_3d_instances is None:
            return None

        # Get arrays
        tk = getattr(pred_3d_instances, 'transformed_keypoints', None)
        k3d = getattr(pred_3d_instances, 'keypoints', None)
        if tk is None or k3d is None:
            return None

        tk = self._to_numpy(tk)
        k3d = self._to_numpy(k3d)
        tk = self._squeeze_kpts(tk)   # (N, K, 2)
        k3d = self._squeeze_kpts(k3d)  # (N, K, 3)

        # Guard: need at least one instance
        if tk.ndim != 3 or k3d.ndim != 3 or tk.shape[0] == 0 or k3d.shape[0] == 0:
            return None

        # Select instance (default: first person)
        xy = tk[instance_index]      # (K, 2) in original image coords
        xyz = k3d[instance_index]    # (K, 3) in model-input coords

        # Filter by keypoint indices
        xy = xy[self.keypoint_indices]    # (NUM_KEYPOINTS, 2)
        xyz = xyz[self.keypoint_indices]  # (NUM_KEYPOINTS, 3)

        # Normalize x,y by original image size
        x_norm = xy[..., 0] / float(img_w)
        y_norm = xy[..., 1] / float(img_h)

        # Use raw z from the pose estimator; do not apply custom offsets
        z = xyz[..., 2]

        # Get visibility scores (always include, ignore add_visible parameter)
        kpt_scores = getattr(pred_3d_instances, 'keypoint_scores', None)
        if kpt_scores is not None:
            kpt_scores = self._to_numpy(kpt_scores)
            # Handle different dimensions
            if kpt_scores.ndim == 2:  # (N, K)
                visible = kpt_scores[instance_index]
            elif kpt_scores.ndim == 3:  # (N, K, 1)
                visible = kpt_scores[instance_index, :, 0]
            else:
                visible = np.ones(len(self.keypoint_indices), dtype=np.float32)

            # Filter by keypoint indices
            visible = visible[self.keypoint_indices]
        else:
            # Default to all visible if scores not available
            visible = np.ones(len(self.keypoint_indices), dtype=np.float32)

        # Stack -> (NUM_KEYPOINTS, 4), do NOT flatten
        out = np.stack([x_norm, y_norm, z, visible], axis=-1).astype(np.float32)

        return out

    @staticmethod
    def _to_numpy(x):
        """Convert torch.Tensor or other array-like objects to numpy arrays."""
        if hasattr(x, 'detach'):
            x = x.detach().cpu().numpy()
        elif hasattr(x, 'cpu'):
            x = x.cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _squeeze_kpts(arr, expect_last: int = 2):
        """
        Ensure shapes:
        - 2D transformed keypoints -> (N, K, 2)
        - 3D keypoints             -> (N, K, 3)
        Removes common singleton dims like (N,1,K,2) -> (N,K,2)
        """
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr

    def close(self):
        """
        Release resources.

        Note: For MMPose, models are typically managed externally
        in worker processes, so this is a no-op.
        """
        pass
