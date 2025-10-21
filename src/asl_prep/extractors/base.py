"""Base classes for landmark extractors."""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class LandmarkExtractor(ABC):
    """
    Abstract base class for landmark extraction.

    Subclasses should implement:
    - process_frame: Extract landmarks from a single video frame
    - close: Release resources (models, GPU memory, etc.)
    """

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame and extract landmarks.

        Args:
            frame: Input video frame (BGR format from OpenCV)

        Returns:
            Flattened numpy array of landmarks, or None if extraction fails

        Examples:
            >>> extractor = SomeExtractor()
            >>> frame = cv2.imread("image.jpg")
            >>> landmarks = extractor.process_frame(frame)
            >>> landmarks.shape
            (255,)  # Or whatever dimension the extractor outputs
        """
        pass

    @abstractmethod
    def close(self):
        """
        Release resources (models, GPU memory, etc.).

        Should be called when the extractor is no longer needed.

        Examples:
            >>> extractor = SomeExtractor()
            >>> # ... use extractor ...
            >>> extractor.close()
        """
        pass
