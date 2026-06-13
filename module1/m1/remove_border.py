import cv2
import numpy as np


def remove_black_border(image: np.ndarray) -> np.ndarray:
    """
    Remove residual thin black border left after SAM page extraction.
    Detects the white page region and crops to its bounding box.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold: keep only bright (page) pixels
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image  # no white region found — return unchanged

    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]
