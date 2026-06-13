import cv2
import numpy as np


def remove_shadow(image: np.ndarray) -> np.ndarray:
    """
    Remove shadow / uneven illumination from a scanned document.

    Works per-channel:
      1. Dilate each channel to estimate the background illumination.
      2. Median-blur the dilated image to smooth the background estimate.
      3. Subtract the foreground from the background (inverted) to normalise.
      4. Normalise each channel to the full [0, 255] range.
    """
    rgb_planes = cv2.split(image)
    result_norm = []

    for plane in rgb_planes:
        # Estimate background illumination
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg      = cv2.medianBlur(dilated, 21)
        diff    = 255 - cv2.absdiff(plane, bg)
        norm    = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        result_norm.append(norm)

    return cv2.merge(result_norm)
