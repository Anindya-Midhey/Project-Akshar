import cv2
import numpy as np


def enhance(image: np.ndarray, mode: str = "bw") -> np.ndarray:
    """
    Enhance a document image using the enhance_new.py pipeline:
      1. Background normalization (median-blur divide) — removes uneven lighting.
      2. Sharpening (unsharp-mask kernel)              — enhances text edges.
      3. Otsu threshold (mode="bw")                   — clean black & white.
      4. Morphological open                            — removes tiny noise dots.

    Args:
        image: Input BGR image.
        mode:  "color"  → sharpened, background-normalized color output  [default for M1]
               "bw"     → Otsu binary output (3-channel BGR for consistency)

    Returns:
        Processed BGR image.
    """
    # Convert to grayscale for per-channel operations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------- Step 1: Background normalization ----------
    bg = cv2.medianBlur(gray, 31)   # estimate background
    norm = cv2.divide(gray, bg, scale=255)

    # --------- Step 2: Sharpen (VERY IMPORTANT) ----------
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(norm, -1, kernel)

    # --------- Step 3: Strong but stable threshold ----------
    _, thresh = cv2.threshold(
        sharp,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------- Step 4: Clean tiny noise (without blur) ----------
    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    if mode == "bw":
        return clean

    elif mode == "color":
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    else:
        return image




