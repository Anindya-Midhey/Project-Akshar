"""
Module 1 — Full Processing Pipeline
====================================
Orchestrates all Module 1 steps sequentially:

  1. Load + resize image
  2. Page-split check (double-page spread detection)
  3. For each page:
     a. SAM page extraction   — isolates document from background
     b. Remove black border   — strips thin dark margins
     c. Shadow removal        — normalises uneven illumination
     d. Enhancement (color)   — sharpens and normalises for readability
  4. Save processed images and return results

Returns a list of result dicts (1 item for single page, 2 for a spread).
"""

import os
import uuid
import base64

import cv2
import numpy as np
from typing import List, Dict, Any

from m1.preprocess    import load_image, resize
from m1.page_split    import maybe_split
from m1.sam_detect    import sam_page_extract
from m1.remove_border import remove_black_border
from m1.shadow        import remove_shadow
from m1.enhance       import enhance

# ---------------------------------------------------------------------------
# Output directory for processed images
# ---------------------------------------------------------------------------
_M1_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(_M1_ROOT, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# SAM model location inside module1/
SAM_CHECKPOINT = os.path.join(_M1_ROOT, "models", "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"


def _encode_image(img: np.ndarray) -> str:
    """Encode a BGR numpy image as a base64 PNG data-URI."""
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def process_image(image_path: str, mode: str = "color") -> List[Dict[str, Any]]:
    """
    Run the full Module 1 pipeline on a single image file.

    Parameters
    ----------
    image_path : str   Absolute path to the uploaded image.
    mode       : str   Enhancement mode — "color" (default) or "bw".

    Returns
    -------
    List of result dicts, one per output page:
        {
            "image_path": "/abs/path/to/processed/m1_<uuid>.png",
            "preview":    "data:image/png;base64,...",
            "width":      <int>,
            "height":     <int>,
        }
    Typically 1 dict; 2 dicts when a double-page spread is detected.
    """
    # ── Step 1: Load + resize ──────────────────────────────────────────────
    image = load_image(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = resize(image)   # target width = 800 px

    # ── Step 2: Page-split check ───────────────────────────────────────────
    # maybe_split returns [single_page] or [left_page, right_page]
    pages = maybe_split(image)

    results: List[Dict[str, Any]] = []

    for page_img in pages:
        # ── Step 3a: SAM page extraction ──────────────────────────────────
        warped = sam_page_extract(
            page_img,
            checkpoint_path=SAM_CHECKPOINT,
            model_type=SAM_MODEL_TYPE,
        )

        # ── Step 3b: Remove residual black border ─────────────────────────
        warped = remove_black_border(warped)

        # ── Step 3c: Shadow removal ───────────────────────────────────────
        warped = remove_shadow(warped)

        # ── Step 3d: Enhancement ─────────────────────────────────────────
        final = enhance(warped, mode=mode)

        # ── Save output ───────────────────────────────────────────────────
        out_id   = str(uuid.uuid4())
        out_path = os.path.join(PROCESSED_DIR, f"m1_{out_id}.png")
        cv2.imwrite(out_path, final)

        h, w = final.shape[:2]
        results.append({
            "image_path": out_path,
            "preview":    _encode_image(final),
            "width":      w,
            "height":     h,
        })

    return results
