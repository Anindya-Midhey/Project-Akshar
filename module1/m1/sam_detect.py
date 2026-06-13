"""
SAM (Segment Anything Model) based page detection.

Copied exactly from Module_1_Latest/Module 1_Sam/utils/sam_detect.py.

Uses Meta's Segment Anything to automatically segment the input image,
then picks the largest mask (assumed to be the book/document page) and
returns a clean page extraction with the background filled white.

Workflow
--------
1. Load SAM model (vit_b checkpoint).
2. Run SamAutomaticMaskGenerator → list of segment dicts.
3. Sort segments by 'area' (descending).
4. Take the largest mask as the page mask.
5. Fill non-mask pixels with white.
6. Crop to the bounding box of the mask.
"""

import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ---------------------------------------------------------------------------
# Module-level cache so the model is loaded only once
# ---------------------------------------------------------------------------
_sam_model      = None
_mask_generator = None


def _get_mask_generator(
    checkpoint_path: str = None,
    model_type: str      = "vit_b",
):
    """Load (and cache) the SAM model + mask generator."""
    global _sam_model, _mask_generator

    if _mask_generator is not None:
        return _mask_generator

    # Default checkpoint: <module1>/backend/models/sam_vit_b_01ec64.pth
    if checkpoint_path is None:
        project_root    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(project_root, "models", "sam_vit_b_01ec64.pth")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"SAM checkpoint not found at:\n  {checkpoint_path}\n"
            f"Download it with:\n"
            f"  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
            f"and place it in module1/backend/models/."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [*] Loading SAM model ({model_type}) on {device} ...")

    _sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    _sam_model.to(device=device)

    _mask_generator = SamAutomaticMaskGenerator(_sam_model)
    print(f"  [OK] SAM model loaded successfully.")
    return _mask_generator


def sam_page_extract(
    image: np.ndarray,
    checkpoint_path: str  = None,
    model_type: str       = "vit_b",
    min_area_ratio: float = 0.05,
    pad: int              = 10,
) -> np.ndarray:
    """
    Detect the document page using SAM and return it with background
    filled white and cropped to the mask bounding box.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (as loaded by cv2.imread).
    checkpoint_path : str, optional
        Path to the SAM checkpoint (.pth). Defaults to models/sam_vit_b_01ec64.pth
    model_type : str
        SAM model type ('vit_h', 'vit_l', or 'vit_b').
    min_area_ratio : float
        Minimum fraction of image area for a valid page mask (default 5%).
    pad : int
        Padding (pixels) to add around the detected mask bounding box.

    Returns
    -------
    np.ndarray
        Extracted page image (BGR) with background filled white.
    """
    mask_gen = _get_mask_generator(checkpoint_path, model_type)

    # SAM expects RGB input
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"  [*] Running SAM segmentation ...")
    masks = mask_gen.generate(image_rgb)
    print(f"  [*] SAM detected {len(masks)} segments.")

    if not masks:
        print("  [WARN] SAM produced no masks - returning original image.")
        return image

    # ── Sort masks by area (largest first) ─────────────────────────────
    masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)

    img_h, img_w = image.shape[:2]
    img_area     = img_h * img_w

    # ── Find the best page mask ────────────────────────────────────────
    # Strategy: The largest mask is usually the page.  But if it covers
    # nearly the entire image (>95%) it might be "background" instead,
    # so in that case we try the second-largest.
    page_mask = None
    for m in masks_sorted:
        ratio = m["area"] / img_area
        if ratio < min_area_ratio:
            break  # too small to be a page
        if ratio > 0.95:
            # Likely the full-image background; skip it
            continue
        page_mask = m["segmentation"]  # bool array h×w
        print(f"  [*] Selected mask with area ratio {ratio:.2%}")
        break

    # If no suitable mask found, fall back to the absolute largest
    if page_mask is None:
        page_mask = masks_sorted[0]["segmentation"]
        ratio     = masks_sorted[0]["area"] / img_area
        print(f"  [*] Fallback: using largest mask (area ratio {ratio:.2%})")

    # ── Fill background with white ─────────────────────────────────────
    result = image.copy()
    result[~page_mask] = [255, 255, 255]

    # ── Crop to bounding box of the mask ───────────────────────────────
    ys, xs = np.where(page_mask)
    if len(ys) == 0:
        return result

    x1 = max(0,     xs.min() - pad)
    y1 = max(0,     ys.min() - pad)
    x2 = min(img_w, xs.max() + pad)
    y2 = min(img_h, ys.max() + pad)

    cropped = result[y1:y2, x1:x2]
    return cropped
