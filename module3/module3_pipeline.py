"""
module3_pipeline.py  (module3)
──────────────────────────────
High-level pipeline entry-point for Module 3.

Supports two modes:
  • PDF mode:   process_pdf_pipeline(pdf_path) — full layout detection + OCR
  • Image mode: process_image_pipeline(image_path) — single-image OCR

Both return a list of block dicts compatible with the rest of the Akshar system.
No Sarvam API key required — all inference is fully local.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# PDF pipeline  (primary path — uses layout detection + OCR)
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf_pipeline(
    pdf_path: str,
    out_json: Optional[str] = None,
    dpi: int = 300,
    layout_filter: Optional[List[str]] = None,
    use_gpu: bool = False,
    visualize: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run full layout-detection + OCR on a PDF and return block dicts.

    Args:
        pdf_path:      Path to the input PDF.
        out_json:      Where to write result JSON.  Defaults to
                       <pdf_basename>_metadata.json next to the PDF.
        dpi:           PDF render DPI (300 recommended).
        layout_filter: e.g. ["title","section_header","paragraph"]
                       None → all text-bearing blocks.
        use_gpu:       Use GPU for OCR inference.
        visualize:     Write annotated debug images.

    Returns:
        List[Dict] — one dict per extracted block.
    """
    if out_json is None:
        base = os.path.splitext(pdf_path)[0]
        out_json = f"{base}_metadata.json"

    from module3.sarvam_client import extract_grounding_metadata  # type: ignore

    return extract_grounding_metadata(
        pdf_path=pdf_path,
        layout_filter=layout_filter,
        dpi=dpi,
        use_gpu=use_gpu,
        visualize=visualize,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image pipeline  (single-image OCR via akshar_api.call_akshar)
# ─────────────────────────────────────────────────────────────────────────────

def process_image_pipeline(image_path: str) -> List[Dict[str, Any]]:
    """
    Run word-level OCR on a single image using the local PaddleOCR engine.

    Args:
        image_path: Path to a JPG/PNG image file.

    Returns:
        List of word dicts:  { "text": str, "bbox": [xmin,ymin,xmax,ymax] }
        Bounding boxes are normalised to [0, 1] relative to image dimensions.
    """
    import cv2
    from module3.akshar_api import call_akshar       # type: ignore
    from module3.bbox_normalizer import normalize_boxes  # type: ignore

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]

    # Step 1: local OCR
    raw_data = call_akshar(image_path)

    # Step 2: normalise bounding boxes to [0, 1]
    normalised = normalize_boxes(raw_data, w, h)

    return normalised


# ─────────────────────────────────────────────────────────────────────────────
# Convenience alias kept for backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

def module3_pipeline(image_path: str) -> List[Dict[str, Any]]:
    """Backward-compatible alias → process_image_pipeline(image_path)."""
    return process_image_pipeline(image_path)