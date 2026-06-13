"""
akshar_api.py  (module3)
────────────────────────
Previously called the Sarvam Akshar cloud API.
Now replaced by the fully-local PaddleOCR pipeline — no API key required.

`call_akshar(image_path)` is kept as the public function so callers do not
need to change their code.  Internally it:
  1. Runs PaddleOCR OCR directly on the image file.
  2. Returns data in a structure compatible with the original Akshar API
     response ({"words": [{"text": ..., "bbox": {...}}, ...]}).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Any

import cv2
import numpy as np

logging.getLogger("ppocr").setLevel(logging.ERROR)

# Singleton OCR engine
_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError:
            raise ImportError(
                "paddleocr >= 3.x required.\n"
                "Activate myenv1: pip install paddleocr paddlepaddle"
            )
        try:
            _ocr_engine = PaddleOCR(use_angle_cls=False, lang="en", device="cpu")
        except TypeError:
            _ocr_engine = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False)
    return _ocr_engine


def call_akshar(image_path: str) -> Dict[str, Any]:
    """
    Run local PaddleOCR on *image_path* and return a dict compatible with
    the original Sarvam Akshar API response format:

        {
            "words": [
                {
                    "text": "Hello",
                    "bbox": {"x": 10, "y": 20, "width": 50, "height": 15}
                },
                ...
            ]
        }

    Args:
        image_path: Absolute or relative path to a JPG/PNG image file.

    Returns:
        Dict with key "words" containing a list of word-level OCR results.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    ocr = _get_ocr_engine()

    # Run OCR
    try:
        result = ocr.predict(
            image_bgr,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        result = ocr.ocr(image_bgr, cls=False)

    words = []
    if result:
        first = result[0]
        if first is None:
            pass
        elif isinstance(first, dict):          # PaddleOCR 3.x
            texts  = first.get("rec_texts",  []) or []
            scores = first.get("rec_scores", []) or []
            polys  = first.get("rec_polys",  []) or []
            for text, score, poly in zip(texts, scores, polys):
                if not (text and text.strip()):
                    continue
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                words.append({
                    "text":  text.strip(),
                    "score": float(score),
                    "bbox":  {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                })
        else:                                  # PaddleOCR 2.x
            for line in first:
                if line is None:
                    continue
                polygon, (text, score) = line
                if not (text and text.strip()):
                    continue
                pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
                x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                words.append({
                    "text":  text.strip(),
                    "score": float(score),
                    "bbox":  {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                })

    return {"words": words}