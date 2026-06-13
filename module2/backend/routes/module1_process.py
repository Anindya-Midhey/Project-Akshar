"""
Module 2 — Module 1 Process Route
===================================
Thin wrapper that imports module1's pipeline and exposes it as a FastAPI
route registered under the /m1 prefix in module2's main.py.

Why this file lives in module2/backend/routes/:
  Module 1 has its own standalone backend folder (module1/backend/) but
  its processing endpoint is served through the same FastAPI app as Module 2
  so the frontend only needs to talk to one server (port 8000).

Path resolution strategy:
  This file computes the absolute path to module1/backend/ at import time
  and inserts it into sys.path with the unique package name "m1" to avoid
  naming conflicts with module2's own "utils" package.
"""

import sys
import os

# ── Resolve module1/backend absolute path ─────────────────────────────────
#   module2/backend/routes/module1_process.py
#                 ↑ routes   ↑ backend   ↑ module2   ↑ project root
_M2_ROUTES_DIR = os.path.dirname(os.path.abspath(__file__))          # .../module2/backend/routes
_M2_BACKEND    = os.path.dirname(_M2_ROUTES_DIR)                      # .../module2/backend
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(_M2_BACKEND))        # .../Sarvam
_M1_BACKEND    = os.path.join(_PROJECT_ROOT, "module1")               # .../module1  (flat, no /backend/)

# Insert module1/backend into sys.path so "from m1.pipeline import ..." works.
# The "m1" package name is unique and does not conflict with module2's "utils".
if _M1_BACKEND not in sys.path:
    sys.path.insert(0, _M1_BACKEND)

# Now import the pipeline (lazy — SAM model is NOT loaded until first request)
from m1.pipeline import process_image  # noqa: E402

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class M1ProcessRequest(BaseModel):
    """Request body for POST /m1/process."""
    image_path: str
    mode: str = "color"   # "color" | "bw"


@router.post("/m1/process", tags=["Module 1"])
async def m1_process(request: M1ProcessRequest):
    """
    Run the full Module 1 SAM-based processing pipeline on a single image.

    Pipeline steps (executed sequentially):
      1. Load + resize to 800 px width
      2. Page-split check (double-page spread → 2 outputs, else 1)
      3. SAM page extraction (background → white, crop to page)
      4. Auto perspective correction (detect page quad, flatten skew/tilt)
      5. Remove residual black border
      6. Shadow removal (per-channel illumination normalisation)
      7. Enhancement (color: background-norm + sharpen)

    Returns
    -------
    {
      "status":        "success",
      "page_count":    1 | 2,
      "pages": [
        {
          "image_path": "/abs/path/to/m1_<uuid>.png",
          "preview":    "data:image/png;base64,...",
          "width":      <int>,
          "height":     <int>
        },
        ...
      ],
      "steps_applied": [...]
    }
    """
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        pages = process_image(request.image_path, mode=request.mode)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Module 1 processing failed: {exc}"
        )

    return {
        "status":        "success",
        "page_count":    len(pages),
        "pages":         pages,
        "steps_applied": [
            "resize",
            "page_split_check",
            "sam_extract",
            "auto_perspective",
            "remove_border",
            "shadow_removal",
            "enhance",
        ],
    }
