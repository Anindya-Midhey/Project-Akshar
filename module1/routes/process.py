"""
Module 1 — Process Route (standalone reference)
Used by module1/backend/main.py for independent testing.
In production, use module2/backend/routes/module1_process.py instead.
"""

import os
import sys

# Ensure module1/backend is in sys.path for "m1.*" imports
_M1_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _M1_BACKEND not in sys.path:
    sys.path.insert(0, _M1_BACKEND)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from m1.pipeline import process_image

router = APIRouter()


class ProcessRequest(BaseModel):
    image_path: str
    mode: str = "color"


@router.post("/m1/process")
async def m1_process(request: ProcessRequest):
    """Run full Module 1 pipeline. See module2/backend/routes/module1_process.py for docs."""
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        pages = process_image(request.image_path, mode=request.mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Module 1 processing failed: {exc}")

    return {
        "status":        "success",
        "page_count":    len(pages),
        "pages":         pages,
        "steps_applied": ["resize", "page_split_check", "sam_extract",
                          "remove_border", "shadow_removal", "enhance"],
    }
