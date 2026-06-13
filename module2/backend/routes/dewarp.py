"""
PROJECT AKSHAR - Dewarp + Deskew Routes
"""

import os
import uuid
import base64
import cv2
from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel

from utils.scantailor import apply_grid_dewarp, analyze_dewarp_grid, apply_custom_grid_dewarp
from utils.dewarp_ml.predictor import run_auto_dewarp, is_model_available
from utils.deskew import deskew_image, apply_manual_deskew

router = APIRouter()

# Directory for processed files
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DewarpRequest(BaseModel):
    """Request model for cylindrical-surface mesh dewarping."""
    image_path: str
    strength: float = 1.0  # 0.5 to 1.5; 1.0 = full correction
    depth_perception: float = 2.0  # camera distance heuristic (1.0–3.0)
    row_curves: Optional[List[List[List[float]]]] = None  # manual grid from frontend


class AutoDewarpRequest(BaseModel):
    """Request model for ML-based automatic dewarping (ICCV 2023 model)."""
    image_path: str


class DeskewRequest(BaseModel):
    """Request model for auto deskew."""
    image_path: str


class ManualDeskewRequest(BaseModel):
    """Request model for manual deskew with user-specified angle."""
    image_path: str
    angle: float  # degrees; positive = CW, negative = CCW


class AnalyzeGridRequest(BaseModel):
    """Request model for grid analysis visualization."""
    image_path: str
    n_cols: int = 20  # number of vertical grid lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_and_encode(image, prefix: str = "processed") -> dict:
    """Save a processed image and return its path + base64 encoding."""
    output_id = str(uuid.uuid4())
    output_path = os.path.join(PROCESSED_DIR, f"{prefix}_{output_id}.png")
    cv2.imwrite(output_path, image)

    _, buffer = cv2.imencode('.png', image)
    b64 = base64.b64encode(buffer).decode('utf-8')
    height, width = image.shape[:2]

    return {
        "image_path": output_path,
        "preview": f"data:image/png;base64,{b64}",
        "width": width,
        "height": height
    }


def _load_image(image_path: str):
    """Load and validate an image from the given path."""
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image file")

    return img


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/dewarp")
async def dewarp_image(request: DewarpRequest):
    """
    Apply ScanTailor-Advanced-style cylindrical-surface mesh dewarping.

    Detects text-line curves to build top/bottom directrix polylines,
    constructs a cylindrical surface model (4-point homography + arc-length
    mapping + generatrix 1-D homography), then remaps with bicubic
    interpolation.

    Args:
        image_path:       Path to source image.
        strength:         Correction strength (default 1.0 = full).
        depth_perception: Camera distance heuristic (1.0–3.0, default 2.0).
        row_curves:       Optional manual grid from the frontend editor.
    """
    img = _load_image(request.image_path)

    try:
        if request.row_curves:
            dewarped = apply_custom_grid_dewarp(
                img, request.row_curves,
                strength=request.strength,
                depth_perception=request.depth_perception,
            )
        else:
            dewarped = apply_grid_dewarp(
                img,
                strength=request.strength,
                depth_perception=request.depth_perception,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mesh dewarp failed: {str(e)}")

    result = _save_and_encode(dewarped, "dewarped")
    return {
        "status": "success",
        "method": "mesh_dewarp",
        "strength": request.strength,
        "depth_perception": request.depth_perception,
        "message": f"ScanTailor cylindrical mesh dewarp applied "
                   f"(strength={request.strength:.2f}, depth={request.depth_perception:.1f})",
        **result
    }


@router.post("/deskew")
async def auto_deskew(request: DeskewRequest):
    """
    Auto-detect skew angle from the document and correct it.

    Uses the automatic deskew logic ported from deskew_img.py:
    projection-profile scoring first, Hough fallback second, min-area fallback
    last, then rotates without cutting the page canvas.
    Returns the detected angle along with the corrected image.
    """
    img = _load_image(request.image_path)

    try:
        deskewed, angle = deskew_image(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deskew failed: {str(e)}")

    result = _save_and_encode(deskewed, "deskewed")
    return {
        "status": "success",
        "method": "auto_deskew",
        "detected_angle": round(angle, 3),
        "message": f"Auto deskew applied (detected angle: {angle:.2f}°)",
        **result
    }


@router.post("/deskew/manual")
async def manual_deskew(request: ManualDeskewRequest):
    """
    Apply a user-specified rotation correction.

    Args:
        image_path: Path to source image.
        angle:      Rotation angle in degrees. Positive = CW, negative = CCW.
    """
    if not -90 <= request.angle <= 90:
        raise HTTPException(
            status_code=400,
            detail="Angle must be between -90 and 90 degrees"
        )

    img = _load_image(request.image_path)

    try:
        deskewed = apply_manual_deskew(img, angle_deg=request.angle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual deskew failed: {str(e)}")

    result = _save_and_encode(deskewed, "deskewed_manual")
    return {
        "status": "success",
        "method": "manual_deskew",
        "angle_applied": request.angle,
        "message": f"Manual deskew applied (angle: {request.angle:.2f}°)",
        **result
    }


@router.post("/dewarp/auto")
async def auto_dewarp_ml(request: AutoDewarpRequest):
    """
    Apply ML-based automatic document dewarping using the ICCV 2023 neural network
    (Foreground and Text-lines Aware Document Image Rectification).

    The model runs fully automatically — no grid interaction required.
    On first call the model is lazy-loaded (~1-2 s); subsequent calls are fast.

    Args:
        image_path: Path to source image.
    """
    if not is_model_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Auto-dewarp model not available. "
                "Please ensure '30.pt' exists in the test dewarp pretrained_models/ folder, "
                "or set the DEWARP_MODEL_PATH environment variable."
            )
        )

    img = _load_image(request.image_path)

    try:
        dewarped = run_auto_dewarp(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML dewarp failed: {str(e)}")

    result = _save_and_encode(dewarped, "auto_dewarped")
    return {
        "status": "success",
        "method": "ml_dewarp",
        "message": "ICCV 2023 neural network auto-dewarp applied",
        **result,
    }


@router.post("/dewarp/analyze-grid")
async def analyze_grid(request: AnalyzeGridRequest):
    """
    Analyze the document warp mesh WITHOUT applying any correction.
    Returns the detected row-curve grid as sampled (x, y) point arrays —
    same data ScanTailor uses to render its blue grid overlay.

    Use this BEFORE applying /dewarp to preview the detected grid on the canvas.

    Returns:
      detected: bool — whether a usable grid was found
      row_curves: list of polylines (one per text row + top/bottom borders)
      col_lines: list of polylines (vertical connectors at evenly-spaced x positions)
      row_count: number of text rows detected
      width, height: image dimensions (for coordinate scaling)
    """
    img = _load_image(request.image_path)

    try:
        grid = analyze_dewarp_grid(img, n_cols=request.n_cols)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid analysis failed: {str(e)}")

    return {
        "status": "success",
        **grid,
    }
