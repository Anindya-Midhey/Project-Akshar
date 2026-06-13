"""
Module 2 - Warp Route
Handles grid-based image warping/dewarping for curved documents.
"""

import os
import uuid
import base64
import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from utils.warp import apply_grid_warp, create_default_grid

router = APIRouter()

# Directory for processed files
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


class WarpRequest(BaseModel):
    """Request model for grid-based warp."""
    image_path: str
    src_points: List[List[float]]  # Original grid positions [[x,y], ...]
    dst_points: List[List[float]]  # User-adjusted positions [[x,y], ...]
    grid_rows: int = 4
    grid_cols: int = 4


class DefaultGridRequest(BaseModel):
    """Request to get the default grid for an image."""
    width: int
    height: int
    grid_rows: int = 4
    grid_cols: int = 4


def save_and_encode(image, prefix: str = "warped") -> dict:
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


@router.post("/warp")
async def warp_image(request: WarpRequest):
    """
    Apply grid-based warp/dewarp transformation.
    
    Input:
        - image_path: path to the source image
        - src_points: original uniform grid positions
        - dst_points: user-adjusted grid positions
        - grid_rows, grid_cols: grid dimensions
        
    Returns: warped image path and base64 preview
    """
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    expected_count = request.grid_rows * request.grid_cols
    if len(request.src_points) != expected_count or len(request.dst_points) != expected_count:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_count} points ({request.grid_rows}x{request.grid_cols}), "
                   f"got src={len(request.src_points)}, dst={len(request.dst_points)}"
        )
    
    img = cv2.imread(request.image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image file")
    
    try:
        warped = apply_grid_warp(
            img,
            request.src_points,
            request.dst_points,
            request.grid_rows,
            request.grid_cols
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warp failed: {str(e)}")
    
    result = save_and_encode(warped, "warped")
    return {
        "status": "success",
        "message": "Grid warp applied",
        **result
    }


@router.post("/warp/grid")
async def get_default_grid(request: DefaultGridRequest):
    """
    Get the default uniform grid points for the given image dimensions.
    """
    grid = create_default_grid(
        request.width, request.height,
        request.grid_rows, request.grid_cols
    )
    return {
        "grid_rows": request.grid_rows,
        "grid_cols": request.grid_cols,
        "points": grid
    }
