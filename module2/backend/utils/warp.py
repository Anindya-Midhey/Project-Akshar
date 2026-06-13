"""
Module 2 - Image Warping Utility
Grid-based image dewarping using control points and OpenCV remap.
Useful for correcting curved/bent document pages.
"""

import cv2
import numpy as np
from scipy.interpolate import griddata


def create_default_grid(width: int, height: int, rows: int = 4, cols: int = 4):
    """
    Create a uniform grid of control points over the image.
    
    Args:
        width: Image width
        height: Image height
        rows: Number of grid rows
        cols: Number of grid columns
        
    Returns:
        List of [x, y] grid points, row by row
    """
    points = []
    for r in range(rows):
        for c in range(cols):
            x = c * (width - 1) / (cols - 1)
            y = r * (height - 1) / (rows - 1)
            points.append([x, y])
    return points


def apply_grid_warp(
    image: np.ndarray,
    src_points: list,
    dst_points: list,
    grid_rows: int = 4,
    grid_cols: int = 4,
) -> np.ndarray:
    """
    Apply grid-based warp transformation.
    
    The user moves control points from src_points to dst_points.
    We compute a smooth remap that deforms the image so that the
    dst_points move to where the src_points are (i.e., straightening
    the document).
    
    Args:
        image: Input image (BGR)
        src_points: Original grid positions [[x,y], ...] (uniform grid)
        dst_points: User-adjusted positions [[x,y], ...] (where the grid was dragged)
        grid_rows: Number of rows in the control grid
        grid_cols: Number of columns in the control grid
        
    Returns:
        Warped image as NumPy array
    """
    h, w = image.shape[:2]
    
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    
    # Create a dense pixel grid for the output image
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    # Generate output coordinate grid
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # We want to find, for each pixel in the OUTPUT image, where to
    # sample from the INPUT image. The user dragged dst -> src, meaning
    # pixel at dst should map to pixel at src in the original.
    # So we interpolate: for each output pixel, compute the source pixel.
    
    # Compute displacement: how much each control point moved
    # displacement = src - dst (source position minus destination position)
    disp_x = src[:, 0] - dst[:, 0]
    disp_y = src[:, 1] - dst[:, 1]
    
    # Interpolate the displacement over the whole image using the dst points
    # as the known locations
    dense_disp_x = griddata(dst, disp_x, (grid_x, grid_y), method='cubic', fill_value=0.0)
    dense_disp_y = griddata(dst, disp_y, (grid_x, grid_y), method='cubic', fill_value=0.0)
    
    # The remap source coordinates: each output pixel samples from
    # (its own position + displacement)
    map_x = grid_x + dense_disp_x.astype(np.float32)
    map_y = grid_y + dense_disp_y.astype(np.float32)
    
    # Apply remap
    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )
    
    return warped
