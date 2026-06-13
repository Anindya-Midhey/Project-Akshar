"""
Module 2 - Interactive Image Workbench: PDF Export Utility
Converts processed images into a downloadable PDF document.
"""

import io
import img2pdf
import cv2
import numpy as np
from PIL import Image
from typing import List


def images_to_pdf(image_paths: List[str]) -> bytes:
    """
    Convert a list of image files into a single PDF document.
    Uses img2pdf for lossless conversion (no re-encoding).
    
    Args:
        image_paths: List of file paths to images
        
    Returns:
        PDF file content as bytes
        
    Raises:
        FileNotFoundError: If any image path doesn't exist
        ValueError: If no valid images provided
    """
    if not image_paths:
        raise ValueError("No image paths provided")
    
    # img2pdf works best with JPEG/PNG files
    # We ensure all images are in a compatible format
    valid_image_data = []
    
    for path in image_paths:
        # Read the image and convert to PNG bytes for img2pdf
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Save to bytes buffer as PNG
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        valid_image_data.append(buf.read())
    
    if not valid_image_data:
        raise ValueError("No valid images found")
    
    # Convert to PDF using img2pdf
    pdf_bytes = img2pdf.convert(valid_image_data)
    
    return pdf_bytes


def numpy_images_to_pdf(images: List[np.ndarray]) -> bytes:
    """
    Convert a list of NumPy image arrays into a single PDF document.
    
    Args:
        images: List of images as NumPy arrays (BGR format)
        
    Returns:
        PDF file content as bytes
    """
    if not images:
        raise ValueError("No images provided")
    
    image_data = []
    for img in images:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        image_data.append(buf.read())
    
    pdf_bytes = img2pdf.convert(image_data)
    return pdf_bytes
