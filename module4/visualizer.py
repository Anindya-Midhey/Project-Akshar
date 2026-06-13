import fitz  # PyMuPDF
import os
from typing import List

def highlight_pdf_region(
    input_pdf_path: str, 
    output_pdf_path: str, 
    highlights: list,
    color: tuple = (1, 0, 0),
    opacity: float = 0.3
):
    """
    Renders translucent highlights over all given bounding boxes on the PDF page.
    This completes the Grounded VQA requirement of Project Akshar.
    
    Args:
        input_pdf_path: Path to the originally clean PDF from Module 2.
        output_pdf_path: Path where the highlighted PDF will be saved for UI display.
        highlights: List of dicts [{"page_num": int, "bbox": [xmin, ymin, xmax, ymax]}]
        color: RGB tuple for the highlight color (normalized to [0, 1]). Default is red.
        opacity: Transparency level (0 is invisible, 1 is opaque).
        
    Returns:
        bool: True if overlays were drawn and saved successfully, False otherwise.
    """
    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(f"Input PDF missing for visualization: {input_pdf_path}")
        
    try:
        # Open document
        doc = fitz.open(input_pdf_path)
        
        for h in highlights:
            page_num = h.get("page_num")
            bbox = h.get("bbox")
            
            # Validate page number bounds (PyMuPDF pages are 0-indexed internally)
            if not page_num or page_num < 1 or page_num > len(doc):
                continue
                
            page = doc[page_num - 1]
            
            # Define the rectangle using the retrieved coordinates.
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            
            # Add a translucent highlight annotation to the page
            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=color, fill=color)
            annot.set_opacity(opacity)
            annot.update()
        
        # Save output
        print(f"Highlighted {len(highlights)} rectangle(s) across the PDF. Saving to {output_pdf_path}...")
        doc.save(output_pdf_path)
        doc.close()
        
        return True
        
    except Exception as e:
        print(f"Error visualizing PDF bounding box: {e}")
        return False

if __name__ == "__main__":
    pass
