import cv2

def load_image(path: str):
    """Load image from disk using OpenCV (BGR format)."""
    return cv2.imread(path)

def resize(image, width: int = 800):
    """Resize image to the given width while preserving aspect ratio."""
    h, w = image.shape[:2]
    ratio = width / w
    return cv2.resize(image, (width, int(h * ratio)))
