"""
Module 1 — Auto Perspective Correction
=======================================
Extracted from perspective_04.py for use inside the Module 1 pipeline.

Public API
----------
    auto_perspective(image: np.ndarray) -> np.ndarray

If a plausible page quad is detected the image is perspective-corrected and
returned. If detection fails the original image is returned unchanged so that
the pipeline always continues without crashing.
"""

from __future__ import annotations

import cv2
import numpy as np


# ── Tunables ────────────────────────────────────────────────────────────────
MIN_PAGE_AREA_RATIO = 0.35
DEFAULT_SHARPNESS   = 0.0   # sharpening is already handled by enhance.py


# ── Helpers ─────────────────────────────────────────────────────────────────

def _resize_for_detection(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image.copy(), 1.0
    scale = max_dim / float(longest)
    resized = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _odd_at_least(value: int, minimum: int) -> int:
    value = max(value, minimum)
    return value if value % 2 == 1 else value + 1


def _build_detection_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    height, width = contrast.shape[:2]
    block_size = _odd_at_least(int(min(height, width) * 0.035), 31)

    adaptive = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, 9,
    )

    low  = max(12, int(np.percentile(contrast, 6)))
    high = min(180, max(low * 3, 55))
    edges = cv2.Canny(contrast, low, high)

    _, light_gray_border = cv2.threshold(contrast, 245, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_or(cv2.bitwise_or(adaptive, edges), light_gray_border)

    close_size = max(3, int(min(height, width) * 0.006))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size)),
        iterations=1,
    )
    dilate_size = max(3, int(min(height, width) * 0.004))
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size)),
        iterations=1,
    )
    return mask


def _order_points(points: np.ndarray) -> np.ndarray:
    points  = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums    = points.sum(axis=1)
    diffs   = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _area_of_quad(points: np.ndarray) -> float:
    return float(cv2.contourArea(_order_points(points).astype(np.float32)))


def _quad_is_plausible(points: np.ndarray, image_shape: tuple) -> bool:
    height, width = image_shape[:2]
    image_area    = float(height * width)
    quad          = _order_points(points)
    quad_area     = _area_of_quad(quad)

    if quad_area < image_area * MIN_PAGE_AREA_RATIO:
        return False

    x_min, x_max = float(np.min(quad[:, 0])), float(np.max(quad[:, 0]))
    y_min, y_max = float(np.min(quad[:, 1])), float(np.max(quad[:, 1]))

    if (x_max - x_min) / float(width) < 0.45:
        return False
    if (y_max - y_min) / float(height) < 0.45:
        return False

    side_lengths = [
        np.linalg.norm(quad[1] - quad[0]),
        np.linalg.norm(quad[2] - quad[1]),
        np.linalg.norm(quad[2] - quad[3]),
        np.linalg.norm(quad[3] - quad[0]),
    ]
    if min(side_lengths) < min(height, width) * 0.30:
        return False

    return True


def _quad_from_contour(contour: np.ndarray) -> np.ndarray:
    hull      = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for eps in (0.012, 0.018, 0.025, 0.035, 0.05, 0.075):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return _order_points(approx.reshape(4, 2))
    points = hull.reshape(-1, 2)
    sums   = points.sum(axis=1)
    diffs  = np.diff(points, axis=1).reshape(-1)
    quad   = np.array(
        [points[np.argmin(sums)], points[np.argmin(diffs)],
         points[np.argmax(sums)], points[np.argmax(diffs)]],
        dtype=np.float32,
    )
    return _order_points(quad)


def _contour_candidate(mask: np.ndarray) -> np.ndarray | None:
    height, width = mask.shape[:2]
    image_area    = float(height * width)
    contours, _   = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray]] = []

    for contour in contours:
        if len(contour) < 20:
            continue
        hull      = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area < image_area * 0.18:
            continue
        quad      = _quad_from_contour(contour)
        quad_area = _area_of_quad(quad)
        if quad_area < image_area * MIN_PAGE_AREA_RATIO:
            continue
        x, y, box_w, box_h = cv2.boundingRect(hull)
        if box_w / float(width) < 0.45 or box_h / float(height) < 0.45:
            continue
        perimeter = cv2.arcLength(hull, True)
        candidates.append((quad_area + perimeter * 10.0, quad))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _first_foreground_points(mask: np.ndarray, side: str, max_scan_ratio: float = 0.33) -> np.ndarray:
    height, width = mask.shape[:2]
    points: list[tuple[int, int]] = []

    if side in {"top", "bottom"}:
        step       = max(1, width // 400)
        scan_limit = int(height * max_scan_ratio)
        for x in range(0, width, step):
            col  = mask[:scan_limit, x] if side == "top" else mask[height - scan_limit:, x]
            hits = np.flatnonzero(col > 0)
            if not len(hits):
                continue
            y = int(hits[0]) if side == "top" else int(height - scan_limit + hits[-1])
            points.append((x, y))
    else:
        step       = max(1, height // 400)
        scan_limit = int(width * max_scan_ratio)
        for y in range(0, height, step):
            row  = mask[y, :scan_limit] if side == "left" else mask[y, width - scan_limit:]
            hits = np.flatnonzero(row > 0)
            if not len(hits):
                continue
            x = int(hits[0]) if side == "left" else int(width - scan_limit + hits[-1])
            points.append((x, y))

    if len(points) < 12:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _fit_line(points: np.ndarray) -> tuple[float, float, float] | None:
    if len(points) < 12:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    a, b, c = float(vy), float(-vx), float(vx * y0 - vy * x0)
    norm = (a * a + b * b) ** 0.5
    if norm < 1e-6:
        return None
    return a / norm, b / norm, c / norm


def _line_distances(points: np.ndarray, line: tuple) -> np.ndarray:
    a, b, c = line
    return np.abs(points[:, 0] * a + points[:, 1] * b + c)


def _keep_outer_points(points: np.ndarray, side: str) -> np.ndarray:
    if len(points) < 12:
        return points
    if side == "top":
        return points[points[:, 1] <= np.percentile(points[:, 1], 72)]
    if side == "bottom":
        return points[points[:, 1] >= np.percentile(points[:, 1], 28)]
    if side == "left":
        return points[points[:, 0] <= np.percentile(points[:, 0], 72)]
    return points[points[:, 0] >= np.percentile(points[:, 0], 28)]


def _robust_border_line(points: np.ndarray, side: str) -> tuple | None:
    filtered = _keep_outer_points(points, side)
    if len(filtered) < 12:
        return None
    for _ in range(4):
        line = _fit_line(filtered)
        if line is None:
            return None
        distances    = _line_distances(filtered, line)
        cutoff       = max(4.0, float(np.percentile(distances, 70)) * 2.2)
        next_filtered = filtered[distances <= cutoff]
        if len(next_filtered) < 12 or len(next_filtered) == len(filtered):
            break
        filtered = next_filtered
    return _fit_line(filtered)


def _intersect_lines(a: tuple, b: tuple) -> np.ndarray | None:
    a1, b1, c1 = a
    a2, b2, c2 = b
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    return np.asarray(
        [(b1 * c2 - b2 * c1) / det, (c1 * a2 - c2 * a1) / det],
        dtype=np.float32,
    )


def _scanline_candidate(mask: np.ndarray) -> np.ndarray | None:
    lines: dict[str, tuple] = {}
    for side in ("top", "right", "bottom", "left"):
        pts  = _first_foreground_points(mask, side, max_scan_ratio=0.45)
        line = _robust_border_line(pts, side)
        if line is None:
            return None
        lines[side] = line

    corners = [
        _intersect_lines(lines["top"],    lines["left"]),
        _intersect_lines(lines["top"],    lines["right"]),
        _intersect_lines(lines["bottom"], lines["right"]),
        _intersect_lines(lines["bottom"], lines["left"]),
    ]
    if any(c is None for c in corners):
        return None

    quad = _order_points(np.asarray(corners, dtype=np.float32))
    h, w = mask.shape[:2]
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)

    if not _quad_is_plausible(quad, mask.shape):
        return None
    return quad


def _detect_page_quad(image: np.ndarray, max_dim: int = 1800) -> np.ndarray | None:
    small, scale = _resize_for_detection(image, max_dim)
    mask         = _build_detection_mask(small)

    candidates = [
        q for q in (_contour_candidate(mask), _scanline_candidate(mask))
        if q is not None and _quad_is_plausible(q, small.shape)
    ]

    if not candidates:
        return None

    quad = max(candidates, key=_area_of_quad)
    return _order_points(quad / scale)


def _warp_page(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    quad = _order_points(quad)
    h, w = image.shape[:2]
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)

    tl, tr, br, bl = quad
    out_w = max(1, int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))))
    out_h = max(1, int(round(max(np.linalg.norm(br - tr), np.linalg.norm(bl - tl)))))

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(
        image, M, (out_w, out_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def auto_perspective(image: np.ndarray, max_dim: int = 1800) -> np.ndarray:
    """
    Detect the page boundary in *image* and apply a perspective warp to
    produce a flat, rectangular crop of the document.

    Parameters
    ----------
    image   : BGR numpy array (any resolution).
    max_dim : Long-side limit used for detection only; warping always
              uses original-resolution pixels.

    Returns
    -------
    Perspective-corrected BGR image.
    If no plausible page quad is found the original image is returned
    unchanged so the pipeline can always continue.
    """
    quad = _detect_page_quad(image, max_dim=max_dim)

    if quad is None:
        # Detection failed — pass the image through untouched.
        return image

    return _warp_page(image, quad)
