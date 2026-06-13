"""
Module 1 — Page Split Detection
Classical page splitting via projection profile and Hough analysis.

Detects a vertical spine line on double-page book spreads and splits
the image into left and right pages.  Returns the original image
untouched when no spine is confidently detected (single-page input).
"""
from __future__ import annotations
import cv2
import numpy as np


# Aspect-ratio gate: if width/height < this, definitely single page
MIN_SPREAD_RATIO = 1.3


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_spine_x(img: np.ndarray) -> int | None:
    """
    Locate the vertical spine line on a double-page spread.
    Runs two independent detectors (projection-profile dip + Hough lines)
    and returns their median when they agree.
    """
    gray = _to_grayscale(img)
    h, w = gray.shape
    cx        = w // 2
    tolerance = int(w * 0.20)
    search_l  = w // 4
    search_r  = 3 * w // 4

    candidates: list[int] = []

    # ── Signal 1: projection-profile dip ──────────────────────────────────
    central   = gray[:, search_l:search_r]
    col_means = central.mean(axis=0).astype(np.float32)
    if col_means.size >= 21:
        col_means = np.convolve(col_means, np.ones(21) / 21.0, mode="same")
    dip_local = int(np.argmin(col_means))
    dip_x = search_l + dip_local
    if abs(dip_x - cx) <= tolerance:
        candidates.append(dip_x)

    # ── Signal 2: Hough line vote ──────────────────────────────────────────
    edges      = cv2.Canny(gray, threshold1=30, threshold2=100, apertureSize=3)
    edge_strip = np.zeros_like(edges)
    edge_strip[:, search_l:search_r] = edges[:, search_l:search_r]

    lines = cv2.HoughLinesP(
        edge_strip,
        rho=1, theta=np.pi / 180,
        threshold=int(h * 0.35),
        minLineLength=int(h * 0.30),
        maxLineGap=int(h * 0.05),
    )
    if lines is not None:
        vertical_xs: list[int] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dy == 0:
                continue
            if dx / dy < np.tan(np.radians(10)):
                mid_x = (x1 + x2) // 2
                if abs(mid_x - cx) <= tolerance:
                    vertical_xs.append(mid_x)

        if vertical_xs:
            xs   = np.array(vertical_xs)
            bins = np.arange(search_l, search_r + 21, 20)
            hist, edges_b = np.histogram(xs, bins=bins)
            peak_bin = int(np.argmax(hist))
            hough_x  = int((edges_b[peak_bin] + edges_b[peak_bin + 1]) / 2)
            if abs(hough_x - cx) <= tolerance:
                candidates.append(hough_x)

    if not candidates:
        return None

    return int(np.median(candidates))


def split_at(img: np.ndarray, x: int) -> tuple[np.ndarray, np.ndarray]:
    """Split image at column x into left and right halves."""
    return img[:, :x].copy(), img[:, x:].copy()


def maybe_split(img: np.ndarray) -> list[np.ndarray]:
    """
    Return a list of page images:
      - [img]           → single page (aspect ratio too narrow or no spine)
      - [left, right]   → double-page spread with detected spine
    """
    h, w  = img.shape[:2]
    ratio = w / h

    if ratio < MIN_SPREAD_RATIO:
        return [img]

    spine_x = detect_spine_x(img)
    if spine_x is None:
        return [img]

    left, right = split_at(img, spine_x)
    return [left, right]
