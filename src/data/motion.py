"""
motion.py -- Optical-flow extrapolation of a radar frame (tasks/plan_motion_input.md).

Used two ways: as a forecaster in its own right (scripts/spike_240km_extrapolation.py)
and as an extra model input channel ("where would the rain be if it just kept moving").

DIS optical flow, not Farneback: Farneback read a known synthetic shift of (10, -4) px
as (0.6-5, -0.3-2) px; DIS recovers it exactly (lesson L034). Backward mapping samples
the motion at the DESTINATION pixel, which is usually dry with raw flow ~0 -- that
stalls rain at its leading edge -- so the flow is smoothed by rain-weighted normalised
convolution and falls back to the median rainy-pixel motion far from any rain.
"""

import cv2
import numpy as np


def _u8(f: np.ndarray) -> np.ndarray:
    """mm/hr -> 0..255 on NEA's log scale (0.5..100 mm/hr ramp)."""
    return np.clip(np.log1p(f) / np.log1p(100.0) * 255, 0, 255).astype(np.uint8)


def advect(prev: np.ndarray, now: np.ndarray, factor: float, smooth_px: float) -> np.ndarray:
    """`now` moved forward by `factor` times the displacement estimated prev -> now.

    prev, now : (H, W) mm/hr, NaN-free, same grid
    factor    : lead / (time between prev and now), e.g. 65 min / 15 min
    smooth_px : Gaussian sigma of the flow smoothing, in pixels
    Rain entering from beyond the grid edge is unknown and set to 0.
    """
    prev = np.asarray(prev, np.float32)
    now = np.asarray(now, np.float32)
    w = ((prev > 0) | (now > 0)).astype(np.float32)
    if w.sum() < 20:                                   # (almost) dry: nothing to move
        return now.copy()
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(_u8(prev), _u8(now), None)
    den = cv2.GaussianBlur(w, (0, 0), smooth_px)
    med = np.median(flow[w > 0], axis=0)
    sm = np.empty_like(flow)
    for c in range(2):
        num = cv2.GaussianBlur(flow[..., c] * w, (0, 0), smooth_px)
        sm[..., c] = np.where(den > 1e-3, num / np.maximum(den, 1e-3), med[c])
    h, w_ = now.shape
    gx, gy = np.meshgrid(np.arange(w_, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(now, gx - factor * sm[..., 0], gy - factor * sm[..., 1],
                     cv2.INTER_NEAREST, borderValue=0)


# 70 km grid: 0.29 km pixels; smooth over ~20 km as the 240 km spike did at 1 km/px.
PX_KM_70 = 0.29
FLOW_GAP_FRAMES = 3                                    # t-4 -> t-1: 15 min


def motion_forecast(frames: np.ndarray, minutes: int) -> np.ndarray:
    """Extrapolation of the last frame `minutes` ahead from a (CF, H, W) mm/hr stack,
    oldest first, using only frames[-1 - FLOW_GAP_FRAMES] and frames[-1]."""
    if frames.shape[0] <= FLOW_GAP_FRAMES:
        raise ValueError(f"motion needs > {FLOW_GAP_FRAMES} context frames")
    return advect(frames[-1 - FLOW_GAP_FRAMES], frames[-1],
                  minutes / (5.0 * FLOW_GAP_FRAMES), 20.0 / PX_KM_70)
