"""Tone mapping, image IO and error metrics."""

from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

from .guiding import luminance


def aces_tonemap(x: np.ndarray) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def srgb_encode(v: np.ndarray) -> np.ndarray:
    return np.where(
        v <= 0.0031308,
        12.92 * v,
        1.055 * np.power(np.maximum(v, 0.0), 1.0 / 2.4) - 0.055,
    )


def tonemap(hdr: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """HDR linear (H,W,3) -> sRGB float [0,1]."""
    x = np.asarray(hdr, dtype=np.float64) * exposure
    return srgb_encode(aces_tonemap(x))


def to_uint8(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_png(path: str, hdr: np.ndarray, exposure: float = 1.0) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if hdr.dtype == np.uint8:
        arr = hdr
    else:
        arr = to_uint8(tonemap(hdr, exposure))
    Image.fromarray(arr, "RGB").save(path)
    return path


def save_raw_png(path: str, rgb01: np.ndarray) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(to_uint8(rgb01), "RGB").save(path)
    return path


def exposure_from_reference(ref_hdr: np.ndarray, key: float = 0.18) -> float:
    lum = luminance(ref_hdr)
    mean_lum = float(np.mean(lum))
    if mean_lum < 1e-8:
        return 1.0
    return float(np.clip(key / mean_lum, 0.05, 20.0))


def image_metrics(
    hdr: np.ndarray,
    ref_hdr: np.ndarray | None,
    exposure: float = 1.0,
) -> dict[str, float]:
    lum = luminance(hdr)
    out: dict[str, float] = {"mean_luma": float(lum.mean())}
    out["mean_rgb"] = [float(hdr[..., c].mean()) for c in range(3)]  # type: ignore[assignment]
    if ref_hdr is None or hdr.shape != ref_hdr.shape:
        return out
    ref_lum = luminance(ref_hdr)
    # Display-referred error with the same exposure for every strategy.
    disp = tonemap(hdr, exposure)
    ref_disp = tonemap(ref_hdr, exposure)
    out["rmse_display"] = float(np.sqrt(np.mean((disp - ref_disp) ** 2)))
    out["mae_display"] = float(np.mean(np.abs(disp - ref_disp)))
    # HDR relative MSE is robust to fireflies and exposure choices.
    rel = (hdr - ref_hdr) ** 2 / (ref_hdr**2 + 0.05**2)
    out["rel_mse"] = float(np.mean(rel))
    out["rmse_luma_hdr"] = float(np.sqrt(np.mean((lum - ref_lum) ** 2)))
    # Mean bias of luminance.
    out["mean_bias"] = float(lum.mean() - ref_lum.mean())
    return out


def roi_rmse(
    hdr: np.ndarray,
    ref_hdr: np.ndarray,
    exposure: float,
    box_norm: tuple[float, float, float, float],
) -> float | None:
    """Display-domain RMSE inside a normalized box (x0,y0,x1,y1 in 0..1)."""
    if ref_hdr is None or hdr.shape != ref_hdr.shape:
        return None
    h, w = hdr.shape[:2]
    x0 = int(box_norm[0] * w); x1 = int(box_norm[2] * w)
    y0 = int(box_norm[1] * h); y1 = int(box_norm[3] * h)
    if x1 <= x0 or y1 <= y0:
        return None
    a = tonemap(hdr[y0:y1, x0:x1], exposure)
    b = tonemap(ref_hdr[y0:y1, x0:x1], exposure)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def error_heatmap(
    hdr: np.ndarray, ref_hdr: np.ndarray, exposure: float = 1.0, gain: float = 4.0
) -> np.ndarray:
    """Amplified absolute display error as a false-color uint8 image."""
    d = np.abs(tonemap(hdr, exposure) - tonemap(ref_hdr, exposure)).mean(axis=2)
    amp = np.clip(d * gain, 0.0, 1.0)
    # simple iron colormap
    r = np.clip(amp * 3.0, 0, 1)
    g = np.clip(amp * 3.0 - 1.0, 0, 1)
    b = np.clip(amp * 3.0 - 2.0, 0, 1)
    return to_uint8(np.stack([r, g, b], axis=-1))


def hstack_png(images: list[np.ndarray], gap: int = 6, bg: int = 24) -> np.ndarray:
    """Horizontally concatenate uint8 RGB images with gaps."""
    h = min(im.shape[0] for im in images)
    resized = []
    for im in images:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            new_w = int(im.shape[1] * scale)
            pil = Image.fromarray(im).resize((new_w, h), Image.BILINEAR)
            resized.append(np.asarray(pil))
        else:
            resized.append(im)
    w = sum(im.shape[1] for im in resized) + gap * (len(resized) - 1)
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    x = 0
    for im in resized:
        canvas[:, x : x + im.shape[1]] = im
        x += im.shape[1] + gap
    return canvas
