"""Finish pass: film grain, vignette, chromatic aberration, bloom.

Ties the procedural layers into one 'photographed' image.
"""
import numpy as np

from .noise import gaussian_blur
from .util import bilinear_wrap


def finish(img: np.ndarray, rng, film_grain: float = 0.018, vignette: float = 0.22,
           bloom: float = 0.18, bloom_thresh: float = 0.78, chroma: float = 0.6,
           emission=None, bloom_emissive: float = 1.6,
           bloom_radius: float = 0.012) -> np.ndarray:
    size = img.shape[0]
    out = img.astype(np.float64).copy()

    emissive_glow = emission is not None and bloom_emissive > 0
    if bloom > 0 or emissive_glow:
        src = np.clip(out - bloom_thresh, 0, 1) * bloom  # bright-region bloom
        if emissive_glow:
            src = src + emission * bloom_emissive        # emitters always bloom
        glow = gaussian_blur(src, max(size * bloom_radius, 0.5))
        out = out + glow * 2.5

    if chroma > 0:
        v = (np.arange(size) + 0.5) / size * 2.0 - 1.0
        X, Y = np.meshgrid(v, v)
        px = (X * 0.5 + 0.5) * size - 0.5
        py = (Y * 0.5 + 0.5) * size - 0.5
        for c, k in ((0, 1.0), (2, -1.0)):  # shift R out, B in
            e = 1.0 + k * chroma * 0.0016
            xs = ((X * e) * 0.5 + 0.5) * size - 0.5
            ys = ((Y * e) * 0.5 + 0.5) * size - 0.5
            out[..., c] = bilinear_wrap(out[..., c], xs, ys)
        _ = px, py

    if vignette > 0:
        v = (np.arange(size) + 0.5) / size * 2.0 - 1.0
        X, Y = np.meshgrid(v, v)
        r2 = X * X + Y * Y
        out *= (1.0 - vignette * np.clip(r2 * 0.55, 0, 1))[..., None]

    if film_grain > 0:
        g = rng.normal(0.0, 1.0, (size, size))
        g = g * 0.75 + gaussian_blur(g, 0.7) * 0.25
        out += (g * film_grain)[..., None]

    return np.clip(out, 0.0, 1.0)
