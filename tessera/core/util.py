"""Shared utilities: seeded RNG, param hashing, image IO."""
import hashlib
import json

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def rng_for(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def param_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def to_image(arr: np.ndarray) -> Image.Image:
    """HxW or HxWx3 float [0,1] -> PIL image."""
    a = np.clip(arr, 0.0, 1.0)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), "RGB")


def save_tile(arr: np.ndarray, path: str, family: str, seed: int, params: dict) -> None:
    img = to_image(arr)
    meta = PngInfo()
    meta.add_text("cram_family", family)
    meta.add_text("cram_seed", str(seed))
    meta.add_text("cram_params", json.dumps(params, sort_keys=True))
    img.save(path, pnginfo=meta, optimize=True)


def downscale(arr: np.ndarray, size: int) -> np.ndarray:
    """Lanczos-downsample a float [0,1] image array to size x size
    (supersampled anti-aliasing)."""
    img = to_image(arr)
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    """Supports descending edges (e0 > e1): result fades 1 -> 0 as x rises."""
    if e0 > e1:
        t = np.clip((x - e1) / (e0 - e1), 0.0, 1.0)
        t = 1.0 - t
    else:
        t = np.clip((x - e0) / max(e1 - e0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def grid_coords(size: int, centered: bool = True):
    """Return (X, Y) in [-1,1] (centered) or [0,1)."""
    if centered:
        v = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    else:
        v = np.arange(size) / size
    X, Y = np.meshgrid(v, v)
    return X, Y


def bilinear_wrap(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample 2D field `img` at float pixel coords (x, y) with wrap-around."""
    h, w = img.shape[:2]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    x0 %= w
    y0 %= h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]
    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    return (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
            + v01 * (1 - fx) * fy + v11 * fx * fy)
