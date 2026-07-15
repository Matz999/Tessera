"""Procedural tileable grain overlays — the 'few KB of grain' realism trick.

Each returns an HxW field centered near 0.5; composite with overlay/multiply.
"""
import numpy as np

from .noise import fbm, gaussian_blur, perlin
from .util import norm01


def paper_fiber(size: int, rng) -> np.ndarray:
    base = fbm(size, rng, octaves=6, freq=16, gain=0.62)
    streak = fbm(size, rng, octaves=4, freq=8)
    idx = (np.arange(size) * 0.25).astype(np.int64) % size  # stretch -> fibers
    streak = streak[:, idx]
    return norm01(base * 0.7 + streak * 0.3)


def stone_speckle(size: int, rng) -> np.ndarray:
    fine = rng.uniform(0, 1, (size, size))
    fine = gaussian_blur(fine, 0.6)
    mid = fbm(size, rng, octaves=5, freq=24, gain=0.6)
    return norm01(fine * 0.55 + mid * 0.45)


def canvas_weave(size: int, rng) -> np.ndarray:
    freq = 64
    xs = np.arange(size) / size
    X, Y = np.meshgrid(xs, xs)
    thread = (np.sin(X * freq * 2 * np.pi) * 0.5 + 0.5) * 0.5 \
        + (np.sin(Y * freq * 2 * np.pi) * 0.5 + 0.5) * 0.5
    wobble = fbm(size, rng, octaves=4, freq=8)
    return norm01(thread * 0.6 + wobble * 0.4)


def brushed_metal(size: int, rng) -> np.ndarray:
    a = rng.uniform(0, 1, (size, size))
    for sigma in (1.0, 2.0):
        f = np.fft.fft2(a)
        fy = np.fft.fftfreq(size)[:, None]
        fx = np.fft.fftfreq(size)[None, :]
        k = np.exp(-2 * np.pi ** 2 * ((sigma * 14) ** 2 * fx ** 2 + sigma ** 2 * fy ** 2))
        a = np.real(np.fft.ifft2(f * k))
    return norm01(a)


def craquelure(size: int, rng) -> np.ndarray:
    """Thin dark crack lines from ridged noise valleys."""
    r = fbm(size, rng, octaves=4, freq=6, ridged=True)
    cracks = np.clip((r - 0.82) / 0.18, 0, 1) ** 1.5
    return 0.5 - cracks * 0.5 + (fbm(size, rng, octaves=3, freq=12) - 0.5) * 0.06


GRAINS = {
    "paper": paper_fiber,
    "stone": stone_speckle,
    "canvas": canvas_weave,
    "metal": brushed_metal,
    "craquelure": craquelure,
}


def apply_grain(img: np.ndarray, grain: np.ndarray, amount: float = 0.5,
                mode: str = "overlay") -> np.ndarray:
    """img HxWx3, grain HxW around 0.5."""
    g = grain[..., None]
    if mode == "multiply":
        out = img * (1.0 - amount + amount * 2.0 * g)
    else:  # overlay
        lo = 2.0 * img * g
        hi = 1.0 - 2.0 * (1.0 - img) * (1.0 - g)
        ov = np.where(img < 0.5, lo, hi)
        out = img * (1 - amount) + ov * amount
    return np.clip(out, 0.0, 1.0)
