"""Seeded, tileable gradient noise + fBm + domain warping. Pure numpy."""
import numpy as np

from .util import bilinear_wrap, norm01


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(size: int, res: int, rng: np.random.Generator) -> np.ndarray:
    """Periodic (tileable) Perlin noise, size x size, lattice of `res` cells."""
    res = max(1, int(res))
    ang = rng.uniform(0, 2 * np.pi, (res + 1, res + 1))
    ang[-1, :] = ang[0, :]
    ang[:, -1] = ang[:, 0]
    gx, gy = np.cos(ang), np.sin(ang)

    d = size // res if size % res == 0 else None
    lin = np.arange(size) * (res / size)
    xi = lin.astype(np.int64)
    xf = lin - xi
    XI, YI = np.meshgrid(xi, xi)
    XF, YF = np.meshgrid(xf, xf)

    def dot_grad(ix, iy, dx, dy):
        return gx[iy, ix] * dx + gy[iy, ix] * dy

    n00 = dot_grad(XI, YI, XF, YF)
    n10 = dot_grad(XI + 1, YI, XF - 1, YF)
    n01 = dot_grad(XI, YI + 1, XF, YF - 1)
    n11 = dot_grad(XI + 1, YI + 1, XF - 1, YF - 1)
    u, v = _fade(XF), _fade(YF)
    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    return nx0 * (1 - v) + nx1 * v  # roughly [-0.7, 0.7]


def fbm(size: int, rng: np.random.Generator, octaves: int = 5, freq: int = 4,
        gain: float = 0.5, ridged: bool = False) -> np.ndarray:
    """Fractal Brownian motion in [0,1]. Tileable."""
    total = np.zeros((size, size))
    amp, f, amps = 1.0, freq, 0.0
    for _ in range(octaves):
        layer = perlin(size, min(f, size // 2), rng)
        if ridged:
            layer = 1.0 - np.abs(layer) * 2.0
        total += amp * layer
        amps += amp
        amp *= gain
        f *= 2
    return norm01(total / amps)


def warped_fbm(size: int, rng: np.random.Generator, octaves: int = 5, freq: int = 4,
               warp: float = 0.3, warp_freq: int = 3, ridged: bool = False) -> np.ndarray:
    """Domain-warped fBm: sample base fBm through two offset fields. Tileable."""
    base = fbm(size, rng, octaves=octaves, freq=freq, ridged=ridged)
    if warp <= 0:
        return base
    wx = fbm(size, rng, octaves=3, freq=warp_freq) - 0.5
    wy = fbm(size, rng, octaves=3, freq=warp_freq) - 0.5
    xs, ys = np.meshgrid(np.arange(size, dtype=np.float64),
                         np.arange(size, dtype=np.float64))
    return bilinear_wrap(base, xs + wx * warp * size, ys + wy * warp * size)


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """FFT gaussian blur with wrap boundary (matches seamless tiles)."""
    if sigma <= 0:
        return a
    h, w = a.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    kernel = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (fx ** 2 + fy ** 2))
    if a.ndim == 2:
        return np.real(np.fft.ifft2(np.fft.fft2(a) * kernel))
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        out[..., c] = np.real(np.fft.ifft2(np.fft.fft2(a[..., c]) * kernel))
    return out
