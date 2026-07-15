"""Family 7: Voronoi / Delaunay mosaic — cracked glaze, stained glass, shattered stone.

Bevelled cells + recessed dark grout + per-cell color jitter + glaze specular.
Seamless: sites are wrapped toroidally.
"""
import numpy as np

from ..core.noise import fbm, gaussian_blur
from ..core.util import norm01, rng_for, smoothstep
from .common import render_material, sample_common

FAMILY = "voronoi"


DIALS = {
    "sites": {"choices": [24, 40, 70, 110, 170]},
    "lloyd": {"lo": 0, "hi": 4, "step": 1},
    "grout": {"lo": 0.005, "hi": 0.06, "step": 0.001},
    "bevel": {"lo": 0.01, "hi": 0.10, "step": 0.001},
    "jitter": {"lo": 0.0, "hi": 0.7, "step": 0.01},
    "height_jitter": {"lo": 0.0, "hi": 0.5, "step": 0.01},
}


def sample_params(rng) -> dict:
    p = sample_common(rng)
    p.update({
        "sites": int(rng.choice([24, 40, 70, 110, 170])),
        "lloyd": int(rng.integers(0, 3)),
        "grout": round(float(rng.uniform(0.010, 0.035)), 4),
        "bevel": round(float(rng.uniform(0.015, 0.06)), 4),
        "jitter": round(float(rng.uniform(0.15, 0.5)), 3),
        "height_jitter": round(float(rng.uniform(0.0, 0.25)), 3),
        "material": str(rng.choice(["ceramic", "glass", "stone"])),
        "ramp": str(rng.choice(["zellige", "celadon", "terracotta", "stainedglass",
                                "lapis_gold", "verdigris", "oxblood", "emerald"])),
    })
    return p


def _voronoi_fields(size, sites_xy):
    """Toroidal F1, F2 distances + nearest site id (chunked over rows)."""
    n = len(sites_xy)
    ys, xs = np.meshgrid(np.arange(size) / size, np.arange(size) / size, indexing="ij")
    f1 = np.full((size, size), 9.0)
    f2 = np.full((size, size), 9.0)
    ids = np.zeros((size, size), dtype=np.int64)
    chunk = max(1, 4_000_000 // (size * size))
    for start in range(0, n, chunk):
        pts = sites_xy[start:start + chunk]
        dx = np.abs(xs[None] - pts[:, 0, None, None])
        dy = np.abs(ys[None] - pts[:, 1, None, None])
        dx = np.minimum(dx, 1.0 - dx)
        dy = np.minimum(dy, 1.0 - dy)
        d = np.sqrt(dx * dx + dy * dy)
        for i in range(d.shape[0]):
            di = d[i]
            closer = di < f1
            f2 = np.where(closer, f1, np.minimum(f2, di))
            ids = np.where(closer, start + i, ids)
            f1 = np.where(closer, di, f1)
    return f1, f2, ids


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    n = params["sites"]
    sites = rng.uniform(0, 1, (n, 2))

    for _ in range(params["lloyd"]):  # cheap Lloyd relaxation on a coarse grid
        g = 128
        _, _, ids = _voronoi_fields(g, sites)
        ys, xs = np.meshgrid(np.arange(g) / g, np.arange(g) / g, indexing="ij")
        for i in range(n):
            m = ids == i
            if m.any():
                # centroid w.r.t. site to respect wrap
                dx = xs[m] - sites[i, 0]
                dy = ys[m] - sites[i, 1]
                dx -= np.round(dx)
                dy -= np.round(dy)
                sites[i] = (sites[i] + np.array([dx.mean(), dy.mean()])) % 1.0

    f1, f2, ids = _voronoi_fields(size, sites)
    edge = f2 - f1  # 0 at cell borders

    grout, bevel = params["grout"], params["bevel"]
    plateau = smoothstep(grout, grout + bevel, edge)  # rounded bevel up from grout
    cell_h = rng.uniform(-1, 1, n)[ids] * params["height_jitter"]
    height = norm01(0.18 + plateau * (0.72 + cell_h * 0.5)
                    + (fbm(size, rng, octaves=4, freq=12) - 0.5) * 0.06)
    height = gaussian_blur(height, size * 0.0012)

    # per-cell tone jitter through the ramp; grout stays dark
    cell_tone = np.clip(0.55 + rng.uniform(-1, 1, n) * params["jitter"], 0.03, 0.97)
    tone = cell_tone[ids] * plateau + 0.04 * (1 - plateau)
    tone = norm01(tone + (fbm(size, rng, octaves=4, freq=10) - 0.5) * 0.10)

    spec_mask = plateau * 0.9 + 0.1  # glaze on cells, matte grout
    return render_material(height, tone, params, rng, gray, spec_mask=spec_mask,
                           ao_radii=(2, 7, 18))
