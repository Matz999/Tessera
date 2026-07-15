"""Family 3: Truchet tilings — arcs on a grid, multiscale (Smith) subdivision.

Emergent op-art mazes rendered as corded/engraved relief. Seamless.
"""
import numpy as np

from ..core.noise import fbm, gaussian_blur
from ..core.util import norm01, rng_for, smoothstep
from .common import render_material, sample_common

FAMILY = "truchet"


DIALS = {
    "grid": {"lo": 3, "hi": 16, "step": 1},
    "subdiv_p": {"lo": 0.0, "hi": 0.6, "step": 0.01},
    "line_w": {"lo": 0.05, "hi": 0.30, "step": 0.005},
    "style": {"choices": ["cord", "engraved"]},
    "double": {"bool": True},
}


def sample_params(rng) -> dict:
    p = sample_common(rng)
    p.update({
        "grid": int(rng.choice([5, 6, 8, 10, 13])),
        "subdiv_p": round(float(rng.uniform(0.0, 0.45)), 3),
        "line_w": round(float(rng.uniform(0.08, 0.22)), 3),
        "style": str(rng.choice(["cord", "engraved"])),
        "double": bool(rng.uniform() < 0.4),  # two concentric arcs per corner
        "material": str(rng.choice(["stone", "bronze", "ceramic", "vellum"])),
        "ramp": str(rng.choice(["ink_vellum", "gold_indigo", "obsidian", "celadon",
                                "lapis_gold", "terracotta", "biolum"])),
    })
    return p


def _cell_arcs(rng, x0, y0, c, depth, subdiv_p, out):
    """Collect (cx, cy, radius) arc-circles for one cell; maybe subdivide."""
    if depth < 2 and rng.uniform() < subdiv_p:
        h = c / 2
        for dx in (0, 1):
            for dy in (0, 1):
                _cell_arcs(rng, x0 + dx * h, y0 + dy * h, h, depth + 1, subdiv_p, out)
        return
    if rng.uniform() < 0.5:
        corners = ((x0, y0), (x0 + c, y0 + c))
    else:
        corners = ((x0 + c, y0), (x0, y0 + c))
    for (cx, cy) in corners:
        out.append((cx, cy, c / 2, x0, y0, c))


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    g = params["grid"]
    c = size / g
    arcs = []
    for gy in range(g):
        for gx in range(g):
            _cell_arcs(rng, gx * c, gy * c, c, 0, params["subdiv_p"], arcs)

    ys, xs = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5, indexing="ij")
    D = np.full((size, size), 1e9)
    for (cx, cy, radius, cx0, cy0, cell) in arcs:
        # clip evaluation to the cell so quarter-arcs stay quarter-arcs
        x_lo, x_hi = int(max(cx0, 0)), int(min(cx0 + cell + 1, size))
        y_lo, y_hi = int(max(cy0, 0)), int(min(cy0 + cell + 1, size))
        if x_lo >= x_hi or y_lo >= y_hi:
            continue
        px = xs[y_lo:y_hi, x_lo:x_hi]
        py = ys[y_lo:y_hi, x_lo:x_hi]
        rr = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        d = np.abs(rr - radius)
        if params["double"]:
            d = np.minimum(d, np.abs(rr - radius * 0.55))
        D[y_lo:y_hi, x_lo:x_hi] = np.minimum(D[y_lo:y_hi, x_lo:x_hi], d)

    w = params["line_w"] * c * 0.5
    cord = smoothstep(w, w * 0.45, D)  # 1 on the line, rounded shoulders

    ground = 0.35 + (fbm(size, rng, octaves=4, freq=8) - 0.5) * 0.14
    if params["style"] == "cord":
        height = ground + cord * 0.5
    else:
        height = ground - cord * 0.42
    height = norm01(gaussian_blur(height, size * 0.0015))

    tone = norm01(0.35 + cord * 0.55 + (fbm(size, rng, octaves=4, freq=6) - 0.5) * 0.12)
    return render_material(height, tone, params, rng, gray, ao_radii=(2, 6, 16))
