"""Family 10: Silicon die shot.

A packed floorplan of rectangular functional blocks — memory arrays (fine
regular gratings), logic fabric (fine speckle), bond pads / capacitors (solid
metal) — separated by oxide channels and crossed by wide top-metal buses.
Colored by a cyclic thin-film interference palette (the blue/purple/gold sheen
of a chip photomicrograph), brighter over metal. Nearly flat: the interest is
color, not relief. Seamless: block rects and buses wrap toroidally.
"""
import numpy as np

from ..core.draw import coords, over, rect, stamp_capsule
from ..core.noise import fbm, gaussian_blur
from ..core.util import norm01, rng_for
from .common import render_material, sample_common

FAMILY = "silicon"

# cyclic thin-film interference ramp: blue -> violet -> magenta -> gold ->
# green -> cyan -> (wraps to blue). Successive metal-layer thicknesses.
THINFILM = np.array([
    (0.10, 0.13, 0.46), (0.34, 0.10, 0.56), (0.60, 0.12, 0.42),
    (0.74, 0.56, 0.18), (0.34, 0.62, 0.34), (0.12, 0.55, 0.63),
    (0.10, 0.13, 0.46),
])

DIALS = {
    "grid": {"choices": [12, 16, 20, 24, 30]},
    "fill_passes": {"lo": 1, "hi": 6, "step": 1},
    "pad_frac": {"lo": 0.0, "hi": 0.5, "step": 0.02},
    "detail": {"lo": 0.5, "hi": 2.0, "step": 0.05},
    "buses": {"lo": 0, "hi": 10, "step": 1},
    "bus_w": {"lo": 0.004, "hi": 0.03, "step": 0.001},
    "film_shift": {"lo": 0.0, "hi": 1.0, "step": 0.01},
    "film_range": {"lo": 0.1, "hi": 1.0, "step": 0.02},
    "channel": {"lo": 0.0, "hi": 0.6, "step": 0.02},
}


def sample_params(rng) -> dict:
    p = sample_common(rng)
    u = rng.uniform
    p.update({
        "grid": int(rng.choice([12, 16, 20, 24, 30])),
        "fill_passes": int(rng.choice([2, 3, 4])),
        "pad_frac": round(float(u(0.05, 0.3)), 3),
        "detail": round(float(u(0.7, 1.6)), 3),
        "buses": int(rng.choice([0, 2, 4, 7])),
        "bus_w": round(float(u(0.006, 0.02)), 4),
        "film_shift": round(float(u(0.0, 1.0)), 3),
        "film_range": round(float(u(0.35, 0.9)), 3),
        "channel": round(float(u(0.15, 0.45)), 3),
        "material": "glass",
        "ramp": "iridescent",   # unused (authored albedo) but kept for metadata
        "spec_tint": round(float(u(0.3, 0.7)), 3),
        "fresnel": round(float(u(0.15, 0.5)), 3),
        "emission": 0.0,        # a die shot has no emitters (pin it to add glow)
    })
    return p


def _thinfilm(t: np.ndarray) -> np.ndarray:
    n = len(THINFILM) - 1
    x = np.clip(t, 0.0, 1.0) * n
    i = np.clip(np.floor(x).astype(int), 0, n - 1)
    f = (x - i)[..., None]
    return THINFILM[i] * (1.0 - f) + THINFILM[i + 1] * f


def _pack(N, rng, passes):
    """Greedy toroidal rectangle packing -> list of (i, j, w, h) grid blocks."""
    occ = np.zeros((N, N), dtype=bool)
    blocks = []
    order = [(i, j) for i in range(N) for j in range(N)]
    for _ in range(passes):
        rng.shuffle(order)
        for (i, j) in order:
            if occ[i, j]:
                continue
            wmax = int(rng.integers(1, max(N // 2, 2) + 1))
            hmax = int(rng.integers(1, max(N // 2, 2) + 1))
            w = 1
            while w < wmax and not occ[i, (j + w) % N]:
                w += 1
            h = 1
            while h < hmax and not occ[(i + h) % N, (j + np.arange(w)) % N].any():
                h += 1
            occ[np.ix_((i + np.arange(h)) % N, (j + np.arange(w)) % N)] = True
            blocks.append((i, j, w, h))
    return blocks


def fields(params: dict, size: int, gray: bool, rng):
    """Pre-render fields (height, tone, spec_mask, emit, albedo) for the mixer.
    albedo is the authored thin-film color; emit is None (a die has no emitters)."""
    X, Y = coords(size)
    N = params["grid"]
    cell = size / N
    soft = max(size / 512.0, 1.0)
    gap = cell * params["channel"] * 0.5 + soft
    det = params["detail"]

    metal = np.zeros((size, size))       # metal-layer coverage (bright)
    micro = np.zeros((size, size))       # fine height detail
    thick = np.full((size, size), params["film_shift"])  # thin-film thickness
    block_area = np.zeros((size, size))

    logic_tex = norm01(fbm(size, rng, octaves=3, freq=int(N * 3)))

    for (i, j, w, h) in _pack(N, rng, params["fill_passes"]):
        cx = (j + w / 2.0) * cell
        cy = (i + h / 2.0) * cell
        hw = w * cell / 2.0 - gap
        hh = h * cell / 2.0 - gap
        if hw <= soft or hh <= soft:
            continue
        m = rect(X, Y, size, cx, cy, hw, hh, soft)
        block_area = np.maximum(block_area, m)
        base_t = params["film_shift"] + (rng.random() - 0.5) * params["film_range"]
        thick = thick * (1.0 - m) + (base_t % 1.0) * m

        big = w >= 2 and h >= 2
        roll = rng.random()
        if big and roll < params["pad_frac"]:                 # bond pad / cap
            metal = over(metal, m, 1.0)
            inner = rect(X, Y, size, cx, cy, hw * 0.62, hh * 0.62, soft)
            metal = over(metal, inner, 0.35)
            micro += (m - inner) * 0.5
        elif rng.random() < 0.55:                             # memory grating
            coord = Y if rng.random() < 0.5 else X
            cyc = round(size / max(6.0 / det, 2.0))            # integer cycles/tile
            grating = 0.5 + 0.5 * np.cos(2 * np.pi * cyc * coord / size)
            metal = over(metal, m, 0.25 + 0.6 * grating)
            micro += m * (grating - 0.5) * 0.6
            thick += m * (grating - 0.5) * 0.05
        else:                                                 # logic fabric
            metal = over(metal, m, 0.25 + 0.55 * logic_tex)
            micro += m * (logic_tex - 0.5) * 0.4

    # wide top-metal buses running edge to edge through the channels
    bw = size * params["bus_w"]
    for _ in range(params["buses"]):
        if rng.random() < 0.5:
            y = rng.integers(N) * cell
            stamp_capsule(metal, 0, y, size - 1, y, bw, soft)
            stamp_capsule(micro, 0, y, size - 1, y, bw, soft)
        else:
            x = rng.integers(N) * cell
            stamp_capsule(metal, x, 0, x, size - 1, bw, soft)
            stamp_capsule(micro, x, 0, x, size - 1, bw, soft)

    channel = 1.0 - block_area
    metal = np.clip(metal, 0.0, 1.0)

    # --- height: subtle. metal/buses slightly proud, channels recessed. A
    #     faint substrate tooth keeps flat metal from mirroring to white. ---
    tooth = (fbm(size, rng, octaves=4, freq=int(N * 2)) - 0.5) * 0.02
    h = (0.4 + block_area * 0.05 + np.clip(micro, -1, 1) * 0.04 + tooth
         - channel * params["channel"] * 0.1)
    h = gaussian_blur(np.clip(h, 0, 1), size * 0.0008)

    # --- albedo: thin-film color, brighter over metal, dark oxide channels ---
    bright = (0.42 + 0.55 * metal) * (1.0 - 0.5 * channel)
    alb = np.clip(_thinfilm(thick) * bright[..., None], 0.0, 1.0)

    # --- spec: metal shiny, oxide matte ---
    sm = np.clip(0.35 + 0.65 * metal - 0.3 * channel, 0.05, 1.0)

    return h, metal, sm, None, alb


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    h, tone, sm, emit, alb = fields(params, size, gray, rng)
    return render_material(h, tone, params, rng, gray, spec_mask=sm,
                           ao_radii=(2, 5, 12), emit_source=emit, albedo=alb)
