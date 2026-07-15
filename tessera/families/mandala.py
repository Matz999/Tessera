"""Family 8: Mandala / rose window — concentric rings of radial motifs.

Built directly in polar space: each ring is a radial profile x petal wave.
Carved-stone cathedral tracery in grey; stained glass in color.
"""
import numpy as np

from ..core.noise import fbm, gaussian_blur
from ..core.util import norm01, rng_for, smoothstep
from .common import render_material, sample_common

FAMILY = "mandala"


# (the per-ring list is authored procedurally — tune it via raw overrides)
DIALS = {
    "boss": {"lo": 0.0, "hi": 0.30, "step": 0.005},
    "terrace": {"choices": [0, 4, 6]},
}


def sample_params(rng) -> dict:
    p = sample_common(rng)
    base_n = int(rng.choice([6, 8, 10, 12, 16]))
    rings = []
    n_rings = int(rng.integers(3, 6))
    for i in range(n_rings):
        rings.append({
            "r": round(0.16 + 0.68 * (i + rng.uniform(0.2, 0.8)) / n_rings, 3),
            "w": round(float(rng.uniform(0.035, 0.10)), 3),
            "petals": int(base_n * int(rng.choice([1, 1, 2]))),
            "sharp": round(float(rng.uniform(1.0, 4.0)), 2),
            "phase": round(float(rng.uniform(0, 1)), 3),
        })
    p.update({
        "rings": rings,
        "boss": round(float(rng.uniform(0.08, 0.16)), 3),
        "terrace": int(rng.choice([0, 4, 6])),
        "material": str(rng.choice(["stone", "bronze", "glass", "ceramic"])),
        "ramp": str(rng.choice(["stainedglass", "lapis_gold", "gold_indigo", "amethyst",
                                "zellige", "oxblood", "iridescent"])),
    })
    return p


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    v = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    X, Y = np.meshgrid(v, v)
    r = np.sqrt(X * X + Y * Y)
    th = np.arctan2(Y, X)

    height = np.zeros((size, size))
    tone = np.zeros((size, size))
    for ring in params["rings"]:
        prof = np.exp(-((r - ring["r"]) ** 2) / (2 * ring["w"] ** 2))
        wave = (0.5 + 0.5 * np.cos(th * ring["petals"] + ring["phase"] * 2 * np.pi))
        wave = smoothstep(0.25, 0.75, wave ** ring["sharp"])  # flat-topped petals
        height += prof * (0.35 + 0.65 * wave)
        tone += prof * (0.3 + 0.7 * wave)

    # carve detail so petals read as chiselled, not airbrushed
    detail = fbm(size, rng, octaves=5, freq=10)
    height *= 0.88 + 0.24 * detail

    # center boss + thin ring mullions between motif rings
    height += np.exp(-(r ** 2) / (2 * params["boss"] ** 2)) * 0.9
    for ring in params["rings"]:
        edge = np.exp(-((r - (ring["r"] + ring["w"] * 1.6)) ** 2) / (2 * 0.006 ** 2))
        height += edge * 0.25

    height = norm01(height)
    if params["terrace"] > 0:
        t = height * params["terrace"]
        f = t - np.floor(t)
        height = (np.floor(t) + smoothstep(0.35, 0.65, f)) / params["terrace"]

    # frame: darken and flatten beyond the outer ring
    rim = smoothstep(0.97, 0.90, r)
    ground = 0.22 + (fbm(size, rng, octaves=4, freq=7) - 0.5) * 0.10
    height = height * rim + ground * (1 - rim)
    height = norm01(gaussian_blur(height, size * 0.002))

    tone = norm01(tone * rim + (fbm(size, rng, octaves=4, freq=5) - 0.5) * 0.12)
    return render_material(height, tone, params, rng, gray, ao_radii=(3, 8, 20))
