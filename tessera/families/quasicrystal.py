"""Family 2: Quasicrystal interference — N plane waves at equal angles.

Shimmering n-fold moiré; evolution of the V1 grating art. Rendered as
engraved/embossed relief so it reads as chased metal or carved lacquer.
"""
import numpy as np

from ..core.noise import fbm, gaussian_blur
from ..core.util import norm01, rng_for
from .common import render_material, sample_common

FAMILY = "quasicrystal"


DIALS = {
    "waves": {"choices": [5, 7, 9, 11]},
    "freq": {"lo": 3.0, "hi": 30.0, "step": 0.5},
    "mode": {"choices": ["cos", "abs", "ridge"]},
    "gamma": {"lo": 0.4, "hi": 3.0, "step": 0.05},
    "terrace": {"choices": [0, 4, 6]},
    "perturb": {"lo": 0.0, "hi": 0.40, "step": 0.01},
}


def sample_params(rng) -> dict:
    p = sample_common(rng)
    p.update({
        "waves": int(rng.choice([5, 7, 9, 11])),
        "freq": round(float(rng.uniform(6.0, 26.0)), 2),
        "mode": str(rng.choice(["cos", "abs", "ridge"])),
        "gamma": round(float(rng.uniform(0.7, 2.2)), 2),
        "terrace": int(rng.choice([0, 0, 4, 6])),
        "perturb": round(float(rng.uniform(0.0, 0.25)), 3),
        "material": str(rng.choice(["bronze", "stone", "ceramic", "glass"])),
        "ramp": str(rng.choice(["gold_indigo", "iridescent", "biolum", "amethyst",
                                "lapis_gold", "obsidian", "verdigris"])),
    })
    return p


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    v = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    X, Y = np.meshgrid(v, v)

    n = params["waves"]
    freq = params["freq"] * np.pi
    field = np.zeros((size, size))
    rot0 = float(rng.uniform(0, np.pi))
    for i in range(n):
        a = rot0 + np.pi * i / n
        phase = float(rng.uniform(0, 2 * np.pi))
        field += np.cos((X * np.cos(a) + Y * np.sin(a)) * freq + phase)
    field /= n

    if params["mode"] == "abs":
        field = np.abs(field)
    elif params["mode"] == "ridge":
        field = 1.0 - np.abs(field)
    else:
        field = field * 0.5 + 0.5
    field = norm01(field) ** params["gamma"]

    if params["perturb"] > 0:  # fBm perturbation so it isn't mathematically sterile
        field = norm01(field + (fbm(size, rng, octaves=5, freq=6) - 0.5) * params["perturb"])

    if params["terrace"] > 0:
        t = field * params["terrace"]
        f = t - np.floor(t)
        s = np.clip((f - 0.35) / 0.3, 0, 1)
        field = (np.floor(t) + s * s * (3 - 2 * s)) / params["terrace"]

    height = gaussian_blur(field, size * 0.0015)
    return render_material(norm01(height), norm01(field), params, rng, gray,
                           ao_radii=(2, 6, 14))
