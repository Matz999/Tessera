"""Family 1: Bevelled symmetric emblems — carved medallions.

Kaleidoscope a warped-noise seed into D-n symmetry, terrace into carved
plateaus, frame in a bevelled disc + border ring, light as stone/bronze/ceramic.
"""
import numpy as np

from ..core.noise import fbm, gaussian_blur, warped_fbm
from ..core.symmetry import kaleido
from ..core.util import norm01, rng_for, smoothstep
from .common import render_material, sample_common

FAMILY = "emblem"


DIALS = {
    "sym": {"choices": [4, 5, 6, 8, 10, 12]},
    "octaves": {"lo": 3, "hi": 8, "step": 1},
    "freq": {"lo": 2, "hi": 10, "step": 1},
    "warp": {"lo": 0.0, "hi": 0.8, "step": 0.01},
    "terrace": {"choices": [0, 3, 4, 5, 6, 7]},
    "rings": {"lo": 0, "hi": 6, "step": 1},
    "ring_freq": {"lo": 1.5, "hi": 9.0, "step": 0.1},
    "border": {"choices": ["ring", "double", "none"]},
}


def sample_params(rng) -> dict:
    p = sample_common(rng)
    p.update({
        "sym": int(rng.choice([4, 5, 6, 8, 10, 12])),
        "octaves": int(rng.integers(4, 7)),
        "freq": int(rng.integers(3, 8)),
        "warp": round(float(rng.uniform(0.05, 0.45)), 3),
        "terrace": int(rng.choice([0, 3, 4, 5, 6, 7])),
        "rings": int(rng.integers(0, 4)),
        "ring_freq": round(float(rng.uniform(2.5, 7.0)), 2),
        "material": str(rng.choice(["stone", "bronze", "ceramic"])),
        "ramp": str(rng.choice(["gold_indigo", "lapis_gold", "verdigris", "terracotta",
                                "oxblood", "amethyst", "zellige", "iridescent"])),
        "border": str(rng.choice(["ring", "double", "none"])),
    })
    return p


def _terrace(h, levels, softness=0.18):
    if levels <= 0:
        return h
    t = h * levels
    f = t - np.floor(t)
    step = smoothstep(0.5 - softness, 0.5 + softness, f)
    return (np.floor(t) + step) / levels


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    v = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    X, Y = np.meshgrid(v, v)
    r = np.sqrt(X * X + Y * Y)

    # seed field -> fold to D-n
    base = warped_fbm(size, rng, octaves=params["octaves"], freq=params["freq"],
                      warp=params["warp"])
    motif = kaleido(base, params["sym"], mirror=True, rot=float(rng.uniform(0, 6.28)))

    # radial ring modulation pulls the eye to center
    if params["rings"] > 0:
        ring = 0.5 + 0.5 * np.cos(r * params["ring_freq"] * 2 * np.pi)
        motif = norm01(motif * (0.72 + 0.28 * ring) + 0.10 * ring)

    motif = _terrace(norm01(motif), params["terrace"])
    motif = gaussian_blur(motif, size * 0.002)  # soften carve edges

    # framing: emblem disc + bevelled rim + border ring(s), stone ground outside
    disc_r = 0.80
    disc = smoothstep(disc_r + 0.02, disc_r - 0.03, r)
    ground = fbm(size, rng, octaves=4, freq=6) * 0.12 + 0.30

    height = ground * (1 - disc) + (0.25 + motif * 0.62) * disc

    def raised_ring(rr, width, amp):
        prof = np.exp(-((r - rr) ** 2) / (2 * (width ** 2)))
        return prof * amp

    if params["border"] in ("ring", "double"):
        height += raised_ring(disc_r + 0.075, 0.022, 0.30)
        height -= raised_ring(disc_r + 0.035, 0.014, 0.22)  # grout channel
    if params["border"] == "double":
        height += raised_ring(disc_r + 0.135, 0.014, 0.22)
    # bevelled rim of the disc itself
    height += raised_ring(disc_r, 0.018, 0.20)

    height = norm01(height)
    tone = disc * (0.22 + 0.70 * norm01(motif)) \
        + (1 - disc) * (0.42 + (ground - 0.36) * 2.2)
    spec_mask = 0.4 + 0.6 * disc  # emblem glossier than ground
    return render_material(height, tone, params, rng, gray, spec_mask=spec_mask)
