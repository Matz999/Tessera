"""Family 8: Neon goo — metaball blobs of glowing slime on a dark substrate.

Sum-of-Gaussians metaball field -> thresholded blobs with domed tops and a
meniscus rim; wet glossy specular + an emissive interior that drives bloom.
The blob fill is fed to render_material as the emission source, so the glow
follows the goo. Seamless: metaball distances wrap toroidally.
"""
import numpy as np

from ..core.noise import fbm, gaussian_blur
from ..core.util import norm01, rng_for, smoothstep
from .common import render_material, sample_common

FAMILY = "goo"


DIALS = {
    "blobs": {"choices": [6, 10, 16, 24, 36, 50]},
    "r_min": {"lo": 0.02, "hi": 0.10, "step": 0.005},
    "r_max": {"lo": 0.05, "hi": 0.25, "step": 0.005},
    "thresh": {"lo": 0.2, "hi": 0.7, "step": 0.01},
    "band": {"lo": 0.02, "hi": 0.20, "step": 0.005},
    "dome": {"lo": 0.2, "hi": 1.2, "step": 0.02},
    "rim": {"lo": 0.0, "hi": 0.8, "step": 0.01},
    "substrate": {"lo": 0.0, "hi": 0.40, "step": 0.01},
}


def sample_params(rng) -> dict:
    p = sample_common(rng)
    u = rng.uniform
    p.update({
        "blobs": int(rng.choice([10, 16, 24, 36, 50])),
        "r_min": round(float(u(0.03, 0.06)), 3),
        "r_max": round(float(u(0.07, 0.14)), 3),
        "thresh": round(float(u(0.35, 0.6)), 3),      # metaball surface level
        "band": round(float(u(0.03, 0.12)), 3),       # blob edge softness
        "dome": round(float(u(0.45, 0.9)), 3),        # bubble bulge
        "rim": round(float(u(0.15, 0.5)), 3),         # meniscus ring height
        "substrate": round(float(u(0.05, 0.22)), 3),  # background relief
        "material": str(rng.choice(["glass", "ceramic"])),
        "ramp": str(rng.choice(["biolum", "iridescent", "amethyst", "emerald",
                                 "verdigris", "oxblood"])),
        # goo glows by default (overrides sample_common's mostly-off emission).
        # glow is driven by the smooth metaball field -> hottest at blob centers,
        # so keep the master modest and let bloom carry the intensity.
        "emission": round(float(u(0.4, 1.0)), 3),
        "emission_gamma": round(float(u(1.1, 2.4)), 3),  # concentrate toward cores
        "emission_white": round(float(u(0.0, 0.3)), 3),
        "bloom": round(float(u(0.18, 0.4)), 3),
        "bloom_emissive": round(float(u(1.5, 3.0)), 3),
    })
    p["emission_thresh"] = p["thresh"]   # glow region == blob region
    p["emission_sharp"] = p["band"]
    return p


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    n = params["blobs"]
    centers = rng.uniform(0, 1, (n, 2))
    radii = rng.uniform(params["r_min"], params["r_max"], n)

    xs, ys = np.meshgrid(np.arange(size) / size, np.arange(size) / size)
    field = np.zeros((size, size))
    for i in range(n):                       # toroidal metaball accumulation
        dx = xs - centers[i, 0]; dx -= np.round(dx)
        dy = ys - centers[i, 1]; dy -= np.round(dy)
        field += np.exp(-(dx * dx + dy * dy) / (2.0 * radii[i] ** 2))
    field = norm01(field)

    thresh, band = params["thresh"], params["band"]
    fill = smoothstep(thresh - band, thresh + band, field)  # blob mask [0,1]
    dome = np.sqrt(np.clip(fill, 0.0, 1.0))                  # rounded bubble top
    rim = np.clip(fill * (1.0 - fill) * 4.0, 0.0, 1.0)       # meniscus ring

    substrate = (fbm(size, rng, octaves=5, freq=6) - 0.5) * params["substrate"]
    height = norm01(0.15 + substrate * 0.5 + dome * params["dome"]
                    + rim * params["rim"])
    height = gaussian_blur(height, size * 0.001)

    # keep blob body in the saturated mid-ramp (not the pale top) so the added
    # emission reads as colored glow rather than blowing out to white
    tone = norm01(fill * 0.5 + dome * 0.12 + 0.08
                  + (fbm(size, rng, octaves=4, freq=8) - 0.5) * 0.05)

    spec_mask = fill * 0.85 + 0.15 + rim * 0.5              # wet blobs, matte floor
    emit = np.clip(field * fill, 0.0, 1.0)                  # gradient, gated to blobs
    return render_material(height, tone, params, rng, gray, spec_mask=spec_mask,
                           ao_radii=(2, 6, 16), emit_source=emit)
