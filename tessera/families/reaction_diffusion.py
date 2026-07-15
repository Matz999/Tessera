"""Family 6: Gray-Scott reaction-diffusion — Turing patterns.

Spots, coral, maze, fingerprint, worms. Simulated on a torus (seamless),
optionally symmetry-folded, then lit as organic relief.
"""
import numpy as np
from PIL import Image

from ..core.noise import gaussian_blur
from ..core.symmetry import kaleido
from ..core.util import norm01, rng_for
from .common import render_material, sample_common

FAMILY = "reaction_diffusion"

# (feed, kill) atlas — named regimes of the Gray-Scott parameter space
REGIMES = {
    "mitosis":     (0.0367, 0.0649),
    "coral":       (0.0545, 0.0620),
    "maze":        (0.0290, 0.0570),
    "worms":       (0.0460, 0.0630),
    "solitons":    (0.0300, 0.0620),
    "fingerprint": (0.0370, 0.0600),
    "holes":       (0.0390, 0.0580),
    "chaos":       (0.0260, 0.0510),
}


DIALS = {
    "regime": {"choices": list(REGIMES.keys())},
    "steps": {"choices": [4000, 6000, 9000]},
    "seed_style": {"choices": ["spots", "rects", "ring"]},
    "fold": {"choices": [0, 4, 6, 8]},
}


def sample_params(rng) -> dict:
    p = sample_common(rng)
    p.update({
        "regime": str(rng.choice(list(REGIMES.keys()))),
        "steps": int(rng.choice([4000, 6000, 9000])),
        "sim_size": 192,
        "seed_style": str(rng.choice(["spots", "rects", "ring"])),
        "fold": int(rng.choice([0, 0, 0, 4, 6, 8])),
        "material": str(rng.choice(["stone", "ceramic", "bronze"])),
        "ramp": str(rng.choice(["biolum", "verdigris", "emerald", "oxblood",
                                "amethyst", "zellige", "iridescent", "bone"])),
    })
    return p


def _laplacian(a):
    return (-a
            + 0.2 * (np.roll(a, 1, 0) + np.roll(a, -1, 0)
                     + np.roll(a, 1, 1) + np.roll(a, -1, 1))
            + 0.05 * (np.roll(np.roll(a, 1, 0), 1, 1) + np.roll(np.roll(a, 1, 0), -1, 1)
                      + np.roll(np.roll(a, -1, 0), 1, 1) + np.roll(np.roll(a, -1, 0), -1, 1)))


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    g = params["sim_size"]
    F, k = REGIMES[params["regime"]]
    Du, Dv, dt = 0.16, 0.08, 1.0

    U = np.ones((g, g))
    V = np.zeros((g, g))
    style = params["seed_style"]
    if style == "spots":
        for _ in range(int(rng.integers(6, 20))):
            cx, cy = rng.integers(0, g, 2)
            rr = int(rng.integers(2, 6))
            ys, xs = np.ogrid[:g, :g]
            m = ((xs - cx) % g) ** 2 + ((ys - cy) % g) ** 2 < rr * rr
            V[m] = 1.0
            U[m] = 0.5
    elif style == "rects":
        for _ in range(int(rng.integers(3, 9))):
            x0, y0 = rng.integers(0, g - 12, 2)
            w, h = rng.integers(4, 14, 2)
            V[y0:y0 + h, x0:x0 + w] = 1.0
            U[y0:y0 + h, x0:x0 + w] = 0.5
    else:  # ring
        ys, xs = np.ogrid[:g, :g]
        r = np.sqrt((xs - g / 2) ** 2 + (ys - g / 2) ** 2)
        m = np.abs(r - g * 0.25) < 3
        V[m] = 1.0
        U[m] = 0.5
    V += rng.uniform(0, 0.02, (g, g))

    for _ in range(params["steps"]):
        uvv = U * V * V
        U += dt * (Du * _laplacian(U) * 4.0 - uvv + F * (1 - U))
        V += dt * (Dv * _laplacian(V) * 4.0 + uvv - (F + k) * V)

    field = norm01(V)
    img = Image.fromarray((field * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    field = np.asarray(img, dtype=np.float64) / 255.0

    if params["fold"] > 0:
        field = kaleido(field, params["fold"], mirror=True)

    height = norm01(gaussian_blur(field, size * 0.002))
    return render_material(height, height, params, rng, gray, ao_radii=(2, 6, 16))
