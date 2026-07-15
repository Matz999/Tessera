"""Emissive channel: self-lit glow added after shading, feeds the bloom halo.

A tile authors (or derives) a *source* field saying where light comes from;
this maps it through a threshold + color + shaping into an HxWx3 emission map.
The map is added post-lighting in `relief.shade` (unaffected by shadow/AO,
like real emitters) and drives the bloom in `finish`.
"""
import numpy as np

from .noise import fbm
from .util import smoothstep

# name -> emission color (r,g,b). Bright/hot; bloom amplifies from here.
EMISSION_COLORS = {
    "neon_cyan":    (0.25, 0.95, 1.00),
    "neon_green":   (0.35, 1.00, 0.45),
    "neon_pink":    (1.00, 0.20, 0.65),
    "neon_magenta": (0.95, 0.15, 1.00),
    "acid":         (0.75, 1.00, 0.15),
    "lava":         (1.00, 0.45, 0.10),
    "ember":        (1.00, 0.25, 0.05),
    "plasma":       (0.55, 0.35, 1.00),
    "sodium":       (1.00, 0.72, 0.20),
    "ice":          (0.55, 0.80, 1.00),
    "blood":        (1.00, 0.08, 0.12),
    "toxic":        (0.55, 1.00, 0.55),
    "white_hot":    (1.00, 0.97, 0.90),
}

# which field selects where the glow lives
EMISSION_SOURCES = ["tone", "height", "crevice", "ridge", "edge", "invert_tone"]


def select_source(name: str, height, tone, curv) -> np.ndarray:
    """Pick the driving field from the tile's geometry (curv = signed curvature)."""
    if name == "height":
        return height
    if name == "invert_tone":
        return 1.0 - tone
    if name == "crevice":       # glow pools in cracks/recesses (lava, circuit gaps)
        return np.clip(-curv, 0.0, 1.0)
    if name == "ridge":         # glowing raised edges
        return np.clip(curv, 0.0, 1.0)
    if name == "edge":          # both-sided edges (wire outlines)
        return np.abs(curv)
    return tone


def emission_rgb(source: np.ndarray, params: dict, rng, size: int) -> np.ndarray:
    """source HxW [0,1] -> HxWx3 emission (pre-master-strength). All dials read
    from params with neutral defaults."""
    thresh = params.get("emission_thresh", 0.6)
    sharp = max(params.get("emission_sharp", 0.12), 1e-3)
    m = smoothstep(thresh - sharp, thresh + sharp, source)
    m = np.power(m, params.get("emission_gamma", 1.0))

    flick = params.get("emission_flicker", 0.0)
    if flick > 0:
        f = fbm(size, rng, octaves=4, freq=int(params.get("emission_flicker_freq", 8)))
        m = m * (1.0 - flick * (1.0 - f))

    col = np.asarray(EMISSION_COLORS.get(params.get("emission_color", "neon_cyan"),
                                         (0.25, 0.95, 1.0)))
    out = m[..., None] * col

    white = params.get("emission_white", 0.35)
    if white > 0:  # hottest cores blow out toward white
        hi = min(thresh + sharp + 0.25, 1.0)
        core = smoothstep(thresh + sharp, hi, source) * m
        out = out + (white * core)[..., None]
    return out
