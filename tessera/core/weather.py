"""Weathering pass: corrosion, flaking paint, drip stains, lichen.

Any family opts in through `weather_*` dials. Effects composite over the
albedo (and adjust spec + height) *after* the base material is built, driven
by the tile's own geometry — corrosion grows from the cavity map, so it pools
in recesses and reads as real decay rather than a pasted-on texture.

`apply_weather(albedo, height, curv, params, rng, size, spec_mask)` returns
`(albedo, spec_mask, height_delta)`; all effects are no-ops at strength 0.
"""
import numpy as np

from .noise import fbm, gaussian_blur
from .palette import apply_ramp
from .util import norm01, smoothstep

# spatially-varied corrosion colors (t in [0,1] -> rgb)
RUST_RAMP = [(0.0, (0.20, 0.09, 0.04)), (0.45, (0.44, 0.19, 0.07)),
             (0.75, (0.63, 0.32, 0.12)), (1.0, (0.76, 0.53, 0.30))]
VERDIGRIS_RAMP = [(0.0, (0.09, 0.16, 0.14)), (0.5, (0.20, 0.44, 0.35)),
                  (0.8, (0.36, 0.62, 0.50)), (1.0, (0.58, 0.78, 0.66))]
LICHEN_RAMP = [(0.0, (0.34, 0.38, 0.28)), (0.6, (0.55, 0.60, 0.46)),
               (1.0, (0.72, 0.76, 0.62))]
CHIP_SUBSTRATE = (0.30, 0.26, 0.21)   # primer/metal under flaked paint
DRIP_COLOR = (0.07, 0.055, 0.045)     # soot / water streak


def _patch_mask(size, rng, freq, coverage, gate=None, bias=0.6, sharp=0.14,
                gamma=1.0):
    """fBm thresholded into patches. coverage 0..1 -> fraction covered.
    gate: optional field the patches prefer (e.g. cavity); bias how strongly."""
    n = fbm(size, rng, octaves=5, freq=int(max(freq, 1)))
    if gate is not None and bias > 0:
        n = norm01(n * ((1.0 - bias) + bias * gate))
    thr = 1.0 - np.clip(coverage, 0.0, 1.0)
    return np.power(smoothstep(thr - sharp, thr + sharp, n), gamma)


def _drips(size, rng, freq, length):
    """Downward streak field (tileable): bright fBm seeds bleed down with decay."""
    seed = np.clip(fbm(size, rng, octaves=4, freq=int(max(freq, 1))) - 0.62, 0, 1) * 3.0
    decay = np.exp(-1.0 / max(length * size * 0.28, 1.0))  # longer runs
    s = seed.copy()
    for _ in range(2):                      # two passes so the streaks wrap
        for i in range(1, size):
            s[i] = np.maximum(s[i], s[i - 1] * decay)
        s[0] = np.maximum(s[0], s[-1] * decay)
    return norm01(s)


def apply_weather(albedo, height, curv, params, rng, size, spec_mask):
    cavity = np.clip(-curv, 0.0, 1.0)       # thin crease lines
    low = norm01(gaussian_blur(height, size * 0.02))  # broad height, lowpassed
    recess = 1.0 - low                      # broad low-lying basins
    corrode = np.clip(0.6 * recess + 0.6 * cavity, 0.0, 1.0)  # where corrosion pools
    exposed = low                           # raised faces (paint flakes here)
    out = albedo.copy()
    sm = spec_mask.copy() if spec_mask is not None else np.ones((size, size))
    hdelta = np.zeros((size, size))
    bias = params.get("weather_cavity_bias", 0.6)

    def blend(mask, color):
        m = mask[..., None]
        return out * (1.0 - m) + np.asarray(color) * m

    rust = params.get("weather_rust", 0.0)
    if rust > 0:
        m = _patch_mask(size, rng, params.get("weather_rust_freq", 8), rust,
                        gate=corrode, bias=bias, gamma=1.1)
        col = apply_ramp(fbm(size, rng, octaves=4,
                             freq=int(params.get("weather_rust_freq", 8)) * 2),
                         RUST_RAMP)
        out = out * (1.0 - m[..., None]) + col * m[..., None]
        sm = sm * (1.0 - 0.85 * m)          # rust is matte
        hdelta += m * 0.03 * (fbm(size, rng, octaves=3, freq=24) - 0.5)  # pitting

    verd = params.get("weather_verdigris", 0.0)
    if verd > 0:
        m = _patch_mask(size, rng, params.get("weather_verd_freq", 7), verd,
                        gate=corrode, bias=bias, gamma=1.0)
        col = apply_ramp(fbm(size, rng, octaves=4,
                             freq=int(params.get("weather_verd_freq", 7)) * 2),
                         VERDIGRIS_RAMP)
        out = out * (1.0 - m[..., None]) + col * m[..., None]
        sm = sm * (1.0 - 0.7 * m)

    chips = params.get("weather_chips", 0.0)
    if chips > 0:                            # flaking paint -> exposed substrate
        m = _patch_mask(size, rng, params.get("weather_chip_freq", 10), chips,
                        gate=exposed, bias=bias * 0.7, sharp=0.04, gamma=1.0)
        edge = np.clip(m - gaussian_blur(m, size * 0.005), 0.0, 1.0)  # torn rim
        out = blend(m, CHIP_SUBSTRATE)
        out = out + (edge * 0.4)[..., None]   # bright torn edge (fresh break)
        hdelta -= m * params.get("weather_chip_depth", 0.05)  # recessed
        sm = sm * (1.0 - 0.5 * m) + edge * 0.4

    drips = params.get("weather_drips", 0.0)
    if drips > 0:
        d = _drips(size, rng, params.get("weather_drip_freq", 6),
                   params.get("weather_drip_len", 0.5)) * drips
        a = np.clip(0.9 * d, 0.0, 0.9)[..., None]  # streaks darken toward soot
        out = out * (1.0 - a) + np.asarray(DRIP_COLOR) * a
        sm = sm * (1.0 - 0.5 * d)

    lichen = params.get("weather_lichen", 0.0)
    if lichen > 0:
        m = _patch_mask(size, rng, params.get("weather_lichen_freq", 9), lichen,
                        gate=np.clip(curv, 0, 1), bias=bias * 0.5,
                        sharp=0.08, gamma=1.4)
        stipple = 0.6 + 0.4 * fbm(size, rng, octaves=4, freq=40)
        col = apply_ramp(fbm(size, rng, octaves=4, freq=16), LICHEN_RAMP) * stipple[..., None]
        out = out * (1.0 - m[..., None]) + col * m[..., None]
        sm = sm * (1.0 - 0.9 * m)            # lichen is dead matte
        hdelta += m * 0.02                   # slightly raised crust

    return np.clip(out, 0.0, 1.0), sm, hdelta
