"""Family 9: Printed circuit board.

Grid-routed copper traces (Manhattan + 45 deg bends) with pads, drilled vias,
ICs (raised epoxy blocks with metal pin rows) and white silkscreen, over a
solder-mask substrate. Multi-material: builds its own albedo/spec/height and
hands them to render_material. LEDs (a subset of pads) drive the emissive glow.
Seamless: every primitive is a tileable wrapped-delta rasterizer.
"""
import numpy as np

from ..core.draw import (coords, over, rect, stamp_capsule, stamp_disk,
                         stamp_rect)
from ..core.noise import fbm, gaussian_blur
from ..core.util import rng_for
from .common import render_material, sample_common

FAMILY = "pcb"

SOLDER = {  # solder-mask colors
    "green":  (0.02, 0.20, 0.09), "red": (0.32, 0.03, 0.04),
    "blue":   (0.03, 0.11, 0.34), "black": (0.03, 0.04, 0.05),
    "purple": (0.17, 0.03, 0.26), "white": (0.80, 0.82, 0.78),
    "yellow": (0.55, 0.48, 0.05), "teal": (0.02, 0.24, 0.24),
}
FINISH = {"gold": (0.80, 0.60, 0.22), "tin": (0.74, 0.76, 0.80),
          "copper": (0.74, 0.44, 0.20)}
CHIP_EPOXY = (0.05, 0.05, 0.06)
PIN_TIN = (0.72, 0.74, 0.78)
SILK = (0.88, 0.90, 0.85)
HOLE = (0.03, 0.03, 0.04)
LED_COLORS = ["neon_green", "neon_cyan", "neon_pink", "sodium", "lava", "ice"]

_DIRS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                  [1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=np.float64)
_DIRW = np.array([3, 3, 3, 3, 1, 1, 1, 1], dtype=np.float64)
_DIRW /= _DIRW.sum()


# family-specific dials for the UI panel (material/ramp live in the shared panel)
DIALS = {
    "grid": {"lo": 6, "hi": 30, "step": 1},
    "traces": {"lo": 2, "hi": 40, "step": 1},
    "trace_w": {"lo": 0.05, "hi": 0.30, "step": 0.005},
    "walk_len": {"lo": 2, "hi": 16, "step": 1},
    "via_frac": {"lo": 0.0, "hi": 1.0, "step": 0.01},
    "chips": {"lo": 0, "hi": 8, "step": 1},
    "led_frac": {"lo": 0.0, "hi": 0.5, "step": 0.01},
    "silk_extra": {"lo": 0, "hi": 16, "step": 1},
    "soldermask": {"choices": list(SOLDER)},
    "finish": {"choices": list(FINISH)},
}


def sample_params(rng) -> dict:
    p = sample_common(rng)
    u = rng.uniform
    p.update({
        "grid": int(rng.choice([10, 12, 16, 20, 26])),
        "traces": int(rng.choice([8, 12, 18, 26])),
        "trace_w": round(float(u(0.10, 0.20)), 3),
        "walk_len": int(rng.choice([4, 6, 8, 11])),
        "via_frac": round(float(u(0.1, 0.5)), 3),
        "chips": int(rng.choice([1, 2, 3, 4])),
        "led_frac": round(float(u(0.0, 0.18)), 3),
        "silk_extra": int(rng.choice([2, 5, 9])),
        "soldermask": str(rng.choice(list(SOLDER))),
        "finish": str(rng.choice(list(FINISH))),
        "material": "ceramic",
        "spec_tint": round(float(u(0.4, 0.9)), 3),
        "fresnel": round(float(u(0.1, 0.45)), 3),
        # LEDs mostly on but gentle; source is the LED pads (emit_source)
        "emission": round(float(rng.choice([0.0, 0.7, 1.0, 1.3])), 3),
        "emission_color": str(rng.choice(LED_COLORS)),
        "emission_thresh": 0.35,
        "emission_white": round(float(u(0.2, 0.5)), 3),
        "bloom": round(float(u(0.15, 0.35)), 3),
    })
    return p


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    X, Y = coords(size)
    N = params["grid"]
    cell = size / N
    tw = cell * params["trace_w"]
    soft = max(size / 512.0, 1.0)

    metal = np.zeros((size, size))
    holes = np.zeros((size, size))
    chip_body = np.zeros((size, size))
    pins = np.zeros((size, size))
    silk = np.zeros((size, size))
    leds = np.zeros((size, size))

    def node(ix, iy):
        return ((ix + 0.5) * cell % size, (iy + 0.5) * cell % size)

    # --- routed traces + their end pads ---
    pads = []
    for _ in range(params["traces"]):
        ix, iy = rng.integers(N), rng.integers(N)
        prev = node(ix, iy); pads.append(prev)
        for _ in range(rng.integers(3, params["walk_len"] + 1)):
            d = _DIRS[rng.choice(8, p=_DIRW)]
            ix, iy = ix + d[0], iy + d[1]
            cur = node(ix, iy)
            stamp_capsule(metal, prev[0], prev[1], cur[0], cur[1], tw, soft)
            prev = cur
        pads.append(prev)

    rp = tw * 1.9
    for (cx, cy) in pads:
        stamp_disk(metal, cx, cy, rp, soft)
        r = rng.random()
        if r < params["via_frac"]:
            stamp_disk(holes, cx, cy, rp * 0.42, soft)
        elif r < params["via_frac"] + params["led_frac"]:
            stamp_disk(leds, cx, cy, rp * 0.8, soft)

    # --- ICs: raised epoxy block + metal pin rows + silkscreen outline ---
    pitch = cell * 0.55
    for _ in range(params["chips"]):
        cx, cy = node(rng.integers(N), rng.integers(N))
        hw = rng.integers(2, 5) * cell * 0.5
        hh = rng.integers(2, 5) * cell * 0.5
        stamp_rect(chip_body, cx, cy, hw, hh, soft)
        plen, pw = cell * 0.34, cell * 0.11
        sides = [("x", -1), ("x", 1)]
        if rng.random() < 0.5:                       # QFP: pins on all 4 sides
            sides += [("y", -1), ("y", 1)]
        for axis, s in sides:
            span = hh if axis == "x" else hw
            k = max(int(2 * span / pitch), 1)
            for j in range(k):
                off = -span + (j + 0.5) * (2 * span / k)
                if axis == "x":
                    px_, py_ = cx + s * (hw + plen * 0.5), cy + off
                    ph, pv = plen * 0.5, pw * 0.5
                else:
                    px_, py_ = cx + off, cy + s * (hh + plen * 0.5)
                    ph, pv = pw * 0.5, plen * 0.5
                stamp_rect(pins, px_, py_, ph, pv, soft)
        # silkscreen border + pin-1 dot
        b = cell * 0.18
        outline = np.clip(rect(X, Y, size, cx, cy, hw + b, hh + b, soft)
                          - rect(X, Y, size, cx, cy, hw + b * 0.4, hh + b * 0.4, soft),
                          0, 1)
        silk = np.maximum(silk, outline)
        stamp_disk(silk, cx - hw * 0.7, cy - hh * 0.7, cell * 0.1, soft)

    # --- extra silkscreen: component footprints + reference ticks ---
    for _ in range(params["silk_extra"]):
        cx, cy = node(rng.integers(N), rng.integers(N))
        hw = rng.integers(1, 3) * cell * 0.5
        hh = cell * 0.5
        outline = np.clip(rect(X, Y, size, cx, cy, hw, hh, soft)
                          - rect(X, Y, size, cx, cy, hw - soft * 2, hh - soft * 2, soft),
                          0, 1)
        silk = np.maximum(silk, outline * 0.8)

    metal = np.clip(metal - holes, 0, 1)
    pins = np.clip(pins - chip_body, 0, 1)            # pins emerge from under body

    # --- height ---
    subtex = (fbm(size, rng, octaves=4, freq=max(N // 2, 4)) - 0.5) * 0.015
    h = np.full((size, size), 0.12) + subtex
    h = np.maximum(h, 0.13 + metal * 0.10)
    h = np.maximum(h, 0.13 + pins * 0.14)
    h = np.maximum(h, 0.14 + chip_body * 0.58)
    h = h + silk * 0.02 - holes * 0.13
    h = gaussian_blur(np.clip(h, 0, 1), size * 0.0009)

    # --- albedo (authored, bypasses the ramp) ---
    solder_col = np.array(SOLDER.get(params["soldermask"], SOLDER["green"]))
    weave = 0.88 + 0.24 * fbm(size, rng, octaves=4, freq=N)
    alb = np.ones((size, size, 3)) * solder_col * weave[..., None]
    alb = over(alb, metal, FINISH.get(params["finish"], FINISH["gold"]))
    alb = over(alb, holes, HOLE)
    alb = over(alb, chip_body, CHIP_EPOXY)
    alb = over(alb, pins, PIN_TIN)
    alb = over(alb, silk, SILK)

    # --- spec mask: metal shiny, mask semigloss, epoxy mid, silk/holes matte ---
    sm = np.full((size, size), 0.35)
    sm = over(sm, metal, 1.0)
    sm = over(sm, chip_body, 0.5)
    sm = over(sm, pins, 1.0)
    sm = over(sm, silk, 0.15)
    sm = over(sm, holes, 0.08)

    return render_material(h, metal, params, rng, gray, spec_mask=sm,
                           ao_radii=(2, 6, 16), emit_source=leds, albedo=alb)
