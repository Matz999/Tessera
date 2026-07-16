"""Family 12: Stamp — emboss a font glyph or an uploaded image into the tile.

The stamp engine (`core.stamp`) turns a glyph (emoji / dingbat / icon / letter)
or an uploaded picture into seamless ink + luminance fields, arranged single /
grid / brick / scattered. This family reads those fields into the shared
material stack: the shape is embossed (raised) or engraved (recessed) into a
textured ground, optionally tinted, made glossy, and lit — so you get carved
stone reliefs, stamped metal plaques, glowing emblems, printed-circuit-style
repeats, etc. Pin `stamp_glyph` / `stamp_image` (via the Stamp strip in the UI
or raw overrides) to choose what gets stamped; pin `emission` to make it glow.
"""
import numpy as np

from ..core.noise import fbm, gaussian_blur
from ..core.stamp import build_fields
from ..core.util import norm01, rng_for
from .common import render_material, sample_common

FAMILY = "stamp"

# fun defaults for a fresh sweep (animals / faces / nature / symbols)
_GLYPHS = ["🐱", "🦊", "🐻", "🦉", "🐙", "🦋", "🌳", "🍄", "👁", "💀",
           "🧠", "☀", "🔥", "🌊", "⚡", "🌛", "☯", "☠", "✶", "♞"]


DIALS = {
    "stamp_arrange":      {"choices": ["single", "grid", "brick", "scatter"]},
    "stamp_scale":        {"lo": 0.1, "hi": 1.3, "step": 0.02},
    "stamp_grid":         {"lo": 1, "hi": 8, "step": 1},
    "stamp_count":        {"lo": 1, "hi": 24, "step": 1},
    "stamp_rotate":       {"lo": 0.0, "hi": 6.283, "step": 0.05},
    "stamp_rot_jitter":   {"lo": 0.0, "hi": 3.14, "step": 0.05},
    "stamp_scale_jitter": {"lo": 0.0, "hi": 0.6, "step": 0.02},
    "stamp_pos_jitter":   {"lo": 0.0, "hi": 0.5, "step": 0.02},
    "stamp_emboss":       {"lo": -1.0, "hi": 1.0, "step": 0.02},
    "stamp_bevel":        {"lo": 0.0, "hi": 1.0, "step": 0.02},
    "stamp_lum_detail":   {"lo": 0.0, "hi": 1.0, "step": 0.02},
    "stamp_threshold":    {"lo": 0.0, "hi": 0.9, "step": 0.02},
    "stamp_edge_only":    {"lo": 0.0, "hi": 1.0, "step": 0.02},
    "stamp_tone":         {"lo": -0.6, "hi": 0.6, "step": 0.02},
    "stamp_spec":         {"lo": -0.5, "hi": 0.8, "step": 0.02},
    "bg_tone":            {"lo": 0.0, "hi": 0.8, "step": 0.02},
    "tone_relief":        {"lo": 0.0, "hi": 0.8, "step": 0.02},
    "bg_tex":             {"lo": 0.0, "hi": 0.4, "step": 0.01},
    "bg_freq":            {"lo": 2, "hi": 12, "step": 1},
    "bg_spec":            {"lo": 0.0, "hi": 0.8, "step": 0.02},
    "stamp_flip_x":       {"bool": True},
    "stamp_flip_y":       {"bool": True},
    "stamp_invert":       {"bool": True},
}


def sample_params(rng) -> dict:
    p = sample_common(rng)
    u = rng.uniform
    arrange = str(rng.choice(["single", "single", "single", "grid", "brick", "scatter"]))
    p.update({
        # --- source (glyph by default; pin stamp_source='image' + stamp_image) ---
        "stamp_source": "glyph",
        "stamp_glyph": str(rng.choice(_GLYPHS)),
        "stamp_font": "emoji",
        "stamp_image": "",
        "stamp_offset": [0.0, 0.0],
        # --- arrangement ---
        "stamp_arrange": arrange,
        "stamp_scale": round(float(u(0.55, 0.78) if arrange == "single" else u(0.42, 0.7)), 3),
        "stamp_grid": int(rng.choice([2, 3, 4])),
        "stamp_count": int(rng.choice([3, 4, 5])),
        "stamp_rotate": round(float(rng.choice([0.0, 0.0, u(0.0, 6.283)])), 3),
        "stamp_rot_jitter": round(float(u(0.0, 0.6) if arrange == "scatter" else 0.0), 3),
        "stamp_scale_jitter": round(float(u(0.0, 0.3) if arrange == "scatter" else 0.0), 3),
        "stamp_pos_jitter": round(float(u(0.0, 0.2) if arrange == "scatter" else 0.0), 3),
        # --- shape ---
        "stamp_threshold": round(float(rng.choice([0.0, 0.0, u(0.2, 0.5)])), 3),
        "stamp_edge_only": round(float(rng.choice([0.0, 0.0, 0.0, u(0.4, 0.9)])), 3),
        "stamp_invert": False,
        "stamp_flip_x": False,
        "stamp_flip_y": False,
        # --- emboss / material coupling ---
        "stamp_emboss": round(float(rng.choice([1.0, 1.0, -1.0]) * u(0.5, 1.0)), 3),
        "stamp_bevel": round(float(u(0.1, 0.6)), 3),
        "stamp_lum_detail": round(float(u(0.2, 0.7)), 3),
        # let relief + cast shadows read the shape (carved look), not a bright
        # inlay — keep tone coupling gentle so stamps don't blow out to white
        "stamp_tone": round(float(u(-0.24, 0.08)), 3),
        "stamp_spec": round(float(u(-0.2, 0.45)), 3),
        # flat plateaus mirror easily — hold the specular down so raised stamps
        # read as carved surface, not a white highlight
        "spec_boost": round(float(u(0.4, 0.95)), 3),
        "spec_pow_boost": round(float(u(0.7, 1.2)), 3),
        # --- ground ---
        "bg_tone": round(float(u(0.32, 0.5)), 3),
        "tone_relief": round(float(u(0.1, 0.32)), 3),
        "bg_tex": round(float(u(0.04, 0.16)), 3),
        "bg_freq": int(rng.choice([3, 4, 6, 8])),
        "bg_spec": round(float(u(0.1, 0.4)), 3),
        # --- look ---
        "material": str(rng.choice(["stone", "bronze", "ceramic", "vellum"])),
        "ramp": str(rng.choice(["terracotta", "oxblood", "gold_indigo", "lapis_gold",
                                "verdigris", "celadon", "zellige", "amethyst"])),
        # ~25% glow to light up the stamped shape (outline-weighted -> neon emblem)
        "emission": round(float(rng.choice([0.0, 0.0, 0.0, u(0.5, 1.0)])), 3),
        "emission_thresh": 0.35,
        "emission_white": round(float(u(0.05, 0.2)), 3),
        # gentler weathering so the stamp stays legible
        "weather_chips": 0.0,
        "weather_rust": round(float(rng.choice([0.0, 0.0, 0.0, u(0.2, 0.45)])), 3),
        "weather_verdigris": round(float(rng.choice([0.0, 0.0, 0.0, u(0.2, 0.4)])), 3),
    })
    return p


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    ink, lum = build_fields(params, size, rng)

    # rounded emboss shoulders (bevel), with interior luminance carving detail
    bev = float(params.get("stamp_bevel", 0.3))
    shoulders = gaussian_blur(ink, size * (0.0015 + 0.03 * bev)) if bev > 0 else ink
    ld = float(params.get("stamp_lum_detail", 0.4))
    relief = shoulders * ((1.0 - ld) + ld * lum)

    # textured ground + signed emboss (raised or engraved). A faint tooth on the
    # stamp faces keeps the flat plateau from mirroring to white specular.
    bg = (fbm(size, rng, octaves=5, freq=int(params.get("bg_freq", 5))) - 0.5)
    bg *= float(params.get("bg_tex", 0.1))
    tooth = (fbm(size, rng, octaves=4, freq=max(int(size / 16), 12)) - 0.5) * 0.02
    h = 0.42 + bg + float(params.get("stamp_emboss", 0.7)) * relief + tooth * shoulders
    h = norm01(gaussian_blur(h, size * 0.0012))

    # tone reads the *material* (ground texture + interior luminance + a gentle
    # ink tint), NOT the emboss step — the shape is carried by relief lighting,
    # cast shadow and AO, so a raised plate reads as carved, not a white sticker.
    bgn = norm01(bg)
    tone = norm01(float(params.get("bg_tone", 0.35))
                  + float(params.get("tone_relief", 0.25)) * bgn
                  + float(params.get("stamp_tone", 0.0)) * ink
                  + 0.22 * ld * lum * ink)
    spec_mask = np.clip(float(params.get("bg_spec", 0.25))
                        + float(params.get("stamp_spec", 0.3)) * ink, 0.02, 1.0)

    # emissive source: glow the outline + bright interior features rather than
    # the flat fill, so a lit stamp reads as a neon emblem, not a white blob.
    gy, gx = np.gradient(shoulders)
    outline = norm01(np.sqrt(gx * gx + gy * gy))
    emit = norm01(outline + 0.5 * ink * (0.3 + 0.7 * lum))

    return render_material(h, tone, params, rng, gray, spec_mask=spec_mask,
                           ao_radii=(3, 8, 20), emit_source=emit)
