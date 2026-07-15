"""Shared material treatment used by all families."""
import numpy as np

from ..core.emissive import (EMISSION_COLORS, EMISSION_SOURCES, emission_rgb,
                             select_source)
from ..core.finish import finish
from ..core.grain import GRAINS, apply_grain
from ..core.noise import fbm
from ..core.palette import GRAY_RAMPS, apply_ramp
from ..core.relief import curvature, normals, shade
from ..core.weather import apply_weather

# Set by every render_material call: the tile's normal map (RGB-encoded),
# harvested by `tessera.render --normals` for the app's WebGL relight shader.
LAST_NORMALS = None

# material -> shading + grain recipe (the 'photographed real thing' layer)
MATERIALS = {
    "stone":   dict(spec=0.07, spec_pow=10, relief=1.25, grain=("stone", 0.38, "multiply"),
                    ambient=0.40, diffuse=0.78),
    "bronze":  dict(spec=0.55, spec_pow=34, relief=1.1, grain=("metal", 0.30, "overlay"),
                    ambient=0.36, diffuse=0.72),
    "ceramic": dict(spec=0.52, spec_pow=64, relief=1.0, grain=("craquelure", 0.42, "multiply"),
                    ambient=0.46, diffuse=0.68),
    "vellum":  dict(spec=0.04, spec_pow=6, relief=0.8, grain=("paper", 0.45, "multiply"),
                    ambient=0.50, diffuse=0.62),
    "glass":   dict(spec=0.65, spec_pow=90, relief=0.9, grain=("stone", 0.18, "multiply"),
                    ambient=0.48, diffuse=0.66),
}

WEAR_COLOR = (0.85, 0.83, 0.78)  # exposed-edge albedo (worn-by-touch)
EMISSION_COLORS_KEYS = list(EMISSION_COLORS)
# strength dials that trigger the (non-trivial) weathering pass
_WEATHER_KEYS = ("weather_rust", "weather_verdigris", "weather_chips",
                 "weather_drips", "weather_lichen")

# UI metadata for every numeric dial: slider range (wider than the sample
# sweep in sample_common — pinning may go beyond what sampling explores).
DIALS = {
    "light_angle":     dict(lo=0.0, hi=6.283, step=0.01, group="key light"),
    "light_elev":      dict(lo=0.05, hi=0.95, step=0.01, group="key light"),
    "key_warmth":      dict(lo=-1.0, hi=1.0, step=0.01, group="key light"),
    "diffuse_boost":   dict(lo=0.0, hi=2.0, step=0.01, group="key light"),
    "ambient_boost":   dict(lo=0.0, hi=2.0, step=0.01, group="key light"),
    "fill_strength":   dict(lo=0.0, hi=1.0, step=0.01, group="fill + rim"),
    "fill_angle":      dict(lo=0.0, hi=6.283, step=0.01, group="fill + rim"),
    "fill_elev":       dict(lo=0.0, hi=0.95, step=0.01, group="fill + rim"),
    "fill_warmth":     dict(lo=-1.0, hi=1.0, step=0.01, group="fill + rim"),
    "rim_strength":    dict(lo=0.0, hi=1.0, step=0.01, group="fill + rim"),
    "shadow_strength": dict(lo=0.0, hi=1.0, step=0.01, group="shadows"),
    "shadow_soft":     dict(lo=0.0, hi=8.0, step=0.1, group="shadows"),
    "shadow_height":   dict(lo=0.0, hi=0.5, step=0.005, group="shadows"),
    "spec_tint":       dict(lo=0.0, hi=1.0, step=0.01, group="specular"),
    "fresnel":         dict(lo=0.0, hi=1.0, step=0.01, group="specular"),
    "spec_boost":      dict(lo=0.0, hi=2.5, step=0.01, group="specular"),
    "spec_pow_boost":  dict(lo=0.2, hi=3.0, step=0.01, group="specular"),
    "rough_amount":    dict(lo=0.0, hi=1.0, step=0.01, group="specular"),
    "rough_freq":      dict(lo=1, hi=20, step=1, group="specular"),
    "edge_wear":       dict(lo=0.0, hi=1.0, step=0.01, group="wear"),
    "grime":           dict(lo=0.0, hi=1.0, step=0.01, group="wear"),
    "wear_radius":     dict(lo=0.5, hi=8.0, step=0.1, group="wear"),
    "wear_spec":       dict(lo=0.0, hi=1.5, step=0.01, group="wear"),
    "emission":            dict(lo=0.0, hi=3.0, step=0.01, group="emissive"),
    "emission_thresh":     dict(lo=0.0, hi=1.0, step=0.01, group="emissive"),
    "emission_sharp":      dict(lo=0.005, hi=0.5, step=0.005, group="emissive"),
    "emission_gamma":      dict(lo=0.3, hi=3.0, step=0.05, group="emissive"),
    "emission_white":      dict(lo=0.0, hi=1.0, step=0.01, group="emissive"),
    "emission_flicker":    dict(lo=0.0, hi=1.0, step=0.01, group="emissive"),
    "emission_flicker_freq": dict(lo=1, hi=24, step=1, group="emissive"),
    "bloom_emissive":      dict(lo=0.0, hi=4.0, step=0.05, group="emissive"),
    "bloom_radius":        dict(lo=0.002, hi=0.06, step=0.001, group="emissive"),
    "weather_rust":        dict(lo=0.0, hi=1.0, step=0.01, group="weathering"),
    "weather_rust_freq":   dict(lo=2, hi=16, step=1, group="weathering"),
    "weather_verdigris":   dict(lo=0.0, hi=1.0, step=0.01, group="weathering"),
    "weather_verd_freq":   dict(lo=2, hi=16, step=1, group="weathering"),
    "weather_chips":       dict(lo=0.0, hi=1.0, step=0.01, group="weathering"),
    "weather_chip_freq":   dict(lo=3, hi=24, step=1, group="weathering"),
    "weather_chip_depth":  dict(lo=0.0, hi=0.15, step=0.005, group="weathering"),
    "weather_drips":       dict(lo=0.0, hi=1.0, step=0.01, group="weathering"),
    "weather_drip_len":    dict(lo=0.0, hi=1.0, step=0.01, group="weathering"),
    "weather_drip_freq":   dict(lo=2, hi=14, step=1, group="weathering"),
    "weather_lichen":      dict(lo=0.0, hi=1.0, step=0.01, group="weathering"),
    "weather_lichen_freq": dict(lo=3, hi=18, step=1, group="weathering"),
    "weather_cavity_bias": dict(lo=0.0, hi=1.0, step=0.01, group="weathering"),
    "relief":          dict(lo=0.2, hi=3.0, step=0.01, group="surface"),
    "ao_strength":     dict(lo=0.0, hi=1.6, step=0.01, group="surface"),
    "grain_boost":     dict(lo=0.0, hi=2.0, step=0.01, group="surface"),
    "film_grain":      dict(lo=0.0, hi=0.08, step=0.001, group="finish"),
    "vignette":        dict(lo=0.0, hi=0.6, step=0.01, group="finish"),
    "bloom":           dict(lo=0.0, hi=0.8, step=0.01, group="finish"),
    "bloom_thresh":    dict(lo=0.4, hi=1.0, step=0.01, group="finish"),
    "chroma":          dict(lo=0.0, hi=3.0, step=0.01, group="finish"),
}


def warmth_rgb(w: float):
    """Warmth dial -1 (cool blue) .. +1 (warm tungsten) -> rgb multiplier."""
    return (1.0 + 0.16 * w, 1.0 + 0.02 * w, 1.0 - 0.20 * w)


def pick_ramp(params: dict, gray: bool) -> str:
    if gray:
        return params.get("gray_ramp", "pewter")
    return params.get("ramp", "pewter")


def render_material(height: np.ndarray, tone: np.ndarray, params: dict, rng,
                    gray: bool, spec_mask=None, ao_radii=(3, 9, 24),
                    do_finish: bool = True, emit_source=None, albedo=None) -> np.ndarray:
    """height: relief field [0,1]. tone: field mapped through the ramp -> albedo.

    Every dial is read from `params` (all optional, neutral defaults), so any
    aspect of the look can be forced via generate overrides — see
    `sample_common` for the full dial list and sweep ranges.

    emit_source: optional HxW [0,1] field a family supplies to drive its glow
    (e.g. goo blob fill); when None the source is picked by `emission_source`.
    albedo: optional HxWx3 authored color, bypassing the ramp (multi-material
    families like pcb build their own); when None `tone` is ramped as usual.
    In gray mode the authored albedo is desaturated to its luminance.
    """
    mat = MATERIALS[params.get("material", "stone")]
    size = height.shape[0]
    px = size / 512.0  # pixel-unit dials are calibrated at 512; scale with res
    global LAST_NORMALS

    if albedo is None:
        # rank-equalize tone so every tile spans the full ramp (no mud-dark tiles)
        flat = tone.ravel()
        ranks = np.empty(flat.size)
        ranks[np.argsort(flat, kind="stable")] = np.linspace(0.0, 1.0, flat.size)
        tone = np.clip(0.55 * tone + 0.45 * ranks.reshape(tone.shape), 0.0, 1.0)
        albedo = apply_ramp(tone, pick_ramp(params, gray))
    else:
        albedo = np.clip(albedo, 0.0, 1.0)
        if gray:
            lum = albedo @ np.array([0.299, 0.587, 0.114])
            albedo = np.stack([lum] * 3, axis=-1)

    grain_name, grain_amt, grain_mode = mat["grain"]
    g = GRAINS[grain_name](size, rng)
    albedo = apply_grain(albedo, g, amount=grain_amt * params.get("grain_boost", 1.0),
                         mode=grain_mode)

    # --- curvature wear: worn bright edges, grime in the creases ---
    curv = curvature(height, params.get("wear_radius", 2.5) * px)
    edge = np.clip(curv, 0.0, 1.0) ** 1.5
    cav = np.clip(-curv, 0.0, 1.0) ** 1.2
    ew = params.get("edge_wear", 0.0)
    if ew > 0:
        m = (edge * ew)[..., None]
        albedo = albedo * (1.0 - m) + np.array(WEAR_COLOR) * m
    gr = params.get("grime", 0.0)
    if gr > 0:
        albedo = albedo * (1.0 - 0.85 * gr * cav)[..., None]

    sm = spec_mask if spec_mask is not None else np.ones_like(height)
    ws = params.get("wear_spec", 0.0)
    if ws > 0:  # touched edges get polished
        sm = sm + edge * ws

    # --- weathering: corrosion / flaking / drips / lichen (reads base geometry) ---
    if any(params.get(k, 0.0) for k in _WEATHER_KEYS):
        albedo, sm, hdelta = apply_weather(albedo, height, curv, params, rng, size, sm)
        height = np.clip(height + hdelta, 0.0, 1.0)

    # normal map (for the app's relight shader) reflects any weathered geometry
    LAST_NORMALS = normals(height, strength=mat["relief"] * params.get("relief", 1.0)) * 0.5 + 0.5

    # --- emissive channel: self-lit glow, feeds bloom ---
    emit_amt = params.get("emission", 0.0)
    emission = None
    if emit_amt > 0 or emit_source is not None:
        src = emit_source if emit_source is not None else select_source(
            params.get("emission_source", "tone"), height, tone, curv)
        emission = emission_rgb(src, params, rng, size) * max(emit_amt, 1e-6)

    # --- spatial roughness: spec_pow becomes a map (smudges, matte patches) ---
    spec_amt = mat["spec"] * params.get("spec_boost", 1.0)
    spec_pow_eff = mat["spec_pow"] * params.get("spec_pow_boost", 1.0)
    ra = params.get("rough_amount", 0.0)
    if ra > 0:
        rough = fbm(size, rng, octaves=4, freq=int(params.get("rough_freq", 6)))
        spec_pow_eff = spec_pow_eff * np.exp((0.5 - rough) * 2.4 * ra)
        spec_amt = spec_amt * (1.0 - 0.5 * ra * rough)

    img = shade(height, albedo,
                light_angle=params.get("light_angle", 2.2),
                light_elev=params.get("light_elev", 0.55),
                relief=mat["relief"] * params.get("relief", 1.0),
                ambient=mat["ambient"] * params.get("ambient_boost", 1.0),
                diffuse=mat["diffuse"] * params.get("diffuse_boost", 1.0),
                spec=spec_amt, spec_pow=spec_pow_eff, spec_mask=sm,
                ao_strength=params.get("ao_strength", 0.9),
                ao_radii=tuple(r * px for r in ao_radii),
                key_tint=warmth_rgb(params.get("key_warmth", 0.0)),
                fill_strength=params.get("fill_strength", 0.0),
                fill_angle=params.get("fill_angle"),
                fill_elev=params.get("fill_elev", 0.3),
                fill_tint=warmth_rgb(params.get("fill_warmth", 0.0)),
                rim_strength=params.get("rim_strength", 0.0),
                shadow_strength=params.get("shadow_strength", 0.0),
                shadow_soft=params.get("shadow_soft", 2.0) * px,
                shadow_height=params.get("shadow_height", 0.15),
                spec_tint=params.get("spec_tint", 0.0),
                fresnel=params.get("fresnel", 0.0),
                emission=emission)
    if do_finish:
        img = finish(img, rng,
                     film_grain=params.get("film_grain", 0.018),
                     vignette=params.get("vignette", 0.22),
                     bloom=params.get("bloom", 0.18),
                     bloom_thresh=params.get("bloom_thresh", 0.78),
                     chroma=params.get("chroma", 0.6),
                     emission=emission,
                     bloom_emissive=params.get("bloom_emissive", 1.6),
                     bloom_radius=params.get("bloom_radius", 0.012))
    return img


def sample_common(rng) -> dict:
    """Params every family shares — the full dial sweep. Any of these can be
    pinned via the overrides box in the UI / `overrides` in the API."""
    u = rng.uniform
    ang = float(u(1.6, 4.4))
    return {
        # key light
        "light_angle": round(ang, 3),
        "light_elev": round(float(u(0.45, 0.7)), 3),
        "key_warmth": round(float(u(-0.35, 0.85)), 3),
        "diffuse_boost": round(float(u(0.85, 1.25)), 3),
        "ambient_boost": round(float(u(0.8, 1.15)), 3),
        # fill + rim light
        "fill_strength": round(float(u(0.0, 0.5)), 3),
        "fill_angle": round(float((ang + np.pi + u(-0.7, 0.7)) % (2 * np.pi)), 3),
        "fill_elev": round(float(u(0.15, 0.5)), 3),
        "fill_warmth": round(float(u(-1.0, 0.25)), 3),
        "rim_strength": round(float(u(0.0, 0.4)), 3),
        # cast shadows
        "shadow_strength": round(float(u(0.15, 0.85)), 3),
        "shadow_soft": round(float(u(0.8, 5.0)), 3),
        "shadow_height": round(float(u(0.04, 0.28)), 3),
        # specular character
        "spec_tint": round(float(u(0.0, 1.0)), 3),
        "fresnel": round(float(u(0.0, 0.6)), 3),
        "spec_boost": round(float(u(0.8, 1.4)), 3),
        "spec_pow_boost": round(float(u(0.7, 1.5)), 3),
        "rough_amount": round(float(u(0.0, 0.75)), 3),
        "rough_freq": int(rng.choice([3, 4, 6, 9, 13])),
        # curvature wear
        "edge_wear": round(float(u(0.0, 0.7)), 3),
        "grime": round(float(u(0.0, 0.65)), 3),
        "wear_radius": round(float(u(1.5, 5.0)), 3),
        "wear_spec": round(float(u(0.0, 0.9)), 3),
        # emissive channel (mostly off for generic families; goo forces it on).
        # source biased to line-fields (crevice/edge/ridge) so a stray high
        # sample glows as thin traces, not a whole-tile white blowout.
        "emission": round(float(rng.choice([0.0, 0.0, 0.0, u(0.3, 1.1)])), 3),
        "emission_color": EMISSION_COLORS_KEYS[int(rng.integers(len(EMISSION_COLORS_KEYS)))],
        "emission_source": str(rng.choice(
            ["crevice", "edge", "ridge", "crevice", "edge", "tone", "invert_tone"])),
        "emission_thresh": round(float(u(0.45, 0.85)), 3),
        "emission_sharp": round(float(u(0.04, 0.3)), 3),
        "emission_gamma": round(float(u(0.6, 2.2)), 3),
        "emission_white": round(float(u(0.0, 0.45)), 3),
        "emission_flicker": round(float(u(0.0, 0.7)), 3),
        "emission_flicker_freq": int(rng.choice([4, 6, 8, 12, 16])),
        "bloom_emissive": round(float(u(0.8, 3.0)), 3),
        "bloom_radius": round(float(u(0.006, 0.03)), 4),
        # weathering (each effect mostly off; freqs/shape always present)
        "weather_rust": round(float(rng.choice([0.0, 0.0, 0.0, u(0.25, 0.8)])), 3),
        "weather_rust_freq": int(rng.choice([4, 6, 8, 11])),
        "weather_verdigris": round(float(rng.choice([0.0, 0.0, 0.0, u(0.25, 0.75)])), 3),
        "weather_verd_freq": int(rng.choice([4, 6, 8, 11])),
        "weather_chips": round(float(rng.choice([0.0, 0.0, 0.0, u(0.2, 0.6)])), 3),
        "weather_chip_freq": int(rng.choice([6, 9, 13, 18])),
        "weather_chip_depth": round(float(u(0.02, 0.09)), 3),
        "weather_drips": round(float(rng.choice([0.0, 0.0, 0.0, u(0.3, 0.8)])), 3),
        "weather_drip_len": round(float(u(0.25, 0.8)), 3),
        "weather_drip_freq": int(rng.choice([4, 6, 8])),
        "weather_lichen": round(float(rng.choice([0.0, 0.0, 0.0, u(0.2, 0.6)])), 3),
        "weather_lichen_freq": int(rng.choice([7, 10, 14])),
        "weather_cavity_bias": round(float(u(0.4, 0.85)), 3),
        # occlusion + surface
        "ao_strength": round(float(u(0.55, 1.25)), 3),
        "grain_boost": round(float(u(0.7, 1.3)), 3),
        # finish
        "film_grain": round(float(u(0.006, 0.032)), 4),
        "vignette": round(float(u(0.05, 0.35)), 3),
        "bloom": round(float(u(0.05, 0.35)), 3),
        "bloom_thresh": round(float(u(0.68, 0.88)), 3),
        "chroma": round(float(u(0.0, 1.3)), 3),
        # grey-first QA
        "gray_ramp": GRAY_RAMPS[int(rng.integers(len(GRAY_RAMPS)))],
    }
