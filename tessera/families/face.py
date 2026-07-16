"""Family 11: Carved mask / idol face.

A bilaterally-symmetric ceremonial mask sculpted from smooth Gaussian bumps and
recesses — raised oval plate, brow ridge, eye sockets/bulges, nose bridge +
tip, nostrils, cheeks, mouth, chin, plus a forehead mark. Symmetry is enforced
by folding x to |x|, so the face is exact left/right. The mask is centered on a
textured background, so the tile stays seamless (only the background wraps).
Eyes drive the emissive channel — pin `emission` for a glowing-eyed idol.
"""
import numpy as np

from ..core.noise import fbm, gaussian_blur
from ..core.util import grid_coords, norm01, rng_for
from .common import render_material, sample_common

FAMILY = "face"

DIALS = {
    "face_rx": {"lo": 0.4, "hi": 0.95, "step": 0.01},
    "face_ry": {"lo": 0.5, "hi": 1.15, "step": 0.01},
    "face_h": {"lo": 0.3, "hi": 1.0, "step": 0.01},
    "brow_h": {"lo": 0.0, "hi": 0.7, "step": 0.01},
    "brow_y": {"lo": -0.55, "hi": -0.05, "step": 0.01},
    "brow_w": {"lo": 0.25, "hi": 0.85, "step": 0.01},
    "brow_angle": {"lo": -0.35, "hi": 0.35, "step": 0.01},
    "eye_x": {"lo": 0.12, "hi": 0.45, "step": 0.01},
    "eye_y": {"lo": -0.30, "hi": 0.10, "step": 0.01},
    "eye_rx": {"lo": 0.05, "hi": 0.22, "step": 0.01},
    "eye_ry": {"lo": 0.04, "hi": 0.18, "step": 0.01},
    "eye_depth": {"lo": -0.6, "hi": 0.6, "step": 0.02},
    "pupil": {"lo": 0.0, "hi": 0.5, "step": 0.02},
    "nose_w": {"lo": 0.04, "hi": 0.22, "step": 0.01},
    "nose_len": {"lo": 0.15, "hi": 0.75, "step": 0.01},
    "nose_h": {"lo": 0.0, "hi": 0.7, "step": 0.01},
    "nostril": {"lo": 0.0, "hi": 0.4, "step": 0.01},
    "mouth_y": {"lo": 0.25, "hi": 0.75, "step": 0.01},
    "mouth_w": {"lo": 0.12, "hi": 0.5, "step": 0.01},
    "mouth_h": {"lo": 0.03, "hi": 0.22, "step": 0.01},
    "mouth_open": {"lo": -0.5, "hi": 0.5, "step": 0.02},
    "mouth_curve": {"lo": -0.35, "hi": 0.35, "step": 0.01},
    "cheek": {"lo": 0.0, "hi": 0.6, "step": 0.01},
    "chin": {"lo": 0.0, "hi": 0.6, "step": 0.01},
    "forehead_mark": {"lo": 0.0, "hi": 0.5, "step": 0.02},
    "bg_tex": {"lo": 0.0, "hi": 0.2, "step": 0.01},
}

# spooky emissive-eye colors
EYE_GLOW = ["lava", "blood", "neon_green", "neon_cyan", "plasma", "sodium",
            "ember", "acid"]


def sample_params(rng) -> dict:
    p = sample_common(rng)
    u = rng.uniform
    p.update({
        "face_rx": round(float(u(0.55, 0.85)), 3),
        "face_ry": round(float(u(0.7, 1.05)), 3),
        "face_h": round(float(u(0.5, 0.9)), 3),
        "brow_h": round(float(u(0.15, 0.55)), 3),
        "brow_y": round(float(u(-0.45, -0.15)), 3),
        "brow_w": round(float(u(0.35, 0.7)), 3),
        "brow_angle": round(float(u(-0.3, 0.3)), 3),
        "eye_x": round(float(u(0.2, 0.38)), 3),
        "eye_y": round(float(u(-0.22, 0.0)), 3),
        "eye_rx": round(float(u(0.08, 0.18)), 3),
        "eye_ry": round(float(u(0.06, 0.14)), 3),
        "eye_depth": round(float(u(-0.5, 0.4)), 3),
        "pupil": round(float(u(0.0, 0.4)), 3),
        "nose_w": round(float(u(0.06, 0.16)), 3),
        "nose_len": round(float(u(0.25, 0.6)), 3),
        "nose_h": round(float(u(0.2, 0.6)), 3),
        "nostril": round(float(u(0.05, 0.3)), 3),
        "mouth_y": round(float(u(0.35, 0.62)), 3),
        "mouth_w": round(float(u(0.18, 0.42)), 3),
        "mouth_h": round(float(u(0.05, 0.16)), 3),
        "mouth_open": round(float(u(-0.4, 0.4)), 3),
        "mouth_curve": round(float(u(-0.3, 0.3)), 3),
        "cheek": round(float(u(0.1, 0.45)), 3),
        "chin": round(float(u(0.1, 0.45)), 3),
        "forehead_mark": round(float(rng.choice([0.0, 0.0, u(0.2, 0.45)])), 3),
        "bg_tex": round(float(u(0.02, 0.09)), 3),
        # gentler weathering than the shared default, so the mask stays legible
        "weather_rust": round(float(rng.choice([0.0, 0.0, 0.0, 0.0, u(0.2, 0.45)])), 3),
        "weather_verdigris": round(float(rng.choice([0.0, 0.0, 0.0, 0.0, u(0.2, 0.4)])), 3),
        "weather_lichen": round(float(rng.choice([0.0, 0.0, 0.0, 0.0, u(0.15, 0.35)])), 3),
        "weather_chips": 0.0,
        "material": str(rng.choice(["stone", "bronze", "ceramic", "vellum"])),
        "ramp": str(rng.choice(["bone", "terracotta", "oxblood", "ink_vellum",
                                "gold_indigo", "verdigris", "celadon"])),
        # ~30% of idols have glowing eyes
        "emission": round(float(rng.choice([0.0, 0.0, u(0.7, 1.4)])), 3),
        "emission_color": str(rng.choice(EYE_GLOW)),
        "emission_thresh": 0.45,
        "emission_white": round(float(u(0.15, 0.35)), 3),
        "bloom_radius": round(float(u(0.008, 0.016)), 4),
    })
    return p


def fields(params: dict, size: int, gray: bool, rng):
    """Pre-render fields (height, tone, spec_mask, emit, albedo) for the mixer.
    emit is the eye glow; albedo is None (tone-ramped, single material)."""
    X, Y = grid_coords(size, centered=True)   # [-1,1], Y increases downward
    p = params

    def gs(cx, cy, rx, ry, shear=0.0):
        """Single smooth 2D Gaussian bump (signed X)."""
        dx = X - cx
        dy = (Y - cy) + shear * (X - cx)
        return np.exp(-((dx / rx) ** 2 + (dy / ry) ** 2))

    def gp(cx, cy, rx, ry, shear=0.0):
        """Bilaterally-symmetric pair (mirror + shear-flip): sum of smooth
        Gaussians, so there is no center-fold crease."""
        return gs(cx, cy, rx, ry, shear) + gs(-cx, cy, rx, ry, -shear)

    # super-Gaussian plate: flat top + crisp rim -> reads as a distinct mask.
    # (X**2 makes it symmetric.) Features are gated by `plate`.
    d2 = (X / p["face_rx"]) ** 2 + ((Y - 0.05) / p["face_ry"]) ** 2
    plate = np.exp(-(np.clip(d2, 0, 4) ** 2.2))
    bgnoise = (fbm(size, rng, octaves=5, freq=5) - 0.5) * p["bg_tex"]
    h = 0.08 + (1.0 - plate) * bgnoise + plate * 0.5 * p["face_h"]
    facemask = plate

    # brow ridge (wide, thin; angled for expression)
    h += p["brow_h"] * 0.16 * plate * gp(p["eye_x"] * 0.9, p["brow_y"],
                                         p["brow_w"] * 0.6, 0.07, shear=p["brow_angle"])
    # cheeks
    h += p["cheek"] * 0.14 * plate * gp(p["eye_x"] * 1.05, 0.28, 0.26, 0.24)
    # nose bridge + tip (centered)
    nose_cy = p["eye_y"] + p["nose_len"] * 0.5
    h += p["nose_h"] * 0.17 * plate * gs(0.0, nose_cy, p["nose_w"], p["nose_len"] * 0.5)
    nose_tip = p["eye_y"] + p["nose_len"]
    h += p["nose_h"] * 0.13 * plate * gs(0.0, nose_tip, p["nose_w"] * 1.5, p["nose_w"] * 1.4)
    # chin
    h += p["chin"] * 0.15 * plate * gs(0.0, p["face_ry"] * 0.7, 0.28, 0.2)

    # eyes: socket (recess if eye_depth<0) or bulge, + raised pupil
    eye = gp(p["eye_x"], p["eye_y"], p["eye_rx"], p["eye_ry"])
    h += p["eye_depth"] * 0.22 * plate * eye
    pupil_mask = gp(p["eye_x"], p["eye_y"], p["eye_rx"] * 0.4, p["eye_ry"] * 0.4)
    h += p["pupil"] * 0.16 * plate * pupil_mask
    # nostrils (recessed)
    h -= p["nostril"] * 0.16 * plate * gp(p["nose_w"] * 1.1, nose_tip,
                                          p["nose_w"] * 0.5, p["nose_w"] * 0.6)
    # mouth: recess (open) or raised lips, curved by mouth_curve (centered)
    mouth = gs(0.0, p["mouth_y"], p["mouth_w"], p["mouth_h"], shear=p["mouth_curve"])
    h += p["mouth_open"] * 0.26 * plate * mouth
    # forehead mark (gem/dot)
    if p["forehead_mark"] > 0:
        h += p["forehead_mark"] * 0.2 * plate * gs(0.0, p["brow_y"] - 0.22, 0.05, 0.05)

    # soften general relief (light)
    h = norm01(gaussian_blur(h, size * 0.0014))

    # tone: relief-based, with a brighter plate so the mask separates from bg
    tone = norm01(0.3 + 0.45 * h + 0.25 * facemask)

    # glossier on the raised plate, matte in the recesses
    spec_mask = 0.25 + 0.6 * facemask

    # eyes glow when emission is on (concentrated on the pupil)
    eye_glow = norm01(pupil_mask + 0.3 * eye)

    return h, tone, spec_mask, eye_glow, None


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    h, tone, sm, emit, alb = fields(params, size, gray, rng)
    return render_material(h, tone, params, rng, gray, spec_mask=sm,
                           ao_radii=(3, 8, 20), emit_source=emit, albedo=alb)
