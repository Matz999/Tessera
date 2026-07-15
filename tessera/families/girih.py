"""Family 4: Islamic girih — Hankin's polygons-in-contact on a square grid.

From each cell-edge midpoint two rays enter the cell at ±contact-angle;
adjacent rays intersect, yielding star-and-cross strapwork. The pattern is
periodic (authentic) and seamless; sweep angle/density/strap width.
"""
import numpy as np

from ..core.noise import fbm, gaussian_blur
from ..core.util import norm01, rng_for, smoothstep
from .common import render_material, sample_common

FAMILY = "girih"


DIALS = {
    "grid": {"lo": 3, "hi": 6, "step": 1},
    "angle_deg": {"lo": 20.0, "hi": 85.0, "step": 0.5},
    "strap_w": {"lo": 0.01, "hi": 0.12, "step": 0.001},
    "outline": {"bool": True},
    "rosette": {"bool": True},
}


def sample_params(rng) -> dict:
    p = sample_common(rng)
    p.update({
        "grid": int(rng.choice([3, 4, 5, 6])),
        "angle_deg": round(float(rng.uniform(30.0, 75.0)), 1),
        "strap_w": round(float(rng.uniform(0.030, 0.075)), 4),
        "outline": bool(rng.uniform() < 0.7),
        "rosette": bool(rng.uniform() < 0.5),  # extra central star layer per cell
        "material": str(rng.choice(["stone", "bronze", "ceramic", "vellum"])),
        "ramp": str(rng.choice(["gold_indigo", "lapis_gold", "zellige", "ink_vellum",
                                "verdigris", "oxblood", "emerald"])),
    })
    return p


def _ray_intersect(p, d, q, e):
    """Intersection of p+t*d and q+s*e (t,s >= 0), else None."""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    s = ((q[0] - p[0]) * d[1] - (q[1] - p[1]) * d[0]) / den
    if t < 0 or s < 0:
        return None
    return (p[0] + t * d[0], p[1] + t * d[1])


def _hankin_segments(theta):
    """Strapwork segments for a unit cell [0,1]^2. Returns list of (x0,y0,x1,y1)."""
    mids = [(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)]
    edges = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]  # edge directions (CCW)

    def rot(v, a):
        return (v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a))

    segs = []
    for i in range(4):
        m_a, e_a = mids[i], edges[i]
        m_b, e_b = mids[(i + 1) % 4], edges[(i + 1) % 4]
        # ray from A rotated +theta into the cell meets ray from B rotated -theta
        d_a = rot(e_a, theta)
        d_b = rot((-e_b[0], -e_b[1]), -theta)
        hit = _ray_intersect(m_a, d_a, m_b, d_b)
        if hit and -0.25 <= hit[0] <= 1.25 and -0.25 <= hit[1] <= 1.25:
            segs.append((m_a[0], m_a[1], hit[0], hit[1]))
            segs.append((m_b[0], m_b[1], hit[0], hit[1]))
        # and the mirrored pair (-theta from A, +theta from B)
        d_a2 = rot(e_a, np.pi - theta)
        d_b2 = rot((-e_b[0], -e_b[1]), -(np.pi - theta))
        hit2 = _ray_intersect(m_a, d_a2, m_b, d_b2)
        if hit2 and -0.25 <= hit2[0] <= 1.25 and -0.25 <= hit2[1] <= 1.25:
            segs.append((m_a[0], m_a[1], hit2[0], hit2[1]))
            segs.append((m_b[0], m_b[1], hit2[0], hit2[1]))
    return segs


def _star_segments(n, r_out, r_in, cx=0.5, cy=0.5, rot0=0.0):
    segs = []
    for k in range(n):
        a0 = rot0 + 2 * np.pi * k / n
        a1 = rot0 + 2 * np.pi * (k + 0.5) / n
        a2 = rot0 + 2 * np.pi * (k + 1) / n
        p0 = (cx + r_out * np.cos(a0), cy + r_out * np.sin(a0))
        p1 = (cx + r_in * np.cos(a1), cy + r_in * np.sin(a1))
        p2 = (cx + r_out * np.cos(a2), cy + r_out * np.sin(a2))
        segs.append((p0[0], p0[1], p1[0], p1[1]))
        segs.append((p1[0], p1[1], p2[0], p2[1]))
    return segs


def _seg_distance_patch(segs, patch_px):
    """Distance field to segments over a unit cell rendered at patch_px."""
    v = (np.arange(patch_px) + 0.5) / patch_px
    X, Y = np.meshgrid(v, v)
    D = np.full((patch_px, patch_px), 1e9)
    for (x0, y0, x1, y1) in segs:
        vx, vy = x1 - x0, y1 - y0
        L2 = vx * vx + vy * vy
        if L2 < 1e-12:
            continue
        t = np.clip(((X - x0) * vx + (Y - y0) * vy) / L2, 0.0, 1.0)
        d = np.sqrt((X - (x0 + t * vx)) ** 2 + (Y - (y0 + t * vy)) ** 2)
        D = np.minimum(D, d)
    return D


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    rng = rng_for(seed)
    g = params["grid"]
    theta = np.deg2rad(params["angle_deg"])
    patch = size // g

    segs = _hankin_segments(theta)
    if params["rosette"]:
        n = int(rng.choice([8, 12, 16]))
        segs += _star_segments(n, 0.30, 0.14, rot0=float(rng.uniform(0, 1)))
    D = _seg_distance_patch(segs, patch)

    reps = size // patch + (1 if size % patch else 0)
    D = np.tile(D, (reps, reps))[:size, :size]

    w = params["strap_w"]
    strap = smoothstep(w, w * 0.5, D)  # D is in cell units

    ground = 0.30 + (fbm(size, rng, octaves=5, freq=8) - 0.5) * 0.12
    height = ground + strap * 0.52
    if params["outline"]:  # incised outline hugging the strap
        outline = smoothstep(w * 1.45, w * 1.1, D) - strap
        height -= np.clip(outline, 0, 1) * 0.18
    height = norm01(gaussian_blur(height, size * 0.0015))

    tone = norm01(0.28 + strap * 0.62 + (fbm(size, rng, octaves=4, freq=6) - 0.5) * 0.10)
    spec_mask = 0.25 + 0.75 * strap  # gilt straps catch the light
    return render_material(height, tone, params, rng, gray, spec_mask=spec_mask,
                           ao_radii=(2, 6, 16))
