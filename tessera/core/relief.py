"""Heightfield -> lit surface: normals, directional light, AO, specular.

This is the material-realism core: every family authors a grayscale
heightfield and albedo, then this module makes it read as physical relief.
"""
import numpy as np

from .noise import gaussian_blur
from .util import bilinear_wrap


def normals(height: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """HxW height [0,1] -> HxWx3 unit normals."""
    gy, gx = np.gradient(height)
    scale = strength * height.shape[0] * 0.5
    nx = -gx * scale
    ny = -gy * scale
    nz = np.ones_like(height)
    mag = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / mag, ny / mag, nz / mag], axis=-1)


def ambient_occlusion(height: np.ndarray, radii=(3, 9, 24), strength: float = 1.0) -> np.ndarray:
    """Multi-scale crevice AO: 0 = open, 1 = fully occluded."""
    ao = np.zeros_like(height)
    for r in radii:
        cavity = gaussian_blur(height, r) - height  # positive in recesses
        ao += np.clip(cavity, 0.0, 1.0)
    ao = np.clip(ao * (2.2 / len(radii)) * strength, 0.0, 1.0)
    return ao


def curvature(height: np.ndarray, radius: float = 2.5) -> np.ndarray:
    """Signed unsharp curvature: positive on ridges/exposed edges, negative in
    creases. Normalized to roughly [-1, 1]."""
    c = height - gaussian_blur(height, radius)
    return np.clip(c / (np.std(c) * 3.0 + 1e-9), -1.0, 1.0)


def cast_shadows(height: np.ndarray, light_angle: float, light_elev: float,
                 height_scale: float = 0.15, softness: float = 2.0) -> np.ndarray:
    """Tileable horizon-march cast shadows: 0 = lit, 1 = fully shadowed.

    Marches toward the light with geometrically growing steps (wrap sampling,
    so shadows tile seamlessly). height_scale: world height of relief as a
    fraction of tile width — the master 'how deep is this carving' dial."""
    size = height.shape[0]
    lz = float(np.clip(light_elev, 0.05, 0.999))
    lxy = np.sqrt(1.0 - lz * lz)
    dx, dy = np.cos(light_angle), np.sin(light_angle)
    Hp = height * (height_scale * size)      # height in pixel units
    slope = lz / max(lxy, 1e-6)              # light ray rise per pixel step
    idx = np.arange(size, dtype=np.float64)
    xs, ys = np.meshgrid(idx, idx)
    occ = np.zeros_like(Hp)
    t = 1.0
    while t < size * 0.35:
        blocker = bilinear_wrap(Hp, xs + dx * t, ys + dy * t)
        occ = np.maximum(occ, blocker - Hp - t * slope)
        t *= 1.22
    sh = np.clip(occ * 0.5, 0.0, 1.0)        # ~2px of blockage = full shadow
    return gaussian_blur(sh, softness) if softness > 0 else sh


def _light_vec(angle: float, elev: float) -> np.ndarray:
    lz = float(np.clip(elev, 0.0, 0.999))
    lxy = np.sqrt(max(1.0 - lz * lz, 0.0))
    return np.array([np.cos(angle) * lxy, np.sin(angle) * lxy, lz])


def shade(height: np.ndarray, albedo: np.ndarray,
          light_angle: float = 2.2, light_elev: float = 0.55,
          relief: float = 1.0, ambient: float = 0.38, diffuse: float = 0.72,
          spec: float = 0.25, spec_pow=28.0, spec_mask=None,
          ao_strength: float = 0.9, ao_radii=(3, 9, 24),
          key_tint=(1.0, 1.0, 1.0),
          fill_strength: float = 0.0, fill_angle=None, fill_elev: float = 0.3,
          fill_tint=(1.0, 1.0, 1.0),
          rim_strength: float = 0.0,
          shadow_strength: float = 0.0, shadow_soft: float = 2.0,
          shadow_height: float = 0.15,
          spec_tint: float = 0.0, fresnel: float = 0.0,
          emission=None) -> np.ndarray:
    """Light a heightfield. albedo HxWx3 (or HxW). Returns HxWx3 in [0,1].

    Dials (all default to the classic single-light look):
      light_angle/light_elev  key light azimuth (rad) / elevation (0..1)
      key_tint / fill_tint    rgb multipliers for light color
      fill_strength/angle/elev  second bounce light (default opposite the key)
      rim_strength            grazing-angle rim glow (uses fill tint)
      shadow_strength/soft/height  horizon-marched cast shadows
      spec / spec_pow         Blinn-Phong; spec_pow may be an HxW roughness map
      spec_mask               HxW glossiness multiplier (glaze/metal regions)
      spec_tint               0 = white highlights .. 1 = metal (albedo-tinted)
      fresnel                 extra grazing-angle specular (metals, glaze)
    """
    if albedo.ndim == 2:
        albedo = np.stack([albedo] * 3, axis=-1)
    n = normals(height, strength=relief)
    nz = n[..., 2]
    L = _light_vec(light_angle, light_elev)
    ndl = np.clip(n[..., 0] * L[0] + n[..., 1] * L[1] + n[..., 2] * L[2], 0.0, 1.0)

    ao = ambient_occlusion(height, radii=ao_radii, strength=ao_strength)
    shadowing = 1.0 - ao * 0.85

    if shadow_strength > 0:
        occ = cast_shadows(height, light_angle, light_elev,
                           height_scale=shadow_height, softness=shadow_soft)
        sun = 1.0 - shadow_strength * occ
    else:
        sun = np.ones_like(height)

    # diffuse stack: ambient + tinted key (shadowed) + tinted fill + rim
    lum = (ambient * shadowing)[..., None] * np.ones(3)
    lum += (diffuse * ndl * sun * shadowing)[..., None] * np.asarray(key_tint)
    ft = np.asarray(fill_tint)
    if fill_strength > 0:
        fa = light_angle + np.pi if fill_angle is None else fill_angle
        Lf = _light_vec(fa, fill_elev)
        ndlf = np.clip(n[..., 0] * Lf[0] + n[..., 1] * Lf[1] + n[..., 2] * Lf[2],
                       0.0, 1.0)
        lum += (fill_strength * ndlf * shadowing)[..., None] * ft
    if rim_strength > 0:
        rim = (1.0 - nz) ** 3.5
        lum += (rim_strength * rim * shadowing)[..., None] * ft

    # Blinn-Phong specular, view straight down.
    H = L + np.array([0.0, 0.0, 1.0])
    H = H / np.linalg.norm(H)
    ndh = np.clip(n[..., 0] * H[0] + n[..., 1] * H[1] + n[..., 2] * H[2], 0.0, 1.0)
    s = (ndh ** spec_pow) * spec
    if spec_mask is not None:
        s = s * spec_mask
    if fresnel > 0:
        s = s + fresnel * spec * ((1.0 - nz) ** 5)
    s = s * sun * shadowing
    if spec_tint > 0:  # metals tint their highlights with the surface color
        tint = albedo / (albedo.max(axis=-1, keepdims=True) + 1e-6)
        spec_col = (1.0 - spec_tint) + spec_tint * tint
    else:
        spec_col = np.ones(3)
    out = albedo * lum + s[..., None] * spec_col
    if emission is not None:
        out = out + emission
    return np.clip(out, 0.0, 1.0)
