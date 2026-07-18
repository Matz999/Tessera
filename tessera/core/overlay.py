"""Post overlays — atmospheric, photographic and generative flourishes layered
onto every family after finish(). Each effect is its own dial, off at 0 and
sampled mostly-off, so ~half of tiles stay clean while breeding gets a big new
surface of things to vary and recombine:

  atmosphere   cloud (cloudy colour gradient), bokeh (defocused orbs),
               soft_focus (dreamy Orton glow)
  generative   shapes (random geometry), lines (hatching / rational-angle line
               families), func_curves (sine / Lissajous / rose / spiral plots)
  photographic glare (lens flare + anamorphic streak), twinkle (star sparkles),
               dust (specks), scratches (film scratches)
  tone         contrast (S-curve punch)

Everything composites through the seamless wrapped primitives / toroidal
distances, so tiles stay tileable.
"""
import numpy as np

from .draw import capsule, coords, disk, rect, ring, stamp_disk
from .noise import fbm, gaussian_blur


def _hsv(h, s, v):
    h = h % 1.0
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return np.array([(v, t, p), (q, v, p), (p, v, t),
                     (p, q, v), (t, p, v), (v, p, q)][i])


def _n01(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-9)


def _screen(img, glow, color):
    """Screen-blend a scalar glow field tinted `color` onto an RGB image."""
    return 1.0 - (1.0 - img) * (1.0 - np.clip(glow, 0, 1)[..., None] * np.asarray(color))


def _over(img, mask, color):
    m = np.clip(mask, 0, 1)[..., None]
    return img * (1.0 - m) + np.asarray(color) * m


def _torus(X, Y, cx, cy, size):
    dx = ((X - cx + size / 2) % size) - size / 2
    dy = ((Y - cy + size / 2) % size) - size / 2
    return dx, dy


# --------------------------------------------------------------- atmosphere
def _cloud(img, p, rng, size, X, Y):
    a = float(p.get("cloud", 0.0))
    if a <= 0:
        return img
    c = _n01(fbm(size, rng, octaves=4, freq=int(p.get("cloud_freq", 2))))
    h = rng.random()
    c1 = _hsv(h, rng.uniform(0.3, 0.7), 1.0)
    c2 = _hsv((h + rng.uniform(0.15, 0.5)) % 1.0, rng.uniform(0.3, 0.7), 0.65)
    grad = c1 * (1 - c[..., None]) + c2 * c[..., None]
    return img * (1 - a * 0.6) + grad * (a * 0.6)


def _bokeh(img, p, rng, size, X, Y):
    a = float(p.get("bokeh", 0.0))
    if a <= 0:
        return img
    base = float(p.get("bokeh_size", 0.08))
    for _ in range(int(1 + a * 18)):
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        r = size * base * rng.uniform(0.5, 1.6)
        dx, dy = _torus(X, Y, cx, cy, size)
        d = np.sqrt(dx * dx + dy * dy)
        g = np.exp(-(d / r) ** 2) * 0.55 + np.clip(1 - np.abs(d - r * 0.9) / (r * 0.16), 0, 1) * 0.45
        img = _screen(img, g * a * rng.uniform(0.3, 0.7), _hsv(rng.random(), rng.uniform(0.2, 0.6), 1.0))
    return img


def _soft_focus(img, p, rng, size, X, Y):
    a = float(p.get("soft_focus", 0.0))
    if a <= 0:
        return img
    r = size * float(p.get("soft_radius", 0.02))
    bl = np.stack([gaussian_blur(img[..., c], r) for c in range(3)], -1)
    glow = 1.0 - (1.0 - img) * (1.0 - bl)          # screen with its own blur
    return img * (1 - a) + glow * a


# --------------------------------------------------------------- generative
def _shapes(img, p, rng, size, X, Y):
    n = int(p.get("shapes", 0))
    if n <= 0:
        return img
    al = float(p.get("shape_alpha", 0.4))
    px = size / 512.0
    for _ in range(n):
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        r = size * rng.uniform(0.05, 0.3)
        col = _hsv(rng.random(), rng.uniform(0.3, 0.9), rng.uniform(0.6, 1.0))
        kind = int(rng.integers(5))
        lw = max(2.0 * px, size * 0.008)
        if kind == 0:
            m = disk(X, Y, size, cx, cy, r, 2 * px)
        elif kind == 1:
            m = ring(X, Y, size, cx, cy, r, r - lw, 2 * px)
        elif kind == 2:
            m = rect(X, Y, size, cx, cy, r, r * rng.uniform(0.4, 1.0), 2 * px)
        elif kind == 3:
            hw, hh = r, r * rng.uniform(0.4, 1.0)
            m = np.clip(rect(X, Y, size, cx, cy, hw, hh, 2 * px)
                        - rect(X, Y, size, cx, cy, hw - lw, hh - lw, 2 * px), 0, 1)
        else:                                       # triangle outline (3 capsules)
            ang = rng.uniform(0, 6.28)
            pts = [(cx + r * np.cos(ang + k * 2.094), cy + r * np.sin(ang + k * 2.094)) for k in range(3)]
            m = np.zeros((size, size))
            for j in range(3):
                ax, ay = pts[j]
                bx, by = pts[(j + 1) % 3]
                m = np.maximum(m, capsule(X, Y, size, ax, ay, bx, by, lw * 0.6, 2 * px))
        img = _over(img, m * al * rng.uniform(0.4, 1.0), col)
    return img


def _lines(img, p, rng, size, X, Y):
    n = int(p.get("lines", 0))
    if n <= 0:
        return img
    al = float(p.get("line_alpha", 0.5))
    for _ in range(n):
        a_, b_ = int(rng.integers(-3, 4)), int(rng.integers(-3, 4))
        if a_ == 0 and b_ == 0:
            b_ = 1
        k = int(rng.integers(2, 14))
        phase = (a_ * X + b_ * Y) / size * k
        tri = np.abs((phase % 1.0) - 0.5)           # 0 on the line, 0.5 between
        w = rng.uniform(0.02, 0.12)
        m = np.clip((w - tri) / (w * 0.5), 0, 1)
        col = _hsv(rng.random(), rng.uniform(0.0, 0.8), rng.uniform(0.7, 1.0))
        img = _over(img, m * al * rng.uniform(0.3, 0.8), col)
    return img


def _functions(img, p, rng, size, X, Y):
    n = int(p.get("func_curves", 0))
    if n <= 0:
        return img
    t = np.linspace(0, 2 * np.pi, 4000)
    for _ in range(n):
        col = _hsv(rng.random(), rng.uniform(0.2, 0.9), 1.0)
        typ = int(rng.integers(6))
        if typ == 0:                                # y = sin(x), seamless (int freq)
            xs = np.linspace(0, size, 4000)
            ys = size / 2 + size * rng.uniform(0.1, 0.35) * np.sin(
                int(rng.integers(1, 8)) * 2 * np.pi * xs / size + rng.uniform(0, 6.28))
        elif typ == 1:                              # Lissajous
            xs = size / 2 + size * 0.42 * np.sin(int(rng.integers(1, 6)) * t + rng.uniform(0, 6.28))
            ys = size / 2 + size * 0.42 * np.sin(int(rng.integers(1, 6)) * t)
        elif typ == 2:                              # rose r=cos(kθ)
            r = size * 0.42 * np.cos(int(rng.integers(2, 7)) * t)
            xs, ys = size / 2 + r * np.cos(t), size / 2 + r * np.sin(t)
        elif typ == 3:                              # spiral
            r = np.linspace(0, size * 0.45, 4000)
            f = int(rng.integers(3, 9))
            xs, ys = size / 2 + r * np.cos(f * t), size / 2 + r * np.sin(f * t)
        elif typ == 4:                              # hypotrochoid (spirograph)
            R, rr = int(rng.integers(3, 9)), int(rng.integers(1, 6))
            d = rng.uniform(0.4, 1.2) * rr
            tt = np.linspace(0, 2 * np.pi * rr / max(int(np.gcd(R, rr)), 1), 4000)
            sc = size * 0.42 / (R + d)
            xs = size / 2 + sc * ((R - rr) * np.cos(tt) + d * np.cos((R - rr) / rr * tt))
            ys = size / 2 + sc * ((R - rr) * np.sin(tt) - d * np.sin((R - rr) / rr * tt))
        else:                                       # de Jong attractor (wrapped scatter)
            a1, b1, c1, d1 = rng.uniform(-2.5, 2.5, 4)
            xs = np.empty(4000); ys = np.empty(4000)
            x = y = 0.1
            for i in range(4000):
                x, y = np.sin(a1 * y) - np.cos(b1 * x), np.sin(c1 * x) - np.cos(d1 * y)
                xs[i], ys[i] = size / 2 + x * size * 0.22, size / 2 + y * size * 0.22
        layer = np.zeros((size, size))
        w = rng.uniform(0.6, 2.0)
        for x, y in zip(xs, ys):
            stamp_disk(layer, float(x), float(y), w, 1.0)   # wrapped -> seamless
        img = _screen(img, np.clip(layer, 0, 1) * rng.uniform(0.5, 0.9), col)
    return img


# --------------------------------------------------------------- photographic
def _glare(img, p, rng, size, X, Y):
    a = float(p.get("glare", 0.0))
    if a <= 0:
        return img
    cx, cy = rng.uniform(0, size), rng.uniform(0, size)
    dx, dy = _torus(X, Y, cx, cy, size)
    d = np.sqrt(dx * dx + dy * dy)
    g = np.exp(-(d / (size * rng.uniform(0.2, 0.5))) ** 2)
    img = _screen(img, g * a, _hsv(rng.uniform(0.08, 0.15), 0.4, 1.0))
    streak = np.exp(-(dy / (size * 0.012)) ** 2) * np.exp(-(dx / (size * 0.42)) ** 2)
    return _screen(img, streak * a * 0.8, (0.6, 0.7, 1.0))


def _twinkle(img, p, rng, size, X, Y):
    n = int(p.get("twinkle", 0))
    if n <= 0:
        return img
    px = size / 512.0
    for _ in range(n):
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        s = px * rng.uniform(2, 6)
        dx, dy = _torus(X, Y, cx, cy, size)
        dot = np.exp(-((dx / s) ** 2 + (dy / s) ** 2))
        cross = (np.exp(-(dy / (s * 0.25)) ** 2) * np.exp(-(dx / (s * 4)) ** 2)
                 + np.exp(-(dx / (s * 0.25)) ** 2) * np.exp(-(dy / (s * 4)) ** 2))
        col = _hsv(rng.uniform(0, 1), rng.uniform(0.0, 0.3), 1.0)
        img = _screen(img, np.clip(dot + cross * 0.7, 0, 1) * rng.uniform(0.5, 1.0), col)
    return img


def _dust(img, p, rng, size, X, Y):
    a = float(p.get("dust", 0.0))
    if a <= 0:
        return img
    m = rng.random((size, size))
    thr = 1.0 - a * 0.012
    img = np.clip(img + (m > thr).astype(float)[..., None] * 0.8, 0, 1)
    return img * (1.0 - (m < (1.0 - thr)).astype(float)[..., None] * 0.7)


def _scratches(img, p, rng, size, X, Y):
    n = int(p.get("scratches", 0))
    if n <= 0:
        return img
    px = size / 512.0
    for _ in range(n):
        x0 = rng.uniform(0, size)
        wav = size * 0.01 * np.sin(Y / size * 2 * np.pi * int(rng.integers(1, 4)) + rng.uniform(0, 6.28))
        dx = ((X - (x0 + wav) + size / 2) % size) - size / 2
        w = px * rng.uniform(0.6, 1.6)
        m = np.clip((w - np.abs(dx)) / w, 0, 1)
        if rng.random() < 0.5:
            img = np.clip(img + (m * rng.uniform(0.4, 0.9))[..., None], 0, 1)
        else:
            img = img * (1.0 - (m * rng.uniform(0.4, 0.8))[..., None])
    return img


# --------------------------------------------------------------- colour grade
def _lum(img):
    return np.clip(img @ np.array([0.299, 0.587, 0.114]), 0, 1)


def _duotone(img, p, rng, size, X, Y):
    a = float(p.get("duotone", 0.0))
    if a <= 0:
        return img
    lum = _lum(img)[..., None]
    c1 = _hsv(float(p.get("duo_h1", 0.62)), 0.6, 0.14)     # shadow tint
    c2 = _hsv(float(p.get("duo_h2", 0.09)), 0.55, 1.0)     # highlight tint
    return img * (1 - a) + (c1 * (1 - lum) + c2 * lum) * a


def _posterize(img, p, rng, size, X, Y):
    lv = int(p.get("posterize", 0))
    if lv < 2:
        return img
    return np.round(np.clip(img, 0, 1) * (lv - 1)) / (lv - 1)


def _halftone(img, p, rng, size, X, Y):
    a = float(p.get("halftone", 0.0))
    if a <= 0:
        return img
    n = int(p.get("halftone_cells", 48))
    lum = _lum(img)
    u = ((X / size * n) % 1.0) - 0.5
    v = ((Y / size * n) % 1.0) - 0.5
    d = np.sqrt(u * u + v * v)
    dot = np.clip((np.sqrt(1 - lum) * 0.72 - d) / 0.07, 0, 1)   # darker -> bigger dot
    ink = _hsv(float(p.get("duo_h1", 0.62)), 0.35, 0.07)
    paper = _hsv(float(p.get("duo_h2", 0.09)), 0.12, 0.98)
    ht = paper * (1 - dot[..., None]) + ink * dot[..., None]
    return img * (1 - a) + ht * a


# --------------------------------------------------------------- display fx
def _aberration(img, p, rng, size, X, Y):
    a = float(p.get("aberration", 0.0))
    if a <= 0:
        return img
    s = max(int(a * size * 0.02), 1)
    r = np.roll(img[..., 0], s, axis=1)             # wrapped -> seamless channel split
    b = np.roll(img[..., 2], -s, axis=1)
    return np.stack([r, img[..., 1], b], -1)


def _scanlines(img, p, rng, size, X, Y):
    a = float(p.get("scanlines", 0.0))
    if a <= 0:
        return img
    n = int(p.get("scanline_freq", 180))
    s = 0.5 + 0.5 * np.cos(2 * np.pi * Y / size * n)   # int n -> seamless
    return img * (1 - a * 0.45 * (1 - s))[..., None]


def _contrast(img, p):
    c = float(p.get("contrast", 0.0))
    if c == 0:
        return img
    x = np.clip(img, 0, 1)
    if c > 0:                                       # S-curve punch
        s = (x - 0.5) * (1 + 2 * c) + 0.5
        return np.clip(0.5 * s + 0.5 * (3 * x * x - 2 * x * x * x), 0, 1) if c > 0.5 else np.clip(s, 0, 1)
    return np.clip((x - 0.5) * (1 + c * 0.8) + 0.5, 0, 1)


_ATMOS = (_cloud, _bokeh, _soft_focus)
_GEN = (_shapes, _lines, _functions)
_GRADE = (_duotone, _posterize, _halftone)
_PHOTO = (_glare, _twinkle, _dust, _scratches)
_DISPLAY = (_aberration, _scanlines)


def apply_overlays(img: np.ndarray, params: dict, rng, size: int) -> np.ndarray:
    """Run every overlay in order (each self-skips when its dial is 0): build
    (atmosphere, generative), grade the colour, add photographic artifacts,
    punch contrast, then display effects (channel split / scanlines) last."""
    X, Y = coords(size)
    for fn in _ATMOS + _GEN + _GRADE + _PHOTO:
        img = fn(img, params, rng, size, X, Y)
    img = _contrast(img, params)
    for fn in _DISPLAY:
        img = fn(img, params, rng, size, X, Y)
    return img
