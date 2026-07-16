"""Stamp engine: turn a font glyph or an uploaded image into a seamless,
tileable coverage field that a family can emboss/engrave into a tile.

Two sources, both deterministic from the tile's params:
  * glyph  — a character rendered with a system font (emoji, dingbats, Segoe
             icons, letters). Fully reproducible from (char, font); no asset.
  * image  — a user-uploaded picture, stored content-addressed under
             <outdir>/_stamps/<sha1>.png and referenced by that filename.

A stamp resolves to two square fields in [0,1]:
  ink — where the stamp *is* (glyph alpha / image alpha or full frame). Drives
        the emboss mask, the albedo inlay, spec and the emissive source.
  lum — interior relief (emoji shading / image luminance), so features carve.

Instances are composited into the size x size tile with wrapped (`% size`)
indices, so a stamp crossing the seam reappears on the far side — every
arrangement (single / grid / brick / scatter) tiles seamlessly.
"""
import hashlib
import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------ asset store
_STORE = os.path.abspath(os.path.join("library", "_stamps"))


def set_store(outdir: str) -> None:
    """Point the uploaded-image store at <outdir>/_stamps (serve/render call this)."""
    global _STORE
    _STORE = os.path.join(os.path.abspath(outdir), "_stamps")


def store_dir() -> str:
    os.makedirs(_STORE, exist_ok=True)
    return _STORE


def save_upload(data: bytes) -> str:
    """Save uploaded image bytes content-addressed; return '<sha1>.png' id."""
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    # bound the stored size — the field is only sampled at 256px anyway
    im.thumbnail((512, 512), Image.LANCZOS)
    sid = hashlib.sha1(im.tobytes()).hexdigest()[:16] + ".png"
    path = os.path.join(store_dir(), sid)
    if not os.path.exists(path):
        im.save(path, optimize=True)
    return sid


# ------------------------------------------------------------------------ fonts
# label -> filename in C:\Windows\Fonts, plus a palette of characters that read
# well as silhouettes (shown in the UI so dingbat letter->symbol maps are usable).
_FONT_DIR = r"C:\Windows\Fonts"
_FONT_CANDIDATES = [
    ("emoji",    "seguiemj.ttf", "🐱🐶🦊🐻🐸🦉🐍🐙🦋🌳🍄👁💀🧠✋🌛☀🔥💧⚡🌊🪐"),
    ("symbols",  "seguisym.ttf", "☠☯☢☣✶✷❀✿❁♛♞⚙⚛⌘⚓⚔⚕⚖⚜∞◈"),
    ("icons",    "SegoeIcons.ttf", ""),
    ("wingdings", "wingding.ttf", "NÜabcdefgWXYZ✆✎✂♥♦♣♠"),
    ("webdings",  "webdings.ttf", "A!\"#$%&'()*+,-./0123456789"),
    ("impact",   "impact.ttf", "AXO@&#8§"),
    ("arialblk", "ariblk.ttf", "AXO@&8"),
]


def fonts() -> list:
    """Available stamp fonts: [{id, label, samples}] (only ones present)."""
    out = []
    for label, fname, samples in _FONT_CANDIDATES:
        if os.path.exists(os.path.join(_FONT_DIR, fname)):
            out.append({"id": label, "label": label, "samples": samples})
    return out


_FONT_FILE = {label: fname for label, fname, _ in _FONT_CANDIDATES}


def _font_path(font_id: str) -> str:
    fname = _FONT_FILE.get(font_id, "seguiemj.ttf")
    return os.path.join(_FONT_DIR, fname)


# ------------------------------------------------------------- source -> fields
def _to_square(a: np.ndarray, res: int) -> np.ndarray:
    """Center a HxW field into a res x res zero-padded square (keeps aspect)."""
    h, w = a.shape
    s = max(h, w)
    sq = np.zeros((s, s), dtype=np.float64)
    y0, x0 = (s - h) // 2, (s - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = a
    img = Image.fromarray((np.clip(sq, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.asarray(img.resize((res, res), Image.LANCZOS), dtype=np.float64) / 255.0


def glyph_fields(char: str, font_id: str, res: int = 256):
    """Render `char` with the chosen font -> (ink, lum) square fields in [0,1]."""
    char = char or "?"
    color = font_id == "emoji"
    font = ImageFont.truetype(_font_path(font_id), 220)
    box = Image.new("RGBA", (320, 320), (0, 0, 0, 0))
    d = ImageDraw.Draw(box)
    try:
        bb = d.textbbox((0, 0), char, font=font, embedded_color=color)
    except TypeError:
        bb = d.textbbox((0, 0), char, font=font)
        color = False
    w, h = max(bb[2] - bb[0], 1), max(bb[3] - bb[1], 1)
    ox, oy = -bb[0] + (320 - w) // 2, -bb[1] + (320 - h) // 2
    d.text((ox, oy), char, font=font, fill=(255, 255, 255, 255), embedded_color=color)
    arr = np.asarray(box, dtype=np.float64) / 255.0
    alpha = arr[..., 3]
    ys, xs = np.where(alpha > 0.02)
    if len(xs) == 0:                     # blank glyph -> tiny dot so it's visible
        alpha[160, 160] = 1.0
        ys, xs = np.where(alpha > 0.0)
    crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    ink = alpha[crop]
    lum = (arr[..., :3] @ np.array([0.299, 0.587, 0.114]))[crop]
    lum = np.where(ink > 0.02, lum, 0.0)
    return _to_square(ink, res), _to_square(lum, res)


def image_fields(image_id: str, res: int = 256):
    """Load an uploaded stamp -> (ink, lum). Alpha (if any) is the mask; a fully
    opaque image fills its square. Luminance carves the interior relief."""
    path = os.path.join(store_dir(), os.path.basename(image_id))
    if not os.path.exists(path):
        return np.zeros((res, res)), np.zeros((res, res))
    im = Image.open(path).convert("RGBA")
    im.thumbnail((res, res), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float64) / 255.0
    alpha = arr[..., 3]
    lum = arr[..., :3] @ np.array([0.299, 0.587, 0.114])
    if alpha.min() > 0.98:               # opaque photo: whole frame is the stamp
        ink = np.ones_like(alpha)
    else:
        ink = alpha
        lum = np.where(alpha > 0.02, lum, 0.0)
    return _to_square(ink, res), _to_square(lum, res)


def source_fields(params: dict, res: int = 256):
    if params.get("stamp_source", "glyph") == "image" and params.get("stamp_image"):
        return image_fields(params["stamp_image"], res)
    return glyph_fields(params.get("stamp_glyph", "☺"), params.get("stamp_font", "emoji"), res)


# --------------------------------------------------- transforms + compositing
def _shape(ink: np.ndarray, lum: np.ndarray, params: dict):
    """Apply threshold / edge-outline / invert to the source ink (and lum)."""
    thr = float(params.get("stamp_threshold", 0.0))
    if thr > 0:
        w = 0.06
        ink = np.clip((ink - (thr - w)) / (2 * w), 0.0, 1.0)
    eo = float(params.get("stamp_edge_only", 0.0))
    if eo > 0:
        gy, gx = np.gradient(ink)
        edge = np.clip(np.sqrt(gx * gx + gy * gy) * (3.0 + 12.0 * eo), 0.0, 1.0)
        ink = ink * (1.0 - eo) + edge * eo
    if params.get("stamp_invert", False):
        ink = 1.0 - ink
        lum = 1.0 - lum
    return ink, lum


def _place(dink, dlum, ink, lum, cx, cy, scale_px, angle, flipx, flipy):
    """Inverse-sample one instance into (dink,dlum), max-composited, seam-wrapped."""
    size = dink.shape[0]
    res = ink.shape[0]
    R = scale_px * 0.75 + 2.0            # window half-extent (diagonal cover)
    x0, x1 = int(np.floor(cx - R)), int(np.ceil(cx + R))
    y0, y1 = int(np.floor(cy - R)), int(np.ceil(cy + R))
    gx = np.arange(x0, x1)[None, :] - cx
    gy = np.arange(y0, y1)[:, None] - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    lx = ca * gx - sa * gy               # rotate into stamp frame
    ly = sa * gx + ca * gy
    u = lx / scale_px + 0.5              # -> [0,1] across the stamp square
    v = ly / scale_px + 0.5
    if flipx:
        u = 1.0 - u
    if flipy:
        v = 1.0 - v
    sx = u * (res - 1)
    sy = v * (res - 1)
    inside = (sx >= 0) & (sx <= res - 1) & (sy >= 0) & (sy <= res - 1)
    sxi = np.clip(sx, 0, res - 1)
    syi = np.clip(sy, 0, res - 1)
    x0i = np.floor(sxi).astype(int); y0i = np.floor(syi).astype(int)
    x1i = np.minimum(x0i + 1, res - 1); y1i = np.minimum(y0i + 1, res - 1)
    fx = sxi - x0i; fy = syi - y0i

    def samp(fld):
        return (fld[y0i, x0i] * (1 - fx) * (1 - fy) + fld[y0i, x1i] * fx * (1 - fy)
                + fld[y1i, x0i] * (1 - fx) * fy + fld[y1i, x1i] * fx * fy) * inside

    si = samp(ink)
    sl = samp(lum)
    rows = np.arange(y0, y1) % size
    cols = np.arange(x0, x1) % size
    idx = np.ix_(rows, cols)
    prev = dink[idx]
    take = si > prev                     # winner-takes interior detail
    dink[idx] = np.where(take, si, prev)
    dlum[idx] = np.where(take, sl, dlum[idx])


def build_fields(params: dict, size: int, rng):
    """Composite the stamp across the chosen arrangement -> (ink, lum) at size."""
    ink0, lum0 = _shape(*source_fields(params), params)
    dink = np.zeros((size, size))
    dlum = np.zeros((size, size))

    arr = params.get("stamp_arrange", "single")
    base_scale = float(params.get("stamp_scale", 0.6))
    ang0 = float(params.get("stamp_rotate", 0.0))
    rjit = float(params.get("stamp_rot_jitter", 0.0))
    sjit = float(params.get("stamp_scale_jitter", 0.0))
    pjit = float(params.get("stamp_pos_jitter", 0.0))
    fx = bool(params.get("stamp_flip_x", False))
    fy = bool(params.get("stamp_flip_y", False))

    def inst(cx, cy, cell):
        a = ang0 + (rng.uniform(-1, 1) * rjit if rjit else 0.0)
        s = cell * base_scale * (1.0 + (rng.uniform(-1, 1) * sjit if sjit else 0.0))
        if pjit:
            cx += rng.uniform(-1, 1) * pjit * cell
            cy += rng.uniform(-1, 1) * pjit * cell
        _place(dink, dlum, ink0, lum0, cx, cy, max(s, 4.0), a, fx, fy)

    if arr == "single":
        off = params.get("stamp_offset", [0.0, 0.0])
        inst(size * (0.5 + off[0]), size * (0.5 + off[1]), size)
    elif arr in ("grid", "brick"):
        n = max(int(params.get("stamp_grid", 3)), 1)
        if arr == "brick" and n % 2:
            n += 1                       # even rows so the half-offset wraps cleanly
        cell = size / n
        for i in range(n):
            row_off = 0.5 * cell if (arr == "brick" and i % 2) else 0.0
            for j in range(n):
                inst((j + 0.5) * cell + row_off, (i + 0.5) * cell, cell)
    elif arr == "scatter":
        count = max(int(params.get("stamp_count", 6)), 1)
        cell = size / max(np.sqrt(count), 1.0)
        for _ in range(count):
            inst(rng.uniform(0, size), rng.uniform(0, size), cell)

    return np.clip(dink, 0, 1), np.clip(dlum, 0, 1)
