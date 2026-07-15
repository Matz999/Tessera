"""Tileable primitive rasterizers (numpy, seamless).

Two flavors:
  * pure `disk/ring/capsule/rect` return a full-field mask (handy for
    compositing/subtraction; use for a handful of primitives).
  * fast `stamp_*` composite one primitive into a field via np.maximum, only
    touching its bounding-box window — O(feature area), not O(tile area). Use
    these in hot loops (hundreds of traces/pads/pins).

Both tile seamlessly. The pure versions measure each pixel's offset by the
nearest toroidal image; the stamp versions rasterize in absolute coordinates
over the bbox and scatter with wrapped (`% size`) indices, so a primitive
crossing a seam writes to both sides. Valid while a feature spans < size.
"""
import numpy as np


def _wrap_delta(a: np.ndarray, size: int) -> np.ndarray:
    """Signed nearest-image difference in [-size/2, size/2)."""
    return (a + size * 0.5) % size - size * 0.5


def coords(size: int):
    """Pixel coordinate grids X, Y in [0, size)."""
    v = np.arange(size, dtype=np.float64)
    return np.meshgrid(v, v)


# --- pure full-field masks ---------------------------------------------------

def disk(X, Y, size, cx, cy, r, soft=1.0) -> np.ndarray:
    dx = _wrap_delta(X - cx, size)
    dy = _wrap_delta(Y - cy, size)
    d = np.sqrt(dx * dx + dy * dy)
    return np.clip((r - d) / max(soft, 1e-6), 0.0, 1.0)


def ring(X, Y, size, cx, cy, r_out, r_in, soft=1.0) -> np.ndarray:
    return np.clip(disk(X, Y, size, cx, cy, r_out, soft)
                   - disk(X, Y, size, cx, cy, r_in, soft), 0.0, 1.0)


def capsule(X, Y, size, ax, ay, bx, by, width, soft=1.0) -> np.ndarray:
    abx = _wrap_delta(bx - ax, size)
    aby = _wrap_delta(by - ay, size)
    dpx = _wrap_delta(X - ax, size)
    dpy = _wrap_delta(Y - ay, size)
    ll = abx * abx + aby * aby
    t = np.clip((dpx * abx + dpy * aby) / (ll + 1e-9), 0.0, 1.0)
    ex = dpx - t * abx
    ey = dpy - t * aby
    d = np.sqrt(ex * ex + ey * ey)
    return np.clip((width - d) / max(soft, 1e-6), 0.0, 1.0)


def rect(X, Y, size, cx, cy, hw, hh, soft=1.0) -> np.ndarray:
    dx = np.abs(_wrap_delta(X - cx, size))
    dy = np.abs(_wrap_delta(Y - cy, size))
    mx = np.clip((hw - dx) / max(soft, 1e-6), 0.0, 1.0)
    my = np.clip((hh - dy) / max(soft, 1e-6), 0.0, 1.0)
    return mx * my


def over(dst: np.ndarray, mask: np.ndarray, value):
    """Composite dst = lerp(dst, value, mask). Scalar field or HxWx3 color."""
    if dst.ndim == 3:
        m = mask[..., None]
        return dst * (1.0 - m) + np.asarray(value, dtype=np.float64) * m
    return dst * (1.0 - mask) + value * mask


# --- fast windowed stamps (max-composite into `field`) -----------------------

def _max_into(field, y0, y1, x0, x1, m):
    size = field.shape[0]
    rows = np.arange(y0, y1) % size
    cols = np.arange(x0, x1) % size
    idx = np.ix_(rows, cols)
    field[idx] = np.maximum(field[idx], m)


def stamp_disk(field, cx, cy, r, soft=1.0):
    R = r + soft + 1.0
    x0, x1 = int(np.floor(cx - R)), int(np.ceil(cx + R))
    y0, y1 = int(np.floor(cy - R)), int(np.ceil(cy + R))
    xs = np.arange(x0, x1)[None, :] - cx
    ys = np.arange(y0, y1)[:, None] - cy
    d = np.sqrt(xs * xs + ys * ys)
    _max_into(field, y0, y1, x0, x1, np.clip((r - d) / max(soft, 1e-6), 0, 1))


def stamp_rect(field, cx, cy, hw, hh, soft=1.0):
    x0, x1 = int(np.floor(cx - hw - soft - 1)), int(np.ceil(cx + hw + soft + 1))
    y0, y1 = int(np.floor(cy - hh - soft - 1)), int(np.ceil(cy + hh + soft + 1))
    dx = np.abs(np.arange(x0, x1)[None, :] - cx)
    dy = np.abs(np.arange(y0, y1)[:, None] - cy)
    mx = np.clip((hw - dx) / max(soft, 1e-6), 0, 1)
    my = np.clip((hh - dy) / max(soft, 1e-6), 0, 1)
    _max_into(field, y0, y1, x0, x1, mx * my)


def stamp_capsule(field, ax, ay, bx, by, width, soft=1.0):
    size = field.shape[0]
    bx = ax + _wrap_delta(bx - ax, size)   # nearest image of the far endpoint
    by = ay + _wrap_delta(by - ay, size)
    R = width + soft + 1.0
    x0, x1 = int(np.floor(min(ax, bx) - R)), int(np.ceil(max(ax, bx) + R))
    y0, y1 = int(np.floor(min(ay, by) - R)), int(np.ceil(max(ay, by) + R))
    dpx = np.arange(x0, x1)[None, :] - ax
    dpy = np.arange(y0, y1)[:, None] - ay
    abx, aby = bx - ax, by - ay
    t = np.clip((dpx * abx + dpy * aby) / (abx * abx + aby * aby + 1e-9), 0, 1)
    ex = dpx - t * abx
    ey = dpy - t * aby
    d = np.sqrt(ex * ex + ey * ey)
    _max_into(field, y0, y1, x0, x1, np.clip((width - d) / max(soft, 1e-6), 0, 1))
