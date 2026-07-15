"""Symmetry folds: kaleidoscope any field into C-n / D-n symmetry."""
import numpy as np

from .util import bilinear_wrap


def kaleido(field: np.ndarray, n: int, mirror: bool = True,
            rot: float = 0.0) -> np.ndarray:
    """Fold a field into n-fold rotational (C-n) or dihedral (D-n) symmetry
    about the tile center. `rot` rotates the whole result."""
    size = field.shape[0]
    v = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    X, Y = np.meshgrid(v, v)
    r = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(Y, X) + rot

    sector = 2.0 * np.pi / n
    t = np.mod(theta, sector)
    if mirror:
        t = np.abs(t - sector / 2.0)  # fold within sector -> D-n

    xs = (r * np.cos(t) * 0.5 + 0.5) * size - 0.5
    ys = (r * np.sin(t) * 0.5 + 0.5) * size - 0.5
    return bilinear_wrap(field, xs, ys)


def radial_blend(field: np.ndarray, n: int, rot: float = 0.0) -> np.ndarray:
    """Average of n rotated copies — softer symmetry (moire-free)."""
    size = field.shape[0]
    v = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    X, Y = np.meshgrid(v, v)
    acc = np.zeros_like(field)
    for k in range(n):
        a = rot + k * 2.0 * np.pi / n
        ca, sa = np.cos(a), np.sin(a)
        xs = ((X * ca - Y * sa) * 0.5 + 0.5) * size - 0.5
        ys = ((X * sa + Y * ca) * 0.5 + 0.5) * size - 0.5
        acc += bilinear_wrap(field, xs, ys)
    return acc / n
