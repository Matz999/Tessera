"""Evolutionary breeding: treat each tile's params as a genome and produce
offspring by crossover + mutation of selected parents.

A tile is (family, seed, params). The *params* are the heritable genome; the
seed only seeds generate()'s internal noise, so each offspring gets a fresh
seed (a new composition) while inheriting the parents' *style* (material,
lighting, weathering, and the family's structural dials). Breeding happens
within a gene pool — same family, and for `wildcard` the same `_sub` generator
(a truchet genome is meaningless to silicon). Mixed selections are grouped and
bred pool-by-pool, offspring spread round-robin across pools.

Ranges/choices come from the same DIALS specs the dial panel uses, so mutation
stays in-gamut and categorical genes (material/ramp/…) resample legally.
"""
import numpy as np

from .core.emissive import EMISSION_COLORS, EMISSION_SOURCES
from .core.palette import COLOR_RAMPS, GRAY_RAMPS
from .families import FAMILIES
from .families.common import DIALS as SHARED_DIALS
from .families.common import MATERIALS
from .families.wildcard import _SUBS as WILD_SUBS
from .render import next_seed, render_genome


def _mod_for(name):
    return FAMILIES.get(name) or WILD_SUBS.get(name)


def param_spec(family: str):
    """(numeric {name: (lo,hi,step)}, choice {name: [options]}) for a family."""
    num = {n: (d["lo"], d["hi"], d["step"]) for n, d in SHARED_DIALS.items()}
    ch = {}
    for name, d in getattr(_mod_for(family), "DIALS", {}).items():
        if "choices" in d:
            ch[name] = list(d["choices"])
        elif d.get("bool"):
            ch[name] = [True, False]
        elif "lo" in d:
            num[name] = (d["lo"], d["hi"], d["step"])
    ch.setdefault("material", list(MATERIALS))
    ch.setdefault("ramp", COLOR_RAMPS)
    ch.setdefault("gray_ramp", GRAY_RAMPS)
    ch.setdefault("emission_color", list(EMISSION_COLORS))
    ch.setdefault("emission_source", list(EMISSION_SOURCES))
    return num, ch


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def crossover(parents: list, num: dict, rng, blend: float) -> dict:
    """Uniform crossover over the union of genes; numeric genes may interpolate
    between two parents (prob `blend`) instead of inheriting one wholesale."""
    child = dict(parents[0])                    # base carries freeform genes
    keys = set().union(*(p.keys() for p in parents))
    for k in keys:
        vals = [p[k] for p in parents if k in p]
        if not vals:
            continue
        if (k in num and len(vals) >= 2 and rng.random() < blend
                and all(_is_num(v) for v in vals)):
            a, b = int(rng.integers(len(vals))), int(rng.integers(len(vals)))
            t = rng.random()
            child[k] = vals[a] * (1.0 - t) + vals[b] * t
        else:
            child[k] = vals[int(rng.integers(len(vals)))]
    return child


def mutate(params: dict, num: dict, ch: dict, rng, m: float) -> dict:
    """Perturb genes: numeric jitter (gaussian, clamped to range), categorical
    resample, bool flip. `m` in [0,1] scales both size and rate."""
    out = dict(params)
    for k, (lo, hi, step) in num.items():
        if k not in out or not _is_num(out[k]):
            continue
        if rng.random() < 0.85:
            v = out[k] + rng.normal(0.0, m * (hi - lo))
            v = min(max(v, lo), hi)
            if float(step).is_integer() and step >= 1:
                out[k] = int(round(v))
            else:
                out[k] = round(float(v), 3)
    for k, opts in ch.items():
        if k not in out or not opts:
            continue
        if all(isinstance(o, bool) for o in opts):        # bool gene
            if rng.random() < m * 0.4:
                out[k] = bool(rng.integers(2))
        elif rng.random() < m * 0.6:                      # categorical gene
            out[k] = opts[int(rng.integers(len(opts)))]   # keep native type
    return out


def breed_batch(parents: list, count: int, size: int, gray: bool, ss: int,
                mutation: float, blend: float, outdir: str) -> list:
    """parents: [{family, seed, params}]. Group into gene pools, breed, render."""
    pools = {}
    for p in parents:
        fam = p.get("family")
        params = p.get("params") or {}
        if fam not in FAMILIES:
            continue
        pools.setdefault(fam, []).append(params)
    if not pools:
        raise ValueError("no valid parents (unknown family?)")

    rng = np.random.default_rng()
    keys = list(pools)
    seed_ctr = {}
    entries = []
    for i in range(count):
        fam = keys[i % len(keys)]
        pool = pools[fam]
        num, ch = param_spec(fam)
        child = mutate(crossover(pool, num, rng, blend), num, ch, rng, mutation)
        if fam not in seed_ctr:
            seed_ctr[fam] = next_seed(fam, outdir)
        seed = seed_ctr[fam]
        seed_ctr[fam] += 1
        entries.append(render_genome(fam, child, seed, size, gray, outdir, ss=ss))
    return entries
