"""Family: wildcard — one procedural that FUSES truchet + face + silicon + pcb.

Not a selector (roll one) but a blender: each generator contributes its own
pre-render fields (height / albedo / spec / emit), and the tile is a smooth
regional mix of all the active ones — truchet weave flowing into silicon
blocks into a carved face into PCB traces, all on a single surface under one
lighting pass. Region ownership comes from softmax over per-generator
large-scale noise (each dominates its own smooth zones); `mix_<name>` weights
bias or disable a generator, `mix_sharp` sets how hard the borders are,
`mix_scale` sets region size.

Each generator's structural params ride in a nested `p_<name>` dict; the shared
render dials (light / material / ramp / weathering / finish) are the tile's own
and apply once to the fused result.
"""
import numpy as np

from ..core.noise import fbm
from ..core.util import downscale, norm01, rng_for
from . import face, pcb, silicon, truchet
from .common import render_material, sample_common

# fusing 2-4 full generators is expensive; cap the internal work resolution and
# upscale the fused result (the mix is soft by nature, so this is invisible).
_MAX_RES = 512

FAMILY = "wildcard"

_SUBS = {m.FAMILY: m for m in (truchet, face, silicon, pcb)}
_NAMES = list(_SUBS)

DIALS = {
    "mix_truchet": {"lo": 0.0, "hi": 1.0, "step": 0.02},
    "mix_face":    {"lo": 0.0, "hi": 1.0, "step": 0.02},
    "mix_silicon": {"lo": 0.0, "hi": 1.0, "step": 0.02},
    "mix_pcb":     {"lo": 0.0, "hi": 1.0, "step": 0.02},
    "mix_sharp":   {"lo": 1.0, "hi": 8.0, "step": 0.1},   # region border hardness
    "mix_scale":   {"lo": 1.0, "hi": 6.0, "step": 0.5},   # region size (noise freq)
}

_SHARED = None


def _shared_keys():
    """Keys owned by the fused tile (shared render dials) — stripped from each
    sub's structural genome so a generator can't override the mix's lighting."""
    global _SHARED
    if _SHARED is None:
        _SHARED = set(sample_common(np.random.default_rng(0))) | {"material", "ramp"}
    return _SHARED


def sample_params(rng) -> dict:
    p = sample_common(rng)
    u = rng.uniform
    sk = _shared_keys()
    for name, mod in _SUBS.items():
        sp = mod.sample_params(rng)
        p["p_" + name] = {k: v for k, v in sp.items() if k not in sk}
        # mostly on, sometimes off or partial -> varied fusions
        p["mix_" + name] = round(float(rng.choice([1.0, 1.0, 1.0, 0.0, u(0.4, 1.0)])), 3)
    if sum(p["mix_" + n] > 0 for n in _NAMES) < 2:      # guarantee a real mix
        for n in list(rng.permutation(_NAMES))[:2]:
            p["mix_" + n] = 1.0
    p["mix_sharp"] = round(float(u(1.5, 4.0)), 3)
    p["mix_scale"] = round(float(rng.choice([1.0, 1.5, 2.0, 3.0])), 3)
    p["material"] = str(rng.choice(["stone", "bronze", "ceramic"]))
    p["ramp"] = str(rng.choice(["gold_indigo", "lapis_gold", "terracotta", "celadon",
                                "obsidian", "biolum", "iridescent", "amethyst"]))
    # glow only matters where pcb LEDs / face eyes exist (guarded in generate);
    # keep it gentle so a lit fusion doesn't wash out
    p["emission"] = round(float(rng.choice([0.0, 0.0, 0.0, u(0.5, 1.0)])), 3)
    p["emission_white"] = round(float(u(0.05, 0.2)), 3)
    return p


def generate(seed: int, params: dict, size: int = 512, gray: bool = False) -> np.ndarray:
    out_size = size
    size = min(size, _MAX_RES)          # bound the multi-generator cost
    rng = rng_for(seed)
    active = [(n, _SUBS[n], float(params.get("mix_" + n, 1.0)))
              for n in _NAMES if float(params.get("mix_" + n, 1.0)) > 0]
    if not active:
        active = [("truchet", _SUBS["truchet"], 1.0)]
    sharp = float(params.get("mix_sharp", 3.0))
    scale = max(int(float(params.get("mix_scale", 3.0))), 1)

    # region ownership: softmax over each generator's large-scale noise + log-bias
    logits = []
    for i, (n, mod, w) in enumerate(active):
        f = fbm(size, rng_for(seed * 131 + i * 17), octaves=3, freq=scale)
        logits.append(sharp * (norm01(f) - 0.5) + np.log(max(w, 1e-3)))
    L = np.stack(logits, 0)
    L -= L.max(axis=0, keepdims=True)
    W = np.exp(L)
    W /= W.sum(axis=0, keepdims=True)                   # (K, size, size), sums to 1

    H = np.zeros((size, size))
    T = np.zeros((size, size))
    S = np.zeros((size, size))
    E = np.zeros((size, size))
    for k, (n, mod, w) in enumerate(active):
        sp = dict(params)
        sp.update(params.get("p_" + n, {}))             # shared + this sub's structure
        h_i, tone_i, sm_i, emit_i, alb_i = mod.fields(sp, size, gray, rng_for(seed * 911 + k))
        wk = W[k]
        H += wk * norm01(h_i)
        T += wk * norm01(tone_i)
        S += wk * (sm_i if sm_i is not None else 0.5)
        if emit_i is not None and emit_i.max() > 0:
            E += wk * norm01(emit_i)

    # one unified material: every generator's pattern carved into the same
    # substrate. render_material ramps + rank-equalizes the fused tone (full
    # contrast, no blowout); silicon/pcb authored colors give way to the ramp.
    emit_src = norm01(E) if E.max() > 0 else None
    rp = params
    if emit_src is None and float(params.get("emission", 0.0)) > 0:
        rp = {**params, "emission": 0.0}   # no emitter present -> don't glow the
        #                                    dark ground (invert_tone -> white)
    img = render_material(norm01(H), norm01(T), rp, rng, gray,
                          spec_mask=np.clip(S, 0.02, 1.0), ao_radii=(3, 8, 18),
                          emit_source=emit_src)
    return downscale(img, out_size) if out_size != size else img
