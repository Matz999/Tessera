from . import (emblem, girih, goo, mandala, quasicrystal, reaction_diffusion,
               stamp, voronoi, wildcard)

# truchet, face, silicon, pcb were pruned from the lineup and folded into the
# `wildcard` grab-bag family (tessera/families/wildcard.py).
FAMILIES = {m.FAMILY: m for m in (
    emblem, quasicrystal, girih, reaction_diffusion, voronoi, mandala,
    goo, stamp, wildcard,
)}
