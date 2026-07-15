from . import (emblem, face, girih, goo, mandala, pcb, quasicrystal,
               reaction_diffusion, silicon, truchet, voronoi)

FAMILIES = {m.FAMILY: m for m in (
    emblem, quasicrystal, truchet, girih, reaction_diffusion, voronoi, mandala,
    goo, pcb, silicon, face,
)}
