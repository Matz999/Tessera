# Tessera

Deterministic procedural tile generator — seamless tiles from seeds + dials,
with a full material-realism stack (relief, lighting, weathering, emissive).

Pipeline: deterministic generators -> batch PNGs -> contact-sheet curation ->
`blessed.json`. Pure numpy + Pillow. Every tile is reproducible: seed + params
are embedded in the PNG (tEXt) and in `library/manifest.json`.

```
pip install -r requirements.txt
```

## Live UI (the everyday loop)

Run the server from the project root (the folder containing `tessera/` and
`library/`), then open the printed URL in a browser:

```
cd procedural_tiles
python -m tessera.serve                    # -> http://127.0.0.1:8000/
python -m tessera.serve --port 9000        # different port
python -m tessera.serve --out experiments  # serve a different tile dir
```

In the browser:

1. **Generate** — in the blue Generate bar pick a family, count (1-16), size,
   optional gray / seed (blank = auto-continue after the family's highest),
   hit **Generate**. The status shows elapsed time (most families run 2-5s
   per 512px tile; reaction_diffusion ~50s). New tiles appear at the top of
   the grid with a yellow NEW badge.
2. **Cull** — hover a tile and click the red **×** to delete it (removes the
   PNG + its normal map, updates the manifest). No confirmation: tiles are
   cheap and reproducible from their seed.
3. **Bless** — click a tile to toggle the green check, then **Save
   blessed.json** writes `library/blessed.json` directly.

Ctrl+C in the terminal stops the server.

## Quick batches from the CLI

```
python -m tessera.render --families truchet --count 4          # 4 new truchet tiles
python -m tessera.render --families emblem,voronoi --count 3   # 3 new per family
```

Seeds auto-continue after each family's highest existing tile, so every run
produces genuinely new tiles — no seed bookkeeping. The manifest updates
incrementally (only new PNGs are read), so a 3-tile batch takes seconds even
with a large library. `library/contact.html` also works standalone over
`file://` (double-click) — same page, just without the Generate bar; Export
downloads blessed.json instead of saving it.

## Full sweeps & other modes

```
python -m tessera.render --all --count 160 --size 512      # everything
python -m tessera.render --families emblem --count 20 --gray   # grey-first QA
python -m tessera.render --seed 5000 --families girih --count 10  # explicit seed range
python -m tessera.render --manifest-only                   # incremental manifest refresh
python -m tessera.render --manifest-only --rescan          # full rebuild from PNG metadata
python -m tessera.render --normals                         # export .n.png normal maps
```

Output: `library/<family>/<seed>_<paramhash>.png`. Re-running an explicit
`--seed` range skips files that already exist, so big sweeps are resumable.
`--out <dir>` renders to a different directory (own manifest + contact sheet) —
handy for throwaway experiments that shouldn't touch the library.

## Curate

Open `library/contact.html` in a browser. Click tiles to bless (green check),
filter by family / blessed state, then **Export blessed.json** and save it as
`library/blessed.json`. Blessed state persists in localStorage between visits.

## Dials

Every tile samples ~35 shading dials (stored in its params — hover a tile's
label in the contact sheet to see them all). In the served UI, the **Dials**
button opens a panel with a slider + numeric input for every dial (plus
material/palette dropdowns): each starts on *auto* (swept per tile); touch it
to pin it — pinned dials go orange and apply to every generate until you hit
its x. The **raw overrides** box takes a JSON object for anything else (e.g.
family-specific params) and wins ties; the CLI equivalent is
`--overrides '{...}'`. The dial groups (`tessera/families/common.py`,
`sample_common` for sweep ranges, `DIALS` for UI ranges):

- **key light** — `light_angle`, `light_elev`, `key_warmth` (-1 cool .. +1
  tungsten), `diffuse_boost`, `ambient_boost`
- **fill + rim** — `fill_strength/angle/elev/warmth`, `rim_strength`
- **cast shadows** — `shadow_strength`, `shadow_soft`, `shadow_height`
  (how deep the carving reads)
- **specular** — `spec_tint` (0 white .. 1 metal), `fresnel`, `spec_boost`,
  `spec_pow_boost`, `rough_amount`/`rough_freq` (matte-vs-gloss patches)
- **curvature wear** — `edge_wear` (worn bright edges), `grime` (dark
  creases), `wear_radius`, `wear_spec` (polished edges)
- **emissive** — self-lit glow that feeds bloom. `emission` (master, 0 =
  off), `emission_color` (neon/lava/plasma/… palette), `emission_source`
  (which field glows: `tone`/`height`/`crevice`/`ridge`/`edge`/`invert_tone`),
  `emission_thresh`/`emission_sharp` (glow cutoff), `emission_gamma`,
  `emission_white` (hot cores blow to white), `emission_flicker`/`_freq`
  (uneven glow), `bloom_emissive` (how hard emitters bloom), `bloom_radius`
- **weathering** — corrosion & decay that reads the tile's own geometry
  (rust/verdigris pool in recesses, paint flakes off raised faces). `weather_rust`
  / `weather_verdigris` (+ `_freq`), `weather_chips` (flaking paint → exposed
  primer) / `weather_chip_freq` / `weather_chip_depth`, `weather_drips`
  (soot/water streaks) / `weather_drip_len` / `_freq`, `weather_lichen` /
  `_freq`, `weather_cavity_bias` (how strongly corrosion favors recesses).
  Each effect is off at 0 and sampled mostly-off, so ~half of tiles stay clean.
- **surface** — `ao_strength`, `grain_boost`, `relief`, `material`, `ramp`
- **finish** — `film_grain`, `vignette`, `bloom`, `bloom_thresh`, `chroma`

Renders are supersampled 2x by default for anti-aliasing (`--ss` / the AA
select in the UI; use 1x for quick drafts or reaction_diffusion, whose sim
cost grows fast with resolution).

## Families

emblem, quasicrystal, truchet, girih, reaction_diffusion (slow — real
Gray-Scott sim), voronoi, mandala, goo (metaball neon slime — glows via the
emissive channel), pcb (routed circuit board — copper traces, ICs, vias,
silkscreen; LEDs drive the glow), silicon (die shot — packed memory/logic/pad
blocks, bus channels, cyclic thin-film interference color), face (carved
ceremonial mask / idol — bilaterally-symmetric brow, eyes, nose, mouth sculpted
from smooth Gaussian bumps on a raised plate; eyes drive the glow, so pin
`emission` for a glowing-eyed idol). Each lives in
`tessera/families/<name>.py` with `sample_params()` (the sweep) and
`generate(seed, params, size, gray)`.

The dial panel is **family-aware**: on top of the shared render dials it shows a
`family: <name>` group with that family's own params (pcb `soldermask`/`finish`/
`grid`/`traces`/…, voronoi `sites`/`grout`/…, goo `blobs`/`dome`/…, etc.),
rebuilt whenever you switch the generate family. Family pins reset on switch;
shared pins persist. Each family declares its panel dials in a module-level
`DIALS` dict. A few procedural params (e.g. mandala's per-ring list) aren't
expressible as a single slider — pin those via the raw-overrides box.

Shared realism stack in `tessera/core/`: fBm/domain-warp noise -> heightfield ->
normals + 3-light rig + cast shadows + AO + metal specular (`relief.py`),
emissive glow (`emissive.py`), corrosion/weathering (`weather.py`), grain
overlays (`grain.py`), palette ramps as data (`palette.py`), kaleidoscope
symmetry (`symmetry.py`), film-grain/vignette/bloom finish (`finish.py`).

Adding a family: write the module with `FAMILY`, `sample_params`, `generate`
(author a heightfield + tone field, call `common.render_material`), then
register it in `tessera/families/__init__.py`. Multi-material families (pcb) author
their own albedo and pass `albedo=` to `render_material` to bypass the ramp;
`tessera/core/draw.py` has seamless tileable primitives (`stamp_capsule/disk/rect`
for hot loops, pure `disk/ring/capsule/rect` for compositing).

Note: PNG metadata keys (`cram_family`, `cram_seed`, `cram_params`) and the
contact sheet's localStorage key keep their original `cram` names — the
existing 1000+ library tiles were written with them, so they stay for
compatibility.
