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
- **finish** — `film_grain`, `vignette`, `bloom`, `bloom_thresh`, `chroma`,
  `contrast` (S-curve punch, biased on)
- **overlays** — atmospheric / generative / photographic flourishes layered on
  every family (`tessera/core/overlay.py`), each off at 0 and sampled mostly-off:
  `cloud` (+`cloud_freq`, cloudy colour gradient), `bokeh` (+`bokeh_size`,
  defocused orbs), `soft_focus` (+`soft_radius`, dreamy Orton glow), `shapes`
  (+`shape_alpha`, random geometry), `lines` (+`line_alpha`, rational-angle line
  families / hatching), `func_curves` (sine / Lissajous / rose / spiral /
  spirograph / de Jong attractor plotted directly), `glare` (lens flare +
  streak), `twinkle` (star sparkles), `dust` (specks), `scratches` (film
  scratches).
- **grade** — colour-grade + display effects: `duotone` (+`duo_h1`/`duo_h2`
  hues, luminance→two-tone), `posterize` (level quantise), `halftone`
  (+`halftone_cells`, newsprint dots), `aberration` (RGB channel split),
  `scanlines` (+`scanline_freq`, CRT lines). Everything composites through the
  seamless wrapped primitives, so tiles stay tileable.

Renders are supersampled 2x by default for anti-aliasing (`--ss` / the AA
select in the UI; use 1x for quick drafts or reaction_diffusion, whose sim
cost grows fast with resolution).

## Breeding loop (curation that drives generation)

Random sampling has a low hit rate and the 60-dial space is impossible to
hand-drive. Instead, **breed** from what you like: every tile's params are a
genome (seed seeds only the noise), so you select favorites and the next batch
is crossover + mutation of their genomes, not a fresh random sweep. Over a few
generations the library converges toward your eye.

In the served UI, the **Breed** bar:

1. Click **select parents: on**, then click tiles to tag them as parents (they
   get a magenta ♥ border). Click again to unpick; **clear** resets. Toggle the
   mode off to go back to blessing.
2. Set **mutation** (0 = offspring hug the parents / refine; ~0.5 = wander far /
   explore) and **blend** (chance a numeric gene interpolates between two parents
   vs. inheriting one whole), pick a **count**, hit **Breed →**. Offspring appear
   at the top with NEW badges; select the best of them as the next generation.

Breeding happens **within a gene pool** — same family, and for `wildcard` the
same `_sub` generator (a truchet genome is meaningless to silicon). A mixed
selection is grouped and bred pool-by-pool, offspring spread round-robin. A
single parent breeds asexually (mutation only — "give me variations of this
one"). Ranges/choices come from the same DIALS the panel uses, so mutation stays
in-gamut. The genetics live in `tessera/breed.py`; API: `POST /api/breed`
`{parents:[{family,seed,params}], count, mutation, blend, size, gray, ss}`.

## Families

emblem, quasicrystal, girih, reaction_diffusion (slow — real Gray-Scott sim),
voronoi, mandala, goo (metaball neon slime — glows via the emissive channel),
stamp (emboss a **font glyph or an uploaded image** into the tile — carved
stone reliefs, stamped metal plaques, glowing neon emblems, repeating motifs;
see below), and wildcard (a **fusion** of the pruned generators — `truchet`, `face`,
`silicon`, `pcb` — carved into one material; see below). Each lives in
`tessera/families/<name>.py` with `sample_params()` (the sweep) and
`generate(seed, params, size, gray)`.

### The wildcard family (fused generators)

Not a selector — a blender. Each of truchet / face / silicon / pcb exposes its
pre-lighting `fields()` (height, tone, spec, glow); wildcard assigns smooth
regions via softmax over per-generator large-scale noise (each dominates its own
zones, cross-fading at the borders), fuses the height + tone, and runs **one**
lighting pass — so a tile is a single carved substrate where truchet's labyrinth
weave flows into silicon's die blocks into a face emboss into PCB traces (LEDs /
eyes still glow). The authored silicon/pcb colors give way to the tile's own
material + ramp, so it reads as one coherent panel, not a four-image collage.
Dials: `mix_truchet` / `mix_face` / `mix_silicon` / `mix_pcb` (bias 0 = off),
`mix_sharp` (border hardness), `mix_scale` (region size) — all breedable, so you
can evolve toward "more silicon, less truchet". Fusing 2–4 full generators is
costly, so the mixer caps its internal work at 512px (`_MAX_RES`) and renders
there **natively** — no supersample, no upscale — so edges stay hard/aliased and
the detail is dense, not blurred (~12s/tile). It doesn't benefit from AA, so
render at 512px. For hard tiles larger than 512, raise `_MAX_RES` (cost grows
with area).

### The stamp family (glyphs + your own images)

Pick `stamp` in the Generate bar and a **Stamp** strip appears:

- **glyph** — choose a font (emoji 🐱💀🌳, symbols ☠☯, Wingdings/Webdings
  silhouettes, Segoe icons, Impact letters) and type/paste a character, or click
  one from the sample palette. Fully reproducible from `(stamp_glyph, stamp_font)`.
- **image** — upload any picture; it's stored content-addressed under
  `library/_stamps/<hash>.png` and embossed into the tile (alpha is respected,
  so transparent logos cut cleanly; opaque photos become a carved relief).

The shape is read into the shared material stack as *geometry* (relief lighting,
cast shadow, AO carry it — it reads as carved, not a flat sticker). Family dials:
`stamp_arrange` (single / grid / brick / scatter), `stamp_scale`, `stamp_grid` /
`stamp_count`, `stamp_rotate` (+ jitters), `stamp_emboss` (signed: raised vs
engraved), `stamp_bevel`, `stamp_lum_detail` (how much the image/emoji interior
carves), `stamp_threshold` / `stamp_edge_only` (outline mode) / `stamp_invert`,
`stamp_tone` / `stamp_spec` (inlay tint & gloss), and the ground (`bg_tone`,
`tone_relief`, `bg_tex`, `bg_freq`, `bg_spec`). Pin `emission` for a neon emblem
(the glow follows the outline + bright features). Every arrangement tiles
seamlessly (instances composite with toroidal wrap). API: `POST /api/upload`
`{data}` → `{id}`; `GET /api/fonts` lists installed stamp fonts.

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
