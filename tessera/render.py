"""Batch renderer.

  python -m tessera.render --families emblem,voronoi --count 32 --size 512 --gray
  python -m tessera.render --all --count 100

Output: library/<family>/<seed>_<paramhash>[.g].png  (params embedded in PNG tEXt)
Deterministic: tile i of a family uses seed = base_seed + i; params drawn from
a param-RNG seeded by the same value, so any tile is reproducible from its name.
"""
import argparse
import json
import os
import re
import time

from .core.util import downscale, param_hash, rng_for, save_tile
from .families import FAMILIES


def next_seed(family: str, outdir: str) -> int:
    """First seed after the highest one already rendered for this family."""
    fam_dir = os.path.join(outdir, family)
    if not os.path.isdir(fam_dir):
        return 1000
    seeds = [int(m.group(1)) for name in os.listdir(fam_dir)
             if name.endswith(".png") and not name.endswith(".n.png")
             and (m := re.match(r"(\d+)_", name))]
    return max(seeds) + 1 if seeds else 1000


def render_batch(family: str, count: int, size: int, gray: bool, base_seed: int,
                 outdir: str, ss: int = 2, overrides: dict = None) -> list:
    """ss: supersample factor (render at size*ss, Lanczos downscale — AA).
    overrides: dict merged over sampled params — pin any dial by hand."""
    mod = FAMILIES[family]
    fam_dir = os.path.join(outdir, family)
    os.makedirs(fam_dir, exist_ok=True)
    ss = max(int(ss), 1)
    entries = []
    t0 = time.time()
    for i in range(count):
        seed = base_seed + i
        params = mod.sample_params(rng_for(seed * 7919 + 13))
        if overrides:
            params.update(overrides)
        h = param_hash(params)
        suffix = ".g" if gray else ""
        name = f"{seed:06d}_{h}{suffix}.png"
        path = os.path.join(fam_dir, name)
        if not os.path.exists(path):
            arr = mod.generate(seed, params, size=size * ss, gray=gray)
            if ss > 1:
                arr = downscale(arr, size)
            save_tile(arr, path, family, seed, params)
        entries.append({"family": family, "seed": seed, "hash": h, "gray": gray,
                        "file": f"{family}/{name}", "params": params})
        if (i + 1) % 8 == 0 or i == count - 1:
            dt = time.time() - t0
            print(f"  {family}: {i + 1}/{count}  ({dt / (i + 1):.2f}s/tile)", flush=True)
    return entries


def render_genome(family: str, params: dict, seed: int, size: int, gray: bool,
                  outdir: str, ss: int = 2) -> dict:
    """Render one tile from an explicit, fully-specified param genome (no
    sample_params) — the breeding path. Same on-disk contract as render_batch."""
    mod = FAMILIES[family]
    params = dict(params)
    ss = max(int(ss), 1)
    h = param_hash(params)
    suffix = ".g" if gray else ""
    name = f"{seed:06d}_{h}{suffix}.png"
    fam_dir = os.path.join(outdir, family)
    os.makedirs(fam_dir, exist_ok=True)
    path = os.path.join(fam_dir, name)
    if not os.path.exists(path):
        arr = mod.generate(seed, params, size=size * ss, gray=gray)
        if ss > 1:
            arr = downscale(arr, size)
        save_tile(arr, path, family, seed, params)
    return {"family": family, "seed": seed, "hash": h, "gray": gray,
            "file": f"{family}/{name}", "params": params}


def render_normals(outdir: str, size: int = 512, ss: int = 2) -> None:
    """Regenerate every manifest tile and save its normal map as <name>.n.png
    (the WebGL relight shader in the app consumes these)."""
    import json as _json
    from .core.util import to_image
    from .families import FAMILIES, common

    ss = max(int(ss), 1)
    with open(os.path.join(outdir, "manifest.json")) as f:
        manifest = _json.load(f)
    todo = [t for t in manifest
            if not os.path.exists(os.path.join(outdir, t["file"][:-4] + ".n.png"))]
    print(f"normal maps: {len(todo)} of {len(manifest)} tiles to do", flush=True)
    t0 = time.time()
    for i, t in enumerate(todo):
        FAMILIES[t["family"]].generate(t["seed"], t["params"], size=size * ss,
                                       gray=t.get("gray", False))
        nmap = common.LAST_NORMALS
        if ss > 1:
            nmap = downscale(nmap, size)
        path = os.path.join(outdir, t["file"][:-4] + ".n.png")
        to_image(nmap).save(path, optimize=True)
        if (i + 1) % 20 == 0 or i == len(todo) - 1:
            dt = time.time() - t0
            print(f"  normals: {i + 1}/{len(todo)}  ({dt / (i + 1):.2f}s/tile)", flush=True)


def write_manifest(outdir: str, rescan: bool = False) -> str:
    """Scan library dir -> manifest.json (params read back from PNG metadata).

    Incremental by default: entries for files already in manifest.json are
    reused (no PNG reopen); entries for deleted files are dropped. `rescan`
    forces re-reading metadata from every PNG."""
    from PIL import Image
    mpath = os.path.join(outdir, "manifest.json")
    known = {}
    if not rescan and os.path.exists(mpath):
        try:
            with open(mpath) as f:
                known = {t["file"]: t for t in json.load(f)}
        except Exception as e:
            print(f"  manifest unreadable, full rescan: {e}")
    tiles = []
    for fam in sorted(os.listdir(outdir)):
        fam_dir = os.path.join(outdir, fam)
        if not os.path.isdir(fam_dir) or fam.startswith("_"):
            continue  # skip asset dirs like _stamps (uploaded stamp images)
        for name in sorted(os.listdir(fam_dir)):
            if not name.endswith(".png") or name.endswith(".n.png"):
                continue
            key = f"{fam}/{name}"
            if key in known:
                tiles.append(known[key])
                continue
            path = os.path.join(fam_dir, name)
            try:
                with Image.open(path) as im:
                    meta = im.text if hasattr(im, "text") else {}
                tiles.append({
                    "file": key,
                    "family": meta.get("cram_family", fam),
                    "seed": int(meta.get("cram_seed", "0")),
                    "gray": name.endswith(".g.png"),
                    "params": json.loads(meta.get("cram_params", "{}")),
                })
            except Exception as e:
                print(f"  skip {path}: {e}")
    with open(mpath, "w") as f:
        json.dump(tiles, f)
    print(f"manifest: {len(tiles)} tiles -> {mpath}")
    return mpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--ss", type=int, default=2,
                    help="supersample factor: render at size*ss, downscale (AA)")
    ap.add_argument("--overrides", default="",
                    help='JSON merged over sampled params, e.g. \'{"edge_wear":0.8}\'')
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed (default: continue after each family's highest)")
    ap.add_argument("--out", default="library")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--rescan", action="store_true",
                    help="rebuild manifest from PNG metadata instead of incrementally")
    ap.add_argument("--normals", action="store_true",
                    help="export .n.png normal maps for every manifest tile")
    args = ap.parse_args()

    from .core.stamp import set_store
    set_store(args.out)  # resolve stamp-family image assets under <out>/_stamps

    if args.manifest_only:
        write_manifest(args.out, rescan=args.rescan)
        return
    if args.normals:
        render_normals(args.out, size=args.size, ss=args.ss)
        return
    overrides = json.loads(args.overrides) if args.overrides else None

    fams = list(FAMILIES) if args.all else [f for f in args.families.split(",") if f]
    for fam in fams:
        if fam not in FAMILIES:
            raise SystemExit(f"unknown family '{fam}' (have: {', '.join(FAMILIES)})")
    for fam in fams:
        base = args.seed if args.seed is not None else next_seed(fam, args.out)
        print(f"[{fam}] rendering {args.count} tiles at {args.size}px "
              f"({'gray' if args.gray else 'color'}), seeds {base}..{base + args.count - 1}",
              flush=True)
        render_batch(fam, args.count, args.size, args.gray, base, args.out,
                     ss=args.ss, overrides=overrides)
    write_manifest(args.out, rescan=args.rescan)
    from .contact_sheet import write_contact
    write_contact(args.out)


if __name__ == "__main__":
    main()
