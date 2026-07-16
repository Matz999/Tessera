"""Serve the tile library with live in-browser generation.

  python -m tessera.serve [--port 8000] [--out library]

Open http://127.0.0.1:8000/ — the contact sheet gains a Generate bar.

Endpoints:
  GET  /               contact sheet UI
  GET  /api/families   known family names
  GET  /api/dials      shared + per-family dial specs for the panel
  GET  /api/fonts      installed stamp fonts (glyph source picker)
  POST /api/generate   {"family", "count", "size", "gray", "seed"?} -> new tiles
  POST /api/breed      {"parents":[{family,seed,params}], count, mutation, blend} -> offspring
  POST /api/delete     {"file": "family/name.png"} -> removes tile (+ normal map)
  POST /api/bless      [entries] -> writes blessed.json next to the library
  POST /api/upload     {"data": base64|dataURL} -> stores a stamp image, {"id"}
Everything under --out is served statically (PNGs, manifest.json, ...).
"""
import argparse
import base64
import json
import os
import re
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .breed import breed_batch
from .contact_sheet import write_contact
from .core.emissive import EMISSION_COLORS, EMISSION_SOURCES
from .core.palette import COLOR_RAMPS, GRAY_RAMPS
from .core.stamp import fonts as stamp_fonts
from .core.stamp import save_upload, set_store
from .families import FAMILIES
from .families.common import DIALS, MATERIALS
from .render import next_seed, render_batch, write_manifest

_RENDER_LOCK = threading.Lock()  # one batch at a time; extra requests queue
_TILE_RE = re.compile(r"^[A-Za-z0-9_]+/\d+_[0-9a-f]{8}(\.g)?\.png$")


class Handler(SimpleHTTPRequestHandler):
    def _json(self, obj, code=200):
        blob = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/contact.html"
        elif self.path == "/api/families":
            return self._json(list(FAMILIES))
        elif self.path == "/api/dials":
            fam = {name: getattr(mod, "DIALS", {}) for name, mod in FAMILIES.items()}
            return self._json({"numeric": DIALS,
                               "choice": {"material": list(MATERIALS),
                                          "ramp": COLOR_RAMPS,
                                          "gray_ramp": GRAY_RAMPS,
                                          "emission_color": list(EMISSION_COLORS),
                                          "emission_source": EMISSION_SOURCES},
                               "families": fam})
        elif self.path == "/api/fonts":
            return self._json(stamp_fonts())
        return super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"null")
        except json.JSONDecodeError:
            return self._json({"error": "invalid JSON body"}, 400)
        try:
            if self.path == "/api/generate":
                return self._generate(body or {})
            if self.path == "/api/breed":
                return self._breed(body or {})
            if self.path == "/api/delete":
                return self._delete(body or {})
            if self.path == "/api/bless":
                return self._bless(body or [])
            if self.path == "/api/upload":
                return self._upload(body or {})
            self._json({"error": f"unknown endpoint {self.path}"}, 404)
        except Exception as e:  # surface render errors to the UI
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _generate(self, body: dict):
        family = body.get("family")
        if family not in FAMILIES:
            return self._json({"error": f"unknown family {family!r}"}, 400)
        count = min(max(int(body.get("count", 3)), 1), 16)
        size = min(max(int(body.get("size", 512)), 64), 1024)
        gray = bool(body.get("gray", False))
        ss = min(max(int(body.get("ss", 2)), 1), 3)
        overrides = body.get("overrides")
        if overrides is not None and not isinstance(overrides, dict):
            return self._json({"error": "overrides must be a JSON object"}, 400)
        with _RENDER_LOCK:
            seed = (int(body["seed"]) if body.get("seed") is not None
                    else next_seed(family, self.directory))
            print(f"[serve] {count} x {family} @ {size}px ss{ss}, seeds "
                  f"{seed}..{seed + count - 1}"
                  + (f", overrides {json.dumps(overrides)}" if overrides else ""),
                  flush=True)
            entries = render_batch(family, count, size, gray, seed, self.directory,
                                   ss=ss, overrides=overrides)
            write_manifest(self.directory)
            write_contact(self.directory)
        self._json({"tiles": entries})

    def _delete(self, body: dict):
        file = body.get("file", "")
        if not _TILE_RE.match(file):  # also blocks any path traversal
            return self._json({"error": f"bad tile path {file!r}"}, 400)
        path = os.path.join(self.directory, *file.split("/"))
        if not os.path.exists(path):
            return self._json({"error": f"{file} not found"}, 404)
        with _RENDER_LOCK:
            os.remove(path)
            npath = path[:-4] + ".n.png"  # normal map, if exported
            if os.path.exists(npath):
                os.remove(npath)
            write_manifest(self.directory)
            write_contact(self.directory)
        print(f"[serve] deleted {file}", flush=True)
        self._json({"ok": True})

    def _breed(self, body: dict):
        """Cross + mutate the params of selected parent tiles into offspring.
        body: {parents:[{family,seed,params}], count, size, gray, ss,
               mutation, blend}."""
        parents = body.get("parents")
        if not isinstance(parents, list) or not parents:
            return self._json({"error": "select at least one parent tile"}, 400)
        count = min(max(int(body.get("count", 6)), 1), 16)
        size = min(max(int(body.get("size", 512)), 64), 1024)
        gray = bool(body.get("gray", False))
        ss = min(max(int(body.get("ss", 2)), 1), 3)
        mutation = min(max(float(body.get("mutation", 0.2)), 0.0), 1.0)
        blend = min(max(float(body.get("blend", 0.5)), 0.0), 1.0)
        with _RENDER_LOCK:
            print(f"[serve] breed {count} from {len(parents)} parents "
                  f"@ {size}px ss{ss}, mut {mutation}, blend {blend}", flush=True)
            try:
                entries = breed_batch(parents, count, size, gray, ss,
                                      mutation, blend, self.directory)
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            write_manifest(self.directory)
            write_contact(self.directory)
        self._json({"tiles": entries})

    def _upload(self, body: dict):
        """Accept a data URL / base64 image -> store it -> return its stamp id.
        The id goes into stamp params as `stamp_image` (source='image')."""
        data = body.get("data", "")
        if "," in data and data.strip().startswith("data:"):
            data = data.split(",", 1)[1]        # strip the data:...;base64, prefix
        try:
            raw = base64.b64decode(data)
        except Exception:
            return self._json({"error": "data must be base64 (optionally a data URL)"}, 400)
        if not raw:
            return self._json({"error": "empty image"}, 400)
        try:
            sid = save_upload(raw)
        except Exception as e:
            return self._json({"error": f"unreadable image: {e}"}, 400)
        print(f"[serve] uploaded stamp {sid} ({len(raw)} bytes)", flush=True)
        self._json({"ok": True, "id": sid, "url": f"_stamps/{sid}"})

    def _bless(self, entries: list):
        path = os.path.join(self.directory, "blessed.json")
        with open(path, "w") as f:
            json.dump(entries, f, indent=1)
        print(f"[serve] blessed.json: {len(entries)} tiles -> {path}", flush=True)
        self._json({"ok": True, "count": len(entries)})

    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):  # mute the tile-image GET spam
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default="library")
    args = ap.parse_args()
    try:  # tile params can carry emoji; keep console prints from crashing on cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    outdir = os.path.abspath(args.out)
    set_store(outdir)  # uploaded stamp images live in <outdir>/_stamps
    if not os.path.exists(os.path.join(outdir, "contact.html")):
        write_manifest(outdir)
        write_contact(outdir)
    srv = ThreadingHTTPServer((args.host, args.port),
                              partial(Handler, directory=outdir))
    print(f"serving {outdir}\n  -> http://{args.host}:{args.port}/  (Ctrl+C to stop)",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
