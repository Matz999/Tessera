"""Build library/contact.html — the curation surface.

Embeds manifest.json inline (works over file://). Click a tile to bless it;
state lives in localStorage; Export downloads blessed.json which you drop
next to the library (library/blessed.json) for the app to use.

When the page is opened through `python -m tessera.serve` it upgrades itself:
a Generate bar appears (renders new tiles via POST /api/generate), the
manifest is re-fetched live, and Export saves blessed.json server-side.

  python -m tessera.contact_sheet [--out library]
"""
import argparse
import json
import os

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Tile library — curation</title>
<style>
  :root { color-scheme: dark; }
  body { background:#111; color:#ddd; font:14px system-ui, sans-serif; margin:0; }
  header { position:sticky; top:0; background:#111d; backdrop-filter:blur(6px);
           border-bottom:1px solid #333; z-index:2; }
  .bar { padding:8px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  .bar + .bar { border-top:1px solid #222; }
  header b { font-size:15px; }
  select, button, input { background:#222; color:#ddd; border:1px solid #444;
           border-radius:6px; padding:5px 10px; font:inherit; }
  button { cursor:pointer; }
  button:disabled { opacity:.45; cursor:default; }
  input[type=number] { width:64px; }
  input[type=text] { width:90px; }
  button.primary { background:#2a4; border-color:#2a4; color:#fff; }
  button.gen { background:#26c; border-color:#26c; color:#fff; }
  label.chk { display:flex; gap:5px; align-items:center; cursor:pointer; user-select:none; }
  #genbar { display:none; }
  #status { color:#8ac; font-variant-numeric:tabular-nums; }
  #dials { display:none; max-height:44vh; overflow-y:auto; padding:6px 16px 12px;
           border-top:1px solid #222; }
  #dials .grid2 { display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr));
           gap:2px 22px; }
  #dials h4 { grid-column:1/-1; margin:10px 0 2px; color:#8ac; font-size:12px;
           text-transform:uppercase; letter-spacing:.08em; }
  .dial { display:flex; gap:8px; align-items:center; padding:1px 0; }
  .dial label { width:112px; font-size:12px; color:#999; overflow:hidden;
           text-overflow:ellipsis; white-space:nowrap; }
  .dial input[type=range] { flex:1; min-width:60px; padding:0; accent-color:#555; }
  .dial input[type=number] { width:70px; padding:2px 6px; font-size:12px;
           color:#666; }
  .dial select { flex:1; padding:2px 6px; font-size:12px; }
  .dial .unpin { width:22px; height:22px; padding:0; border-radius:50%;
           font-size:12px; line-height:1; color:#777; }
  .dial.pinned label { color:#fc3; }
  .dial.pinned input[type=range] { accent-color:#fc3; }
  .dial.pinned input[type=number] { color:#fc3; border-color:#a82; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr));
          gap:8px; padding:12px; }
  .cell { position:relative; aspect-ratio:1; cursor:pointer; border-radius:8px;
          overflow:hidden; outline:3px solid transparent; outline-offset:-3px; }
  .cell img { width:100%; height:100%; object-fit:cover; display:block; }
  .cell.blessed { outline-color:#3c5; }
  .cell.blessed::after { content:"\\2713"; position:absolute; top:6px; right:8px;
          background:#3c5; color:#fff; border-radius:50%; width:22px; height:22px;
          display:grid; place-items:center; font-weight:bold; }
  .cell.new { outline-color:#fc3; }
  .cell.new.blessed { outline-color:#3c5; }
  .cell.new::before { content:"NEW"; position:absolute; top:6px; left:8px;
          background:#fc3; color:#000; border-radius:4px; padding:1px 6px;
          font-size:10px; font-weight:700; }
  .cell .tag { position:absolute; left:0; bottom:0; right:0; padding:3px 7px;
          background:#000a; font-size:11px; color:#bbb; opacity:0; transition:opacity .15s; }
  .cell:hover .tag { opacity:1; }
  .cell .del { position:absolute; top:6px; left:8px; width:22px; height:22px;
          background:#c33; color:#fff; border:none; border-radius:50%; padding:0;
          display:grid; place-items:center; font-size:13px; font-weight:700;
          line-height:1; opacity:0; transition:opacity .15s; }
  .cell:hover .del { opacity:1; }
  .cell.new .del { top:26px; }  /* keep clear of the NEW badge */
  .count { color:#3c5; font-weight:600; }
</style></head><body>
<header>
  <div class="bar">
    <b>Tile library</b>
    <select id="famFilter"><option value="">all families</option></select>
    <select id="modeFilter">
      <option value="">all</option><option value="color">color</option><option value="gray">gray</option>
    </select>
    <select id="blessFilter">
      <option value="">all tiles</option><option value="blessed">blessed only</option>
      <option value="unblessed">unblessed only</option>
    </select>
    <span><span class="count" id="count">0</span> blessed / <span id="total">0</span></span>
    <button class="primary" id="export">Export blessed.json</button>
    <button id="clear">Clear all</button>
  </div>
  <div class="bar" id="genbar">
    <b style="color:#8ac">Generate</b>
    <select id="genFam"></select>
    <label class="chk">count <input id="genCount" type="number" min="1" max="16" value="3"></label>
    <label class="chk">size <select id="genSize">
      <option>256</option><option selected>512</option><option>768</option><option>1024</option>
    </select></label>
    <label class="chk"><input type="checkbox" id="genGray"> gray</label>
    <label class="chk">AA <select id="genSS">
      <option value="1">1&#215;</option><option value="2" selected>2&#215;</option>
      <option value="3">3&#215;</option>
    </select></label>
    <input id="genSeed" type="text" placeholder="seed: auto">
    <button id="dialsBtn">Dials &#9662;</button>
    <input id="genOv" type="text" style="width:190px"
           placeholder='raw overrides: {"boss":0.2}' title="JSON merged over sampled params + pinned dials (wins ties)">
    <button class="gen" id="genBtn">Generate</button>
    <span id="status"></span>
  </div>
  <div id="dials">
    <div class="grid2" id="famDialGrid"></div>
    <div class="grid2" id="dialGrid"></div>
  </div>
</header>
<div class="grid" id="grid"></div>
<script>
const MANIFEST_INLINE = __MANIFEST__;
let MANIFEST = MANIFEST_INLINE;
const SERVED = location.protocol.startsWith("http");
const KEY = "cram_blessed_tiles";
let blessed = new Set(JSON.parse(localStorage.getItem(KEY) || "[]"));
const newFiles = new Set();
const $ = id => document.getElementById(id);

function populateFilters() {
  const sel = $("famFilter"), cur = sel.value;
  sel.innerHTML = '<option value="">all families</option>';
  [...new Set(MANIFEST.map(t => t.family))].sort().forEach(f => {
    const o = document.createElement("option"); o.value = o.textContent = f;
    sel.appendChild(o);
  });
  sel.value = cur;
  $("total").textContent = MANIFEST.length;
}

function save() {
  localStorage.setItem(KEY, JSON.stringify([...blessed]));
  $("count").textContent = blessed.size;
}

function render() {
  const fam = $("famFilter").value, mode = $("modeFilter").value,
        bf = $("blessFilter").value, grid = $("grid");
  grid.innerHTML = "";
  const list = [...MANIFEST].sort((a, b) =>
      newFiles.has(b.file) - newFiles.has(a.file));  // fresh tiles first
  for (const t of list) {
    if (fam && t.family !== fam) continue;
    if (mode === "gray" && !t.gray) continue;
    if (mode === "color" && t.gray) continue;
    if (bf === "blessed" && !blessed.has(t.file)) continue;
    if (bf === "unblessed" && blessed.has(t.file)) continue;
    const cell = document.createElement("div");
    cell.className = "cell" + (blessed.has(t.file) ? " blessed" : "")
                            + (newFiles.has(t.file) ? " new" : "");
    const img = document.createElement("img");
    img.loading = "lazy"; img.src = t.file;
    const tag = document.createElement("div");
    tag.className = "tag"; tag.textContent = t.family + " #" + t.seed;
    tag.title = JSON.stringify(t.params, null, 1);  // hover the label = all dials
    cell.append(img, tag);
    if (SERVED) {
      const del = document.createElement("button");
      del.className = "del"; del.textContent = "\\u00d7";
      del.title = "delete tile";
      del.onclick = e => { e.stopPropagation(); deleteTile(t, cell); };
      cell.appendChild(del);
    }
    cell.onclick = () => {
      blessed.has(t.file) ? blessed.delete(t.file) : blessed.add(t.file);
      cell.classList.toggle("blessed"); save();
    };
    grid.appendChild(cell);
  }
}

// ---- dial panel (server mode): pin any generation dial to a value ----
const pinned = {};        // shared render dials
const familyPinned = {};  // the selected family's own dials (reset on switch)
let DIALSPEC = null;

function coerce(v) {      // "4" -> 4, "true" -> true, "cord" -> "cord"
  if (v === "true") return true;
  if (v === "false") return false;
  if (v !== "" && !isNaN(v)) return Number(v);
  return v;
}
function updateDialsBtn() {
  const n = Object.keys(pinned).length + Object.keys(familyPinned).length;
  $("dialsBtn").textContent = "Dials" + (n ? " (" + n + " pinned)" : "") + " \\u25be";
}

function numericDial(name, d, pins) {
  const row = document.createElement("div"); row.className = "dial";
  const lab = document.createElement("label");
  lab.textContent = name; lab.title = name;
  const rng = document.createElement("input");
  rng.type = "range"; rng.min = d.lo; rng.max = d.hi; rng.step = d.step;
  rng.value = (d.lo + d.hi) / 2;
  const num = document.createElement("input");
  num.type = "number"; num.min = d.lo; num.max = d.hi; num.step = d.step;
  num.placeholder = "auto";
  const un = document.createElement("button");
  un.className = "unpin"; un.textContent = "\\u00d7"; un.title = "back to auto";
  const pin = v => { pins[name] = v; rng.value = v; num.value = v;
                     row.classList.add("pinned"); updateDialsBtn(); };
  rng.oninput = () => pin(+rng.value);
  num.oninput = () => {
    if (num.value === "") { delete pins[name]; row.classList.remove("pinned"); }
    else { const v = parseFloat(num.value); if (!isNaN(v)) pin(v); }
    updateDialsBtn();
  };
  un.onclick = () => { delete pins[name]; num.value = "";
                       row.classList.remove("pinned"); updateDialsBtn(); };
  row.append(lab, rng, num, un);
  return row;
}

function choiceDial(name, options, pins) {
  const row = document.createElement("div"); row.className = "dial";
  const lab = document.createElement("label"); lab.textContent = name; lab.title = name;
  const sel = document.createElement("select");
  const auto = document.createElement("option");
  auto.value = ""; auto.textContent = "auto";
  sel.appendChild(auto);
  options.forEach(o => { const e = document.createElement("option");
                         e.value = e.textContent = o; sel.appendChild(e); });
  sel.onchange = () => {
    if (sel.value === "") { delete pins[name]; row.classList.remove("pinned"); }
    else { pins[name] = coerce(sel.value); row.classList.add("pinned"); }
    updateDialsBtn();
  };
  row.append(lab, sel);
  return row;
}

async function buildDials() {
  DIALSPEC = await (await fetch("/api/dials")).json();
  const grid = $("dialGrid");
  const h = document.createElement("h4"); h.textContent = "material + palette";
  grid.appendChild(h);
  for (const [name, opts] of Object.entries(DIALSPEC.choice))
    grid.appendChild(choiceDial(name, opts, pinned));
  let group = null;
  for (const [name, d] of Object.entries(DIALSPEC.numeric)) {
    if (d.group !== group) {
      group = d.group;
      const hg = document.createElement("h4"); hg.textContent = group;
      grid.appendChild(hg);
    }
    grid.appendChild(numericDial(name, d, pinned));
  }
  buildFamilyDials($("genFam").value);
}

function buildFamilyDials(family) {
  const g = $("famDialGrid"); g.innerHTML = "";
  for (const k in familyPinned) delete familyPinned[k];  // family switch clears its pins
  const spec = (DIALSPEC && DIALSPEC.families && DIALSPEC.families[family]) || {};
  if (Object.keys(spec).length) {
    const h = document.createElement("h4"); h.textContent = "family: " + family;
    g.appendChild(h);
    for (const [name, d] of Object.entries(spec)) {
      if (d.choices) g.appendChild(choiceDial(name, d.choices, familyPinned));
      else if (d.bool) g.appendChild(choiceDial(name, ["true", "false"], familyPinned));
      else g.appendChild(numericDial(name, d, familyPinned));
    }
  }
  updateDialsBtn();
}

async function deleteTile(t, cell) {
  const r = await fetch("/api/delete", { method: "POST",
                                         body: JSON.stringify({file: t.file}) });
  const data = await r.json();
  if (!r.ok) { $("status").textContent = "error: " + (data.error || r.statusText); return; }
  MANIFEST = MANIFEST.filter(x => x.file !== t.file);
  blessed.delete(t.file); newFiles.delete(t.file); save();
  populateFilters();
  cell.remove();  // no full re-render: keeps scroll position while culling
  $("status").textContent = "deleted " + t.file;
}

async function refreshManifest() {
  MANIFEST = await (await fetch("manifest.json?t=" + Date.now())).json();
  populateFilters(); render();
}

async function generate() {
  const body = { family: $("genFam").value, count: +$("genCount").value || 1,
                 size: +$("genSize").value, gray: $("genGray").checked,
                 ss: +$("genSS").value };
  const s = $("genSeed").value.trim();
  if (s) body.seed = +s;
  const ov = Object.assign({}, pinned, familyPinned);  // panel first, raw JSON wins ties
  const raw = $("genOv").value.trim();
  if (raw) {
    try { Object.assign(ov, JSON.parse(raw)); }
    catch (e) { $("status").textContent = "bad overrides JSON: " + e.message; return; }
  }
  if (Object.keys(ov).length) body.overrides = ov;
  $("genBtn").disabled = true;
  const t0 = Date.now();
  const tick = setInterval(() => { $("status").textContent =
      "rendering " + body.count + " \\u00d7 " + body.family + "\\u2026 "
      + ((Date.now() - t0) / 1000).toFixed(0) + "s"; }, 250);
  try {
    const r = await fetch("/api/generate", { method: "POST", body: JSON.stringify(body) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);
    data.tiles.forEach(t => newFiles.add(t.file));
    await refreshManifest();
    $("status").textContent = "done: " + data.tiles.length + " tiles in "
        + ((Date.now() - t0) / 1000).toFixed(1) + "s";
  } catch (e) {
    $("status").textContent = "error: " + e.message;
  } finally {
    clearInterval(tick);
    $("genBtn").disabled = false;
  }
}

$("export").onclick = async () => {
  const out = MANIFEST.filter(t => blessed.has(t.file))
    .map(t => ({file: t.file, family: t.family, seed: t.seed, params: t.params}));
  if (SERVED) {
    const r = await fetch("/api/bless", { method: "POST", body: JSON.stringify(out) });
    $("status").textContent = r.ok
        ? "blessed.json saved (" + out.length + " tiles)" : "save failed";
    return;
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)],
                                        {type: "application/json"}));
  a.download = "blessed.json"; a.click();
};
$("clear").onclick = () => {
  if (confirm("Unbless everything?")) { blessed.clear(); save(); render(); }
};
$("genBtn").onclick = generate;
$("famFilter").onchange = render;
$("modeFilter").onchange = render;
$("blessFilter").onchange = render;

$("dialsBtn").onclick = () => {
  const p = $("dials");
  p.style.display = p.style.display === "block" ? "none" : "block";
};

if (SERVED) {
  $("genbar").style.display = "flex";
  $("export").textContent = "Save blessed.json";
  $("genFam").onchange = () => buildFamilyDials($("genFam").value);
  fetch("/api/families").then(r => r.json()).then(fams => {
    fams.forEach(f => {
      const o = document.createElement("option"); o.value = o.textContent = f;
      $("genFam").appendChild(o);
    });
    buildDials();  // after genFam has options, so the family group matches selection
  });
  refreshManifest();
}
populateFilters(); save(); render();
</script></body></html>
"""


def write_contact(outdir: str) -> str:
    mpath = os.path.join(outdir, "manifest.json")
    with open(mpath) as f:
        manifest = json.load(f)
    html = TEMPLATE.replace("__MANIFEST__", json.dumps(manifest))
    cpath = os.path.join(outdir, "contact.html")
    with open(cpath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"contact sheet: {len(manifest)} tiles -> {cpath}")
    return cpath


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="library")
    args = ap.parse_args()
    write_contact(args.out)
