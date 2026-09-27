"""Run the G1c fish from a fresh clone. No login, no 2.5 GB download: numpy and scipy only.

    cd brain
    python -m fishbrain.demo                          # replay two published gate trials, check their hashes
    python -m fishbrain.demo loom --az -45            # a looming shadow on the left
    python -m fishbrain.demo prey --az 15 --speed 60  # a prey dot on the right
    python -m fishbrain.demo recede --side left       # the control the G1c fish failed
    python -m fishbrain.demo loom --ablate c          # remove the model's crossing pathway
    python -m fishbrain.demo loom --lesion unilateral_left
    python -m fishbrain.demo loom --g-c 8 --seed my-fish

The fish is the one G1c gated (evidence/G1c-gate.md): the reduced Fish1 network plus the frozen bridge, gains
g_a = 1 and g_c = 4. Its five files are in brain/kit/ and are copied into brain/data/ on first run, after their
sha256 is checked. Every run prints the network digest, the input hash and the spike hash, and writes a spike
raster (SVG) and a receipt (JSON) to brain/runs/. Change the stimulus, the seed, the gains, the bridge or the
lesions and you have a different fish; the digests say exactly which one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from fishbrain import bridge as BR
from fishbrain import cells as C
from fishbrain import readouts as RO
from fishbrain import sim as S
from fishbrain import stimuli as ST

BRAIN = Path(__file__).resolve().parent.parent
KIT = BRAIN / "kit"
RUNS = BRAIN / "runs"
RECORD = BRAIN.parent / "evidence" / "data" / "g1c-data" / "gate" / "gate_results.json"

G1C_GAINS = (1, 4)
G1C_NETWORK_DIGEST = "0b6f3369f21f5c626d9ba1ba4bc02dfefd1e51b945b3355fcd6730c9aa083b4a"
REPLAY = (("loom_az-45_lv120", "G1-seed-0"), ("prey_az+15_v60", "G1-seed-0"))

KIT_FILES = {
    "cells_identified.json": C.CELLS_JSON,
    "readouts.json": RO.READOUTS_JSON,
    "bridge_spec.json": BR.SPEC_PATH,
    "topology_intact.npz": BR.NETDIR / "topology_intact.npz",
    "topology_intact.json": BR.NETDIR / "topology_intact.json",
}

ROWS = [("Mauthner", "Mauthner"), ("MiD2cm", "MiD2cm"), ("MiD3cm", "MiD3cm"), ("nIII_dorsal", "eye muscles (nIII)"),
        ("nMLF", "nMLF"), ("turning", "turning SPNs")]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_kit() -> None:
    """Check brain/kit against its SHA256SUMS and put each file where the engine reads it."""
    sums = {}
    for line in (KIT / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[name.strip()] = digest
    if set(sums) != set(KIT_FILES):
        raise SystemExit(f"brain/kit/SHA256SUMS lists {sorted(sums)}, expected {sorted(KIT_FILES)}")
    for name, dest in KIT_FILES.items():
        src = KIT / name
        if _sha(src) != sums[name]:
            raise SystemExit(f"brain/kit/{name} does not match its sha256; re-clone the repo")
        if dest.exists():
            if _sha(dest) != sums[name]:
                raise SystemExit(f"{dest.relative_to(BRAIN)} exists and differs from brain/kit/{name}. "
                                 f"Move it away to run the demo on the G1c fish.")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)


def build_fish(g_a: int, g_c: int, ablate: List[str]):
    topo = BR.load_topology("intact")
    bw = BR.network(topo, g_a, g_c, ablate)
    if (g_a, g_c) == G1C_GAINS and not ablate and bw.net.digest() != G1C_NETWORK_DIGEST:
        raise SystemExit(f"network digest {bw.net.digest()} is not the G1c fish's {G1C_NETWORK_DIGEST}")
    return bw


def condition(spec: dict, kind: str, a: argparse.Namespace) -> dict:
    """A gate condition built exactly as bridge.make_condition builds the spec's own."""
    if kind == "loom":
        p = dict(spec["stimuli"]["held_out"]["loom"][0])
        p["azimuth_deg"], p["l_over_v_ms"] = float(a.az), float(a.lv)
        if p["azimuth_deg"] == 0:
            raise SystemExit("--az 0 has no side; use a non-zero azimuth")
        _, p["expansion_ms"] = ST.loom_schedule(p["l_over_v_ms"], p["start_deg"], p["end_deg"])
        return BR.make_condition("loom", p)
    if kind == "prey":
        p = dict(spec["stimuli"]["held_out"]["prey"][0])
        p["azimuth_deg"], p["speed_deg_s"] = float(a.az), float(a.speed)
        if p["azimuth_deg"] == 0:
            raise SystemExit("--az 0 has no side; use a non-zero azimuth")
        return BR.make_condition("prey", p)
    if kind == "recede":
        return BR.make_condition("recede", side=a.side)
    return BR.make_condition(kind)


def published_conditions(spec: dict) -> Dict[str, dict]:
    out = {}
    for which in ("held_out_loom", "held_out_prey", "controls"):
        for c in BR.conditions(spec, which):
            out[c["name"]] = c
    return out


def run_trial(bw, retina, cond: dict, seed: str, lesion: Optional[str] = None) -> dict:
    stim = BR.render(cond)
    rates, diag = retina.rates(stim)
    net = bw.net
    lesioned: List[int] = []
    if lesion:
        sets = BR.lesion_sets()
        if lesion not in sets:
            raise SystemExit(f"unknown lesion {lesion!r}; choose from {sorted(sets)}")
        have = set(net.ids.tolist())
        lesioned = [x for x in sets[lesion] if x in have]
        net = net.lesion(np.asarray(lesioned, np.uint64))
    t0 = time.perf_counter()
    sim = S.Simulator(net, seed=seed)
    sim.add_poisson(bw.stim_idx, rates, frame_ms=BR.FRAME_MS, name="retina")
    raster = sim.run(rates.shape[0] * BR.FRAME_MS)
    wall = time.perf_counter() - t0
    sets = BR.deciding_sets()
    rc = BR.readout_counts(raster, net, sets, tuple(cond["window"]))
    score = BR.score_trial(rc, cond)
    return {"cond": cond, "seed": seed, "raster": raster, "net": net, "rates_sha256": diag["rates_sha256"],
            "stim_idx": bw.stim_idx,
            "score": score, "wall_s": wall, "lesion": lesion, "lesioned_ids": lesioned, "sets": sets}


def say(tr: dict) -> str:
    """The decision in plain words."""
    sc, cond = tr["score"], tr["cond"]
    if sc["escape"]:
        parts = [f"ESCAPE {sc['escape_to']} (the {sc['cell_side']} escape cell fired first, {sc['escape_t_ms']:.1f} ms)"]
    else:
        parts = ["no escape"]
    if sc["strike"]:
        side = sc.get("strike_side")
        parts.append("strike ahead" if side == "ahead" else f"strike {side}")
    if sc.get("turn"):
        parts.append(f"turn {sc['turn']}")
    if cond.get("side") in ("left", "right"):
        parts.append(f"stimulus {cond['side']}")
    return " · ".join(parts)


def receipt(tr: dict, bw) -> dict:
    sc = tr["score"]
    return {
        "condition": tr["cond"]["name"], "stimulus": {k: v for k, v in tr["cond"].items() if k in ("kind", "params", "side")},
        "window_ms": list(tr["cond"]["window"]), "seed": tr["seed"],
        "network_digest": tr["net"].digest(), "bridged_digest": bw.digest, "gains": {"g_a": bw.g_a, "g_c": bw.g_c},
        "ablate": list(bw.ablate), "lesion": tr["lesion"], "lesioned_ids": tr["lesioned_ids"],
        "params_digest": S.LIFParams().digest(), "rates_sha256": tr["rates_sha256"],
        "spike_hash": S.spike_hash(tr["raster"]), "n_spikes": int(len(tr["raster"].neurons)),
        "decision": {k: sc.get(k) for k in ("escape", "initiator", "cell_side", "escape_to", "escape_t_ms", "strike",
                                            "strike_side", "turn", "turn_index", "m_spikes")},
        "note": "Simulated. The G1c fish: the Fish1 wiring plus a disclosed bridge. See evidence/G1c-gate.md "
                "for how much of each decision runs on measured synapses.",
    }


def raster_svg(tr: dict, title: str) -> str:
    """Spikes of the deciding cells and the eye's input, one SVG, no dependencies."""
    raster, net, sets, cond = tr["raster"], tr["net"], tr["sets"], tr["cond"]
    t = raster.steps * raster.dt_ms
    T = float(max(t.max() if len(t) else 0.0, cond["window"][1]))
    W, left, right, top, row_h = 960, 150, 24, 64, 34
    rows = [(f"{k}_{s}", f"{label} {s[0].upper()}") for k, label in ROWS for s in ("left", "right")]
    H = top + (len(rows) + 2) * row_h + 70
    x = lambda ms: left + (W - left - right) * ms / T
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="JetBrains Mono, Menlo, monospace">',
           f'<rect width="{W}" height="{H}" fill="#000"/>',
           f'<text x="24" y="34" fill="#fff" font-size="15" font-weight="700">{_esc(title)}</text>']
    w0, w1 = cond["window"]
    out.append(f'<rect x="{x(w0):.1f}" y="{top - 8}" width="{x(w1) - x(w0):.1f}" height="{(len(rows) + 2) * row_h}" '
               f'fill="#45ff59" fill-opacity="0.06"/>')
    bins = np.zeros(int(T // 10) + 1)
    m = np.isin(raster.neurons, tr["stim_idx"])
    np.add.at(bins, (t[m] // 10).astype(int), 1)
    y0 = top + row_h
    peak = bins.max() or 1.0
    pts = " ".join(f"{x(i * 10):.1f},{y0 - (row_h - 6) * v / peak:.1f}" for i, v in enumerate(bins))
    out.append(f'<text x="24" y="{y0 - 10}" fill="#8e9390" font-size="12">eye input (tectum)</text>')
    out.append(f'<polyline points="{pts}" fill="none" stroke="#45ff59" stroke-width="1"/>')
    for r, (key, label) in enumerate(rows):
        y = top + (r + 2) * row_h
        out.append(f'<text x="24" y="{y - 10}" fill="#c9cdca" font-size="12">{label}</text>')
        out.append(f'<line x1="{left}" y1="{y}" x2="{W - right}" y2="{y}" stroke="#1c2120"/>')
        idx = [i for _, i in BR._group_idx(net, {key: sets[key]})[key] if i >= 0]
        if not idx:
            continue
        m = np.isin(raster.neurons, idx)
        colour = "#ff4a3d" if key.startswith(("Mauthner", "MiD")) else "#45ff59"
        for ts in np.unique(np.round(t[m], 0)):
            out.append(f'<line x1="{x(ts):.1f}" y1="{y - row_h + 10}" x2="{x(ts):.1f}" y2="{y - 2}" '
                       f'stroke="{colour}" stroke-width="1"/>')
    sc = tr["score"]
    if sc["escape"] and sc["escape_t_ms"] is not None:
        xe = x(sc["escape_t_ms"])
        out.append(f'<line x1="{xe:.1f}" y1="{top - 8}" x2="{xe:.1f}" y2="{top + (len(rows) + 1) * row_h}" '
                   f'stroke="#ff4a3d" stroke-dasharray="3 3"/>')
    yb = top + (len(rows) + 1) * row_h + 22
    for ms in range(0, int(T) + 1, 500):
        out.append(f'<text x="{x(ms):.1f}" y="{yb}" fill="#8e9390" font-size="11" text-anchor="middle">{ms}</text>')
    out.append(f'<text x="{W - right}" y="{yb}" fill="#8e9390" font-size="11" text-anchor="end">ms</text>')
    out.append(f'<text x="24" y="{H - 30}" fill="#c9cdca" font-size="12">{_esc(say(tr))}</text>')
    out.append(f'<text x="24" y="{H - 12}" fill="#8e9390" font-size="11">Simulated spikes on the G1c fish (Fish1 wiring '
               f'+ a disclosed bridge) · spike hash {S.spike_hash(raster)[:16]}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_outputs(tr: dict, bw, stem: str) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{stem}.json").write_text(json.dumps(receipt(tr, bw), indent=2, default=float) + "\n")
    title = f"{tr['cond']['name']} · seed {tr['seed']} · g_a {bw.g_a} g_c {bw.g_c}"
    if bw.ablate:
        title += f" · bridge removed: {','.join(bw.ablate)}"
    if tr["lesion"]:
        title += f" · lesion {tr['lesion']}"
    (RUNS / f"{stem}.svg").write_text(raster_svg(tr, title))
    return RUNS / f"{stem}.svg"


def replay(bw, retina, spec: dict) -> int:
    """Re-run two held-out G1c trials and compare with the published record."""
    record = json.loads(RECORD.read_text())["trials"]["intact"]
    conds = published_conditions(spec)
    ok = True
    for name, seed in REPLAY:
        want = next(r for r in record[name] if r["seed"] == seed)["spike_hash"]
        tr = run_trial(bw, retina, conds[name], seed)
        got = S.spike_hash(tr["raster"])
        same = got == want
        ok &= same
        svg = write_outputs(tr, bw, f"replay-{name}-{seed}")
        print(f"\n{name}  seed {seed}  ({tr['wall_s']:.1f} s)")
        print(f"  {say(tr)}")
        print(f"  spike hash  {got}")
        print(f"  published   {want}  {'MATCH' if same else 'DIFFERENT'}")
        print(f"  raster      {svg.relative_to(BRAIN)}")
    print("\nBoth trials reproduce the published record bit for bit." if ok else
          "\nA trial differs from the published record. Open an issue with your platform and the two hashes.")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fishbrain.demo", description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", nargs="?", choices=["loom", "prey", "recede", "dim", "blank"],
                    help="omit to replay two published gate trials")
    ap.add_argument("--az", type=float, default=-45.0, help="stimulus azimuth in degrees; negative = left (default -45)")
    ap.add_argument("--lv", type=float, default=120.0, help="loom l/v in ms: smaller = faster approach (default 120)")
    ap.add_argument("--speed", type=float, default=60.0, help="prey speed in deg/s (default 60)")
    ap.add_argument("--side", choices=["left", "right"], default="left", help="side for recede (default left)")
    ap.add_argument("--seed", default="G1-seed-0", help="any string; the gate used G1-seed-0 ... G1-seed-7")
    ap.add_argument("--g-a", type=int, default=G1C_GAINS[0], help="bridge gain for class (a) links, 1-32 (G1c: 1)")
    ap.add_argument("--g-c", type=int, default=G1C_GAINS[1], help="bridge gain for class (c) links, 1-32 (G1c: 4)")
    ap.add_argument("--ablate", default="", help="remove bridge links: a, c, all, or a pathway prefix (E1, P, ...)")
    ap.add_argument("--lesion", default=None, help="silence cells: bilateral, unilateral_left, unilateral_right, ...")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    ensure_kit()
    spec = BR.load_spec()
    ablate = [x for x in a.ablate.split(",") if x]
    bw = build_fish(a.g_a, a.g_c, ablate)
    print(f"fish: {bw.net.n:,} cells, {bw.net.nnz:,} connections, network digest {bw.net.digest()[:16]}")
    print("building the eye (about 10 s)...", flush=True)
    retina = BR.GateRetina(bw.stim_ids, cache=False)
    print(f"ready in {time.perf_counter() - t0:.1f} s", flush=True)
    if a.kind is None:
        if (a.g_a, a.g_c) != G1C_GAINS or ablate:
            raise SystemExit("the replay checks the G1c fish; drop --g-a/--g-c/--ablate or name a stimulus")
        return replay(bw, retina, spec)
    cond = condition(spec, a.kind, a)
    tr = run_trial(bw, retina, cond, a.seed, a.lesion)
    stem = f"{cond['name']}-{a.seed}" + (f"-ablate-{'+'.join(ablate)}" if ablate else "") + \
           (f"-lesion-{a.lesion}" if a.lesion else "") + ("" if (a.g_a, a.g_c) == G1C_GAINS else f"-g{a.g_a}-{a.g_c}")
    svg = write_outputs(tr, bw, stem)
    print(f"\n{cond['name']}  seed {a.seed}  ({tr['wall_s']:.1f} s)")
    print(f"  {say(tr)}")
    print(f"  spike hash  {S.spike_hash(tr['raster'])}")
    print(f"  raster      {svg.relative_to(BRAIN)}")
    print(f"  receipt     {svg.with_suffix('.json').relative_to(BRAIN)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
