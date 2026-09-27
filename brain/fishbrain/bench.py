"""FISHBRAIN G1 benchmark: real-time factor of the Fish1 brain on this machine (task C).

Rows (each in its own subprocess, so peak memory is that row's own):
  full        every one of the 13,458,709 segments and every nonzero pair of the brain of record
  forward     the fireable set F: cells reachable from the stimulated tectum through excitatory pairs
  reduced     the exact functional subnetwork used by the gate (brain.build_bundle)
  reduced_hot the reduced network at w_syn = 20 mV, where activity reaches Mauthner (event cost)

Workload: the real right-loom retina rates over the busiest second (the last 1,000 ms of the
expansion, 5 ms frames) on the 2,048 stimulated cells of the reduced network. In `full` and
`forward` the other stimulated tectal cells (which cannot reach a readout) get, per frame, the mean
rate of the kept stimulated cells in their own lobe: a benchmark-only approximation, stated.
Shiu's LIF, dt 0.1 ms, seed "bench-0". RTF = simulated ms / elapsed ms; CPU time is the honest
number on a shared machine, wall time is what a viewer waits.

    python -m fishbrain.bench            # all rows -> data/bench/g1_bench.json
"""
from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

from fishbrain import brain as B
from fishbrain import sim as S

OUT = B.DATA / "bench" / "g1_bench.json"
ROWS = ("reduced", "reduced_hot", "forward", "full")
SIM_MS = 1000.0


def _rss_peak() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def _loom_window():
    b = B.build_bundle(B.GATE_SIGN_RULE, "intact")
    ret = B.Retina(b.stim_ids)
    rates, diag = B.cached_rates(ret, "loom_right")
    w = B.windows_ms()
    end = int(round((w["loom"][0] + w["loom_expansion_ms"]) / B.RETINA.frame_ms))
    n = int(SIM_MS / B.RETINA.frame_ms)
    return b, rates[end - n:end], diag


def _extend(b, seg, stim_all):
    """Rates for every stimulated id: the kept cells' own, the rest their lobe's per-frame mean."""
    ident = B.identified()
    side = dict(zip(ident["tectum"].tolist(), ident["tectum_side"].tolist()))
    kept_side = np.array([side[int(s)] for s in b.stim_ids])
    out = np.zeros((seg.shape[0], len(stim_all)))
    pos = {int(s): i for i, s in enumerate(b.stim_ids.tolist())}
    lobe_mean = {s: seg[:, kept_side == s].mean(1) for s in ("left", "right")}
    for j, s in enumerate(stim_all.tolist()):
        i = pos.get(int(s))
        out[:, j] = seg[:, i] if i is not None else lobe_mean[side[int(s)]]
    return out


def run_row(name: str) -> dict:
    t0 = time.perf_counter()
    b, seg, diag = _loom_window()
    params = S.LIFParams()
    if name in ("reduced", "reduced_hot"):
        net, stim_idx, rates = b.net, b.stim_idx, seg
        if name == "reduced_hot":
            params = S.LIFParams(w_syn_mV=20.0)
    elif name == "forward":
        _, fb = B.build_bundle(B.GATE_SIGN_RULE, "intact", keep_forward=True)
        net, stim_idx, rates = fb.net, fb.idx(fb.stim_ids), _extend(b, seg, fb.stim_ids)
    elif name == "full":
        net, stim_ids = B.full_network(B.GATE_SIGN_RULE)
        stim_idx, rates = net.index_of(stim_ids), _extend(b, seg, stim_ids)
    else:
        raise ValueError(name)
    build_s = time.perf_counter() - t0
    rss_build = _rss_peak()
    sim = S.Simulator(net, seed="bench-0", params=params)
    sim.add_poisson(stim_idx, rates, frame_ms=B.RETINA.frame_ms, name="retina")
    la0 = os.getloadavg()
    c0, w0 = time.process_time(), time.perf_counter()
    r = sim.run(SIM_MS)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    steps = r.n_steps
    return {"row": name, "neurons": int(net.n), "pairs": int(net.nnz), "stimulated": int(len(stim_idx)),
            "w_syn_mV": params.w_syn_mV, "dt_ms": params.dt_ms, "simulated_ms": SIM_MS, "steps": steps,
            "spikes": int(len(r.neurons)), "syn_events": int(sim.stats["syn_events"]),
            "cpu_s": round(cpu, 3), "wall_s": round(wall, 3),
            "rtf_cpu": round(SIM_MS / 1000.0 / cpu, 4), "rtf_wall": round(SIM_MS / 1000.0 / wall, 4),
            "cpu_us_per_step": round(cpu / steps * 1e6, 2), "wall_us_per_step": round(wall / steps * 1e6, 2),
            "network_bytes": int(net.nbytes()), "build_s": round(build_s, 1),
            "rss_peak_after_build_bytes": rss_build, "rss_peak_bytes": _rss_peak(),
            "loadavg_start": la0, "loadavg_end": os.getloadavg(), "rates_sha256_of_window": diag["rates_sha256"]}


def main(rows=ROWS) -> dict:
    res = {"machine": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
           "python": platform.python_version(), "numpy": np.__version__,
           "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": []}
    for name in rows:
        p = subprocess.run([sys.executable, "-m", "fishbrain.bench", "--row", name], capture_output=True, text=True,
                           cwd=str(B.ROOT))
        if p.returncode != 0:
            res["rows"].append({"row": name, "error": p.stderr[-2000:]})
        else:
            res["rows"].append(json.loads(p.stdout.strip().splitlines()[-1]))
        print(json.dumps(res["rows"][-1]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FISHBRAIN G1 benchmark")
    ap.add_argument("--row", choices=ROWS, default=None)
    a = ap.parse_args()
    if a.row:
        print(json.dumps(run_row(a.row)))
    else:
        main()
