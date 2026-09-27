"""Export FISHBRAIN simulator cases for the JS kernel (js/fishbrain-sim.mjs) with Python's answers.

A case is one JSON file: the wiring (CSR), the LIF parameters, the seed string, every input source
in the order it is added, the run() calls, and the reference engine's results after each call
(spike_hash of the window, state_digest, stats) plus the derived constants' bits and the digests.

The Python results are computed from the JSON itself (python_sim(case) rebuilds the Simulator from
the decoded case), so a case that matches proves the file is a complete description of the run.

    .venv/bin/python js/export_case.py --out /tmp/cases            # synthetic cases
    .venv/bin/python js/export_case.py --out data/js --real        # + the real gate network

Run with the brain venv. Large arrays (rates, currents) travel as base64 little-endian float64, so
there is no decimal round trip anywhere; uint64 neuron ids travel as decimal strings.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BRAIN = HERE.parent
sys.path.insert(0, str(BRAIN))

from fishbrain import sim as S  # noqa: E402

FORMAT = "fishbrain-js-case-v1"
PARAM_FIELDS = ("v_rest_mV", "v_reset_mV", "v_thresh_mV", "tau_m_ms", "tau_syn_ms", "refractory_ms",
                "delay_ms", "w_syn_mV", "poisson_weight_mV", "dt_ms")


# ---------------------------------------------------------------------------------------------
# encoding

def f64b64(a) -> dict:
    arr = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    return {"f64_b64": base64.b64encode(arr.tobytes()).decode("ascii"), "shape": list(arr.shape)}


def enc_values(x):
    """Scalar -> JSON float (Python's repr, which JS parses exactly); arrays -> base64 float64."""
    a = np.asarray(x, dtype=np.float64)
    return float(a) if a.ndim == 0 else f64b64(a)


def dec_values(x):
    if isinstance(x, dict):
        raw = base64.b64decode(x["f64_b64"])
        return np.frombuffer(raw, dtype="<f8").reshape(x["shape"]).astype(np.float64)
    return float(x)


def f32_hex(x) -> str:
    return np.asarray([x], "<f4").tobytes().hex()


def f64_hex(x) -> str:
    return np.asarray([x], "<f8").tobytes().hex()


def network_json(net: S.Network, with_ids: bool) -> dict:
    return {"n": int(net.n),
            "ids": [str(int(i)) for i in net.ids] if with_ids else None,
            "indptr": net.indptr.astype(np.int64).tolist(),
            "indices": net.indices.astype(np.int64).tolist(),
            "counts": net.counts.astype(np.int64).tolist(),
            "dead_indices": np.flatnonzero(net.dead).astype(np.int64).tolist()}


def source(kind, neurons, values, start_ms=0.0, stop_ms=None, frame_ms=None, name=None, weight_mV=None) -> dict:
    s = {"type": kind, "neurons": [int(i) for i in np.asarray(neurons).ravel()],
         "start_ms": float(start_ms), "stop_ms": None if stop_ms is None else float(stop_ms),
         "frame_ms": None if frame_ms is None else float(frame_ms), "name": name}
    if kind == "poisson":
        s["rate_hz"] = enc_values(values)
        s["weight_mV"] = None if weight_mV is None else float(weight_mV)
    else:
        s["amp_mV"] = enc_values(values)
    return s


def make_case(name, net, params: dict, seed: str, sources, runs, with_ids=False, meta=None) -> dict:
    params = {k: float(v) for k, v in params.items()}          # repr(20) != repr(20.0) in the digest
    case = {"format": FORMAT, "name": name, "seed": seed, "params": params,
            "network": network_json(net, with_ids), "sources": sources, "runs": runs, "meta": meta or {}}
    case["expected"] = expected(case)
    return case


# ---------------------------------------------------------------------------------------------
# the reference run, rebuilt from the JSON

def python_sim(case: dict, cls=S.Simulator):
    N = case["network"]
    n = int(N["n"])
    ids = np.arange(n, dtype=np.uint64) if N.get("ids") is None else np.array([int(x) for x in N["ids"]], np.uint64)
    dead = np.zeros(n, bool)
    dead[np.asarray(N["dead_indices"], np.int64)] = True
    net = S.Network(ids, np.asarray(N["indptr"], np.int64), np.asarray(N["indices"], np.int32),
                    np.asarray(N["counts"], np.int32), dead=dead)
    params = S.LIFParams(**{k: float(v) for k, v in case["params"].items()})
    sim = cls(net, seed=case["seed"], params=params)
    for s in case["sources"]:
        kw = dict(start_ms=s["start_ms"], stop_ms=s["stop_ms"], frame_ms=s["frame_ms"], name=s["name"])
        idx = np.asarray(s["neurons"], np.int64)
        if s["type"] == "poisson":
            sim.add_poisson(idx, dec_values(s["rate_hz"]), weight_mV=s["weight_mV"], **kw)
        elif s["type"] == "current":
            sim.add_current(idx, dec_values(s["amp_mV"]), **kw)
        else:
            raise ValueError(s["type"])
    return sim, net, params


def _run_all(case, cls):
    sim, net, params = python_sim(case, cls)
    out = []
    t0 = time.process_time()
    for r in case["runs"]:
        ras = sim.run(r.get("duration_ms"), n_steps=r.get("n_steps"))
        out.append({"spike_hash": S.spike_hash(ras), "state_digest": sim.state_digest(),
                    "n_spikes": int(len(ras.neurons)), "n_steps": int(ras.n_steps), "stats": {k: int(v) for k, v in sim.stats.items()}})
    cpu = time.process_time() - t0
    return sim, net, params, out, cpu


def expected(case: dict) -> dict:
    sim, net, params, runs, cpu = _run_all(case, S.Simulator)
    full = sim.raster()
    d = params.derived()
    w_bits = sim._w.view(np.uint32)
    case["weights_f32_bits"] = w_bits.astype(np.int64).tolist()     # input for the weight-bit mutation control
    exp = {
        "runs": runs,
        "raster_hash": S.spike_hash(full),
        "n_spikes": int(len(full.neurons)),
        "spike_counts": np.bincount(full.neurons, minlength=net.n).astype(np.int64).tolist(),
        "params_repr": {k: repr(getattr(params, k)) for k in PARAM_FIELDS},
        "params_digest": params.digest(),
        "network_digest": net.digest(),
        "weights_sha256": hashlib.sha256(sim._w.astype("<f4").tobytes()).hexdigest(),
        "derived": {"a": f32_hex(d["a"]), "b": f32_hex(d["b"]), "c": f32_hex(d["c"]), "k_i": f32_hex(d["k_i"]),
                    "theta": f32_hex(d["theta"]), "u_reset": f32_hex(d["u_reset"]),
                    "c64": f64_hex(d["c64"]), "k_i64": f64_hex(d["k_i64"]),
                    "n_ref": int(d["n_ref"]), "n_delay": int(d["n_delay"])},
        "poisson_seeds": {s.name: str(S.derive_seed(case["seed"], "poisson", s.name)) for s in sim.sources if s.kind == "poisson"},
        "python_cpu_s": round(cpu, 3),
    }
    # the activity-gated engine must agree too (it is the one make_simulator picks for big networks)
    try:
        from fishbrain.sim_fast import ActiveSimulator
        _, _, _, runs_fast, _ = _run_all(case, ActiveSimulator)
        exp["active_simulator_identical"] = runs_fast == runs
    except ImportError:
        exp["active_simulator_identical"] = None
    return exp


# ---------------------------------------------------------------------------------------------
# synthetic cases

def synthetic_cases() -> list:
    cases = []

    # 1. small, mostly excitatory, strong synapses; scalar + framed Poisson + a current; uneven runs
    net = S.synthetic_graph(300, 9000, seed="js-s1", frac_type1=0.8)
    rates = np.linspace(0.0, 120.0, 6 * 50).reshape(6, 50)
    cases.append(make_case(
        "s1-small-excit-dt0.1", net, {"w_syn_mV": 2.5}, "0x9f3c0000js-s1",
        [source("poisson", np.arange(0, 300, 3), 30.0, name="bg"),
         source("poisson", np.arange(10, 60), rates, frame_ms=25.0, weight_mV=12.0),          # default name poisson1
         source("current", np.arange(200, 240), 4.0, start_ms=20.0, stop_ms=140.0, name="drive")],
        [{"duration_ms": 37.3}, {"duration_ms": 50.0}, {"n_steps": 0}, {"duration_ms": 112.7}, {"n_steps": 1}]))

    # 2. mostly inhibitory, dt 0.2, lesion, per-neuron (K,) rates, (F, K) currents with negatives, p >= 1
    net = S.synthetic_graph(1000, 30000, seed="js-s2", frac_type1=0.35).lesion_indices(np.arange(0, 1000, 13))
    k_rates = ((np.arange(500) * 37) % 101).astype(np.float64) * 0.8
    amps = (((np.arange(12 * 200) * 7919) % 211).astype(np.float64) - 90.0) / 10.0
    cases.append(make_case(
        "s2-inhib-dt0.2-lesion", net, {"dt_ms": 0.2, "w_syn_mV": 4.0}, "0xabc",
        [source("current", np.arange(1, 1000, 5), amps.reshape(12, 200), frame_ms=10.0, start_ms=4.0, name="ramp"),
         source("poisson", np.arange(0, 1000, 2), k_rates, weight_mV=30.0, name="per-neuron"),
         source("poisson", [5, 17, 29], 6000.0, start_ms=40.0, stop_ms=41.0, weight_mV=2.0, name="certain")],
        [{"n_steps": 333}, {"duration_ms": 41.0}, {"n_steps": 1}, {"duration_ms": 158.8}]))

    # 3. dense multi-synapse pairs, bursty; odd time constants; a rate below 2^-40 per step; zero rate;
    #    a framed source whose frames run out before its stop, one frame at p >= 1
    net = S.synthetic_graph(120, 24000, seed="js-s3", frac_type1=0.7)
    fr = np.zeros((4, 10))
    fr[1, :] = 20000.0
    fr[2, ::2] = 300.0
    cases.append(make_case(
        "s3-dense-bursty", net,
        {"w_syn_mV": 1.2, "tau_m_ms": 15.0, "tau_syn_ms": 3.0, "refractory_ms": 1.0, "delay_ms": 0.5},
        "block 0000000000000000000000000000000000000000000000000000000000000000",
        [source("poisson", np.arange(120), 60.0, weight_mV=20.0, name="bg"),
         source("poisson", np.arange(0, 120, 7), 1e-9, name="tiny"),
         source("poisson", np.arange(3, 120, 11), 0.0, name="zero"),
         source("poisson", np.arange(100, 110), fr, frame_ms=5.0, start_ms=12.0, stop_ms=90.0, weight_mV=9.0, name="frames")],
        [{"duration_ms": 25.5}, {"duration_ms": 74.5}, {"duration_ms": 100.0}]))

    # 4. tau_m == tau_syn (the other branch of c), no delay (one slot), refractory of one step (no freeze),
    #    reset below rest, a framed Poisson source starting late
    net = S.synthetic_graph(600, 15000, seed="js-s4", frac_type1=0.6)
    fr4 = np.tile(np.array([[0.0], [80.0], [15.0]]), (1, 60))
    cases.append(make_case(
        "s4-equal-tau-no-delay", net,
        {"tau_m_ms": 10.0, "tau_syn_ms": 10.0, "delay_ms": 0.0, "refractory_ms": 0.1, "w_syn_mV": 3.0,
         "v_reset_mV": -60.0, "poisson_weight_mV": 50.0},
        "seed with spaces | and pipes",
        [source("poisson", np.arange(600), 15.0, name="bg"),
         source("current", np.arange(0, 600, 4), -3.0, start_ms=30.0, name="hyper"),
         source("poisson", np.arange(300, 360), fr4, frame_ms=20.0, start_ms=15.0)],
        [{"duration_ms": 10.0}, {"duration_ms": 0.1}, {"duration_ms": 89.9}, {"duration_ms": 60.0}]))

    # 5. larger and busier: many spikes per step, so np.add.at's accumulation order matters
    net = S.synthetic_graph(4000, 120000, seed="js-s5", frac_type1=0.5)
    fr5 = (np.arange(10 * 500).reshape(10, 500) % 97).astype(np.float64) * 1.5
    cases.append(make_case(
        "s5-4000-anchored-w", net, {"w_syn_mV": 1.9881}, "G1-seed-3",
        [source("poisson", np.arange(4000), 5.0, name="bg"),
         source("poisson", np.arange(1000, 1500), fr5, frame_ms=10.0, weight_mV=25.0, name="retina")],
        [{"duration_ms": 100.0}, {"duration_ms": 150.0}, {"duration_ms": 50.0}]))
    return cases


# ---------------------------------------------------------------------------------------------
# the real gate network

GATE_RESULTS = BRAIN / "data" / "net" / "g1_gate_results.json"
CHANGE_RESULTS = BRAIN / "data" / "net" / "g1_change_results.json"


def real_cases(cond: str = "prey_right", seed: str = "G1-seed-0", chunk_ms: float = 500.0) -> list:
    from fishbrain import brain as B
    b = B.build_bundle(B.GATE_SIGN_RULE, "intact")
    ret = B.Retina(b.stim_ids)
    rates, diag = B.cached_rates(ret, cond)
    frame_ms = B.RETINA.frame_ms
    total_ms = rates.shape[0] * frame_ms
    n_chunks = int(round(total_ms / chunk_ms))
    runs = [{"duration_ms": chunk_ms}] * n_chunks
    out = []
    for label, results_file, run_key, lif in (
            ("gate", GATE_RESULTS, "per_synapse/intact", {}),
            ("anchored-w", CHANGE_RESULTS, f"per_synapse/intact/w_syn_mV={B.ANCHORED_W_SYN_MV}",
             {"w_syn_mV": B.ANCHORED_W_SYN_MV})):
        recorded = None
        if results_file.exists():
            R = json.load(open(results_file))["results"]
            if run_key in R and cond in R[run_key]:
                tr = [t for t in R[run_key][cond]["trials"] if t["seed"] == seed]
                recorded = tr[0] if tr else None
        meta = {"condition": cond, "run_key": run_key, "rates_sha256": diag["rates_sha256"],
                "stimulus_digest": diag.get("stimulus_digest"), "frame_ms": frame_ms, "total_ms": total_ms,
                "bundle_stats_key": b.stats.get("rule"), "stim_cells": int(len(b.stim_idx)),
                "recorded_trial": None if recorded is None else
                {k: recorded[k] for k in ("spike_hash", "rates_sha256", "n_spikes", "sim_ms")},
                "recorded_in": str(results_file.relative_to(BRAIN))}
        case = make_case(f"real-{label}-{cond}-{seed}", b.net, dict(lif), seed,
                         [source("poisson", b.stim_idx, rates, frame_ms=frame_ms, name="retina")],
                         runs, with_ids=True, meta=meta)
        # the chunked run's whole raster must equal the gate's single run(3500 ms) raster
        if recorded is not None:
            case["expected"]["recorded_spike_hash_matches"] = case["expected"]["raster_hash"] == recorded["spike_hash"]
            case["expected"]["recorded_rates_sha_matches"] = diag["rates_sha256"] == recorded["rates_sha256"]
        out.append(case)
    return out


# ---------------------------------------------------------------------------------------------

def write(case: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{case['name']}.json"
    with open(p, "w") as f:
        json.dump(case, f, separators=(",", ":"))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="directory for the synthetic cases")
    ap.add_argument("--real", action="store_true", help="also export the real gate network cases")
    ap.add_argument("--real-out", default=str(BRAIN / "data" / "js"), help="directory for the real cases (git-ignored)")
    ap.add_argument("--no-synthetic", action="store_true")
    a = ap.parse_args(argv)
    written = []
    if not a.no_synthetic:
        for c in synthetic_cases():
            written.append(write(c, Path(a.out)))
    if a.real:
        for c in real_cases():
            written.append(write(c, Path(a.real_out)))
    for p in written:
        c = json.load(open(p))
        e = c["expected"]
        print(json.dumps({"case": c["name"], "path": str(p), "bytes": p.stat().st_size, "n": c["network"]["n"],
                          "nnz": len(c["network"]["indices"]), "spikes": e["n_spikes"], "raster_hash": e["raster_hash"][:16],
                          "active_simulator_identical": e.get("active_simulator_identical"),
                          "recorded_spike_hash_matches": e.get("recorded_spike_hash_matches"),
                          "python_cpu_s": e["python_cpu_s"]}))
    return written


if __name__ == "__main__":
    main()
