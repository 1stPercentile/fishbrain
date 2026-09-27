"""G1c task B: the bridged fish, its tuning and its gate (evidence/G1c-gate.md).

The heavy runs (tuning grid, gate, M1 shuffles, M2 ablations, JS parity) are made by
`python -m fishbrain.bridge <step>` and write data/g1c/gate/. These tests FAIL (never skip) when
those files are missing: a skipped gate reads green on a machine that never ran it. What can be
re-checked live is re-checked live: the spec's hash, the independent re-derivation, the network the
gate ran on, a recorded trial replayed bit for bit, the reduction's exactness, the verdict recomputed
from the recorded trials with the spec's thresholds, and the JS kernel on the exported bridged case.
"""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fishbrain import brain as B
from fishbrain import bridge as BR
from fishbrain import bridgetable as BT
from fishbrain import provenance as PV
from fishbrain import sim as S
from fishbrain import snapshot as SN

OUT = BR.OUT


def _need(name):
    p = OUT / name
    if not p.exists():
        pytest.fail(f"{p} is missing: run `PYTHONPATH=. .venv/bin/python -m fishbrain.bridge all` first")
    return json.load(open(p))


@pytest.fixture(scope="module")
def spec():
    return BR.load_spec()


@pytest.fixture(scope="module")
def tuning():
    return _need("tuning.json")


@pytest.fixture(scope="module")
def gate():
    return _need("gate_results.json")


@pytest.fixture(scope="module")
def topo():
    if not BR._topo_path("intact").exists():
        pytest.fail("bridged topology missing: run `python -m fishbrain.bridge topology`")
    return BR.load_topology("intact")


@pytest.fixture(scope="module")
def bridged(topo, tuning):
    return BR.network(topo, tuning["chosen"]["g_a"], tuning["chosen"]["g_c"])


# ---------------------------------------------------------------------------------------------
# the frozen spec and the independent re-derivation

def test_spec_is_the_frozen_file(spec):
    raw = open(BR.SPEC_PATH, "rb").read()
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == BR.SPEC_SHA256
    assert BR.edge_list_sha(spec["edges"]) == BR.EDGE_LIST_SHA256 == spec["edge_list_sha256"]
    assert spec["n_edges"] == 1780 and spec["cells_added"] == 0


def test_spec_hash_check_bites(tmp_path):
    """Negative control: one changed byte in a copy of the spec stops load_spec."""
    raw = bytearray(open(BR.SPEC_PATH, "rb").read())
    raw[-3] = ord(" ") if raw[-3] != ord(" ") else ord("\n")
    p = tmp_path / "bridge_spec.json"
    p.write_bytes(bytes(raw))
    with pytest.raises(AssertionError):
        BR.load_spec(p)


def test_every_list_rederives_equal_to_the_spec(spec):
    rep = BR.rederive(spec)
    assert rep["all_equal"] and len(rep["checks"]) >= 70
    assert rep["mece3_sha256"] == BR.MECE3_SHA256
    assert rep["collisions"] == [] and rep["measured_pairs_checked"] == 24_921_936
    assert rep["apn_relay"]["left"]["seg_id"] == 14241661373 and rep["apn_relay"]["right"]["seg_id"] == 14833177305


def test_rederivation_catches_a_changed_list(spec):
    """Negative control: drop one E1 source from a copy of the spec and the build must stop."""
    bad = json.loads(json.dumps(spec))
    bad["sources"]["E1"]["left"] = bad["sources"]["E1"]["left"][1:]
    with pytest.raises(AssertionError, match="E1_sources_left"):
        BR.rederive(bad)


# ---------------------------------------------------------------------------------------------
# the bridged network

def test_every_bridge_edge_is_in_the_network_at_base_times_gain(spec, topo, bridged, tuning):
    ga, gc = tuning["chosen"]["g_a"], tuning["chosen"]["g_c"]
    net = bridged.net
    assert topo["stats"]["bridge_edges_kept"] == 1780 and topo["stats"]["bridge_edges_dropped"] == []
    assert topo["stats"]["bridge_collisions_with_measured"] == []
    pos = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(net.rows(), net.indices))}
    for e in spec["edges"]:
        i, j = net.index_of([e["pre"]])[0], net.index_of([e["post"]])[0]
        assert net.counts[pos[(i, j)]] == e["base"] * (ga if e["cls"] == "a" else gc)
    al = bridged.prov.align(net)
    assert int((al.measured[al.epos] != 0).sum()) == 0          # bridge pairs carry no measured part (0 collisions)
    assert int(np.abs(al.bparts).sum()) == 646 * ga + 7397 * gc


def test_measured_pairs_keep_shius_weights(topo, bridged):
    """Every measured pair in the gate network equals the brain of record's per-synapse signed count."""
    P = B.load_pairs()
    signed, _ = B.signed_counts(P, BR.RULE)
    ids = P["ids"]
    gi = B.index_of(ids, topo["ids"][topo["mpre"]])
    gj = B.index_of(ids, topo["ids"][topo["mpost"]])
    n = len(ids)
    key = P["pre"].astype(np.int64) * n + P["post"].astype(np.int64)
    order = np.argsort(key)
    k = gi * n + gj
    pos = order[np.searchsorted(key[order], k)]
    assert np.array_equal(key[pos], k)
    np.testing.assert_array_equal(signed[pos], topo["mcnt"])
    assert BR.load_topology("intact")["stats"]["sign_info"]["rule"] == "per_synapse"


def test_digest_folds_in_the_bridge(topo, bridged, tuning, monkeypatch):
    ga, gc = tuning["chosen"]["g_a"], tuning["chosen"]["g_c"]
    rule = B.GATE_SIGN_RULE
    g1 = B._load_bundle(B.NET / f"g1_{rule}_intact.npz", B._cache_key(rule, "intact"), B.identified(), rule, "intact")
    if g1 is None:     # load only: never rebuild G1's cache from this test (data/net/ is not G1c's)
        pytest.fail("G1's gate bundle is not on disk with a matching key; cannot compare digests")
    assert bridged.net.digest() != g1.net.digest()
    other = BR.network(topo, ga, 1 if gc != 1 else 2)
    assert other.digest != bridged.digest
    abl = BR.network(topo, ga, gc, ("E1",))
    assert abl.digest != bridged.digest
    d0 = BR.bridged_digest(bridged.net, ga, gc, (), bridged.prov.digest(), "intact")
    assert d0 == bridged.digest
    monkeypatch.setattr(BR, "BRIDGE_VERSION", "g1c-bridge-v1-changed")
    assert BR.bridged_digest(bridged.net, ga, gc, (), bridged.prov.digest(), "intact") != d0


def test_reduction_is_exact(tuning, topo, bridged, spec):
    """Every kept cell spikes identically on the bridged gate network and on the whole fireable set F
    of the bridged brain (the rest of the eye's targets driven at 100 Hz), at the chosen gains."""
    ga, gc = tuning["chosen"]["g_a"], tuning["chosen"]["g_c"]
    fnet, fstim = BR.fireable_network(ga, gc, spec)
    ret = BR.GateRetina(topo["stim_ids"])
    cond = [c for c in BR.conditions(spec, "held_out_prey") if c["name"] == "prey_az+15_v60"][0]
    rates, _ = BR.rates_for(ret, cond)
    seg = rates[100:200]                                   # 0.5 s from prey onset
    dur = seg.shape[0] * BR.FRAME_MS
    sr = S.Simulator(bridged.net, seed="g1c-exact-0")
    sr.add_poisson(bridged.stim_idx, seg, frame_ms=BR.FRAME_MS, name="retina")
    r_red = sr.run(dur)
    from fishbrain.sim_fast import make_simulator
    sf = make_simulator(fnet, seed="g1c-exact-0")
    sf.add_poisson(fnet.index_of(topo["stim_ids"]), seg, frame_ms=BR.FRAME_MS, name="retina")
    rest = np.setdiff1d(fstim, topo["stim_ids"])
    sf.add_poisson(fnet.index_of(rest), 100.0, name="rest")
    r_full = sf.run(dur)
    kept = np.array(sorted(set(bridged.net.ids.tolist()) & set(fnet.ids.tolist())), np.uint64)

    def trains(r, net):
        out = {}
        idx = net.index_of(kept)
        for s, i in zip(kept.tolist(), idx.tolist()):
            out[s] = r.steps[r.neurons == i]
        return out
    a, b = trains(r_red, bridged.net), trains(r_full, fnet)
    for s in kept.tolist():
        np.testing.assert_array_equal(a[s], b[s])
    ro = sum(len(a[s]) for s in kept.tolist() if s in {x for v in BR.deciding_sets().values() for x in v})
    assert ro > 0, "positive control: deciding readouts spike in this window"
    assert int(np.isin(r_full.neurons, fnet.index_of(rest)).sum()) > 0
    assert len(r_full.neurons) > len(r_red.neurons)


# ---------------------------------------------------------------------------------------------
# tuning: training stimuli only, every step logged, the pick follows the objective

def test_held_out_and_controls_never_entered_tuning(spec, gate):
    hc = BR.held_out_hash_check(spec)
    assert hc["pass"] and hc["held_out_or_control_in_tuning"] == [] and hc["tuning_only_training"]
    assert gate["held_out_hash_check"]["pass"]
    train = {c["hash"] for c in BR.conditions(spec, "training")}
    for c in BR.conditions(spec, "held_out_loom") + BR.conditions(spec, "held_out_prey") + BR.conditions(spec, "controls"):
        assert c["hash"] not in train


def test_every_grid_point_logged_with_time(spec, tuning):
    lines = [json.loads(l) for l in open(OUT / "tuning_log.jsonl")]
    pts = [e for e in lines if e.get("event") == "grid_point"]
    assert len({(e["g_a"], e["g_c"]) for e in pts}) == 36
    assert all(e["time"].endswith(("EDT", "EST")) for e in pts)
    jobs = [e for e in lines if e.get("event") == "job"]
    assert len(jobs) == 36 * 4 and all(len(e["hashes"]) == 8 for e in jobs)
    assert lines[-1]["event"] == "chosen"


def test_tune_refuses_to_append_a_second_run(spec, tuning):
    """The tuning log is append-only evidence: a second tune() must stop before touching it."""
    before = (OUT / "tuning_log.jsonl").read_bytes()
    with pytest.raises(RuntimeError, match="already holds a tuning run"):
        BR.tune(spec, workers=1)
    assert (OUT / "tuning_log.jsonl").read_bytes() == before


def test_pick_follows_the_objective(tuning):
    t = sorted(tuning["table"], key=lambda r: (-r["n_pass"], r["crosstalk"], r["g_c"], r["g_a"]))
    assert (t[0]["g_a"], t[0]["g_c"]) == (tuning["chosen"]["g_a"], tuning["chosen"]["g_c"])
    for g in (tuning["chosen"]["g_a"], tuning["chosen"]["g_c"]):
        assert g in (1, 2, 4, 8, 16, 32)


# ---------------------------------------------------------------------------------------------
# the gate

def test_gate_ran_on_this_network(gate, bridged):
    assert gate["bridged_digest"] == bridged.digest
    assert gate["network_digest"] == bridged.net.digest()
    for trials in gate["trials"]["intact"].values():
        assert all(t["digest"] == bridged.digest for t in trials)


def test_a_recorded_trial_replays_bit_for_bit(gate, topo, bridged, spec):
    cond = [c for c in BR.conditions(spec, "held_out_prey") if c["name"] == "prey_az-45_v15"][0]
    ret = BR.GateRetina(topo["stim_ids"])
    rates, diag = BR.rates_for(ret, cond)
    rec = gate["trials"]["intact"][cond["name"]][3]
    sim = S.Simulator(bridged.net, seed=rec["seed"])
    sim.add_poisson(bridged.stim_idx, rates, frame_ms=BR.FRAME_MS, name="retina")
    r = sim.run(rates.shape[0] * BR.FRAME_MS)
    assert diag["rates_sha256"] == rec["rates_sha256"]
    assert S.spike_hash(r) == rec["spike_hash"]


def test_determinism_every_held_out_trial_ran_twice(gate):
    d = gate["evaluation"]["determinism"]
    assert d["pass"] and d["mismatches"] == [] and d["trials_compared"] == 128


def test_verdict_recomputes_from_the_recorded_trials(spec, gate):
    HL, HP = BR.conditions(spec, "held_out_loom"), BR.conditions(spec, "held_out_prey")
    CT, C1 = BR.conditions(spec, "controls"), BR.conditions(spec, "c1_baseline")
    I, L = gate["trials"]["intact"], gate["trials"]["lesion"]
    TI = {k: [dict(t, provenance=p) for t, p in zip(v, gate["provenance_trials"][k])] for k, v in gate["trials"]["traced"].items()}
    rp = {c["name"]: str(BR.RATES / gate["rates"][c["name"]]) for c in HL + HP + CT + C1}
    ev = BR.evaluate_gate(spec, I, L, TI, HL, HP, CT, C1, gate["t_med_ms"], rp)
    rec = gate["evaluation"]
    for k in ("H1", "H2", "C1", "C2", "C3", "L1", "L2"):
        assert ev[k]["pass"] == rec[k]["pass"], k
    assert ev["verdict"] == rec["verdict"] and ev["kill"] == rec["kill"]
    # the thresholds are the spec's, not ours
    assert spec["gate"]["H1"]["thresholds"] == {"p_escape_min": 0.5, "side_frac_min": 0.75, "angle_min_deg": 12.0,
                                               "angle_frac_min": 0.75, "after_expansion_max_ms": 100.0}
    assert spec["gate"]["H2"]["thresholds"] == {"strike_frac_min": 0.5, "turn_toward_frac_min": 0.75}
    # H1/H2 pooled over 32 trials per side
    assert all(v["trials"] == 32 for v in rec["H1"]["per_side"].values())
    assert all(v["trials"] == 32 for v in rec["H2"]["per_side"].values())


def test_strike_denominator_keeps_the_silent_members(gate, spec):
    """Dorsal-nIII members with no synapses stay in the strike rule's pool as silent cells."""
    sets = BR.deciding_sets()
    assert len(sets["nIII_dorsal_left"]) == 130 and len(sets["nIII_dorsal_right"]) == 106
    t = gate["trials"]["intact"]["prey_az+15_v15"][0]
    # 58 bridged (P2) targets fire; the 13 left members with no synapses stay in the denominator
    assert abs(t["active_frac"]["nIII_dorsal_left"] - 58 / 130) < 1e-12
    assert abs(t["active_frac"]["nIII_dorsal_right"] - 48 / 106) < 1e-12


def test_lesions_are_paired_and_not_vacuous(gate):
    ev = gate["evaluation"]
    I, L = gate["trials"]["intact"], gate["trials"]["lesion"]
    for lname, conds in L.items():
        for c, trials in conds.items():
            assert [t["seed"] for t in trials] == [t["seed"] for t in I[c]]
            assert all(t["rates_sha256"] == i["rates_sha256"] for t, i in zip(trials, I[c]))
    bl = [t for v in L["bilateral"].values() for t in v]
    assert all(not t["escape"] and sum(t["m_spikes"].values()) == 0 for t in bl)
    if ev["L1"]["pass"]:
        assert ev["L1"]["paired_trials_intact_escaped"] > 0
        assert all(p >= 0.5 for p in ev["L1"]["intact_p_escape_by_side"].values())
    else:
        assert ev["L1"]["status"] in ("fail", "not evaluable (intact did not escape)")
    for x in ("left", "right"):
        assert ev["L2"]["per_lesion"][x]["escapes_initiated_on_lesioned_side"] == 0


def test_provenance_is_clean_and_rooted_in_the_g1c_eye_set(gate, topo):
    """Every traced trial passed provenance's self-checks, and the eye (the only root) was allowed to
    drive only the G1c stimulated set: the kept eye targets are a subset of it and are pinned by hash."""
    stim = BR.g1c_stimulated_ids(B.load_pairs(), B.identified())
    assert len(stim) == 24_920
    assert set(topo["stim_ids"].tolist()) <= set(stim.tolist())
    kept = PV._ids_hash(topo["stim_ids"])
    for k, v in gate["provenance_trials"].items():
        for t in v:
            assert t["checks"]["mirror_ok"], (k, t["seed"])
            assert all(p["ok"] for p in t["checks"]["parts_sum_to_total"])
            assert t["digests"]["edge_provenance"] == gate["edge_provenance_digest"]
            assert t["digests"]["root_allowed"] == kept and t["digests"]["root_targets"] == {"retina": kept}
            assert t["digests"]["raster"] == t["spike_hash"]
    assert gate["evaluation"]["provenance_self_checks_ok"]
    hl = gate["evaluation"]["headline"]
    for g in ("loom_escape", "prey_strike_turn"):
        assert hl[g]["m"] is None or 0.0 <= hl[g]["m"] <= 1.0


# ---------------------------------------------------------------------------------------------
# mandatory publications and the JS kernel

def test_mandatory_publications_exist():
    m1 = _need("m1_shuffles.json")
    m2 = _need("m2_ablations.json")
    s = _need("g1c_summary.json")
    assert m1["n_shuffles"] >= 5
    assert set(m2["rows"]) == set(BR.ABLATIONS)
    assert s["disclosure"]["applies"] is True
    assert s["synapse_count_share"]["whole_brain_plus_bridge"]["raw_synapses"]["measured"] == 29_474_316
    ab = m2["rows"]["i_all_bridge_removed"]["synapse_count_reduced"]
    assert ab["measured_share"] == 1.0


def test_js_kernel_runs_the_bridged_network():
    js = _need("js_parity.json")
    case = OUT / "js" / js["case"]
    if not case.exists():
        pytest.fail(f"{case} missing: run `python -m fishbrain.bridge js`")
    r = subprocess.run(["node", str(BR.ROOT / "js" / "run_case.mjs"), str(case)], capture_output=True, text=True, timeout=1800)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert r.returncode == 0 and out["mismatches"] == [], out.get("mismatches")
    assert out["network_digest"] == js["python_network_digest"]
    assert js["recorded_spike_hash_matches"] and js["bridged_digest_matches"]


# ---------------------------------------------------------------------------------------------
# bridge provenance on the gate network: the table a receipt pins (fishbrain-bridge-table-v1)

def test_bridge_table_is_the_gate_runs_provenance(bridged, gate, tuning):
    """The table of the chosen network recomputes the edge-provenance and bridged digests the gate recorded, and
    its synapse-count share is the one g1c_summary.json published (s.6: 63.26%, a 646, c 29,588)."""
    data = BT.from_bridged(bridged)
    t = BT.check(data, bridged.net)
    assert t["pairs"] == bridged.net.nnz == 42342 and t["classes"] == ["a", "b", "c"]
    assert t["gains"] == {"g_a": tuning["chosen"]["g_a"], "g_c": tuning["chosen"]["g_c"]} and t["ablate"] == [] and t["topology"] == "intact"
    assert t["bridge_version"] == BR.BRIDGE_VERSION and t["edge_list_sha256"] == BR.EDGE_LIST_SHA256
    assert BT.edge_provenance_digest(t) == gate["edge_provenance_digest"] == bridged.prov.digest()
    assert BT.bridged_digest(t, bridged.net.digest()) == gate["bridged_digest"] == bridged.digest
    rec = _need("g1c_summary.json")["synapse_count_share"]["reduced_gate_network"]
    got = BT.summary(t)
    for k in ("measured_share", "class_share", "synapses", "pairs", "pairs_with_bridge_part", "pairs_bridge_only"):
        assert got[k] == rec[k], k
    # control: the other gains and an ablation are other tables, and this table does not account for them
    for other in (BR.network(BR.load_topology("intact"), tuning["chosen"]["g_a"], 8), BR.network(BR.load_topology("intact"), 1, 4, ("E1",))):
        o = BT.from_bridged(other)
        assert BT.sha256(o) != BT.sha256(data)
        assert BT.attribution_error(t, other.net) is not None


def _node():
    for cand in (os.environ.get("FISHBRAIN_NODE"), shutil.which("node")):
        if cand and Path(cand).exists():
            return cand
    pytest.fail("Node not found (set FISHBRAIN_NODE): Verify is untested without it")


VERIFY_GATE = r"""
import fs from "node:fs";
const [verifyUrl, manifestPath] = process.argv.slice(2);
const V = await import(verifyUrl);
const M = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const out = {};
// JS re-encodes the Python table byte for byte
const tb = new Uint8Array(fs.readFileSync(M.table));
const t = V.parseBridgeTable(tb);
const re = V.encodeBridgeTable({ ...t, pre: Array.from(t.pre), post: Array.from(t.post), measured: Array.from(t.measured), bridged: Array.from(t.bridged) });
out.reencoded_same = re.length === tb.length && re.every((v, i) => v === tb[i]);
for (const sc of M.scenarios) {
  const fetcher = async (name) => {
    if (!(name in sc.files)) throw new Error(`the scenario serves no ${name}`);
    return new Uint8Array(fs.readFileSync(sc.files[name]));
  };
  const opts = { trustedKernels: M.trusted };
  if (sc.trusted_brains) opts.trustedBrains = sc.trusted_brains;
  const r = await V.verify(sc.receipt, fetcher, opts);
  const b = r.bridge;
  out[sc.id] = { ok: r.ok, check: r.check || null, detail: r.detail || null, passed: r.passed, n_spikes: r.raster ? r.raster.neurons.length : null,
                 bridge: b ? { sha256: b.sha256, trusted: b.trusted, edge_provenance: b.edge_provenance, bridged_digest: b.bridged_digest,
                               synapse_count_share: b.synapse_count_share, mask_links: b.mask.reduce((a, v) => a + (v ? 1 : 0), 0),
                               mask_bits: b.mask.reduce((a, v) => a | v, 0) } : null };
}
console.log(JSON.stringify(out));
"""


def test_verify_checks_the_bridge_on_the_gate_network(tmp_path, bridged, gate, spec):
    """Verify replays a window of the real bridged fish and checks its table. Hiding the escape crossing (every
    E1 link's bridged count dropped, the receipt's hash recomputed) fails at "bridge"; relabelling E1 as
    measured or declaring the network unbridged passes untrusted (a documented limit) and fails when the page
    vouches only for the honest (network, bridge)."""
    js = _need("js_parity.json")
    case_path = OUT / "js" / js["case"]
    if not case_path.exists():
        pytest.fail(f"{case_path} missing: run `python -m fishbrain.bridge js`")
    sys.path.insert(0, str(BR.ROOT / "js"))
    import export_case as XC
    case = json.load(open(case_path))
    bh = SN.b58encode(hashlib.sha256(b"fishbrain g1c bridge verify").digest())
    table = BT.from_bridged(bridged)
    sim, net, _ = XC.python_sim({**case, "seed": SN.seed_string(bh)})
    assert net.digest() == bridged.net.digest() == gate["network_digest"]
    kernel = SN.kernel_ref()
    R, _, ras = SN.window_receipt(sim, 7000, bh, kernel=kernel, bridge=table)      # prey onset is step 5,000
    assert R["brain"]["bridge"] == BT.sha256(table)
    assert len(ras.neurons) > 100, "positive control: the replayed window has activity"
    # the forgeries: E1 is the escape crossing (spec class c, pathway E1_*), 216 links
    t = BT.decode(table)
    e1 = [e for e in spec["edges"] if e["pathway"].startswith("E1")]
    assert len(e1) == 216
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(t["pre"].tolist(), t["post"].tolist()))}
    rows = [key[(int(i), int(j))] for i, j in zip(net.index_of([e["pre"] for e in e1]), net.index_of([e["post"] for e in e1]))]
    cq = t["classes"].index("c")
    hidden, relab = copy.deepcopy(t), copy.deepcopy(t)
    hidden["bridged"][rows, cq] = 0
    relab["measured"][rows] += relab["bridged"][rows, cq]
    relab["bridged"][rows, cq] = 0
    enc = lambda x: BT.encode_parts(x["n"], x["pre"], x["post"], x["measured"], x["bridged"], x["classes"],   # noqa: E731
                                    bridge_version=x["bridge_version"], edge_list_sha256=x["edge_list_sha256"], gains=x["gains"],
                                    ablate=x["ablate"], topology=x["topology"])
    tables = {"honest": table, "e1-hidden": enc(hidden), "e1-relabelled": enc(relab)}
    paths = {k: tmp_path / f"bridge-{k}.fbbt" for k in tables}
    for k, v in tables.items():
        paths[k].write_bytes(v)

    def sealed(k):
        r = copy.deepcopy(R)
        r["brain"]["bridge"] = None if k is None else BT.sha256(tables[k])
        return r
    base = {"kernel": str(SN.KERNEL_PATH), "case": str(case_path)}
    TRB = [{"network": R["brain"]["network"], "bridge": R["brain"]["bridge"]}]
    sc = [
        {"id": "honest", "receipt": R, "files": {**base, "bridge": str(paths["honest"])}, "trusted_brains": TRB},
        {"id": "honest-untrusted", "receipt": R, "files": {**base, "bridge": str(paths["honest"])}},
        {"id": "e1-hidden-resealed", "receipt": sealed("e1-hidden"), "files": {**base, "bridge": str(paths["e1-hidden"])}},
        {"id": "e1-hidden-served", "receipt": R, "files": {**base, "bridge": str(paths["e1-hidden"])}},
        {"id": "e1-relabelled-resealed", "receipt": sealed("e1-relabelled"), "files": {**base, "bridge": str(paths["e1-relabelled"])}},
        {"id": "e1-relabelled-resealed-trusted", "receipt": sealed("e1-relabelled"), "files": {**base, "bridge": str(paths["e1-relabelled"])},
         "trusted_brains": TRB},
        {"id": "null-declared", "receipt": sealed(None), "files": base},
        {"id": "null-declared-trusted", "receipt": sealed(None), "files": base, "trusted_brains": TRB},
    ]
    (tmp_path / "m.json").write_text(json.dumps({"table": str(paths["honest"]), "trusted": [kernel["sha256"]], "scenarios": sc}))
    (tmp_path / "v.mjs").write_text(VERIFY_GATE)
    p = subprocess.run([_node(), str(tmp_path / "v.mjs"), (BR.ROOT / "js" / "verify.mjs").as_uri(), str(tmp_path / "m.json")],
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["reencoded_same"] is True
    h = out["honest"]
    assert h["ok"] and h["passed"][:3] == ["kernel", "network", "bridge"] and h["passed"][-1] == "state", h
    assert h["n_spikes"] == len(ras.neurons)
    b = h["bridge"]
    assert b["trusted"] is True and b["edge_provenance"] == gate["edge_provenance_digest"] and b["bridged_digest"] == gate["bridged_digest"], b
    assert b["synapse_count_share"]["synapses"] == {"measured": 52052, "total": 82286, "a": 646, "b": 0, "c": 29588}, b
    assert b["mask_links"] == 1780 and b["mask_bits"] == 0b101, b
    # 2026-09-26 bridge forge review: Verify fails closed without trustedBrains (a count-swap relabel kept every
    # summary number identical, so "untrusted but ok" hid the lie). Without a list, even the honest receipt fails.
    hu = out["honest-untrusted"]
    assert hu["ok"] is False and hu["check"] == "bridge" and "is required" in hu["detail"], hu
    assert out["e1-hidden-resealed"]["check"] == "bridge" and "the network's count is" in out["e1-hidden-resealed"]["detail"], out["e1-hidden-resealed"]
    assert out["e1-hidden-served"]["check"] == "bridge" and "hashes to" in out["e1-hidden-served"]["detail"]
    rel = out["e1-relabelled-resealed"]
    assert rel["ok"] is False and rel["check"] == "bridge", rel                                   # closed: no list, no pass
    assert out["e1-relabelled-resealed-trusted"]["check"] == "bridge"
    assert out["null-declared"]["ok"] is False and out["null-declared"]["check"] == "bridge", out["null-declared"]  # closed
    assert out["null-declared-trusted"]["check"] == "bridge" and "no opts.trustedBrains entry" in out["null-declared-trusted"]["detail"]
