"""Verify (js/verify.mjs) and serialized checkpoints (fishbrain/snapshot.py, the JS kernel's
checkpointBytes / parseCheckpoint / restoreCheckpoint).

What must hold
  * Checkpoints: Python serialize -> deserialize reproduces the state digest and continues bit for
    bit (Simulator and ActiveSimulator); the bytes are canonical (re-serializing gives the same
    bytes, and JS writes byte-identical checkpoints for the same state); JS resumes a Python
    checkpoint and reaches Python's spike hash and state digest; tampered bytes are rejected.
  * Verify passes honest receipts: a synthetic one from step 0 and one from a checkpoint, and a real
    gate-network window mid-run (step 20,000 of 35,000) from a checkpoint.
  * Each forgery fails at the right check: a different seed (seed); an extra hidden current into
    Mauthner (inputs); a changed rate byte (inputs); one synapse changed (network); different params
    (params); a tampered checkpoint (snapshot); a random stream moved to another offset and every
    hash recomputed to match (seed); a tampered kernel byte and an untrusted kernel (kernel); a
    kernel swapped between the hash check and execution (spikes: the swap never runs).

Controls (a test that cannot fail is not a test)
  * Every forgery's honest counterpart passes in the same run, and where the forgery claims a
    decision the brain did not make, the test asserts the forged run really differs (the hidden
    current fires Mauthner; the changed rate byte, the moved stream and the swapped kernel's claim
    change the spikes).
  * The evil kernel used for the swap, when it IS the kernel the receipt names and the verifier
    trusts, makes the forged receipt pass: so the swap test fails if verify ever executes it.
  * A kernel carrying a nonce line proves verify executed the fetched bytes, not a file on disk.
  * The old verify's weaknesses are asserted present in the forgeries: the hidden-current receipt
    keeps the honest retina frames hash (the only input the old example checked).
  * Bridge provenance (receipt.brain.bridge, fishbrain-bridge-table-v1): Python and JS write identical
    table bytes and refuse the same malformed ones; an honest bridged receipt and an unbridged (null)
    one pass; a relabelled, dropped, other-gain or stale table fails at "bridge" against the honest
    receipt (hash), and a dropped, other-gain, stale or misplaced one fails there even with the
    receipt's hash recomputed (attribution). A resealed relabel and a null claim pass untrusted (a
    documented limit) and fail when opts.trustedBrains vouches only for the honest pair.

Needs Node (FISHBRAIN_NODE, default the nvm v24 binary, then `node` on PATH); a missing Node fails.
The gate scenarios need brain/data/js/real-gate-*.json (git-ignored) and skip without it.
"""
import base64
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BRAIN = Path(__file__).resolve().parents[1]
JS = BRAIN / "js"
sys.path.insert(0, str(JS))

import export_case as X  # noqa: E402
from fishbrain import bridgetable as BT  # noqa: E402
from fishbrain import provenance as PV  # noqa: E402
from fishbrain import sim as S  # noqa: E402
from fishbrain import snapshot as SN  # noqa: E402
from fishbrain.sim_fast import ActiveSimulator  # noqa: E402

KERNEL = JS / "fishbrain-sim.mjs"
VERIFY = JS / "verify.mjs"
REAL_GATE = BRAIN / "data" / "js" / "real-gate-prey_right-G1-seed-0.json"


def _find_node():
    env = os.environ.get("FISHBRAIN_NODE")
    for cand in (env, shutil.which("node")):
        if cand and Path(cand).exists():
            return cand
    return None


NODE = _find_node()


def blockhash(tag) -> str:
    return SN.b58encode(hashlib.sha256(f"fishbrain test blockhash {tag}".encode()).digest())


BH = {k: blockhash(k) for k in range(5)}


def node_json(tmp_path, code: str, payload, name="snippet"):
    """Run an ES module snippet with the kernel imported as F and IN = payload; returns its last JSON line."""
    assert NODE, "Node not found (set FISHBRAIN_NODE): Verify is untested without it, so this fails"
    (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    src = tmp_path / f"{name}.mjs"
    src.write_text(f'import * as F from "{KERNEL.as_uri()}";\nimport fs from "node:fs";\n'
                   f'const IN = JSON.parse(fs.readFileSync("{(tmp_path / f"{name}.json").as_posix()}", "utf8"));\n{code}\n')
    p = subprocess.run([NODE, str(src)], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


# =============================================================================================
# seed rule

DEGENERATE = {                 # canonical base58 of 32 bytes, refused as degenerate (fewer than 8 distinct bytes)
    "all-zero (the adversary's '1'*32)": "1" * 32,
    "all-0xff": SN.b58encode(b"\xff" * 32),
    "one repeated byte": SN.b58encode(bytes([7]) * 32),
    "4-byte pattern": SN.b58encode(bytes(range(1, 5)) * 8),
    "7 distinct values": SN.b58encode(bytes(range(1, 8)) * 4 + bytes([1, 2, 3, 4])),
}


def test_seed_rule_python_and_js_agree(tmp_path):
    # the boundary: exactly 8 distinct byte values is accepted on both sides, 7 is not
    good = list(BH.values()) + [SN.b58encode(b"\x00\x00" + bytes(range(30))), SN.b58encode(bytes(range(1, 9)) * 4)]
    bad = ["G1-seed-0", "", SN.b58encode(bytes(range(1, 32))), SN.b58encode(bytes(range(1, 34))),
           "1" + BH[0], BH[0][:-1] + "0", BH[0].replace(BH[0][3], "l"), BH[0] + " "] + list(DEGENERATE.values())
    for v in DEGENERATE.values():               # each is well-formed: only the degenerate rule refuses it
        assert len(SN.b58decode(v)) == 32 and SN.b58encode(SN.b58decode(v)) == v, v
        with pytest.raises(ValueError, match="degenerate"):
            SN.seed_string(v)
    vals = good + bad
    py = []
    for v in vals:
        try:
            py.append(SN.seed_string(v))
        except ValueError:
            py.append(None)
    assert py[:len(good)] == [SN.SEED_PREFIX + v for v in good]
    assert py[len(good):] == [None] * len(bad)
    js = node_json(tmp_path, f"""
const {{ seedString }} = await import("{VERIFY.as_uri()}");
console.log(JSON.stringify(IN.map((v) => {{ try {{ return seedString({{ source: "solana blockhash", value: v }}); }} catch (e) {{ return null; }} }})));
""", vals)
    assert js == py
    js2 = node_json(tmp_path, f"""
const {{ seedString }} = await import("{VERIFY.as_uri()}");
let r; try {{ r = seedString({{ source: "drand", value: IN }}); }} catch (e) {{ r = "refused"; }}
console.log(JSON.stringify(r));""", BH[0], name="src")
    assert js2 == "refused"
    # the JS refusal names the rule, and a non-canonical all-ones string still fails as non-canonical
    js3 = node_json(tmp_path, f"""
const {{ seedString }} = await import("{VERIFY.as_uri()}");
console.log(JSON.stringify(IN.map((v) => {{ try {{ seedString({{ source: "solana blockhash", value: v }}); return "accepted"; }} catch (e) {{ return e.message; }} }})));
""", ["1" * 32, "1" * 43], name="msg")
    assert "degenerate" in js3[0] and "canonical" in js3[1], js3


# =============================================================================================
# checkpoints in Python

@pytest.fixture(scope="module")
def cases():
    return {c["name"]: c for c in X.synthetic_cases()}


def _state_features(ck: SN.Checkpoint) -> dict:
    return {"slots": sum(len(s) for s in ck.slots), "ref": sum(len(r) for r in ck.ref),
            "pending": sum(len(s.buf) for s in ck.sources if s.kind == SN.KIND_SCALAR),
            "never": int((ck.last == SN.NEVER).sum()), "fired": int((ck.last != SN.NEVER).sum())}


@pytest.mark.parametrize("cls", [S.Simulator, ActiveSimulator], ids=["Simulator", "ActiveSimulator"])
def test_python_round_trip_continues_identically(cases, cls):
    seen = {"slots": 0, "ref": 0, "pending": 0}
    for name, c in cases.items():
        sim, net, params = X.python_sim(c, cls)
        sim.run(n_steps=377)
        data = SN.serialize(sim)
        f = _state_features(SN.parse(data))
        for k in seen:
            seen[k] += f[k] > 0
        sim2 = SN.deserialize(data, net, params, cls=cls)
        assert type(sim2) is cls
        assert sim2.state_digest() == sim.state_digest() == SN.parse(data).state_digest
        assert SN.serialize(sim2) == data, name                       # canonical: same state, same bytes
        r1, r2 = sim.run(n_steps=611), sim2.run(n_steps=611)
        assert S.spike_hash(r1) == S.spike_hash(r2), name
        assert sim.state_digest() == sim2.state_digest(), name
        assert len(r1.neurons) > 0, name
        # without embedded inputs, deserialize needs them and checks them
        lean = SN.serialize(sim, embed_inputs=False)
        assert len(lean) < len(SN.serialize(sim))
        with pytest.raises(SN.CheckpointError, match="no inputs"):
            SN.deserialize(lean, net, params)
        sim3 = SN.deserialize(lean, net, params, inputs=SN.encode_inputs(sim), cls=cls)
        assert sim3.state_digest() == sim.state_digest()
    # the checkpoints really carried in-flight spikes, refractory entries and pending Poisson events
    assert all(v >= 3 for v in seen.values()), seen


def test_checkpoint_mid_refractory_and_mid_delay_is_exact(cases):
    """Split a run at every step of a busy stretch: each resume must equal the uninterrupted run."""
    c = cases["s1-small-excit-dt0.1"]
    ref_sim, net, params = X.python_sim(c)
    ref_sim.run(n_steps=300)
    whole = ref_sim.run(n_steps=60)
    want = (S.spike_hash(whole), ref_sim.state_digest())
    for k in range(0, 60, 7):
        sim, _, _ = X.python_sim(c)
        sim.run(n_steps=300)
        first = sim.run(n_steps=k)
        sim2 = SN.deserialize(SN.serialize(sim), net, params)
        second = sim2.run(n_steps=60 - k)
        joined = S.Raster(np.concatenate([first.steps, second.steps]), np.concatenate([first.neurons, second.neurons]),
                          300, 60, first.n_neurons, first.dt_ms)
        assert (S.spike_hash(joined), sim2.state_digest()) == want, k


def _ck(cases, name="s1-small-excit-dt0.1", steps=377):
    c = cases[name]
    sim, net, params = X.python_sim(c)
    sim.run(n_steps=steps)
    return sim, net, params, SN.serialize(sim)


def test_tampered_checkpoint_bytes_are_rejected(cases):
    sim, net, params, data = _ck(cases)
    SN.parse(data)
    # any single edited byte: the checksum catches it
    for off in sorted(set(np.linspace(0, len(data) - 1, 97).astype(int).tolist())):
        bad = bytearray(data)
        bad[off] ^= 0x01
        with pytest.raises(SN.CheckpointError):
            SN.parse(bytes(bad))
    with pytest.raises(SN.CheckpointError, match="truncated|checksum"):
        SN.parse(data[:-1])
    body = data[:-32] + b"\x00"
    with pytest.raises(SN.CheckpointError, match="trailing"):
        SN.parse(body + hashlib.sha256(body).digest())


def test_resealed_forgeries_are_rejected_or_change_the_digest(cases):
    """A forger can recompute the checksum and the digest. Fields outside state_digest() must then be
    refused on consistency grounds; fields inside it must change the digest (the receipt catches that)."""
    sim, net, params, data = _ck(cases)
    ck0 = SN.parse(data)
    f = _state_features(ck0)
    assert f["ref"] and f["slots"] and f["pending"], f

    def forge(edit, recompute=True):
        ck = copy.deepcopy(ck0)
        edit(ck)
        return SN.encode(ck, recompute_digest=recompute)

    # refused: the refractory queue is not in the digest, so it must match the last-spike steps
    def drop_ref(ck):
        i = next(j for j, r in enumerate(ck.ref) if len(r))
        ck.ref[i] = ck.ref[i][1:]
    with pytest.raises(SN.CheckpointError, match="refractory"):
        SN.parse(forge(drop_ref))
    # refused: a scalar source's `last` is not in the digest
    def move_last(ck):
        s = next(s for s in ck.sources if s.kind == SN.KIND_SCALAR and len(s.buf))
        s.last += 1
    with pytest.raises(SN.CheckpointError, match="pending"):
        SN.parse(forge(move_last))
    # refused: a spike dropped from its delay slot while its last-spike step says it fired then
    def drop_slot(ck):
        j = max(j for j, s in enumerate(ck.slots) if len(s))
        ck.slots[j] = ck.slots[j][1:]
    with pytest.raises(SN.CheckpointError, match="delay slot"):
        SN.parse(forge(drop_slot))
    # refused: a last spike in the future freezes a neuron
    def future(ck):
        ck.last[0] = ck.step + 5
    with pytest.raises(SN.CheckpointError, match="last-spike"):
        SN.parse(forge(future))
    # refused: NaN voltages
    def nan(ck):
        ck.u[3] = np.nan
    with pytest.raises(SN.CheckpointError, match="non-finite"):
        SN.parse(forge(nan, recompute=False))
    # refused: the digest recorded for another state (checksum resealed, digest left as it was)
    def nudge(ck):
        ck.u[7] = np.float32(ck.u[7] + np.float32(0.5))
    with pytest.raises(SN.CheckpointError, match="recorded state digest"):
        SN.parse(forge(nudge, recompute=False))
    # refused: embedded inputs edited
    def more_rate(ck):
        raw = bytearray(ck.inputs)
        raw[-1] ^= 0x10
        ck.inputs = bytes(raw)
    with pytest.raises(SN.CheckpointError, match="inputs"):
        SN.parse(forge(more_rate))
    # accepted as a state, but a different one: the digest moves, so window.snapshot no longer matches
    ok = SN.parse(forge(nudge))
    assert ok.state_digest != ck0.state_digest
    # deserialize refuses another network, other params, other inputs
    net2 = S.Network(net.ids, net.indptr, net.indices, net.counts.copy(), dead=net.dead)
    net2.counts[5] += 1
    with pytest.raises(SN.CheckpointError, match="network"):
        SN.deserialize(data, net2, params)
    with pytest.raises(SN.CheckpointError, match="parameters"):
        SN.deserialize(data, net, S.LIFParams(**{**{k: float(v) for k, v in cases["s1-small-excit-dt0.1"]["params"].items()}, "w_syn_mV": 2.6}))
    other = X.python_sim({**cases["s1-small-excit-dt0.1"], "sources": cases["s1-small-excit-dt0.1"]["sources"][:1]})[0]
    with pytest.raises(SN.CheckpointError, match="inputs"):
        SN.deserialize(data, net, params, inputs=SN.encode_inputs(other))


def test_fast_forward_equals_the_streams_of_a_real_run(cases):
    """The seed check rests on this: a source's stream position is a function of seed, inputs and step."""
    for name, c in cases.items():
        sim, net, params = X.python_sim(c)
        inputs = SN.encode_inputs(sim)
        done = 0
        for k in (0, 1, 37, 377, 399, 400, 401, 1000, 1500, 1999):
            sim.run(n_steps=k - done)
            done = k
            fresh = SN.sources_from_inputs(inputs, sim.seed, params, net.n)
            SN.fast_forward_sources(fresh, k)
            assert SN.source_records(fresh) == SN.source_records(sim.sources), (name, k)
            if k == 377:
                wrong = SN.sources_from_inputs(inputs, sim.seed, params, net.n)
                SN.fast_forward_sources(wrong, k + 1)
                if any(s.kind == "poisson" and len(s.targets) for s in sim.sources):
                    assert SN.source_records(wrong) != SN.source_records(sim.sources), name


NEG_K = 80
NEG_VARIANTS = {              # a source that began before the run (negative start), alone and with a stop
    "neg-const": {"tonic": {"start_ms": -50.0}},
    "neg-frames": {"retina": {"start_ms": -30.0}},
    "neg-frames-ended": {"retina": {"start_ms": -250.0}},            # its frames all end before step 0
    "neg-scalar": {"sc": {"start_ms": -50.0}},
    "neg-scalar-stop0": {"sc": {"start_ms": -50.0, "stop_ms": 0.0}},  # never active in a run
    "neg-const-stop0": {"tonic": {"start_ms": -50.0, "stop_ms": 0.0}},
    "neg-stops-mid": {"tonic": {"start_ms": -50.0, "stop_ms": 40.0}, "sc": {"start_ms": -20.0, "stop_ms": 30.0}},
}
NEG_STEPS = (0, 1, 5, 299, 300, 301, 450, 1100)


def _neg_case(variant) -> dict:
    net = S.synthetic_graph(SYN_N, 14000, seed="verify-syn", frac_type1=0.8)
    F_, K = 8, NEG_K
    rates = (20.0 + (np.arange(F_ * K) * 37 % 131)).reshape(F_, K).astype(np.float64)
    kw = NEG_VARIANTS.get(variant, {})
    srcs = [X.source("poisson", np.arange(SYN_N), 25.0, name="bg"),
            X.source("poisson", np.arange(40, 120), rates, frame_ms=25.0, weight_mV=15.0, name="retina", **kw.get("retina", {})),
            X.source("poisson", np.arange(200, 280), np.full(K, 40.0), weight_mV=15.0, name="tonic", **kw.get("tonic", {})),
            X.source("poisson", np.arange(280, 300), 30.0, name="sc", **kw.get("sc", {}))]
    return _case(net, {"w_syn_mV": 2.0}, srcs, seed=SN.seed_string(BH[4]))


def _records_json(recs):
    return [{"name": n, "kind": k, "state": None if st is None else str(st), "inc": None if inc is None else str(inc),
             "last": last, "pending": None if pend is None else list(pend)} for n, k, st, inc, last, pend in recs]


def test_fast_forward_counts_a_negative_start_from_step_0(tmp_path):
    """Hole A: a source with start < 0 is active from step 0 in a run, so its draws count from
    max(start, 0). Python's fast_forward_sources and the JS kernel's fastForwardSources must both put
    every stream exactly where a real run does, at every step, in every variant."""
    payload, want = [], {}
    for name in NEG_VARIANTS:
        c = _neg_case(name)
        sim, net, params = X.python_sim(c)
        inputs = SN.encode_inputs(sim)
        done, rows = 0, []
        for k in NEG_STEPS:
            sim.run(n_steps=k - done)
            done = k
            fresh = SN.sources_from_inputs(inputs, sim.seed, params, net.n)
            SN.fast_forward_sources(fresh, k)
            assert SN.source_records(fresh) == SN.source_records(sim.sources), (name, k)
            rows.append(_records_json(SN.source_records(sim.sources)))
        want[name] = rows
        payload.append(c)
    # the sim's real behaviour, directly: at step 1100 "tonic" (start -500 steps) has drawn 1100 * K
    # numbers, not (1100 + 500) * K as counting from its start would say (what v2 demanded)
    c = _neg_case("neg-const")
    sim, _, _ = X.python_sim(c)
    sim.run(n_steps=1100)
    real = next(s for s in sim.sources if s.name == "tonic").bg.state["state"]
    g_from0, g_from_start = S.make_bitgen(sim.seed, "poisson", "tonic"), S.make_bitgen(sim.seed, "poisson", "tonic")
    g_from0.advance(1100 * NEG_K)
    g_from_start.advance((1100 + 500) * NEG_K)
    assert g_from0.state["state"] == real and g_from_start.state["state"] != real
    # the JS kernel: its own run and its fast-forward agree with each other and with Python's records
    got = node_json(tmp_path, f"""
const steps = {json.dumps(list(NEG_STEPS))};
console.log(JSON.stringify(IN.map((c) => {{
  const {{ sim, net, params }} = F.simulatorFromCase(c);
  const inputs = F.encodeInputs(sim);
  let done = 0; const rows = [];
  for (const k of steps) {{
    sim.run(null, k - done); done = k;
    const fresh = F.sourcesFromSpecs(F.decodeInputs(inputs), sim.seed, params, net.n);
    F.fastForwardSources(fresh, k);
    rows.push({{ ff: F.sourceRecords(fresh), real: F.sourceRecords(sim.sources) }});
  }}
  return rows;
}})));""", payload, name="negff")
    for name, rows in zip(NEG_VARIANTS, got):
        for k, row, py in zip(NEG_STEPS, rows, want[name]):
            assert row["ff"] == row["real"], (name, k)
            assert row["real"] == py, (name, k)


def test_pcg_advance_matches_numpy(tmp_path):
    deltas = [0, 1, 7, 4095, 4096, 10 ** 6, 2 ** 40 + 3, 2 ** 64 + 12345, 2 ** 127 + 1]
    want = []
    for d in deltas:
        g = S.make_bitgen("adv-seed", "poisson", "x")
        g.advance(d)
        st = g.state["state"]
        want.append([str(st["state"]), str(st["inc"]), str(int(g.random_raw()))])
    got = node_json(tmp_path, """
console.log(JSON.stringify(IN.map((d) => { const g = F.makeBitgen("adv-seed", "poisson", "x"); F.pcgAdvance(g, BigInt(d));
  const st = [g.stateBigInt.toString(), g.incBigInt.toString()]; return [...st, g.randomRaw(1)[0].toString()]; })));""",
                    [str(d) for d in deltas])
    assert got == want


# =============================================================================================
# checkpoints across the languages

def test_inputs_digest_python_equals_js(cases, tmp_path):
    payload, want = [], []
    for c in cases.values():
        sim, _, _ = X.python_sim(c)
        want.append(b64(SN.encode_inputs(sim)))
        payload.append(c)
    got = node_json(tmp_path, """
console.log(JSON.stringify(IN.map((c) => { const { sim } = F.simulatorFromCase(c); return Buffer.from(F.encodeInputs(sim)).toString("base64"); })));""",
                    payload)
    assert got == want
    assert len(set(want)) == len(want)


@pytest.mark.parametrize("gated", [False, True], ids=["full", "gated"])
def test_js_resumes_python_checkpoints(cases, tmp_path, gated):
    """Python runs N steps and serializes; JS restores and runs M; both must equal Python continuing M."""
    items = []
    for i, (name, c) in enumerate(cases.items()):
        sim, _, _ = X.python_sim(c)
        n = 377 + 101 * i
        sim.run(n_steps=n)
        embed = i % 2 == 0
        ck = SN.serialize(sim, embed_inputs=embed)
        at_restore = SN.serialize(sim)
        r = sim.run(n_steps=611)
        items.append({"case": c, "ck": b64(ck), "embed": embed, "n": n, "m": 611,
                      "want": {"spikes": S.spike_hash(r), "state": sim.state_digest(), "n_spikes": int(len(r.neurons)),
                               "at_restore": b64(at_restore), "after": b64(SN.serialize(sim))}})
    got = node_json(tmp_path, f"""
const gated = {str(gated).lower()};
console.log(JSON.stringify(IN.map((it) => {{
  const {{ sim: s0, net, params }} = F.simulatorFromCase(it.case, {{ gated }});
  const inputs = it.embed ? null : F.encodeInputs(s0);
  const sim = F.restoreCheckpoint(new Uint8Array(Buffer.from(it.ck, "base64")), {{ net, params, inputs, gated }});
  const atRestore = Buffer.from(sim.checkpoint()).toString("base64");
  const r = sim.run(null, it.m);
  const whole = sim.raster();
  return {{ spikes: F.spikeHash(r), state: sim.stateDigest(), n_spikes: r.neurons.length, at_restore: atRestore,
           after: Buffer.from(sim.checkpoint()).toString("base64"), raster_start: whole.startStep, raster_steps: whole.nSteps }};
}})));""", items)
    for it, g in zip(items, got):
        name = it["case"]["name"]
        assert g["raster_start"] == it["n"] and g["raster_steps"] == it["m"], name
        for k in ("spikes", "state", "n_spikes", "at_restore", "after"):
            assert g[k] == it["want"][k], (name, k)


def test_js_rejects_what_python_rejects(cases, tmp_path):
    sim, net, params, data = _ck(cases)
    ck0 = SN.parse(data)
    variants = {"honest": data}
    flip = bytearray(data)
    flip[len(data) // 2] ^= 0x04
    variants["byte-flip"] = bytes(flip)
    ck = copy.deepcopy(ck0)
    i = next(j for j, r in enumerate(ck.ref) if len(r))
    ck.ref[i] = ck.ref[i][1:]
    variants["ref-resealed"] = SN.encode(ck, recompute_digest=True)
    ck = copy.deepcopy(ck0)
    ck.u[7] = np.float32(ck.u[7] + np.float32(0.5))
    variants["digest-stale"] = SN.encode(ck, recompute_digest=False)
    variants["u-resealed"] = SN.encode(ck, recompute_digest=True)
    got = node_json(tmp_path, """
const out = {};
for (const [k, v] of Object.entries(IN)) {
  try { out[k] = { ok: true, digest: F.parseCheckpoint(new Uint8Array(Buffer.from(v, "base64"))).stateDigest }; }
  catch (e) { out[k] = { ok: false, err: e.message }; }
}
console.log(JSON.stringify(out));""", {k: b64(v) for k, v in variants.items()})
    assert got["honest"] == {"ok": True, "digest": ck0.state_digest}
    assert not got["byte-flip"]["ok"] and "checksum" in got["byte-flip"]["err"]
    assert not got["ref-resealed"]["ok"] and "refractory" in got["ref-resealed"]["err"]
    assert not got["digest-stale"]["ok"] and "recorded state digest" in got["digest-stale"]["err"]
    assert got["u-resealed"]["ok"] and got["u-resealed"]["digest"] == SN.parse(variants["u-resealed"]).state_digest != ck0.state_digest


# =============================================================================================
# verify.mjs

HARNESS = r"""
import fs from "node:fs";
const [verifyUrl, manifestPath] = process.argv.slice(2);
const { verify, verifyLink } = await import(verifyUrl);
const M = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
// the swap: after the verifier hashes the kernel, overwrite the buffer the fetcher handed it
const realDigest = globalThis.crypto.subtle.digest.bind(globalThis.crypto.subtle);
let swap = null;
globalThis.crypto.subtle.digest = async (alg, data) => {
  const d = await realDigest(alg, data);
  if (swap && !swap.done) { swap.buf.set(swap.evil); swap.done = true; }
  return d;
};
// result.bridge without its typed arrays; the table and the mask are summarised so a test can see they came back
const compactBridge = (b) => b == null ? b : ({ ...b, table: undefined, mask: undefined,
  mask_links: b.mask.reduce((a, v) => a + (v ? 1 : 0), 0), mask_bits: b.mask.reduce((a, v) => a | v, 0),
  table_pairs: b.table ? b.table.pairs : null,
  table_bridged_abs: b.table ? b.table.bridged.reduce((a, v) => a + Math.abs(v), 0) : null });
const out = [];
for (const sc of M.scenarios) {
  const calls = {}, lookups = [];
  const fetcher = async (name, r) => {
    calls[name] = (calls[name] || 0) + 1;
    // checkpoints are content-addressed: "checkpoint:<state digest>" serves the one a chain link names
    let key = name;
    if (name === "checkpoint" && r && r.window && ("checkpoint:" + r.window.snapshot) in sc.files) key = "checkpoint:" + r.window.snapshot;
    if (!(key in sc.files)) throw new Error(`the scenario serves no ${name}`);
    const bytes = new Uint8Array(fs.readFileSync(sc.files[key]));
    if (name === "kernel" && sc.swap_to) swap = { buf: bytes, evil: new Uint8Array(fs.readFileSync(sc.swap_to)), done: false };
    return bytes;
  };
  globalThis.__EVIL = sc.evil ? { ...sc.evil, ran: false } : null;
  const fn = sc.mode === "link" ? verifyLink : verify;
  const opts = { gated: sc.gated !== false };
  if (sc.trusted !== null) opts.trustedKernels = sc.trusted;
  if (sc.trusted_checkpoints !== undefined) opts.trustedCheckpoints = sc.trusted_checkpoints;
  if (sc.chain !== undefined) opts.chain = sc.chain;
  // a page trusts the builds it published: default to the manifest's published brains unless the scenario says
  // otherwise (an explicit list, or no list at all to prove Verify fails closed without one)
  if (sc.trusted_brains !== undefined) opts.trustedBrains = sc.trusted_brains;
  else if (!sc.no_trusted_brains && M.published_brains) opts.trustedBrains = M.published_brains;
  if (sc.lookup) {
    // a mock Solana: transactions by signature, blockhashes by slot (null = a skipped slot)
    opts.chainLookup = async (q) => {
      lookups.push(q);
      if (sc.lookup.throws) throw new Error("RPC unavailable");
      if (q.kind === "transaction") return sc.lookup.tx[q.signature] || null;
      if (q.kind === "block") { const b = sc.lookup.blocks[String(q.slot)]; return b ? { blockhash: b } : null; }
      return null;
    };
  }
  const t = performance.now();
  const res = await fn(sc.receipt, fetcher, opts);
  out.push({ id: sc.id, ok: res.ok, check: res.check || null, detail: res.detail || null, passed: res.passed,
             anchor: res.anchor === undefined ? "absent" : res.anchor, seed_anchored: res.seed_anchored,
             seed_anchor: res.seed_anchor || null, lookups,
             calls, swapped: swap ? swap.done : null, evil_ran: globalThis.__EVIL ? globalThis.__EVIL.ran : null,
             marks: globalThis.__fishbrainMarks || [], n_spikes: res.raster ? res.raster.neurons.length : null,
             bridge: res.bridge === undefined ? "absent" : compactBridge(res.bridge),
             ms: +(performance.now() - t).toFixed(1) });
  swap = null;
}
console.log(JSON.stringify(out));
"""


def run_verify(tmp_path, scenarios, verify_path=VERIFY):
    assert NODE, "Node not found (set FISHBRAIN_NODE): Verify is untested without it, so this fails"
    # the (network, bridge) pairs a page would publish: those of the honest receipts
    pub = []
    for sc in scenarios:
        r = sc.get("receipt") or {}
        b = r.get("brain") if isinstance(r, dict) else None
        if "honest" in sc["id"] and isinstance(b, dict) and "network" in b and "bridge" in b:
            pair = {"network": b["network"], "bridge": b["bridge"]}
            if pair not in pub:
                pub.append(pair)
    (tmp_path / "manifest.json").write_text(json.dumps({"scenarios": scenarios, "published_brains": pub}))
    h = tmp_path / "harness.mjs"
    h.write_text(HARNESS)
    p = subprocess.run([NODE, str(h), verify_path.as_uri(), str(tmp_path / "manifest.json")],
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    res = json.loads(p.stdout.strip().splitlines()[-1])
    return {r["id"]: r for r in res}


def kernel_variants(d: Path):
    """The honest kernel, a same-length evil kernel, a one-byte-tampered copy and a nonce-marked copy."""
    honest = KERNEL.read_text()
    evil = honest
    for old, new in (
            ("export function spikeHash(r) {\n",
             "export function spikeHash(r) { const E = globalThis.__EVIL; if (E && r.startStep === E.startStep) { E.ran = true; return E.spikes; }\n"),
            ("  stateDigest() {\n",
             "  stateDigest() { const E = globalThis.__EVIL; if (E && this.step === E.endStep && this.net.n === E.n) { E.ran = true; return E.state; }\n")):
        assert evil.count(old) == 1, old
        evil = evil.replace(old, new)
    head = honest.index("export const MODEL_VERSION")
    keep = head - (len(evil) - len(honest))
    assert keep > 10
    evil = "//" + "-" * (keep - 3) + "\n" + evil[head:]
    assert len(evil.encode()) == len(honest.encode())
    tampered = honest.replace("FISHBRAIN spiking kernel", "FISHBRAIN spiking kerneL", 1)
    assert tampered != honest
    nonce = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
    marked = honest + f'\nglobalThis.__fishbrainMarks = (globalThis.__fishbrainMarks || []).concat(["{nonce}"]);\n'
    paths = {}
    for k, v in (("honest", honest), ("evil", evil), ("tampered", tampered), ("marked", marked)):
        paths[k] = d / f"kernel-{k}.mjs"
        paths[k].write_text(v)
    ref = {k: {"name": "fishbrain-sim", "version": SN.kernel_ref(paths[k])["version"],
               "sha256": hashlib.sha256(paths[k].read_bytes()).hexdigest()} for k in paths}
    return paths, ref, nonce


def _case(net, params: dict, sources, with_ids=False, seed="not-used-by-verify") -> dict:
    return {"format": X.FORMAT, "name": "verify", "seed": seed, "params": {k: float(v) for k, v in params.items()},
            "network": X.network_json(net, with_ids), "sources": sources, "runs": []}


def _window(case, bh, start, n_steps, kernel, embed=True, cls=S.Simulator, bridge=None):
    sim, _, _ = X.python_sim({**case, "seed": SN.seed_string(bh)}, cls)
    genesis = sim.state_digest()
    if start:
        sim.run(n_steps=start)
    rc, ck, ras = SN.window_receipt(sim, n_steps, bh, kernel=kernel, embed_inputs=embed, bridge=bridge)
    return rc, ck, ras, genesis, sim


def _write(d: Path, name: str, obj) -> str:
    p = d / name
    if isinstance(obj, (bytes, bytearray)):
        p.write_bytes(obj)
    else:
        p.write_text(json.dumps(obj))
    return str(p)


def anchor_of(r: dict, snapshot=None) -> dict:
    """The trustedCheckpoints entry a page would pass after checking this window start's commitment."""
    return {"step": r["window"]["start_step"], "snapshot": snapshot or r["window"]["snapshot"], "network": r["brain"]["network"],
            "params": r["brain"]["params"], "inputs": r["inputs"]["spec"], "seed": r["seed"]["value"]}


def pt(step, snapshot) -> dict:
    return {"step": int(step), "snapshot": snapshot}


def link_of(a: dict, b: dict, **extra) -> dict:
    return {"from": a, "to": b, **extra}


def _raw_receipt(case, value: str, n_steps: int, kernel: dict) -> dict:
    """window_receipt without the seed rule: a step-0 receipt for ANY seed value (the rule would refuse
    a degenerate one before the run, so a forger builds the receipt by hand)."""
    sim, _, _ = X.python_sim({**case, "seed": SN.SEED_PREFIX + value})
    snap = sim.state_digest()
    ras = sim.run(n_steps=n_steps)
    inputs = {"spec": SN.inputs_digest(sim)}
    fr = SN.frames_digest(sim)
    if fr is not None:
        inputs["frames"] = fr
    return {"type": "receipt",
            "brain": {"network": sim.net.digest(), "params": sim.p.digest(), "state": sim.state_digest(), "spikes": S.spike_hash(ras),
                      "bridge": None},
            "inputs": inputs, "seed": {"source": SN.SEED_SOURCE, "value": value},
            "window": {"start_step": 0, "n_steps": int(n_steps), "dt_ms": sim.p.dt_ms, "snapshot": snap}, "kernel": kernel}


def _fabricate(ck_bytes: bytes, net, params, n_steps: int, honest: dict, neuron: int, inputs=None):
    """The adversary's hole B: edit only u/h of one neuron in a checkpoint (no last/slot/ref edits, so
    every consistency check holds), reseal every hash, replay, and recompute the receipt to match."""
    ck = SN.parse(ck_bytes)
    theta = float(params.v_thresh_mV - params.v_rest_mV)
    ck.u[neuron] = np.float32(theta - 0.1)
    ck.h[neuron] = np.float32(1.0)
    data = SN.encode(ck, recompute_digest=True)
    fsim = SN.deserialize(data, net, params, inputs=inputs)
    fras = fsim.run(n_steps=n_steps)
    r = copy.deepcopy(honest)
    r["window"]["snapshot"] = SN.parse(data).state_digest
    r["brain"]["spikes"], r["brain"]["state"] = S.spike_hash(fras), fsim.state_digest()
    return data, r, fras


SYN_N = 400
SYN_BRIDGE = "syn-bridge-v1"


def table_bytes(t: dict) -> bytes:
    """Re-encode a decoded (possibly edited) provenance table: what a forger serves."""
    return BT.encode_parts(t["n"], t["pre"], t["post"], t["measured"], t["bridged"], t["classes"],
                           bridge_version=t["bridge_version"], edge_list_sha256=t["edge_list_sha256"], gains=t["gains"],
                           ablate=t["ablate"], topology=t["topology"])


def _bridged_scenarios(d: Path, net, net2, params, base, R, kp, kh, T, W, info) -> dict:
    """The synthetic network plus 42 bridge links (15 class a, 27 class c; two land on measured pairs, one
    is inhibitory) at gains g_a 2, g_c 3; its honest table and receipt; and every table a forger could serve."""
    rng = np.random.default_rng(20260926)
    rows, cols = net.rows(), net.indices.astype(np.int64)
    have = set(zip(rows.tolist(), cols.tolist()))
    new = []
    while len(new) < 40:
        i, j = (int(x) for x in rng.integers(0, SYN_N, 2))
        if i != j and (i, j) not in have and (i, j) not in new:
            new.append((i, j))
    coll = [(int(rows[k]), int(cols[k])) for k in rng.choice(np.flatnonzero(net.counts > 0), 2, replace=False)]
    pairs, cls = new + coll, ["a"] * 15 + ["c"] * 27
    b_base = [int(x) for x in rng.integers(1, 4, len(pairs))]
    b_base[3] = -b_base[3]
    edges = sorted([p[0], p[1], b, c] for p, b, c in zip(pairs, b_base, cls))
    edge_sha = hashlib.sha256(json.dumps(edges, separators=(",", ":")).encode()).hexdigest()

    def build(meas_net, g_a, g_c):
        cnt = [b * (g_a if c == "a" else g_c) for b, c in zip(b_base, cls)]
        bn = S.Network.from_pairs(np.concatenate([meas_net.rows(), [p[0] for p in pairs]]),
                                  np.concatenate([meas_net.indices.astype(np.int64), [p[1] for p in pairs]]),
                                  np.concatenate([meas_net.counts.astype(np.int64), cnt]), SYN_N, ids=meas_net.ids,
                                  meta={"bridge_version": SYN_BRIDGE})
        prov = PV.EdgeProvenance(PV.BRIDGE_CLASSES, [p[0] for p in pairs], [p[1] for p in pairs], cnt, cls)
        tb = BT.encode(bn, prov, bridge_version=SYN_BRIDGE, edge_list_sha256=edge_sha, gains={"g_a": g_a, "g_c": g_c},
                       topology="synthetic")
        return bn, prov, tb

    bnet, bprov, TB = build(net, 2, 3)
    bnet5, _, TB5 = build(net, 2, 5)                     # a different gain on the same pairs
    bnet_prev, _, TB_stale = build(net2, 2, 3)           # the build before the measured wiring changed by one synapse
    bcase, bcase5 = _case(bnet, params, base), _case(bnet5, params, base)
    RB, _, rasB, _, _ = _window(bcase, BH[1], T, W, kh, bridge=TB)
    RB5, _, _, _, _ = _window(bcase5, BH[1], T, W, kh, bridge=TB5)
    tB = BT.decode(TB)
    cq = tB["classes"].index("c")

    rel = copy.deepcopy(tB)                              # half the class-c links relabelled measured
    half = np.flatnonzero(rel["bridged"][:, cq] != 0)[::2]
    rel["measured"][half] += rel["bridged"][half, cq]
    rel["bridged"][half, cq] = 0
    only = np.flatnonzero(np.any(tB["bridged"] != 0, axis=1) & (tB["measured"] == 0))
    drop = copy.deepcopy(tB)                             # one bridge link zeroed, its count left unattributed
    drop["bridged"][only[0]] = 0
    row = {k: v for k, v in tB.items()}                  # one bridge link's row deleted
    keep = np.arange(tB["pairs"]) != only[0]
    for k in ("pre", "post", "measured", "bridged"):
        row[k] = tB[k][keep]
    row["pairs"] = int(keep.sum())
    mis = None                                           # one bridge link named on a pair the network does not have
    for k in only.tolist():
        i, j = int(tB["pre"][k]), int(tB["post"][k])
        lo = int(tB["post"][k - 1]) if k and tB["pre"][k - 1] == i else -1
        hi = int(tB["post"][k + 1]) if k + 1 < tB["pairs"] and tB["pre"][k + 1] == i else SYN_N
        cand = [x for x in (j - 1, j + 1) if lo < x < hi]
        if cand:
            mis = copy.deepcopy(tB)
            mis["post"][k] = cand[0]
            break
    assert mis is not None
    TB_rel, TB_drop, TB_row, TB_mis = table_bytes(rel), table_bytes(drop), table_bytes(row), table_bytes(mis)
    TB_zero = BT.encode(net, PV.EdgeProvenance.measured_only(), bridge_version=SYN_BRIDGE, edge_list_sha256=edge_sha,
                        gains={"g_a": 2, "g_c": 3}, topology="synthetic")
    tables = {"honest": TB, "relabelled": TB_rel, "dropped": TB_drop, "dropped-row": TB_row, "other-gain": TB5,
              "stale": TB_stale, "misplaced": TB_mis, "zero": TB_zero}
    info["bridged"] = {
        "spikes_differ": RB["brain"]["spikes"] != R["brain"]["spikes"], "n_spikes": int(len(rasB.neurons)),
        "prov_digest": bprov.digest(), "pv_share": PV.synapse_count_share(bnet, bprov),
        "bridged_digest": BT.bridged_digest(tB, bnet.digest()), "network": bnet.digest(), "sha": BT.sha256(TB),
        "rows_with_part": int(np.any(tB["bridged"] != 0, axis=1).sum()), "bridged_abs": int(np.abs(tB["bridged"]).sum()),
        # what each forged table says about the honest network and about the network it came from
        "attribution": {k: BT.attribution_error(BT.decode(v), bnet) for k, v in tables.items()},
        "own": {"other-gain": BT.attribution_error(BT.decode(TB5), bnet5), "stale": BT.attribution_error(BT.decode(TB_stale), bnet_prev)},
        "digests_differ": {"other-gain": bnet5.digest() != bnet.digest(), "stale": bnet_prev.digest() != bnet.digest()},
        "relabel": {"bridged_count": BT.bridged_count(rel), "measured_share": BT.summary(rel)["measured_share"],
                    "honest_measured_share": BT.summary(tB)["measured_share"]},
        "misplaced_pair_in_net": (int(mis["pre"][k]), int(mis["post"][k])) in set(zip(bnet.rows().tolist(), bnet.indices.tolist())),
    }

    def resealed(r, data):
        r = copy.deepcopy(r)
        r["brain"]["bridge"] = None if data is None else BT.sha256(data)
        return r

    fb = {"kernel": str(kp["honest"]), "case": _write(d, "case-bridged.json", bcase)}
    fb5 = {"kernel": str(kp["honest"]), "case": _write(d, "case-bridged-g5.json", bcase5)}
    paths = {k: _write(d, f"bridge-{k}.fbbt", v) for k, v in tables.items()}
    served = lambda k, f=fb: {**f, "bridge": paths[k]}                       # noqa: E731
    TRB = [{"network": RB["brain"]["network"], "bridge": RB["brain"]["bridge"]}]
    missing = copy.deepcopy(R)
    del missing["brain"]["bridge"]
    H = [kh["sha256"]]
    files = {"kernel": str(kp["honest"]), "case": _write(d, "case-clean-b.json", _case(net, params, base))}
    sc = [
        {"id": "syn-bridged-honest-0", "receipt": RB, "files": served("honest"), "trusted": H},
        {"id": "syn-bridged-honest-trusted", "receipt": RB, "files": served("honest"), "trusted": H, "trusted_brains": TRB},
        {"id": "syn-bridged-other-gain-own", "receipt": RB5, "files": served("other-gain", fb5), "trusted": H,
         "trusted_brains": [{"network": RB5["brain"]["network"], "bridge": RB5["brain"]["bridge"]}]},
        {"id": "syn-bridged-no-trust-list", "receipt": RB, "files": served("honest"), "trusted": H, "no_trusted_brains": True},
        {"id": "syn-honest-0-trusted-unbridged", "receipt": R, "files": files, "trusted": H,
         "trusted_brains": [{"network": R["brain"]["network"], "bridge": None}]},
        {"id": "syn-honest-0-trust-lists-only-the-bridged", "receipt": R, "files": files, "trusted": H, "trusted_brains": TRB},
        # the honest receipt, a different table served: the hash stops each
        {"id": "syn-bridge-relabelled", "receipt": RB, "files": served("relabelled"), "trusted": H},
        {"id": "syn-bridge-dropped", "receipt": RB, "files": served("dropped"), "trusted": H},
        {"id": "syn-bridge-other-gain", "receipt": RB, "files": served("other-gain"), "trusted": H},
        {"id": "syn-bridge-stale", "receipt": RB, "files": served("stale"), "trusted": H},
        # the receipt's hash recomputed for the forged table: the attribution stops each
        {"id": "syn-bridge-dropped-resealed", "receipt": resealed(RB, TB_drop), "files": served("dropped"), "trusted": H},
        {"id": "syn-bridge-dropped-row-resealed", "receipt": resealed(RB, TB_row), "files": served("dropped-row"), "trusted": H},
        {"id": "syn-bridge-other-gain-resealed", "receipt": resealed(RB, TB5), "files": served("other-gain"), "trusted": H},
        {"id": "syn-bridge-stale-resealed", "receipt": resealed(RB, TB_stale), "files": served("stale"), "trusted": H},
        {"id": "syn-bridge-misplaced-resealed", "receipt": resealed(RB, TB_mis), "files": served("misplaced"), "trusted": H},
        # limits: a resealed relabel and a null claim account for every count; only trustedBrains stops them
        {"id": "syn-bridge-relabelled-resealed", "receipt": resealed(RB, TB_rel), "files": served("relabelled"), "trusted": H},
        {"id": "syn-bridge-relabelled-resealed-trusted", "receipt": resealed(RB, TB_rel), "files": served("relabelled"), "trusted": H,
         "trusted_brains": TRB},
        {"id": "syn-bridge-null-declared", "receipt": resealed(RB, None), "files": fb, "trusted": H},
        {"id": "syn-bridge-null-declared-trusted", "receipt": resealed(RB, None), "files": fb, "trusted": H, "trusted_brains": TRB},
        # an unbridged network may not publish an all-measured table; the field is required; the table must be served
        {"id": "syn-bridge-empty-table", "receipt": resealed(R, TB_zero), "files": {**files, "bridge": paths["zero"]}, "trusted": H},
        {"id": "syn-bridge-missing-field", "receipt": missing, "files": files, "trusted": H},
        {"id": "syn-bridge-not-served", "receipt": RB, "files": fb, "trusted": H},
        {"id": "syn-bridge-trust-list-bare-digest", "receipt": RB, "files": served("honest"), "trusted": H,
         "trusted_brains": [RB["brain"]["network"]]},
        # isolated structural guards: the trust list names the forged pair, so only the structural check can refuse it
        {"id": "syn-bridge-dropped-resealed-iso", "receipt": resealed(RB, TB_drop), "files": served("dropped"), "trusted": H,
         "trusted_brains": [{"network": RB["brain"]["network"], "bridge": BT.sha256(TB_drop)}]},
        {"id": "syn-bridge-misplaced-resealed-iso", "receipt": resealed(RB, TB_mis), "files": served("misplaced"), "trusted": H,
         "trusted_brains": [{"network": RB["brain"]["network"], "bridge": BT.sha256(TB_mis)}]},
        {"id": "syn-bridge-empty-table-iso", "receipt": resealed(R, TB_zero), "files": {**files, "bridge": paths["zero"]}, "trusted": H,
         "trusted_brains": [{"network": R["brain"]["network"], "bridge": BT.sha256(TB_zero)}]},
    ]
    return {"scenarios": sc, "tables": tables, "RB": RB, "bnet": bnet, "bprov": bprov, "bcase": bcase, "edge_sha": edge_sha}


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    d = tmp_path_factory.mktemp("verify-syn")
    kp, kref, nonce = kernel_variants(d)
    net = S.synthetic_graph(SYN_N, 14000, seed="verify-syn", frac_type1=0.8)
    params = {"w_syn_mV": 2.0}
    F, K = 8, 80
    rates = (20.0 + (np.arange(F * K) * 37 % 131)).reshape(F, K).astype(np.float64)
    base = [X.source("poisson", np.arange(SYN_N), 25.0, name="bg"),
            X.source("poisson", np.arange(40, 120), rates, frame_ms=25.0, weight_mV=15.0, name="retina"),
            X.source("current", np.arange(300, 330), 5.0, start_ms=30.0, stop_ms=150.0, name="drive")]
    clean = _case(net, params, base)
    kh = kref["honest"]
    T, W = 0, 600               # window from step 0
    CS, CW = 1100, 500          # window from a checkpoint
    R, _, ras, genesis, _ = _window(clean, BH[1], T, W, kh)
    RC, ckC, rasC, _, _ = _window(clean, BH[1], CS, CW, kh)
    case_clean = _write(d, "case-clean.json", clean)
    ck_path = _write(d, "ck-1100.fbck", ckC)
    info = {"R": R, "RC": RC, "n_spikes": len(ras.neurons), "n_spikes_ck": len(rasC.neurons)}

    # a hidden current source: the run the forger really made
    hidden = _case(net, params, base + [X.source("current", [5, 6, 7], 30.0, start_ms=5.0, stop_ms=50.0, name="gain")])
    RH, _, rasH, _, _ = _window(hidden, BH[1], T, W, kh)
    info["hidden"] = {"spikes_differ": RH["brain"]["spikes"] != R["brain"]["spikes"],
                      "frames_same": RH["inputs"]["frames"] == R["inputs"]["frames"],
                      "spec_differs": RH["inputs"]["spec"] != R["inputs"]["spec"]}
    forged_hidden = copy.deepcopy(RH)
    forged_hidden["inputs"] = copy.deepcopy(R["inputs"])            # claims the clean inputs

    # one rate byte changed in the retina frames
    rate_src = copy.deepcopy(base)
    raw = bytearray(base64.b64decode(rate_src[1]["rate_hz"]["f64_b64"]))
    raw[8 * 5 + 6] ^= 0xFF                                           # element 5 (frame 0), a high mantissa byte
    rate_src[1]["rate_hz"]["f64_b64"] = b64(bytes(raw))
    ratecase = _case(net, params, rate_src)
    RR, _, _, _, _ = _window(ratecase, BH[1], T, W, kh)
    info["rate"] = {"spikes_differ": RR["brain"]["spikes"] != R["brain"]["spikes"]}
    forged_rate = copy.deepcopy(RR)
    forged_rate["inputs"] = copy.deepcopy(R["inputs"])

    # one synapse changed
    net2 = S.Network(net.ids, net.indptr, net.indices, net.counts.copy(), dead=net.dead)
    net2.counts[int(np.argmax(net2.counts > 0))] += 1
    netcase = _case(net2, params, base)
    RN, _, _, _, _ = _window(netcase, BH[1], T, W, kh)
    forged_net = copy.deepcopy(RN)
    forged_net["brain"]["network"] = R["brain"]["network"]

    # different params
    parcase = _case(net, {"w_syn_mV": 2.05}, base)
    RP, _, _, _, _ = _window(parcase, BH[1], T, W, kh)
    info["params_spikes_differ"] = RP["brain"]["spikes"] != R["brain"]["spikes"]
    forged_par = copy.deepcopy(RP)
    forged_par["brain"]["params"] = R["brain"]["params"]

    # a decision the brain did not make, for the kernel attacks: the hidden run's spikes and state
    forged_claim = copy.deepcopy(R)
    forged_claim["brain"]["spikes"], forged_claim["brain"]["state"] = RH["brain"]["spikes"], RH["brain"]["state"]
    evil = {"startStep": T, "endStep": T + W, "n": SYN_N, "spikes": RH["brain"]["spikes"], "state": RH["brain"]["state"]}

    def rc_with(r, **kw):
        r = copy.deepcopy(r)
        for k, v in kw.items():
            r[k] = v
        return r

    seed_forged = copy.deepcopy(R)
    seed_forged["seed"]["value"] = BH[2]
    seed_forged_ck = copy.deepcopy(RC)
    seed_forged_ck["seed"]["value"] = BH[2]
    shifted = copy.deepcopy(R)
    shifted["window"]["n_steps"] = W + 1

    # a random stream moved ahead, every hash recomputed to match (snapshot, spikes, state): the
    # retina by one step's draws (the window's spikes change), and the scalar "bg" stream by one
    # 4096-draw batch (its pending events cover the window, so only the end state changes)
    ck0 = SN.parse(ckC)
    lif = S.LIFParams(**{k: float(v) for k, v in params.items()})

    def move(name, draws):
        ck = copy.deepcopy(ck0)
        src = next(s for s in ck.sources if s.name == name)
        g = np.random.PCG64()
        g.state = {"bit_generator": "PCG64", "state": {"state": src.state, "inc": src.inc}, "has_uint32": 0, "uinteger": 0}
        g.advance(draws)
        src.state = int(g.state["state"]["state"])
        data = SN.encode(ck, recompute_digest=True)
        msim = SN.deserialize(data, net, lif)
        mras = msim.run(n_steps=CW)
        r = copy.deepcopy(RC)
        r["window"]["snapshot"] = SN.parse(data).state_digest
        r["brain"]["spikes"], r["brain"]["state"] = S.spike_hash(mras), msim.state_digest()
        return data, r

    moved_bytes, moved = move("retina", K)
    moved_bg_bytes, moved_bg = move("bg", 4096)
    info["moved_spikes_differ"] = moved["brain"]["spikes"] != RC["brain"]["spikes"]
    info["moved_bg"] = {"spikes_same": moved_bg["brain"]["spikes"] == RC["brain"]["spikes"],
                        "state_differs": moved_bg["brain"]["state"] != RC["brain"]["state"]}
    # one voltage edited in the checkpoint, every hash resealed
    ck = copy.deepcopy(ck0)
    i = int(np.argmax(ck.u))
    ck.u[i] = np.float32(ck.u[i] + np.float32(3.0))
    resealed = SN.encode(ck, recompute_digest=True)

    files = {"kernel": str(kp["honest"]), "case": case_clean}
    H = [kh["sha256"]]

    # ---- B: anchoring a checkpoint (chain links, trusted checkpoints) and the fabricated state ----
    R6, ck6, _, _, _ = _window(clean, BH[1], 600, CS - 600, kh)          # the honest state at step 600
    assert R6["brain"]["state"] == RC["window"]["snapshot"]              # 600 -> 1100 reaches the 1100 checkpoint
    g0, s6, s11 = pt(0, genesis), pt(600, R6["window"]["snapshot"]), pt(CS, RC["window"]["snapshot"])
    ck6_key = f"checkpoint:{s6['snapshot']}"
    ck6_path = _write(d, "ck-600.fbck", ck6)
    chain_g11 = [link_of(g0, s11)]
    chain_2 = [link_of(g0, s6), link_of(s6, s11)]
    fab_bytes, fab, fab_ras = _fabricate(ckC, net, lif, CW, RC, neuron=350)
    s11f = pt(CS, fab["window"]["snapshot"])
    fab_files = {**files, "checkpoint": _write(d, "ck-fabricated.fbck", fab_bytes)}
    info["fabricated"] = {"spikes_differ": fab["brain"]["spikes"] != RC["brain"]["spikes"],
                          "fires_350": int((fab_ras.neurons == 350).sum()), "honest_fires_350": int((rasC.neurons == 350).sum()),
                          "snapshot_differs": fab["window"]["snapshot"] != RC["window"]["snapshot"]}
    other_inputs = {**anchor_of(RC), "inputs": RH["inputs"]["spec"]}
    link_1600 = {"kernel": kh, "seed": RC["seed"], "brain": {"network": RC["brain"]["network"], "params": RC["brain"]["params"]},
                 "inputs": RC["inputs"], "from": s11, "to": pt(CS + CW, RC["brain"]["state"])}
    link_fab = {**link_1600, "from": s11f, "to": pt(CS + CW, fab["brain"]["state"])}

    # ---- A: a source that began before the run (the adversary's neg-* scenarios) ----
    neg_base = [base[0], base[1], X.source("poisson", np.arange(200, 280), np.full(K, 40.0), start_ms=-50.0, weight_mV=15.0, name="tonic"), base[2]]
    zero_base = [base[0], base[1], X.source("poisson", np.arange(200, 280), np.full(K, 40.0), weight_mV=15.0, name="tonic"), base[2]]
    negcase, zerocase = _case(net, params, neg_base), _case(net, params, zero_base)
    RNg, ckN, rasN, genN, _ = _window(negcase, BH[1], CS, CW, kh)
    RZ, ckZ, _, _, _ = _window(zerocase, BH[1], CS, CW, kh)
    ckM = SN.parse(ckN)
    tonic = next(s for s in ckM.sources if s.name == "tonic")
    g = np.random.PCG64()
    g.state = {"bit_generator": "PCG64", "state": {"state": tonic.state, "inc": tonic.inc}, "has_uint32": 0, "uinteger": 0}
    g.advance(500 * K)                                     # where v2's fast-forward (counting from start -500) put it
    tonic.state = int(g.state["state"]["state"])
    moved_neg_bytes = SN.encode(ckM, recompute_digest=True)
    msim = SN.deserialize(moved_neg_bytes, net, lif)
    mras = msim.run(n_steps=CW)
    RMn = copy.deepcopy(RNg)
    RMn["window"]["snapshot"] = SN.parse(moved_neg_bytes).state_digest
    RMn["brain"]["spikes"], RMn["brain"]["state"] = S.spike_hash(mras), msim.state_digest()
    info["neg"] = {"moved_spikes_differ": RMn["brain"]["spikes"] != RNg["brain"]["spikes"], "n_spikes": int(len(rasN.neurons))}
    neg_files = {**files, "case": _write(d, "case-neg.json", negcase)}
    neg_link = {"kernel": kh, "seed": RNg["seed"], "brain": {"network": RNg["brain"]["network"], "params": RNg["brain"]["params"]},
                "inputs": RNg["inputs"], "from": pt(0, genN)}

    # ---- C: a degenerate seed, and the seed anchor against a mock Solana ----
    zero_seed = _raw_receipt(clean, "1" * 32, W, kh)
    commit = "5RunCommitSig1111111111111111111111111111111111111111111111111111111111111111111111"
    memo = SN.run_commit_memo(R["brain"]["network"], R["brain"]["params"], kh["sha256"], 1001)
    Rs = copy.deepcopy(R)
    Rs["seed"]["commit"] = commit
    chain_ok = {"tx": {commit: {"slot": 1000, "memo": memo}}, "blocks": {"1001": BH[1]}}

    def lookup(tx_slot=1000, memo_text=memo, block=BH[1], **kw):
        return {"tx": {commit: {"slot": tx_slot, "memo": memo_text}}, "blocks": {"1001": block}, **kw}

    # ---- the adversary's secondary cases (limits, not holes): honest receipts for other inputs ----
    weight_src = copy.deepcopy(base)
    weight_src[1]["weight_mV"] = 60.0                                   # the retina's weight x4, same rates
    weightcase = _case(net, params, weight_src)
    RW, _, rasW, _, _ = _window(weightcase, BH[1], T, W, kh)
    retarget_src = copy.deepcopy(base)
    retarget_src[1] = X.source("poisson", np.arange(320, 400), rates, frame_ms=25.0, weight_mV=15.0, name="retina")
    retargetcase = _case(net, params, retarget_src)
    RT, _, _, _, _ = _window(retargetcase, BH[1], T, W, kh)
    info["limits"] = {"weight": {"frames_same": RW["inputs"]["frames"] == R["inputs"]["frames"], "spec_differs": RW["inputs"]["spec"] != R["inputs"]["spec"],
                                 "spikes_differ": RW["brain"]["spikes"] != R["brain"]["spikes"]},
                      "retarget": {"frames_same": RT["inputs"]["frames"] == R["inputs"]["frames"], "spec_differs": RT["inputs"]["spec"] != R["inputs"]["spec"],
                                   "spikes_differ": RT["brain"]["spikes"] != R["brain"]["spikes"]}}
    unbound = copy.deepcopy(R)
    unbound["learned"] = {"run": "evo-0003", "gen": 12, "params": hashlib.sha256(b"never checked").hexdigest()}
    unbound["chain"], unbound["trade"], unbound["id"] = {"tx": "not-a-signature", "memo": "0" * 64}, "x-44", "f" * 64
    flipped = copy.deepcopy(R)
    flipped["brain"]["spikes"] = ("0" if flipped["brain"]["spikes"][0] != "0" else "1") + flipped["brain"]["spikes"][1:]

    # ---- D: bridge provenance (receipt.brain.bridge; check "bridge") ----
    bridge = _bridged_scenarios(d, net, net2, params, base, R, kp, kh, T, W, info)

    ckf = {**files, "checkpoint": ck_path}
    TR = [anchor_of(RC)]
    sc = [
        {"id": "syn-honest-0", "receipt": R, "files": files, "trusted": H},
        {"id": "syn-honest-0-full", "receipt": R, "files": files, "trusted": H, "gated": False},
        {"id": "syn-honest-ck", "receipt": RC, "files": ckf, "trusted": H, "trusted_checkpoints": TR},
        # B: every way to anchor the honest checkpoint passes; every missing or wrong anchor fails closed
        {"id": "syn-honest-ck-chain", "receipt": RC, "files": ckf, "trusted": H, "chain": chain_g11},
        {"id": "syn-honest-ck-chain-fetched", "receipt": RC, "files": {**ckf, "chain": _write(d, "chain-g11.json", chain_g11)}, "trusted": H},
        {"id": "syn-honest-ck-chain-2", "receipt": RC, "files": {**ckf, ck6_key: ck6_path}, "trusted": H, "chain": chain_2},
        {"id": "syn-honest-ck-chain-from-trusted", "receipt": RC, "files": {**ckf, ck6_key: ck6_path}, "trusted": H,
         "chain": [link_of(s6, s11)], "trusted_checkpoints": [anchor_of(R6)]},
        {"id": "syn-honest-ck-chain-from-untrusted", "receipt": RC, "files": {**ckf, ck6_key: ck6_path}, "trusted": H, "chain": [link_of(s6, s11)]},
        {"id": "syn-honest-ck-unanchored", "receipt": RC, "files": ckf, "trusted": H},
        {"id": "syn-honest-ck-bare-digest", "receipt": RC, "files": ckf, "trusted": H, "trusted_checkpoints": [RC["window"]["snapshot"]]},
        {"id": "syn-honest-0-bare-digest", "receipt": R, "files": files, "trusted": H, "trusted_checkpoints": [R["window"]["snapshot"]]},
        {"id": "syn-honest-ck-trusted-other-inputs", "receipt": RC, "files": ckf, "trusted": H, "trusted_checkpoints": [other_inputs]},
        {"id": "syn-honest-ck-chain-other-seed", "receipt": RC, "files": ckf, "trusted": H,
         "chain": [link_of(g0, s11, seed={"source": SN.SEED_SOURCE, "value": BH[2]})]},
        {"id": "syn-honest-ck-chain-broken", "receipt": RC, "files": {**ckf, ck6_key: ck6_path}, "trusted": H,
         "chain": [link_of(g0, s6), link_of(pt(600, genesis), s11)]},
        {"id": "syn-fabricated", "receipt": fab, "files": fab_files, "trusted": H},
        {"id": "syn-fabricated-honest-chain", "receipt": fab, "files": fab_files, "trusted": H, "chain": chain_g11},
        {"id": "syn-fabricated-forged-chain", "receipt": fab, "files": fab_files, "trusted": H, "chain": [link_of(g0, s11f)]},
        {"id": "syn-fabricated-trusted-control", "receipt": fab, "files": fab_files, "trusted": H, "trusted_checkpoints": [anchor_of(fab)]},
        {"id": "syn-link-unanchored", "mode": "link", "receipt": link_1600, "files": ckf, "trusted": H},
        {"id": "syn-link-anchored-by-chain", "mode": "link", "receipt": link_1600, "files": ckf, "trusted": H, "chain": chain_g11},
        {"id": "syn-link-from-fabricated", "mode": "link", "receipt": link_fab, "files": fab_files, "trusted": H},
        # A: negative start
        {"id": "neg-honest-ck", "receipt": RNg, "files": {**neg_files, "checkpoint": _write(d, "ck-neg.fbck", ckN)}, "trusted": H,
         "trusted_checkpoints": [anchor_of(RNg)]},
        {"id": "neg-honest-ck-chain", "receipt": RNg, "files": {**neg_files, "checkpoint": _write(d, "ck-neg.fbck", ckN)}, "trusted": H,
         "chain": [link_of(pt(0, genN), pt(CS, RNg["window"]["snapshot"]))]},
        # the forger's own digest is trusted here, so only the stream check can stop it
        {"id": "neg-stream-moved", "receipt": RMn, "files": {**neg_files, "checkpoint": _write(d, "ck-neg-moved.fbck", moved_neg_bytes)},
         "trusted": H, "trusted_checkpoints": [anchor_of(RMn)]},
        {"id": "neg-start0-ck", "receipt": RZ, "files": {**files, "case": _write(d, "case-zero.json", zerocase), "checkpoint": _write(d, "ck-zero.fbck", ckZ)},
         "trusted": H, "trusted_checkpoints": [anchor_of(RZ)]},
        {"id": "neg-link-genesis-to-honest", "mode": "link", "receipt": {**neg_link, "to": pt(CS, RNg["window"]["snapshot"])}, "files": neg_files, "trusted": H},
        {"id": "neg-link-genesis-to-moved", "mode": "link", "receipt": {**neg_link, "to": pt(CS, RMn["window"]["snapshot"])}, "files": neg_files, "trusted": H},
        # C: the seed
        {"id": "syn-seed-degenerate", "receipt": zero_seed, "files": files, "trusted": H},
        {"id": "syn-seed-anchored", "receipt": Rs, "files": files, "trusted": H, "lookup": chain_ok},
        {"id": "syn-seed-anchor-no-lookup", "receipt": Rs, "files": files, "trusted": H},
        {"id": "syn-seed-anchor-no-commit", "receipt": R, "files": files, "trusted": H, "lookup": chain_ok},
        {"id": "syn-seed-anchor-rpc-down", "receipt": Rs, "files": files, "trusted": H, "lookup": lookup(throws=True)},
        {"id": "syn-seed-anchor-wrong-blockhash", "receipt": Rs, "files": files, "trusted": H, "lookup": lookup(block=BH[2])},
        {"id": "syn-seed-anchor-commit-after", "receipt": Rs, "files": files, "trusted": H, "lookup": lookup(tx_slot=1001)},
        {"id": "syn-seed-anchor-skipped-slot", "receipt": Rs, "files": files, "trusted": H, "lookup": lookup(block=None)},
        {"id": "syn-seed-anchor-other-brain", "receipt": Rs, "files": files, "trusted": H,
         "lookup": lookup(memo_text=SN.run_commit_memo(R["brain"]["network"], RP["brain"]["params"], kh["sha256"], 1001))},
        {"id": "syn-seed-anchor-unpinned", "receipt": Rs, "files": files, "trusted": H,
         "lookup": lookup(memo_text=memo.rsplit(" ", 1)[0])},
        {"id": "syn-seed-anchor-not-a-memo", "receipt": Rs, "files": files, "trusted": H, "lookup": lookup(memo_text="gm")},
        # the adversary's secondary cases: limits of what verify can know (see test_limits_...)
        {"id": "syn-retina-weight-x4", "receipt": RW, "files": {**files, "case": _write(d, "case-weight.json", weightcase)}, "trusted": H},
        {"id": "syn-retina-retargeted", "receipt": RT, "files": {**files, "case": _write(d, "case-retarget.json", retargetcase)}, "trusted": H},
        {"id": "syn-unbound-fields", "receipt": unbound, "files": files, "trusted": H},
        {"id": "syn-spikes-flipped", "receipt": flipped, "files": files, "trusted": H},
        {"id": "syn-seed-0", "receipt": seed_forged, "files": files, "trusted": H},
        {"id": "syn-seed-ck", "receipt": seed_forged_ck, "files": {**files, "checkpoint": ck_path}, "trusted": H},
        {"id": "syn-stream-moved", "receipt": moved, "files": {**files, "checkpoint": _write(d, "ck-moved.fbck", moved_bytes)}, "trusted": H},
        {"id": "syn-stream-moved-bg", "receipt": moved_bg, "files": {**files, "checkpoint": _write(d, "ck-moved-bg.fbck", moved_bg_bytes)}, "trusted": H},
        {"id": "syn-ck-resealed", "receipt": RC, "files": {**files, "checkpoint": _write(d, "ck-resealed.fbck", resealed)}, "trusted": H},
        {"id": "syn-hidden-current", "receipt": forged_hidden, "files": {**files, "case": _write(d, "case-hidden.json", hidden)}, "trusted": H},
        {"id": "syn-rate-byte", "receipt": forged_rate, "files": {**files, "case": _write(d, "case-rate.json", ratecase)}, "trusted": H},
        {"id": "syn-network", "receipt": forged_net, "files": {**files, "case": _write(d, "case-net.json", netcase)}, "trusted": H},
        {"id": "syn-params", "receipt": forged_par, "files": {**files, "case": _write(d, "case-par.json", parcase)}, "trusted": H},
        {"id": "syn-window-shift", "receipt": shifted, "files": files, "trusted": H},
        {"id": "syn-kernel-byte", "receipt": R, "files": {**files, "kernel": str(kp["tampered"])}, "trusted": H},
        {"id": "syn-kernel-untrusted", "receipt": rc_with(R, kernel=kref["evil"]), "files": {**files, "kernel": str(kp["evil"])}, "trusted": H},
        {"id": "syn-kernel-no-trust-list", "receipt": R, "files": files, "trusted": None},
        {"id": "syn-kernel-swap", "receipt": forged_claim, "files": files, "trusted": H, "swap_to": str(kp["evil"]), "evil": evil},
        {"id": "syn-evil-kernel-control", "receipt": rc_with(forged_claim, kernel=kref["evil"]),
         "files": {**files, "kernel": str(kp["evil"])}, "trusted": [kref["evil"]["sha256"]], "evil": evil},
        {"id": "syn-kernel-mark", "receipt": rc_with(R, kernel=kref["marked"]), "files": {**files, "kernel": str(kp["marked"])},
         "trusted": [kref["marked"]["sha256"]]},
    ] + bridge["scenarios"]
    info.update(nonce=nonce, genesis=genesis)
    return {"scenarios": sc, "info": info, "dir": d, "kernels": kp, "kref": kref, "bridge": bridge}


@pytest.fixture(scope="module")
def syn_results(syn, tmp_path_factory):
    return run_verify(tmp_path_factory.mktemp("verify-syn-run"), syn["scenarios"])


def test_verify_is_browser_safe_and_never_imports_the_kernel_statically():
    src = VERIFY.read_text()
    code = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", src, flags=re.S))
    assert not re.search(r"^\s*import\b", code, flags=re.M), "verify.mjs must have no static import"
    assert "fishbrain-sim" not in re.sub(r'"fishbrain-sim"', "", code), "verify.mjs must not name a kernel path"
    for pat in (r"node:", r"\brequire\s*\(", r"\bprocess\.", r"\bBuffer\b"):
        assert not re.search(pat, code), pat
    # the kernel is executed only from the verified bytes
    assert "await import(url)" in code and "data:text/javascript;base64," in code


def test_syn_honest_receipts_pass(syn, syn_results):
    info = syn["info"]
    assert info["n_spikes"] > 50 and info["n_spikes_ck"] > 50, info
    for sid in ("syn-honest-0", "syn-honest-0-full", "syn-honest-ck"):
        r = syn_results[sid]
        assert r["ok"], (sid, r)
        assert r["passed"] == ["kernel", "network", "bridge", "params", "inputs", "snapshot", "seed", "anchor", "spikes", "state"], (sid, r["passed"])
        assert r["calls"]["kernel"] == 1
    assert syn_results["syn-honest-0"]["n_spikes"] == info["n_spikes"]
    assert syn_results["syn-honest-ck"]["n_spikes"] == info["n_spikes_ck"]
    assert "checkpoint" not in syn_results["syn-honest-0"]["calls"]
    assert syn_results["syn-honest-ck"]["calls"] == {"kernel": 1, "case": 1, "checkpoint": 1}   # trusted: no chain fetched
    assert syn_results["syn-honest-0"]["anchor"] == {"by": "genesis"}
    assert syn_results["syn-honest-ck"]["anchor"] == {"by": "trusted"}
    # without a chainLookup nothing checked the blockhash against the chain, and the result says so
    for sid in ("syn-honest-0", "syn-honest-ck"):
        assert syn_results[sid]["seed_anchored"] is False and "chainLookup" in syn_results[sid]["seed_anchor"]["reason"], sid


SYN_EXPECT = {                 # scenario -> the check it must fail at (None: it must pass)
    "syn-honest-0": None, "syn-honest-0-full": None, "syn-honest-ck": None,
    "syn-seed-0": "seed", "syn-seed-ck": "seed", "syn-stream-moved": "seed", "syn-stream-moved-bg": "seed",
    "syn-ck-resealed": "snapshot",
    "syn-hidden-current": "inputs", "syn-rate-byte": "inputs", "syn-network": "network", "syn-params": "params",
    "syn-window-shift": "spikes", "syn-kernel-byte": "kernel", "syn-kernel-untrusted": "kernel",
    "syn-kernel-no-trust-list": "kernel", "syn-kernel-swap": "spikes", "syn-evil-kernel-control": None,
    "syn-kernel-mark": None,
    # B: anchoring
    "syn-honest-ck-chain": None, "syn-honest-ck-chain-fetched": None, "syn-honest-ck-chain-2": None,
    "syn-honest-ck-chain-from-trusted": None, "syn-honest-ck-chain-from-untrusted": "anchor",
    "syn-honest-ck-unanchored": "anchor", "syn-honest-ck-bare-digest": "anchor", "syn-honest-ck-trusted-other-inputs": "anchor",
    "syn-honest-0-bare-digest": "anchor",           # a malformed trust list fails even where genesis anchors
    "syn-honest-ck-chain-other-seed": "anchor", "syn-honest-ck-chain-broken": "anchor",
    "syn-fabricated": "anchor", "syn-fabricated-honest-chain": "anchor", "syn-fabricated-forged-chain": "anchor",
    "syn-fabricated-trusted-control": None,
    "syn-link-unanchored": None, "syn-link-anchored-by-chain": None, "syn-link-from-fabricated": None,
    # A: negative start
    "neg-honest-ck": None, "neg-honest-ck-chain": None, "neg-stream-moved": "seed", "neg-start0-ck": None,
    "neg-link-genesis-to-honest": None, "neg-link-genesis-to-moved": "state",
    # C: the seed
    "syn-seed-degenerate": "seed", "syn-seed-anchored": None, "syn-seed-anchor-no-lookup": None,
    "syn-seed-anchor-no-commit": None, "syn-seed-anchor-rpc-down": None, "syn-seed-anchor-wrong-blockhash": "seed",
    "syn-seed-anchor-commit-after": "seed", "syn-seed-anchor-skipped-slot": "seed", "syn-seed-anchor-other-brain": "seed",
    "syn-seed-anchor-unpinned": "seed", "syn-seed-anchor-not-a-memo": "seed",
    # the adversary's secondary cases: honest receipts for other inputs pass (a limit, see test_limits_*)
    "syn-retina-weight-x4": None, "syn-retina-retargeted": None, "syn-unbound-fields": None,
    "syn-spikes-flipped": "spikes",
    # D: bridge provenance
    "syn-bridged-honest-0": None, "syn-bridged-honest-trusted": None, "syn-bridged-other-gain-own": None,
    "syn-honest-0-trusted-unbridged": None, "syn-honest-0-trust-lists-only-the-bridged": "bridge",
    "syn-bridge-relabelled": "bridge", "syn-bridge-dropped": "bridge", "syn-bridge-other-gain": "bridge", "syn-bridge-stale": "bridge",
    "syn-bridge-dropped-resealed": "bridge", "syn-bridge-dropped-row-resealed": "bridge", "syn-bridge-other-gain-resealed": "bridge",
    "syn-bridge-stale-resealed": "bridge", "syn-bridge-misplaced-resealed": "bridge",
    "syn-bridge-relabelled-resealed": "bridge", "syn-bridge-relabelled-resealed-trusted": "bridge",  # closed: trustedBrains required
    "syn-bridge-null-declared": "bridge", "syn-bridge-null-declared-trusted": "bridge",              # closed: trustedBrains required
    "syn-bridged-no-trust-list": "bridge",                                                            # no list at all fails closed
    "syn-bridge-dropped-resealed-iso": "bridge", "syn-bridge-misplaced-resealed-iso": "bridge", "syn-bridge-empty-table-iso": "bridge",
    "syn-bridge-empty-table": "bridge", "syn-bridge-missing-field": "receipt", "syn-bridge-not-served": "bridge",
    "syn-bridge-trust-list-bare-digest": "bridge",
}


def test_every_syn_scenario_has_an_expectation(syn):
    assert sorted(SYN_EXPECT) == sorted(s["id"] for s in syn["scenarios"])


@pytest.mark.parametrize("sid", [k for k, v in SYN_EXPECT.items() if v])
def test_syn_forgery_fails_at_the_right_check(syn, syn_results, sid):
    check = SYN_EXPECT[sid]
    r = syn_results[sid]
    assert r["ok"] is False and r["check"] == check, (sid, r)
    order = ["receipt", "kernel", "network", "bridge", "params", "inputs", "snapshot", "seed", "anchor", "spikes", "state"]
    assert all(order.index(p) < order.index(check) for p in r["passed"]), r["passed"]


@pytest.mark.parametrize("sid", [k for k, v in SYN_EXPECT.items() if v is None])
def test_syn_honest_scenario_passes(syn_results, sid):
    r = syn_results[sid]
    assert r["ok"] is True and r["check"] is None, (sid, r)


def test_syn_forgeries_claim_something_the_brain_did_not_do(syn, syn_results):
    i = syn["info"]
    assert i["hidden"] == {"spikes_differ": True, "frames_same": True, "spec_differs": True}, i["hidden"]
    assert i["rate"]["spikes_differ"] and i["params_spikes_differ"] and i["moved_spikes_differ"], i
    # the moved stream and the resealed voltage got past the snapshot check; only the streams / the
    # receipt's snapshot could stop them
    for sid in ("syn-stream-moved", "syn-stream-moved-bg"):
        assert "snapshot" in syn_results[sid]["passed"] and "random stream" in syn_results[sid]["detail"], sid
    # a move the replay alone cannot see: same spikes in the window, caught only by the stream check
    assert i["moved_bg"] == {"spikes_same": True, "state_differs": True}, i["moved_bg"]


# Controls: each mutant re-opens one hole in a copy of verify.mjs; the scenario that guards that hole
# must then change outcome, while the honest receipts still pass (so the mutant copy really runs).
VERIFY_MUTANTS = {
    # hash and execute the buffer the fetcher returned, not a private copy (the swap lands)
    "no-private-copy": ([('const bytes = await privateBytes(await fetcher("kernel", ctx));',
                          'const fetched = await fetcher("kernel", ctx); const bytes = fetched instanceof Uint8Array ? fetched : await privateBytes(fetched);', 1)],
                        ["syn-kernel-swap"]),
    # execute the kernel file next to the page instead of the verified bytes (the refuted design)
    "static-kernel": ([("try { F = await importFromBytes(bytes); }", f'try {{ F = await import("{KERNEL.as_uri()}"); }}', 1)],
                      ["syn-kernel-mark"]),
    "ignore-trust-list": ([("if (!trustedKernels.includes(kernel.sha256)) return", "if (false) return", 1)],
                          ["syn-kernel-untrusted"]),
    "no-kernel-hash": ([("if (sha !== kernel.sha256) return", "if (false) return", 1)], ["syn-kernel-byte"]),
    "no-step0-seed-check": ([("if (d0 !== snapshot) {", "if (false) {", 1)], ["syn-seed-0"]),
    "no-stream-check": ([("if (JSON.stringify(want[q]) !== JSON.stringify(got[q])) {", "if (false) {", 1)],
                        ["syn-stream-moved", "syn-stream-moved-bg"]),
    "no-snapshot-check": ([("if (ck.stateDigest !== snapshot) return", "if (false) return", 1)], ["syn-ck-resealed"]),
    # the refuted example: only the retina frames are pinned
    "frames-only": ([("if (spec !== receipt.inputs.spec) return", "if (false) return", 1)], ["syn-hidden-current"]),
    "no-network-check": ([("if (netDigest !== receipt.brain.network) return", "if (false) return", 1)], ["syn-network"]),
    "no-params-check": ([("if (parDigest !== receipt.brain.params) return", "if (false) return", 1)], ["syn-params"]),
    # B: trust a self-consistent checkpoint (the adversary's fabricated state then passes)
    "no-anchor-check": ([('if (an.fail) return fail("anchor", an.fail);', "if (false) return;", 1)], ["syn-fabricated"]),
    # B: accept a chain without replaying its links (a chain that claims the forged state passes)
    "chain-link-not-replayed": ([("if (d !== l.to.snapshot) {", "if (false) {", 1)], ["syn-fabricated-forged-chain"]),
    # B: accept a chain that ends somewhere else (the honest chain vouches for the forged state)
    "chain-end-unchecked": ([("if (last.step !== start.step || last.snapshot !== start.snapshot) {", "if (false) {", 1)],
                            ["syn-fabricated-honest-chain"]),
    # B: accept a chain rooted in an unanchored checkpoint
    "chain-root-unchecked": ([("if (!root) return", "if (false) return", 1)], ["syn-honest-ck-chain-from-untrusted"]),
    # B: a trusted entry that does not bind the inputs
    "trusted-ignores-inputs": ([(" && t.inputs === R.inputs.spec", "", 1)], ["syn-honest-ck-trusted-other-inputs"]),
    # C: accept a degenerate blockhash
    "no-degenerate-seed-check": ([("if (distinct < MIN_DISTINCT_BYTES) {", "if (false) {", 1)], ["syn-seed-degenerate"]),
    # C: a run-commit after the seed slot, or a blockhash that is not that slot's
    "seed-anchor-no-slot-order": ([("if (!(seedSlot > tx.slot)) {", "if (false) {", 1)], ["syn-seed-anchor-commit-after"]),
    "seed-anchor-no-blockhash-check": ([("if (block.blockhash !== receipt.seed.value) {", "if (false) {", 1)],
                                       ["syn-seed-anchor-wrong-blockhash"]),
    # D: serve any table, whatever the receipt pinned (a relabelled one then passes against the honest receipt)
    "bridge-hash-unchecked": ([("if (sha !== claim) return", "if (false) return", 1)], ["syn-bridge-relabelled"]),
    # D: accept a table whose parts do not add up to the network's counts (a dropped link, another gain, a stale build)
    "bridge-totals-unchecked": ([("if (sum !== net.counts[k]) return", "if (false) return", 1)],
                                ["syn-bridge-dropped-resealed-iso"]),
    # D: accept a table that names a bridge part on a pair the network does not have (the UI would colour the wrong link)
    "bridge-pairs-unchecked": ([("if (t.pre[k] !== i || t.post[k] !== net.indices[k]) return", "if (false) return", 1)],
                               ["syn-bridge-misplaced-resealed-iso"]),
    # D: a trustedBrains list that is validated but never compared
    "trusted-brains-ignored": ([("if (!tb.list.some((e) => e.network === netDigest && e.bridge === claim)) {", "if (false) {", 1)],
                               ["syn-bridge-relabelled-resealed-trusted", "syn-bridge-null-declared-trusted"]),
    # D: let an unbridged network publish an all-measured table instead of null
    "bridge-null-rule-unchecked": ([("if (!s.pairs_with_bridge_part) {", "if (false) {", 1)], ["syn-bridge-empty-table-iso"]),
}
HONEST = ["syn-honest-0", "syn-honest-ck", "syn-bridged-honest-0", "syn-bridged-honest-trusted"]


def _outcome(r, nonce):
    return (r["ok"], r["check"], nonce in r["marks"]) if r["id"] == "syn-kernel-mark" else (r["ok"], r["check"])


@pytest.mark.parametrize("mutant", list(VERIFY_MUTANTS))
def test_verify_mutant_is_caught(syn, syn_results, tmp_path, mutant):
    src = VERIFY.read_text()
    edits, guards = VERIFY_MUTANTS[mutant]
    mutated = src
    for old, new, n in edits:
        assert mutated.count(old) == n, f"{mutant}: expected {n} copies of {old!r}"
        mutated = mutated.replace(old, new)
    mp = tmp_path / f"verify-{mutant}.mjs"
    mp.write_text(mutated)
    by_id = {s["id"]: s for s in syn["scenarios"]}
    run = [by_id[i] for i in HONEST + guards]
    got = run_verify(tmp_path, run, verify_path=mp)
    nonce = syn["info"]["nonce"]
    for i in HONEST:
        assert got[i]["ok"], (mutant, i, got[i])                                  # the mutant copy works
    for i in guards:
        assert _outcome(got[i], nonce) != _outcome(syn_results[i], nonce), (mutant, i, got[i])


def test_kernel_swap_never_executes_the_swapped_bytes(syn_results):
    swap, control = syn_results["syn-kernel-swap"], syn_results["syn-evil-kernel-control"]
    # control: when the evil kernel is the one named and trusted, it makes the forged receipt pass
    assert control["ok"] is True and control["evil_ran"] is True, control
    # the swap really happened after hashing, verify ran the bytes it hashed, and the forgery failed
    assert swap["swapped"] is True and swap["evil_ran"] is False, swap
    assert swap["calls"]["kernel"] == 1 and swap["ok"] is False and swap["check"] == "spikes", swap


def test_verify_executes_the_fetched_kernel_bytes(syn, syn_results):
    r = syn_results["syn-kernel-mark"]
    assert r["ok"], r
    assert syn["info"]["nonce"] in r["marks"], r["marks"]
    assert syn["info"]["nonce"] not in KERNEL.read_text()


# ---------------------------------------------------------------------------------------------
# B: a checkpoint is anchored or refused

def test_anchor_paths_say_how_the_start_was_anchored(syn_results):
    r = syn_results
    assert r["syn-honest-ck-chain"]["anchor"] == {"by": "chain", "root": "genesis", "links": 1}
    assert r["syn-honest-ck-chain"]["calls"] == {"kernel": 1, "case": 1, "checkpoint": 1}      # link 0 starts at genesis
    assert r["syn-honest-ck-chain-fetched"]["anchor"] == {"by": "chain", "root": "genesis", "links": 1}
    assert r["syn-honest-ck-chain-fetched"]["calls"]["chain"] == 1
    # two links: the second restores the 600 checkpoint, fetched content-addressed by its digest
    assert r["syn-honest-ck-chain-2"]["anchor"] == {"by": "chain", "root": "genesis", "links": 2}
    assert r["syn-honest-ck-chain-2"]["calls"] == {"kernel": 1, "case": 1, "checkpoint": 2}
    assert r["syn-honest-ck-chain-from-trusted"]["anchor"] == {"by": "chain", "root": "trusted", "links": 1}
    # the kernel and the case are fetched once per call, however many links there are
    for sid in ("syn-honest-ck-chain", "syn-honest-ck-chain-2", "syn-honest-ck-chain-from-trusted"):
        assert r[sid]["calls"]["kernel"] == 1 and r[sid]["calls"]["case"] == 1, sid
    # the honest checkpoint with no anchor fails closed, after its own checks held
    un = r["syn-honest-ck-unanchored"]
    assert un["passed"] == ["kernel", "network", "bridge", "params", "inputs", "snapshot", "seed"], un
    assert "not anchored" in un["detail"] and un["calls"]["chain"] == 1, un
    assert "bare digest" in r["syn-honest-ck-bare-digest"]["detail"]
    assert "neither genesis" in r["syn-honest-ck-chain-from-untrusted"]["detail"]
    assert "another run" in r["syn-honest-ck-chain-other-seed"]["detail"]
    assert "not where link 0 ended" in r["syn-honest-ck-chain-broken"]["detail"]


def test_fabricated_state_is_real_and_only_the_anchor_stops_it(syn, syn_results):
    """Hole B: u/h of one neuron edited in the checkpoint, every hash resealed. The checkpoint is
    self-consistent and its streams are honest, so it passes snapshot and seed; the anchor refuses it."""
    i = syn["info"]["fabricated"]
    assert i["spikes_differ"] and i["snapshot_differs"] and i["fires_350"] > i["honest_fires_350"], i
    r = syn_results
    for sid in ("syn-fabricated", "syn-fabricated-honest-chain", "syn-fabricated-forged-chain"):
        assert r[sid]["ok"] is False and r[sid]["check"] == "anchor", (sid, r[sid])
        assert r[sid]["passed"] == ["kernel", "network", "bridge", "params", "inputs", "snapshot", "seed"], (sid, r[sid]["passed"])
    assert "chain ends at step 1100" in r["syn-fabricated-honest-chain"]["detail"]
    assert "chain link 0: from step 0 the brain reaches" in r["syn-fabricated-forged-chain"]["detail"]
    # control: when the page vouches for the forged state, nothing else stops it; the anchor is the check
    ctl = r["syn-fabricated-trusted-control"]
    assert ctl["ok"] is True and ctl["anchor"] == {"by": "trusted"}, ctl


def test_a_link_is_conditional_and_says_whether_it_is_anchored(syn_results):
    r = syn_results
    un, ch, fab = r["syn-link-unanchored"], r["syn-link-anchored-by-chain"], r["syn-link-from-fabricated"]
    assert un["ok"] and un["anchor"] is None and "anchor" not in un["passed"], un
    assert ch["ok"] and ch["anchor"] == {"by": "chain", "root": "genesis", "links": 1} and "anchor" in ch["passed"], ch
    # the link from the fabricated state holds as a link; it anchors nothing, and says so
    assert fab["ok"] and fab["anchor"] is None, fab
    assert "chain" not in un["calls"]                                   # a link never fetches a chain
    assert r["neg-link-genesis-to-honest"]["anchor"] == {"by": "genesis"}


# ---------------------------------------------------------------------------------------------
# A: a source that began before the run

def test_negative_start_forgery_is_real_and_fails_at_the_streams(syn, syn_results):
    i = syn["info"]["neg"]
    assert i["moved_spikes_differ"] and i["n_spikes"] > 50, i
    r = syn_results
    assert r["neg-honest-ck"]["ok"] and r["neg-honest-ck-chain"]["ok"] and r["neg-start0-ck"]["ok"]
    mv = r["neg-stream-moved"]
    assert mv["check"] == "seed" and "snapshot" in mv["passed"] and "tonic" in mv["detail"], mv
    assert r["neg-link-genesis-to-moved"]["check"] == "state"


def test_old_fast_forward_reopens_hole_a(syn, tmp_path):
    """Control: a kernel with v2's count (draws from the negative start) must fail the honest receipt
    at "seed" and pass the stream-moved forgery, exactly what the adversary found."""
    src = KERNEL.read_text()
    old_edits = [("const count = e - BigInt(first);", "const count = e - BigInt(s.start);"),
                 ("if (end - 1 >= first) s.hits(", "if (end - 1 >= s.start) s.hits(")]
    mutated = src
    for a, b in old_edits:
        assert mutated.count(a) == 1, a
        mutated = mutated.replace(a, b)
    kp = tmp_path / "kernel-v2-count.mjs"
    kp.write_text(mutated)
    ref = {"name": "fishbrain-sim", "version": SN.kernel_ref(kp)["version"], "sha256": hashlib.sha256(kp.read_bytes()).hexdigest()}
    by_id = {s["id"]: s for s in syn["scenarios"]}
    run = []
    for sid in ("neg-honest-ck", "neg-stream-moved", "syn-honest-ck"):
        s = copy.deepcopy(by_id[sid])
        s["receipt"]["kernel"] = ref
        s["files"]["kernel"] = str(kp)
        s["trusted"] = [ref["sha256"]]
        run.append(s)
    got = run_verify(tmp_path, run)
    assert got["syn-honest-ck"]["ok"], got["syn-honest-ck"]                        # the copy works where starts are >= 0
    assert got["neg-honest-ck"]["ok"] is False and got["neg-honest-ck"]["check"] == "seed", got["neg-honest-ck"]
    assert got["neg-stream-moved"]["ok"] is True, got["neg-stream-moved"]


# ---------------------------------------------------------------------------------------------
# C: the seed

def test_reference_writer_carries_the_run_commit(cases):
    """window_receipt(commit=...) writes seed.commit; run_commit_memo refuses what verify could not parse."""
    c = cases["s1-small-excit-dt0.1"]
    sim, _, _ = X.python_sim({**c, "seed": SN.seed_string(BH[1])})
    r, _, _ = SN.window_receipt(sim, 20, BH[1], commit="SigOfTheRunCommit")
    assert r["seed"] == {"source": SN.SEED_SOURCE, "value": BH[1], "commit": "SigOfTheRunCommit"}
    sim2, _, _ = X.python_sim({**c, "seed": SN.seed_string(BH[1])})
    assert "commit" not in SN.window_receipt(sim2, 20, BH[1])[0]["seed"]
    h = "ab" * 32
    assert SN.run_commit_memo(h, h, h, 7) == f"fishbrain:run:v1 network={h} params={h} kernel={h} seed_slot=7"
    for bad in ((h[:-1], h, h, 7), (h, h, h.upper(), 7), (h, h, h, -1), (h, h, h, 10 ** 16), (h, h, h, "7")):
        with pytest.raises(ValueError):
            SN.run_commit_memo(*bad)
    # the degenerate rule also binds the reference writer: it will not produce a receipt for "1"*32
    with pytest.raises(ValueError, match="degenerate"):
        SN.window_receipt(sim2, 20, "1" * 32)


def test_degenerate_seed_is_refused(syn, syn_results):
    r = syn_results["syn-seed-degenerate"]
    assert r["check"] == "seed" and "degenerate" in r["detail"] and r["passed"] == ["kernel"], r


def test_seed_anchor_against_a_mock_chain(syn_results):
    r = syn_results
    ok = r["syn-seed-anchored"]
    assert ok["ok"] and ok["seed_anchored"] is True, ok
    assert ok["seed_anchor"]["slot"] == 1001 and ok["seed_anchor"]["commit_slot"] == 1000, ok
    # verify asked the chain the two questions it needs, in order, and nothing else
    assert [q["kind"] for q in ok["lookups"]] == ["transaction", "block"] and ok["lookups"][1]["slot"] == 1001, ok["lookups"]
    # missing pieces leave ok alone and say seed_anchored: false, with the reason
    for sid, why in (("syn-seed-anchor-no-lookup", "chainLookup"), ("syn-seed-anchor-no-commit", "seed.commit"),
                     ("syn-seed-anchor-rpc-down", "RPC unavailable")):
        assert r[sid]["ok"] is True and r[sid]["seed_anchored"] is False and why in r[sid]["seed_anchor"]["reason"], (sid, r[sid])
    # the chain contradicting the receipt is a forgery: it fails at "seed", before anything is replayed
    for sid, why in (("syn-seed-anchor-wrong-blockhash", "blockhash is"), ("syn-seed-anchor-commit-after", "not before"),
                     ("syn-seed-anchor-skipped-slot", "no block"), ("syn-seed-anchor-other-brain", "commits to params"),
                     ("syn-seed-anchor-unpinned", "seed slot"), ("syn-seed-anchor-not-a-memo", "not a fishbrain:run:v1")):
        assert r[sid]["check"] == "seed" and why in r[sid]["detail"] and r[sid]["seed_anchored"] is False, (sid, r[sid])
        assert r[sid]["passed"] == ["kernel"], (sid, r[sid]["passed"])


# ---------------------------------------------------------------------------------------------
# limits the adversary also showed (documented in js/README.md, not holes verify can close)

def test_limits_inputs_spec_is_only_as_good_as_its_commitment(syn, syn_results):
    """The retina's weight x4, or its targets moved, with the spec recomputed: an honest receipt for
    other inputs. The frames hash is unchanged and the decision differs; only the spec differs from
    the honest run's, so a page must pin the committed inputs.spec. Anchoring the start state does
    not anchor the inputs (state_digest does not cover rates, weights or targets)."""
    lim = syn["info"]["limits"]
    # a 15 mV kick already crosses the 7 mV threshold, so x4 fires the same cells; the retarget does not
    assert lim["weight"]["frames_same"] and lim["weight"]["spec_differs"], lim["weight"]
    assert lim["retarget"] == {"frames_same": True, "spec_differs": True, "spikes_differ": True}, lim["retarget"]
    for sid in ("syn-retina-weight-x4", "syn-retina-retargeted", "syn-unbound-fields"):
        assert syn_results[sid]["ok"] is True, sid
    # verify reads none of learned / chain / trade / id: they are the stage's to bind
    assert syn_results["syn-spikes-flipped"]["check"] == "spikes"


# ---------------------------------------------------------------------------------------------
# D: bridge provenance (fishbrain-bridge-table-v1, receipt.brain.bridge, check "bridge")

def _cols(t: dict) -> dict:
    return {"n": int(t["n"]), "pre": [int(x) for x in t["pre"]], "post": [int(x) for x in t["post"]],
            "measured": [int(x) for x in t["measured"]], "bridged": [int(x) for x in np.asarray(t["bridged"]).reshape(-1)],
            "classes": list(t["classes"]), "bridge_version": t["bridge_version"], "edge_list_sha256": t["edge_list_sha256"],
            "gains": {k: int(v) for k, v in t["gains"].items()}, "ablate": list(t["ablate"]), "topology": t["topology"]}


def _extra_tables():
    """Shapes the synthetic network does not have: no pairs, no classes, ablations, gains given unsorted, +-2^48."""
    empty = BT.encode_parts(5, [], [], [], np.zeros((0, 0), np.int64), [], bridge_version="v", edge_list_sha256="0" * 64,
                            gains={}, ablate=[], topology="")
    big = BT.encode_parts(7, [0, 0, 3, 6], [1, 6, 2, 0], [2 ** 48, -3, 0, 5], [[0, 0, 1], [-(2 ** 48), 0, 0], [0, 7, 0], [0, 0, 0]],
                          ["a", "b", "c"], bridge_version="g1c-bridge-v1", edge_list_sha256="ab" * 32,
                          gains={"g_c": 32, "g_a": 1}, ablate=["P2", "P1", "all"], topology="m1_3")
    return {"empty": empty, "big": big}


def test_bridge_table_python_and_js_write_identical_bytes(syn, tmp_path):
    """Both encoders write the same bytes from the same columns; JS parses Python's bytes and re-encodes them
    exactly; JS's recomputed digests and synapse-count share equal Python's (ablations and gains included)."""
    tables = {**syn["bridge"]["tables"], **_extra_tables()}
    dec = {k: BT.decode(v) for k, v in tables.items()}
    for k, t in dec.items():
        assert BT.encode_parts(t["n"], t["pre"], t["post"], t["measured"], t["bridged"], t["classes"],
                               bridge_version=t["bridge_version"], edge_list_sha256=t["edge_list_sha256"], gains=t["gains"],
                               ablate=t["ablate"], topology=t["topology"]) == tables[k], k      # canonical: decode, encode, same bytes
    nd = "c" * 64
    payload = {k: {"cols": _cols(t), "b64": b64(tables[k])} for k, t in dec.items()}
    js = node_json(tmp_path, f"""
const V = await import("{VERIFY.as_uri()}");
const b64 = (u) => Buffer.from(u).toString("base64");
const out = {{}};
for (const [k, x] of Object.entries(IN)) {{
  const enc = V.encodeBridgeTable(x.cols);
  const t = V.parseBridgeTable(new Uint8Array(Buffer.from(x.b64, "base64")));
  const re = V.encodeBridgeTable({{ ...t, pre: Array.from(t.pre), post: Array.from(t.post), measured: Array.from(t.measured), bridged: Array.from(t.bridged) }});
  out[k] = {{ enc: b64(enc), re: b64(re), gains: t.gains, ablate: t.ablate, summary: V.bridgeSummary(t),
             prov: await V.edgeProvenanceDigest(t), bridged: await V.bridgedDigest(t, "{nd}") }};
}}
console.log(JSON.stringify(out));""", payload)
    for k, t in dec.items():
        assert js[k]["enc"] == b64(tables[k]), k
        assert js[k]["re"] == b64(tables[k]), k
        assert js[k]["gains"] == t["gains"] and js[k]["ablate"] == t["ablate"], k
        assert js[k]["summary"] == BT.summary(t), k
        assert js[k]["prov"] == BT.edge_provenance_digest(t) and js[k]["bridged"] == BT.bridged_digest(t, nd), k
    # the recomputed digests are the ones the provenance and bridge code write
    info = syn["info"]["bridged"]
    assert BT.edge_provenance_digest(dec["honest"]) == info["prov_digest"] == syn["bridge"]["bprov"].digest()
    assert dec["big"]["ablate"] == ["P1", "P2", "all"] and list(dec["big"]["gains"]) == ["g_a", "g_c"]


def _malformed(syn) -> dict:
    """Byte-level edits of valid tables, each a way the canonical form can be broken."""
    TB = syn["bridge"]["tables"]["honest"]
    t = BT.decode(TB)
    P, K = t["pairs"], len(t["classes"])
    hdr = len(TB) - 8 * P * (3 + K)

    def put(data, col, row, value):
        b = bytearray(data)
        off = hdr + 8 * (col * P + row)
        b[off:off + 8] = int(value).to_bytes(8, "little", signed=True)
        return bytes(b)

    def sub(data, old, new):
        assert data.count(old) == 1, old
        return data.replace(old, new)
    swapped = put(put(TB, 0, 0, t["pre"][1]), 0, 1, t["pre"][0])
    swapped = put(put(swapped, 1, 0, t["post"][1]), 1, 1, t["post"][0])
    big = _extra_tables()["big"]
    np_off = hdr - 8
    return {
        "trailing byte": TB + b"\x00", "truncated": TB[:-1], "magic": b"X" + TB[1:],
        "rows out of order": swapped, "value over 2^48": put(TB, 2, 0, 2 ** 48 + 1), "int64 min": put(TB, 3, 0, -(2 ** 63)),
        "pre outside n": put(TB, 0, P - 1, t["n"]),
        "non-ASCII class": sub(TB, b"\x01\x00\x00\x00b", b"\x01\x00\x00\x00\xe9"),
        "duplicate class": sub(TB, b"\x01\x00\x00\x00b", b"\x01\x00\x00\x00a"),
        "gain keys out of order": sub(TB, b"g_a", b"g_d"), "gain key not g_": sub(TB, b"g_a", b"x_a"),
        "non-ASCII topology": sub(TB, b"synthetic", b"synthet\x80c"),
        "ablations out of order": sub(sub(big, b"P1", b"Q9"), b"P2", b"P1").replace(b"Q9", b"P2"),
        "one pair too many": TB[:np_off] + (P + 1).to_bytes(8, "little") + TB[np_off + 8:],
    }


def test_bridge_table_refusals_agree(syn, tmp_path):
    """Python decode and JS parseBridgeTable refuse the same malformed bytes, and accept the valid ones; both
    encoders refuse the same invalid columns."""
    bad = _malformed(syn)
    good = {**{k: v for k, v in syn["bridge"]["tables"].items()}, **_extra_tables()}
    for k, v in bad.items():
        with pytest.raises(BT.TableError):
            BT.decode(v)
    js = node_json(tmp_path, f"""
const V = await import("{VERIFY.as_uri()}");
const r = {{}};
for (const [k, b] of Object.entries(IN)) {{ try {{ V.parseBridgeTable(new Uint8Array(Buffer.from(b, "base64"))); r[k] = "accepted"; }} catch (e) {{ r[k] = "refused: " + e.message; }} }}
console.log(JSON.stringify(r));""", {**{f"bad:{k}": b64(v) for k, v in bad.items()}, **{f"good:{k}": b64(v) for k, v in good.items()}})
    for k in bad:
        assert js[f"bad:{k}"].startswith("refused"), (k, js[f"bad:{k}"])
    for k in good:
        assert js[f"good:{k}"] == "accepted", (k, js[f"good:{k}"])
    # invalid columns: both encoders refuse
    t = _cols(BT.decode(syn["bridge"]["tables"]["honest"]))
    cases = {"rows out of order": {"pre": t["pre"][1::-1] + t["pre"][2:], "post": t["post"][1::-1] + t["post"][2:]},
             "value over 2^48": {"measured": [2 ** 48 + 1] + t["measured"][1:]},
             "index outside n": {"n": max(t["pre"])},
             "non-ASCII topology": {"topology": "synth\u00e9tic"}, "duplicate ablation": {"ablate": ["E1", "E1"]},
             "duplicate class": {"classes": ["a", "a", "c"]}, "bad edge sha": {"edge_list_sha256": "AB" * 32},
             "gain key": {"gains": {"gain_a": 2}}}
    for k, over in cases.items():
        c = {**t, **over}
        with pytest.raises(BT.TableError):
            BT.encode_parts(c["n"], c["pre"], c["post"], c["measured"], np.asarray(c["bridged"]).reshape(len(c["pre"]), -1),
                            c["classes"], bridge_version=c["bridge_version"], edge_list_sha256=c["edge_list_sha256"],
                            gains=c["gains"], ablate=c["ablate"], topology=c["topology"])
    js2 = node_json(tmp_path, f"""
const V = await import("{VERIFY.as_uri()}");
const r = {{}};
for (const [k, c] of Object.entries(IN)) {{ try {{ V.encodeBridgeTable(c); r[k] = "accepted"; }} catch (e) {{ r[k] = "refused"; }} }}
console.log(JSON.stringify(r));""", {k: {**t, **over} for k, over in cases.items()}, name="enc")
    assert js2 == {k: "refused" for k in cases}, js2


def test_window_receipt_writes_the_bridge(syn, cases):
    """receipt.brain.bridge is always written: null for an unbridged network, the table's sha256 for a bridged
    one. The writer refuses a table that does not account for its network, an all-measured table, and None
    for a network built with a bridge."""
    B, tables = syn["bridge"], syn["bridge"]["tables"]
    c = cases["s1-small-excit-dt0.1"]
    sim, _, _ = X.python_sim({**c, "seed": SN.seed_string(BH[1])})
    r, _, _ = SN.window_receipt(sim, 20, BH[1])
    assert "bridge" in r["brain"] and r["brain"]["bridge"] is None
    assert B["RB"]["brain"]["bridge"] == BT.sha256(tables["honest"])
    for k, why in (("other-gain", "the network's count is"), ("dropped", "the network's count is"),
                   ("dropped-row", "exactly once"), ("misplaced", "the table names")):
        sim, _, _ = X.python_sim({**B["bcase"], "seed": SN.seed_string(BH[1])})
        with pytest.raises(BT.TableError, match=why):
            SN.window_receipt(sim, 20, BH[1], bridge=tables[k])
    sim0, _, _ = X.python_sim({**c, "seed": SN.seed_string(BH[1])})
    with pytest.raises(BT.TableError, match="neurons"):
        SN.window_receipt(sim0, 20, BH[1], bridge=tables["honest"])                   # another network (n differs)
    with pytest.raises(BT.TableError, match="declares brain.bridge null"):          # an all-measured table
        SN.window_receipt(S.Simulator(_clean_net(syn), seed=SN.seed_string(BH[1])), 20, BH[1], bridge=tables["zero"])
    # a network whose meta says it was built with a bridge cannot declare null; with its table it writes the hash
    net = B["bnet"]
    assert net.meta["bridge_version"] == SYN_BRIDGE
    with pytest.raises(ValueError, match="cannot declare brain.bridge null"):
        SN.window_receipt(S.Simulator(net, seed=SN.seed_string(BH[1])), 20, BH[1])
    r2 = SN.window_receipt(S.Simulator(net, seed=SN.seed_string(BH[1])), 20, BH[1], bridge=tables["honest"])[0]
    assert r2["brain"]["bridge"] == BT.sha256(tables["honest"])


def _clean_net(syn):
    """The unbridged synthetic wiring (the syn fixture's measured network), rebuilt from its case file."""
    case = json.loads(Path(next(s for s in syn["scenarios"] if s["id"] == "syn-honest-0")["files"]["case"]).read_text())
    return X.python_sim({**case, "seed": SN.seed_string(BH[1])})[1]


def test_bridged_receipt_passes_and_returns_the_table(syn, syn_results):
    info = syn["info"]["bridged"]
    assert info["spikes_differ"] and info["n_spikes"] > 50, info          # the bridge changes what the brain does
    r = syn_results["syn-bridged-honest-0"]
    assert r["ok"] and r["passed"] == ["kernel", "network", "bridge", "params", "inputs", "snapshot", "seed", "anchor", "spikes", "state"], r
    assert r["calls"] == {"kernel": 1, "case": 1, "bridge": 1}, r["calls"]
    b = r["bridge"]
    assert b["sha256"] == info["sha"] and b["bridged"] is True and b["classes"] == ["a", "b", "c"], b
    assert b["gains"] == {"g_a": 2, "g_c": 3} and b["ablate"] == [] and b["topology"] == "synthetic" and b["bridge_version"] == SYN_BRIDGE
    assert b["edge_provenance"] == info["prov_digest"] and b["bridged_digest"] == info["bridged_digest"], b
    pv = info["pv_share"]
    for k in ("measured_share", "class_share", "synapses", "pairs", "pairs_with_bridge_part", "pairs_bridge_only"):
        assert b["synapse_count_share"][k] == pv[k], k                     # provenance.synapse_count_share, exactly
    assert pv["pairs_with_bridge_part"] == 42 and pv["pairs_bridge_only"] == 40 and 0 < pv["measured_share"] < 1, pv
    assert b["mask_links"] == info["rows_with_part"] == 42 and b["mask_bits"] == 0b101, b          # classes a and c
    assert b["table_pairs"] == pv["pairs"] and b["table_bridged_abs"] == info["bridged_abs"], b
    assert b["decision_path_share"]["computed"] is False and "server" in b["decision_path_share"]["source"], b
    assert b["trusted"] is True, b                                        # the published build is trusted
    assert syn_results["syn-bridged-honest-trusted"]["bridge"]["trusted"] is True
    assert syn_results["syn-bridged-other-gain-own"]["ok"]                 # the other gain's table is right for ITS network
    # unbridged: null, nothing fetched, every count measured
    u = syn_results["syn-honest-0"]
    assert u["bridge"]["sha256"] is None and u["bridge"]["bridged"] is False and "bridge" not in u["calls"], u
    assert u["bridge"]["synapse_count_share"]["measured_share"] == 1 and u["bridge"]["mask_links"] == 0, u["bridge"]
    assert u["bridge"]["synapse_count_share"]["synapses"]["measured"] == int(np.abs(_clean_net(syn).counts.astype(np.int64)).sum())
    assert syn_results["syn-honest-0-trusted-unbridged"]["bridge"]["trusted"] is True


def test_bridge_forgeries_are_real(syn):
    """Each forged table really differs, and says what the test claims it says: the relabel still sums to the
    network's counts (so only the hash, or trustedBrains, can stop it) and lowers the bridged share; the
    other gain and the stale build are the honest tables of THEIR networks; the misplaced pair is not a pair."""
    info = syn["info"]["bridged"]
    tables = syn["bridge"]["tables"]
    assert len({v for v in tables.values()}) == len(tables)
    att = info["attribution"]
    assert att["honest"] is None and att["relabelled"] is None, att
    for k in ("dropped", "dropped-row", "other-gain", "stale", "misplaced", "zero"):
        assert att[k] is not None, k
    assert info["own"] == {"other-gain": None, "stale": None} and info["digests_differ"] == {"other-gain": True, "stale": True}
    rel = info["relabel"]
    assert rel["bridged_count"] > 0 and rel["measured_share"] > rel["honest_measured_share"], rel
    assert info["misplaced_pair_in_net"] is False


BRIDGE_REASONS = {
    "syn-bridge-relabelled": "hashes to", "syn-bridge-dropped": "hashes to", "syn-bridge-other-gain": "hashes to",
    "syn-bridge-stale": "hashes to",
    "syn-bridge-dropped-resealed": "the network's count is", "syn-bridge-other-gain-resealed": "the network's count is",
    "syn-bridge-stale-resealed": "the network's count is", "syn-bridge-dropped-row-resealed": "attributed exactly once",
    "syn-bridge-misplaced-resealed": "the table names",
    "syn-bridge-relabelled-resealed-trusted": "no opts.trustedBrains entry", "syn-bridge-null-declared-trusted": "no opts.trustedBrains entry",
    "syn-honest-0-trust-lists-only-the-bridged": "no opts.trustedBrains entry",
    "syn-bridge-relabelled-resealed": "no opts.trustedBrains entry", "syn-bridge-null-declared": "no opts.trustedBrains entry",
    "syn-bridged-no-trust-list": "opts.trustedBrains is required",
    "syn-bridge-dropped-resealed-iso": "the network's count is", "syn-bridge-misplaced-resealed-iso": "the table names",
    "syn-bridge-empty-table-iso": "declares brain.bridge null",
    "syn-bridge-empty-table": "declares brain.bridge null", "syn-bridge-not-served": "serves no bridge",
    "syn-bridge-trust-list-bare-digest": "{network, bridge}",
}


@pytest.mark.parametrize("sid", list(BRIDGE_REASONS))
def test_bridge_forgery_fails_at_bridge_for_its_reason(syn_results, sid):
    r = syn_results[sid]
    assert r["ok"] is False and r["check"] == "bridge" and BRIDGE_REASONS[sid] in r["detail"], (sid, r)
    assert r["passed"] == ["kernel", "network"], (sid, r["passed"])


def test_bridge_field_is_required(syn_results):
    """A receipt without brain.bridge fails at "receipt": absent is not null (null is itself a claim)."""
    m = syn_results["syn-bridge-missing-field"]
    assert m["check"] == "receipt" and "brain.bridge" in m["detail"] and "absent is not null" in m["detail"], m
    assert m["passed"] == [], m


def test_limits_a_resealed_relabel_or_a_null_claim_needs_trusted_brains(syn, syn_results):
    """Closed 2026-09-26 (bridge forge review): the table accounts for every count, so a relabel with the receipt's
    hash recomputed, or a bridged network declaring null, used to pass untrusted. A count-swap relabel kept every
    summary number identical, so "untrusted but ok" hid the lie. Verify now fails closed: trustedBrains is required,
    and against the published builds both forgeries fail at "bridge"."""
    for sid in ("syn-bridge-relabelled-resealed", "syn-bridge-null-declared",
                "syn-bridge-relabelled-resealed-trusted", "syn-bridge-null-declared-trusted"):
        r = syn_results[sid]
        assert r["ok"] is False and r["check"] == "bridge" and "no opts.trustedBrains entry" in r["detail"], (sid, r)
    n = syn_results["syn-bridged-no-trust-list"]
    assert n["ok"] is False and n["check"] == "bridge" and "is required" in n["detail"], n


# ---------------------------------------------------------------------------------------------
# the real gate network, a window mid-run from a checkpoint

G_START, G_N = 20000, 3000


@pytest.fixture(scope="module")
def gate(tmp_path_factory):
    if not REAL_GATE.exists():
        pytest.skip(f"no real gate case at {REAL_GATE} (git-ignored; export with js/export_case.py --real)")
    d = tmp_path_factory.mktemp("verify-gate")
    kp, kref, _ = kernel_variants(d)
    kh = kref["honest"]
    raw = json.load(open(REAL_GATE))
    # the served case keeps its own "seed": "G1-seed-0"; verify must ignore it
    clean = {k: raw[k] for k in ("format", "name", "seed", "params", "network", "sources", "runs")}
    assert clean["seed"] == "G1-seed-0"
    case_path = _write(d, "gate-case.json", clean)

    R3, ck3, ras3, genesis, sim3 = _window(clean, BH[3], G_START, G_N, kh, embed=False)
    inputs = SN.encode_inputs(sim3)
    ck_path = _write(d, "gate-ck.fbck", ck3)
    ck0 = SN.parse(ck3)
    net, params = sim3.net, sim3.p

    # the reviewer's forgery: a receipt naming seed 0 over the seed-3 run
    seed_forged = copy.deepcopy(R3)
    seed_forged["seed"]["value"] = BH[0]
    # ... and with the checkpoint's own seed string rewritten to seed 0 (not in the digest, so the
    # snapshot still matches): only the random streams can tell
    ck = copy.deepcopy(ck0)
    ck.seed = SN.seed_string(BH[0])
    ck_seed = _write(d, "gate-ck-seed0.fbck", SN.encode(ck))
    assert SN.parse(Path(ck_seed).read_bytes()).state_digest == ck0.state_digest

    # a moved random stream: retina one step ahead, every hash recomputed so only the seed check can see it
    ck = copy.deepcopy(ck0)
    retina = next(s for s in ck.sources if s.name == "retina")
    g = np.random.PCG64()
    g.state = {"bit_generator": "PCG64", "state": {"state": retina.state, "inc": retina.inc}, "has_uint32": 0, "uinteger": 0}
    g.advance(2048)
    retina.state = int(g.state["state"]["state"])
    ck_moved = SN.encode(ck, recompute_digest=True)
    moved_sim = SN.deserialize(ck_moved, net, params, inputs=inputs)
    moved_ras = moved_sim.run(n_steps=G_N)
    moved = copy.deepcopy(R3)
    moved["window"]["snapshot"] = SN.parse(ck_moved).state_digest
    moved["brain"]["spikes"], moved["brain"]["state"] = S.spike_hash(moved_ras), moved_sim.state_digest()
    ck_moved_path = _write(d, "gate-ck-moved.fbck", ck_moved)

    # a tampered checkpoint: one byte flipped; and one voltage edited with every hash resealed
    flip = bytearray(ck3)
    flip[len(ck3) // 3] ^= 0x20
    ck_flip = _write(d, "gate-ck-flip.fbck", bytes(flip))
    ck = copy.deepcopy(ck0)
    i = int(np.argmax(ck.u))
    ck.u[i] = np.float32(ck.u[i] + np.float32(3.0))
    ck_resealed = SN.encode(ck, recompute_digest=True)
    ck_resealed_path = _write(d, "gate-ck-resealed.fbck", ck_resealed)

    # a hidden 30 mV current into the left Mauthner cell during the window
    from fishbrain import brain as B
    ids = [int(x) for x in clean["network"]["ids"]]
    m_left = ids.index(int(B.identified()["mauthner"]["left"]))
    hidden = copy.deepcopy(clean)
    hidden["sources"] = clean["sources"] + [X.source("current", [m_left], 30.0, start_ms=G_START * 0.1,
                                                     stop_ms=(G_START + G_N) * 0.1, name="gain")]
    RH, ckH, rasH, _, _ = _window(hidden, BH[3], G_START, G_N, kh, embed=False)
    forged_hidden = copy.deepcopy(RH)
    forged_hidden["inputs"] = copy.deepcopy(R3["inputs"])

    info = {
        "n_spikes": int(len(ras3.neurons)),
        "moved_spikes_differ": moved["brain"]["spikes"] != R3["brain"]["spikes"],
        "resealed_digest_differs": SN.parse(ck_resealed).state_digest != ck0.state_digest,
        "mauthner_clean": int((ras3.neurons == m_left).sum()), "mauthner_hidden": int((rasH.neurons == m_left).sum()),
        "hidden_frames_same": RH["inputs"]["frames"] == R3["inputs"]["frames"],
        "frames_is_recorded_rates": R3["inputs"]["frames"] == raw["meta"]["rates_sha256"],
    }
    files = {"kernel": str(kp["honest"]), "case": case_path, "checkpoint": ck_path}
    H = [kh["sha256"]]
    link = {"kernel": kh, "seed": R3["seed"], "brain": {"network": R3["brain"]["network"], "params": R3["brain"]["params"]},
            "inputs": R3["inputs"], "from": {"step": G_START, "snapshot": R3["window"]["snapshot"]},
            "to": {"step": G_START + G_N, "snapshot": R3["brain"]["state"]}}
    genesis_link = {**link, "from": {"step": 0, "snapshot": genesis}, "to": {"step": G_START, "snapshot": R3["window"]["snapshot"]}}
    wrong_link = {**link, "to": {"step": G_START + G_N, "snapshot": SN.parse(ck_resealed).state_digest}}

    # hole B on the real brain (the adversary's gate-FORGED-neural-state): only the left Mauthner
    # cell's u/h nudged so it fires, every hash resealed, the receipt recomputed to match
    m_right = ids.index(int(B.identified()["mauthner"]["right"]))
    fab_bytes, fab, fab_ras = _fabricate(ck3, net, params, G_N, R3, neuron=m_left, inputs=inputs)
    fab_path = _write(d, "gate-ck-fabricated.fbck", fab_bytes)
    info["fabricated"] = {"mauthner_left": int((fab_ras.neurons == m_left).sum()), "mauthner_left_clean": info["mauthner_clean"],
                          "mauthner_right": int((fab_ras.neurons == m_right).sum()), "mauthner_right_clean": int((ras3.neurons == m_right).sum()),
                          "spikes_differ": fab["brain"]["spikes"] != R3["brain"]["spikes"]}
    TR = [anchor_of(R3)]
    s_gen, s_start = {"step": 0, "snapshot": genesis}, {"step": G_START, "snapshot": R3["window"]["snapshot"]}
    fab_files = {**files, "checkpoint": fab_path}
    sc = [
        {"id": "gate-honest", "receipt": R3, "files": files, "trusted": H, "trusted_checkpoints": TR},
        {"id": "gate-honest-full", "receipt": R3, "files": files, "trusted": H, "gated": False, "trusted_checkpoints": TR},
        {"id": "gate-honest-chain", "receipt": R3, "files": files, "trusted": H, "chain": [link_of(s_gen, s_start)]},
        {"id": "gate-fabricated-mauthner", "receipt": fab, "files": fab_files, "trusted": H},
        {"id": "gate-fabricated-mauthner-honest-chain", "receipt": fab, "files": fab_files, "trusted": H, "chain": [link_of(s_gen, s_start)]},
        {"id": "gate-fabricated-mauthner-forged-chain", "receipt": fab, "files": fab_files, "trusted": H,
         "chain": [link_of(s_gen, {"step": G_START, "snapshot": fab["window"]["snapshot"]})]},
        {"id": "gate-fabricated-mauthner-trusted-control", "receipt": fab, "files": fab_files, "trusted": H, "trusted_checkpoints": [anchor_of(fab)]},
        {"id": "gate-link-genesis-to-fabricated", "mode": "link", "files": {k: v for k, v in files.items() if k != "checkpoint"}, "trusted": H,
         "receipt": {**genesis_link, "to": {"step": G_START, "snapshot": fab["window"]["snapshot"]}}},
        {"id": "gate-seed", "receipt": seed_forged, "files": files, "trusted": H},
        {"id": "gate-seed-resealed", "receipt": seed_forged, "files": {**files, "checkpoint": ck_seed}, "trusted": H},
        {"id": "gate-stream-moved", "receipt": moved, "files": {**files, "checkpoint": ck_moved_path}, "trusted": H},
        {"id": "gate-ck-flip", "receipt": R3, "files": {**files, "checkpoint": ck_flip}, "trusted": H},
        {"id": "gate-ck-resealed", "receipt": R3, "files": {**files, "checkpoint": ck_resealed_path}, "trusted": H},
        {"id": "gate-hidden-mauthner", "receipt": forged_hidden,
         "files": {**files, "case": _write(d, "gate-case-hidden.json", hidden), "checkpoint": _write(d, "gate-ck-hidden.fbck", ckH)},
         "trusted": H},
        {"id": "gate-link", "mode": "link", "receipt": link, "files": files, "trusted": H},
        {"id": "gate-link-genesis", "mode": "link", "receipt": genesis_link, "files": {k: v for k, v in files.items() if k != "checkpoint"}, "trusted": H},
        {"id": "gate-link-wrong", "mode": "link", "receipt": wrong_link, "files": files, "trusted": H},
    ]
    return {"scenarios": sc, "info": info, "moved_python_replay": S.spike_hash(moved_ras)}


@pytest.fixture(scope="module")
def gate_results(gate, tmp_path_factory):
    return run_verify(tmp_path_factory.mktemp("verify-gate-run"), gate["scenarios"])


def test_gate_honest_window_from_a_checkpoint_passes(gate, gate_results):
    assert gate["info"]["n_spikes"] > 100, gate["info"]
    assert gate["info"]["frames_is_recorded_rates"], "inputs.frames must equal the G1 gate's recorded rates_sha256"
    for sid in ("gate-honest", "gate-honest-full", "gate-honest-chain"):
        r = gate_results[sid]
        assert r["ok"], (sid, r)
        assert r["passed"] == ["kernel", "network", "bridge", "params", "inputs", "snapshot", "seed", "anchor", "spikes", "state"]
        assert r["n_spikes"] == gate["info"]["n_spikes"]
        assert r["calls"] == {"kernel": 1, "case": 1, "checkpoint": 1}
    assert gate_results["gate-honest"]["anchor"] == {"by": "trusted"}
    # the full 20,000-step replay from genesis, in the same call as the window
    assert gate_results["gate-honest-chain"]["anchor"] == {"by": "chain", "root": "genesis", "links": 1}


def test_gate_fabricated_mauthner_state_fails_at_the_anchor(gate, gate_results):
    """Hole B on the real brain: the fabricated left Mauthner fires in the window (an escape the fish
    did not make), and the checkpoint is self-consistent. No anchor, the honest chain, and a chain that
    claims the forged state all fail at "anchor"; vouching for it (the control) is all it would take."""
    f = gate["info"]["fabricated"]
    assert f["spikes_differ"] and f["mauthner_left"] > f["mauthner_left_clean"], f
    for sid in ("gate-fabricated-mauthner", "gate-fabricated-mauthner-honest-chain", "gate-fabricated-mauthner-forged-chain"):
        r = gate_results[sid]
        assert r["ok"] is False and r["check"] == "anchor", (sid, r)
        assert r["passed"] == ["kernel", "network", "bridge", "params", "inputs", "snapshot", "seed"], (sid, r["passed"])
    ctl = gate_results["gate-fabricated-mauthner-trusted-control"]
    assert ctl["ok"] is True and ctl["anchor"] == {"by": "trusted"}, ctl
    assert gate_results["gate-link-genesis-to-fabricated"]["check"] == "state"


@pytest.mark.parametrize("sid,check", [
    ("gate-seed", "seed"), ("gate-seed-resealed", "seed"), ("gate-stream-moved", "seed"),
    ("gate-ck-flip", "snapshot"), ("gate-ck-resealed", "snapshot"), ("gate-hidden-mauthner", "inputs"),
])
def test_gate_forgery_fails_at_the_right_check(gate_results, sid, check):
    r = gate_results[sid]
    assert r["ok"] is False and r["check"] == check, (sid, r)


def test_gate_forgeries_are_real(gate, gate_results):
    i = gate["info"]
    # the hidden current changed what Mauthner did, while the retina frames (all the old verify
    # checked) stayed identical
    assert i["mauthner_hidden"] > i["mauthner_clean"], i
    assert i["hidden_frames_same"], i
    # the moved stream changed the decision window, and the resealed voltage changed the state
    assert i["moved_spikes_differ"] and i["resealed_digest_differs"], i
    # the seed-rewritten checkpoint passed the snapshot check and failed only on its streams
    assert "snapshot" in gate_results["gate-seed-resealed"]["passed"]
    assert "snapshot" in gate_results["gate-stream-moved"]["passed"]
    assert "random stream" in gate_results["gate-stream-moved"]["detail"]


def test_gate_checkpoint_chain_links(gate_results):
    for sid in ("gate-link", "gate-link-genesis"):
        r = gate_results[sid]
        assert r["ok"], (sid, r)
        assert "spikes" not in r["passed"] and r["passed"][-1] == "state"
    assert "checkpoint" not in gate_results["gate-link-genesis"]["calls"]
    # a link is conditional: 20,000 -> 23,000 holds, and says its start is not anchored by itself
    assert gate_results["gate-link"]["anchor"] is None and gate_results["gate-link-genesis"]["anchor"] == {"by": "genesis"}
    bad = gate_results["gate-link-wrong"]
    assert bad["ok"] is False and bad["check"] == "state", bad
