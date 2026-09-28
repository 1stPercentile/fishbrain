"""The JS kernel (js/fishbrain-sim.mjs) must reproduce sim.Simulator bit for bit.

This is what the Verify button rests on: a viewer's browser re-runs the fish's brain and must get the
identical spike hash and state digest as the Python engine. Each case (js/export_case.py) carries the
wiring, parameters, seed string, inputs and run() calls, plus Python's answers after every call. Node
runs the case through the kernel (js/run_case.mjs), and every hash, digest, stat and derived constant
is compared twice: by run_case.mjs itself (its exit code) and again here, in Python.

Cases
  * synthetic: exported fresh by `export_case.py --out <tmp>` on every run, so they always reflect
    the current sim.py. They always run; a missing Node binary is a failure, never a skip.
  * real: brain/data/js/real-*.json (the 8,654-neuron gate network, 35,000 steps, git-ignored,
    16 MB each). The only skip in this module is for these, when that directory holds none. Their
    stored answers are re-derived here from the current sim.py (so a stale file cannot pass), and
    their whole-run raster hash is checked against the G1 trial recorded in data/net/.

Controls (a test that cannot fail is not a test)
  * Kernel mutants: copies of the kernel with one deliberate change each (integrate order, a
    Math.fround removed, synaptic delay and refractory limit off by one, a PCG64 multiplier limb and
    the 128-bit multiplier, delivery order, the Poisson batch size, and one gating-only change). Each
    must fail parity (exit 1, digest mismatches) on every case that exercises the changed path, in
    both modes. Where a case cannot see a mutant (e.g. zero delay has one slot, so a delay off by one
    names the same slot) the test asserts that case PASSES and says why: that pass is the positive
    control showing --kernel loads a working kernel, so a broken harness cannot fake a catch.
  * An unmodified copy of the kernel loaded through --kernel passes every case.
  * One float32 weight bit flipped in a case, run with --use-exported-weights, must change the state
    digest; the same run without the flip must match. Bit 22 (top mantissa bit) is caught on every
    case. Bit 0 (one ULP) is caught on s5 and s2 but washes out by the run boundaries on s1, s3 and s4
    (the target's h decays or is reset, and two adjacent float32 values round to the same product),
    so the one-ULP check uses s5. Weights in a real Verify are derived from counts and params, which
    are hashed (network_digest, params_digest), so the washout does not open a gap there.
  * One synapse count changed by one must change the network digest, the weights and the dynamics.

Needs Node (FISHBRAIN_NODE, then `node` on PATH).
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
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

BRAIN = Path(__file__).resolve().parents[1]
JS = BRAIN / "js"
sys.path.insert(0, str(JS))

import export_case as X  # noqa: E402
from fishbrain import sim as S  # noqa: E402

KERNEL = JS / "fishbrain-sim.mjs"
RUN_CASE = JS / "run_case.mjs"
REAL_DIR = BRAIN / "data" / "js"


def _find_node():
    env = os.environ.get("FISHBRAIN_NODE")
    for cand in (env, shutil.which("node")):
        if cand and Path(cand).exists():
            return cand
    return None


NODE = _find_node()


def _need_node():
    assert NODE, ("Node not found (set FISHBRAIN_NODE). JS parity is unverified without it, "
                  "so this fails rather than skips.")


# ---------------------------------------------------------------------------------------------
# helpers

def run_node(case_path, *flags, kernel=None, timeout=600):
    """Run one case through run_case.mjs; returns (exit code, its JSON output). Exit 2 = error."""
    _need_node()
    cmd = [NODE, str(RUN_CASE), str(case_path), *flags]
    if kernel is not None:
        cmd += ["--kernel", str(kernel)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.startswith("{")]
    assert lines, f"no output from node (rc {p.returncode}): {p.stderr[-2000:]}"
    out = json.loads(lines[-1])
    assert "error" not in out, out.get("error")
    assert p.returncode in (0, 1), (p.returncode, p.stderr[-2000:])
    return p.returncode, out


def mismatches(case, out, exported_weights=False):
    """Every field where the JS output differs from Python's expected block, computed here."""
    E = case["expected"]
    bad = []
    if len(out["runs"]) != len(E["runs"]):
        bad.append("n_runs")
    for i, (er, jr) in enumerate(zip(E["runs"], out["runs"])):
        for k in ("spike_hash", "state_digest", "n_spikes", "n_steps", "stats"):
            if er[k] != jr[k]:
                bad.append(f"runs[{i}].{k}")
    for k in ("raster_hash", "n_spikes", "params_repr", "params_digest", "network_digest", "derived", "poisson_seeds"):
        if E[k] != out[k]:
            bad.append(k)
    if not exported_weights and E["weights_sha256"] != out["weights_sha256"]:
        bad.append("weights_sha256")
    return bad


def hash_mismatches(bad):
    """The mismatches a Verify would show: a window's spike hash or state digest, or the raster hash."""
    return [b for b in bad if b.endswith("spike_hash") or b.endswith("state_digest") or b == "raster_hash"]


def node_eval(tmp_path, code: str, payload) -> dict:
    _need_node()
    (tmp_path / "in.json").write_text(json.dumps(payload))
    src = tmp_path / "snippet.mjs"
    src.write_text(f'import * as F from "{KERNEL.as_uri()}";\nimport fs from "node:fs";\n'
                   f'const IN = JSON.parse(fs.readFileSync("{(tmp_path / "in.json").as_posix()}", "utf8"));\n{code}\n')
    p = subprocess.run([NODE, str(src)], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------------------------
# synthetic cases: exported fresh, through the script's own command line

SYNTHETIC_NAMES = ["s1-small-excit-dt0.1", "s2-inhib-dt0.2-lesion", "s3-dense-bursty",
                   "s4-equal-tau-no-delay", "s5-4000-anchored-w"]
MODES = [pytest.param(False, id="full"), pytest.param(True, id="gated")]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("js-cases")
    p = subprocess.run([sys.executable, str(JS / "export_case.py"), "--out", str(out)],
                       capture_output=True, text=True, timeout=600, cwd=str(BRAIN))
    assert p.returncode == 0, p.stderr[-3000:]
    paths = sorted(out.glob("*.json"))
    assert [q.stem for q in paths] == sorted(SYNTHETIC_NAMES), [q.stem for q in paths]
    return {q.stem: (q, json.load(open(q))) for q in paths}


def test_node_is_available():
    _need_node()
    v = subprocess.run([NODE, "--version"], capture_output=True, text=True).stdout.strip()
    assert int(v.lstrip("v").split(".")[0]) >= 18, v       # BigInt, atob, TextEncoder as globals


def test_synthetic_cases_are_nontrivial(synthetic):
    for name, (_, c) in synthetic.items():
        E = c["expected"]
        last = E["runs"][-1]["stats"]
        assert E["n_spikes"] > 1000, name
        assert last["syn_events"] > 10_000 and last["poisson_events"] > 1000, name
        assert E["active_simulator_identical"] is True, name
    # the variety the cases claim is really there
    params = {n: c["params"] for n, (_, c) in synthetic.items()}
    assert {p.get("dt_ms", 0.1) for p in params.values()} == {0.1, 0.2}
    assert any(c["network"]["dead_indices"] for _, c in synthetic.values())
    kinds = {(s["type"], "scalar" if not isinstance(s.get("rate_hz", s.get("amp_mV")), dict) else
              len(s.get("rate_hz", s.get("amp_mV"))["shape"])) for _, c in synthetic.values() for s in c["sources"]}
    assert {("poisson", "scalar"), ("poisson", 1), ("poisson", 2), ("current", "scalar"), ("current", 2)} <= kinds
    derived = [c["expected"]["derived"] for _, c in synthetic.values()]
    assert {d["n_delay"] for d in derived} >= {0, 18} and {d["n_ref"] for d in derived} >= {1, 22}


@pytest.mark.parametrize("gated", MODES)
@pytest.mark.parametrize("name", SYNTHETIC_NAMES)
def test_parity_synthetic(synthetic, name, gated):
    path, case = synthetic[name]
    rc, out = run_node(path, *(["--gated"] if gated else []))
    assert out["gated"] is gated                 # gating really on (it needs theta >= 0), or really off
    assert mismatches(case, out) == []
    assert out["mismatches"] == [] and rc == 0
    assert out["weights_match_exported"] is True
    assert out["raster_hash"] == case["expected"]["raster_hash"]


# ---------------------------------------------------------------------------------------------
# real gate network cases (brain/data/js, git-ignored)

REAL_PATHS = sorted(REAL_DIR.glob("real-*.json"))
REAL_SKIP = (f"no real-*.json in {REAL_DIR}: regenerate with "
             "`.venv/bin/python js/export_case.py --out /tmp/js-cases --real` (needs brain/data/net)")
REAL_PARAMS = ([pytest.param(p.stem, id=p.stem) for p in REAL_PATHS] or
               [pytest.param(None, id="absent", marks=pytest.mark.skip(reason=REAL_SKIP))])


@lru_cache(maxsize=None)
def _real(name):
    path = REAL_DIR / f"{name}.json"
    with open(path) as f:
        return path, json.load(f)


def _recorded_trial(case):
    """The G1 trial this case reproduces, read live from the results file (not the case's copy)."""
    m = case["meta"]
    results = BRAIN / m["recorded_in"]
    assert results.exists(), f"{results} is missing; the real case cannot be anchored"
    R = json.load(open(results))["results"]
    trials = [t for t in R[m["run_key"]][m["condition"]]["trials"] if t["seed"] == case["seed"]]
    assert len(trials) == 1, (m["run_key"], m["condition"], case["seed"])
    return trials[0]


@pytest.mark.parametrize("gated", MODES)
@pytest.mark.parametrize("name", REAL_PARAMS)
def test_parity_real_gate_network(name, gated):
    path, case = _real(name)
    assert case["network"]["n"] == 8654 and len(case["runs"]) == 7, name
    assert sum(r["n_steps"] for r in case["expected"]["runs"]) == 35_000
    rc, out = run_node(path, *(["--gated"] if gated else []))
    assert out["gated"] is gated
    assert mismatches(case, out) == [], name
    assert out["mismatches"] == [] and rc == 0, name
    trial = _recorded_trial(case)
    assert out["raster_hash"] == trial["spike_hash"], name
    assert out["n_spikes"] == trial["n_spikes"], name


@pytest.mark.parametrize("name", REAL_PARAMS)
def test_real_case_answers_are_current(name):
    """The file's expected block must equal what the current sim.py computes from the same case."""
    _, case = _real(name)
    E = case["expected"]
    sim, net, params = X.python_sim(case, S.Simulator)
    assert net.digest() == E["network_digest"] and params.digest() == E["params_digest"]
    for i, r in enumerate(case["runs"]):
        ras = sim.run(r.get("duration_ms"), n_steps=r.get("n_steps"))
        er = E["runs"][i]
        assert S.spike_hash(ras) == er["spike_hash"], (name, i)
        assert sim.state_digest() == er["state_digest"], (name, i)
        assert {k: int(v) for k, v in sim.stats.items()} == er["stats"], (name, i)
    assert S.spike_hash(sim.raster()) == E["raster_hash"]
    trial = _recorded_trial(case)
    assert E["raster_hash"] == trial["spike_hash"] and case["meta"]["rates_sha256"] == trial["rates_sha256"]


@pytest.mark.parametrize("name", REAL_PARAMS)
def test_js_hashes_the_retina_frames_like_python(name, tmp_path):
    """What the fish saw: the kernel's sha256 over the decoded rate frames equals the recorded rates_sha256."""
    _, case = _real(name)
    (src,) = [s for s in case["sources"] if s["name"] == "retina"]
    got = node_eval(tmp_path, "console.log(JSON.stringify({h: F.sha256Hex(F.decodeF64(IN.b64))}));",
                    {"b64": src["rate_hz"]["f64_b64"]})
    assert got["h"] == case["meta"]["rates_sha256"] == _recorded_trial(case)["rates_sha256"]


def test_js_hashes_synthetic_frames_like_python(synthetic, tmp_path):
    blobs = [s[k]["f64_b64"] for _, c in synthetic.values() for s in c["sources"]
             for k in ("rate_hz", "amp_mV") if isinstance(s.get(k), dict)]
    assert len(blobs) >= 5
    got = node_eval(tmp_path, "console.log(JSON.stringify(IN.map((b) => F.sha256Hex(F.decodeF64(b)))));", blobs)
    assert got == [hashlib.sha256(base64.b64decode(b)).hexdigest() for b in blobs]


# ---------------------------------------------------------------------------------------------
# seeding, constants, repr

def test_seed_chain_matches_numpy(tmp_path):
    ents = [0, 1, 2 ** 32 - 1, 2 ** 32, 2 ** 64 + 5, (7 << 64) | (3 << 32) | 9, (1 << 280) | 12345,
            S.derive_seed("G1-seed-0", "poisson", "retina"), S.derive_seed("0xabc"), S.derive_seed("x", "a", "b|c")]
    ref = []
    for e in ents:
        ss = np.random.SeedSequence(e)
        bg = np.random.PCG64(e)
        st = bg.state["state"]
        ref.append({"e": str(e), "pool": [int(x) for x in ss.pool],
                    "gen": [str(int(x)) for x in ss.generate_state(4, np.uint64)],
                    "state": str(st["state"]), "inc": str(st["inc"]),
                    "raw": [str(int(x)) for x in bg.random_raw(9)],
                    "json_after": json.dumps(bg.state, sort_keys=True, default=str)})
    seeds = [["G1-seed-0", "poisson", "retina"], ["0xabc"], ["x", "a", "b|c"], ["unicodé ✓", "t"]]
    got = node_eval(tmp_path, """
const out = IN.ents.map((v) => { const e = BigInt(v.e); const g = new F.PCG64(e);
  const st = [g.stateBigInt.toString(), g.incBigInt.toString()];
  const raw = Array.from(g.randomRaw(9), String);
  return {e: v.e, pool: F.seedSequencePool(e), gen: F.seedSequenceState64(e).map(String), state: st[0], inc: st[1], raw, json_after: g.stateJSON()}; });
const seeds = IN.seeds.map((s) => F.deriveSeed(...s).toString());
console.log(JSON.stringify({out, seeds}));""", {"ents": ref, "seeds": seeds})
    assert got["out"] == ref
    assert got["seeds"] == [str(S.derive_seed(*s)) for s in seeds]


def test_fast_paths_match_single_steps(tmp_path):
    """fillRaw (four jumping lanes) and drawSelected (jump table) leave the same outputs and state as next()."""
    e = str(S.derive_seed("G1-seed-0", "poisson", "retina"))
    bg = np.random.PCG64(int(e))
    raw = bg.random_raw(4096 + 13 + 700)
    ref = {"fill": [str(int(x)) for x in raw[:4096]], "sel": [str(int(raw[4096 + 13 + i])) for i in (0, 5, 699)],
           "state": str(bg.state["state"]["state"])}
    got = node_eval(tmp_path, """
const g = new F.PCG64(BigInt(IN.e));
const H = new Uint32Array(4096), L = new Uint32Array(4096);
g.fillRaw(4096, H, L);
const fill = Array.from(H, (h, i) => ((BigInt(h) << 32n) | BigInt(L[i])).toString());
for (let i = 0; i < 13; i++) g.next();
const sel = Int32Array.of(0, 5, 699), SH = new Uint32Array(3), SL = new Uint32Array(3);
g.drawSelected(700, sel, 3, SH, SL);
console.log(JSON.stringify({fill, sel: Array.from(SH, (h, i) => ((BigInt(h) << 32n) | BigInt(SL[i])).toString()), state: g.stateBigInt.toString()}));""",
                    {"e": e})
    assert got == ref


def test_derived_constants_and_params_digest_grid(tmp_path):
    grid = []
    for dt in (0.1, 0.2, 0.05, 0.25, 0.5):
        for tm, ts in ((20.0, 5.0), (10.0, 10.0), (7.3, 2.9), (35.0, 0.6)):
            for w in (0.275, 1.9881, 3.0):
                grid.append({"dt_ms": dt, "tau_m_ms": tm, "tau_syn_ms": ts, "w_syn_mV": w,
                             "refractory_ms": 2.5 if dt in (0.5, 0.25) else 2.2,
                             "delay_ms": 1.5 if dt in (0.5, 0.25) else 1.8})
    ref = []
    for g in grid:
        p = S.LIFParams(**g)
        d = p.derived()
        ref.append({"digest": p.digest(), "c64": X.f64_hex(d["c64"]), "k_i64": X.f64_hex(d["k_i64"]),
                    "a": X.f32_hex(d["a"]), "b": X.f32_hex(d["b"]), "n_ref": d["n_ref"], "n_delay": d["n_delay"]})
    got = node_eval(tmp_path, """
const out = IN.map((g) => { const p = new F.LIFParams(g); const d = p.derived();
  return {digest: p.digest(), c64: F.f64Hex(d.c64), k_i64: F.f64Hex(d.k_i64), a: F.f32Hex(d.a), b: F.f32Hex(d.b), n_ref: d.n_ref, n_delay: d.n_delay}; });
console.log(JSON.stringify(out));""", grid)
    assert len(ref) == 60
    assert got == ref


def test_python_float_repr_and_steps(tmp_path):
    xs = [0.1, 0.2, 0.275, 1.9881, 20.0, -52.0, 68.75, 1e-05, 0.0001, 1e16, 1e17, 123456789012345678.0,
          5e-324, 1.7976931348623157e308, -0.0, 0.0, 2.2, 1.8, 3.14159, 1.5e-7, 9.999999999999999e22]
    got = node_eval(tmp_path, "console.log(JSON.stringify(IN.map(F.pyFloatRepr)));", xs)
    assert got == [repr(x) for x in xs]
    # 0.15 and 0.05 are not whole steps of 0.1 (1.5 would round half-even to 2): both must refuse
    ms = [0.0, 2.2, 1.8, 37.3, 112.7, 158.8, 3500.0, 0.30000000000000004, 1e-12, 0.15, 0.05, 0.25]
    dt = 0.1
    ref = []
    for m in ms:
        try:
            ref.append(S._steps(m, dt, "x"))
        except ValueError:
            ref.append("error")
    got = node_eval(tmp_path, """
console.log(JSON.stringify(IN.ms.map((m) => { try { return F.stepsOf(m, IN.dt, "x"); } catch (e) { return "error"; } })));""",
                    {"ms": ms, "dt": dt})
    assert got == ref
    assert ref[1] == 22 and ref.count("error") == 3, ref


def test_kernel_is_browser_safe():
    """No module imports and no Node-only globals: the file loads as-is in a browser <script type=module>."""
    src = KERNEL.read_text()
    code = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", src, flags=re.S))     # comments out
    for pat, what in ((r"^\s*import\b", "an import statement"), (r"\bimport\s*\(", "a dynamic import"),
                      (r"node:", "a node: specifier"), (r"\brequire\s*\(", "require()"),
                      (r"\bprocess\.", "process"), (r"\bBuffer\b", "Buffer"), (r"__dirname|__filename", "CommonJS paths")):
        hits = [ln for ln in code.splitlines() if re.search(pat, ln)]
        assert not hits, f"kernel uses {what}: {hits[:3]}"
    # and it really is loadable with nothing but its own text (no relative imports to resolve)
    assert "export class Simulator" in src and "export function simulatorFromCase" in src


# ---------------------------------------------------------------------------------------------
# controls: kernel mutants

S2, S4 = "s2-inhib-dt0.2-lesion", "s4-equal-tau-no-delay"
FUSED = ("            const v = Math.fround(Math.fround(u[i] * a) + h[i]);\n"
         "            u[i] = v;\n"
         "            h[i] = h[i] * b;\n")
NO_SCALAR_POISSON = "s2 has no scalar Poisson source with 0 < p < 1 (its array sources jump through the table)"

# name -> (edits as (old, new, occurrences expected in the kernel), {case: why it cannot see it},
#          why full mode cannot see it or None)
KERNEL_MUTANTS = {
    # integrate u*a + h with one rounding: what an FMA or a float64 accumulator would do
    "integrate-fround-removed": (
        [("const v = Math.fround(Math.fround(u[i] * a) + h[i]);", "const v = Math.fround(u[i] * a + h[i]);", 2),
         ("u[i] = Math.fround(u[i] * a) + h[i];", "u[i] = u[i] * a + h[i];", 2)], {}, None),
    # operation order: decay h before adding it to u (sim.py adds, then decays)
    "integrate-h-decays-first": (
        [(FUSED, "            h[i] = h[i] * b;\n"
                 "            const v = Math.fround(Math.fround(u[i] * a) + h[i]);\n"
                 "            u[i] = v;\n", 2)], {}, None),
    # deliver spikes fired n_delay - 1 steps ago
    "delay-off-by-one": (
        [("slots[(((k - nDelay) % nslot) + nslot) % nslot]", "slots[(((k - nDelay + 1) % nslot) + nslot) % nslot]", 1)],
        {S4: "s4 has delay_ms 0: one slot, so both indices name it"}, None),
    # refractory one step shorter
    "refractory-off-by-one": (
        [("const lim = k - nRef;", "const lim = k - nRef + 1;", 1)],
        {S4: "s4 has a one-step refractory period: neither limit ever freezes a neuron"}, None),
    # one 16-bit limb of the PCG64 multiplier in the single-step next()
    "pcg64-step-limb": (
        [("c += s0 * 0x2360 + ", "c += s0 * 0x2361 + ", 1)], {S2: NO_SCALAR_POISSON}, None),
    # the 128-bit PCG64 multiplier used for seeding and every jump table
    "pcg64-multiplier": (
        [("PCG_MULT_128 = 0x2360ED051FC65DA44385DF649FCCF645n", "PCG_MULT_128 = 0x2360ED051FC65DA44385DF649FCCF647n", 1)],
        {}, None),
    # add the arriving rows in reverse order (np.add.at's order matters for float32 sums)
    "reverse-delivery-order": (
        [("for (let q = 0; q < arriving.length; q++) {\n        const src = arriving[q]",
          "for (let q = arriving.length - 1; q >= 0; q--) {\n        const src = arriving[q]", 1)], {}, None),
    # a scalar Poisson batch of 4095 raw draws instead of numpy's 4096
    "poisson-batch-4095": (
        [("bg.fillRaw(4096, RH, RL);", "bg.fillRaw(4095, RH, RL);", 1),
         ("for (let d = 0; d < 4096; d++) {", "for (let d = 0; d < 4095; d++) {", 1)], {S2: NO_SCALAR_POISSON}, None),
    # gated only: a Poisson kick no longer wakes its target
    "gated-poisson-untouched": (
        [("u[t] = u[t] + wt; if (gated && !inSet[t]) this._touch(t);", "u[t] = u[t] + wt;", 1)],
        {}, "the full engine integrates every neuron and keeps no active set"),
}


def _mutant(tmp_path, name):
    src = KERNEL.read_text()
    mutated = src
    for old, new, n in KERNEL_MUTANTS[name][0]:
        assert mutated.count(old) == n, f"{name}: expected {n} copies of {old!r}, found {mutated.count(old)}"
        mutated = mutated.replace(old, new)
    assert mutated != src
    kp = tmp_path / f"fishbrain-sim-{name}.mjs"
    kp.write_text(mutated)
    return kp


def test_unmodified_kernel_copy_passes(synthetic, tmp_path):
    kp = tmp_path / "fishbrain-sim-copy.mjs"
    shutil.copyfile(KERNEL, kp)
    for name in SYNTHETIC_NAMES:
        path, case = synthetic[name]
        rc, out = run_node(path, kernel=kp)
        assert out["kernel"] == kp.name
        assert rc == 0 and out["mismatches"] == [] and mismatches(case, out) == [], name


@pytest.mark.parametrize("gated", MODES)
@pytest.mark.parametrize("mutant", list(KERNEL_MUTANTS))
def test_kernel_mutant_is_caught(synthetic, tmp_path, mutant, gated):
    kp = _mutant(tmp_path, mutant)
    _, blind_cases, blind_full = KERNEL_MUTANTS[mutant]
    report, caught = {}, 0
    for name in SYNTHETIC_NAMES:
        path, case = synthetic[name]
        rc, out = run_node(path, *(["--gated"] if gated else []), kernel=kp)
        assert out["kernel"] == kp.name and out["gated"] is gated
        bad = mismatches(case, out)
        assert set(out["mismatches"]) == set(bad), (name, out["mismatches"], bad)   # two comparators agree
        why_blind = blind_cases.get(name) or (blind_full if not gated else None)
        report[name] = (rc, hash_mismatches(bad)[:2], why_blind)
        if why_blind:
            # positive control: the mutant kernel loads and runs; this case cannot see the change
            assert rc == 0 and bad == [], (mutant, name, why_blind, bad)
        else:
            assert rc == 1 and hash_mismatches(bad), (mutant, name, rc, bad)
            caught += 1
    print(mutant, "gated" if gated else "full", report)
    # every mutant is caught by some case in every mode, except a gating-only change in full mode
    assert caught > 0 or (blind_full and not gated), (mutant, report)


# ---------------------------------------------------------------------------------------------
# controls: case perturbations

def _busiest_synapse(case):
    """The first outgoing synapse of the neuron that fires most in the reference raster."""
    counts = np.asarray(case["expected"]["spike_counts"])
    indptr = np.asarray(case["network"]["indptr"])
    for pre in np.argsort(-counts, kind="stable"):
        if counts[pre] > 0 and indptr[pre + 1] > indptr[pre]:
            return int(pre), int(indptr[pre])
    raise AssertionError("no firing neuron has an outgoing synapse")


@pytest.mark.parametrize("gated", MODES)
@pytest.mark.parametrize("name,bit", [("s1-small-excit-dt0.1", 22), ("s5-4000-anchored-w", 22),
                                      ("s5-4000-anchored-w", 0)], ids=["s1-bit22", "s5-bit22", "s5-bit0-ulp"])
def test_weight_bit_flip_is_detected(synthetic, tmp_path, name, bit, gated):
    path, case = synthetic[name]
    mode = ["--gated"] if gated else []
    E = case["expected"]
    # positive control: the exported weight bits, unmodified, give Python's answers
    rc0, out0 = run_node(path, "--use-exported-weights", *mode)
    assert out0["exported_weights"] is True and out0["weights_match_exported"] is True
    assert out0["weights_sha256"] == E["weights_sha256"]
    assert rc0 == 0 and mismatches(case, out0, exported_weights=True) == []
    # the perturbation: one bit of one weight the run really uses
    pre, j = _busiest_synapse(case)
    mut = copy.deepcopy(case)
    mut["weights_f32_bits"][j] ^= 1 << bit
    w = np.array([case["weights_f32_bits"][j], mut["weights_f32_bits"][j]], np.uint32).view(np.float32)
    assert w[0] != 0 and w[0] != w[1] and np.isfinite(w[1])
    p = tmp_path / "flipped.json"
    p.write_text(json.dumps(mut))
    rc1, out1 = run_node(p, "--use-exported-weights", *mode)
    assert out1["weights_sha256"] != E["weights_sha256"]           # the flipped weights were loaded
    bad = mismatches(case, out1, exported_weights=True)
    assert rc1 == 1 and set(out1["mismatches"]) == set(bad)
    assert any(b.endswith("state_digest") for b in bad), bad


def test_synapse_count_change_is_detected(synthetic, tmp_path):
    path, case = synthetic["s2-inhib-dt0.2-lesion"]
    pre, j = _busiest_synapse(case)
    mut = copy.deepcopy(case)
    mut["network"]["counts"][j] += 1 if mut["network"]["counts"][j] != -1 else 2
    p = tmp_path / "count.json"
    p.write_text(json.dumps(mut))
    rc, out = run_node(p)
    bad = mismatches(case, out)
    assert rc == 1 and "network_digest" in bad and "weights_sha256" in bad
    assert any(b.endswith("state_digest") for b in bad), bad
