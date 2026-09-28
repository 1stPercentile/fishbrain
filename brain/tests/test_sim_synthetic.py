"""Simulator and stimuli tests on synthetic graphs only (G1, task B2).

Run from brain:  .venv/bin/python -m pytest tests/test_sim_synthetic.py
"""
import hashlib
import math
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

BRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BRAIN)

from fishbrain import sim  # noqa: E402
from fishbrain import stimuli as st  # noqa: E402
from fishbrain.sim import LIFParams, Network, Simulator, spike_hash  # noqa: E402

SEED = "0x9f3c2a7e5d1b4c8a0e6f2d9b7a5c3e1f0d8b6a4c2e0f9d7b5a3c1e9f7d5b3a1c"  # block-hash shaped; guard:public synthetic test seed, not a key or a real block
EXC_FIRST = {1: +1, 2: -1}


def chain(n_links=4, count=1000, n_extra=1, ids=None):
    """0 -> 1 -> ... -> n_links, plus unconnected neurons."""
    pre = list(range(n_links))
    post = list(range(1, n_links + 1))
    n = n_links + 1 + n_extra
    return Network.from_pairs(pre, post, [count] * n_links, n=n, ids=ids)


def fire_once(sim_, idx, at_ms=0.0):
    """A Poisson source with p >= 1 for one step: a guaranteed kick at `at_ms`."""
    dt = sim_.p.dt_ms
    sim_.add_poisson([idx], 2000.0 / dt * 1000.0, start_ms=at_ms, stop_ms=at_ms + dt, name=f"kick{idx}@{at_ms}")


def first_spike(r, i):
    s = r.steps[r.neurons == i]
    return int(s[0]) if len(s) else None


# ---------------------------------------------------------------------------------------------
# seeds and parameters

def test_seed_is_sha256_of_string():
    assert sim.derive_seed(SEED) == int.from_bytes(hashlib.sha256(SEED.encode()).digest(), "big")
    assert sim.derive_seed(SEED, "poisson", "a") == int.from_bytes(
        hashlib.sha256(f"{SEED}|poisson|a".encode()).digest(), "big")
    a = sim.make_bitgen(SEED).random_raw(4)
    b = np.random.PCG64(int.from_bytes(hashlib.sha256(SEED.encode()).digest(), "big")).random_raw(4)
    assert np.array_equal(a, b)


def test_defaults_are_shiu_2024():
    p = LIFParams()
    assert (p.v_rest_mV, p.v_reset_mV, p.v_thresh_mV) == (-52.0, -52.0, -45.0)
    assert (p.tau_m_ms, p.tau_syn_ms, p.refractory_ms, p.delay_ms) == (20.0, 5.0, 2.2, 1.8)
    assert p.w_syn_mV == 0.275 and p.poisson_weight_mV == 250 * 0.275 and p.dt_ms == 0.1
    d = p.derived()
    assert d["n_ref"] == 22 and d["n_delay"] == 18
    assert d["a"] == np.float32(math.exp(-0.1 / 20.0)) and d["b"] == np.float32(math.exp(-0.1 / 5.0))
    assert p.digest() == LIFParams().digest() != LIFParams(w_syn_mV=0.3).digest()
    with pytest.raises(ValueError):
        LIFParams(dt_ms=0.25)  # 1.8 ms is not a whole number of 0.25 ms steps


# ---------------------------------------------------------------------------------------------
# dynamics against closed forms

def test_integrator_matches_analytic_psp():
    """One presynaptic spike through `count` synapses gives the alpha-like PSP of
    dv/dt = (v0 - v + g)/tm, dg/dt = -g/ts, g(0) = w:  u(t) = w ts/(tm - ts) (e^-t/tm - e^-t/ts)."""
    count = 100
    net = Network.from_pairs([0], [1], [count], n=2)
    p = LIFParams()
    s = Simulator(net, SEED, p)
    fire_once(s, 0)
    u1 = []
    for _ in range(600):
        s.run(n_steps=1)
        u1.append(float(s.u[1]))
    u1 = np.array(u1)
    r = s.raster()
    assert first_spike(r, 0) == 1 and len(r.neurons) == 1           # neuron 1 stays subthreshold
    arrival = 1 + p.derived()["n_delay"]                               # g jumps in step 19's synapse phase
    assert np.all(u1[:arrival + 1] == 0.0)
    k = np.arange(arrival + 1, 600)
    t = (k - arrival) * p.dt_ms
    w = count * p.w_syn_mV
    tm, ts = p.tau_m_ms, p.tau_syn_ms
    analytic = w * ts / (tm - ts) * (np.exp(-t / tm) - np.exp(-t / ts))
    assert np.max(np.abs(u1[k] - analytic)) < 5e-5
    t_peak = math.log(tm / ts) * tm * ts / (tm - ts)                   # 9.242 ms
    assert abs(t[np.argmax(u1[k])] - t_peak) <= p.dt_ms
    assert abs(u1[k].max() / w - 0.15749) < 1e-3


@pytest.mark.parametrize("dt", [0.1, 0.2])
def test_refractory_period_is_exact(dt):
    p = LIFParams(dt_ms=dt)
    s = Simulator(chain(1, count=0, n_extra=0), SEED, p)
    s.add_current([0], 1e5)
    r = s.run(20.0)
    isi = np.diff(r.steps[r.neurons == 0])
    assert len(isi) > 5 and np.all(isi == round(2.2 / dt))


def test_refractory_neuron_is_frozen_but_still_receives():
    """Brian2 '(unless refractory)': during t_rfc neither v nor g integrates, but synaptic input
    still lands on g and is felt once the neuron recovers."""
    count = 100
    net = Network.from_pairs([0], [1], [count], n=2)
    p = LIFParams()
    s = Simulator(net, SEED, p)
    fire_once(s, 0)          # neuron 0 fires at step 1 -> arrives at neuron 1 in step 19
    fire_once(s, 1)          # neuron 1 fires at step 1 -> refractory through step 22
    u1, h1 = [], []
    for _ in range(30):
        s.run(n_steps=1)
        u1.append(float(s.u[1]))
        h1.append(float(s.h[1]))
    r = s.raster()
    assert first_spike(r, 0) == 1 and first_spike(r, 1) == 1
    cw = np.float32(count * p.w_syn_mV * p.derived()["c64"])
    assert all(u == 0.0 for u in u1[1:23])                            # v held at reset, steps 1..22
    assert h1[18] == 0.0 and h1[19] == h1[20] == h1[21] == h1[22] == cw   # g lands, does not decay
    assert u1[23] == cw and h1[23] == np.float32(cw * p.derived()["b"])    # step 23 integrates


def test_spike_resets_g():
    s = Simulator(chain(1, count=2000), SEED)
    fire_once(s, 0)
    for _ in range(60):
        s.run(n_steps=1)
        if len(s.raster().neurons) and 1 in s.raster().neurons:
            break
    assert first_spike(s.raster(), 1) == s.step - 1
    assert s.h[1] == 0.0 and s.u[1] == 0.0


def test_poisson_kick_during_refractory_is_deferred_not_lost():
    """Brian2's PoissonInput writes v even while refractory; v is then frozen and the neuron
    fires on its first non-refractory step (flycoinrh instead discards such kicks)."""
    s = Simulator(chain(1, count=0, n_extra=0), SEED)
    fire_once(s, 0, at_ms=0.0)     # fires at step 1
    fire_once(s, 0, at_ms=0.5)     # kick at step 5, inside the refractory window
    r = s.run(5.0)
    assert list(r.steps[r.neurons == 0]) == [1, 1 + 22]


def test_transmission_delay_is_exact():
    p = LIFParams()
    s = Simulator(chain(1, count=2000), SEED, p)
    fire_once(s, 0)
    r = s.run(10.0)
    pre, post = first_spike(r, 0), first_spike(r, 1)
    assert pre == 1
    assert post is not None and post >= pre + p.derived()["n_delay"] + 1


# ---------------------------------------------------------------------------------------------
# the four behaviours the task names

def test_chain_excitation_fires_downstream():
    net = chain(4, count=1000, n_extra=1)
    s = Simulator(net, SEED)
    s.add_poisson([0], 40.0, stop_ms=500.0)
    r = s.run(520.0)
    c = r.counts()
    assert c[0] > 5
    assert all(c[k] >= 0.8 * c[0] for k in (1, 2, 3, 4)), c
    assert c[5] == 0                                                   # unconnected neuron
    firsts = [first_spike(r, k) for k in range(5)]
    hops = np.diff(firsts)
    assert np.all(hops >= 19) and np.all(hops <= 60), firsts           # delay + rise time per hop


def _inhibition_counts(sign_map, drive_a, drive_i):
    # A=0 -(type 1, 400 syn)-> T=1 <-(type 2, 1500 syn)- I=2
    net = Network.from_synapses([0, 2], [1, 1], [1, 2], sign_map, counts=[400, 1500], ids=[0, 1, 2])
    s = Simulator(net, SEED)
    if drive_a:
        s.add_poisson([0], 40.0, name="A")
    if drive_i:
        s.add_poisson([2], 200.0, name="I")
    return s.run(1000.0).counts()


def test_inhibition_suppresses():
    base = _inhibition_counts(EXC_FIRST, True, False)
    inh = _inhibition_counts(EXC_FIRST, True, True)
    assert base[1] >= 0.8 * base[0] > 20
    assert inh[1] <= 0.1 * base[1], (base, inh)
    assert inh[0] == base[0]                                           # A itself is untouched


def test_sign_comes_from_the_map_not_the_code():
    flipped = {1: -1, 2: +1}
    a_only = _inhibition_counts(flipped, True, False)
    i_only = _inhibition_counts(flipped, False, True)
    assert a_only[0] > 20 and a_only[1] == 0                           # type 1 now inhibits
    assert i_only[1] > 20                                              # type 2 now excites
    with pytest.raises(ValueError):
        Network.from_synapses([0], [1], [3], EXC_FIRST, ids=[0, 1])   # unknown type value


def test_mixed_type_pairs_net_out():
    net = Network.from_synapses([5, 5, 5, 6], [6, 6, 6, 5], [1, 1, 2, 2], EXC_FIRST, ids=[5, 6])
    assert net.nnz == 2
    i5, i6 = net.index_of([5, 6])
    dense = np.zeros((2, 2), int)
    dense[np.repeat(np.arange(2), np.diff(net.indptr)), net.indices] = net.counts
    assert dense[i5, i6] == 1 and dense[i6, i5] == -1


def test_lesion_removes_the_neurons_influence():
    ids = np.array([10**12 + 7 * k for k in range(6)], dtype=np.uint64)   # segment-id-like ids
    net = chain(4, count=1000, n_extra=1, ids=ids)
    s0 = Simulator(net, SEED)
    s0.add_poisson([0], 40.0, name="drive")
    r0 = s0.run(500.0)
    les = net.lesion([ids[2]])
    assert les.nnz == net.nnz - 2 and les.dead[2] and les.n == net.n
    s1 = Simulator(les, SEED)
    s1.add_poisson([0], 40.0, name="drive")
    s1.add_poisson([2], 100.0, name="direct")                        # even direct drive can't fire it
    r1 = s1.run(500.0)
    c0, c1 = r0.counts(), r1.counts()
    assert c0[3] > 0 and c0[4] > 0
    assert c1[2] == 0 and c1[3] == 0 and c1[4] == 0
    for k in (0, 1):                                                  # upstream spikes unchanged
        assert np.array_equal(r0.steps[r0.neurons == k], r1.steps[r1.neurons == k])
    assert "lesioned_ids" in les.meta and les.meta["lesioned_ids"] == [int(ids[2])]


def _strong_random_net(n=300, k_out=8, count=60, seed="net"):
    bg = sim.make_bitgen(seed, "test-net")
    pre = np.repeat(np.arange(n), k_out)
    post = (bg.random_raw(n * k_out) % np.uint64(n)).astype(np.int64)
    sign = np.where(bg.random_raw(n * k_out) < np.uint64(int(0.8 * 2 ** 64)), 1, -1)
    keep = pre != post
    return Network.from_pairs(pre[keep], post[keep], sign[keep] * count, n=n)


def _row_sorted_counts(net):
    rows = net.rows()
    o = np.lexsort((net.counts, rows))
    return rows[o], net.counts[o]


def test_shuffle_preserves_degrees_and_changes_the_raster():
    net = _strong_random_net()
    sh = net.shuffled_copy("shuffle-seed-1")
    assert sh.meta["shuffle"]["unrepaired_pairs"] == 0
    assert sh.nnz == net.nnz
    assert np.array_equal(sh.out_degree(), net.out_degree())
    assert np.array_equal(sh.in_degree(), net.in_degree())
    assert np.array_equal(sh.out_strength(), net.out_strength())
    ra, ca = _row_sorted_counts(net)
    rb, cb = _row_sorted_counts(sh)
    assert np.array_equal(ra, rb) and np.array_equal(ca, cb)            # each neuron's outgoing signs
    assert not np.any(sh.rows() == sh.indices)                          # no self-loops introduced
    assert sh.digest() != net.digest()
    assert net.shuffled_copy("shuffle-seed-1").digest() == sh.digest()
    assert net.shuffled_copy("shuffle-seed-2").digest() != sh.digest()

    def run(g):
        s = Simulator(g, SEED)
        s.add_poisson(np.arange(0, 300, 10), 20.0, name="drive")
        return s.run(300.0)

    r0, r1 = run(net), run(sh)
    assert r0.counts()[np.arange(0, 300, 10)].sum() > 0
    assert spike_hash(r0) != spike_hash(r1)
    assert not np.array_equal(r0.counts(), r1.counts())
    # the undriven population differs, not just the timing
    undriven = np.setdiff1d(np.arange(300), np.arange(0, 300, 10))
    assert np.count_nonzero(r0.counts()[undriven]) > 10
    assert not np.array_equal(r0.counts()[undriven] > 0, r1.counts()[undriven] > 0)


# ---------------------------------------------------------------------------------------------
# determinism

SCENARIO = textwrap.dedent('''
    import numpy as np
    from fishbrain import sim

    def build(seed, params=None):
        net = sim.synthetic_graph(2000, 40000, seed="scenario-graph")
        s = sim.Simulator(net, seed, params or sim.LIFParams(w_syn_mV=10.0))
        s.add_poisson(np.arange(0, 2000, 10), 20.0, name="bg")
        rates = np.linspace(0.0, 80.0, 10)[:, None] * np.ones((1, 100))
        s.add_poisson(np.arange(1000, 1100), rates, frame_ms=5.0, name="frames")
        s.add_current(np.arange(500, 550), 12.0, start_ms=20.0, stop_ms=60.0, name="step")
        return s

    def scenario(seed):
        s = build(seed)
        r = s.run(100.0)
        return sim.spike_hash(r) + ":" + s.state_digest() + ":" + str(len(r.neurons))
''')


def _scenario():
    ns = {}
    exec(SCENARIO, ns)
    return ns


def test_same_seed_same_hash_in_process():
    ns = _scenario()
    a, b = ns["scenario"](SEED), ns["scenario"](SEED)
    assert a == b
    assert int(a.split(":")[2]) > 100                                  # it actually spiked


def test_same_seed_same_hash_in_a_subprocess():
    ns = _scenario()
    here = ns["scenario"](SEED)
    env = dict(os.environ, PYTHONPATH=BRAIN, PYTHONHASHSEED="12345")
    code = SCENARIO + "\nimport sys\nprint(scenario(sys.argv[1]))\n"
    out = subprocess.run([sys.executable, "-c", code, SEED], cwd=BRAIN, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == here


def test_different_seed_different_hash():
    ns = _scenario()
    assert ns["scenario"](SEED).split(":")[0] != ns["scenario"](SEED[:-1] + "d").split(":")[0]


GOLDEN = {
    # Recorded 2026-09-25 on the Apple M5 Mac (macOS 26.6.2, Python 3.9.6, numpy 2.0.2, scipy 1.13.1).
    # The same code must reproduce these bits on any machine. If this fails on another machine
    # while the other determinism tests pass there, cross-machine replay is broken: report it,
    # do not re-record it. Re-record only for a deliberate model change, with a version bump.
    "params": "958a1404a0e6dfab7682b5f9c4419aa5d3e25963cc18f582cbd99a4d69284717",
    "scenario": "459892940b54b9f2e103a9dbba5a7a5b5eb1e8011d9b663a5e5667867e610f20:"
                "c918d0e111599e5efe20e8640c8f3a6093dd30d0f9a38d6e76b6514c7601928f:23141",
}


def test_golden_hashes_pin_the_model():
    assert LIFParams().digest() == GOLDEN["params"]
    assert _scenario()["scenario"](SEED) == GOLDEN["scenario"]


def test_split_runs_and_snapshots_replay_exactly():
    ns = _scenario()
    whole = ns["build"](SEED)
    whole.run(100.0)
    split = ns["build"](SEED)
    split.run(37.0)
    split.run(63.0)
    assert spike_hash(whole.raster()) == spike_hash(split.raster())
    assert whole.state_digest() == split.state_digest()

    s = ns["build"](SEED)
    s.run(50.0)
    snap = s.snapshot()
    d_snap = s.state_digest()
    first = spike_hash(s.run(50.0))
    end1 = s.state_digest()
    s.restore(snap)
    assert s.state_digest() == d_snap
    assert spike_hash(s.run(50.0)) == first and s.state_digest() == end1


def test_spike_hash_list_form_matches_raster_form():
    r = sim.Raster(np.array([0, 0, 2]), np.array([1, 3, 0]), 0, 4, 5, 0.1)
    assert spike_hash(r) == spike_hash([[1, 3], [], [0], []], n_neurons=5)
    assert spike_hash(r) != spike_hash([[1, 3], [], [0]], n_neurons=5)  # trailing silence counts


# ---------------------------------------------------------------------------------------------
# Poisson generators

def _isolated(n):
    return Network(np.arange(n, dtype=np.uint64), np.zeros(n + 1, np.int64),
                   np.zeros(0, np.int32), np.zeros(0, np.int32))


def test_scalar_poisson_rate_and_spread():
    n, rate = 2000, 20.0
    s = Simulator(_isolated(n), SEED)
    s.add_poisson(np.arange(n), rate, name="bg")
    s.run(1000.0)
    ev = s.stats["poisson_events"]
    assert abs(ev - n * rate) < 5 * math.sqrt(n * rate)
    c = s.raster().counts()
    assert 15.0 < c.var() < 25.0                                       # Poisson: var ~ mean = 20


def test_poisson_frames_land_in_their_own_steps():
    """(F, K) rates with frame_ms: frame j covers [start + j*frame, start + (j+1)*frame), nothing
    leaks into neighbours, and the source stops drawing once its frames run out."""
    rates = np.zeros((5, 1))
    rates[2, 0] = 1e9                                # p >= 1: a kick on every step of frame 2
    s = Simulator(_isolated(1), SEED)
    s.add_poisson([0], rates, start_ms=3.0, frame_ms=1.0, name="frames")   # frame 2 = steps 50..59
    s.run(n_steps=85)                                # frames end at step 80
    src = s.sources[0]
    state_after_frames = dict(src.bg.state["state"])
    s.run(n_steps=115)
    assert src.bg.state["state"] == state_after_frames                      # silent, no draws
    assert s.stats["poisson_events"] == 10
    r = s.raster()
    # kick at 50 -> spike at 51; kicks 52..59 arrive while refractory -> deferred to step 73
    assert list(r.steps) == [51, 73]


def test_current_frames_land_in_their_own_steps():
    amps = np.zeros((8, 1))
    amps[1, 0] = amps[7, 0] = 1e5                    # frames 1 and 7 saturate; 5 steps per frame
    s = Simulator(_isolated(1), SEED)
    s.add_current([0], amps, start_ms=2.0, frame_ms=0.5, name="step")        # frame 1 = steps 25..29
    u0 = []
    for _ in range(150):
        s.run(n_steps=1)
        u0.append(float(s.u[0]))
    r = s.raster()
    assert all(u == 0.0 for u in u0[:25])
    assert list(r.steps) == [25, 55]                 # frame 7 = steps 55..59; nothing after 60


def test_array_poisson_rates():
    n = 2000
    rates = np.linspace(0.0, 40.0, n)
    s = Simulator(_isolated(n), SEED)
    s.add_poisson(np.arange(n), rates, name="per-neuron")
    s.run(1000.0)
    ev = s.stats["poisson_events"]
    assert abs(ev - rates.sum()) < 5 * math.sqrt(rates.sum())
    c = s.raster().counts()
    assert c[:100].sum() < c[-100:].sum() / 5


# ---------------------------------------------------------------------------------------------
# stimuli

def test_looming_follows_l_over_v():
    L = st.looming(azimuth_deg=90.0, l_over_v_ms=240.0, start_deg=4.0, end_deg=140.0)
    tc, t_end = st.loom_schedule(240.0, 4.0, 140.0)
    size = L.trace["size_deg"]
    assert abs(size[0] - 4.0) < 1e-9
    assert np.all(np.diff(size) >= 0)
    for t in (0.0, 1000.0, 6000.0, 6700.0):
        i = L.frame_at(t)
        assert abs(size[i] - math.degrees(2 * math.atan(240.0 / (tc - L.time_ms(i))))) < 1e-9
    assert abs(st.loom_angle_deg(t_end) - 140.0) < 1e-9
    early, late = L[0], L[len(L) - 1]
    assert early["left"] == [] and late["left"] == []                  # az 90 is outside the left eye
    assert 0 < len(early["right"]) < len(late["right"])
    x = late["right"][0]
    assert isinstance(late["right"], list) and len(x) == 3 and all(isinstance(v, float) for v in x)
    assert all(-1.0 <= c < 0.0 for _, _, c in late["right"])


def test_receding_is_the_loom_reversed():
    L = st.looming()
    R = st.receding()
    t_end = L.params["expansion_ms"]
    sz = R.trace["size_deg"]
    assert abs(sz[0] - 140.0) < 1e-9 and np.all(np.diff(sz) <= 1e-12)
    for i in (0, 100, 3000, 6700):
        assert abs(sz[i] - st.loom_angle_deg(t_end - R.time_ms(i))) < 1e-9


def test_dimming_matches_loom_luminance():
    D = st.dimming()
    L = st.looming()
    omega_eye = st.EyeModel().solid_angle_sr()
    expect = -st.disc_solid_angle_sr(L.trace["size_deg"]) / omega_eye
    assert np.allclose(D.trace["contrast"], expect, atol=1e-12)
    f = D.arrays(len(D) - 1)
    masks = st.eye_masks(st.RES_DEG, st.EyeModel())
    for e in st.EYES:
        assert len(f[e]) == masks[e].sum()
        assert np.all(f[e][:, 2] == f[e][0, 2]) and f[e][0, 2] < -0.5   # uniform, dark
    one = st.dimming(eyes="left").arrays(10)
    assert len(one["right"]) == 0 and len(one["left"]) > 0


def test_prey_dot_size_position_and_eye():
    P = st.prey_dot(30.0, size_deg=3.0, speed_deg_s=30.0, sweep_deg=20.0)
    az = P.trace["az_deg"]
    assert az[0] == 30.0 and az.min() >= 20.0 - 1e-9 and az.max() <= 40.0 + 1e-9
    assert abs(az[P.frame_at(1000.0 / 3.0)] - 40.0) < 0.05              # 10 deg in 1/3 s at 30 deg/s
    f = P.arrays(0)
    assert len(f["left"]) == 0 and 0 < len(f["right"]) <= 6
    area = np.abs(f["right"][:, 2]).sum() * st.RES_DEG ** 2
    assert 0.5 * math.pi * 1.5 ** 2 < area < 1.5 * math.pi * 1.5 ** 2
    assert np.all(f["right"][:, 2] > 0)                                # bright spot
    left = st.prey_dot(-30.0).arrays(0)
    assert len(left["right"]) == 0 and len(left["left"]) > 0
    front = st.prey_dot(0.0, sweep_deg=0.0).arrays(0)
    assert len(front["left"]) > 0 and len(front["right"]) > 0          # binocular zone
    tiny = st.prey_dot(30.0, size_deg=1.0).arrays(0)
    assert len(tiny["right"]) > 0                                      # sub-lattice dot still seen


def test_stimuli_are_deterministic():
    a, b = st.looming(), st.looming()
    assert a.digest() == b.digest() != st.looming(azimuth_deg=-90.0).digest()
    for i in (0, 3000, len(a) - 1):
        for e in st.EYES:
            assert np.array_equal(a.arrays(i)[e], b.arrays(i)[e])
