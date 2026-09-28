"""G1c provenance (the pre-registered decision-path share): synthetic cases with hand-computed
answers, the bookkeeping mirroring the simulator, and the gate bundle when its data are present.

Run from brain:  .venv/bin/python -m pytest -q tests/test_provenance.py
"""
import os
import sys

import numpy as np
import pytest

BRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BRAIN)

from fishbrain import provenance as P  # noqa: E402
from fishbrain import sim as S  # noqa: E402

CL = ("a", "b", "c")
BIG = 1000          # synapse count that fires the next cell from one presynaptic spike (Shiu weights)


def kick(sim, idx, at_ms=0.0, name="retina"):
    """A Poisson source with p >= 1 for one step: a guaranteed kick of every idx at `at_ms`."""
    dt = sim.p.dt_ms
    sim.add_poisson(np.atleast_1d(idx), 2000.0 / dt * 1000.0, start_ms=at_ms, stop_ms=at_ms + dt, name=name)


def net_of(edges, n):
    """edges: (pre, post, count, klass or None). Returns the summed Network and its EdgeProvenance."""
    pre = [e[0] for e in edges]
    post = [e[1] for e in edges]
    cnt = [e[2] for e in edges]
    br = [e for e in edges if e[3] is not None]
    net = S.Network.from_pairs(pre, post, cnt, n=n)
    prov = P.EdgeProvenance(CL, [e[0] for e in br], [e[1] for e in br], [e[2] for e in br], [e[3] for e in br])
    return net, prov


def run(net, prov, inputs, ms=40.0, labels=None, windows=None, **kw):
    r, tr = P.run_traced(net, "prov-test", inputs, duration_ms=ms, prov=prov, source_labels=labels,
                         windows=windows or {"all": (0.0, ms)}, **kw)
    assert_checks(tr)
    return r, tr


def assert_checks(tr):
    c = tr.checks
    assert c["mirror_ok"], c["mirror"]
    assert all(w["ok"] for w in c["parts_sum_to_total"]), c["parts_sum_to_total"]
    assert c["simulator"]["ok"], c["simulator"]
    if len(tr.spike_comp):
        assert np.allclose(tr.spike_comp.sum(axis=1), 1.0, atol=1e-6)
        assert (tr.spike_comp >= 0).all()


def m_of(tr, i, w="all"):
    return tr.share([i], w)["m"]


def spikes(tr, i):
    return [tr.spike_share(int(p)) for p in tr.spikes_of(i)]


# ---------------------------------------------------------------------------------------------
# the pre-registered cases

def test_all_measured_chain_gives_m_1():
    net, prov = net_of([(0, 1, BIG, None), (1, 2, BIG, None), (2, 3, BIG, None)], 5)
    r, tr = run(net, prov, lambda s: kick(s, 0))
    for i in (1, 2, 3):
        assert m_of(tr, i) == 1.0
        sp = spikes(tr, i)
        assert len(sp) == 1 and sp[0]["carried_measured"] == 1.0 and sp[0]["m"] == 1.0
    assert tr.share([4], "all")["label"] == "no drive"
    assert tr.synapse_count["measured_share"] == 1.0


def test_bridge_in_the_middle_zeroes_everything_downstream():
    net, prov = net_of([(0, 1, BIG, None), (1, 2, BIG, "c"), (2, 3, BIG, None)], 5)
    r, tr = run(net, prov, lambda s: kick(s, 0))
    assert m_of(tr, 1) == 1.0
    for i in (2, 3):
        sh = tr.share([i], "all")
        assert sh["m"] == 0.0
        assert sh["by_path"] == {"c": 1.0}
        assert sh["bridge_through"] == {"a": 0.0, "b": 0.0, "c": 1.0}
        assert sh["bridge_only"]["c"] == 1.0
        assert spikes(tr, i)[0]["carried"] == {"c": 1.0}
    assert tr.synapse_count["class_share"]["c"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("cm,cb,expect", [(BIG, BIG, 0.5), (3 * BIG, BIG, 0.75), (BIG, 3 * BIG, 0.25)])
def test_convergence_gives_the_weighted_share(cm, cb, expect):
    # 0 and 1 are eye-driven on the same step; 0 -> 2 measured, 1 -> 2 bridge (a); 2 -> 3 measured
    net, prov = net_of([(0, 2, cm, None), (1, 2, cb, "a"), (2, 3, BIG, None)], 4)
    r, tr = run(net, prov, lambda s: kick(s, [0, 1]))
    assert m_of(tr, 2) == pytest.approx(expect, abs=1e-12)
    assert tr.share([2], "all")["by_path"] == pytest.approx({"measured": expect, "a": 1 - expect}, abs=1e-12)
    first = spikes(tr, 2)[0]                      # both arrive on one step: the membrane holds them in ratio
    assert first["carried_measured"] == pytest.approx(expect, abs=1e-12)
    assert m_of(tr, 3) == pytest.approx(expect, abs=1e-12)     # a measured synapse passes the source's m on


def test_eye_input_alone_is_labelled_chosen():
    net, prov = net_of([(0, 1, BIG, None)], 2)
    r, tr = run(net, prov, lambda s: kick(s, 0))
    sh = tr.share([0], "all")
    assert sh["m"] is None and sh["label"] == P.EYE_LABEL == "chosen (eye model)"
    assert sh["chosen_share"] == {P.EYE_LABEL: 1.0}
    assert sh["link_drive_mV_ms"] == 0.0 and sh["root_drive_mV_ms"][P.EYE_LABEL] > 0
    sp = spikes(tr, 0)[0]
    assert sp["label"] == P.EYE_LABEL and sp["m"] is None and sp["direct_root_share"] == 1.0
    assert sp["carried_measured"] == 1.0          # the eye model is the root, never a bridge
    assert m_of(tr, 1) == 1.0 and tr.share([1], "all")["chosen_share"][P.EYE_LABEL] == 0.0


def test_mutation_control_flags_are_what_the_metric_reads():
    edges = [(0, 1, BIG, None), (1, 2, BIG, None), (2, 3, BIG, None)]
    base_net, base_prov = net_of(edges, 4)
    r0, t0 = run(base_net, base_prov, lambda s: kick(s, 0))
    assert [m_of(t0, i) for i in (1, 2, 3)] == [1.0, 1.0, 1.0]
    for flip, want in ((1, [1.0, 0.0, 0.0]), (0, [0.0, 0.0, 0.0]), (2, [1.0, 1.0, 0.0])):
        e = [(p, q, c, "b" if j == flip else None) for j, (p, q, c, _) in enumerate(edges)]
        net, prov = net_of(e, 4)
        r, tr = run(net, prov, lambda s: kick(s, 0))
        assert [m_of(tr, i) for i in (1, 2, 3)] == want
        # the flags are bookkeeping only: same wiring, same spikes, a different provenance digest
        assert tr.digests["network"] == t0.digests["network"]
        assert tr.digests["raster"] == t0.digests["raster"]
        assert tr.digests["edge_provenance"] != t0.digests["edge_provenance"]
    # both convergent inputs bridged -> 0; neither -> 1
    for kl, want in ((("a", "c"), 0.0), ((None, None), 1.0)):
        net, prov = net_of([(0, 2, BIG, kl[0]), (1, 2, BIG, kl[1])], 3)
        r, tr = run(net, prov, lambda s: kick(s, [0, 1]))
        assert m_of(tr, 2) == want
    net, prov = net_of([(0, 2, BIG, "a"), (1, 2, BIG, "c")], 3)
    r, tr = run(net, prov, lambda s: kick(s, [0, 1]))
    sh = tr.share([2], "all")
    assert sh["bridge_through"] == pytest.approx({"a": 0.5, "b": 0.0, "c": 0.5})


# ---------------------------------------------------------------------------------------------
# time order, masks, inhibition, inputs

def test_time_order_a_later_bridge_does_not_rewrite_an_earlier_spike():
    # 0 (eye at 0 ms) -> 2 measured; 1 (eye at 10 ms) -> 2 bridge (b); 2 -> 3 measured
    net, prov = net_of([(0, 2, BIG, None), (1, 2, BIG, "b"), (2, 3, BIG, None)], 4)

    def inputs(s):
        kick(s, 0, 0.0, "retina")
        kick(s, 1, 10.0, "retina2")
    r, tr = run(net, prov, inputs, labels={"retina2": "eye"},
                windows={"all": (0.0, 40.0), "early": (0.0, 5.0), "late": (5.0, 40.0)})
    s2 = spikes(tr, 2)
    assert len(s2) == 2
    assert s2[0]["carried"] == {"measured": 1.0}      # the reset wiped it; the bridge came after
    assert s2[1]["carried"] == {"b": 1.0}
    assert m_of(tr, 2, "early") == 1.0 and m_of(tr, 2, "late") == 0.0
    assert m_of(tr, 2, "all") == pytest.approx(0.5, abs=1e-12)
    s3 = spikes(tr, 3)
    assert [s["carried_measured"] for s in s3] == [1.0, 0.0]


def test_window_counts_arrivals_not_emissions():
    # 0 fires at 0.1 ms; its spike reaches 1 at 1.9 ms (delay 1.8 ms)
    net, prov = net_of([(0, 1, BIG, None)], 2)
    r, tr = run(net, prov, lambda s: kick(s, 0), windows={"w": (1.0, 40.0), "before": (0.0, 1.0)})
    assert m_of(tr, 1, "w") == 1.0
    assert tr.share([1], "before")["label"] == "no drive"


def test_two_hops_mix_hand_computed():
    # X=2 carries 0.5 (0 measured + 1 bridged, same step); Y=4 gets X (measured) and C=3 (measured)
    net, prov = net_of([(0, 2, BIG, None), (1, 2, BIG, "a"), (2, 4, BIG, None), (3, 4, BIG, None)], 5)
    r, tr = run(net, prov, lambda s: kick(s, [0, 1, 3]))
    assert m_of(tr, 4) == pytest.approx((0.5 * BIG + 1.0 * BIG) / (2 * BIG), abs=1e-12)
    ys = spikes(tr, 4)
    assert [s["carried_measured"] for s in ys] == pytest.approx([1.0, 0.5], abs=1e-12)


def test_paths_through_two_classes_get_both_bits():
    net, prov = net_of([(0, 1, BIG, "a"), (1, 2, BIG, "c"), (2, 3, BIG, None)], 4)
    r, tr = run(net, prov, lambda s: kick(s, 0))
    sh = tr.share([3], "all")
    assert sh["by_path"] == {"a+c": 1.0}
    assert sh["bridge_through"] == {"a": 1.0, "b": 0.0, "c": 1.0}
    assert sh["bridge_only"] == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_inhibitory_bridge_is_reported_beside_not_inside_m():
    net, prov = net_of([(0, 2, BIG, None), (1, 2, -300, "c")], 3)
    r, tr = run(net, prov, lambda s: kick(s, [0, 1]))
    sh = tr.share([2], "all")
    assert sh["m"] == 1.0
    assert sh["inhibitory"]["m"] == 0.0 and sh["inhibitory"]["bridge_through"]["c"] == 1.0
    assert sh["m_abs"] == pytest.approx(BIG / (BIG + 300), abs=1e-12)


def test_inhibition_is_not_carried_by_a_spike():
    # 2 fires on a bridged excitatory input (a) while a measured inhibitory input also sits on it
    net, prov = net_of([(1, 2, BIG, "a"), (0, 2, -300, None), (2, 3, BIG, None)], 4)
    r, tr = run(net, prov, lambda s: kick(s, [0, 1]))
    sp = spikes(tr, 2)[0]
    assert sp["carried"] == {"a": 1.0} and sp["m"] == 0.0
    assert m_of(tr, 3) == 0.0 and tr.share([3], "all")["bridge_only"]["a"] == 1.0
    assert tr.share([2], "all")["inhibitory"]["m"] == 1.0


def test_mixed_pair_parts_stay_apart():
    # 0 -> 1 carries +3000 measured and -1000 bridge (c) on one pair; the network holds +2000
    net, prov = net_of([(0, 1, 3 * BIG, None), (0, 1, -BIG, "c")], 2)
    assert net.nnz == 1 and int(net.counts[0]) == 2 * BIG
    r, tr = run(net, prov, lambda s: kick(s, 0))
    sh = tr.share([1], "all")
    assert sh["m"] == 1.0 and sh["inhibitory"]["bridge_through"]["c"] == 1.0
    assert sh["m_abs"] == pytest.approx(0.75, abs=1e-12)
    sc = tr.synapse_count
    assert sc["synapses"] == {"measured": 3000, "total": 4000, "a": 0, "b": 0, "c": 1000}


def test_bridge_labelled_input_carries_its_class():
    net, prov = net_of([(0, 1, BIG, None)], 2)
    r, tr = run(net, prov, lambda s: kick(s, 0, 0.0, "fill"), labels={"fill": "bridge:b"})
    assert tr.share([0], "all")["m"] == 0.0 and tr.share([0], "all")["by_path"] == {"b": 1.0}
    assert m_of(tr, 1) == 0.0 and tr.share([1], "all")["bridge_through"]["b"] == 1.0


def test_unlabelled_or_bad_inputs_fail_closed():
    net, prov = net_of([(0, 1, BIG, None)], 2)
    with pytest.raises(ValueError, match="no provenance label"):
        P.run_traced(net, "x", lambda s: kick(s, 0, 0.0, "tonic"), duration_ms=5.0, prov=prov)
    with pytest.raises(ValueError, match="bridge class"):
        P.run_traced(net, "x", lambda s: kick(s, 0, 0.0, "tonic"), duration_ms=5.0, prov=prov,
                     source_labels={"tonic": "bridge:z"})
    with pytest.raises(ValueError, match="unknown bridge class"):
        P.EdgeProvenance(CL, [0], [1], [5], ["z"])


def test_a_root_is_a_target_set_not_a_name():
    # 1 is a fragment the eye model must not reach: driving it through "retina" would hide a bridge
    net, prov = net_of([(0, 2, BIG, None), (1, 2, BIG, None)], 3)
    with pytest.raises(ValueError, match="outside the pre-registered root set"):
        P.run_traced(net, "x", lambda s: kick(s, [0, 1]), duration_ms=5.0, prov=prov, root_targets=[0])
    with pytest.raises(ValueError, match="outside the pre-registered root set"):
        P.run_traced(net, "x", lambda s: kick(s, 1, 0.0, "tonic"), duration_ms=5.0, prov=prov,
                     source_labels={"tonic": "chosen:tonic"}, root_targets=[0])
    # the same link as a bridge input is allowed, and is counted as a bridge
    def inputs(s):
        kick(s, 0)
        kick(s, 1, 0.0, "fill")
    r, tr = run(net, prov, inputs, labels={"fill": "bridge:a"}, root_targets=[0])
    assert tr.checks["root_guard"]["enforced"] and tr.checks["root_guard"]["allowed"] == 1
    assert m_of(tr, 2) == pytest.approx(0.5, abs=1e-12)
    # the root target set is pinned: another set, another digest
    r2, tr2 = run(net, prov, lambda s: kick(s, [0, 1]), root_targets=[0, 1])
    r3, tr3 = run(net, prov, lambda s: kick(s, [0]), root_targets=[0, 1])
    assert tr2.digests["root_targets"]["retina"] != tr3.digests["root_targets"]["retina"]
    assert tr2.digests["root_allowed"] == tr3.digests["root_allowed"] != tr.digests["root_allowed"]


def test_needs_a_fresh_run_and_pristine_sources():
    net, prov = net_of([(0, 1, BIG, None)], 2)
    sim = S.Simulator(net, seed="x")
    kick(sim, 0)
    sim.run(5.0)
    with pytest.raises(ValueError, match="before it ran"):
        P.trace(net, sim.raster(), sim, prov=prov)
    r2 = sim.run(5.0)
    with pytest.raises(ValueError, match="fresh simulator"):
        P.trace(net, r2, [], prov=prov)


def test_lesion_skips_dead_pairs_and_unknown_pairs_raise():
    net, prov = net_of([(0, 1, BIG, None), (1, 2, BIG, "c"), (2, 3, BIG, None)], 4)
    les = net.lesion_indices([2])
    al = prov.align(les)
    assert al.skipped_dead == 1 and len(al.epos) == 0
    r, tr = run(les, prov, lambda s: kick(s, 0))
    assert m_of(tr, 1) == 1.0 and tr.share([3], "all")["label"] == "no drive"
    bad = P.EdgeProvenance(CL, [0], [3], [5], ["a"])
    with pytest.raises(ValueError, match="not in the network"):
        bad.align(net)


def test_synapse_count_share_hand_computed():
    edges = [(0, 1, 2, None), (1, 2, 2, None), (2, 3, 2, None), (3, 4, 4, "b"), (0, 2, 3, None), (0, 2, -1, "a")]
    net, prov = net_of(edges, 5)
    sc = P.synapse_count_share(net, prov)
    assert sc["synapses"] == {"measured": 9, "total": 14, "a": 1, "b": 4, "c": 0}
    assert sc["measured_share"] == pytest.approx(9 / 14)
    assert sc["class_share"] == pytest.approx({"a": 1 / 14, "b": 4 / 14, "c": 0.0})
    assert (sc["pairs"], sc["pairs_with_bridge_part"], sc["pairs_bridge_only"]) == (5, 2, 1)
    onto = P.synapse_count_share(net, prov, onto=[4])
    assert onto["measured_share"] == 0.0 and onto["synapses"]["b"] == 4


def test_provenance_digest_pins_the_table():
    a = P.EdgeProvenance(CL, [0, 1], [1, 2], [3, 4], ["a", "c"])
    b = P.EdgeProvenance(CL, [1, 0], [2, 1], [4, 3], ["c", "a"])          # same table, other order
    c = P.EdgeProvenance(CL, [0, 1], [1, 2], [3, 4], ["a", "b"])
    assert a.digest() == b.digest() != c.digest()
    d = P.EdgeProvenance(CL, [0, 0], [1, 1], [2, 1], ["a", "a"])         # duplicates sum
    assert d.parts.tolist() == [[3, 0, 0]]


# ---------------------------------------------------------------------------------------------
# the bookkeeping mirrors the simulator on a random bridged network

def random_bridged(seed="prov-random", n=400, n_syn=12000, frac_bridge=0.2):
    base = S.synthetic_graph(n, n_syn, seed=seed, frac_type1=0.8)       # synthetic signs {1: +1, 2: -1}
    rows, cols, cnt = base.rows(), base.indices.astype(np.int64), base.counts.astype(np.int64)
    rng = np.random.Generator(np.random.PCG64(7))
    k_old = rng.choice(len(rows), int(frac_bridge * len(rows)), replace=False)
    n_new = 800
    b_pre = np.concatenate([rows[k_old], rng.integers(0, n, n_new)])
    b_post = np.concatenate([cols[k_old], rng.integers(0, n, n_new)])
    b_cnt = np.concatenate([rng.integers(-2, 4, len(k_old)), rng.integers(-2, 5, n_new)])
    b_cls = rng.choice(list(CL), len(b_pre))
    keep = b_cnt != 0
    b_pre, b_post, b_cnt, b_cls = b_pre[keep], b_post[keep], b_cnt[keep], b_cls[keep]
    net = S.Network.from_pairs(np.concatenate([rows, b_pre]), np.concatenate([cols, b_post]),
                               np.concatenate([cnt, b_cnt]), n)
    present = set(zip(net.rows().tolist(), net.indices.tolist()))
    ok = np.array([(int(p), int(q)) in present for p, q in zip(b_pre, b_post)])   # drop pairs that netted to 0
    return net, P.EdgeProvenance(CL, b_pre[ok], b_post[ok], b_cnt[ok], b_cls[ok])


RAND_LABELS = {"background": "chosen:background", "drive": "chosen:current", "fill": "bridge:b"}
RAND_WINDOWS = {"early": (0.0, 50.0), "late": (50.0, 150.0), "all": (0.0, 150.0)}


def rand_inputs(sim):
    sim.add_poisson(np.arange(0, 40), 60.0, start_ms=5.0, stop_ms=120.0, name="retina")
    sim.add_poisson(np.arange(40, 60), np.linspace(0, 40, 80).reshape(4, 20), frame_ms=25.0, name="background")
    sim.add_current(np.arange(60, 65), 9.0, start_ms=20.0, stop_ms=60.0, name="drive")
    sim.add_poisson(np.arange(65, 70), 50.0, start_ms=0.0, stop_ms=150.0, name="fill")


@pytest.fixture(scope="module")
def rand_run():
    net, prov = random_bridged()
    params = S.LIFParams(w_syn_mV=2.0)
    r, tr = P.run_traced(net, "prov-rand", rand_inputs, duration_ms=150.0, params=params, prov=prov,
                         source_labels=RAND_LABELS, windows=RAND_WINDOWS)
    return net, prov, params, r, tr


def test_random_network_mirrors_the_simulator(rand_run):
    net, prov, params, r, tr = rand_run
    assert_checks(tr)
    mir = tr.checks["mirror"]
    assert len(r.neurons) > 200 and tr.checks["syn_events"] > 5000          # the check saw real traffic
    assert mir["min_margin_fired_mV"] < 0.1 and mir["max_free_minus_theta_mV"] > -0.1   # and near-threshold cases
    assert int((tr.spike_root < 0.5).sum()) > 20                              # spikes caused by synapses
    assert tr.checks["live_channels"] > 10                                    # many masks were live


def test_random_network_parts_sum_to_one(rand_run):
    net, prov, params, r, tr = rand_run
    for w in RAND_WINDOWS:
        m = tr.m(w)
        fin = np.isfinite(m)
        assert fin.sum() > 100 and ((m[fin] >= 0) & (m[fin] <= 1)).all()
        for i in tr.touched[fin][:60]:
            sh = tr.share([int(i)], w)
            assert sum(sh["by_path"].values()) == pytest.approx(1.0, abs=1e-9)
            assert sh["m"] == pytest.approx(sh["by_path"].get("measured", 0.0), abs=1e-12)
            for c in CL:
                assert sh["bridge_through"][c] >= sh["bridge_only"][c] - 1e-12
        # the pooled share is the drive-weighted mean of the cells' m
        pooled = tr.share(tr.touched[fin], w)
        drive = np.array([tr.share([int(i)], w)["link_drive_mV_ms"] for i in tr.touched[fin][:80]])
        ms = m[fin][:80]
        sub = tr.share(tr.touched[fin][:80], w)
        assert sub["m"] == pytest.approx(float((drive * ms).sum() / drive.sum()), rel=1e-9)
        assert 0 < pooled["m"] < 1
    # early + late windows add up to the whole run
    for i in tr.touched[:50]:
        e, l, a = (tr.drive[w][:, np.searchsorted(tr.touched, i)] for w in ("early", "late", "all"))
        assert np.allclose(e + l, a, rtol=1e-12, atol=1e-12)


def test_random_network_fast_simulator_gives_the_same_trace(rand_run):
    net, prov, params, r, tr = rand_run
    r2, tr2 = P.run_traced(net, "prov-rand", rand_inputs, duration_ms=150.0, params=params, prov=prov,
                           source_labels=RAND_LABELS, windows=RAND_WINDOWS, fast=True)
    # make_simulator picks the reference Simulator below its size cutoff; force the active one too
    from fishbrain.sim_fast import ActiveSimulator
    import copy
    sim = ActiveSimulator(net, seed="prov-rand", params=params)
    rand_inputs(sim)
    pristine = copy.deepcopy(sim.sources)
    r3 = sim.run(150.0)
    tr3 = P.trace(net, r3, pristine, params, prov, RAND_LABELS, RAND_WINDOWS, sim_after=sim)
    for t in (tr2, tr3):
        assert t.digests == tr.digests
        assert np.array_equal(t.spike_comp, tr.spike_comp)
        for w in RAND_WINDOWS:
            assert np.array_equal(t.drive[w], tr.drive[w])


def test_simulator_is_untouched(rand_run):
    net, prov, params, r, tr = rand_run
    sim = S.Simulator(net, seed="prov-rand", params=params)
    rand_inputs(sim)
    plain = sim.run(150.0)
    assert S.spike_hash(plain) == S.spike_hash(r) == tr.digests["raster"]


def test_report_names_the_escaping_cell_and_pools_the_deciders():
    # 0 (eye, 0 ms) -> 2 measured; 1 (eye, 5 ms) -> 3 bridge (c): cell 2 fires first
    net, prov = net_of([(0, 2, BIG, None), (1, 3, BIG, "c")], 4)

    def inputs(s):
        kick(s, 0, 0.0, "retina")
        kick(s, 1, 5.0, "retina2")
    r, tr = run(net, prov, inputs, labels={"retina2": "eye"}, windows={"decision": (0.0, 40.0)})
    groups = {"Mauthner_left": np.array([2]), "Mauthner_right": np.array([], np.int64),
              "turning_left": np.array([3]), "nMLF_left": np.array([], np.int64)}
    rep = P.report(tr, groups, "decision")
    assert rep["escape"]["initiator"] == "Mauthner_left" and rep["escape"]["escape_to"] == "right"
    assert rep["groups"]["escape"]["m"] == 1.0 and rep["groups"]["turn"]["m"] == 0.0
    assert rep["escape"]["first_spike"]["Mauthner_left"]["carried_measured"] == 1.0
    assert rep["decision_path_groups"] == ["escape", "turn"]
    assert rep["decision_path_share"]["m"] == pytest.approx(0.5, abs=1e-12)
    assert rep["decision_path_share"]["bridge_through"]["c"] == pytest.approx(0.5, abs=1e-12)
    assert set(rep["readouts_absent_from_network"]) == {"Mauthner_right", "nMLF_left"}
    import json
    json.dumps(P.to_json(rep))


# ---------------------------------------------------------------------------------------------
# the gate bundle (skipped when the brain of record is not on this machine)

DATA = os.path.join(BRAIN, "data")
HAVE_DATA = os.path.exists(os.path.join(DATA, "net", "g1_per_synapse_intact.npz")) and \
    os.path.exists(os.path.join(DATA, "readouts.json"))


@pytest.fixture(scope="module")
def gate():
    if not HAVE_DATA:
        pytest.skip("gate bundle not on this machine")
    from fishbrain import brain as B
    from fishbrain import readouts as RO
    b = B.build_bundle()
    rates, diag = B.cached_rates(B.Retina(b.stim_ids), "loom_left")
    groups, missing = P.readout_groups(b.net, RO.readout_sets(RO.load()))
    return B, b, rates, diag, groups


def test_gate_trial_all_measured(gate):
    B, b, rates, diag, groups = gate
    w = B.window_for("loom_left")
    ref = B.run_trial(b, rates, "G1-seed-0", "loom_left", rates_sha=diag["rates_sha256"])
    r, tr = P.trial_traced(b, rates, "G1-seed-0", windows={"decision": w})
    assert S.spike_hash(r) == ref.spike_hash                 # the same run as the gate's
    assert_checks(tr)
    assert tr.checks["root_guard"]["enforced"] and tr.checks["root_guard"]["root_sources"] == {"retina": len(b.stim_idx)}
    m = tr.m("decision")
    fin = np.isfinite(m)
    assert fin.sum() > 100 and (m[fin] == 1.0).all()         # no bridge -> every wired cell is measured
    assert tr.synapse_count["measured_share"] == 1.0
    rep = P.report(tr, groups, "decision")
    for k, v in rep["per_readout"].items():
        assert v["m"] in (None, 1.0), k


def test_gate_eye_set_of_record_and_the_guard(gate):
    B, b, rates, diag, groups = gate
    seg = P.eye_segments_of_record()
    assert len(seg) == b.stats["stim_info"]["stimulated"] == 24913
    assert np.isin(b.stim_ids, seg).all()
    # a "retina" that also drives a cell outside the stimulated set (here the left Mauthner) is refused
    extra = np.uint64(B.mauthner_ids(b)["left"])
    assert extra not in seg
    ids = np.sort(np.append(b.stim_ids, extra))
    fake = B.Bundle(b.net, ids, b.readouts, b.stats, b.variant, b.rule)
    col = int(np.searchsorted(ids, extra))
    r2 = np.insert(rates[:20], col, 0.0, axis=1)
    with pytest.raises(ValueError, match="outside the pre-registered root set"):
        P.trial_traced(fake, r2, "G1-seed-0")


def test_gate_trial_exercise_flags(gate):
    """Exercise only (not a bridge): 10% of the gate network's pairs flagged a or c. The spikes do
    not change; the bookkeeping must still mirror the run and sum to one."""
    B, b, rates, diag, groups = gate
    net = b.net
    rng = np.random.Generator(np.random.PCG64(11))
    k = rng.choice(net.nnz, net.nnz // 10, replace=False)
    rows = net.rows()
    prov = P.EdgeProvenance(CL, rows[k], net.indices[k], net.counts[k], rng.choice(["a", "c"], len(k)))
    w = B.window_for("loom_left")
    r, tr = P.trial_traced(b, rates, "G1-seed-0", prov=prov, windows={"decision": w})
    assert_checks(tr)
    m = tr.m("decision")
    fin = np.isfinite(m)
    assert fin.sum() > 100 and (m[fin] < 1.0).any() and (m[fin] > 0.0).any()
    sc = tr.synapse_count
    assert sc["measured_share"] + sum(sc["class_share"].values()) == pytest.approx(1.0)
