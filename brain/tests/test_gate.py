"""G1 gate on the real Fish1 wiring (task C). Pre-registered criteria: evidence/G1-gate.md s.3.

Heavy: the module fixture runs the whole protocol once (8 seeds x 8 conditions on the intact brain,
Mauthner lesion, whole-brain shuffle) in a process pool and writes data/net/g1_gate_results.json.
Needs data/synapses.parquet and data/cells_identified.json (git-ignored). Without them every test
FAILS loudly: a skipped gate would read green on a fresh clone or a server that never ran the brain.
"""
import os

import numpy as np
import pytest

from fishbrain import brain as B
from fishbrain import cells as C
from fishbrain import sim as S
from fishbrain import stimuli as ST

WORKERS = int(os.environ.get("FISHBRAIN_WORKERS", "4"))


@pytest.fixture(scope="module", autouse=True)
def brain_of_record_on_disk():
    if not (B.SYNAPSES.exists() and C.CELLS_JSON.exists()):
        pytest.fail("brain of record not on disk (data/ is git-ignored): run `python -m fishbrain.pull` "
                    "and `python -m fishbrain.cells` first. The gate cannot pass without the wiring.")


@pytest.fixture(scope="module")
def pairs():
    return B.load_pairs()


@pytest.fixture(scope="module")
def ident():
    return B.identified()


@pytest.fixture(scope="module")
def intact():
    return B.build_bundle(B.GATE_SIGN_RULE, "intact")


@pytest.fixture(scope="module")
def retina(intact):
    return B.Retina(intact.stim_ids)


@pytest.fixture(scope="module")
def gate():
    return B.gate_run(workers=WORKERS, write=True)


# ---------------------------------------------------------------------------------------------
# the network

def test_wiring_is_the_brain_of_record(pairs):
    assert pairs["n_synapses"] == 29_474_316
    assert pairs["type1_synapses"] == 15_883_209 and pairs["type2_synapses"] == 13_591_107
    assert pairs["n_segments"] == 13_458_709
    assert int(pairs["t1"].sum()) + int(pairs["t2"].sum()) == 29_474_316


def test_mauthner_counts_match_b1(pairs, ident):
    ids = pairs["ids"]
    nsyn = pairs["t1"].astype(np.int64) + pairs["t2"]
    want = {"left": (5849, 1294, 4555, 38), "right": (4173, 629, 3544, 25)}
    for side, (n_in, n1, n2, n_out) in want.items():
        i = int(B.index_of(ids, ident["mauthner"][side])[0])
        into = pairs["post"] == i
        assert int(nsyn[into].sum()) == n_in
        assert int(pairs["t1"][into].sum()) == n1 and int(pairs["t2"][into].sum()) == n2
        assert int(nsyn[pairs["pre"] == i].sum()) == n_out


def test_type2_is_excitatory_by_ground_truth(pairs):
    r = B.ei_polarity_check(pairs, write=False)
    exc = r["habenular_axons_classified_true_excitatory"]
    inh = r["inhibitory_cells_predicted_as_inhibitory"]
    assert exc["cells_majority_type2"] == exc["segments_with_outputs"] == 70
    assert inh["cells_majority_type1"] >= 0.85 * inh["segments_with_outputs"]
    assert B.SIGN_MAP == {1: -1, 2: +1}


def test_stimulated_set_is_tectal_and_two_hops_from_mauthner(intact, ident):
    stim = set(intact.stim_ids.tolist())
    assert stim <= set(ident["tectum"].tolist())
    assert not stim & C.never_stimulate()
    ro = {r["seg_id"] for r in B.readout_table(ident)}
    assert not stim & ro
    hops = intact.stats["mauthner_min_hops_from_stimulated"]
    assert hops["left"] >= 2 and hops["right"] >= 2, hops
    # no stimulated cell synapses directly onto any readout, in the kept network
    ro_idx = set(intact.ro_idx.tolist())
    rows = intact.net.rows()
    direct = np.isin(rows, intact.stim_idx) & np.isin(intact.net.indices, list(ro_idx))
    assert not direct.any()


def test_retina_is_tectum_drive(retina):
    """The matrix used per frame equals cells.tectum_drive on random multi-point frames."""
    rng = np.random.default_rng(7)
    for _ in range(3):
        vals, frame = {}, {}
        for e in ST.EYES:
            k = rng.choice(len(retina.pts[e]), 200, replace=False)
            v = np.zeros(len(retina.pts[e]))
            v[k] = rng.uniform(0, 1, len(k))
            vals[e] = v
            frame[e] = [(float(retina.az[retina.pts[e][j]]), float(retina.el[retina.pts[e][j]]), float(v[j])) for j in k]
        d_direct = C.tectum_drive(frame)[retina.rows]
        d_matrix = retina.drive_of_points(vals)
        assert np.max(np.abs(d_direct - d_matrix)) < 1e-9


def test_whole_field_dimming_never_reaches_the_tectum(retina):
    """Documents that the dimming control is decided by the eye model (balanced surround)."""
    rates, diag = B.cached_rates(retina, "dim")
    assert float(rates.max()) == 0.0, float(rates.max())


EXACT_W_SYN_MV = 20.0   # a weight at which activity reaches the readouts (none does at Shiu's 0.275)


def _trains(r, net, seg_ids):
    idx = net.index_of(seg_ids)
    order = np.argsort(r.neurons, kind="stable")
    n, s = r.neurons[order], r.steps[order]
    lo, hi = np.searchsorted(n, idx), np.searchsorted(n, idx, side="right")
    return {int(k): s[a:b] for k, a, b in zip(seg_ids.tolist(), lo, hi)}


def test_reduction_is_exact():
    """Every kept cell spikes identically on the reduced network and on the whole fireable set F.

    This tests reduce_graph(), not the gate model: at Shiu's weight nothing downstream of the
    tectum fires, so the check runs at EXACT_W_SYN_MV where readouts do fire (positive control).
    """
    b, fb = B.build_bundle(B.GATE_SIGN_RULE, "intact", keep_forward=True)
    ret = B.Retina(b.stim_ids)
    rates, _ = B.cached_rates(ret, "loom_right")
    seg = rates[-300:-200]                   # 0.5 s at the end of the expansion
    p = S.LIFParams(w_syn_mV=EXACT_W_SYN_MV)
    dur = seg.shape[0] * B.RETINA.frame_ms
    sim_r = S.Simulator(b.net, seed="exact-0", params=p)
    sim_r.add_poisson(b.stim_idx, seg, frame_ms=B.RETINA.frame_ms, name="retina")
    r_red = sim_r.run(dur)
    sim_f = S.Simulator(fb.net, seed="exact-0", params=p)
    sim_f.add_poisson(fb.idx(b.stim_ids), seg, frame_ms=B.RETINA.frame_ms, name="retina")
    rest = np.setdiff1d(fb.stim_ids, b.stim_ids)   # stimulated cells that cannot reach a readout
    sim_f.add_poisson(fb.idx(rest), 100.0, name="rest")
    r_full = sim_f.run(dur)
    kept_in_F = np.array(sorted(set(b.net.ids.tolist()) & set(fb.net.ids.tolist())), np.uint64)
    tr_red, tr_full = _trains(r_red, b.net, kept_in_F), _trains(r_full, fb.net, kept_in_F)
    for sid in kept_in_F.tolist():
        np.testing.assert_array_equal(tr_red[sid], tr_full[sid])
    not_in_F = np.array(sorted(set(b.net.ids.tolist()) - set(fb.net.ids.tolist())), np.uint64)
    assert all(len(v) == 0 for v in _trains(r_red, b.net, not_in_F).values())
    ro = sum(len(tr_red.get(int(s), [])) for s in b.ro_ids.tolist())
    assert ro > 0, "positive control: readouts must spike in this window"
    assert int(np.isin(r_full.neurons, fb.idx(rest)).sum()) > 0, "the extra stimulated cells did fire"
    assert len(r_full.neurons) > len(r_red.neurons), "F holds activity that the reduction dropped"


# ---------------------------------------------------------------------------------------------
# (a) looming and its controls

@pytest.mark.parametrize("side", ["left", "right"])
def test_a_loom_fires_the_predicted_mauthner_in_window(gate, side):
    ev = gate["evaluation"][f"a_loom_{side}"]
    assert ev["pass"], ev


@pytest.mark.parametrize("side", ["left", "right"])
def test_control_receding_fires_mauthner_less(gate, side):
    ev = gate["evaluation"][f"control_recede_{side}"]
    assert ev["pass"], ev


def test_control_dimming_fires_mauthner_less(gate):
    ev = gate["evaluation"]["control_dim"]
    assert ev["pass"], ev


# ---------------------------------------------------------------------------------------------
# (b) prey and its control

@pytest.mark.parametrize("side", ["left", "right"])
def test_b_prey_turns_toward_the_dot(gate, side):
    ev = gate["evaluation"][f"b_prey_{side}"]
    assert ev["pass"], ev


def test_control_no_stimulus_no_systematic_turn(gate):
    ev = gate["evaluation"]["control_blank"]
    # A silent network passes "no systematic turn" by construction, so the control only counts
    # when the turn readout is live: prey must have moved at least one turning cell.
    prey_live = any(sum(gate["evaluation"][f"b_prey_{s}"][k]) > 0
                    for s in ("left", "right") for k in ("mean_left_cell_spikes", "mean_right_cell_spikes"))
    assert prey_live and not ev.get("vacuous_no_spikes"), ("vacuous: the turning readout never fired, "
                                                         "so this control tests nothing", ev)
    assert ev["pass"], ev


# ---------------------------------------------------------------------------------------------
# (c) lesion, (d) shuffle, (e) determinism

def test_c_mauthner_lesion_removes_escape_spikes(gate):
    ev = gate["evaluation"]["c_lesion"]
    # Silence after a lesion means nothing unless the intact fish escaped: a lesioned cell cannot
    # fire by construction. Require intact Mauthner spikes to a loom first.
    intact = sum(gate["evaluation"][f"a_loom_{s}"]["mean_mauthner_spikes"] for s in ("left", "right"))
    assert intact > 0, ("vacuous: the intact Mauthner never fired to a loom, so the lesion removes "
                        "nothing", ev)
    assert ev["trials"] == 8 and ev["pass"], ev


def test_d_shuffled_wiring_changes_the_decision(gate):
    ev = gate["evaluation"]["d_shuffle"]
    assert ev["pass"], ev


def test_e_same_seed_same_spikes(intact, retina, gate):
    stim = B.make_stim("prey_right")
    r1, d1 = retina.rates(stim)
    r2, d2 = retina.rates(stim)
    assert d1["rates_sha256"] == d2["rates_sha256"]
    t1 = B.run_trial(intact, r1, "G1-seed-0")
    t2 = B.run_trial(intact, r2, "G1-seed-0")
    assert t1.spike_hash == t2.spike_hash and t1.n_spikes > 0
    t3 = B.run_trial(intact, r1, "G1-seed-0 ")
    assert t3.spike_hash != t1.spike_hash
    # the pool worker (another process) produced the same raster and the same rates
    g = gate["results"][f"{B.GATE_SIGN_RULE}/intact"]["prey_right"]["trials"][0]
    assert g["seed"] == "G1-seed-0" and g["spike_hash"] == t1.spike_hash
    assert g["rates_sha256"] == d1["rates_sha256"]
