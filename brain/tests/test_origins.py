"""Tests for fishbrain.origins (G1c measure): fragment origins, the retinal-afferent rule, the share bounds.

Two kinds of test:
- synthetic (no data): planted fragments whose class is known, including the positive control the task names
  (a fragment spanning tectum -> hindbrain must classify as an observed tectal origin), on a small synthetic atlas;
- data: data/g1c/*.json built by `python -m fishbrain.origins ...`. Without them the data tests FAIL, not skip
  (a skipped positive control would read green on a clone that never ran).
"""
import json
import math

import numpy as np
import pytest

from fishbrain import origins as O

UM = O.UM_PER_VOXEL


# =============================================================================================
# a small synthetic atlas: a "tectum" block at the anterior end, hindbrain behind it
# =============================================================================================

def _atlas():
    shape = (60, 20, 20)                     # 4 um voxels: 246 x 82 x 77 um
    m0 = np.full(shape, O.M0_HINDBRAIN, np.uint8)
    m1 = np.zeros(shape, np.uint8)
    m2 = np.zeros(shape, np.uint8)
    m2[0:15, :, 0:10] = 18                   # tectal neuropil: x < 61 um, dorsal half
    m2[0:15, :, 10:20] = 17                  # SPV
    m0[:, 0:2, :] = O.M0_GANGLIA             # a ganglion strip at the fish's right edge (low y)
    return O.Atlas(m0, m1, m2)


def _line(p0_um, p1_um, n):
    """n points (8 nm voxel coordinates) on the segment p0 -> p1 given in um."""
    t = np.linspace(0, 1, n)[:, None]
    return (np.asarray(p0_um) + t * (np.asarray(p1_um) - np.asarray(p0_um))) / UM


def _classify(at, cloud_vox, contact_vox, sides=None, **kw):
    L = at.lookup(cloud_vox)
    Lc = at.lookup(contact_vox)
    sides = np.ones(len(cloud_vox), np.int8) if sides is None else sides
    return O.classify_fragment(L["dT"], cloud_vox * UM, sides, L["m0"], contact_vox * UM, Lc["dT"], **kw)


def test_atlas_lookup_and_distance():
    at = _atlas()
    inside = np.array([[10 * 512, 5 * 512, 2 * 128]], float)            # voxel (10, 5, 2): neuropil
    assert at.lookup(inside)["dT"][0] == 0 and at.lookup(inside)["m2"][0] == 18
    three_back = np.array([[17 * 512 + 5, 5 * 512, 2 * 128]], float)     # voxel 17: 3 voxels behind the last T voxel (14)
    assert at.lookup(three_back)["dT"][0] == pytest.approx(3 * 4.096, rel=1e-5)
    assert at.vox(np.array([[511.9, 512.0, 127.9]]))[0][0] == 0 and at.vox(np.array([[511.9, 512.0, 127.9]]))[1][0] == 1
    far = np.array([[1e9, 5 * 512, 1e9]], float)                        # clipped, never out of bounds
    assert at.lookup(far)["m0"][0] == O.M0_HINDBRAIN and at.lookup(np.array([[5.0, -5.0, 5.0]]))["m0"][0] == O.M0_GANGLIA


def test_positive_control_fragment_spanning_tectum_to_hindbrain_is_observed_tectal():
    """The task's positive control: a synthetic fragment spanning tectum -> hindbrain must classify as (a)."""
    at = _atlas()
    cloud = _line([30, 40, 20], [220, 40, 60], 40)       # from inside the neuropil back to the hindbrain
    contact = cloud[-3:]
    c = _classify(at, cloud, contact, sides=np.ones(40, np.int8))
    assert c["cls"] == "O-a1" and c["lobe"] == 1 and c["n_T"] >= 2
    c = _classify(at, cloud, contact, sides=np.zeros(40, np.int8))
    assert c["cls"] == "O-a1" and c["lobe"] == 0               # the lobe tag follows the tectal synapses' side


def test_tract_only_local_far_and_ganglion():
    at = _atlas()
    # runs from the contact toward the tectum and stops 25 um short: tract-only (O-b), not O-a
    run = _line([85, 40, 40], [220, 40, 60], 30)
    c = _classify(at, run, run[-2:])
    assert c["cls"] == "O-b" and c["n_T"] == 0 and c["progress_um"] >= O.PROGRESS_UM and not c["edge"]
    # a local blob near the contact: O-c
    rng = np.random.default_rng(0)
    blob = (np.array([200, 40, 60]) + rng.normal(0, 3, (12, 3))) / UM
    assert _classify(at, blob, blob[:2])["cls"] == "O-c"
    # a fragment that reaches 40 um further posterior, away from the tectum: O-d3
    away = _line([200, 40, 60], [240, 40, 60], 20)
    assert _classify(at, away, away[:2])["cls"] == "O-d3"
    # two synapses in the ganglion strip: O-d2 (before O-b and O-d3)
    g = np.vstack([blob, np.array([[200, 3, 60], [201, 3, 60]]) / UM])
    assert _classify(at, g, blob[:2])["cls"] == "O-d2"


def test_single_tectal_synapse_is_not_enough_and_is_flagged_edge():
    at = _atlas()
    cloud = np.vstack([np.array([[40, 40, 20]]) / UM, _line([90, 40, 60], [100, 40, 60], 10)])
    c = _classify(at, cloud, cloud[-2:])
    assert c["n_T"] == 1 and c["cls"] != "O-a1" and c["edge"]


def test_priority_tectal_cell_then_tectal_end_then_soma():
    at = _atlas()
    blob = _line([200, 40, 60], [205, 40, 60], 5)
    c = _classify(at, blob, blob[:1], is_tectal=True, tectal_side=0)
    assert c["cls"] == "O-a0" and c["lobe"] == 0
    assert _classify(at, blob, blob[:1], soma=True)["cls"] == "O-d1"
    span = _line([30, 40, 20], [220, 40, 60], 40)
    assert _classify(at, span, span[-2:], soma=True)["cls"] == "O-a1"   # an observed tectal end beats a soma elsewhere


def test_compartment_order():
    dz = np.array([25.0, 25.0, 5.0, 5.0, 19.9])
    lat = np.array([60.0, 30.0, 60.0, 30.0, 49.9])
    assert list(O.compartment(dz, lat, 50.0)) == ["ventral", "ventral", "lateral", "soma", "soma"]


def test_afferent_rule():
    n_in = np.array([0, 0, 0, 0, 1, 1, 0, 0])
    n_out = np.array([5, 5, 5, 5, 20, 5, 0, 5])
    out_np = np.array([4, 3, 5, 5, 20, 5, 0, 5])
    soma = np.array([0, 0, 1, 0, 0, 0, 0, 0], bool)
    tect = np.array([0, 0, 0, 1, 0, 0, 0, 0], bool)
    assert list(O.afferent_rule(n_in, n_out, out_np, soma, tect)) == [1, 0, 0, 0, 0, 0, 0, 1]
    assert list(O.afferent_rule(n_in, n_out, out_np, soma, tect, near_zero=True)) == [1, 0, 0, 0, 1, 0, 0, 1]


def test_share_bounds():
    b = O.share_bounds(exc_M=2, exc_A=6, exc_B=12, x_M=1e-4, q_A=0.01, q_B=0.02, exc_all=100)
    assert b["LB_syn"] == pytest.approx(0.1) and b["UB_syn"] == pytest.approx(0.4)
    assert b["LB_walk"] == pytest.approx(1e-4 / 0.0301, rel=1e-6) and b["UB_walk"] == pytest.approx(0.0101 / 0.0301, rel=1e-6)
    assert b["all_inputs_M"] == pytest.approx(0.02) and b["all_inputs_M_plus_A"] == pytest.approx(0.08)
    z = O.share_bounds(0, 0, 0, 0, 0, 0, 10)
    assert z["LB_syn"] is None and z["UB_walk"] is None          # nothing visual reaches the cell: undefined, not 100%


# =============================================================================================
# data: data/g1c/*.json (built by `python -m fishbrain.origins classify|real|shuffles|oa|meshes|assemble|share`)
# =============================================================================================

def _j(name):
    p = O.OUT / name
    assert p.exists(), f"{p} missing: run python -m fishbrain.origins first"
    return json.load(open(p))


@pytest.fixture(scope="module")
def origins():
    return _j("origins.json")


def test_atlas_lookup_is_valid_on_real_cells(origins):
    a = origins["atlas_check"]
    assert a["pass"] and a["share_in_mece1_tectum"] >= 0.8 and a["mauthner_left"]["48_within_one_voxel"]


def test_pipeline_reproduces_task_R_direct_contacts_exactly(origins):
    pc = origins["pipeline_check"]
    assert pc["pass"]
    for pop, (i, c) in {"m_system": (5, 0), "mauthner": (4, 0), "strike": (3, 1), "turning": (2, 0)}.items():
        assert (pc[pop]["ipsi"], pc[pop]["contra"]) == (i, c)


def test_positive_and_negative_controls(origins):
    pos = origins["controls"]["positive_tectal_cells"]["summary"]["classes"]
    neg = origins["controls"]["negative_caudal_hindbrain"]["summary"]["classes"]
    assert pos["O-a0"]["share"] + pos["O-a1"]["share"] >= 0.5          # the rule fires inside the tectum
    assert neg["O-a0"]["fragments"] + neg["O-a1"]["fragments"] == 0     # and not 150+ um away from it


def test_class_parts_sum_to_independent_totals(origins):
    """Assert parts against the total: per readout side, the classes (and compartments) sum to the input synapses
    counted independently from the pair arrays."""
    from fishbrain import brain as B
    from fishbrain import structure as S
    P = B.load_pairs()
    n_all = np.bincount(P["post"], weights=P["t1"] + P["t2"], minlength=len(P["ids"]))
    n_exc = np.bincount(P["post"], weights=P["t2"], minlength=len(P["ids"]))
    pops = S.readout_populations()
    for pop in O.DECIDING:
        for side in O.SIDES:
            ix = B.index_of(P["ids"], np.array(pops[pop][side], np.uint64))
            ix = np.unique(ix[ix >= 0])
            s = origins["readouts"][pop][side]["summary"]
            assert s["input_synapses"] == int(n_all[ix].sum()) and s["input_exc_synapses"] == int(n_exc[ix].sum())
            assert sum(c["synapses"] for c in s["classes"].values()) == s["input_synapses"]
            assert sum(c["exc_synapses"] for c in s["classes"].values()) == s["input_exc_synapses"]
            assert sum(c["share"] for c in s["classes"].values()) == pytest.approx(1.0, abs=1e-5)
            if "compartments" in s:
                assert sum(v["synapses"] for v in s["compartments"].values()) == s["input_synapses"]
                for k, v in s["compartments"].items():
                    assert sum(c["synapses"] for c in v["classes"].values()) == v["synapses"]


def test_observed_tectal_inputs_land_on_the_ventral_dendrite(origins):
    """Post hoc observation, pinned: every O-a fragment on either Mauthner contacts only the ventral dendrite."""
    for side in O.SIDES:
        fr = [f for f in origins["readouts"]["mauthner"][side]["fragments"] if f["cls"] in ("O-a0", "O-a1")]
        assert len(fr) == 8 and all(f["comp"]["ventral"][0] == f["syn"] for f in fr)


def test_the_eye_holds_no_synapse(origins):
    e = origins["eye"]
    assert e["retina_voxels_4096"] == 38184 and e["synapses_in_retina"] == 0 and e["cave_somas_in_retina"] == 0
    assert e["synapses_total"] == 29474316


def test_afferent_set_satisfies_its_rule_on_data():
    import pandas as pd
    from fishbrain import brain as B
    from fishbrain import cells as C
    a = np.load(O.OUT / "afferents_primary.npz")
    acc = np.load(O.OUT / "scan_acc.npz")
    idx = a["idx"]
    assert len(idx) == 1284858
    assert (acc["in_all"][idx] == 0).all() and (acc["out_all"][idx] >= 1).all()
    assert (acc["out_np"][idx] >= O.AFF_NEUROPIL_FRAC * acc["out_all"][idx]).all()
    ag = pd.read_csv(O.REF / "agglomerated_segments_and_soma_ids.csv")
    somas = set(int(s) for s in ag.hires_id_agglo.values if s != 0)
    segs = set(int(s) for s in a["seg_id"])
    assert not segs & somas and not segs & set(int(s) for s in C.tectum_seg_ids(C.load()))
    # a deliberately wrong set must fail the same check: tectal cells' own outputs are not >= 80% in the neuropil
    P = B.load_pairs()
    t = B.index_of(P["ids"], C.tectum_seg_ids(C.load()))
    t = t[t >= 0]
    assert not (acc["in_all"][t] == 0).all()


def test_reach_from_afferents_shuffles_and_verdict():
    from fishbrain import structure as S
    r = _j("reach.json")
    assert len(r["shuffles"]["A"]) == 20
    for s in r["shuffles"]["A"]:
        assert s["repair"]["unrepaired_pairs"] == 0 and s["checks"]["in_degree_preserved"] and s["checks"]["out_degree_preserved"]
    rf = r["real"]["flat"]
    assert json.loads(json.dumps(S.verdict(rf, r["nullA"]))) == r["verdict"]
    v = r["verdict"]["positive_controls"]
    assert v["PC1"]["pass"] and v["PC1"]["value"] >= 0.5
    for p, sides in r["real"]["detail"]["reach"].items():
        for side, row in sides.items():
            vals = [row["any_le1"], row["any_le2"], row["any_le3"]]
            assert all(0 <= x <= 1 for x in vals) and vals == sorted(vals)
            assert row["any_le1"] == 0.0                     # no afferent synapses onto any readout


def test_share_rows_are_consistent():
    s = _j("share.json")
    for pop in O.DECIDING:
        for side in O.SIDES:
            for lobe in O.SIDES:
                r = s[pop][side][f"from_{lobe}_lobe"]
                assert r["exc_M"] + r["exc_A"] + r["exc_B"] <= r["exc_all"]
                b = O.share_bounds(r["exc_M"], r["exc_A"], r["exc_B"], r["x_M_walk"], r["q_A"], r["q_B"], r["exc_all"])
                for k in ("LB_syn", "UB_syn", "LB_walk", "UB_walk"):
                    assert b[k] == r[k]
                    assert r[k] is None or 0.0 <= r[k] <= 1.0
                if r["LB_syn"] is not None:
                    assert r["LB_syn"] <= r["UB_syn"] and r["LB_walk"] <= r["UB_walk"]
                assert r["relation"] == ("ipsi" if lobe == side else "contra")
        assert s[pop]["pairing"]["primary"] == O.PAIRING[pop]


def test_mesh_sensitivity_row():
    m = _j("meshes.json")
    assert m["picks"] == 2 * O.MESH_TOP_PER_SIDE and m["errors"] == 0


def test_pinned_numbers(origins):
    """Key measured numbers, pinned so a rebuild that changes them fails loudly."""
    ms = {s: origins["readouts"]["m_system"][s]["summary"]["classes"] for s in O.SIDES}
    assert (ms["left"]["O-a1"]["synapses"], ms["right"]["O-a1"]["synapses"]) == (47, 50)
    assert (ms["left"]["O-a1"]["contra"]["synapses"], ms["right"]["O-a1"]["contra"]["synapses"]) == (1, 0)
    assert origins["spanning"]["segments"] == 91 and origins["spanning"]["touching_a_deciding_readout"] == 27
    s = _j("share.json")
    assert s["mauthner"]["right"]["from_left_lobe"]["exc_M"] == 0.0     # the escaping cell for a right-eye loom
    assert s["m_system"]["right"]["from_left_lobe"]["UB_syn"] == pytest.approx(0.00615, abs=1e-6)
    assert s["m_system"]["left"]["from_right_lobe"]["UB_syn"] == pytest.approx(0.002246, abs=1e-6)
