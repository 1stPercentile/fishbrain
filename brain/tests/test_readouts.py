"""Tests for fishbrain.readouts (task R): atlas populations, the M-system, and the decision rules.

Offline: reads data/readouts.json (built by `python -m fishbrain.readouts`), data/cells_identified.json,
B1's cached atlas box data/ref/mece2_1024_hindbrain.npz and the lore-id -> segment table. Nothing here
reads the network or the build cache. Without readouts.json every data test FAILS (a skipped
positive control would read green on a clone that never built the readouts).
"""
import numpy as np
import pytest

from fishbrain import cells as C
from fishbrain import readouts as R

SIDES = ("left", "right")


@pytest.fixture(scope="module")
def ro():
    if not R.READOUTS_JSON.exists():
        pytest.fail("data/readouts.json missing: run `python -m fishbrain.readouts` first")
    return R.load()


@pytest.fixture(scope="module")
def cells():
    return C.load()


@pytest.fixture(scope="module")
def b1_box():
    d = np.load(R.REF / "mece2_1024_hindbrain.npz")
    return d["M"], d["b0"].astype(int), R.F1024


def _strict_names(points, box):
    M, b0, f = box
    q = (np.asarray(points, float) // f - b0).astype(int)
    # the 3 x 3 x 3 vote needs its whole neighbourhood inside B1's box to be comparable
    inside = ((q >= 1) & (q < np.array(M.shape) - 1)).all(1)
    lab = np.zeros(len(points), int)
    if inside.any():
        lab[inside] = R.vote_labels(np.asarray(points, float)[inside], M, b0, f)
    return [R.LABEL_NAME.get(int(l), "unlabelled" if l == 0 else f"other:{l}") for l in lab], inside


def _members(ro, name, side, pset="strict"):
    grp = next(g for g, r in R.POPULATIONS.items() if name in r)
    return {m["seg_id"] for m in ro["populations"][pset][grp][name][side]}


# ---------------------------------------------------------------- positive control: named SPNs

def test_named_spns_map_into_their_own_populations(ro, cells, b1_box):
    """Positive control. Labels recomputed here from B1's cached box (independent of the build's
    fetched box): every named SPN whose soma sits in its own-named atlas region must be a member of
    that region's population on its own side, and B1's 11 of 12 must reproduce."""
    spn = cells["spn"]
    names, inside = _strict_names([s["pos"] for s in spn], b1_box)
    assert inside.all(), "every named SPN lies inside B1's hindbrain box"
    in_rs = [n in R.RS_LABELS for n in names]
    own = [n == s["name"] for n, s in zip(names, spn)]
    assert sum(in_rs) == 12 and sum(own) == 11, (sum(in_rs), sum(own))
    hits = 0
    for s, o in zip(spn, own):
        if o:
            assert s["seg_id"] in _members(ro, s["name"], s["side"]), (s["lore_id"], s["name"], s["side"])
            hits += 1
    assert hits == 11
    v = ro["spn_validation"]["strict"]
    assert v["strict"]["own_named_region"] == 11 and v["strict"]["in_any_rs_region"] == 12
    assert v["named_cell_in_own_population"]["all"] == 11


def test_positive_control_fails_on_swapped_sides(ro, cells, b1_box):
    """The check above must be able to fail: with left and right populations swapped, none of the
    11 own-region named cells (all on the right) is found."""
    spn = cells["spn"]
    names, _ = _strict_names([s["pos"] for s in spn], b1_box)
    found = sum(s["seg_id"] in _members(ro, s["name"], R.other(s["side"]))
                for n, s in zip(names, spn) if n == s["name"])
    assert found == 0


def test_own_name_matches_beat_the_name_permutation_null(cells, b1_box):
    spn = cells["spn"]
    names, _ = _strict_names([s["pos"] for s in spn], b1_box)
    lab = np.array(names)
    true = np.array([s["name"] for s in spn])
    obs = int((lab == true).sum())
    rng = R._rng("test-permutation")
    null = np.array([(lab == true[rng.permutation(len(true))]).sum() for _ in range(2000)])
    assert obs == 11 and (null >= obs).mean() < 0.01 and null.mean() < 3


def test_mirror_test_recorded_and_left_populations_exist(ro):
    m = ro["spn_validation"]["strict"]["mirror"]
    assert m["lands_on_other_side"] == 50
    assert m["strict_own_named_region"] >= 8, "reflected right SPNs should mostly land in the same-named LEFT region"
    for grp, regions in R.POPULATIONS.items():
        for name in regions:
            for side in SIDES:
                assert len(ro["populations"]["strict"][grp][name][side]) >= 1, (name, side)


# ---------------------------------------------------------------- membership consistency

def test_every_member_is_on_its_side_soma_bearing_and_in_its_region(ro, cells, b1_box):
    import pandas as pd
    agg = pd.read_csv(R.REF / "agglomerated_segments_and_soma_ids.csv").set_index("lores_id")["hires_id_agglo"]
    axes = cells["axes"]
    checked = 0
    for grp, regions in ro["populations"]["strict"].items():
        for name, sides in regions.items():
            for side, mem in sides.items():
                if not mem:
                    continue
                pos = np.array([m["pos"] for m in mem], float)
                assert (C.side_of(pos, axes) == side).all(), (name, side)
                assert all(m["lat_um"] >= R.SIDE_AMBIGUOUS_UM for m in mem)
                for m in mem:
                    assert m["seg_id"] > 0
                    assert all(int(agg[l]) == m["seg_id"] for l in m["lore_ids"])
                # region label recomputed from B1's box for single-soma members inside it
                single = [m for m in mem if len(m["lore_ids"]) == 1]
                labs, inside = _strict_names([m["pos"] for m in single], b1_box)
                for m, l, ok in zip(single, labs, inside):
                    if ok:
                        assert l == name, (name, m["seg_id"], l)
                        checked += 1
    assert checked > 200


def test_dilated_populations_contain_the_strict_ones(ro):
    for grp, regions in ro["populations"]["strict"].items():
        for name, sides in regions.items():
            for side, mem in sides.items():
                assert {m["seg_id"] for m in mem} <= _members(ro, name, side, "dilated_5um"), (name, side)


def test_readout_sets_are_disjoint_and_nIII_dorsal_is_the_dorsal_half(ro):
    rs = ro["readout_sets"]["strict"]
    keys = [k for k in rs if not k.startswith("nIII_") or k.startswith("nIII_dorsal")]
    seen = {}
    for k in keys:
        for s in rs[k]:
            assert s not in seen, (s, k, seen.get(s))
            seen[s] = k
    for side in SIDES:
        mem = ro["populations"]["strict"]["strike"]["nIII"][side]
        z_d = [m["pos"][2] for m in mem if m["nIII_dorsal"]]
        z_v = [m["pos"][2] for m in mem if not m["nIII_dorsal"]]
        assert max(z_d) <= min(z_v), "dorsal = low z in Fish1"
        assert abs(len(z_d) - len(z_v)) <= 1


# ---------------------------------------------------------------- the M-system

def test_m_system(ro, cells):
    m = ro["m_system"]
    for side in SIDES:
        assert m["Mauthner"][side]["seg_id"] == int(cells["mauthner"][side]["seg_id"])
    ctl = m["identification"]["positive_control_r4"]
    assert m["identification"]["positive_control_passed"]
    assert ctl["left"]["seg_id"] == 15773771512 and ctl["right"]["seg_id"] == 16304202596
    assert (ctl["left"]["soma_volume_um3"], ctl["right"]["soma_volume_um3"]) == (271.1, 247.6)
    axes = cells["axes"]
    picked = set()
    for h, rhomb in (("MiD2cm", "r5"), ("MiD3cm", "r6")):
        cands = m["identification"]["candidates"][h]
        for side in SIDES:
            seg = m[h][side]["task_criteria"]
            c = next(c for c in cands[side]["candidates"] if c["seg_id"] == seg)
            assert c["soma_volume_um3"] == max(x["soma_volume_um3"] for x in cands[side]["candidates"])
            assert m[h][side]["task_criteria_input_rank"] == 1, "largest soma also has the most inputs"
            assert C.side_of(np.array([c["pos"]], float), axes)[0] == side
            assert c["frac_in_region"] > 0.5
            picked.add(seg)
    assert len(picked) == 4 and not picked & {15773771512, 16304202596}
    ls = ro["lesion_sets"]
    assert len(ls["task_criteria"]) == 6 and {15773771512, 16304202596} <= set(ls["task_criteria"])
    assert set(ls["union"]) == set(ls["task_criteria"]) | set(ls["preregistered_crossing"])


# ---------------------------------------------------------------- decision rules (synthetic)

def _strike_in(nl, nr, el, er):
    return {"nMLF_left": nl, "nMLF_right": nr, "nIII_dorsal_left": el, "nIII_dorsal_right": er}


ACTIVE = [2, 1, 0, 3, 1]      # 80% active
QUIET = [0, 0, 0, 0, 0]


def test_strike_right_and_its_mirror():
    d = R.strike_decision(_strike_in([0, 1, 0, 0, 0], [3, 2, 4, 1, 0], ACTIVE, ACTIVE))
    assert d["strike"] and d["side"] == "right" and d["trade"] == "buy"
    m = R.strike_decision(_strike_in([3, 2, 4, 1, 0], [0, 1, 0, 0, 0], ACTIVE, ACTIVE))
    assert m["strike"] and m["side"] == "left"
    assert d["side_index"] == pytest.approx(-m["side_index"])


def test_no_strike_without_convergence_or_drive():
    strong = [5, 5, 5, 5, 5]
    assert not R.strike_decision(_strike_in(QUIET, strong, ACTIVE, QUIET))["strike"], "one eye only is not convergence"
    assert not R.strike_decision(_strike_in(QUIET, QUIET, ACTIVE, ACTIVE))["strike"], "convergence without nMLF drive"
    assert not R.strike_decision(_strike_in(QUIET, QUIET, QUIET, QUIET))["strike"]


def test_symmetric_nmlf_strikes_ahead_and_population_size_does_not_bias_side():
    d = R.strike_decision(_strike_in([1, 1, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0], ACTIVE, ACTIVE))
    assert d["strike"] and d["side"] == "ahead" and d["side_index"] == 0.0


def test_escape_goes_away_from_the_first_m_series_cell():
    silent = {f"{c}_{s}": None for c in R.ESCAPE_RULE["cells"] for s in SIDES}
    zero = {k: 0 for k in silent}
    assert not R.escape_decision(silent, zero)["escape"]
    e = R.escape_decision({**silent, "Mauthner_right": 12.0, "Mauthner_left": 30.0},
                          {**zero, "Mauthner_right": 1, "Mauthner_left": 1})
    assert e["escape"] and e["cell_side"] == "right" and e["escape_to"] == "left" and e["trade"] == "sell"
    h = R.escape_decision({**silent, "MiD2cm_left": 8.0}, {**zero, "MiD2cm_left": 2})
    assert h["initiator"] == "MiD2cm_left" and h["escape_to"] == "right"
    t = R.escape_decision({**silent, "Mauthner_left": 5.0, "MiD3cm_right": 5.0},
                          {**zero, "Mauthner_left": 1, "MiD3cm_right": 1})
    assert t["escape"] and t["cell_side"] is None


def test_turn_index_sign():
    assert R.turn_index({"turning_left": [0, 0, 1], "turning_right": [2, 2, 0, 1]})["turn"] == "right"
    assert R.turn_index({"turning_left": [3, 1], "turning_right": [0, 0]})["turn"] == "left"
    assert R.turn_index({"turning_left": [0], "turning_right": [0]})["turn"] is None


# ---------------------------------------------------------------- wiring checks recorded with their null

def test_wiring_checks_recorded_with_null(ro):
    w = ro["wiring"]
    assert w is not None and w["shuffles"] >= 20
    for base in ("nMLF", "nIII_dorsal", "turning", "Mauthner", "MiD2cm", "MiD3cm"):
        r = w["routing"][base]
        assert "null_mean" in r["fireable_ipsi_index_(ipsi-contra)/(ipsi+contra)"]
        assert "null_mean" in r["direct_ipsi_synapses"]
