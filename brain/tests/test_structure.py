"""Tests for fishbrain.structure (task S): does the Fish1 wiring carry side and stimulus information?

Two kinds of test:
- synthetic (no data): the side / selectivity index, the walk, and the shuffle, on planted graphs whose
  answer is known (positive controls that must succeed, and a no-routing case that must read 0);
- data: data/structure/structure.json (built by `python -m fishbrain.structure run`) plus an
  independent recomputation of hop-1 and hop-2 weights from the pair arrays. Without structure.json the
  data tests FAIL, not skip (a skipped positive control would read green on a clone that never ran).
"""
import json
import math

import numpy as np
import pytest

from fishbrain import structure as T

SIDES = ("left", "right")


# =============================================================================================
# synthetic: the index
# =============================================================================================

@pytest.mark.parametrize("fL,fR,gL,gR", [(1.0, 1.11, 2.0, 1.0), (11799, 13121, 543, 265), (0.3, 5.0, 1e-6, 3e-6)])
def test_side_index_is_zero_without_routing(fL, fR, gL, gR):
    """w = f(source) x g(target): lobe size and a target's own bias cancel exactly."""
    assert abs(T.side_index(fL * gL, fL * gR, fR * gL, fR * gR)) < 1e-12


def test_side_index_extremes_signs_and_undefined():
    assert T.side_index(1, 0, 0, 1) == 1.0          # each source to its own target
    assert T.side_index(0, 1, 1, 0) == -1.0         # crossed
    assert T.side_index(0, 0, 1, 1) is None         # source 1 reaches neither target
    assert T.side_index(-1, 0, 0, -1) == -1.0       # signed: inhibition onto the own side reads as crossed
    assert T.side_index(3, 1, 1, 3) == pytest.approx(0.5)


def _planted(crossed: bool, n_src: int = 60, n_rel: int = 8, n_tgt: int = 6, noise: int = 0, seed: int = 0):
    """Left sources -> left relays -> targets (right if crossed, else left); mirror for the right.
    Returns (Wiring, pre, post, t1, t2, n, groups) with groups = {left_src, right_src, left_tgt, right_tgt}.
    `noise` random extra pairs among all nodes (no self-loops, no duplicates)."""
    rng = np.random.default_rng(seed)
    Ls, Rs = np.arange(n_src), np.arange(n_src, 2 * n_src)
    Lr, Rr = 2 * n_src + np.arange(n_rel), 2 * n_src + n_rel + np.arange(n_rel)
    base = 2 * n_src + 2 * n_rel
    Lt, Rt = base + np.arange(n_tgt), base + n_tgt + np.arange(n_tgt)
    n = base + 2 * n_tgt + 400
    pairs = {}
    for src, rel, tgt in ((Ls, Lr, Rt if crossed else Lt), (Rs, Rr, Lt if crossed else Rt)):
        for s in src:
            for r in rng.choice(rel, 3, replace=False):
                pairs[(int(s), int(r))] = (0, int(rng.integers(1, 4)))
        for r in rel:
            for t in tgt:
                pairs[(int(r), int(t))] = (0, 2)
    k = 0
    while k < noise:
        a, b = (int(x) for x in rng.integers(0, n, 2))
        if a != b and (a, b) not in pairs:
            pairs[(a, b)] = (int(rng.integers(0, 3)), int(rng.integers(1, 3)))
            k += 1
    key = sorted(pairs)
    pre = np.array([p for p, _ in key], np.int32)
    post = np.array([q for _, q in key], np.int32)
    t1 = np.array([pairs[x][0] for x in key], np.int32)
    t2 = np.array([pairs[x][1] for x in key], np.int32)
    return T.Wiring(n, pre, post, t1, t2), pre, post, t1, t2, n, {"Ls": Ls, "Rs": Rs, "Lt": Lt, "Rt": Rt}


def _si(w, g, kind="exc", hop="cum"):
    def lobe(src):
        s = np.zeros(w.n)
        s[src] = 1.0
        return T._hopdict(w.walk(s, kind))[hop]
    L, R = lobe(g["Ls"]), lobe(g["Rs"])
    return T.side_index(L[g["Lt"]].mean(), L[g["Rt"]].mean(), R[g["Lt"]].mean(), R[g["Rt"]].mean())


def test_planted_crossed_route_reads_minus_one_and_ipsilateral_plus_one():
    """PC3, positive control: a planted route must be read exactly, through the module's own walk."""
    w, *_, g = _planted(crossed=True)
    assert _si(w, g, hop="cum") == -1.0
    assert _si(w, g, hop="hop2") == -1.0
    assert _si(w, g, hop="hop1") is None             # no direct contact: undefined, not 0
    w, *_, g = _planted(crossed=False)
    assert _si(w, g, hop="cum") == 1.0


def test_walk_is_input_fraction_products():
    """x_2 at a target = sum over relays of (share of the relay's input from the source) x (share of the
    target's input from the relay)."""
    w, pre, post, t1, t2, n, g = _planted(crossed=True, noise=500, seed=3)
    s = np.zeros(n)
    s[g["Ls"]] = 1.0
    x2 = w.walk(s, "exc")[1]
    nin = np.bincount(post, weights=(t1 + t2).astype(float), minlength=n)
    src = np.zeros(n, bool)
    src[g["Ls"]] = True
    x1 = np.bincount(post, weights=np.where(src[pre], t2, 0) / nin[post], minlength=n)
    ref = np.bincount(post, weights=x1[pre] * t2 / nin[post], minlength=n)
    assert np.allclose(x2, ref, rtol=1e-12, atol=0)


def test_planted_route_beats_its_shuffles_which_scatter_around_zero():
    """The null must be able to say no: shuffles of a crossed graph scatter around 0, the real one is
    beyond all of them."""
    w, pre, post, t1, t2, n, g = _planted(crossed=True, noise=3000, seed=1)
    real = _si(w, g)
    assert real < -0.5          # measured -0.654: the noise pairs dilute the planted route
    null = []
    for k in range(20):
        p2, rep = T.shuffle_post(pre, post, n, f"test-shuffle-{k}")
        assert rep["unrepaired_pairs"] == 0
        v = _si(T.Wiring(n, pre, p2, t1, t2), g)
        if v is not None:
            null.append(v)
    s = T.null_summary(real, null)
    assert len(null) >= 15
    assert abs(s["mean"]) < 0.3
    assert s["beyond_all"] and s["z"] < -3


def test_selectivity_index_reads_a_planted_type_route():
    """SEL is side_index with (loom, prey) as sources and (escape, strike) as targets."""
    w, *_, g = _planted(crossed=False)
    # loom = left sources -> left targets ("escape"); prey = right sources -> right targets ("strike")
    assert _si(w, g) == 1.0
    # the same numbers through the selectivity convention
    def lobe(src):
        s = np.zeros(w.n)
        s[src] = 1.0
        return T._hopdict(w.walk(s, "exc"))["cum"]
    L, R = lobe(g["Ls"]), lobe(g["Rs"])
    assert T.side_index(L[g["Lt"]].mean(), L[g["Rt"]].mean(), R[g["Lt"]].mean(), R[g["Rt"]].mean()) == 1.0


# =============================================================================================
# synthetic: the shuffle
# =============================================================================================

def _random_pairs(n=3000, m=40000, seed=2):
    rng = np.random.default_rng(seed)
    p, q = rng.integers(0, n, m), rng.integers(0, n, m)
    _, u = np.unique(p * n + q, return_index=True)
    p, q = p[u], q[u]
    keep = p != q
    p, q = p[keep], q[keep]
    o = np.argsort(p, kind="stable")
    return p[o], q[o], n


def test_shuffle_preserves_degrees_counts_and_has_no_duplicates():
    p, q, n = _random_pairs()
    q2, rep = T.shuffle_post(p, q, n, "G1b-structure-test")
    assert rep["unrepaired_pairs"] == 0
    assert np.array_equal(np.bincount(q, minlength=n), np.bincount(q2, minlength=n))   # in-degree
    assert len(np.unique(p.astype(np.int64) * n + q2)) == len(p)                        # no duplicates
    assert not np.any(p == q2)                                                           # no self-loops
    assert np.mean(q2 != q) > 0.99                                                       # it did shuffle
    q3, _ = T.shuffle_post(p, q, n, "G1b-structure-test")
    assert np.array_equal(q2, q3)                                                        # deterministic
    q4, _ = T.shuffle_post(p, q, n, "G1b-structure-test-2")
    assert not np.array_equal(q2, q4)


def test_side_class_shuffle_keeps_each_pairs_class():
    p, q, n = _random_pairs(seed=5)
    side = (np.arange(n) % 3 == 0).astype(np.int64)      # an uneven two-way split
    grp = side[p] * 2 + side[q]
    q2, rep = T.shuffle_post(p, q, n, "G1b-structure-test-B", groups=grp)
    assert rep["unrepaired_pairs"] == 0
    assert np.array_equal(side[p] * 2 + side[q2], grp)
    assert np.array_equal(np.bincount(q, minlength=n), np.bincount(q2, minlength=n))
    assert len(np.unique(p.astype(np.int64) * n + q2)) == len(p)


# =============================================================================================
# data
# =============================================================================================

@pytest.fixture(scope="module")
def st():
    if not T.STRUCTURE_JSON.exists():
        pytest.fail("data/structure/structure.json missing: run `python -m fishbrain.structure run` first")
    return json.load(open(T.STRUCTURE_JSON))


@pytest.fixture(scope="module")
def real_graph():
    """The brain of record through the module (for the independent recomputation)."""
    from fishbrain import brain as B
    P = B.load_pairs()
    return P


def test_pipeline_direct_contacts_reproduce_task_R(st):
    """Hop-1 synapses from each lobe onto each readout equal readouts.json (task R, other code)."""
    ro = json.load(open(T.READOUTS_JSON))["wiring"]["routing"]
    names = {"mauthner": "Mauthner", "strike": "nMLF", "turning": "turning", "nIII_dorsal": "nIII_dorsal",
             "forward": "forward"}
    d = st["real"]["detail"]["direct"]
    for mine, theirs in names.items():
        assert d[mine]["ipsi"] == ro[theirs]["direct_ipsi_synapses"]["observed"], mine
        assert d[mine]["contra"] == ro[theirs]["direct_contra_synapses"]["observed"], mine
    assert (d["nIII_dorsal"]["ipsi"], d["nIII_dorsal"]["contra"]) == (16, 3)
    assert (d["forward"]["ipsi"], d["forward"]["contra"]) == (18, 0)


def test_positive_controls_pass_and_the_verdict_is_valid(st):
    v = st["verdict"]
    pc = v["positive_controls"]
    assert pc["PC1"]["pass"] and pc["PC1"]["value"] >= T.PC1_MIN
    assert pc["PC1"]["value"] > pc["PC1"]["null_max_abs"]
    assert pc["PC2"]["pass"] and pc["PC2"]["value"] >= T.PC2_MIN and pc["PC2"]["z"] >= T.Z_MIN
    assert v["call"] != "INVALID"


def test_enough_shuffles_each_preserving_degrees(st):
    A, Bn = st["shuffles"]["A"], st["shuffles"]["B"]
    assert len(A) >= T.MIN_SHUFFLES
    assert len(A) == T.N_SHUFFLES and len(Bn) == T.N_SIDE_SHUFFLES
    assert len({s["seed"] for s in A}) == len(A)
    for s in A + Bn:
        assert s["repair"]["unrepaired_pairs"] == 0
        assert s["checks"]["in_degree_preserved"] and s["checks"]["out_degree_preserved"]
    for s in Bn:
        assert s["checks"]["side_class_preserved"]


def test_verdict_recomputes_from_the_recorded_numbers(st):
    v = T.verdict(st["real"]["flat"], st["nullA"])
    assert v["call"] == st["verdict"]["call"]
    assert v["n_side_separations"] == st["verdict"]["n_side_separations"]
    assert v["n_stimulus_separations"] == st["verdict"]["n_stimulus_separations"]
    for key, s in st["nullA"].items():
        if s["real"] is not None and s["z"] is not None and s["sd"]:
            assert s["z"] == pytest.approx((s["real"] - s["mean"]) / s["sd"])


def test_reachability_is_bounded_and_monotone(st):
    for p, sides in st["real"]["detail"]["reach"].items():
        for side, r in sides.items():
            for tag in ("any", "exc"):
                a, b, c = r[f"{tag}_le1"], r[f"{tag}_le2"], r[f"{tag}_le3"]
                assert 0 <= a <= b <= c <= 1, (p, side, tag)
            assert r["orphan_share"] + r["any_le3"] <= 1 + 1e-9
            assert r["orphan_share"] > 0.5, "the orphan ceiling: most input comes from segments with no input"


def test_independent_recompute_of_hop1_and_hop2(st, real_graph):
    """Hop-1 and hop-2 excitatory weights onto the M-system, recomputed with bincounts straight from the
    pair arrays (not the sparse matrices), must equal the stored values; so must PC1 at hop 1."""
    from fishbrain import brain as B
    from fishbrain import cells as C
    P = real_graph
    ids, pre, post = P["ids"], P["pre"].astype(np.int64), P["post"].astype(np.int64)
    t1, t2 = P["t1"].astype(np.int64), P["t2"].astype(np.int64)
    n = len(ids)
    nin = np.bincount(post, weights=(t1 + t2).astype(float), minlength=n)
    cells = C.load()
    tix = B.index_of(ids, C.tectum_seg_ids(cells))
    side = C.tectum_map(cells).side
    pops = T.readout_populations()
    lw = st["real"]["detail"]["lobe_weights"]["exc"]
    for lobe in SIDES:
        src = np.zeros(n, bool)
        src[tix[(tix >= 0) & (side == lobe)]] = True
        x1 = np.bincount(post, weights=np.where(src[pre], t2, 0) / nin[post], minlength=n)
        x2 = np.bincount(post, weights=x1[pre] * t2 / nin[post], minlength=n)
        for tgt in SIDES:
            m = B.index_of(ids, np.array(pops["m_system"][tgt], np.uint64))
            assert x1[m].mean() == pytest.approx(lw["hop1"]["m_system"][lobe][tgt], rel=1e-5, abs=1e-12)
            assert x2[m].mean() == pytest.approx(lw["hop2"]["m_system"][lobe][tgt], rel=1e-5, abs=1e-12)
    # Mauthner-left gets 2 excitatory synapses from the left lobe out of 5,849 inputs (task R, G1-gate)
    assert lw["hop1"]["mauthner"]["left"]["left"] == pytest.approx(2 / 5849, rel=1e-5)


def test_pinned_key_numbers(st):
    """The numbers the note quotes. A rebuild that moves them must be explained, not re-pinned."""
    f = st["real"]["flat"]
    pins = PINS
    for key, val in pins.items():
        assert f[key] == pytest.approx(val, abs=5e-4), key
    assert st["verdict"]["call"] == PINNED_CALL


def test_tectal_output_is_local_and_recorded(st):
    """The tectum's own output synapses: none reach the other lobe, and the non-tectal targets sit on the
    lobe's own side at tectal x positions (neuropil), far anterior of the Mauthner cells (x ~70,400)."""
    t = st["tectal_output"]
    assert t["output_synapses"] == 95169 and t["tectal_cells_with_outputs"] == 8893
    assert t["share_onto_other_lobe_tectal"] == 0.0
    assert t["share_onto_same_lobe_tectal"] + t["share_onto_non_tectal"] == pytest.approx(1.0)
    assert t["non_tectal_share_on_the_lobes_own_side"] > 0.99
    assert t["non_tectal_target_x_percentiles_5_25_50_75_95"][4] < 65000


def test_shuffle_replays_bit_for_bit(st):
    """Shuffle A-00 rebuilt from its seed in a fresh process gave every flat value identically."""
    r = st["replay_check"]
    assert r is not None and r["identical"] and r["keys"] > 500 and not r["differing_keys"]


def test_null_B_never_claims_a_separation(st):
    """Null B is centred near SI = +1 (laterality alone), so no 'separates' / beyond_all flag is reported."""
    for group in ("side", "stimulus"):
        for row in st["nullB_sensitivity"][group].values():
            assert "separates" not in row and "beyond_all" not in row
    for row in st["nullB"].values():
        assert "beyond_all" not in row and "separates" not in row
    assert st["nullB"]["SI|exc|cum|m_system"]["mean"] > 0.9       # laterality alone predicts the real 0.976


def test_power_limit_is_recorded_honestly(st):
    """The pre-registered |z| >= 3 bar cannot be reached in the predicted (contralateral / loom->escape)
    direction for any verdict item, because the Null-A spread is that wide. Pinned so a rebuild that
    changes this is noticed; the note says it in words."""
    A = st["nullA"]
    for key, sign in (("SI|exc|cum|m_system", -1), ("SI|exc|cum|turning", -1),
                      ("SEL|exc|cum|E_vs_strike|mean", +1), ("SEL|exc|cum|E_vs_turning|mean", +1)):
        s = A[key]
        assert abs((sign - s["mean"]) / s["sd"]) < T.Z_MIN, key


# filled from the run of record (2026-09-26 02:58 EDT, 20 + 10 shuffles); see G1b-structure.md
PINS = {
    "SI|exc|cum|m_system": 0.975946, "SI|exc|cum|turning": 0.409624, "SI|exc|cum|strike": 0.335074,
    "SEL|exc|cum|E_vs_strike|mean": -0.247371, "SEL|exc|cum|E_vs_turning|mean": -0.532884,
    "PC1|exc|hop1": 1.0, "PC2|exc|hop1|mean": 0.96779,
    "SI|exc|h23|m_system": 0.838956, "SI|exc|h23|turning": -0.020956, "SI|exc|h23|strike": -0.17316,
    "reach|any_le3|m_system|both": 0.025898, "reach|exc_le3|m_system|both": 0.017921,
    "reach|orphan_share|m_system|both": 0.702945, "Jglobal|hop1": 0.000258, "Jglobal|hop2": 0.017305,
}
PINNED_CALL = "ABSENT"
