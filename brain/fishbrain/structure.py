"""FISHBRAIN G1b structure (task S): does the Fish1 wiring carry side and stimulus information?

Graph analysis only; nothing is simulated. Pre-registration, definitions, results and every choice:
``launches/fishbrain/evidence/G1b-structure.md``.

1. **Path weight.** Every (pre, post) pair of the brain of record becomes an input fraction:
   excitatory E[j, i] = t2(i->j) / n_in(j), signed S[j, i] = (t2 - t1)(i->j) / n_in(j), where n_in(j)
   counts all of j's input synapses. For a source vector s, x_k = W^k s is the share of each cell's
   input traceable to the source through walks of exactly k steps (a linear rate model with unit
   gains). The primary statistic is the cumulative excitatory weight x_1 + x_2 + x_3.
2. **Side index** of a readout pair (X_L, X_R) driven by the two tectal lobes::

       SI = 1/2 [ (a - b) / (|a| + |b|) + (d - c) / (|c| + |d|) ]

   with a = w(left lobe -> X_L), b = w(left lobe -> X_R), c = w(right lobe -> X_L),
   d = w(right lobe -> X_R). +1 = each lobe routes to its own side, -1 = to the other side. It is
   exactly 0 when w factorises as source x target, so lobe size and a target's own input bias cancel.
3. **Selectivity index**: the same formula with (loom patch, prey patch) as the sources and
   (escape, strike | turning) as the targets, pooled over both sides.
4. **Intermediate sets** from each lobe to each readout (hop-1 and hop-2 cells on its <=3-step paths)
   and their Jaccard overlap; **reachability** of each readout's input from the tectum.
5. **Nulls.** Null A: degree-preserving whole-brain shuffle, the algorithm of
   ``sim.Network.shuffled_copy`` applied to the raw pairs so each pair keeps (pre, t1, t2). Null B
   (sensitivity): the same, but posts are permuted only within a (pre side, post side) class.

Run: ``python -m fishbrain.structure run`` (real wiring, then the shuffles, resumable; each shuffle
is written to ``data/structure/shuffles/`` as it finishes), then ``python -m fishbrain.structure
assemble`` writes ``data/structure/structure.json``.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = DATA / "structure"
STRUCTURE_JSON = OUT / "structure.json"
SHUFFLE_DIR = OUT / "shuffles"
READOUTS_JSON = DATA / "readouts.json"
CODE_VERSION = "g1b-structure-v1"

SIDES = ("left", "right")
CONTRA = {"left": "right", "right": "left"}     # an eye drives the opposite tectal lobe (B1)
HOPS = ("hop1", "hop2", "hop3", "cum", "h23")   # h23 = hops 2 + 3 (post hoc, added 02:34 EDT; see the note s.9)
KINDS = ("exc", "sig")

# ---- pre-registered 2026-09-26 02:27 EDT (G1b-structure.md s.0) ----
N_SHUFFLES = 20
N_SIDE_SHUFFLES = 10
MIN_SHUFFLES = 5
SHUFFLE_SEED = "G1b-structure-shuffle-{k}"
SIDE_SHUFFLE_SEED = "G1b-structure-sideshuffle-{k}"
PATCHES = {"loom": (90.0, 30.0, 30.0), "prey": (20.0, 0.0, 1.5)}   # (|azimuth|, elevation, radius), degrees
SEP_MIN = 0.10          # |index| floor for a separation
Z_MIN = 3.0             # |z| floor against Null A
PC1_MIN = 0.5           # lobe -> own-lobe tectal cells, hop 1
PC2_MIN = 0.1           # loom zone vs prey zone, hop 1
ZONE_FRAC = 0.10        # top 10% of a patch's drive within the driven lobe
SIDE_VERDICT = ("m_system", "turning", "strike")
SEL_VERDICT = ("E_vs_strike", "E_vs_turning")
PRIMARY = ("exc", "cum")
# literature sign of SI (lobe -> readout): -1 contralateral, +1 ipsilateral, None = no prediction
PREDICTED_SI = {
    "m_system": {"sign": -1, "source": "Dunn et al. 2016 via brain.MAUTHNER_PREDICTION (ipsilateral to the stimulus = contralateral to the lobe)"},
    "mauthner": {"sign": -1, "source": "as m_system"},
    "turning": {"sign": -1, "source": "Huang et al. 2013: RoV3/MiV1/MiV2 fire for turns to their own side; prey on the right drives the left lobe and turns right"},
    "strike": {"sign": None, "source": "conflict: Thiele 2014 + strike toward prey -> -1; Gahtan 2005 anatomy (ipsilateral tectal dendrites) and task R's direct contacts -> +1",
               "thiele": -1, "gahtan_anatomy": +1},
    "nIII_dorsal": {"sign": None, "source": "convergence needs both medial recti; no side prediction"},
    "forward": {"sign": None, "source": "no side prediction"},
    "turning_dilated": {"sign": -1, "source": "as turning"},
    "strike_dilated": {"sign": None, "source": "as strike"},
}
PREDICTED_SEL = +1      # loom -> escape, prey -> strike / turning
JACCARD_POPS = ("m_system", "mauthner", "turning", "strike", "nIII_dorsal")


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =============================================================================================
# 1. indices (pure functions)
# =============================================================================================

def side_index(a: float, b: float, c: float, d: float) -> Optional[float]:
    """1/2 [(a - b)/(|a| + |b|) + (d - c)/(|c| + |d|)]; None if a source reaches neither target.

    a = source1 -> target1, b = source1 -> target2, c = source2 -> target1, d = source2 -> target2.
    +1: each source routes only to "its" target; -1: only to the other; 0: w = f(source) g(target)."""
    n1, n2 = abs(a) + abs(b), abs(c) + abs(d)
    if not (n1 > 0 and n2 > 0) or not all(map(math.isfinite, (a, b, c, d))):
        return None
    return 0.5 * ((a - b) / n1 + (d - c) / n2)


def jaccard(A: np.ndarray, Bm: np.ndarray) -> Optional[float]:
    u = int(np.count_nonzero(A | Bm))
    return None if u == 0 else int(np.count_nonzero(A & Bm)) / u


def weighted_jaccard(fa: np.ndarray, fb: np.ndarray) -> Optional[float]:
    """sum min / sum max of two nonnegative flow vectors, each normalised to sum 1."""
    sa, sb = float(fa.sum()), float(fb.sum())
    if not (sa > 0 and sb > 0):
        return None
    pa, pb = fa / sa, fb / sb
    return float(np.minimum(pa, pb).sum() / np.maximum(pa, pb).sum())


def null_summary(real: Optional[float], null: Sequence[Optional[float]]) -> dict:
    """z and two-sided empirical p of a real value against a null sample (None = undefined)."""
    v = np.array([x for x in null if x is not None and math.isfinite(x)], float)
    out = {"real": real, "n_null": len(null), "n_defined": int(len(v)), "null": [None if x is None else float(x) for x in null]}
    if real is None or not len(v):
        out.update({"mean": float(v.mean()) if len(v) else None, "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                    "z": None, "p_two_sided": None, "beyond_all": None})
        return out
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    z = (real - m) / sd if sd > 0 else (math.inf if real != m else 0.0) * (1 if real >= m else -1)
    k = int((np.abs(v - m) >= abs(real - m) - 1e-15).sum())
    out.update({"mean": m, "sd": sd, "min": float(v.min()), "max": float(v.max()), "z": z,
                "p_two_sided": (1 + k) / (1 + len(v)),
                "beyond_all": bool(abs(real) > np.abs(v).max())})
    return out


# =============================================================================================
# 2. the wiring as input-fraction matrices
# =============================================================================================

class Wiring:
    """Pairs (pre -> post, t1, t2) as sparse input-fraction matrices W[kind] (rows = post)."""

    def __init__(self, n: int, pre: np.ndarray, post: np.ndarray, t1: np.ndarray, t2: np.ndarray, label: str = ""):
        self.n, self.label = int(n), label
        self.pre = np.asarray(pre, np.int32)
        self.post = np.asarray(post, np.int32)
        self.t1 = np.asarray(t1, np.int32)
        self.t2 = np.asarray(t2, np.int32)
        nsyn = (self.t1 + self.t2).astype(np.float64)
        self.n_in = np.bincount(self.post, weights=nsyn, minlength=n)
        self.in_deg = np.bincount(self.post, minlength=n)
        self.out_deg = np.bincount(self.pre, minlength=n)
        order = np.argsort(self.post, kind="stable")
        indptr = np.zeros(n + 1, np.int64)
        np.cumsum(self.in_deg, out=indptr[1:])
        if indptr[-1] < 2 ** 31 - 1:
            indptr = indptr.astype(np.int32)
        idx = self.pre[order]
        inv = 1.0 / self.n_in[self.post[order]]
        self.W = {"exc": sp.csr_matrix((self.t2[order] * inv, idx, indptr), shape=(n, n), copy=False),
                  "sig": sp.csr_matrix(((self.t2[order] - self.t1[order]) * inv, idx, indptr), shape=(n, n), copy=False)}
        del order, inv

    def walk(self, s: np.ndarray, kind: str, hops: int = 3) -> List[np.ndarray]:
        W = self.W[kind]
        out, x = [], np.asarray(s, np.float64)
        for _ in range(hops):
            x = W @ x
            out.append(x)
        return out

    def back(self, y: np.ndarray, kind: str = "exc") -> np.ndarray:
        return self.W[kind].T @ y

    def forward_csr(self, mask: Optional[np.ndarray] = None):
        """CSR over pre (pairs are kept in pre order), optionally restricted to mask."""
        pre, post = (self.pre, self.post) if mask is None else (self.pre[mask], self.post[mask])
        if len(pre) > 1 and np.any(pre[1:] < pre[:-1]):
            o = np.argsort(pre, kind="stable")
            pre, post = pre[o], post[o]
        indptr = np.zeros(self.n + 1, np.int64)
        np.cumsum(np.bincount(pre, minlength=self.n), out=indptr[1:])
        return indptr, post


def bfs_levels(n, indptr, indices, sources) -> np.ndarray:
    from fishbrain import brain as B
    return B.bfs_levels(n, indptr, indices, sources)


# =============================================================================================
# 3. nulls
# =============================================================================================

def shuffle_post(pre: np.ndarray, post: np.ndarray, n: int, seed: str, groups: Optional[np.ndarray] = None,
                 max_rounds: int = 200) -> Tuple[np.ndarray, dict]:
    """sim.Network.shuffled_copy's algorithm on raw pairs: pre (and each pair's counts) stay put,
    post ends are permuted (within `groups` if given), duplicate pairs and self-loops are repaired by
    swapping post ends with random partners (from the same group) until none remain."""
    from fishbrain import sim as S
    nnz = len(post)
    bg = S.make_bitgen(seed, "shuffle")
    post64 = np.asarray(post, np.int64)
    if groups is None:
        cols = post64[np.argsort(bg.random_raw(nnz), kind="stable")]
        gid = gpos = gstart = gsize = None
    else:
        gid = np.asarray(groups, np.int64)
        gpos = np.argsort(gid, kind="stable")
        gsize = np.bincount(gid)
        gstart = np.concatenate([[0], np.cumsum(gsize)[:-1]])
        cols = post64.copy()
        for g in range(len(gsize)):
            P = gpos[gstart[g]:gstart[g] + gsize[g]]
            if len(P):
                cols[P] = post64[P][np.argsort(bg.random_raw(len(P)), kind="stable")]
    rows = np.asarray(pre, np.int64)
    rounds, n_bad = 0, 0
    for rounds in range(1, max_rounds + 1):
        key = rows * n + cols
        order = np.argsort(key, kind="stable")
        ks = key[order]
        del key
        bad_mask = np.zeros(nnz, bool)
        bad_mask[order[1:][ks[1:] == ks[:-1]]] = True
        del order, ks
        bad_mask |= rows == cols
        bad = np.flatnonzero(bad_mask)
        n_bad = len(bad)
        if n_bad == 0:
            break
        r = bg.random_raw(n_bad)
        if gid is None:
            partners = (r % np.uint64(nnz)).astype(np.int64)
        else:
            g = gid[bad]
            partners = gpos[gstart[g] + (r % gsize[g].astype(np.uint64)).astype(np.int64)]
        ok = ~bad_mask[partners]
        first = np.zeros(n_bad, bool)
        first[np.unique(partners, return_index=True)[1]] = True
        ok &= first
        b, pp = bad[ok], partners[ok]
        tmp = cols[b].copy()
        cols[b] = cols[pp]
        cols[pp] = tmp
    return cols.astype(np.int32), {"rounds": rounds, "unrepaired_pairs": int(n_bad)}


def segment_sides(ids: np.ndarray) -> np.ndarray:
    """Per segment: 1 = fish's left (high y), 0 = right, from its synapse centroid (cells.parquet)
    against the fitted midline. Every segment in the pairs has at least one synapse."""
    import pyarrow.parquet as pq
    from fishbrain import cells as C
    t = pq.read_table(DATA / "cells.parquet", columns=["seg_id", "cx", "cy"])
    seg = t.column("seg_id").to_numpy().astype(np.uint64)
    if len(seg) != len(ids) or not np.array_equal(seg, ids):
        raise ValueError("cells.parquet is not aligned with the pair ids")
    axes = C.load()["axes"]
    cx = t.column("cx").to_numpy().astype(float)
    cy = t.column("cy").to_numpy().astype(float)
    return (cy > C.midline_y(cx, axes)).astype(np.int8)


# =============================================================================================
# 4. context: sources, readouts, patches (fixed from the real wiring)
# =============================================================================================

@dataclass
class Context:
    n: int
    ids: np.ndarray
    tect: Dict[str, np.ndarray]            # in-graph tectal indices per lobe
    tect_mask: np.ndarray
    pops: Dict[str, Dict[str, np.ndarray]]  # pop -> side -> indices of members with input synapses
    pops_all: Dict[str, Dict[str, np.ndarray]]  # every in-graph member
    patch: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]  # kind -> eye -> (idx, drive)
    zones: Dict[str, Dict[str, np.ndarray]]  # eye -> {loom, prey} -> tectal indices (hop-1 PC2 targets)
    tect_recv: Dict[str, np.ndarray]        # tectal cells with input synapses, per lobe (PC1 targets)
    info: dict


def readout_populations(ro: Optional[dict] = None) -> Dict[str, Dict[str, List[int]]]:
    ro = json.load(open(READOUTS_JSON)) if ro is None else ro
    rs, dl = ro["readout_sets"]["strict"], ro["readout_sets"]["dilated_5um"]
    pops = {
        "m_system": {s: list(rs[f"Mauthner_{s}"]) + list(rs[f"MiD2cm_{s}"]) + list(rs[f"MiD3cm_{s}"]) for s in SIDES},
        "mauthner": {s: list(rs[f"Mauthner_{s}"]) for s in SIDES},
        "turning": {s: list(rs[f"turning_{s}"]) for s in SIDES},
        "strike": {s: list(rs[f"nMLF_{s}"]) for s in SIDES},
        "nIII_dorsal": {s: list(rs[f"nIII_dorsal_{s}"]) for s in SIDES},
        "forward": {s: list(rs[f"forward_{s}"]) for s in SIDES},
        "turning_dilated": {s: list(dl[f"turning_{s}"]) for s in SIDES},
        "strike_dilated": {s: list(dl[f"nMLF_{s}"]) for s in SIDES},
    }
    return pops


def patch_points(kind: str, eye: str) -> list:
    from fishbrain import cells as C
    az, el, r = PATCHES[kind]
    return C.disk(az if eye == "right" else -az, el, r)


def build_context(P: dict, n_in: np.ndarray) -> Context:
    from fishbrain import brain as B
    from fishbrain import cells as C
    ids = P["ids"]
    n = len(ids)
    cells = C.load()
    tseg = C.tectum_seg_ids(cells)
    tside = C.tectum_map(cells).side
    tix = B.index_of(ids, tseg)
    excl = B.index_of(ids, np.array(sorted(C.never_stimulate(cells)), np.uint64))
    excl = excl[excl >= 0]
    tect = {s: np.unique(tix[(tix >= 0) & (tside == s)]) for s in SIDES}
    if np.intersect1d(np.concatenate(list(tect.values())), excl).size:
        raise AssertionError("a tectal source is in the exclusion list")
    tect_mask = np.zeros(n, bool)
    for s in SIDES:
        tect_mask[tect[s]] = True
    pops, pops_all, dropped = {}, {}, {}
    for name, sides in readout_populations().items():
        pops[name], pops_all[name] = {}, {}
        for s, segs in sides.items():
            ix = B.index_of(ids, np.array(segs, np.uint64))
            ing = np.unique(ix[ix >= 0])
            recv = ing[n_in[ing] > 0]
            if np.intersect1d(ing, np.flatnonzero(tect_mask)).size:
                raise AssertionError(f"readout {name}_{s} overlaps the tectal sources")
            pops[name][s], pops_all[name][s] = recv, ing
            dropped[f"{name}_{s}"] = {"listed": len(segs), "in_graph": int(len(ing)), "with_inputs": int(len(recv))}
    patch = {k: {} for k in PATCHES}
    zones = {}
    tect_recv = {s: tect[s][n_in[tect[s]] > 0] for s in SIDES}
    drive_info = {}
    for eye in SIDES:
        lobe = CONTRA[eye]
        dv = {}
        for k in PATCHES:
            d = C.tectum_drive({eye: patch_points(k, eye)}, cells)
            ok = (tix >= 0) & (d > 0)
            if np.any(tside[ok] != lobe):
                raise AssertionError("a patch drove the wrong lobe")
            idx, val = tix[ok], d[ok]
            o = np.argsort(idx)
            patch[k][eye] = (idx[o], val[o])
            full = np.zeros(n)
            full[idx] = val
            dv[k] = full
            drive_info[f"{k}_{eye}_eye"] = {"points": len(patch_points(k, eye)), "cells_driven": int(len(idx)),
                                            "drive_sum": float(val.sum()), "drive_max": float(val.max()),
                                            "lobe": lobe}
        cand = tect_recv[lobe]
        m = max(1, int(round(ZONE_FRAC * len(cand))))
        zl = cand[np.argsort(-dv["loom"][cand], kind="stable")[:m]]
        zp = cand[np.argsort(-dv["prey"][cand], kind="stable")[:m]]
        both = np.intersect1d(zl, zp)
        zones[eye] = {"loom": np.setdiff1d(zl, both), "prey": np.setdiff1d(zp, both)}
        drive_info[f"zones_{eye}_eye"] = {"size": m, "overlap_removed": int(len(both))}
    info = {"tectal_in_graph": {s: int(len(tect[s])) for s in SIDES},
            "tectal_with_inputs": {s: int(len(tect_recv[s])) for s in SIDES},
            "populations": dropped, "patches": drive_info}
    return Context(n, ids, tect, tect_mask, pops, pops_all, patch, zones, tect_recv, info)


# =============================================================================================
# 5. the measurements on one wiring
# =============================================================================================

def _mean(x: np.ndarray, idx: np.ndarray) -> float:
    return float(x[idx].mean()) if len(idx) else float("nan")


def _hopdict(xs: List[np.ndarray]) -> Dict[str, np.ndarray]:
    return {"hop1": xs[0], "hop2": xs[1], "hop3": xs[2], "cum": xs[0] + xs[1] + xs[2], "h23": xs[1] + xs[2]}


def _r(v: Optional[float], nd: int = 6):
    return None if v is None or not math.isfinite(v) else round(float(v), nd)


def measure(w: Wiring, ctx: Context) -> dict:
    """Every number for one wiring. Returns {"detail": nested, "flat": {key: scalar}}."""
    t0 = time.time()
    n = ctx.n
    flat: Dict[str, Optional[float]] = {}
    det: dict = {"lobe_weights": {}, "SI": {}, "patch_weights": {}, "SEL": {}, "SSI": {}, "PC1": {}, "PC2": {}}
    # ---------------- lobes -> readouts
    LW = {k: {h: {p: {lobe: {} for lobe in SIDES} for p in ctx.pops} for h in HOPS} for k in KINDS}
    PCW = {k: {h: {lobe: {} for lobe in SIDES} for h in HOPS} for k in KINDS}
    keepE = {}
    for lobe in SIDES:
        s = np.zeros(n)
        s[ctx.tect[lobe]] = 1.0
        for kind in KINDS:
            xs = w.walk(s, kind)
            hd = _hopdict(xs)
            for h, x in hd.items():
                for p, sides in ctx.pops.items():
                    for side, idx in sides.items():
                        LW[kind][h][p][lobe][side] = _mean(x, idx)
                for side in SIDES:
                    PCW[kind][h][lobe][side] = _mean(x, ctx.tect_recv[side])
            if kind == "exc":
                keepE[lobe] = (xs[0], xs[1])
            del xs, hd
    for kind in KINDS:
        det["SI"][kind], det["PC1"][kind] = {}, {}
        for h in HOPS:
            det["SI"][kind][h] = {}
            for p in ctx.pops:
                q = LW[kind][h][p]
                si = side_index(q["left"]["left"], q["left"]["right"], q["right"]["left"], q["right"]["right"])
                det["SI"][kind][h][p] = _r(si)
                flat[f"SI|{kind}|{h}|{p}"] = si
            q = PCW[kind][h]
            pc = side_index(q["left"]["left"], q["left"]["right"], q["right"]["left"], q["right"]["right"])
            det["PC1"][kind][h] = _r(pc)
            flat[f"PC1|{kind}|{h}"] = pc
    det["lobe_weights"] = {k: {h: {p: {lobe: {s: _r(v, 9) for s, v in d2.items()} for lobe, d2 in d1.items()}
                                   for p, d1 in LW[k][h].items()} for h in HOPS} for k in KINDS}
    det["pc1_weights"] = {k: {h: {lobe: {s: _r(v, 9) for s, v in PCW[k][h][lobe].items()} for lobe in SIDES}
                              for h in HOPS} for k in KINDS}
    # ---------------- patches -> readouts (per side and pooled)
    PW = {k: {h: {} for h in HOPS} for k in KINDS}
    Z = {k: {} for k in KINDS}
    for pk, eyes in ctx.patch.items():
        for eye, (idx, val) in eyes.items():
            s = np.zeros(n)
            s[idx] = val
            for kind in KINDS:
                hd = _hopdict(w.walk(s, kind))
                for h, x in hd.items():
                    row = {}
                    for p, sides in ctx.pops.items():
                        row[p] = {side: _mean(x, ix) for side, ix in sides.items()}
                        row[p]["both"] = _mean(x, np.concatenate([sides["left"], sides["right"]]))
                    PW[kind][h][f"{pk}|{eye}"] = row
                Z[kind][f"{pk}|{eye}"] = {z: _mean(hd["hop1"], ctx.zones[eye][z]) for z in ("loom", "prey")}
                del hd
    contrasts = {"E_vs_strike": ("m_system", "strike"), "E_vs_turning": ("m_system", "turning"),
                 "E_vs_nIII_dorsal": ("m_system", "nIII_dorsal"), "mauthner_vs_strike": ("mauthner", "strike"),
                 "E_vs_forward": ("m_system", "forward")}
    for kind in KINDS:
        det["SEL"][kind], det["SSI"][kind], det["PC2"][kind] = {}, {}, {}
        for h in HOPS:
            det["SEL"][kind][h] = {}
            for cname, (E, X) in contrasts.items():
                per = {}
                for eye in SIDES:
                    L, Pr = PW[kind][h][f"loom|{eye}"], PW[kind][h][f"prey|{eye}"]
                    per[eye] = side_index(L[E]["both"], L[X]["both"], Pr[E]["both"], Pr[X]["both"])
                vals = [v for v in per.values() if v is not None]
                mean = float(np.mean(vals)) if len(vals) == 2 else None
                det["SEL"][kind][h][cname] = {"left_eye": _r(per["left"]), "right_eye": _r(per["right"]), "mean": _r(mean)}
                flat[f"SEL|{kind}|{h}|{cname}|mean"] = mean
                for eye in SIDES:
                    flat[f"SEL|{kind}|{h}|{cname}|{eye}_eye"] = per[eye]
            det["SSI"][kind][h] = {}
            for pk in PATCHES:
                for p in ctx.pops:
                    L, R_ = PW[kind][h][f"{pk}|left"][p], PW[kind][h][f"{pk}|right"][p]
                    # +1 = the readout on the stimulus side (left-eye patch -> X_L, right-eye patch -> X_R)
                    v = side_index(L["left"], L["right"], R_["left"], R_["right"])
                    det["SSI"][kind][h][f"{pk}->{p}"] = _r(v)
                    flat[f"SSI|{kind}|{h}|{pk}|{p}"] = v
        per = {eye: side_index(Z[kind][f"loom|{eye}"]["loom"], Z[kind][f"loom|{eye}"]["prey"],
                               Z[kind][f"prey|{eye}"]["loom"], Z[kind][f"prey|{eye}"]["prey"]) for eye in SIDES}
        vals = [v for v in per.values() if v is not None]
        mean = float(np.mean(vals)) if len(vals) == 2 else None
        det["PC2"][kind]["hop1"] = {"left_eye": _r(per["left"]), "right_eye": _r(per["right"]), "mean": _r(mean)}
        flat[f"PC2|{kind}|hop1|mean"] = mean
    det["patch_weights"] = {k: {h: {src: {p: {s: _r(v, 9) for s, v in d.items()} for p, d in row.items()}
                                    for src, row in PW[k][h].items()} for h in ("hop1", "cum", "h23")} for k in KINDS}
    # ---------------- intermediate sets (excitatory) and Jaccard
    det["jaccard"] = {}
    excl0 = ctx.tect_mask
    for p in JACCARD_POPS:
        det["jaccard"][p] = {}
        for side in SIDES:
            T = ctx.pops[p][side]
            if not len(T):
                continue
            y = np.zeros(n)
            y[T] = 1.0 / len(T)
            b1 = w.back(y)
            b2 = w.back(b1)
            excl = excl0.copy()
            excl[ctx.pops_all[p][side]] = True
            ok12, ok1 = ((b1 > 0) | (b2 > 0)) & ~excl, (b1 > 0) & ~excl
            I1 = {lobe: (keepE[lobe][0] > 0) & ok12 for lobe in SIDES}
            I2 = {lobe: (keepE[lobe][1] > 0) & ok1 for lobe in SIDES}
            f1 = {lobe: np.where(I1[lobe], keepE[lobe][0] * (b1 + b2), 0.0) for lobe in SIDES}
            f2 = {lobe: np.where(I2[lobe], keepE[lobe][1] * b1, 0.0) for lobe in SIDES}
            row = {}
            for tag, I, f in (("hop1", I1, f1), ("hop2", I2, f2)):
                j, wj = jaccard(I["left"], I["right"]), weighted_jaccard(f["left"], f["right"])
                row[tag] = {"size_left_lobe": int(I["left"].sum()), "size_right_lobe": int(I["right"].sum()),
                            "shared": int((I["left"] & I["right"]).sum()), "jaccard": _r(j), "weighted_jaccard": _r(wj),
                            "flow_left_lobe": _r(float(f["left"].sum()), 12), "flow_right_lobe": _r(float(f["right"].sum()), 12)}
                flat[f"J|{tag}|{p}|{side}"] = j
                flat[f"wJ|{tag}|{p}|{side}"] = wj
            det["jaccard"][p][side] = row
            del b1, b2, excl, ok12, ok1, I1, I2, f1, f2
    g = {}
    for tag, k in (("hop1", 0), ("hop2", 1)):
        A_ = (keepE["left"][k] > 0) & ~excl0
        B_ = (keepE["right"][k] > 0) & ~excl0
        g[tag] = {"size_left_lobe": int(A_.sum()), "size_right_lobe": int(B_.sum()), "shared": int((A_ & B_).sum()),
                  "jaccard": _r(jaccard(A_, B_)),
                  "weighted_jaccard": _r(weighted_jaccard(np.where(A_, keepE["left"][k], 0.0), np.where(B_, keepE["right"][k], 0.0)))}
        flat[f"Jglobal|{tag}"] = g[tag]["jaccard"]
        flat[f"wJglobal|{tag}"] = g[tag]["weighted_jaccard"]
    det["jaccard"]["global"] = g
    del keepE
    # ---------------- reachability and direct contacts
    src = np.concatenate([ctx.tect["left"], ctx.tect["right"]])
    ip, ix = w.forward_csr()
    d_any = bfs_levels(n, ip, ix, src)
    del ip, ix
    emask = w.t2 > 0
    ip, ix = w.forward_csr(emask)
    d_exc = bfs_levels(n, ip, ix, src)
    del ip, ix
    orphan = (w.n_in == 0) & ~ctx.tect_mask
    lobe_of = np.full(n, -1, np.int8)
    lobe_of[ctx.tect["left"]], lobe_of[ctx.tect["right"]] = 0, 1
    det["reach"], det["direct"] = {}, {}
    nsyn = (w.t1 + w.t2).astype(np.int64)
    for p in ctx.pops:
        det["reach"][p], det["direct"][p] = {}, {}
        for side in list(SIDES) + ["both"]:
            members = ctx.pops_all[p][side] if side != "both" else np.concatenate([ctx.pops_all[p]["left"], ctx.pops_all[p]["right"]])
            m = np.zeros(n, bool)
            m[members] = True
            sel = np.flatnonzero(m[w.post])
            pr = w.pre[sel].astype(np.int64)
            a_syn, e_syn = nsyn[sel], w.t2[sel].astype(np.int64)
            tot, tot_e = int(a_syn.sum()), int(e_syn.sum())
            da, de = d_any[pr], d_exc[pr]
            row = {"input_synapses": tot, "input_exc_synapses": tot_e,
                   "orphan_share": _r(a_syn[orphan[pr]].sum() / tot if tot else None),
                   "orphan_share_exc": _r(e_syn[orphan[pr]].sum() / tot_e if tot_e else None)}
            for k in (1, 2, 3):
                row[f"any_le{k}"] = _r(a_syn[(da >= 0) & (da <= k - 1)].sum() / tot if tot else None)
                row[f"exc_le{k}"] = _r(e_syn[(de >= 0) & (de <= k - 1)].sum() / tot_e if tot_e else None)
            row["min_hops_any"] = int(d_any[members][d_any[members] >= 0].min()) if (d_any[members] >= 0).any() else None
            row["min_hops_exc"] = int(d_exc[members][d_exc[members] >= 0].min()) if (d_exc[members] >= 0).any() else None
            if side != "both":
                row["walk_share_exc_cum"] = _r(LW["exc"]["cum"][p]["left"][side] + LW["exc"]["cum"][p]["right"][side], 9)
                lo = lobe_of[pr]
                det["direct"][p][side] = {"from_left_lobe": int(a_syn[lo == 0].sum()), "from_right_lobe": int(a_syn[lo == 1].sum()),
                                          "exc_from_left_lobe": int(e_syn[lo == 0].sum()), "exc_from_right_lobe": int(e_syn[lo == 1].sum())}
            for key in ("orphan_share", "any_le1", "any_le2", "any_le3", "exc_le1", "exc_le2", "exc_le3"):
                flat[f"reach|{key}|{p}|{side}"] = row[key]
            det["reach"][p][side] = row
        dd = det["direct"][p]
        ipsi = dd["left"]["from_left_lobe"] + dd["right"]["from_right_lobe"]
        contra = dd["right"]["from_left_lobe"] + dd["left"]["from_right_lobe"]
        dd["ipsi"], dd["contra"] = ipsi, contra
        flat[f"direct|ipsi|{p}"], flat[f"direct|contra|{p}"] = float(ipsi), float(contra)
    det["tectum_reach"] = {"cells_within_1_any": int(((d_any >= 0) & (d_any <= 1)).sum()),
                           "cells_within_2_any": int(((d_any >= 0) & (d_any <= 2)).sum()),
                           "cells_within_1_exc": int(((d_exc >= 0) & (d_exc <= 1)).sum()),
                           "cells_within_2_exc": int(((d_exc >= 0) & (d_exc <= 2)).sum()),
                           "orphan_segments": int(orphan.sum())}
    det["seconds"] = round(time.time() - t0, 1)
    return {"detail": det, "flat": {k: (None if v is None or not math.isfinite(v) else float(v)) for k, v in flat.items()}}


# =============================================================================================
# 6. verdict (from real flat values and their Null-A summaries)
# =============================================================================================

def separation(real: Optional[float], summ: Mapping) -> dict:
    ok = (real is not None and summ.get("z") is not None and abs(real) >= SEP_MIN
          and bool(summ.get("beyond_all")) and abs(summ["z"]) >= Z_MIN)
    return {"separates": bool(ok), "value": real, "z": summ.get("z"), "p": summ.get("p_two_sided"),
            "beyond_all": summ.get("beyond_all"), "null_mean": summ.get("mean"), "null_sd": summ.get("sd")}


def verdict(real: Mapping[str, Optional[float]], nullA: Mapping[str, Mapping]) -> dict:
    k, h = PRIMARY
    side = {p: separation(real.get(f"SI|{k}|{h}|{p}"), nullA.get(f"SI|{k}|{h}|{p}", {})) for p in SIDE_VERDICT}
    stim = {c: separation(real.get(f"SEL|{k}|{h}|{c}|mean"), nullA.get(f"SEL|{k}|{h}|{c}|mean", {})) for c in SEL_VERDICT}
    pc1v, pc1s = real.get("PC1|exc|hop1"), nullA.get("PC1|exc|hop1", {})
    pc2v, pc2s = real.get("PC2|exc|hop1|mean"), nullA.get("PC2|exc|hop1|mean", {})
    pc1 = bool(pc1v is not None and pc1v >= PC1_MIN and pc1s.get("beyond_all"))
    pc2 = bool(pc2v is not None and pc2v >= PC2_MIN and pc2s.get("beyond_all") and (pc2s.get("z") or 0) >= Z_MIN)
    n_side = sum(v["separates"] for v in side.values())
    n_stim = sum(v["separates"] for v in stim.values())
    if not (pc1 and pc2):
        call = "INVALID"
    elif n_side >= 2 and n_stim >= 1:
        call = "CARRIES"
    elif n_side + n_stim >= 1:
        call = "WEAK"
    else:
        call = "ABSENT"
    direction = {}
    for p, v in side.items():
        pred = PREDICTED_SI[p]
        sgn = None if v["value"] is None or v["value"] == 0 else (1 if v["value"] > 0 else -1)
        direction[p] = {"measured_sign": sgn, "predicted_sign": pred["sign"], "separates": v["separates"],
                        "matches": None if pred["sign"] is None or sgn is None else sgn == pred["sign"]}
        if p == "strike" and sgn is not None:
            direction[p]["matches_thiele"] = sgn == pred["thiele"]
            direction[p]["matches_gahtan_anatomy"] = sgn == pred["gahtan_anatomy"]
    for c, v in stim.items():
        sgn = None if v["value"] is None or v["value"] == 0 else (1 if v["value"] > 0 else -1)
        direction[c] = {"measured_sign": sgn, "predicted_sign": PREDICTED_SEL, "separates": v["separates"],
                        "matches": None if sgn is None else sgn == PREDICTED_SEL}
    return {"call": call, "side": side, "stimulus": stim, "n_side_separations": n_side, "n_stimulus_separations": n_stim,
            "positive_controls": {"PC1": {"pass": pc1, "value": pc1v, "min": PC1_MIN, "null_max_abs": _maxabs(pc1s)},
                                  "PC2": {"pass": pc2, "value": pc2v, "min": PC2_MIN, "z": pc2s.get("z"),
                                          "null_max_abs": _maxabs(pc2s)}},
            "direction": direction,
            "rule": {"primary": f"{k}|{h}", "sep_min": SEP_MIN, "z_min": Z_MIN, "side_readouts": SIDE_VERDICT,
                     "stimulus_contrasts": SEL_VERDICT}}


def _maxabs(s: Mapping) -> Optional[float]:
    v = [abs(x) for x in s.get("null", []) if x is not None]
    return max(v) if v else None


# =============================================================================================
# 7. runner
# =============================================================================================

def load_real():
    from fishbrain import brain as B
    P = B.load_pairs()
    w = Wiring(len(P["ids"]), P["pre"], P["post"], P["t1"], P["t2"], label="real")
    ctx = build_context(P, w.n_in)
    return P, w, ctx


def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    os.replace(tmp, path)


def run(n_a: int = N_SHUFFLES, n_b: int = N_SIDE_SHUFFLES, only_real: bool = False):
    SHUFFLE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P, w, ctx = load_real()
    _log("loaded", round(time.time() - t0, 1), "s")
    real_p = SHUFFLE_DIR / "real.json"
    if not real_p.exists():
        m = measure(w, ctx)
        m["context"] = ctx.info
        m["built"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        _dump(real_p, m)
        _log("real done", m["detail"]["seconds"], "s")
    if only_real:
        return
    in0, out0 = w.in_deg.copy(), w.out_deg.copy()
    del w
    sides = None
    jobs = [("A", k, SHUFFLE_SEED.format(k=k)) for k in range(n_a)] + [("B", k, SIDE_SHUFFLE_SEED.format(k=k)) for k in range(n_b)]
    for kind, k, seed in jobs:
        path = SHUFFLE_DIR / f"{kind}-{k:02d}.json"
        if path.exists():
            continue
        if kind == "B" and sides is None:
            sides = segment_sides(P["ids"])
        m = one_shuffle(P, ctx, kind, k, sides, in0, out0)
        _dump(path, m)
        _log(f"shuffle {kind}-{k}", m["repair"], m["checks"], m["seconds_total"], "s")


def one_shuffle(P: dict, ctx: Context, kind: str, k: int, sides: Optional[np.ndarray], in0: np.ndarray,
                out0: np.ndarray) -> dict:
    """Build shuffle `kind`-`k` (A: whole brain; B: within (pre side, post side) class) and measure it."""
    t1s = time.time()
    pre, t1, t2, n = P["pre"], P["t1"], P["t2"], len(P["ids"])
    seed = (SHUFFLE_SEED if kind == "A" else SIDE_SHUFFLE_SEED).format(k=k)
    groups = None
    if kind == "B":
        groups = (sides[pre].astype(np.int64) * 2 + sides[P["post"]]).astype(np.int8)
    post, rep = shuffle_post(pre, P["post"], n, seed, groups=groups)
    del groups
    ws = Wiring(n, pre, post, t1, t2, label=f"{kind}-{k}")
    checks = {"in_degree_preserved": bool(np.array_equal(ws.in_deg, in0)),
              "out_degree_preserved": bool(np.array_equal(ws.out_deg, out0))}
    if kind == "B":
        g0 = sides[pre].astype(np.int64) * 2 + sides[P["post"]]
        g1 = sides[pre].astype(np.int64) * 2 + sides[post]
        checks["side_class_preserved"] = bool(np.array_equal(g0, g1))
        del g0, g1
    m = measure(ws, ctx)
    m.update({"null": kind, "k": k, "seed": seed, "repair": rep, "checks": checks,
              "seconds_total": round(time.time() - t1s, 1)})
    return m


def replay_check(kind: str = "A", k: int = 0) -> dict:
    """Rebuild one stored shuffle from its seed and compare every flat value (determinism, measured)."""
    stored = json.load(open(SHUFFLE_DIR / f"{kind}-{k:02d}.json"))
    P, w, ctx = load_real()
    in0, out0 = w.in_deg.copy(), w.out_deg.copy()
    del w
    sides = segment_sides(P["ids"]) if kind == "B" else None
    m = one_shuffle(P, ctx, kind, k, sides, in0, out0)
    a, b = stored["flat"], m["flat"]
    diff = [key for key in a if a[key] != b.get(key)]
    return {"shuffle": f"{kind}-{k:02d}", "keys": len(a), "identical": len(diff) == 0, "differing_keys": diff[:20],
            "checked": time.strftime("%Y-%m-%d %H:%M:%S %Z")}


def tectal_output_summary(P: Optional[dict] = None) -> dict:
    """Where the tectum's own output synapses land (descriptive, real wiring only; post hoc, 02:37 EDT)."""
    from fishbrain import brain as B
    from fishbrain import cells as C
    import pyarrow.parquet as pq
    P = B.load_pairs() if P is None else P
    ids = P["ids"]
    n = len(ids)
    cells = C.load()
    tix = B.index_of(ids, C.tectum_seg_ids(cells))
    tside = C.tectum_map(cells).side
    lobe = np.full(n, -1, np.int8)
    ok = tix >= 0
    lobe[tix[ok]] = np.where(tside[ok] == "left", 1, 0)
    seg_side = segment_sides(ids)
    pre, post = P["pre"], P["post"]
    ns = (P["t1"] + P["t2"]).astype(np.int64)
    sel = lobe[pre] >= 0
    lp, lq, s, e = lobe[pre[sel]], lobe[post[sel]], ns[sel], P["t2"][sel].astype(np.int64)
    tot = int(s.sum())
    nont = lq < 0
    q = post[sel][nont]
    cx = pq.read_table(DATA / "cells.parquet", columns=["cx"]).column("cx").to_numpy()
    tx = np.array([c["pos"][0] for c in cells["tectum"]], float)
    out_counts = np.bincount(pre[sel], weights=s, minlength=n)[np.concatenate([np.flatnonzero(lobe == 0), np.flatnonzero(lobe == 1)])]
    return {"tectal_cells_in_graph": int(ok.sum()), "tectal_cells_with_outputs": int((out_counts > 0).sum()),
            "output_synapses": tot, "output_synapses_type2": int(e.sum()),
            "share_onto_same_lobe_tectal": round(float(s[lq == lp].sum() / tot), 6),
            "share_onto_other_lobe_tectal": round(float(s[(lq >= 0) & (lq != lp)].sum() / tot), 6),
            "share_onto_non_tectal": round(float(s[nont].sum() / tot), 6),
            "non_tectal_share_on_the_lobes_own_side": round(float(s[nont & (seg_side[post[sel]] == lp)].sum() / s[nont].sum()), 6),
            "non_tectal_target_x_percentiles_5_25_50_75_95": [round(float(v)) for v in np.percentile(cx[q], [5, 25, 50, 75, 95])],
            "tectal_soma_x_percentiles_5_50_95": [round(float(v)) for v in np.percentile(tx, [5, 50, 95])],
            "distinct_non_tectal_targets": int(len(np.unique(q))),
            "note": "x in 8 nm voxels, anterior -> posterior; the Mauthner somas sit at x ~70,400"}


def _load_shuffles(kind: str) -> List[dict]:
    return [json.load(open(p)) for p in sorted(SHUFFLE_DIR.glob(f"{kind}-*.json"))]


def assemble(write: bool = True) -> dict:
    real = json.load(open(SHUFFLE_DIR / "real.json"))
    A, Bn = _load_shuffles("A"), _load_shuffles("B")
    if len(A) < MIN_SHUFFLES:
        raise RuntimeError(f"only {len(A)} Null-A shuffles; the floor is {MIN_SHUFFLES}")
    rf = real["flat"]
    nullA = {k: null_summary(v, [s["flat"].get(k) for s in A]) for k, v in rf.items()}
    nullB = {k: null_summary(v, [s["flat"].get(k) for s in Bn]) for k, v in rf.items()} if Bn else {}
    for v in nullB.values():            # Null B is centred near +1: |real| > max|null| means nothing there
        v.pop("beyond_all", None)
    ver = verdict(rf, nullA)
    # Null B is centred far from 0 (laterality alone gives SI ~ +1), so |real| > max|null| means nothing
    # there: only the mean, sd, z and empirical p are reported for it, never a "separates" flag.
    def _b(key):
        s = nullB.get(key, {})
        return {"real": s.get("real"), "null_mean": s.get("mean"), "null_sd": s.get("sd"), "z": s.get("z"),
                "p_two_sided": s.get("p_two_sided"), "n_defined": s.get("n_defined")}
    side_B = {p: _b(f"SI|exc|cum|{p}") for p in SIDE_VERDICT} if Bn else {}
    stim_B = {c: _b(f"SEL|exc|cum|{c}|mean") for c in SEL_VERDICT} if Bn else {}
    power = {}
    for key, sign in (("SI|exc|cum|m_system", -1), ("SI|exc|cum|turning", -1), ("SI|exc|cum|strike", -1),
                      ("SEL|exc|cum|E_vs_strike|mean", 1), ("SEL|exc|cum|E_vs_turning|mean", 1)):
        s = nullA.get(key, {})
        if s.get("sd"):
            power[key] = {"z_at_plus_1": (1 - s["mean"]) / s["sd"], "z_at_minus_1": (-1 - s["mean"]) / s["sd"],
                          "predicted_direction": sign, "max_abs_z_in_predicted_direction": abs((sign - s["mean"]) / s["sd"])}
    replay_p = OUT / "replay_check.json"
    from fishbrain import brain as B
    meta = json.load(open(B.NET / "pairs_meta.json"))
    out = {
        "schema": {"real": "measure() output on the brain of record: detail (nested) and flat (key -> scalar)",
                   "nullA": "per flat key: real, null values, mean, sd, z, p_two_sided (empirical), beyond_all",
                   "nullB": "the same against the side-class shuffle (sensitivity)",
                   "verdict": "pre-registered rule (G1b-structure.md s.0) on the primary statistic",
                   "flat keys": "SI|kind|hop|pop, SEL|kind|hop|contrast|mean|<eye>_eye, SSI|kind|hop|patch|pop, "
                                "PC1|kind|hop, PC2|kind|hop1|mean, J|hop|pop|side, wJ|..., Jglobal|hop, "
                                "reach|measure|pop|side, direct|ipsi|pop"},
        "built": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "code_version": CODE_VERSION,
        "wiring_content_hash": B.WIRING_CONTENT_HASH, "pairs": meta,
        "preregistered": {"time": "2026-09-26 02:27 EDT", "patches": PATCHES, "sep_min": SEP_MIN, "z_min": Z_MIN,
                          "pc1_min": PC1_MIN, "pc2_min": PC2_MIN, "zone_frac": ZONE_FRAC, "primary": PRIMARY,
                          "predicted_si": PREDICTED_SI, "predicted_sel": PREDICTED_SEL,
                          "n_shuffles": N_SHUFFLES, "n_side_shuffles": N_SIDE_SHUFFLES,
                          "seeds": [SHUFFLE_SEED, SIDE_SHUFFLE_SEED]},
        "real": real,
        "shuffles": {"A": [{"k": s["k"], "seed": s["seed"], "repair": s["repair"], "checks": s["checks"],
                            "seconds_total": s["seconds_total"]} for s in A],
                     "B": [{"k": s["k"], "seed": s["seed"], "repair": s["repair"], "checks": s["checks"],
                            "seconds_total": s["seconds_total"]} for s in Bn]},
        "nullA": nullA, "nullB": nullB,
        "nullA_detail_means": _detail_means(A),
        "verdict": ver,
        "nullB_sensitivity": {"side": side_B, "stimulus": stim_B,
                              "note": "beyond_all in nullB rows assumes a null centred at 0 and does not apply; use z and p"},
        "power": {"formula": "largest |z| an index in [-1, 1] can reach = |(+-1 - null mean) / null sd|",
                  "items": power},
        "tectal_output": tectal_output_summary(),
        "replay_check": json.load(open(replay_p)) if replay_p.exists() else None,
    }
    if write:
        _dump(STRUCTURE_JSON, out)
    return out


def _detail_means(sh: List[dict]) -> dict:
    """Null-A means of the reachability and Jaccard detail (for side-by-side tables)."""
    if not sh:
        return {}
    out = {"tectum_reach": {}, "reach": {}, "direct": {}}
    for k in sh[0]["detail"]["tectum_reach"]:
        out["tectum_reach"][k] = float(np.mean([s["detail"]["tectum_reach"][k] for s in sh]))
    for p, sides in sh[0]["detail"]["reach"].items():
        out["reach"][p] = {}
        for side, row in sides.items():
            out["reach"][p][side] = {k: (float(np.mean([s["detail"]["reach"][p][side][k] for s in sh
                                                         if s["detail"]["reach"][p][side][k] is not None]))
                                         if any(s["detail"]["reach"][p][side][k] is not None for s in sh) else None)
                                     for k, v in row.items() if isinstance(v, (int, float)) or v is None}
    for p, d in sh[0]["detail"]["direct"].items():
        out["direct"][p] = {"ipsi": float(np.mean([s["detail"]["direct"][p]["ipsi"] for s in sh])),
                            "contra": float(np.mean([s["detail"]["direct"][p]["contra"] for s in sh]))}
    return out


def main(argv: Sequence[str]) -> None:
    cmd = argv[0] if argv else "run"
    if cmd == "run":
        run()
        assemble()
    elif cmd == "real":
        run(only_real=True)
    elif cmd == "replay":
        r = replay_check()
        _dump(OUT / "replay_check.json", r)
        print(json.dumps(r, indent=1))
    elif cmd == "assemble":
        o = assemble()
        print(json.dumps(o["verdict"], indent=1, default=str))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
