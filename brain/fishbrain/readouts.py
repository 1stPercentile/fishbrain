"""FISHBRAIN readouts (task R): bilateral motor populations, the M-system, and the strike and
escape decision rules.

What this module builds (``python -m fishbrain.readouts``) and writes to ``data/readouts.json``:

1. **Atlas populations.** Every CAVE soma (v709) whose position sits in one of the Fish1 atlas's
   reticulospinal / oculomotor regions (``mece2_231218``) is assigned to that region and to the
   fish's own side, then mapped to its ``seg_241003_agg241003`` segment. Membership rule = B1's
   plurality vote over a 3 x 3 x 3 neighbourhood at the 1024 x 1024 x 960 nm mip (``cells._label_votes``),
   so the "named SPNs land in their own-named region" check is comparable with B1's. The nearest
   region within r um is computed too, as a sensitivity curve only.
2. **Validation.** The 50 named HMI SPNs (49 on the right) against those populations: strict
   own-name matches, overlap of named cells with the atlas populations, a name-permutation null, a
   midline mirror test (a right SPN reflected to the left must land in the same-named left region),
   and a comparison with B1's 30 unconfirmed left mirror picks.
3. **M-system.** MiD2cm (r5) and MiD3cm (r6) on each side, found the way B1 found Mauthner: the soma
   segmentation ``lores_cbs_231218`` read at 512 x 512 x 60 nm around the atlas MiD2 / MiD3 region,
   somas ranked by volume, then input synapse count and the agglomerated mesh (lateral dendrite,
   axon crossing the midline). The same code run on B1's cached r4 box must return both Mauthner
   cells (positive control).
4. **Decision rules** (pure functions, no data needed): ``strike_decision`` (buy), ``escape_decision``
   (sell), ``turn_index``. Their thresholds are fixed in ``STRIKE_RULE`` / ``ESCAPE_RULE`` and were
   written before any simulation used these populations.
5. **Wiring-only checks.** Direct synapses from each tectal lobe onto each population, and the
   excitatory input each population receives from cells that can fire when one lobe is driven
   (G1's ``lobe_prediction`` method), against a seeded post-permutation null.

Every rule that is ours is listed in ``evidence/G1b-readouts.md`` ("Choices").
Coordinates: 8 x 8 x 30 nm voxels (x anterior->posterior, y fish's right->left, z dorsal->ventral).

Caches (atlas and soma boxes read from the public bucket) go to ``$FISHBRAIN_READOUTS_CACHE``
(default: a folder in the system temp dir). They are not needed by the tests, which read
``data/readouts.json`` plus files already in ``data/``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from fishbrain import cells as C

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
REF = DATA / "ref"
READOUTS_JSON = DATA / "readouts.json"
CACHE = Path(os.environ.get("FISHBRAIN_READOUTS_CACHE", Path(tempfile.gettempdir()) / "fishbrain_readouts_cache"))
CODE_VERSION = "g1b-readouts-v1"

UM_PER_VOXEL = np.array([0.008, 0.008, 0.030])   # 8 x 8 x 30 nm
F1024 = np.array([128, 128, 32])                # 1024 x 1024 x 960 nm mip, in 8 nm voxels
F512 = np.array([64, 64, 2])                    # 512 x 512 x 60 nm mip, in 8 nm voxels
BOX_MARGIN_8NM = np.array([1500, 1500, 60])     # B1's margin around the region boxes (12 um, 12 um, 1.8 um)
SIDE_AMBIGUOUS_UM = 3.0                         # B1's tectum rule: within 3 um of the midline, side is ambiguous
SOMA_BOX_MARGIN_UM = 20.0                       # B1's Mauthner box margin (somas must be whole)

# ---- atlas labels (mece2_231218 segment_properties, read 2026-09-25) ----
POPULATIONS = {
    # Huang et al. 2013: RoV3, MiV1, MiV2 needed for turning (phototaxis / OMR); lateralised to the turn side
    "turning": {"RoV3": 28, "MiV1": 47, "MiV2": 34},
    # HMI classifier's forward class (B1 section 4)
    "forward": {"RoM1": 24, "MiM1": 53, "MiR1": 49, "MiR2": 54, "RoL-R1": 44},
    # Mauthner + its serial homologs (Liu & Fetcho 1999; Kohashi & Oda 2008)
    "m_system": {"Mauthner": 48, "MiD2": 38, "MiD3": 55},
    # strike (buy): nMLF (Gahtan 2005; Thiele 2014) and the oculomotor nucleus (Greaney 2017)
    "strike": {"NucMLF": 16, "nIII": 15},
}
# Reticulospinal / oculomotor labels present in the atlas and NOT used by any readout here.
UNUSED_LABELS = {"RoL2": 21, "RoM2": 22, "RoM3": 27, "Spiral Fiber Neuron Posterior cluster": 29, "RoL3": 31,
                 "Spiral Fiber Neuron Anterior cluster": 32, "Mauthner Cell Axon Cap": 52, "MiT": 60,
                 "CaD": 61, "CaV": 62, "nIV": 43}
REGION_LABELS = {name: lab for grp in POPULATIONS.values() for name, lab in grp.items()}
ALL_LABELS = {**REGION_LABELS, **UNUSED_LABELS}
LABEL_NAME = {v: k for k, v in ALL_LABELS.items()}
# every RS label used to compute "nearest region" and the SPN checks (nIII / nIV are not reticulospinal)
RS_LABELS = {k: v for k, v in ALL_LABELS.items() if k not in ("nIII", "nIV")}
# mece1 rhombomere labels (segment_properties of mece1_231218)
RHOMBOMERE = {27: "r1", 24: "r2", 25: "r3", 28: "r4", 26: "r5", 29: "r6", 30: "r7", 31: "caudal hindbrain",
              20: "tegmentum", 21: "tectum"}

# nIII: IR/MR motoneurons sit predominantly in DORSAL nIII and innervate the ipsilateral eye
# (Greaney et al. 2017, J Comp Neurol, PMC5116274, read 2026-09-26). The atlas does not split nIII.
NIII_DORSAL_FRACTION = 0.5    # our proxy: the dorsal half (by z rank) of each side's atlas nIII somas

# ---- decision rules: fixed 2026-09-26 before any simulation used these populations ----
STRIKE_RULE = {
    "conv_min_active_frac": 0.2,   # each side's dorsal-nIII pool: >= 20% of cells spike in the window
    "drive_min_active_frac": 0.2,  # at least one side's nMLF: >= 20% of cells spike in the window
    "side_index_min": 0.2,         # |SI| >= 0.2 (a 1.5 : 1 ratio of mean spikes per cell) names a side
}
ESCAPE_RULE = {
    "cells": ("Mauthner", "MiD2cm", "MiD3cm"),   # the M-series (Liu & Fetcho 1999)
    "min_spikes": 1,                               # one spike of any M-series cell is an escape
}
TURN_RULE = {"side_index_min": 0.0}                # G1's turn index sign, no dead zone


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def other(side: str) -> str:
    return {"left": "right", "right": "left"}[side]


# =============================================================================================
# 1. atlas boxes
# =============================================================================================

def _atlas_regions() -> dict:
    return C._atlas()


def atlas_box_bounds() -> tuple:
    """1024-mip voxel box (b0, b1) covering every label in ALL_LABELS plus B1's margin."""
    a = _atlas_regions()["mece2_231218"]
    keys = [str(v) for v in ALL_LABELS.values()]
    lo = np.maximum(np.min([a[k]["bbox_lo"] for k in keys], 0) - BOX_MARGIN_8NM, 0)
    hi = np.max([a[k]["bbox_hi"] for k in keys], 0) + BOX_MARGIN_8NM
    return (np.array(lo) // F1024).astype(int), (np.array(hi) // F1024 + 1).astype(int)


def _fetch(layer: str, mip, b0, b1) -> np.ndarray:
    from cloudvolume import CloudVolume
    import warnings
    warnings.filterwarnings("ignore")
    cv = CloudVolume(C.PUBLIC + layer, mip=mip, use_https=True, progress=False, fill_missing=True)
    return np.asarray(cv[b0[0]:b1[0], b0[1]:b1[1], b0[2]:b1[2]])[..., 0]


def atlas_box() -> dict:
    """mece2 and mece1 labels at the 1024 nm mip over atlas_box_bounds() (cached, uint8)."""
    b0, b1 = atlas_box_bounds()
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"atlas_1024_{'_'.join(map(str, b0))}_{'_'.join(map(str, b1))}.npz"
    if path.exists():
        d = np.load(path)
        M2, M1 = d["M2"], d["M1"]
    else:
        M2 = _fetch("mece2_231218", [1024, 1024, 960], b0, b1)
        M1 = _fetch("mece1_231218", [1024, 1024, 960], b0, b1)
        if M2.max() > 255 or M1.max() > 255:
            raise ValueError("atlas label above 255")
        M2, M1 = M2.astype(np.uint8), M1.astype(np.uint8)
        np.savez_compressed(path, M2=M2, M1=M1, b0=b0)
    return {"M2": M2, "M1": M1, "b0": b0, "b1": b1, "f": F1024, "cache": str(path)}


def check_against_b1_box(box: dict) -> dict:
    """The fetched mece2 box must equal B1's cached hindbrain box wherever they overlap."""
    d = np.load(REF / "mece2_1024_hindbrain.npz")
    Mb, bb = d["M"], d["b0"].astype(int)
    lo = np.maximum(bb, box["b0"])
    hi = np.minimum(bb + np.array(Mb.shape), box["b1"])
    A = box["M2"][lo[0] - box["b0"][0]:hi[0] - box["b0"][0], lo[1] - box["b0"][1]:hi[1] - box["b0"][1],
                  lo[2] - box["b0"][2]:hi[2] - box["b0"][2]]
    B = Mb[lo[0] - bb[0]:hi[0] - bb[0], lo[1] - bb[1]:hi[1] - bb[1], lo[2] - bb[2]:hi[2] - bb[2]]
    return {"b1_box_b0": bb.tolist(), "b1_box_shape": list(Mb.shape), "overlap_voxels": int(A.size),
            "equal_voxels": int((A == B).sum()), "identical": bool(np.array_equal(A, B))}


def _in_box(pts: np.ndarray, b0, shape, f) -> np.ndarray:
    q = np.asarray(pts, float) // f - b0
    return ((q >= 0) & (q < np.array(shape))).all(1)


def region_voxel_centres_um(M: np.ndarray, b0, f, label: int) -> np.ndarray:
    idx = np.argwhere(M == label)
    return (idx + b0 + 0.5) * f * UM_PER_VOXEL


def nearest_region(pts8: np.ndarray, box: dict, labels: Mapping[str, int]) -> tuple:
    """For each point: (name of the nearest region voxel's label, distance um to that voxel centre),
    over the given labels. Also returns the full distance matrix (points x labels)."""
    from scipy.spatial import cKDTree
    p = np.asarray(pts8, float) * UM_PER_VOXEL
    names = list(labels)
    D = np.full((len(p), len(names)), np.inf)
    for j, n in enumerate(names):
        c = region_voxel_centres_um(box["M2"], box["b0"], box["f"], labels[n])
        if len(c):
            D[:, j] = cKDTree(c).query(p)[0]
    k = np.argmin(D, 1)
    return np.array(names)[k], D[np.arange(len(p)), k], D, names


def vote_labels(pts8: np.ndarray, M: np.ndarray, b0, f) -> np.ndarray:
    """B1's rule: plurality mece label over the 3 x 3 x 3 neighbourhood (cells._label_votes)."""
    lab, _ = C._label_votes(M, b0, f, np.asarray(pts8, float))
    return lab


# =============================================================================================
# 2. soma-bearing segments per region and side
# =============================================================================================

def _graph_degrees() -> dict:
    """Input / output synapse counts per segment from the cached pair arrays (brain of record)."""
    from fishbrain import brain as B
    P = B.load_pairs()
    n = len(P["ids"])
    ns = (P["t1"].astype(np.int64) + P["t2"])
    n_in = np.bincount(P["post"], weights=ns, minlength=n).astype(np.int64)
    n_out = np.bincount(P["pre"], weights=ns, minlength=n).astype(np.int64)
    in2 = np.bincount(P["post"], weights=P["t2"], minlength=n).astype(np.int64)
    return {"ids": P["ids"], "n_in": n_in, "n_out": n_out, "in_type2": in2}


def _seg_stats(segs: Sequence[int], deg: dict) -> List[dict]:
    from fishbrain.brain import index_of
    idx = index_of(deg["ids"], np.asarray(segs, np.uint64)) if len(segs) else np.array([], np.int64)
    out = []
    for s, i in zip(segs, idx):
        if i < 0:
            out.append({"in_graph": False, "n_in": 0, "n_out": 0, "in_type2": 0})
        else:
            out.append({"in_graph": True, "n_in": int(deg["n_in"][i]), "n_out": int(deg["n_out"][i]),
                        "in_type2": int(deg["in_type2"][i])})
    return out


def build_populations(box: dict, somas, agg, axes, deg: dict, dilate_um: Optional[float] = None) -> tuple:
    """Assign every CAVE soma in the atlas box to a region and side; keep soma-bearing segments.

    dilate_um None: strict (B1's plurality vote). dilate_um r: the strict label if it is one of
    ALL_LABELS; else, for a soma in no atlas region at all (vote 0), the nearest ALL_LABELS region
    whose voxel centre lies within r um. Somas inside any other atlas region are never reassigned.
    Returns (populations dict, per-soma table, stats)."""
    import pandas as pd
    M2, b0, f = box["M2"], box["b0"], box["f"]
    P = somas[["id", "x", "y", "z"]].values.astype(float)
    inb = _in_box(P[:, 1:], b0, M2.shape, f)
    S = somas[inb].copy()
    pts = S[["x", "y", "z"]].values.astype(float)
    S["label"] = vote_labels(pts, M2, b0, f)
    if dilate_um is not None:
        near, dist, _, _ = nearest_region(pts, box, ALL_LABELS)
        near_lab = np.array([ALL_LABELS[n] for n in near])
        strict = S["label"].values
        S["label"] = np.where(np.isin(strict, list(ALL_LABELS.values())), strict,
                              np.where((strict == 0) & (dist <= dilate_um), near_lab, strict))
    S["rhombomere"] = vote_labels(pts, box["M1"], b0, f)
    S["side"] = C.side_of(pts, axes)
    S["lat_um"] = np.abs(S.y.values - C.midline_y(S.x.values, axes)) * UM_PER_VOXEL[1]
    S["seg_id"] = S.id.map(lambda i: int(agg.get(int(i), 0))).astype(np.int64)
    all_counts = agg[agg > 0].value_counts()
    stats = {"somas_in_box": int(len(S))}
    pops, dropped = {}, {}
    for group, regions in POPULATIONS.items():
        for name, lab in regions.items():
            R = S[S.label == lab]
            d = {"atlas_somas": int(len(R)), "no_segment": int((R.seg_id == 0).sum()),
                 "side_ambiguous_lt_3um": int((R.lat_um < SIDE_AMBIGUOUS_UM).sum()), "merged_dropped": 0}
            R = R[(R.seg_id > 0) & (R.lat_um >= SIDE_AMBIGUOUS_UM)]
            for side in ("left", "right"):
                G = R[R.side == side]
                members = []
                for seg, g in G.groupby("seg_id"):
                    n_all = int(all_counts.get(seg, 0))
                    if n_all > len(g):          # the segment also holds a soma outside this region/side
                        d["merged_dropped"] += len(g)
                        continue
                    p = g[["x", "y", "z"]].values.mean(0)
                    members.append({"seg_id": int(seg), "lore_ids": [int(i) for i in g.id],
                                    "pos": [int(round(t)) for t in p],
                                    "lat_um": round(float(g.lat_um.mean()), 1),
                                    "rhombomere": RHOMBOMERE.get(int(g.rhombomere.mode().iloc[0]),
                                                                 str(int(g.rhombomere.mode().iloc[0])))})
                for m, st in zip(members, _seg_stats([m["seg_id"] for m in members], deg)):
                    m.update(st)
                members.sort(key=lambda m: (m["pos"][0], m["pos"][1], m["pos"][2]))
                pops.setdefault(group, {}).setdefault(name, {})[side] = members
            dropped[name] = d
    # nIII: dorsal half per side = the IR/MR proxy (Greaney 2017: IR/MR predominantly dorsal nIII)
    for side in ("left", "right"):
        mem = pops["strike"]["nIII"][side]
        z = np.array([m["pos"][2] for m in mem], float)
        order = np.argsort(z, kind="stable")
        k = int(math.ceil(NIII_DORSAL_FRACTION * len(mem)))
        dorsal = set(order[:k].tolist())
        for i, m in enumerate(mem):
            m["nIII_dorsal"] = i in dorsal
        stats[f"nIII_dorsal_{side}_z_cut_8nm"] = float(z[order[k - 1]]) if k else None
    stats["per_region"] = dropped
    return pops, S, stats


def population_ids(pops: dict, group: str, name: str, side: str, only=None) -> List[int]:
    mem = pops[group][name][side]
    return [m["seg_id"] for m in mem if (only is None or m.get(only))]


def symmetry(pops: dict) -> dict:
    """Left/right member counts per region with a two-sided binomial test against 50:50."""
    from scipy.stats import binomtest
    out = {}
    for group, regions in pops.items():
        for name, sides in regions.items():
            L, R = len(sides["left"]), len(sides["right"])
            p = binomtest(L, L + R, 0.5).pvalue if L + R else None
            out[name] = {"left": L, "right": R, "asymmetry_(L-R)/(L+R)": round((L - R) / (L + R), 3) if L + R else None,
                         "binomial_p": None if p is None else round(float(p), 4),
                         "left_with_inputs": sum(1 for m in sides["left"] if m["n_in"] > 0),
                         "right_with_inputs": sum(1 for m in sides["right"] if m["n_in"] > 0)}
    return out


def atlas_region_symmetry(box: dict, axes: dict) -> dict:
    """Voxel counts of each used atlas region on each side of the fitted midline (1024 mip)."""
    out = {}
    for name, lab in {**REGION_LABELS}.items():
        idx = np.argwhere(box["M2"] == lab)
        c8 = (idx + box["b0"] + 0.5) * box["f"]
        s = C.side_of(c8, axes)
        L, R = int((s == "left").sum()), int((s == "right").sum())
        out[name] = {"left_voxels": L, "right_voxels": R,
                     "asymmetry": round((L - R) / (L + R), 3) if L + R else None}
    return out


# =============================================================================================
# 3. validation against the named HMI SPNs (positive control) and B1's mirror picks
# =============================================================================================

DILATION_UM = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0)
PERMUTATIONS = 10000
PERMUTATION_SEED = "G1b-spn-name-permutation"


def _rng(seed: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(hashlib.sha256(seed.encode()).hexdigest(), 16)))


def mirror_points(pts8: np.ndarray, axes: dict) -> np.ndarray:
    """Reflect points across the fitted midline (y' = 2 y_mid(x) - y)."""
    p = np.array(pts8, float)
    p[:, 1] = 2 * C.midline_y(p[:, 0], axes) - p[:, 1]
    return p


def _label_names(labels: np.ndarray) -> List[str]:
    return [LABEL_NAME.get(int(l), "unlabelled" if int(l) == 0 else f"other:{int(l)}") for l in labels]


def validate_spn(box: dict, cells: dict, pops: dict) -> dict:
    """Named HMI SPNs as the positive control for the atlas populations."""
    axes = cells["axes"]
    spn = cells["spn"]
    pos = np.array([s["pos"] for s in spn], float)
    names = np.array([s["name"] for s in spn])
    M2, b0, f = box["M2"], box["b0"], box["f"]

    def score(points):
        lab = _label_names(vote_labels(points, M2, b0, f))
        near, dist, D, cols = nearest_region(points, box, RS_LABELS)
        return np.array(lab), near, dist

    lab, near, dist = score(pos)
    in_rs = np.isin(lab, list(RS_LABELS))
    own = lab == names
    b1_labels = [s["atlas_region"] for s in spn]
    b1_short = [r.split("/")[-1] if r != "unlabelled" else "unlabelled" for r in b1_labels]
    per_cell = []
    for i, s in enumerate(spn):
        per_cell.append({"lore_id": s["lore_id"], "seg_id": s["seg_id"], "name": s["name"], "side": s["side"],
                         "class": s["class"], "prox": s["prox"], "strict_region": lab[i],
                         "b1_region": b1_short[i], "nearest_rs_region": str(near[i]),
                         "nearest_rs_distance_um": round(float(dist[i]), 2)})
    # the strict vote must reproduce B1's labels exactly (same box on the overlap, same rule)
    b1_agree = int(sum(a == (b if b in RS_LABELS or b == "unlabelled" else b) for a, b in zip(lab, b1_short)))
    # overlap of named cells with the atlas populations (same name, same side)
    seg_in_pop = []
    for s, l in zip(spn, lab):
        grp = next((g for g, r in POPULATIONS.items() if s["name"] in r), None)
        ids = {m["seg_id"] for m in pops[grp][s["name"]][s["side"]]} if grp else set()
        seg_in_pop.append(s["seg_id"] in ids)
    seg_in_pop = np.array(seg_in_pop)
    turning = np.array([s["class"] == "turning" for s in spn])
    right = np.array([s["side"] == "right" for s in spn])
    turn_pop_right = {m["seg_id"] for n in POPULATIONS["turning"] for m in pops["turning"][n]["right"]}
    # name-permutation null: shuffle the 50 names, count own-name matches
    rng = _rng(PERMUTATION_SEED)
    null_strict, null_near = [], []
    for _ in range(PERMUTATIONS):
        perm = rng.permutation(len(names))
        null_strict.append(int((lab == names[perm]).sum()))
        null_near.append(int((near == names[perm]).sum()))
    null_strict, null_near = np.array(null_strict), np.array(null_near)
    own_near = int((near == names).sum())
    # dilation curve (sensitivity only; the populations use the strict vote)
    curve = []
    for r in DILATION_UM:
        w = dist <= r
        curve.append({"r_um": r, "within_r": int(w.sum()), "own_name": int((w & (near == names)).sum())})
    # mirror test: reflect each named cell to the other side and label it again
    mpos = mirror_points(pos, axes)
    mlab, mnear, mdist = score(mpos)
    mside = C.side_of(mpos, axes)
    m_in_rs = np.isin(mlab, list(RS_LABELS))
    mirror_curve = []
    for r in DILATION_UM:
        w = mdist <= r
        mirror_curve.append({"r_um": r, "within_r": int(w.sum()), "own_name": int((w & (mnear == names)).sum())})
    for i in range(len(spn)):
        per_cell[i].update({"mirror_side": str(mside[i]), "mirror_strict_region": mlab[i],
                            "mirror_nearest_rs_region": str(mnear[i]),
                            "mirror_nearest_rs_distance_um": round(float(mdist[i]), 2)})
    return {
        "n_named": len(spn),
        "strict": {"in_any_rs_region": int(in_rs.sum()), "own_named_region": int(own.sum()),
                   "mismatches": [{"lore_id": spn[i]["lore_id"], "name": names[i], "region": lab[i]}
                                  for i in np.flatnonzero(in_rs & ~own)],
                   "agrees_with_b1_label": b1_agree},
        "named_cell_in_own_population": {"all": int(seg_in_pop.sum()),
                                         "turning_right": int((seg_in_pop & turning & right).sum()),
                                         "turning_right_named": int((turning & right).sum()),
                                         "right_turning_population_size": len(turn_pop_right)},
        "permutation_null": {"permutations": PERMUTATIONS, "seed": PERMUTATION_SEED,
                             "strict_mean": round(float(null_strict.mean()), 3),
                             "strict_max": int(null_strict.max()),
                             "strict_p_ge_observed": float((null_strict >= own.sum()).mean()),
                             "nearest_observed": own_near,
                             "nearest_mean": round(float(null_near.mean()), 3),
                             "nearest_max": int(null_near.max()),
                             "nearest_p_ge_observed": float((null_near >= own_near).mean())},
        "nearest_region_curve": curve,
        "mirror": {"lands_on_other_side": int((mside != np.array([s["side"] for s in spn])).sum()),
                   "strict_in_any_rs_region": int(m_in_rs.sum()),
                   "strict_own_named_region": int((mlab == names).sum()),
                   "nearest_own_name_any_distance": int((mnear == names).sum()),
                   "nearest_region_curve": mirror_curve},
        "per_cell": per_cell,
    }


def compare_mirror_picks(cells: dict, pops: dict) -> dict:
    """Where B1's unconfirmed other-side mirror picks fall relative to the atlas populations."""
    all_pop = {}
    for grp, regions in pops.items():
        for name, sides in regions.items():
            for side, mem in sides.items():
                for m in mem:
                    all_pop[m["seg_id"]] = (name, side)
    rows, n_any, n_own, n_turn, n_turn_own = [], 0, 0, 0, 0
    for m in cells["spn_mirror_candidates"]:
        if not m.get("seg_id"):
            continue
        hit = all_pop.get(int(m["seg_id"]))
        own = bool(hit and hit[0] == m["name"] and hit[1] == m["side"])
        n_any += bool(hit)
        n_own += own
        if m["class"] == "turning" and m["side"] == "left":
            n_turn += 1
            n_turn_own += own
        rows.append({"of_lore_id": m["of_lore_id"], "seg_id": int(m["seg_id"]), "name": m["name"], "side": m["side"],
                     "in_atlas_population": None if not hit else f"{hit[0]}/{hit[1]}", "own_name": own})
    return {"mirror_picks_with_segment": len(rows), "in_any_atlas_population": n_any,
            "in_own_named_population_same_side": n_own,
            "left_turning_picks": n_turn, "left_turning_picks_in_own_population": n_turn_own, "rows": rows}


# =============================================================================================
# 4. the M-system: MiD2cm (r5) and MiD3cm (r6), found the way B1 found Mauthner
# =============================================================================================

M_HOMOLOGS = {"MiD2cm": ("MiD2", 38), "MiD3cm": ("MiD3", 55)}
M_TOP_K = 5
SOMA_VOXEL_UM3 = 0.512 * 0.512 * 0.060


def soma_box(label: int) -> dict:
    """lores_cbs_231218 somas plus mece2 / mece1 labels at 512 x 512 x 60 nm over the atlas region's
    bbox (4096-mip, cells._atlas) plus a 20 um margin, cached in CACHE."""
    a = _atlas_regions()["mece2_231218"][str(label)]
    margin = SOMA_BOX_MARGIN_UM / UM_PER_VOXEL
    b0 = ((np.array(a["bbox_lo"]) - margin) / F512).astype(int)
    b1 = ((np.array(a["bbox_hi"]) + margin) / F512).astype(int)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"somabox_512x512x60_label{label}_{'_'.join(map(str, b0))}.npz"
    if path.exists():
        d = np.load(path)
        return {"L": d["L"], "M2": d["M2"], "M1": d["M1"], "b0": d["b0"], "f": F512, "cache": str(path)}
    L = _fetch("lores_cbs_231218", [512, 512, 60], b0, b1)
    M2 = _fetch("mece2_231218", [512, 512, 60], b0, b1).astype(np.uint8)
    M1 = _fetch("mece1_231218", [512, 512, 60], b0, b1).astype(np.uint8)
    np.savez_compressed(path, L=L, M2=M2, M1=M1, b0=b0)
    return {"L": L, "M2": M2, "M1": M1, "b0": b0, "f": F512, "cache": str(path)}


def b1_r4_box() -> dict:
    """B1's cached r4 box (lores_cbs_231218 + mece2 at 512 x 512 x 60 nm), read-only."""
    d = np.load(REF / "r4_box_512x512x60.npz")
    return {"L": d["L"], "M2": d["M"], "M1": None, "b0": d["b0"].astype(int), "f": F512,
            "cache": str(REF / "r4_box_512x512x60.npz")}


def soma_table(box: dict, label: int, axes: dict):
    """Per soma in the box: volume, centroid (8 nm), clipped flag, fraction of voxels in `label`,
    plurality mece1 rhombomere, side."""
    import pandas as pd
    L = box["L"]
    ids, inv = np.unique(L.ravel(), return_inverse=True)
    inv = inv.ravel()
    n = len(ids)
    cnt = np.bincount(inv, minlength=n)
    cen = np.empty((n, 3))
    for k in range(3):   # one axis at a time: coordinate of every voxel along axis k
        shape = [1, 1, 1]
        shape[k] = L.shape[k]
        coord = np.broadcast_to(np.arange(L.shape[k], dtype=np.float32).reshape(shape), L.shape).ravel()
        cen[:, k] = np.bincount(inv, weights=coord, minlength=n) / cnt
        del coord
    # a soma is clipped when any of its voxels lies on a face of the box (same as B1's min/max test)
    inv3 = inv.reshape(L.shape)
    on_face = np.zeros(n, bool)
    for face in (inv3[0], inv3[-1], inv3[:, 0], inv3[:, -1], inv3[:, :, 0], inv3[:, :, -1]):
        on_face[np.unique(face)] = True
    frac = np.bincount(inv, weights=(box["M2"] == label).ravel(), minlength=n) / cnt
    c8 = (cen + box["b0"] + 0.5) * box["f"]
    df = pd.DataFrame({"lore": ids.astype(np.int64), "vox": cnt, "vol_um3": cnt * SOMA_VOXEL_UM3,
                       "x": c8[:, 0], "y": c8[:, 1], "z": c8[:, 2],
                       "clipped": on_face, "frac_in_region": frac})
    if box.get("M1") is not None:
        m1 = box["M1"].ravel().astype(np.int64)
        key = inv.astype(np.int64) * 256 + m1
        uk, uc = np.unique(key, return_counts=True)
        best = {}
        for k, c in zip(uk.tolist(), uc.tolist()):
            s, lab = divmod(k, 256)
            if lab and (s not in best or c > best[s][1]):
                best[s] = (lab, c)
        df["rhombomere"] = [RHOMBOMERE.get(best[i][0], str(best[i][0])) if i in best else "unlabelled"
                            for i in range(n)]
    else:
        df["rhombomere"] = None
    df = df[(df.lore > 0) & ~df.clipped].copy()
    df["side"] = C.side_of(df[["x", "y", "z"]].values, axes)
    df["lat_um"] = np.abs(df.y.values - C.midline_y(df.x.values, axes)) * UM_PER_VOXEL[1]
    return df


def mesh_features(seg: int, soma8: Sequence[float], side: str, axes: dict) -> dict:
    """Agglomerated mesh (lod 3) of one segment: reach from the soma and midline crossing."""
    from cloudvolume import CloudVolume
    import warnings
    warnings.filterwarnings("ignore")
    cv = CloudVolume(C.PUBLIC + "seg_241003_agg241003", mip=0, use_https=True, progress=False)
    m = cv.mesh.get(int(seg), lod=3)
    m = m[int(seg)] if isinstance(m, dict) else m
    v = m.vertices / np.array(C.VOXEL_NM)
    if not len(v):
        return {"n_vertices_lod3": 0}
    sp = np.asarray(soma8, float)
    ymid = C.midline_y(v[:, 0], axes)
    own = 1 if side == "left" else -1
    past = np.sign(v[:, 1] - ymid) != own
    lateral = (v[:, 1] - sp[1]) * own            # > 0 = away from the midline
    return {"n_vertices_lod3": int(len(v)),
            "lateral_reach_um": round(float(max(lateral.max(), 0) * UM_PER_VOXEL[1]), 1),
            "medial_reach_um": round(float(max(-lateral.min(), 0) * UM_PER_VOXEL[1]), 1),
            "ventral_reach_um": round(float((v[:, 2].max() - sp[2]) * UM_PER_VOXEL[2]), 1),
            "posterior_reach_um": round(float((v[:, 0].max() - sp[0]) * UM_PER_VOXEL[0]), 1),
            "anterior_reach_um": round(float((sp[0] - v[:, 0].min()) * UM_PER_VOXEL[0]), 1),
            "fraction_vertices_past_midline": round(float(past.mean()), 4),
            "max_past_midline_um": round(float(np.abs(v[past, 1] - ymid[past]).max() * UM_PER_VOXEL[1]), 1)
            if past.any() else 0.0}


def identify_largest(box: dict, label: int, axes: dict, agg, deg: dict, somas, require_crossing: bool,
                     top_k: int = M_TOP_K, with_mesh: bool = True) -> dict:
    """The selection rule written before the r5/r6 boxes were read (evidence note, s.4):
    rank unclipped somas overlapping the atlas region by volume per side; pick the largest whose
    mesh crosses the midline (if require_crossing), else the largest."""
    df = soma_table(box, label, axes)
    p99, med = float(np.percentile(df.vol_um3, 99)), float(np.median(df.vol_um3))
    so = somas.set_index("id")
    out = {"method": {"box_b0_512": [int(t) for t in box["b0"]], "box_shape": list(box["L"].shape),
                      "cache": box["cache"], "somas_unclipped": int(len(df)),
                      "median_soma_um3": round(med, 1), "p99_soma_um3": round(p99, 1),
                      "rule": "largest unclipped soma overlapping the region"
                              + (", whose mesh crosses the midline" if require_crossing else "")}}
    for side in ("left", "right"):
        d = df[(df.side == side) & (df.frac_in_region > 0)].sort_values("vox", ascending=False).head(top_k)
        cands = []
        for _, r in d.iterrows():
            seg = int(agg.get(int(r.lore), 0))
            st = _seg_stats([seg], deg)[0] if seg else {"in_graph": False, "n_in": 0, "n_out": 0, "in_type2": 0}
            cp = so.loc[int(r.lore), ["x", "y", "z"]].values.astype(float) if int(r.lore) in so.index \
                else np.array([r.x, r.y, r.z])
            c = {"lore_id": int(r.lore), "seg_id": seg, "soma_volume_um3": round(float(r.vol_um3), 1),
                 "frac_in_region": round(float(r.frac_in_region), 2), "rhombomere": r.rhombomere,
                 "lat_um": round(float(r.lat_um), 1), "pos": [int(round(t)) for t in cp], **st}
            if with_mesh and seg:
                try:
                    c["mesh"] = mesh_features(seg, cp, side, axes)
                except Exception as e:   # a failed mesh read is recorded, never silently skipped
                    c["mesh"] = {"error": type(e).__name__}
            cands.append(c)
        pick, flag = None, None
        for c in cands:
            crosses = c.get("mesh", {}).get("fraction_vertices_past_midline", 0) > 0
            if not require_crossing or crosses:
                pick = c
                break
        if pick is None and cands:
            pick, flag = cands[0], "crossing not seen in the top candidates"
        if pick is None:
            out[side] = {"seg_id": None, "flag": "no candidate"}
            continue
        others = [c for c in cands if c is not pick]
        out[side] = {
            "seg_id": pick["seg_id"], "lore_id": pick["lore_id"], "pos": pick["pos"],
            "soma_volume_um3": pick["soma_volume_um3"], "flag": flag,
            "rank_by_volume_among_overlapping": 1 + cands.index(pick),
            "evidence": {
                "volume_ratio_to_next_candidate": round(pick["soma_volume_um3"] / max(1e-9, max(
                    [c["soma_volume_um3"] for c in others] or [1e-9])), 2) if others else None,
                "volume_ratio_to_box_p99": round(pick["soma_volume_um3"] / p99, 2),
                "frac_in_region": pick["frac_in_region"], "rhombomere": pick["rhombomere"],
                "lat_um": pick["lat_um"], "input_synapses": pick["n_in"], "output_synapses": pick["n_out"],
                "input_rank_among_candidates": 1 + sorted([c["n_in"] for c in cands], reverse=True).index(pick["n_in"]),
                "mesh": pick.get("mesh")},
            "candidates": cands,
        }
    return out


def mirror_consistency(ident: dict, axes: dict) -> Optional[dict]:
    """Distance (um) between the left pick and the mirror image of the right pick."""
    if not (ident.get("left", {}).get("seg_id") and ident.get("right", {}).get("seg_id")):
        return None
    l = np.array(ident["left"]["pos"], float)
    r = mirror_points(np.array([ident["right"]["pos"]], float), axes)[0]
    d = (l - r) * UM_PER_VOXEL
    return {"distance_um": round(float(np.sqrt((d ** 2).sum())), 1),
            "dx_um": round(float(d[0]), 1), "dy_um": round(float(d[1]), 1), "dz_um": round(float(d[2]), 1)}


def shared_afferent_evidence(m_ident: Mapping[str, dict], mauthner: Mapping[str, int], axes: dict) -> dict:
    """Post-hoc check (written after the candidate table was seen): M-series cells share VIIIth-nerve
    input with the M-cell (Nakayama & Oda 2004). For every candidate: input synapses more than 50 um
    lateral of the midline, and synapses from presynaptic segments that also contact the same-side
    Mauthner more than 50 um laterally (B1's lateral-dendrite zone), raw and as a fraction."""
    import pyarrow.parquet as pq
    segs = sorted({int(c["seg_id"]) for h in m_ident.values() for s in ("left", "right")
                   for c in h[s]["candidates"] if c["seg_id"]} | {int(v) for v in mauthner.values()})
    d = pq.read_table(DATA / "synapses.parquet", columns=["pre", "post", "x", "y"],
                      filters=[("post", "in", segs)]).to_pandas()
    d["lat_um"] = (d.y.values - C.midline_y(d.x.values, axes)) * UM_PER_VOXEL[1]
    sgn = {"left": 1, "right": -1}
    aff = {}
    for s in ("left", "right"):
        m = d[(d.post == mauthner[s]) & (d.lat_um * sgn[s] > 50)]
        aff[s] = set(m.pre.astype(np.int64).tolist())
    out = {"mauthner_far_lateral_presynaptic_segments": {s: len(aff[s]) for s in aff}}
    for h, res in m_ident.items():
        for s in ("left", "right"):
            rows = []
            for c in res[s]["candidates"]:
                x = d[d.post == c["seg_id"]]
                lat = int((x.lat_um * sgn[s] > 50).sum())
                pre = x.pre.astype(np.int64)
                shared = int(pre.isin(aff[s]).sum())
                rows.append({"lore_id": c["lore_id"], "seg_id": c["seg_id"], "soma_volume_um3": c["soma_volume_um3"],
                             "n_in": c["n_in"], "inputs_gt50um_lateral": lat,
                             "synapses_from_same_side_M_lateral_afferents": shared,
                             "shared_fraction_of_lateral_inputs": round(shared / lat, 3) if lat else None,
                             "synapses_from_other_side_M_lateral_afferents": int(pre.isin(aff[other(s)]).sum())})
            out[f"{h}_{s}"] = rows
    return out


def choose_m_homologs(m_ident: Mapping[str, dict], axes: dict) -> dict:
    """Two picks per homolog and side, both reported:
    task_criteria: largest soma overlapping the atlas region (the task's region + size + input rule,
        as for Mauthner; identify_largest with require_crossing=False gives the same cell here);
    preregistered_crossing: the rule written before the boxes were read (largest that crosses)."""
    out = {}
    for h, res in m_ident.items():
        row = {}
        for s in ("left", "right"):
            cands = res[s]["candidates"]
            size_pick = max(cands, key=lambda c: c["soma_volume_um3"]) if cands else None
            row[s] = {"task_criteria": None if size_pick is None else size_pick["seg_id"],
                      "task_criteria_lore_id": None if size_pick is None else size_pick["lore_id"],
                      "task_criteria_input_rank": None if size_pick is None else
                      1 + sorted([c["n_in"] for c in cands], reverse=True).index(size_pick["n_in"]),
                      "preregistered_crossing": res[s]["seg_id"], "preregistered_crossing_lore_id": res[s].get("lore_id"),
                      "agree": bool(size_pick and size_pick["seg_id"] == res[s]["seg_id"])}
        # bilateral pairing of each pick set
        for key in ("task_criteria", "preregistered_crossing"):
            pos = {}
            for s in ("left", "right"):
                c = next((c for c in res[s]["candidates"] if c["seg_id"] == row[s][key]), None)
                pos[s] = None if c is None else c["pos"]
            row[f"{key}_mirror_distance"] = mirror_consistency({s: {"seg_id": row[s][key], "pos": pos[s]}
                                                                for s in ("left", "right")}, axes)
        out[h] = row
    return out


# =============================================================================================
# 5. decision rules (pure functions; thresholds fixed in STRIKE_RULE / ESCAPE_RULE / TURN_RULE)
# =============================================================================================

def pool_stats(spikes: Sequence) -> dict:
    """Per-cell spike counts in the decision window -> size, fraction of cells active, mean spikes/cell."""
    a = np.asarray(spikes, float).ravel()
    if a.size == 0:
        return {"n": 0, "active_frac": 0.0, "mean_spikes": 0.0}
    return {"n": int(a.size), "active_frac": float((a > 0).mean()), "mean_spikes": float(a.mean())}


def side_index(right_mean: float, left_mean: float) -> float:
    """(R - L) / (R + L); 0 when both are 0. Positive = right."""
    s = right_mean + left_mean
    return 0.0 if s <= 0 else (right_mean - left_mean) / s


def strike_decision(spikes: Mapping[str, Sequence], rule: Mapping = STRIKE_RULE) -> dict:
    """The strike (buy) readout.

    spikes: per-cell spike counts in one decision window for "nMLF_left", "nMLF_right",
    "nIII_dorsal_left", "nIII_dorsal_right" (cell order does not matter).
      1. Convergence: both eyes rotate nasally, so BOTH medial-rectus pools must be active. MR
         motoneurons are in dorsal nIII and drive the ipsilateral eye (Greaney 2017). Gate:
         min(active_frac(nIII_dorsal_left), active_frac(nIII_dorsal_right)) >= conv_min_active_frac.
      2. Drive: max(active_frac(nMLF_left), active_frac(nMLF_right)) >= drive_min_active_frac.
      3. Side: nMLF on one side deflects the tail to that same side (Thiele et al. 2014, unilateral
         optogenetic stimulation; unilateral ablation biases the tail to the intact side), so
         SI = (mean nMLF_right - mean nMLF_left) / (sum). SI >= side_index_min -> "right";
         SI <= -side_index_min -> "left"; otherwise "ahead".
    Strike = 1 and 2. Returns {"strike", "side" (None if no strike), "trade" ("buy" or None), parts}.
    """
    nl, nr = pool_stats(spikes["nMLF_left"]), pool_stats(spikes["nMLF_right"])
    el, er = pool_stats(spikes["nIII_dorsal_left"]), pool_stats(spikes["nIII_dorsal_right"])
    conv = min(el["active_frac"], er["active_frac"]) >= rule["conv_min_active_frac"]
    drive = max(nl["active_frac"], nr["active_frac"]) >= rule["drive_min_active_frac"]
    si = side_index(nr["mean_spikes"], nl["mean_spikes"])
    strike = bool(conv and drive)
    side = None
    if strike:
        side = "right" if si >= rule["side_index_min"] else "left" if si <= -rule["side_index_min"] else "ahead"
    return {"strike": strike, "side": side, "trade": "buy" if strike else None, "converged": bool(conv),
            "drive": bool(drive), "side_index": si, "nMLF": {"left": nl, "right": nr},
            "nIII_dorsal": {"left": el, "right": er}}


def escape_decision(first_spike_ms: Mapping[str, Optional[float]], spikes: Mapping[str, int],
                    rule: Mapping = ESCAPE_RULE) -> dict:
    """The escape (sell) readout over the M-series.

    first_spike_ms / spikes: keyed "<cell>_<side>" for cell in ESCAPE_RULE["cells"] (Mauthner,
    MiD2cm, MiD3cm), side left/right; first spike time in the window (None = silent) and count.
    Escape = any M-series cell reaches min_spikes. The initiating cell is the earliest first spike;
    an M-series cell's axon crosses the midline, so the body bends away from the cell's side and the
    escape goes to the OTHER side (Dunn et al. 2016: right-field looms evoke escapes to the left).
    Ties in time go to the side with more M-series spikes; a full tie gives side None."""
    fired = {k: v for k, v in first_spike_ms.items() if v is not None and spikes.get(k, 0) >= rule["min_spikes"]}
    if not fired:
        return {"escape": False, "trade": None, "initiator": None, "cell_side": None, "escape_to": None}
    t0 = min(fired.values())
    first = sorted(k for k, v in fired.items() if v == t0)
    sides = {k.rsplit("_", 1)[1] for k in first}
    if len(sides) == 1:
        cell_side = sides.pop()
    else:
        tot = {s: sum(spikes.get(f"{c}_{s}", 0) for c in rule["cells"]) for s in ("left", "right")}
        cell_side = None if tot["left"] == tot["right"] else max(tot, key=tot.get)
    return {"escape": True, "trade": "sell", "initiator": first[0] if len(first) == 1 else first,
            "first_spike_ms": t0, "cell_side": cell_side, "escape_to": None if cell_side is None else other(cell_side)}


def turn_index(spikes: Mapping[str, Sequence], rule: Mapping = TURN_RULE) -> dict:
    """G1's turn index over the atlas turning populations (RoV3 + MiV1 + MiV2 per side):
    TI = (mean right - mean left) / (sum), mean spikes per cell. TI > 0 = right turn.
    These cells steer phototaxis / OMR turns (Huang et al. 2013), NOT hunting J-turns (Lau et al. 2025)."""
    l, r = pool_stats(spikes["turning_left"]), pool_stats(spikes["turning_right"])
    ti = side_index(r["mean_spikes"], l["mean_spikes"])
    turn = None if (l["mean_spikes"] + r["mean_spikes"]) == 0 else \
        ("right" if ti > rule["side_index_min"] else "left" if ti < -rule["side_index_min"] else None)
    return {"turn_index": ti, "turn": turn, "left": l, "right": r}


# =============================================================================================
# 6. readout sets (segment ids) and the wiring-only checks
# =============================================================================================

def readout_sets(ro: dict, population_set: str = "strict", m_pick: str = "task_criteria") -> Dict[str, List[int]]:
    """Segment ids per readout key, from a readouts.json dict. Keys: nMLF_*, nIII_dorsal_*, nIII_*,
    turning_*, forward_*, Mauthner_*, MiD2cm_*, MiD3cm_*, for * in left/right."""
    pops = ro["populations"][population_set]
    out = {}
    for s in ("left", "right"):
        out[f"nMLF_{s}"] = [m["seg_id"] for m in pops["strike"]["NucMLF"][s]]
        out[f"nIII_{s}"] = [m["seg_id"] for m in pops["strike"]["nIII"][s]]
        out[f"nIII_dorsal_{s}"] = [m["seg_id"] for m in pops["strike"]["nIII"][s] if m["nIII_dorsal"]]
        out[f"turning_{s}"] = [m["seg_id"] for n in POPULATIONS["turning"] for m in pops["turning"][n][s]]
        out[f"forward_{s}"] = [m["seg_id"] for n in POPULATIONS["forward"] for m in pops["forward"][n][s]]
        out[f"Mauthner_{s}"] = [int(ro["m_system"]["Mauthner"][s]["seg_id"])]
        for h in M_HOMOLOGS:
            out[f"{h}_{s}"] = [int(ro["m_system"][h][s][m_pick])]
    return out


WIRING_SHUFFLES = 20
WIRING_SEED = "G1b-wiring-post-permutation"


def wiring_checks(sets: Mapping[str, List[int]], n_shuffles: int = WIRING_SHUFFLES) -> dict:
    """Wiring only, no simulation. For each tectal lobe driven alone (G1's stimulated cells of that
    lobe), and each readout set: excitatory synapses from cells that can fire (reachable from the
    lobe through excitatory pairs, per-synapse sign), per target cell, and the minimum hop count.
    Also direct synapses from ALL of B1's tectal cells of each lobe. Null: the pairs' postsynaptic
    ends permuted (pre, counts and signs kept; seeded; duplicates and self-pairs not repaired),
    n_shuffles times, same stimulated cells."""
    from fishbrain import brain as B
    P = B.load_pairs()
    ident = B.identified()
    ids = P["ids"]
    n = len(ids)
    signed, _ = B.signed_counts(P, B.GATE_SIGN_RULE)
    stim, _ = B.stimulated_set(P, ident)
    tside = dict(zip(ident["tectum"].tolist(), ident["tectum_side"].tolist()))
    stim_side = np.array([tside[int(i)] for i in ids[stim]])
    t_idx = B.index_of(ids, ident["tectum"])
    t_ok = t_idx >= 0
    lobe_of = np.full(n, -1, np.int8)
    lobe_of[t_idx[t_ok]] = np.where(ident["tectum_side"][t_ok] == "left", 0, 1)
    keys = list(sets)
    tgt = np.full(n, -1, np.int32)
    n_cells = {}
    for k, key in enumerate(keys):
        ix = B.index_of(ids, np.array(sets[key], np.uint64))
        ix = ix[ix >= 0]
        if (tgt[ix] >= 0).any():
            raise ValueError(f"readout set {key} overlaps an earlier set; wiring targets must be disjoint")
        tgt[ix] = k
        n_cells[key] = int(len(ix))
    pre = P["pre"].astype(np.int64)
    pos = signed > 0
    nsyn = P["t1"].astype(np.int64) + P["t2"]

    def measure(post: np.ndarray) -> dict:
        ip, ix = B._csr(n, pre[pos], post[pos])
        res = {}
        tp = tgt[post]
        for li, lobe in enumerate(("left", "right")):
            d = B.bfs_levels(n, ip, ix, stim[stim_side == lobe])
            F = d >= 0
            sel = pos & F[pre] & (tp >= 0)
            exc = np.bincount(tp[sel], weights=signed[sel], minlength=len(keys))
            seld = (lobe_of[pre] == li) & (tp >= 0)
            direct = np.bincount(tp[seld], weights=nsyn[seld], minlength=len(keys))
            hops = {}
            for k, key in enumerate(keys):
                dd = d[tgt == k]
                dd = dd[dd >= 0]
                hops[key] = int(dd.min()) if dd.size else None
            res[lobe] = {key: {"exc_from_fireable": int(exc[k]), "per_cell": exc[k] / max(1, n_cells[key]),
                               "direct_from_lobe": int(direct[k]), "min_hops": hops[key]}
                         for k, key in enumerate(keys)}
            res[lobe]["_fireable"] = int(F.sum())
        return res

    post0 = P["post"].astype(np.int64)
    intact = measure(post0)
    rng = _rng(WIRING_SEED)
    null = []
    for i in range(n_shuffles):
        t0 = time.time()
        null.append(measure(post0[rng.permutation(len(post0))]))
        _log("wiring shuffle", i + 1, "of", n_shuffles, round(time.time() - t0, 1), "s")
    bases = sorted({k.rsplit("_", 1)[0] for k in keys})

    def routing(r: dict, base: str) -> dict:
        """Ipsilateral vs contralateral routing, lobe -> same-side target vs other-side target.
        Each target appears once as ipsi and once as contra, so a target's own input bias cancels."""
        pc = {(lobe, s): r[lobe][f"{base}_{s}"]["per_cell"] for lobe in ("left", "right") for s in ("left", "right")}
        dr = {(lobe, s): r[lobe][f"{base}_{s}"]["direct_from_lobe"] for lobe in ("left", "right") for s in ("left", "right")}
        f_ip, f_co = pc[("left", "left")] + pc[("right", "right")], pc[("left", "right")] + pc[("right", "left")]
        d_ip, d_co = dr[("left", "left")] + dr[("right", "right")], dr[("left", "right")] + dr[("right", "left")]
        return {"fireable_ipsi_index": side_index(f_ip, f_co), "direct_ipsi": d_ip, "direct_contra": d_co,
                "direct_ipsi_fraction": (d_ip / (d_ip + d_co)) if (d_ip + d_co) else float("nan")}

    def summary(obs: float, nl: np.ndarray) -> dict:
        return {"observed": round(float(obs), 3), "null_mean": round(float(nl.mean()), 3), "null_sd": round(float(nl.std()), 3),
                "null_p2.5": round(float(np.percentile(nl, 2.5)), 3), "null_p97.5": round(float(np.percentile(nl, 97.5)), 3),
                "p_two_sided": round(float((np.abs(nl - nl.mean()) >= abs(obs - nl.mean()) - 1e-12).mean()), 3)}

    out = {}
    for base in bases:
        o = routing(intact, base)
        nr = [routing(r, base) for r in null]
        row = {"per_lobe": {lobe: {f"{s}_target": {kk: (round(v, 2) if isinstance(v, float) else v)
                                                   for kk, v in intact[lobe][f"{base}_{s}"].items()}
                                   for s in ("left", "right")} for lobe in ("left", "right")},
               "fireable_ipsi_index_(ipsi-contra)/(ipsi+contra)": summary(o["fireable_ipsi_index"],
                                                                          np.array([x["fireable_ipsi_index"] for x in nr])),
               "direct_ipsi_synapses": summary(o["direct_ipsi"], np.array([x["direct_ipsi"] for x in nr], float)),
               "direct_contra_synapses": summary(o["direct_contra"], np.array([x["direct_contra"] for x in nr], float)),
               "direct_ipsi_fraction": None if math.isnan(o["direct_ipsi_fraction"]) else summary(
                   o["direct_ipsi_fraction"], np.array([x["direct_ipsi_fraction"] for x in nr
                                                        if not math.isnan(x["direct_ipsi_fraction"])], float)),
               "direct_binomial_p_vs_half": None if not (o["direct_ipsi"] + o["direct_contra"]) else round(float(
                   __import__("scipy.stats", fromlist=["binomtest"]).binomtest(
                       int(o["direct_ipsi"]), int(o["direct_ipsi"] + o["direct_contra"]), 0.5).pvalue), 4)}
        for lobe in ("left", "right"):
            row[f"{lobe}_lobe_target_laterality_(R-L)/(R+L)"] = round(side_index(
                intact[lobe][f"{base}_right"]["per_cell"], intact[lobe][f"{base}_left"]["per_cell"]), 3)
        out[base] = row
    return {"n_cells": n_cells, "fireable": {l: intact[l]["_fireable"] for l in ("left", "right")},
            "null_fireable_mean": {l: float(np.mean([r[l]["_fireable"] for r in null])) for l in ("left", "right")},
            "shuffles": n_shuffles, "seed": WIRING_SEED, "routing": out,
            "stimulated": {"left_lobe": int((stim_side == "left").sum()), "right_lobe": int((stim_side == "right").sum())},
            "tectal_segments_by_lobe": {"left": int((lobe_of == 0).sum()), "right": int((lobe_of == 1).sum())}}


# =============================================================================================
# 7. build
# =============================================================================================

SCHEMA = {
    "populations": "{strict|dilated_5um: {group: {region: {left|right: [member]}}}}; member = {seg_id, lore_ids, "
                   "pos (8 nm voxels), lat_um, rhombomere, in_graph, n_in, n_out, in_type2, nIII_dorsal (nIII only)}. "
                   "strict = B1's plurality vote; dilated_5um = post hoc sensitivity set",
    "population_stats": "{strict|dilated_5um: {somas_in_box, per_region: {atlas_somas, no_segment, "
                        "side_ambiguous_lt_3um, merged_dropped}, nIII_dorsal_*_z_cut_8nm}}",
    "symmetry": "{strict|dilated_5um: {region: {left, right, asymmetry, binomial_p, *_with_inputs}}}",
    "atlas_region_symmetry": "{region: {left_voxels, right_voxels, asymmetry}} at the 1024 nm mip",
    "spn_validation": "{strict|dilated_5um: named-SPN positive control, permutation null, mirror test, per_cell}",
    "mirror_picks_vs_atlas": "B1's 49 unconfirmed other-side picks against the atlas populations",
    "m_system": "{Mauthner: {left|right: {seg_id, lore_id}}, MiD2cm|MiD3cm: {left|right: {task_criteria, "
                "preregistered_crossing, ...}, *_mirror_distance, confidence}, identification: full candidate "
                "tables, positive_control_r4, shared_afferents}",
    "lesion_sets": "{task_criteria|preregistered_crossing|union: [seg_id]} Liu & Fetcho 1999 M-series lesion",
    "readout_sets": "{population_set: {key: [seg_id]}} ready for the simulator (see readout_sets())",
    "rules": "STRIKE_RULE, ESCAPE_RULE, TURN_RULE, NIII_DORSAL_FRACTION and their sources",
    "wiring": "wiring-only laterality per lobe with the post-permutation null (wiring_checks())",
    "atlas_box": "b0/b1 (1024 mip voxels), shape, sha256 of the mece2 and mece1 boxes, equality with B1's box",
    "labels": "mece2 labels used, and the reticulospinal/oculomotor labels left unused",
}

M_CONFIDENCE = {   # set after reading the evidence (G1b-readouts.md s.4); not computed
    "MiD2cm": {"left": "moderate", "right": "high"},
    "MiD3cm": {"left": "low (contested)", "right": "moderate"},
}


def build(out_path: os.PathLike = READOUTS_JSON, wiring: bool = True) -> dict:
    t00 = time.time()
    cells = C.load()
    axes = cells["axes"]
    _log("atlas box")
    box = atlas_box()
    b1_check = check_against_b1_box(box)
    if not b1_check["identical"]:
        raise AssertionError("fetched atlas box differs from B1's cached box on the overlap")
    somas = C._somas()
    agg = C._agg_map()
    deg = _graph_degrees()
    _log("populations")
    pops, pstats, sym, val = {}, {}, {}, {}
    for key, r in (("strict", None), ("dilated_5um", 5.0)):
        pops[key], _, pstats[key] = build_populations(box, somas, agg, axes, deg, dilate_um=r)
        sym[key] = symmetry(pops[key])
        val[key] = validate_spn(box, cells, pops[key])
    mirror_cmp = {k: compare_mirror_picks(cells, pops[k]) for k in pops}
    _log("M-system: positive control on B1's r4 box")
    control = identify_largest(b1_r4_box(), 48, axes, agg, deg, somas, require_crossing=True)
    expect = {s: int(cells["mauthner"][s]["seg_id"]) for s in ("left", "right")}
    control_ok = all(control[s]["seg_id"] == expect[s] for s in ("left", "right"))
    m_ident = {}
    for h, (reg, lab) in M_HOMOLOGS.items():
        _log("M-system:", h)
        m_ident[h] = identify_largest(soma_box(lab), lab, axes, agg, deg, somas, require_crossing=True)
    picks = choose_m_homologs(m_ident, axes)
    shared = shared_afferent_evidence(m_ident, expect, axes)
    m_system = {"Mauthner": {s: {"seg_id": expect[s], "lore_id": int(cells["mauthner"][s]["lore_id"]),
                                 "source": "cells_identified.json (B1)"} for s in ("left", "right")}}
    for h in M_HOMOLOGS:
        m_system[h] = {**picks[h], "confidence": M_CONFIDENCE[h]}
    m_system["identification"] = {"candidates": m_ident, "positive_control_r4": control,
                                  "positive_control_passed": control_ok, "shared_afferents": shared}
    lesion = {}
    for key in ("task_criteria", "preregistered_crossing"):
        lesion[key] = sorted({expect["left"], expect["right"]} |
                             {int(picks[h][s][key]) for h in M_HOMOLOGS for s in ("left", "right")})
    lesion["union"] = sorted(set(lesion["task_criteria"]) | set(lesion["preregistered_crossing"]))
    ro = {"populations": pops, "m_system": m_system}
    rsets = {k: readout_sets(ro, k) for k in pops}
    wires = None
    if wiring:
        _log("wiring checks")
        wires = wiring_checks({k: v for k, v in rsets["strict"].items() if not re.match(r"nIII_(left|right)$", k)})
    out = {
        "schema": SCHEMA,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_version": CODE_VERSION,
        "cells_identified_built": cells.get("built"),
        "atlas_box": {"b0_1024": box["b0"].tolist(), "b1_1024": box["b1"].tolist(), "shape": list(box["M2"].shape),
                      "mip_nm": [1024, 1024, 960], "sha256_mece2": _sha_arr(box["M2"]), "sha256_mece1": _sha_arr(box["M1"]),
                      "source": "precomputed://gs://fish1-public/mece2_231218 and mece1_231218 (public, no login)",
                      "equals_b1_hindbrain_box_on_overlap": b1_check},
        "labels": {"used": {g: r for g, r in POPULATIONS.items()}, "unused_present_in_atlas": UNUSED_LABELS},
        "populations": pops,
        "population_stats": pstats,
        "symmetry": sym,
        "atlas_region_symmetry": atlas_region_symmetry(box, axes),
        "spn_validation": val,
        "mirror_picks_vs_atlas": mirror_cmp,
        "m_system": m_system,
        "lesion_sets": lesion,
        "readout_sets": rsets,
        "rules": {"strike": STRIKE_RULE, "escape": {**ESCAPE_RULE, "cells": list(ESCAPE_RULE["cells"])},
                  "turn": TURN_RULE, "nIII_dorsal_fraction": NIII_DORSAL_FRACTION,
                  "strike_text": strike_decision.__doc__, "escape_text": escape_decision.__doc__,
                  "turn_text": turn_index.__doc__},
        "wiring": wires,
        "build_seconds": round(time.time() - t00, 1),
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    _log("wrote", out_path)
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.str_):
        return str(o)
    raise TypeError(type(o))


def load(path: os.PathLike = READOUTS_JSON) -> dict:
    with open(path) as fh:
        return json.load(fh)


if __name__ == "__main__":
    import sys
    build(wiring="--no-wiring" not in sys.argv)
