"""FISHBRAIN G1c measure: what is real before bridging.

Graph and geometry only; nothing is simulated. Pre-registration, results and every choice:
``evidence/G1c-measure.md`` (section 0 was written before any number here was computed).

1. **Fragment origins.** Every segment that synapses onto a deciding readout (the M-system, turning, strike) gets a
   synapse cloud (all its pre and post synapses in ``synapses.parquet``) and atlas labels (mece0/1/2 at 4096 nm). Its
   origin class, first match wins:
   O-a0 tectal cell (B1), O-a1 observed tectal end (>= 2 cloud synapses inside the tectum mask), O-d1 soma elsewhere,
   O-d2 ganglion, O-b tract-only (the cloud runs >= 20 um from the readout toward the tectum), O-d3 other far origin
   (the cloud reaches >= 30 um from its contacts), O-c local or terminal. M-system inputs are split by compartment
   (ventral dendrite, lateral dendrite = VIIIth nerve zone, soma / proximal).
2. **Reach from the retina's endings.** The retinal-afferent set (n_in = 0, >= 80% of outputs in the tectal neuropil,
   no soma, not tectal) replaces the tectal somata as the source of ``structure.measure``, placed retinotopically with
   B1's own ``cells.TectumMap``; the same 20 Null-A shuffles as G1b.
3. **The eye.** What the atlas Retina region holds (synapses, segments, somas) and whether any of it reaches the tectum.
4. **Predicted decision-path share.** Graph estimates (synapse view and walk view, upper and lower bounds) of the share
   of a deciding readout's drive that would travel over measured synapses from the eye input.

Run: ``python -m fishbrain.origins all`` (scan, classify, eye, afferents, then the shuffles and the share), or the
steps one at a time: ``scan``, ``classify``, ``eye``, ``real``, ``shuffles``, ``meshes``, ``share``, ``assemble``.
Outputs go to ``data/g1c/``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
REF = DATA / "ref"
OUT = DATA / "g1c"
SHUFFLE_DIR = OUT / "shuffles"
CODE_VERSION = "g1c-measure-v1"

SIDES = ("left", "right")
OTHER = {"left": "right", "right": "left"}
UM_PER_VOXEL = np.array([0.008, 0.008, 0.030])      # 8 x 8 x 30 nm voxels
ATLAS_F = np.array([512, 512, 128])                  # 8 nm voxels per 4096 nm atlas voxel (x, y), 30 nm per 3840 nm (z)
ATLAS_SAMPLING_UM = (4.096, 4.096, 3.84)

# ---- pre-registered 2026-09-26 03:52 EDT (G1c-measure.md s.0) ----
TECT_LABELS = (17, 18)          # mece2: Stratum Periventriculare, Neuropil
NEUROPIL = 18
M0_RETINA, M0_GANGLIA, M0_HINDBRAIN = 1, 3, 6
M1_TECTUM, M1_CAUDAL_HB = 21, 31
M2_MAUTHNER = 48
MIN_TECT_SYN = 2                # O-a1: cloud synapses inside the tectum mask
PROGRESS_UM = 20.0              # O-b: median d_T of contacts - min d_T of cloud
FAR_UM = 30.0                   # O-d3: max distance of the cloud from its contacts' centroid
EDGE_UM = 8.0                   # flag only: min d_T <= 8 um but fewer than MIN_TECT_SYN synapses inside
VENTRAL_DZ_UM = 20.0            # ventral dendrite: >= 20 um ventral of the soma
LAT_MAUTHNER_UM = 50.0          # lateral dendrite of the Mauthner cells (B1's VIIIth-nerve-zone threshold)
LAT_MID_EXTRA_UM = 10.0         # lateral dendrite of MiD2cm / MiD3cm: soma lat + 10 um
AFF_NEUROPIL_FRAC = 0.8         # afferent: >= 80% of outputs in the tectal neuropil
AFF_NEAR_ZERO = 0.1             # sensitivity set: n_in <= 0.1 n_out
SIDE_AMBIGUOUS_UM = 3.0         # B1's midline rule
SPAN_X_VOX = 67_500             # 540 um: "posterior of the tectum" for the brain-wide spanning count
N_CONTROL = 60
POS_SEED, NEG_SEED = "G1c-pos-control", "G1c-neg-control"
NEG_MIN_INPUTS = 20
MESH_TOP_PER_SIDE = 10
DECIDING = ("m_system", "mauthner", "turning", "strike")
CLASSES = ("O-a0", "O-a1", "O-d1", "O-d2", "O-b", "O-d3", "O-c")
PAIRING = {"m_system": "contra", "mauthner": "contra", "turning": "contra", "strike": "both"}


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else
                  (int(o) if isinstance(o, np.integer) else (float(o) if isinstance(o, np.floating) else str(o))))
    os.replace(tmp, path)


def _rng(seed: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))


def _r(v, nd=6):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), nd)


# =============================================================================================
# 1. atlas and geometry (pure given arrays)
# =============================================================================================

class Atlas:
    """mece0/1/2 at 4096 nm plus the distance (um) to the tectum mask mece2 in TECT_LABELS."""

    def __init__(self, m0: np.ndarray, m1: np.ndarray, m2: np.ndarray):
        from scipy.ndimage import distance_transform_edt
        self.m0, self.m1, self.m2 = (np.asarray(a).astype(np.uint8) for a in (m0, m1, m2))
        self.T = np.isin(self.m2, TECT_LABELS)
        self.dT = distance_transform_edt(~self.T, sampling=ATLAS_SAMPLING_UM).astype(np.float32)
        self.shape = np.array(self.m0.shape)

    @classmethod
    def load(cls) -> "Atlas":
        return cls(*(np.load(REF / f"mece{k}_231218_4096.npy") for k in (0, 1, 2)))

    def vox(self, xyz) -> tuple:
        q = (np.atleast_2d(np.asarray(xyz, float)) // ATLAS_F).astype(np.int64)
        q = np.clip(q, 0, self.shape - 1)
        return q[:, 0], q[:, 1], q[:, 2]

    def lookup(self, xyz) -> Dict[str, np.ndarray]:
        i, j, k = self.vox(xyz)
        return {"m0": self.m0[i, j, k], "m1": self.m1[i, j, k], "m2": self.m2[i, j, k], "dT": self.dT[i, j, k]}


def to_um(xyz) -> np.ndarray:
    return np.atleast_2d(np.asarray(xyz, float)) * UM_PER_VOXEL


def lateral_um(xyz, axes: dict) -> np.ndarray:
    from fishbrain import cells as C
    p = np.atleast_2d(np.asarray(xyz, float))
    return np.abs(p[:, 1] - C.midline_y(p[:, 0], axes)) * UM_PER_VOXEL[1]


def side_code(xyz, axes: dict) -> np.ndarray:
    """1 = fish's left (y above the fitted midline), 0 = right."""
    from fishbrain import cells as C
    p = np.atleast_2d(np.asarray(xyz, float))
    return (p[:, 1] > C.midline_y(p[:, 0], axes)).astype(np.int8)


def compartment(dz_um: np.ndarray, lat_um: np.ndarray, lat_min_um: float) -> np.ndarray:
    """Per input synapse: 'ventral' (dz >= 20 um), else 'lateral' (lat >= lat_min), else 'soma'."""
    dz_um, lat_um = np.asarray(dz_um, float), np.asarray(lat_um, float)
    return np.where(dz_um >= VENTRAL_DZ_UM, "ventral", np.where(lat_um >= lat_min_um, "lateral", "soma"))


def classify_fragment(cloud_dT: np.ndarray, cloud_um: np.ndarray, cloud_side: np.ndarray, cloud_m0: np.ndarray,
                      contact_um: np.ndarray, contact_dT: np.ndarray, is_tectal: bool = False,
                      tectal_side: Optional[int] = None, soma: bool = False) -> dict:
    """Origin class of one fragment relative to one readout (first match wins; thresholds pre-registered).

    cloud_*: every synapse of the fragment (pre or post); contact_*: its synapses onto the readout.
    Sides are 1 = left, 0 = right. Returns cls, lobe (1/0/None), and the geometry used."""
    cloud_dT = np.asarray(cloud_dT, float)
    inside = cloud_dT <= 0.0
    n_T = int(inside.sum())
    min_dT = float(cloud_dT.min()) if len(cloud_dT) else math.inf
    c0 = np.asarray(contact_um, float).reshape(-1, 3).mean(0)
    far = float(np.sqrt(((np.asarray(cloud_um, float).reshape(-1, 3) - c0) ** 2).sum(1)).max()) if len(cloud_dT) else 0.0
    progress = float(np.median(contact_dT)) - min_dT if len(cloud_dT) else 0.0
    far_idx = int(np.argmax(((np.asarray(cloud_um, float).reshape(-1, 3) - c0) ** 2).sum(1))) if len(cloud_dT) else -1
    out = {"n_T": n_T, "min_dT": _r(min_dT, 2), "progress_um": _r(progress, 2), "far_um": _r(far, 2),
           "far_idx": far_idx, "edge": bool(n_T < MIN_TECT_SYN and min_dT <= EDGE_UM), "lobe": None}
    if is_tectal:
        out.update(cls="O-a0", lobe=tectal_side)
    elif n_T >= MIN_TECT_SYN:
        sides = np.asarray(cloud_side)[inside]
        out.update(cls="O-a1", lobe=int(np.mean(sides) >= 0.5))
    elif soma:
        out["cls"] = "O-d1"
    elif int((np.asarray(cloud_m0) == M0_GANGLIA).sum()) >= 2:
        out["cls"] = "O-d2"
    elif progress >= PROGRESS_UM:
        out["cls"] = "O-b"
    elif far >= FAR_UM:
        out["cls"] = "O-d3"
    else:
        out["cls"] = "O-c"
    return out


def afferent_rule(n_in: np.ndarray, n_out: np.ndarray, out_neuropil: np.ndarray, soma: np.ndarray, tectal: np.ndarray,
                  near_zero: bool = False) -> np.ndarray:
    """Retinal-afferent mask: (near) zero inputs, outputs >= 80% in the tectal neuropil, no soma, not tectal."""
    n_in, n_out, out_np = (np.asarray(a, np.int64) for a in (n_in, n_out, out_neuropil))
    quiet = (n_in <= AFF_NEAR_ZERO * n_out) if near_zero else (n_in == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_out > 0, out_np / np.maximum(n_out, 1), 0.0)
    return quiet & (n_out >= 1) & (frac >= AFF_NEUROPIL_FRAC) & ~np.asarray(soma, bool) & ~np.asarray(tectal, bool)


# =============================================================================================
# 2. shared inputs: pairs, identities, somas
# =============================================================================================

class World:
    """Everything the steps share, loaded once."""

    def __init__(self):
        from fishbrain import brain as B
        from fishbrain import cells as C
        from fishbrain import structure as S
        self.B, self.C, self.S = B, C, S
        t0 = time.time()
        self.P = B.load_pairs()
        self.ids = self.P["ids"]
        self.n = len(self.ids)
        self.cells = C.load()
        self.axes = self.cells["axes"]
        self.atlas = Atlas.load()
        tm = C.tectum_map(self.cells)
        tix = B.index_of(self.ids, tm.seg_ids)
        self.tect_side = np.full(self.n, -1, np.int8)
        ok = tix >= 0
        self.tect_side[tix[ok]] = (tm.side[ok] == "left").astype(np.int8)
        self.tectal = self.tect_side >= 0
        self.soma_seg, self.soma_pos = self._somas()
        self.soma = np.zeros(self.n, bool)
        self.soma[self.soma_seg] = True
        self.pops = S.readout_populations()
        _log("world loaded", round(time.time() - t0, 1), "s")

    def _somas(self):
        import pandas as pd
        ag = pd.read_csv(REF / "agglomerated_segments_and_soma_ids.csv")
        ag = ag[ag.hires_id_agglo != 0]
        cave = pd.read_parquet(REF / "cave_somas_full.parquet", columns=["id", "pt_position"])
        pos = dict(zip(cave.id.values, cave.pt_position.values))
        segs = ag.hires_id_agglo.values.astype(np.uint64)
        ix = self.B.index_of(self.ids, segs)
        keep = ix >= 0
        soma_pos = {}
        for i, lore in zip(ix[keep], ag.lores_id.values[keep]):
            if int(i) not in soma_pos and lore in pos:
                soma_pos[int(i)] = np.asarray(pos[lore], float)
        return np.unique(ix[keep]), soma_pos

    def members(self, pop: str, side: str) -> np.ndarray:
        ix = self.B.index_of(self.ids, np.array(self.pops[pop][side], np.uint64))
        return np.unique(ix[ix >= 0])

    def m_somas(self) -> Dict[int, dict]:
        """Soma position and lateral-dendrite threshold of each M-system cell (index -> info)."""
        ro = json.load(open(DATA / "readouts.json"))["m_system"]
        out = {}
        for s in SIDES:
            seg = int(self.cells["mauthner"][s]["seg_id"])
            pos = np.array(self.cells["mauthner"][s]["pos"], float)
            out[int(self.B.index_of(self.ids, seg)[0])] = {"name": "Mauthner", "side": s, "seg_id": seg, "pos": pos,
                                                            "lat_min_um": LAT_MAUTHNER_UM}
            for nm in ("MiD2cm", "MiD3cm"):
                sid = int(ro[nm][s]["task_criteria"])
                cand = [c for c in ro["identification"]["candidates"][nm][s]["candidates"] if int(c["seg_id"]) == sid][0]
                p = np.array(cand["pos"], float)
                out[int(self.B.index_of(self.ids, sid)[0])] = {
                    "name": nm, "side": s, "seg_id": sid, "pos": p,
                    "lat_min_um": float(lateral_um(p, self.axes)[0]) + LAT_MID_EXTRA_UM}
        return out


# =============================================================================================
# 3. the synapse scan (one streaming pass over synapses.parquet)
# =============================================================================================

ACC = ("out_all", "out_np", "out_T", "cnt_T", "cnt_hb_post", "cnt_gang", "cnt_ret", "out_ret", "in_all")


def scan(world: World, watch: np.ndarray) -> dict:
    """Per-segment accumulators over every synapse, plus the full rows of every synapse touching `watch`."""
    import pyarrow.parquet as pq
    n, ids, at = world.n, world.ids, world.atlas
    acc = {k: np.zeros(n, np.int32) for k in ACC}
    wmask = np.zeros(n, bool)
    wmask[watch] = True
    keep = {k: [] for k in ("pre", "post", "type", "x", "y", "z", "m0", "m1", "m2", "dT")}
    glob = {"synapses": 0, "in_retina": 0, "in_retina_type2": 0, "in_tectum": 0, "in_neuropil": 0, "in_ganglia": 0}
    pf = pq.ParquetFile(DATA / "synapses.parquet")
    t0 = time.time()
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["pre", "post", "type", "x", "y", "z"])
        pre = np.searchsorted(ids, t.column("pre").to_numpy()).astype(np.int32)
        post = np.searchsorted(ids, t.column("post").to_numpy()).astype(np.int32)
        ty = t.column("type").to_numpy()
        xyz = np.stack([t.column(c).to_numpy() for c in ("x", "y", "z")], 1).astype(np.float64)
        del t
        L = at.lookup(xyz)
        inT = L["dT"] <= 0
        np_ = L["m2"] == NEUROPIL
        hbp = (L["m0"] == M0_HINDBRAIN) & (xyz[:, 0] >= SPAN_X_VOX)
        gang = L["m0"] == M0_GANGLIA
        ret = L["m0"] == M0_RETINA
        bc = lambda idx: np.bincount(idx, minlength=n).astype(np.int32)
        acc["out_all"] += bc(pre)
        acc["in_all"] += bc(post)
        acc["out_np"] += bc(pre[np_])
        acc["out_T"] += bc(pre[inT])
        acc["out_ret"] += bc(pre[ret])
        for m, key in ((inT, "cnt_T"), (hbp, "cnt_hb_post"), (gang, "cnt_gang"), (ret, "cnt_ret")):
            acc[key] += bc(pre[m]) + bc(post[m])
        glob["synapses"] += len(pre)
        glob["in_retina"] += int(ret.sum())
        glob["in_retina_type2"] += int((ret & (ty == 2)).sum())
        glob["in_tectum"] += int(inT.sum())
        glob["in_neuropil"] += int(np_.sum())
        glob["in_ganglia"] += int(gang.sum())
        sel = wmask[pre] | wmask[post]
        for k, v in (("pre", pre), ("post", post), ("type", ty), ("x", xyz[:, 0]), ("y", xyz[:, 1]), ("z", xyz[:, 2]),
                     ("m0", L["m0"]), ("m1", L["m1"]), ("m2", L["m2"]), ("dT", L["dT"])):
            keep[k].append(np.asarray(v)[sel])
        if rg % 5 == 0:
            _log(f"scan row group {rg + 1}/{pf.num_row_groups}", round(time.time() - t0, 1), "s")
    rows = {k: np.concatenate(v) for k, v in keep.items()}
    rows["x"], rows["y"], rows["z"] = (rows[c].astype(np.float32) for c in ("x", "y", "z"))
    _log("scan done", glob, "watched rows", len(rows["pre"]), round(time.time() - t0, 1), "s")
    return {"acc": acc, "rows": rows, "glob": glob}


class Clouds:
    """Per-segment synapse clouds of the watched segments (both roles), from scan rows."""

    def __init__(self, rows: dict, watch: np.ndarray, n: int, axes: dict):
        wmask = np.zeros(n, bool)
        wmask[watch] = True
        a = np.flatnonzero(wmask[rows["pre"]])
        b = np.flatnonzero(wmask[rows["post"]])
        seg = np.concatenate([rows["pre"][a], rows["post"][b]]).astype(np.int64)
        r = np.concatenate([a, b])
        role = np.concatenate([np.zeros(len(a), np.int8), np.ones(len(b), np.int8)])   # 0 = pre (output), 1 = post
        o = np.argsort(seg, kind="stable")
        self.seg, self.r, self.role = seg[o], r[o], role[o]
        self.rows = rows
        self.xyz = np.stack([rows["x"], rows["y"], rows["z"]], 1).astype(np.float64)
        self.um = self.xyz * UM_PER_VOXEL
        self.side = side_code(self.xyz, axes)
        self.useg, self.start = np.unique(self.seg, return_index=True)
        self.stop = np.append(self.start[1:], len(self.seg))

    def of(self, i: int) -> np.ndarray:
        """Row indices of segment i's cloud (each synapse once per role it plays)."""
        k = np.searchsorted(self.useg, i)
        if k >= len(self.useg) or self.useg[k] != i:
            return np.zeros(0, np.int64)
        return self.r[self.start[k]:self.stop[k]]


# =============================================================================================
# 4. step 1: fragment origins
# =============================================================================================

def fragments_of(world: World, members: np.ndarray):
    """Presynaptic segments of a member set, with their all-type and type-2 synapse counts onto it."""
    P = world.P
    m = np.zeros(world.n, bool)
    m[members] = True
    sel = np.flatnonzero(m[P["post"]])
    pre = P["pre"][sel].astype(np.int64)
    a = (P["t1"][sel] + P["t2"][sel]).astype(np.int64)
    e = P["t2"][sel].astype(np.int64)
    u, inv = np.unique(pre, return_inverse=True)
    return u, np.bincount(inv, weights=a).astype(np.int64), np.bincount(inv, weights=e).astype(np.int64)


def readout_sets(world: World) -> Dict[str, Dict[str, np.ndarray]]:
    return {p: {s: world.members(p, s) for s in SIDES} for p in DECIDING}


def control_sets(world: World, acc: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """Pseudo-readouts: 60 B1 tectal cells with inputs (positive) and 60 caudal-hindbrain soma segments (negative)."""
    n_in = np.bincount(world.P["post"], weights=(world.P["t1"] + world.P["t2"]), minlength=world.n)
    tect = np.flatnonzero(world.tectal & (n_in > 0))
    pos = np.sort(_rng(POS_SEED).choice(tect, N_CONTROL, replace=False))
    si = np.array(sorted(world.soma_pos), np.int64)
    sp = np.array([world.soma_pos[i] for i in si], float)
    ok = (n_in[si] >= NEG_MIN_INPUTS) & ~world.tectal[si] & (world.atlas.lookup(sp)["m1"] == M1_CAUDAL_HB)
    cand = si[ok]
    neg = np.sort(_rng(NEG_SEED).choice(cand, N_CONTROL, replace=False))
    return {"positive_tectal_cells": pos, "negative_caudal_hindbrain": neg, "negative_pool_size": int(len(cand)),
            "positive_pool_size": int(len(tect))}


def classify_readout(world: World, clouds: Clouds, members: np.ndarray, side_code_readout: Optional[int],
                     compartments: Optional[Dict[int, dict]] = None) -> dict:
    """Classify every fragment onto `members` and sum synapse weights per class (and per compartment)."""
    rows = clouds.rows
    u, a_cnt, e_cnt = fragments_of(world, members)
    mset = np.zeros(world.n, bool)
    mset[members] = True
    frag = []
    for f, ca, ce in zip(u, a_cnt, e_cnt):
        cl = clouds.of(int(f))
        con = cl[(rows["pre"][cl] == f) & mset[rows["post"][cl]]]
        con = np.unique(con)
        if len(con) != ca:
            raise AssertionError(f"fragment {f}: {len(con)} contact rows vs {ca} synapses in the pairs")
        c = classify_fragment(rows["dT"][cl], clouds.um[cl], clouds.side[cl], rows["m0"][cl], clouds.um[con],
                              rows["dT"][con], is_tectal=bool(world.tectal[f]),
                              tectal_side=int(world.tect_side[f]) if world.tectal[f] else None,
                              soma=bool(world.soma[f]))
        far_row = cl[c.pop("far_idx")] if len(cl) else None
        rec = {"idx": int(f), "seg_id": int(world.ids[f]), "syn": int(ca), "exc": int(ce), "cloud": int(len(cl)),
               "cloud_unique": int(len(np.unique(cl))), **c}
        if far_row is not None:
            rec["far_m1"], rec["far_m2"] = int(rows["m1"][far_row]), int(rows["m2"][far_row])
        if rec["cls"] == "O-d1":
            p = world.soma_pos.get(int(f))
            if p is not None:
                L = world.atlas.lookup(p)
                rec["soma_m0"], rec["soma_m1"], rec["soma_m2"] = int(L["m0"][0]), int(L["m1"][0]), int(L["m2"][0])
        if side_code_readout is not None and rec["lobe"] is not None:
            rec["lobe_rel"] = "ipsi" if rec["lobe"] == side_code_readout else "contra"
        if compartments is not None:
            comp = {"ventral": [0, 0], "lateral": [0, 0], "soma": [0, 0]}
            for rr in con:
                info = compartments[int(rows["post"][rr])]
                dz = (float(rows["z"][rr]) - info["pos"][2]) * UM_PER_VOXEL[2]
                lat = float(lateral_um([[rows["x"][rr], rows["y"][rr], rows["z"][rr]]], world.axes)[0])
                k = str(compartment(np.array([dz]), np.array([lat]), info["lat_min_um"])[0])
                comp[k][0] += 1
                comp[k][1] += int(rows["type"][rr] == 2)
            rec["comp"] = comp
        frag.append(rec)
    return {"fragments": frag, "summary": summarize(frag, compartments is not None)}


def summarize(frag: List[dict], with_comp: bool = False) -> dict:
    tot_a = sum(f["syn"] for f in frag)
    tot_e = sum(f["exc"] for f in frag)
    out = {"fragments": len(frag), "input_synapses": tot_a, "input_exc_synapses": tot_e, "classes": {}}
    for c in CLASSES:
        fs = [f for f in frag if f["cls"] == c]
        a, e = sum(f["syn"] for f in fs), sum(f["exc"] for f in fs)
        row = {"fragments": len(fs), "synapses": a, "exc_synapses": e,
               "share": _r(a / tot_a if tot_a else None), "share_exc": _r(e / tot_e if tot_e else None)}
        if c in ("O-a0", "O-a1"):
            for rel in ("ipsi", "contra"):
                fr = [f for f in fs if f.get("lobe_rel") == rel]
                row[rel] = {"fragments": len(fr), "synapses": sum(f["syn"] for f in fr), "exc_synapses": sum(f["exc"] for f in fr)}
        out["classes"][c] = row
    out["edge_flagged"] = {"fragments": sum(f["edge"] for f in frag), "synapses": sum(f["syn"] for f in frag if f["edge"])}
    out["singletons"] = {"fragments": sum(f["cloud_unique"] == 1 for f in frag),
                         "synapses": sum(f["syn"] for f in frag if f["cloud_unique"] == 1)}
    if with_comp:
        out["compartments"] = {}
        for k in ("ventral", "lateral", "soma"):
            ta = sum(f["comp"][k][0] for f in frag)
            te = sum(f["comp"][k][1] for f in frag)
            row = {"synapses": ta, "exc_synapses": te, "classes": {}}
            for c in CLASSES:
                fs = [f for f in frag if f["cls"] == c]
                a = sum(f["comp"][k][0] for f in fs)
                e = sum(f["comp"][k][1] for f in fs)
                row["classes"][c] = {"fragments": sum(f["comp"][k][0] > 0 for f in fs), "synapses": a, "exc_synapses": e,
                                     "share": _r(a / ta if ta else None), "share_exc": _r(e / te if te else None)}
            out["compartments"][k] = row
    return out


# =============================================================================================
# 5. step 2: the retinal-afferent source set
# =============================================================================================

def afferent_sets(world: World, acc: dict) -> dict:
    import pyarrow.parquet as pq
    t = pq.read_table(DATA / "cells.parquet", columns=["seg_id", "n_in", "n_out", "out_cx", "out_cy", "out_cz"])
    seg = t.column("seg_id").to_numpy().astype(np.uint64)
    if len(seg) != world.n or not np.array_equal(seg, world.ids):
        raise ValueError("cells.parquet is not aligned with the pair ids")
    n_in, n_out = t.column("n_in").to_numpy(), t.column("n_out").to_numpy()
    oc = np.stack([t.column(c).to_numpy() for c in ("out_cx", "out_cy", "out_cz")], 1)
    del t
    checks = {"n_out_matches_scan": bool(np.array_equal(n_out, acc["out_all"])),
              "n_in_matches_scan": bool(np.array_equal(n_in, acc["in_all"]))}
    res = {"checks": checks}
    for name, nz in (("primary", False), ("near_zero", True)):
        m = afferent_rule(n_in, n_out, acc["out_np"], world.soma, world.tectal, near_zero=nz)
        idx = np.flatnonzero(m)
        side = side_code(oc[idx], world.axes)
        lat = lateral_um(oc[idx], world.axes)
        keep = lat >= SIDE_AMBIGUOUS_UM
        res[name] = {"idx": idx[keep], "side": side[keep], "pos": oc[idx[keep]],
                     "n_rule": int(len(idx)), "n_dropped_midline": int((~keep).sum()),
                     "out_synapses": int(n_out[idx[keep]].sum()), "in_synapses": int(n_in[idx[keep]].sum())}
    return res


def afferent_context(world: World, w, aff: dict):
    """A structure.Context whose sources are the afferents (per lobe) and whose patches drive them."""
    from fishbrain import cells as C
    S = world.S
    base = S.build_context(world.P, w.n_in)
    idx, side, pos = aff["idx"], aff["side"], aff["pos"]
    sname = np.where(side == 1, "left", "right")
    fake = {"axes": world.axes, "tectum": [{"seg_id": int(world.ids[i]), "pos": [float(v) for v in p], "side": str(s)}
                                           for i, p, s in zip(idx, pos, sname)]}
    tmap = C.TectumMap(fake)
    tect = {s: np.sort(idx[sname == s]) for s in SIDES}
    tmask = np.zeros(world.n, bool)
    tmask[idx] = True
    ro = np.concatenate([v for d in base.pops_all.values() for v in d.values()])
    if tmask[ro].any():
        raise AssertionError("a readout member is in the afferent set")
    patch = {k: {} for k in S.PATCHES}
    info = {}
    for eye in SIDES:
        lobe = S.CONTRA[eye]
        for k in S.PATCHES:
            d = tmap.drive({eye: S.patch_points(k, eye)})
            ok = d > 0
            if np.any(sname[ok] != lobe):
                raise AssertionError("a patch drove the wrong lobe's afferents")
            ii, vv = idx[ok], d[ok]
            o = np.argsort(ii)
            patch[k][eye] = (ii[o], vv[o])
            info[f"{k}_{eye}_eye"] = {"afferents_driven": int(ok.sum()), "drive_sum": float(vv.sum()),
                                      "drive_max": float(vv.max()), "lobe": lobe}
    ctx = S.Context(world.n, world.ids, tect, tmask, base.pops, base.pops_all, patch, base.zones, base.tect_recv,
                    {"afferents_per_lobe": {s: int(len(tect[s])) for s in SIDES},
                     "tectal_targets_with_inputs": base.info["tectal_with_inputs"], "populations": base.info["populations"],
                     "patches": info, "zones": {e: {k: int(len(v)) for k, v in z.items()} for e, z in base.zones.items()}})
    return ctx, tmap


# =============================================================================================
# 6. step 3: the eye
# =============================================================================================

def eye_report(world: World, acc: dict, glob: dict) -> dict:
    import pandas as pd
    at = world.atlas
    cave = pd.read_parquet(REF / "cave_somas_full.parquet", columns=["id", "pt_position"])
    p = np.vstack(cave.pt_position.values).astype(float)
    L = at.lookup(p)
    in_ret = L["m0"] == M0_RETINA
    seg_ret = acc["cnt_ret"] > 0
    soma_ret = [i for i, q in world.soma_pos.items() if int(at.lookup(q)["m0"][0]) == M0_RETINA]
    both = seg_ret & (acc["out_T"] > 0)
    return {
        "retina_voxels_4096": int((at.m0 == M0_RETINA).sum()),
        "synapses_in_retina": glob["in_retina"], "synapses_in_retina_type2": glob["in_retina_type2"],
        "segments_with_any_synapse_in_retina": int(seg_ret.sum()),
        "segments_with_retina_synapses_and_outputs_in_tectum": int(both.sum()),
        "their_output_synapses_in_tectum": int(acc["out_T"][both].sum()),
        "cave_somas_in_retina": int(in_ret.sum()),
        "soma_segments_with_soma_in_retina": len(soma_ret),
        "soma_segments_in_retina_with_outputs_in_tectum": int(sum(acc["out_T"][i] > 0 for i in soma_ret)),
        "synapses_total": glob["synapses"], "synapses_in_tectum_mask": glob["in_tectum"],
        "synapses_in_neuropil": glob["in_neuropil"], "synapses_in_ganglia": glob["in_ganglia"],
    }


# =============================================================================================
# 7. step 4: predicted decision-path share
# =============================================================================================

def exc_bfs(world: World, sources: np.ndarray) -> np.ndarray:
    P = world.P
    m = P["t2"] > 0
    ip, ix = world.B._csr(world.n, P["pre"][m].astype(np.int64), P["post"][m].astype(np.int64))
    return world.B.bfs_levels(world.n, ip, ix, sources)


def share_bounds(exc_M: float, exc_A: float, exc_B: float, x_M: float, q_A: float, q_B: float, exc_all: float) -> dict:
    """The pre-registered estimates. Synapse view: equal activity per synapse; walk view: linear, unit gains."""
    def frac(a, b):
        return None if not b > 0 else a / b
    return {"LB_syn": _r(frac(exc_M, exc_M + exc_A + exc_B)), "UB_syn": _r(frac(exc_M + exc_A, exc_M + exc_A + exc_B)),
            "LB_walk": _r(frac(x_M, x_M + q_A + q_B), 9), "UB_walk": _r(frac(x_M + q_A, x_M + q_A + q_B), 9),
            "all_inputs_M": _r(frac(exc_M, exc_all), 9), "all_inputs_M_plus_A": _r(frac(exc_M + exc_A, exc_all), 9)}


def decision_share(world: World, origins: dict, aff: dict, real: dict) -> dict:
    """Per deciding readout x side x stimulated lobe (see the note's s.0 step 4)."""
    P = world.P
    n_in = np.bincount(P["post"], weights=(P["t1"] + P["t2"]), minlength=world.n)
    dist = {lobe: exc_bfs(world, aff["idx"][aff["side"] == (1 if lobe == "left" else 0)]) for lobe in SIDES}
    LW = real["detail"]["lobe_weights"]["exc"]["cum"]
    affmask = np.zeros(world.n, bool)
    affmask[aff["idx"]] = True
    out = {"afferents_among_fragments": {}}
    for pop in DECIDING:
        out[pop] = {}
        for side in SIDES:
            mem = world.members(pop, side)
            recv = mem[n_in[mem] > 0]
            frag = origins["readouts"][pop][side]["fragments"]
            tag = {}
            for f in frag:
                if affmask[f["idx"]]:
                    t = tag.setdefault(f["cls"], {"fragments": 0, "synapses": 0, "exc_synapses": 0})
                    t["fragments"] += 1
                    t["synapses"] += f["syn"]
                    t["exc_synapses"] += f["exc"]
            out["afferents_among_fragments"][f"{pop}_{side}"] = tag
            sel = np.flatnonzero(np.isin(P["post"], recv))
            pre, t2 = P["pre"][sel].astype(np.int64), P["t2"][sel].astype(np.float64)
            frac_e = t2 / n_in[P["post"][sel]]
            exc_all = float(t2.sum())
            for lobe in SIDES:
                d = dist[lobe]
                inM = (d[pre] >= 0) & (d[pre] <= 2)
                lobe_code = 1 if lobe == "left" else 0
                A = {f["idx"] for f in frag if f["cls"] in ("O-a0", "O-a1") and f["lobe"] == lobe_code}
                Bset = {f["idx"] for f in frag if f["cls"] == "O-b"}
                inA = np.isin(pre, list(A)) & ~inM
                inB = np.isin(pre, list(Bset)) & ~inM & ~inA
                q = lambda m: float(np.bincount(np.searchsorted(recv, P["post"][sel][m]), weights=frac_e[m],
                                                minlength=len(recv)).mean()) if len(recv) else 0.0
                x_M = float(LW[pop][lobe][side]) if LW[pop][lobe][side] is not None else 0.0
                row = {"exc_all": exc_all, "exc_M": float(t2[inM].sum()), "exc_A": float(t2[inA].sum()),
                       "exc_B": float(t2[inB].sum()), "segments_M": int(len(np.unique(pre[inM]))),
                       "segments_A": int(len(np.unique(pre[inA]))), "segments_B": int(len(np.unique(pre[inB]))),
                       "x_M_walk": x_M, "q_A": q(inA), "q_B": q(inB),
                       "relation": "ipsi" if lobe == side else "contra"}
                row.update(share_bounds(row["exc_M"], row["exc_A"], row["exc_B"], x_M, row["q_A"], row["q_B"], exc_all))
                # post hoc (04:20 EDT, after the pre-registered row was seen): how the estimate moves with the size of
                # the bridge's (b) pool. near20 = O-b fragments whose cloud comes within 20 um of the tectum; none = a
                # bridge that drives only the observed-origin fragments.
                row["variants_post_hoc"] = {}
                for vname, keepf in (("near20", lambda f: f["min_dT"] is not None and f["min_dT"] <= 20.0),
                                     ("none", lambda f: False)):
                    Bv = {f["idx"] for f in frag if f["cls"] == "O-b" and keepf(f)}
                    inBv = np.isin(pre, list(Bv)) & ~inM & ~inA
                    eb, qb = float(t2[inBv].sum()), q(inBv)
                    row["variants_post_hoc"][vname] = {"exc_B": eb, "q_B": qb, **share_bounds(
                        row["exc_M"], row["exc_A"], eb, x_M, row["q_A"], qb, exc_all)}
                out[pop][side] = out[pop].get(side, {})
                out[pop][side][f"from_{lobe}_lobe"] = row
        # the behaviourally correct pairing (contra) and the "no crossing" pairing (ipsi), both stimulated lobes
        pair = {"primary": PAIRING[pop]}
        for rel in ("contra", "ipsi"):
            pair[rel] = {f"{lobe}_lobe_to_{side}": out[pop][side][f"from_{lobe}_lobe"] for lobe in SIDES for side in SIDES
                         if (side != lobe) == (rel == "contra")}
        out[pop]["pairing"] = pair
    return out


# =============================================================================================
# 8. the mesh sensitivity row (public agglomeration meshes; read-only network)
# =============================================================================================

def mesh_extent(seg_ids: Sequence[int], atlas: Atlas, axes: dict) -> List[dict]:
    import warnings
    warnings.filterwarnings("ignore")
    from cloudvolume import CloudVolume
    cv = CloudVolume("precomputed://gs://fish1-public/seg_241003_agg241003", mip=0, use_https=True, progress=False)
    res = []
    for s in seg_ids:
        try:
            m = cv.mesh.get(int(s), lod=3)
            m = m[int(s)] if isinstance(m, dict) else m
            v = np.asarray(m.vertices, float) / np.array([8.0, 8.0, 30.0])
        except Exception as e:  # report, never hide
            res.append({"seg_id": int(s), "error": type(e).__name__})
            continue
        if not len(v):
            res.append({"seg_id": int(s), "vertices": 0})
            continue
        L = atlas.lookup(v)
        um = v * UM_PER_VOXEL
        res.append({"seg_id": int(s), "vertices": int(len(v)), "min_dT": _r(float(L["dT"].min()), 2),
                    "vertices_in_T": int((L["dT"] <= 0).sum()),
                    "x_min_um": _r(um[:, 0].min(), 1), "x_max_um": _r(um[:, 0].max(), 1),
                    "extent_um": [_r(e, 1) for e in (um.max(0) - um.min(0))]})
    return res


# =============================================================================================
# 9. runner
# =============================================================================================

def _watch(world: World, ctrl: dict) -> np.ndarray:
    ws = []
    for pop in DECIDING:
        for s in SIDES:
            ws.append(fragments_of(world, world.members(pop, s))[0])
    for k in ("positive_tectal_cells", "negative_caudal_hindbrain"):
        ws.append(fragments_of(world, ctrl[k])[0])
    return np.unique(np.concatenate(ws))


def atlas_check(world: World) -> dict:
    tm = world.C.tectum_map(world.cells)
    L = world.atlas.lookup(tm.pos)
    share = float((L["m1"] == M1_TECTUM).mean())
    share_T = float((L["dT"] <= 0).mean())
    out = {"tectal_somas": int(len(tm.pos)), "share_in_mece1_tectum": _r(share), "share_in_tectum_mask_17_18": _r(share_T),
           "share_in_SPV_17": _r(float((L["m2"] == 17).mean()))}
    for s in SIDES:
        p = np.array(world.cells["mauthner"][s]["pos"], float)
        i, j, k = world.atlas.vox(p)
        box = world.atlas.m2[max(i[0] - 1, 0):i[0] + 2, max(j[0] - 1, 0):j[0] + 2, max(k[0] - 1, 0):k[0] + 2]
        out[f"mauthner_{s}"] = {"label": int(world.atlas.m2[i, j, k][0]), "48_within_one_voxel": bool((box == M2_MAUTHNER).any())}
    out["pass"] = bool(share >= 0.8 and out["mauthner_left"]["48_within_one_voxel"])
    return out


def run_classify(world: World) -> dict:
    t0 = time.time()
    chk = atlas_check(world)
    _log("atlas check", chk)
    if not chk["pass"]:
        raise RuntimeError("atlas lookup failed its validity check; no class may be used")
    ctrl = control_sets(world)
    watch = _watch(world, ctrl)
    _log("watched fragments", len(watch))
    sc = scan(world, watch)
    np.savez_compressed(OUT / "scan_acc.npz", **sc["acc"])
    clouds = Clouds(sc["rows"], watch, world.n, world.axes)
    msom = world.m_somas()
    res = {"atlas_check": chk, "glob": sc["glob"], "readouts": {}, "controls": {}}
    for pop in DECIDING:
        res["readouts"][pop] = {}
        for s in SIDES:
            mem = world.members(pop, s)
            comp = msom if pop in ("m_system", "mauthner") else None
            res["readouts"][pop][s] = classify_readout(world, clouds, mem, 1 if s == "left" else 0, comp)
            _log(pop, s, json.dumps(res["readouts"][pop][s]["summary"]["classes"])[:400])
    for k in ("positive_tectal_cells", "negative_caudal_hindbrain"):
        r = classify_readout(world, clouds, ctrl[k], None, None)
        res["controls"][k] = {"cells": [int(world.ids[i]) for i in ctrl[k]], "summary": r["summary"]}
    res["controls"]["pool_sizes"] = {"positive": ctrl["positive_pool_size"], "negative": ctrl["negative_pool_size"]}
    # brain-wide: does any segment span tectum -> hindbrain in synapse space?
    acc = sc["acc"]
    span = (acc["cnt_T"] >= MIN_TECT_SYN) & (acc["cnt_hb_post"] >= 2)
    ro_all = np.unique(np.concatenate([world.members(p, s) for p in DECIDING for s in SIDES]))
    frag_all = fragments_of(world, ro_all)[0]
    res["spanning"] = {"segments": int(span.sum()), "their_synapses_in_T": int(acc["cnt_T"][span].sum()),
                       "their_synapses_in_hindbrain_posterior": int(acc["cnt_hb_post"][span].sum()),
                       "soma_bearing": int((span & world.soma).sum()), "tectal_b1": int((span & world.tectal).sum()),
                       "touching_a_deciding_readout": int(span[frag_all].sum()),
                       "rule": "cnt_T >= 2 and cnt(mece0 6 hindbrain, x >= 67,500) >= 2, either role"}
    res["eye"] = eye_report(world, acc, sc["glob"])
    res["pipeline_check"] = pipeline_check(res)
    res["seconds"] = round(time.time() - t0, 1)
    res["built"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    _dump(OUT / "origins.json", res)
    return res


def pipeline_check(res: dict) -> dict:
    """O-a0 contacts vs task R / G1b direct tectal contacts (ipsi / contra), exact."""
    expect = {"m_system": (5, 0), "mauthner": (4, 0), "strike": (3, 1), "turning": (2, 0)}
    out = {}
    for pop, (ei, ec) in expect.items():
        ipsi = sum(res["readouts"][pop][s]["summary"]["classes"]["O-a0"]["ipsi"]["synapses"] for s in SIDES)
        contra = sum(res["readouts"][pop][s]["summary"]["classes"]["O-a0"]["contra"]["synapses"] for s in SIDES)
        out[pop] = {"ipsi": ipsi, "contra": contra, "expected": [ei, ec], "match": (ipsi, contra) == (ei, ec)}
    out["pass"] = all(v["match"] for v in out.values() if isinstance(v, dict))
    return out


def run_meshes(world: World) -> dict:
    o = json.load(open(OUT / "origins.json"))
    picks = []
    for s in SIDES:
        fr = [f for f in o["readouts"]["mauthner"][s]["fragments"] if f["comp"]["ventral"][1] > 0]
        fr.sort(key=lambda f: (-f["comp"]["ventral"][1], f["seg_id"]))
        picks += [dict(f, side=s) for f in fr[:MESH_TOP_PER_SIDE]]
    ext = mesh_extent([f["seg_id"] for f in picks], world.atlas, world.axes)
    rows = []
    for f, e in zip(picks, ext):
        rows.append({"side": f["side"], "seg_id": f["seg_id"], "cls": f["cls"], "ventral_exc_contacts": f["comp"]["ventral"][1],
                     "cloud_min_dT": f["min_dT"], "cloud_far_um": f["far_um"], "mesh": e,
                     "mesh_reaches_T_where_cloud_did_not": bool(e.get("vertices_in_T", 0) > 0 and f["cls"] not in ("O-a0", "O-a1"))})
    res = {"picks": len(rows), "rows": rows,
           "n_mesh_reaches_T": sum(r["mesh"].get("vertices_in_T", 0) > 0 for r in rows),
           "n_reaches_T_where_cloud_did_not": sum(r["mesh_reaches_T_where_cloud_did_not"] for r in rows),
           "errors": sum("error" in r["mesh"] for r in rows), "built": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    _dump(OUT / "meshes.json", res)
    return res


def run_oa(world: World) -> dict:
    """The O-a fragments in detail: where their tectal end is, which role its synapses play, what feeds them there,
    and the tectal column they sit in (median preferred field of the 20 nearest B1 tectal somas; a choice)."""
    import pyarrow.parquet as pq
    from scipy.spatial import cKDTree
    o = json.load(open(OUT / "origins.json"))
    touch: Dict[int, list] = {}
    for pop in DECIDING:
        for s in SIDES:
            for f in o["readouts"][pop][s]["fragments"]:
                if f["cls"] in ("O-a0", "O-a1"):
                    touch.setdefault(f["idx"], []).append({"readout": f"{pop}_{s}", "syn": f["syn"], "exc": f["exc"],
                                                           "comp": f.get("comp"), "lobe_rel": f.get("lobe_rel"),
                                                           "cls": f["cls"]})
    fset = np.array(sorted(touch), np.int64)
    mask = np.zeros(world.n, bool)
    mask[fset] = True
    aff = np.load(OUT / "afferents_primary.npz")
    affmask = np.zeros(world.n, bool)
    affmask[aff["idx"]] = True
    keep = {k: [] for k in ("pre", "post", "type", "x", "y", "z")}
    pf = pq.ParquetFile(DATA / "synapses.parquet")
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["pre", "post", "type", "x", "y", "z"])
        pre = np.searchsorted(world.ids, t.column("pre").to_numpy()).astype(np.int64)
        post = np.searchsorted(world.ids, t.column("post").to_numpy()).astype(np.int64)
        sel = mask[pre] | mask[post]
        for k, v in (("pre", pre), ("post", post), ("type", t.column("type").to_numpy()),
                     ("x", t.column("x").to_numpy()), ("y", t.column("y").to_numpy()), ("z", t.column("z").to_numpy())):
            keep[k].append(np.asarray(v)[sel])
    R = {k: np.concatenate(v) for k, v in keep.items()}
    xyz = np.stack([R["x"], R["y"], R["z"]], 1).astype(float)
    L = world.atlas.lookup(xyz)
    tm = world.C.tectum_map(world.cells)
    tree = cKDTree(tm.pos * UM_PER_VOXEL)
    # post hoc (04:30 EDT): how hard G1b's loom and prey patches drive each column under B1's map (the eye that
    # drives the fragment's lobe; mean over the 20 nearest somas, as a share of that patch's maximum in the lobe)
    patch_drive = {}
    for k in world.S.PATCHES:
        for eye in SIDES:
            d = world.C.tectum_drive({eye: world.S.patch_points(k, eye)}, world.cells)
            patch_drive[(k, world.S.CONTRA[eye])] = d / d.max()
    out = []
    for f in fset:
        for_role = {}
        for role, col in (("out", "pre"), ("in", "post")):
            m = R[col] == f
            inT = m & (L["dT"] <= 0)
            for_role[role] = {"synapses": int(m.sum()), "in_tectum": int(inT.sum()),
                              "in_tectum_neuropil": int((inT & (L["m2"] == NEUROPIL)).sum())}
            if role == "in":
                partners = R["pre"][inT]
                for_role[role]["tectal_inputs_from_afferents"] = int(affmask[partners].sum())
                for_role[role]["tectal_inputs_from_b1_tectal_cells"] = int(world.tectal[partners].sum())
                for_role[role]["all_inputs_from_afferents"] = int(affmask[R["pre"][m]].sum())
        cl = (R["pre"] == f) | (R["post"] == f)
        tT = cl & (L["dT"] <= 0)
        rec = {"idx": int(f), "seg_id": int(world.ids[f]), "readouts": touch[int(f)], "roles": for_role,
               "soma": bool(world.soma[f]), "b1_tectal": bool(world.tectal[f]),
               "x_um_range": [_r(xyz[cl, 0].min() * UM_PER_VOXEL[0], 1), _r(xyz[cl, 0].max() * UM_PER_VOXEL[0], 1)],
               "z_um_range": [_r(xyz[cl, 2].min() * UM_PER_VOXEL[2], 1), _r(xyz[cl, 2].max() * UM_PER_VOXEL[2], 1)]}
        if tT.any():
            c = xyz[tT].mean(0)
            rec["tectal_end_8nm"] = [round(float(v)) for v in c]
            rec["tectal_end_side"] = "left" if side_code(c, world.axes)[0] == 1 else "right"
            rec["tectal_end_m2"] = {str(k): int(v) for k, v in zip(*np.unique(L["m2"][tT], return_counts=True))}
            _, nn = tree.query(c * UM_PER_VOXEL, k=20)
            rec["column_field_deg"] = {"theta": _r(float(np.median(tm.theta[nn])), 1), "elev": _r(float(np.median(tm.elev[nn])), 1),
                                       "nearest_somas_um": _r(float(np.linalg.norm(tm.pos[nn[0]] * UM_PER_VOXEL - c * UM_PER_VOXEL)), 1)}
            rec["patch_drive_rel_post_hoc"] = {k: _r(float(patch_drive[(k, rec["tectal_end_side"])][nn].mean()), 6)
                                               for k in world.S.PATCHES}
        out.append(rec)
    res = {"fragments": out, "n": len(out), "built": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "note": "column_field_deg = median B1 preferred field of the 20 tectal somas nearest the tectal-end centroid (choice)"}
    _dump(OUT / "oa_fragments.json", res)
    return res


def _aff_payload(aff: dict) -> dict:
    return {k: v for k, v in aff.items() if k not in ("idx", "side", "pos")}


def run_real(world: World, which: str = "primary"):
    acc = dict(np.load(OUT / "scan_acc.npz"))
    affs = afferent_sets(world, acc)
    aff = affs[which]
    w = world.S.Wiring(world.n, world.P["pre"], world.P["post"], world.P["t1"], world.P["t2"], label="real")
    ctx, tmap = afferent_context(world, w, aff)
    m = world.S.measure(w, ctx)
    m["context"] = ctx.info
    m["afferents"] = {k: _aff_payload(v) for k, v in affs.items() if k != "checks"} | {"checks": affs["checks"]}
    m["afferent_placement"] = {"theta_range": [float(tmap.theta.min()), float(tmap.theta.max())],
                               "elev_range": [float(tmap.elev.min()), float(tmap.elev.max())]}
    m["built"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    name = "real.json" if which == "primary" else f"real_{which}.json"
    _dump(SHUFFLE_DIR / name, m)
    np.savez_compressed(OUT / f"afferents_{which}.npz", idx=aff["idx"], side=aff["side"], pos=aff["pos"],
                        seg_id=world.ids[aff["idx"]])
    _log("real done", which, m["detail"]["seconds"], "s")
    return m, w, ctx


def run_shuffles(world: World, n_a: int = 20, n_b: int = 0):
    acc = dict(np.load(OUT / "scan_acc.npz"))
    aff = afferent_sets(world, acc)["primary"]
    w = world.S.Wiring(world.n, world.P["pre"], world.P["post"], world.P["t1"], world.P["t2"], label="real")
    ctx, _ = afferent_context(world, w, aff)
    in0, out0 = w.in_deg.copy(), w.out_deg.copy()
    del w
    sides = None
    jobs = [("A", k) for k in range(n_a)] + [("B", k) for k in range(n_b)]
    for kind, k in jobs:
        path = SHUFFLE_DIR / f"{kind}-{k:02d}.json"
        if path.exists():
            continue
        if kind == "B" and sides is None:
            sides = world.S.segment_sides(world.ids)
        m = world.S.one_shuffle(world.P, ctx, kind, k, sides, in0, out0)
        _dump(path, m)
        _log(f"shuffle {kind}-{k}", m["repair"], m["checks"], m["seconds_total"], "s")


def assemble() -> dict:
    from fishbrain import structure as S
    real = json.load(open(SHUFFLE_DIR / "real.json"))
    A = [json.load(open(p)) for p in sorted(SHUFFLE_DIR.glob("A-*.json"))]
    Bn = [json.load(open(p)) for p in sorted(SHUFFLE_DIR.glob("B-*.json"))]
    if len(A) < S.MIN_SHUFFLES:
        raise RuntimeError(f"only {len(A)} Null-A shuffles")
    rf = real["flat"]
    nullA = {k: S.null_summary(v, [s["flat"].get(k) for s in A]) for k, v in rf.items()}
    nullB = {k: S.null_summary(v, [s["flat"].get(k) for s in Bn]) for k, v in rf.items()} if Bn else {}
    for v in nullB.values():
        v.pop("beyond_all", None)
    ver = S.verdict(rf, nullA)
    g1b = json.load(open(S.STRUCTURE_JSON)) if S.STRUCTURE_JSON.exists() else None
    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "code_version": CODE_VERSION,
           "sources": "retinal afferents (primary rule), see origins.afferent_rule",
           "real": real, "nullA": nullA, "nullB": nullB, "verdict": ver,
           "shuffles": {"A": [{"k": s["k"], "seed": s["seed"], "repair": s["repair"], "checks": s["checks"]} for s in A],
                        "B": [{"k": s["k"], "seed": s["seed"], "repair": s["repair"], "checks": s["checks"]} for s in Bn]},
           "nullA_detail_means": S._detail_means(A),
           "g1b_reference": None if g1b is None else {"reach": g1b["real"]["detail"]["reach"],
                                                      "verdict_call": g1b["verdict"]["call"],
                                                      "lobe_weights_exc_cum": g1b["real"]["detail"]["lobe_weights"]["exc"]["cum"]}}
    _dump(OUT / "reach.json", out)
    return out


def run_share(world: World) -> dict:
    origins = json.load(open(OUT / "origins.json"))
    real = json.load(open(SHUFFLE_DIR / "real.json"))
    acc = dict(np.load(OUT / "scan_acc.npz"))
    aff = afferent_sets(world, acc)["primary"]
    res = decision_share(world, origins, aff, real)
    res["built"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    _dump(OUT / "share.json", res)
    return res


def main(argv: Sequence[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SHUFFLE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = argv[0] if argv else "all"
    world = World()
    if cmd in ("classify", "all"):
        run_classify(world)
    if cmd in ("real", "all"):
        run_real(world, "primary")
        run_real(world, "near_zero")
    if cmd in ("shuffles", "all"):
        n_a = int(argv[1]) if len(argv) > 1 else 20
        n_b = int(argv[2]) if len(argv) > 2 else 0
        run_shuffles(world, n_a, n_b)
    if cmd in ("meshes", "all"):
        run_meshes(world)
    if cmd in ("assemble", "all"):
        o = assemble()
        print(json.dumps(o["verdict"]["call"]))
    if cmd in ("oa", "all"):
        run_oa(world)
    if cmd in ("share", "all"):
        run_share(world)
    if cmd not in ("classify", "real", "shuffles", "meshes", "assemble", "oa", "share", "all"):
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
