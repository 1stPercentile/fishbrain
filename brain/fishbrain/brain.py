"""FISHBRAIN G1 brain: the Fish1 wiring as a spiking network, its eye and its readouts (task C).

Pipeline (every step is measured or listed as our choice in evidence/G1-gate.md):

1. ``load_pairs()`` reads ``data/synapses.parquet`` (all 29,474,316 synapses of the brain of record)
   and aggregates them per (pre segment, post segment) pair with the type-1 and type-2 counts of
   every pair (cached as .npy under ``data/net/``).
2. ``signed_counts()`` applies the E/I sign. Type 2 = excitatory, type 1 = inhibitory, measured
   against the Fish1 authors' ground-truth cell lists (``ei_polarity_check()``). The gate uses the
   brain of record as published: each synapse signs itself (``GATE_SIGN_RULE = "per_synapse"``);
   Dale's-law majority per segment is kept only as a sensitivity row.
3. ``stimulated_set()``: vision enters only tectal cells from ``cells_identified.json``, never an
   excluded cell, and never a tectal cell with a direct synapse onto any readout (Mauthner, SPN,
   SPN mirror candidate), so every readout is at least 2 hops from the input.
4. ``reduce_graph()`` keeps exactly the neurons that can matter for a readout when input only
   enters the stimulated set: those that can fire (reachable from a stimulated cell through
   excitatory pairs; Shiu's LIF has no background input and no rebound, so nothing else ever
   fires) AND can reach a readout through pairs whose presynaptic cell can fire. Readouts are kept
   regardless. ``tests/test_gate.py`` checks that readout spike trains are identical on the reduced
   network and on the whole fireable set.
5. ``Retina`` turns stimulus frames into per-tectal-cell Poisson rates: OFF-transient, balanced
   centre-surround front end, then ``cells.tectum_drive`` (used as a linear operator, built column
   by column from single-point calls and checked against direct calls), then 150 Hz x clip(drive).
6. ``run_trial()`` simulates with ``sim.Simulator`` (Shiu et al. 2024 LIF, dt 0.1 ms) and returns
   every readout's spikes, the raster hash and the hash of the rates fed in.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fishbrain import cells as C
from fishbrain import sim as S
from fishbrain import stimuli as ST

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
NET = DATA / "net"
SYNAPSES = DATA / "synapses.parquet"
EI_CHECK = DATA / "ei_polarity_check.json"
POLARITY_STATE = DATA / "ref" / "evaluation_polarity_assignment.json"
CODE_VERSION = "g1-brain-v2"

# =============================================================================================
# 0. PRE-REGISTERED PARAMETERS (fixed 2026-09-25 before the first stimulus run; G1-gate.md s.3)
# =============================================================================================

# ---- E/I: measured (data/ei_polarity_check.json), not assumed ----
SIGN_MAP = {1: -1, 2: +1}          # type 1 = inhibitory, type 2 = excitatory
SIGN_RULES = ("per_synapse", "dale_majority")
GATE_SIGN_RULE = "per_synapse"     # the brain of record as published; dale_majority = sensitivity row
SIGN_RULE = GATE_SIGN_RULE

WIRING_CONTENT_HASH = "09a71038502546afd12082fbde9f538fdc0704e940de33e860db1f0962d25af2"  # pull.py


@dataclass(frozen=True)
class RetinaParams:
    """The eye's front end. Every value is our choice unless a source is named."""
    res_deg: float = 2.0             # stimuli.py lattice spacing (B2 default)
    frame_ms: float = 5.0            # stimulus frame period fed to the brain
    surround_sigma_deg: float = 10.0  # Gaussian surround of the centre-surround stage (choice)
    surround_radius_deg: float = 30.0  # surround truncated at 3 sigma (choice)
    tau_hp_ms: float = 100.0         # transient (high-pass) time constant (choice)
    w_off: float = 1.0               # OFF channel weight
    w_on: float = 0.0                # ON channel weight: dark looms escape, bright looms rarely (Temizer 2015)
    r_max_hz: float = 150.0          # a fully driven tectal cell fires at Shiu's r_poi = 150 Hz (model.py)
    rate_round_hz: float = 1e-6      # rates rounded before the simulator sees them
    zero_tol: float = 1e-9           # retinal values below this are exactly 0

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()


RETINA = RetinaParams()

# ---- stimulus protocol (B2's stimuli.py with its defaults unless stated) ----
PROTOCOL = {
    "loom": {"azimuth_deg": 90.0, "elevation_deg": 0.0, "l_over_v_ms": 240.0, "start_deg": 4.0,
             "end_deg": 140.0, "contrast": -1.0, "pre_ms": 500.0, "hold_ms": 500.0},
    "recede": {"static_ms": 1000.0},   # the 140 deg disc sits still for 1 s (onset window), then recedes
    "dim": {"eyes": "both", "mode": "matched", "pre_ms": 500.0, "hold_ms": 500.0},
    "prey": {"azimuth_deg": 30.0, "elevation_deg": 0.0, "size_deg": 3.0, "speed_deg_s": 30.0,
             "sweep_deg": 20.0, "contrast": 1.0, "duration_ms": 3000.0, "pre_ms": 500.0},
    "blank": {"duration_ms": 3500.0, "pre_ms": 500.0},
}

# ---- gate criteria ----
GATE = {
    "n_seeds": 8,                     # per condition; the same seed strings in every condition (paired)
    "seed_fmt": "G1-seed-{k}",
    "crit_angle_min_deg": 12.0,       # Temizer 2015 critical angle 21.7 +- 4.9 deg, minus 2 SD
    "sensorimotor_allow_ms": 100.0,   # Dunn 2016: escape follows the threshold angle by 81 ms
    "p_escape_min": 0.5,              # loom: >= half the seeds fire a Mauthner cell in the window
    "side_frac_min": 0.75,            # loom: first Mauthner to fire is the predicted one
    "timing_frac_min": 0.75,          # loom: first spike inside the plausible window
    "control_ratio_max": 0.5,         # receding/dimming: Mauthner spikes per trial <= half of loom's
    "turn_frac_min": 0.75,            # prey: turn index has the predicted sign in >= 75% of seeds
}
MAUTHNER_PREDICTION = "ipsilateral_to_stimulus"   # Dunn et al. 2016 (see G1-gate.md s.4)
SHUFFLE_SEED = "G1-whole-brain-shuffle"
SUBNET_SHUFFLE_SEED = "G1-subnetwork-shuffle"


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(str(p).encode())
        h.update(b"|")
    return h.hexdigest()


def _file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# =============================================================================================
# 1. wiring
# =============================================================================================

def _read_synapse_columns(cols=("pre", "post", "type")):
    import pyarrow.parquet as pq
    tb = pq.read_table(SYNAPSES, columns=list(cols))
    out = {c: tb.column(c).to_numpy() for c in cols}
    del tb
    return out


def load_pairs(rebuild: bool = False) -> Dict[str, np.ndarray]:
    """Per-pair synapse counts over the whole brain of record.

    Returns ids (uint64, every segment in pre U post, sorted), pre/post (int32 indices into ids),
    t1/t2 (int32 type-1 / type-2 synapse counts of each pair), and n_synapses (int).
    """
    NET.mkdir(parents=True, exist_ok=True)
    names = ("ids", "pre", "post", "t1", "t2")
    paths = {k: NET / f"pairs_{k}.npy" for k in names}
    meta_p = NET / "pairs_meta.json"
    if not rebuild and all(p.exists() for p in paths.values()) and meta_p.exists():
        out = {k: np.load(paths[k], mmap_mode=None) for k in names}
        out.update(json.load(open(meta_p)))
        return out
    t0 = time.time()
    col = _read_synapse_columns()
    pre_id, post_id, ty = col["pre"], col["post"], col["type"]
    n_syn = len(pre_id)
    types = sorted(int(t) for t in np.unique(ty))
    if types != [1, 2]:
        raise ValueError(f"unexpected type values {types}")
    ids = np.unique(np.concatenate([np.unique(pre_id), np.unique(post_id)]))
    pre = np.searchsorted(ids, pre_id).astype(np.int64)
    del pre_id
    post = np.searchsorted(ids, post_id).astype(np.int64)
    del post_id
    n = len(ids)
    key = pre * n + post
    del pre, post
    order = np.argsort(key, kind="stable")
    ks = key[order]
    del key
    t1_sorted = (ty[order] == 1).astype(np.int32)
    del order, ty
    start = np.flatnonzero(np.concatenate([[True], ks[1:] != ks[:-1]]))
    uk = ks[start]
    del ks
    cnt = np.diff(np.concatenate([start, [n_syn]])).astype(np.int32)
    t1 = np.add.reduceat(t1_sorted, start).astype(np.int32)
    del t1_sorted
    t2 = cnt - t1
    out = {"ids": ids, "pre": (uk // n).astype(np.int32), "post": (uk % n).astype(np.int32), "t1": t1, "t2": t2}
    del uk
    for k in names:
        np.save(paths[k], out[k])
    meta = {"n_synapses": int(n_syn), "n_pairs": int(len(t1)), "n_segments": int(n),
            "type1_synapses": int(t1.sum()), "type2_synapses": int(t2.sum()),
            "build_seconds": round(time.time() - t0, 1), "source": str(SYNAPSES.name)}
    json.dump(meta, open(meta_p, "w"), indent=1)
    out.update(meta)
    _log("pairs built", meta)
    return out


def segment_output_types(P: Dict[str, np.ndarray]):
    """Per segment: (type-1 output synapses, type-2 output synapses)."""
    n = len(P["ids"])
    o1 = np.bincount(P["pre"], weights=P["t1"], minlength=n).astype(np.int64)
    o2 = np.bincount(P["pre"], weights=P["t2"], minlength=n).astype(np.int64)
    return o1, o2


def signed_counts(P: Dict[str, np.ndarray], rule: str = SIGN_RULE):
    """Signed synapse count per pair under a sign rule. Returns (signed int32, info dict).

    dale_majority: every output of a segment takes the sign of the majority type over ALL of that
        segment's output synapses (type 2 -> +1, type 1 -> -1). Ties -> the pair keeps the
        per-synapse net (t2 - t1), so a tied segment with a one-each pair contributes 0 there.
    per_synapse:   each synapse signs itself; a pair's weight is t2 - t1 (B2's default rule).
    """
    if rule not in SIGN_RULES:
        raise ValueError(rule)
    t1, t2 = P["t1"].astype(np.int64), P["t2"].astype(np.int64)
    per_syn = SIGN_MAP[2] * t2 + SIGN_MAP[1] * t1
    info = {"rule": rule, "sign_map": {str(k): v for k, v in SIGN_MAP.items()}}
    if rule == "per_synapse":
        s = per_syn
    else:
        o1, o2 = segment_output_types(P)
        seg_sign = np.where(o2 > o1, SIGN_MAP[2], np.where(o1 > o2, SIGN_MAP[1], 0)).astype(np.int64)
        ps = seg_sign[P["pre"]]
        s = np.where(ps != 0, ps * (t1 + t2), per_syn)
        has_out = (o1 + o2) > 0
        info.update({"segments_with_outputs": int(has_out.sum()),
                     "segments_excitatory": int((has_out & (seg_sign > 0)).sum()),
                     "segments_inhibitory": int((has_out & (seg_sign < 0)).sum()),
                     "segments_tied": int((has_out & (seg_sign == 0)).sum()),
                     "synapses_from_tied_segments": int((t1 + t2)[ps == 0].sum()),
                     "synapses_resigned_by_dale": int(np.where(ps > 0, t1, np.where(ps < 0, t2, 0)).sum())})
    s = s.astype(np.int32)
    info.update({"pairs": int(len(s)), "pairs_positive": int((s > 0).sum()), "pairs_negative": int((s < 0).sum()),
                 "pairs_zero_dropped": int((s == 0).sum()),
                 "synapses_in_zero_pairs": int((t1 + t2)[s == 0].sum())})
    return s, info


def ei_polarity_check(P: Optional[Dict[str, np.ndarray]] = None, write: bool = True) -> dict:
    """Output-type counts of the authors' ground-truth E/I cell lists (neuroglancer state
    evaluation_polarity_assignment.json, same agglomeration seg_241003_agg241003)."""
    P = load_pairs() if P is None else P
    d = json.load(open(POLARITY_STATE))
    o1, o2 = segment_output_types(P)
    ids = P["ids"]
    res = {}
    for l in d["layers"]:
        if l.get("type") != "segmentation" or not l.get("segments"):
            continue
        q = np.array([int(s) for s in l["segments"]], np.uint64)
        pos = np.searchsorted(ids, q)
        ok = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == q)
        i = pos[ok]
        a, b = o1[i], o2[i]
        w = (a + b) > 0
        res[l["name"]] = {"n_segments": int(len(q)), "segments_with_outputs": int(w.sum()),
                          "out_type1": int(a.sum()), "out_type2": int(b.sum()),
                          "cells_majority_type1": int((a > b).sum()), "cells_majority_type2": int((b > a).sum()),
                          "cells_tied_with_outputs": int(((a == b) & w).sum())}
    if write:
        json.dump(res, open(EI_CHECK, "w"), indent=1)
    return res


# =============================================================================================
# 2. identified cells and graph search
# =============================================================================================

def identified(cells: Optional[dict] = None) -> dict:
    cells = C.load() if cells is None else cells
    spn = cells["spn"]
    mir = [m for m in cells["spn_mirror_candidates"] if m.get("seg_id")]
    return {
        "mauthner": {s: int(cells["mauthner"][s]["seg_id"]) for s in ("left", "right")},
        "spn": [{"seg_id": int(s["seg_id"]), "side": s["side"], "class": s["class"], "name": s["name"],
                 "prox": bool(s["prox"]), "confirmed": True} for s in spn],
        "spn_mirror": [{"seg_id": int(m["seg_id"]), "side": m["side"], "class": m["class"], "name": m["name"],
                        "confirmed": False, "of_lore_id": m["of_lore_id"]} for m in mir],
        "tectum": C.tectum_seg_ids(cells).astype(np.uint64),
        "tectum_side": C.tectum_map(cells).side,
        "exclusions": np.array(sorted(C.never_stimulate(cells)), np.uint64),
    }


def _csr(n, src, dst):
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, dst[order].astype(np.int64)


def bfs_levels(n, indptr, indices, sources) -> np.ndarray:
    """Multi-source BFS hop count (int32, -1 = unreachable)."""
    dist = np.full(n, -1, np.int32)
    front = np.unique(np.asarray(sources, np.int64))
    dist[front] = 0
    lvl = 0
    while len(front):
        starts = indptr[front]
        cnt = indptr[front + 1] - starts
        tot = int(cnt.sum())
        if not tot:
            break
        csum = np.cumsum(cnt)
        g = np.arange(tot, dtype=np.int64) + np.repeat(starts - (csum - cnt), cnt)
        nb = np.unique(indices[g])
        nb = nb[dist[nb] < 0]
        lvl += 1
        dist[nb] = lvl
        front = nb
    return dist


def index_of(ids: np.ndarray, q) -> np.ndarray:
    q = np.atleast_1d(np.asarray(q, np.uint64))
    pos = np.searchsorted(ids, q)
    ok = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == q)
    return np.where(ok, pos, -1).astype(np.int64)




def readout_table(ident: dict) -> List[dict]:
    """Every readout cell: group (mauthner|spn|spn_mirror), side, class, name, confirmed."""
    rows = [{"seg_id": int(ident["mauthner"][s]), "group": "mauthner", "side": s, "class": "escape",
             "name": f"Mauthner-{s}", "confirmed": True} for s in ("left", "right")]
    rows += [{"seg_id": c["seg_id"], "group": "spn", "side": c["side"], "class": c["class"], "name": c["name"],
              "confirmed": True, "prox": c["prox"]} for c in ident["spn"]]
    rows += [{"seg_id": c["seg_id"], "group": "spn_mirror", "side": c["side"], "class": c["class"],
              "name": c["name"], "confirmed": False} for c in ident["spn_mirror"]]
    return rows


def readout_ids(ident: dict) -> Dict[str, np.ndarray]:
    return {"mauthner": np.array([ident["mauthner"]["left"], ident["mauthner"]["right"]], np.uint64),
            "spn": np.array([c["seg_id"] for c in ident["spn"]], np.uint64),
            "spn_mirror": np.array([c["seg_id"] for c in ident["spn_mirror"]], np.uint64)}


# =============================================================================================
# 3. the stimulated set and the exact functional subnetwork
# =============================================================================================

def stimulated_set(P: Dict[str, np.ndarray], ident: dict) -> Tuple[np.ndarray, dict]:
    """Indices (into P["ids"]) of the tectal cells that receive visual input.

    Rule (ours): B1's tectal cells, that have at least one synapse in the brain of record, minus
    B1's never-stimulate exclusions, minus every tectal cell with a direct synapse (any type) onto
    any readout. Raises if an excluded or readout cell would be stimulated.
    """
    ids = P["ids"]
    t_all = index_of(ids, ident["tectum"])
    present = t_all[t_all >= 0]
    side_of = dict(zip(ident["tectum"].tolist(), ident["tectum_side"].tolist()))
    excl = index_of(ids, ident["exclusions"])
    excl = excl[excl >= 0]
    ro = np.concatenate([v for v in readout_ids(ident).values()])
    ro_idx = index_of(ids, ro)
    ro_idx = ro_idx[ro_idx >= 0]
    is_ro = np.zeros(len(ids), bool)
    is_ro[ro_idx] = True
    is_t = np.zeros(len(ids), bool)
    is_t[present] = True
    nsyn = P["t1"].astype(np.int64) + P["t2"]
    direct = is_t[P["pre"]] & is_ro[P["post"]] & (nsyn > 0)
    removed = []
    for k in np.flatnonzero(direct):
        removed.append({"tectal_seg": int(ids[P["pre"][k]]), "lobe": side_of[int(ids[P["pre"][k]])],
                        "readout_seg": int(ids[P["post"][k]]), "synapses": int(nsyn[k]),
                        "type1": int(P["t1"][k]), "type2": int(P["t2"][k])})
    drop = np.zeros(len(ids), bool)
    drop[excl] = True
    drop[P["pre"][direct]] = True
    stim = np.unique(present[~drop[present]])
    if np.intersect1d(stim, excl).size or np.intersect1d(stim, ro_idx).size:
        raise AssertionError("an excluded or readout cell is in the stimulated set")
    info = {"tectum_listed": int(len(ident["tectum"])), "tectum_in_graph": int(len(present)),
            "excluded_overlap": int(np.intersect1d(present, excl).size),
            "direct_readout_partners_removed": int(np.unique(P["pre"][direct]).size),
            "direct_pairs": removed, "stimulated": int(len(stim)),
            "stimulated_left_lobe": int(sum(side_of[int(i)] == "left" for i in ids[stim])),
            "stimulated_right_lobe": int(sum(side_of[int(i)] == "right" for i in ids[stim]))}
    return stim, info


def reduce_graph(n: int, pre: np.ndarray, post: np.ndarray, signed: np.ndarray, nsyn: np.ndarray,
                 stim: np.ndarray, readouts: Dict[str, np.ndarray]) -> dict:
    """The exact functional subnetwork for input into `stim` (see module docstring).

    F = cells reachable from a stimulated cell through positive (excitatory) pairs.
    B = cells in F from which a readout in F is reachable through nonzero pairs inside F.
    keep = (F & B) | readouts. Returns index arrays into 0..n-1, hop counts and statistics.
    """
    pre = pre.astype(np.int64)
    post = post.astype(np.int64)
    nz = signed != 0
    all_ro = np.unique(np.concatenate([v[v >= 0] for v in readouts.values()]))
    pos = signed > 0
    ip, ix = _csr(n, pre[pos], post[pos])
    d_pos = bfs_levels(n, ip, ix, stim)
    del ip, ix
    ip, ix = _csr(n, pre[nz], post[nz])
    d_all = bfs_levels(n, ip, ix, stim)
    del ip, ix
    F = d_pos >= 0
    inF = nz & F[pre] & F[post]
    ip, ix = _csr(n, post[inF], pre[inF])
    d_back = bfs_levels(n, ip, ix, all_ro[F[all_ro]])
    del ip, ix
    keep = F & (d_back >= 0)
    keep[all_ro] = True
    keep_idx = np.flatnonzero(keep)
    pair_keep = nz & keep[pre] & keep[post] & F[pre]
    stats = {"segments_total": int(n), "pairs_total": int(len(signed)), "pairs_nonzero": int(nz.sum()),
             "synapses_in_nonzero_pairs": int(nsyn[nz].sum()),
             "stimulated": int(len(stim)), "can_fire_F": int(F.sum()), "F_pairs": int(inF.sum()),
             "F_and_reach_readout": int((F & (d_back >= 0)).sum()),
             "kept_neurons": int(len(keep_idx)), "kept_pairs": int(pair_keep.sum()),
             "kept_synapses": int(nsyn[pair_keep].sum()), "stimulated_kept": int(keep[stim].sum()),
             "readouts_can_fire": {k: [bool(F[i]) if i >= 0 else None for i in v.tolist()] for k, v in readouts.items()},
             "readouts_hops_excitatory": {k: [int(d_pos[i]) if i >= 0 else None for i in v.tolist()] for k, v in readouts.items()},
             "readouts_hops_any": {k: [int(d_all[i]) if i >= 0 else None for i in v.tolist()] for k, v in readouts.items()}}
    return {"keep_idx": keep_idx, "pair_keep": pair_keep, "F": F, "inF": inF, "d_pos": d_pos, "d_all": d_all,
            "stats": stats}


# =============================================================================================
# 4. network bundles (cached)
# =============================================================================================

@dataclass
class Bundle:
    """A simulator network with its stimulated cells and readouts, all by segment id."""
    net: S.Network
    stim_ids: np.ndarray            # uint64 segment ids of the driven tectal cells in net, sorted
    readouts: List[dict]            # readout_table rows present in net
    stats: dict
    variant: str
    rule: str

    def __post_init__(self):
        self.stim_idx = self.net.index_of(self.stim_ids)
        self.ro_ids = np.array([r["seg_id"] for r in self.readouts], np.uint64)
        self.ro_idx = self.net.index_of(self.ro_ids)

    def idx(self, seg_ids) -> np.ndarray:
        return self.net.index_of(seg_ids)


def _cache_key(rule: str, variant: str) -> str:
    meta = json.load(open(NET / "pairs_meta.json"))
    return _sha(CODE_VERSION, WIRING_CONTENT_HASH, meta["n_pairs"], meta["n_synapses"],
                _file_sha(C.CELLS_JSON), rule, variant, SHUFFLE_SEED, SUBNET_SHUFFLE_SEED)[:16]


def _save_bundle(path: Path, ids, pre, post, cnt, stim_ids, stats, key):
    np.savez(path, ids=ids, pre=pre.astype(np.int32), post=post.astype(np.int32), cnt=cnt.astype(np.int32),
             stim_ids=stim_ids)
    json.dump({"key": key, "stats": stats}, open(str(path).replace(".npz", ".json"), "w"), indent=1, default=int)


def _load_bundle(path: Path, key: str, ident: dict, rule: str, variant: str) -> Optional[Bundle]:
    js = Path(str(path).replace(".npz", ".json"))
    if not (path.exists() and js.exists()):
        return None
    meta = json.load(open(js))
    if meta.get("key") != key:
        return None
    d = np.load(path)
    net = S.Network.from_pairs(d["pre"], d["post"], d["cnt"], len(d["ids"]), ids=d["ids"],
                               meta={"rule": rule, "variant": variant, "sign_map": {str(k): v for k, v in SIGN_MAP.items()}})
    present = set(d["ids"].tolist())
    ro = [r for r in readout_table(ident) if r["seg_id"] in present]
    return Bundle(net, d["stim_ids"], ro, meta["stats"], variant, rule)


def _bundle_from_reduction(ids, pre, post, cnt, nsyn, stim, ident, rule, variant, extra) -> Tuple[Bundle, tuple]:
    ro = {k: index_of(ids, v) for k, v in readout_ids(ident).items()}
    R = reduce_graph(len(ids), pre, post, cnt, nsyn, stim, ro)
    keep, pk = R["keep_idx"], R["pair_keep"]
    loc = np.full(len(ids), -1, np.int64)
    loc[keep] = np.arange(len(keep))
    kids = ids[keep]
    lpre, lpost, lcnt = loc[pre[pk]], loc[post[pk]], cnt[pk]
    stim_ids = np.sort(ids[stim[np.isin(stim, keep)]])
    stats = dict(R["stats"])
    stats.update(extra)
    stats["mauthner_min_hops_from_stimulated"] = {s: stats["readouts_hops_any"]["mauthner"][i]
                                                  for i, s in enumerate(("left", "right"))}
    net = S.Network.from_pairs(lpre, lpost, lcnt, len(kids), ids=kids,
                               meta={"rule": rule, "variant": variant, "sign_map": {str(k): v for k, v in SIGN_MAP.items()}})
    present = set(kids.tolist())
    rows = [r for r in readout_table(ident) if r["seg_id"] in present]
    return Bundle(net, stim_ids, rows, stats, variant, rule), (kids, lpre, lpost, lcnt, stim_ids, R)


def build_bundle(rule: str = GATE_SIGN_RULE, variant: str = "intact", rebuild: bool = False,
                 keep_forward: bool = False):
    """The reduced network for a sign rule and a wiring variant, cached in data/net/.

    variant: "intact"      the brain of record
             "shuffled"    whole-brain degree-preserving shuffle (sim.Network.shuffled_copy over all
                           13.46M segments and every nonzero pair), then reduced again
             "subshuffled" the intact reduced network, shuffled inside itself (same neurons)
    keep_forward=True also returns the whole fireable set F as a Network (intact only), for the
    exactness test.
    """
    cells = C.load()
    ident = identified(cells)
    NET.mkdir(parents=True, exist_ok=True)
    key = _cache_key(rule, variant)
    path = NET / f"g1_{rule}_{variant}.npz"
    fpath = NET / f"g1_{rule}_{variant}_forward.npz"
    if not rebuild and not keep_forward:
        b = _load_bundle(path, key, ident, rule, variant)
        if b is not None:
            return b
    if not rebuild and keep_forward:
        b = _load_bundle(path, key, ident, rule, variant)
        fb = _load_bundle(fpath, key, ident, rule, variant + "_forward")
        if b is not None and fb is not None:
            return b, fb
    if variant == "subshuffled":
        base = build_bundle(rule, "intact")
        sh = base.net.shuffled_copy(SUBNET_SHUFFLE_SEED)
        rows = sh.rows()
        stats = dict(base.stats)
        stats["subnetwork_shuffle"] = sh.meta.get("shuffle")
        b = Bundle(sh, base.stim_ids, base.readouts, stats, variant, rule)
        _save_bundle(path, sh.ids, rows, sh.indices, sh.counts, base.stim_ids, stats, key)
        return b
    t0 = time.time()
    P = load_pairs()
    ids = P["ids"]
    signed, sinfo = signed_counts(P, rule)
    stim, stim_info = stimulated_set(P, ident)
    extra = {"sign_info": sinfo, "stim_info": stim_info, "rule": rule, "variant": variant}
    if variant == "intact":
        pre, post, cnt = P["pre"], P["post"], signed
        nsyn = P["t1"].astype(np.int64) + P["t2"]
    elif variant == "shuffled":
        nz = signed != 0
        full = S.Network.from_pairs(P["pre"][nz], P["post"][nz], signed[nz], len(ids), ids=ids)
        del P
        _log("whole-brain shuffle: n", full.n, "pairs", full.nnz)
        t1 = time.time()
        sh = full.shuffled_copy(SHUFFLE_SEED)
        extra["shuffle"] = dict(sh.meta.get("shuffle", {}), seconds=round(time.time() - t1, 1),
                                pairs=int(sh.nnz), neurons=int(sh.n))
        del full
        pre, post, cnt = sh.rows(), sh.indices.astype(np.int64), sh.counts
        nsyn = np.abs(cnt).astype(np.int64)
        del sh
    else:
        raise ValueError(variant)
    b, (kids, lpre, lpost, lcnt, stim_ids, R) = _bundle_from_reduction(ids, pre, post, cnt, nsyn, stim, ident, rule, variant, extra)
    b.stats["build_seconds"] = round(time.time() - t0, 1)
    _save_bundle(path, kids, lpre, lpost, lcnt, stim_ids, b.stats, key)
    if variant == "intact":
        # the whole fireable set F with every nonzero pair inside it (exactness test only)
        F, inF = R["F"], R["inF"]
        fidx = np.flatnonzero(F)
        loc = np.full(len(ids), -1, np.int64)
        loc[fidx] = np.arange(len(fidx))
        fs = {"n": int(len(fidx)), "pairs": int(inF.sum())}
        _save_bundle(fpath, ids[fidx], loc[pre[inF]], loc[post[inF]], cnt[inF], np.sort(ids[stim]), fs, key)
    if keep_forward:
        return b, _load_bundle(fpath, key, ident, rule, variant + "_forward")
    return b


def full_network(rule: str = GATE_SIGN_RULE) -> Tuple[S.Network, np.ndarray]:
    """The whole brain of record as one Network (every segment, every nonzero pair) and the
    internal indices of the stimulated cells in it. Used by the benchmark only."""
    P = load_pairs()
    ident = identified()
    signed, _ = signed_counts(P, rule)
    stim, _ = stimulated_set(P, ident)
    nz = signed != 0
    net = S.Network.from_pairs(P["pre"][nz], P["post"][nz], signed[nz], len(P["ids"]), ids=P["ids"])
    return net, np.sort(P["ids"][stim])


# =============================================================================================
# 5. the eye: stimulus frames -> tectal Poisson rates
# =============================================================================================

class Retina:
    """OFF-transient, centre-surround front end feeding cells.tectum_drive.

    Per eye, per lattice point p (B2's 2-degree lattice, points inside that eye's field only):
        c_p(t)  contrast of frame t (stimuli.py; 0 = background, -1 = black)
        hp_p    = c_p(t) - m_p ;  m_p += alpha (c_p(t) - m_p),  alpha = 1 - exp(-frame/tau_hp)
        y_p     = hp_p - sum_q S_pq hp_q     S = Gaussian surround (sigma 10 deg, cut at 30 deg,
                                            rows renormalised inside the eye's field, so any uniform
                                            change of the whole field gives y = 0 exactly)
        r_p     = w_off max(0, -y_p) + w_on max(0, y_p)
    tectal drive d = tectum_drive(points r) (linear in the point values; W is built from single-point
    calls to cells.tectum_drive and checked against direct calls), rate = r_max * clip(d, 0, 1).
    """

    def __init__(self, stim_ids: np.ndarray, cells: Optional[dict] = None, params: RetinaParams = RETINA,
                 eye: ST.EyeModel = ST.EyeModel(), cache: bool = True):
        self.p = params
        self.eye = eye
        self.cells = C.load() if cells is None else cells
        self.stim_ids = np.asarray(stim_ids, np.uint64)
        tm = C.tectum_map(self.cells)
        order = np.argsort(tm.seg_ids)
        pos = np.searchsorted(tm.seg_ids[order], self.stim_ids)
        if np.any(tm.seg_ids[order][np.minimum(pos, len(order) - 1)] != self.stim_ids):
            raise ValueError("a stimulated id is not a tectal cell")
        self.rows = order[pos]                       # rows of tectum_drive's output
        self.K = len(self.stim_ids)
        az, el, xyz = ST.lattice(params.res_deg)
        self.az, self.el, self.xyz = az, el, xyz
        masks = ST.eye_masks(params.res_deg, eye)
        self.pts = {e: np.flatnonzero(masks[e]) for e in ST.EYES}
        self.key_sorted, self.key_order = self._keys(az, el)
        self.alpha = 1.0 - math.exp(-params.frame_ms / params.tau_hp_ms)
        self.S = {e: self._surround(self.pts[e]) for e in ST.EYES}
        self.W = self._load_or_build_W(cache)

    @staticmethod
    def _key(az, el):
        return np.round(np.asarray(az) * 1e6).astype(np.int64) * 1_000_000_000 + np.round((np.asarray(el) + 90.0) * 1e6).astype(np.int64)

    def _keys(self, az, el):
        k = self._key(az, el)
        o = np.argsort(k)
        return k[o], o

    def lattice_index(self, az, el) -> np.ndarray:
        k = self._key(az, el)
        pos = np.searchsorted(self.key_sorted, k)
        if np.any(self.key_sorted[np.minimum(pos, len(self.key_sorted) - 1)] != k):
            raise ValueError("stimulus point not on the lattice")
        return self.key_order[pos]

    def _surround(self, pts):
        import scipy.sparse as sp
        from scipy.spatial import cKDTree
        xyz = self.xyz[pts]
        chord = 2.0 * math.sin(math.radians(self.p.surround_radius_deg) / 2.0)
        pairs = cKDTree(xyz).query_pairs(chord, output_type="ndarray")
        i = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(len(pts))])
        j = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(len(pts))])
        dot = np.clip((xyz[i] * xyz[j]).sum(1), -1.0, 1.0)
        d = np.degrees(np.arccos(dot))
        w = np.exp(-d ** 2 / (2.0 * self.p.surround_sigma_deg ** 2))
        M = sp.csr_matrix((w, (i, j)), shape=(len(pts), len(pts)))
        M.sort_indices()
        rs = np.asarray(M.sum(1)).ravel()
        M = sp.diags(1.0 / rs) @ M
        return M.tocsr()

    def _load_or_build_W(self, cache):
        key = _sha("W", _file_sha(C.CELLS_JSON), self.stim_ids, self.p.res_deg, dataclasses.asdict(self.eye))[:16]
        path = NET / f"retina_W_{key}.npz"
        if cache and path.exists():
            d = np.load(path)
            return {e: d[e] for e in ST.EYES}
        W = {}
        for e in ST.EYES:
            cols = np.zeros((self.K, len(self.pts[e])), np.float64)
            for j, p in enumerate(self.pts[e]):
                cols[:, j] = C.tectum_drive({e: [(float(self.az[p]), float(self.el[p]), 1.0)]}, self.cells)[self.rows]
            W[e] = cols
        if cache:
            NET.mkdir(parents=True, exist_ok=True)
            np.savez(path, **W)
        return W

    def drive_of_points(self, eye_points: Dict[str, np.ndarray]) -> np.ndarray:
        """Tectal drive (stimulated cells) for per-eye lattice values through W."""
        d = np.zeros(self.K)
        with np.errstate(all="ignore"):   # macOS Accelerate raises spurious FP flags in matmul
            for e, v in eye_points.items():
                d += self.W[e] @ v
        if not np.isfinite(d).all():
            raise FloatingPointError("non-finite tectal drive")
        return d

    def frame_contrast(self, stim, i) -> Dict[str, np.ndarray]:
        arr = stim.arrays(i)
        out = {}
        for e in ST.EYES:
            c = np.zeros(len(self.pts[e]))
            a = arr[e]
            if len(a):
                g = self.lattice_index(a[:, 0], a[:, 1])
                loc = np.searchsorted(self.pts[e], g)
                if np.any(self.pts[e][np.minimum(loc, len(self.pts[e]) - 1)] != g):
                    raise ValueError("stimulus point outside the eye mask")
                c[loc] = a[:, 2]
            out[e] = c
        return out

    def rates(self, stim) -> Tuple[np.ndarray, dict]:
        """(F, K) Poisson rates in Hz, one row per stimulus frame; plus per-frame diagnostics."""
        if abs(stim.dt_ms - self.p.frame_ms) > 1e-9:
            raise ValueError(f"stimulus frames are {stim.dt_ms} ms; the retina runs at {self.p.frame_ms} ms")
        n = len(stim)
        drive = np.zeros((n, self.K))
        m = {e: np.zeros(len(self.pts[e])) for e in ST.EYES}
        off_tot = np.zeros(n)
        for i in range(n):
            c = self.frame_contrast(stim, i)
            for e in ST.EYES:
                hp = c[e] - m[e]
                m[e] += self.alpha * hp
                if not np.any(np.abs(hp) > self.p.zero_tol):
                    continue
                y = hp - self.S[e] @ hp
                y[np.abs(y) <= self.p.zero_tol] = 0.0
                r = self.p.w_off * np.maximum(0.0, -y) + self.p.w_on * np.maximum(0.0, y)
                if r.any():
                    off_tot[i] += r.sum()
                    with np.errstate(all="ignore"):   # spurious Accelerate FP flags; checked below
                        drive[i] += self.W[e] @ r
        if not np.isfinite(drive).all():
            raise FloatingPointError("non-finite tectal drive")
        rates = self.p.r_max_hz * np.clip(drive, 0.0, 1.0)
        q = self.p.rate_round_hz
        rates = np.round(rates / q) * q
        diag = {"retina_output_per_frame": off_tot, "tectal_drive_sum_per_frame": drive.sum(1),
                "cells_driven_per_frame": (rates > 0).sum(1),
                "expected_tectal_input_spikes": float(rates.sum() * self.p.frame_ms / 1000.0),
                "rates_sha256": hashlib.sha256(rates.astype("<f8").tobytes()).hexdigest()}
        return rates, diag


# =============================================================================================
# 6. protocol stimuli
# =============================================================================================

def _az(side: str, az: float) -> float:
    if side not in ("left", "right"):
        raise ValueError(side)
    return az if side == "right" else -az


def stim_loom(side: str, params: RetinaParams = RETINA):
    L = PROTOCOL["loom"]
    return ST.looming(_az(side, L["azimuth_deg"]), L["elevation_deg"], L["l_over_v_ms"], L["start_deg"], L["end_deg"],
                      L["contrast"], pre_ms=L["pre_ms"], hold_ms=L["hold_ms"], dt_ms=params.frame_ms,
                      res_deg=params.res_deg)


def stim_recede(side: str, params: RetinaParams = RETINA):
    """The loom reversed (140 -> 4 deg), preceded by the 140 deg disc standing still for
    static_ms after pre_ms of blank, then held at 4 deg for hold_ms."""
    L = PROTOCOL["loom"]
    hold0 = PROTOCOL["recede"]["static_ms"]
    _, t_end = ST.loom_schedule(L["l_over_v_ms"], L["start_deg"], L["end_deg"])
    dt = params.frame_ms
    n = ST._n_frames(L["pre_ms"] + hold0 + t_end + L["hold_ms"], dt)
    t = ST._times(n, dt)
    rel = t - L["pre_ms"] - hold0
    size = ST.loom_angle_deg(np.clip(t_end - rel, 0.0, None), L["l_over_v_ms"], L["start_deg"], L["end_deg"])
    size = np.where(rel < 0, L["end_deg"], np.where(rel > t_end, L["start_deg"], size))
    visible = t >= L["pre_ms"]
    size = np.where(visible, size, 0.0)
    params_d = dict(azimuth_deg=_az(side, L["azimuth_deg"]), elevation_deg=L["elevation_deg"],
                    l_over_v_ms=L["l_over_v_ms"], start_deg=L["start_deg"], end_deg=L["end_deg"],
                    contrast=L["contrast"], pre_ms=L["pre_ms"], static_ms=hold0, hold_ms=L["hold_ms"],
                    expansion_ms=t_end, note="G1 receding with a static onset period (brain.py)")
    return ST.Stimulus("receding", dt, ST._disc_trace(t, _az(side, L["azimuth_deg"]), L["elevation_deg"], size,
                                                       L["contrast"], visible), params_d, ST.EyeModel(), params.res_deg)


def stim_dim(params: RetinaParams = RETINA):
    L, D = PROTOCOL["loom"], PROTOCOL["dim"]
    return ST.dimming(L["l_over_v_ms"], L["start_deg"], L["end_deg"], eyes=D["eyes"], mode=D["mode"],
                      pre_ms=D["pre_ms"], hold_ms=D["hold_ms"], dt_ms=params.frame_ms, res_deg=params.res_deg)


def stim_prey(side: str, params: RetinaParams = RETINA):
    Pp = PROTOCOL["prey"]
    return ST.prey_dot(_az(side, Pp["azimuth_deg"]), Pp["elevation_deg"], Pp["size_deg"], Pp["speed_deg_s"],
                       Pp["sweep_deg"], Pp["contrast"], Pp["duration_ms"], direction=1 if side == "right" else -1,
                       pre_ms=Pp["pre_ms"], dt_ms=params.frame_ms, res_deg=params.res_deg)


def stim_blank(params: RetinaParams = RETINA):
    return ST.blank(PROTOCOL["blank"]["duration_ms"], dt_ms=params.frame_ms, res_deg=params.res_deg)


def windows_ms() -> dict:
    """Analysis windows [start, stop) in ms from stimulus start, per condition."""
    L = PROTOCOL["loom"]
    _, t_end = ST.loom_schedule(L["l_over_v_ms"], L["start_deg"], L["end_deg"])
    a = GATE["sensorimotor_allow_ms"]
    s0 = L["pre_ms"]
    r0 = L["pre_ms"] + PROTOCOL["recede"]["static_ms"]
    return {"loom": (s0, s0 + t_end + a), "dim": (PROTOCOL["dim"]["pre_ms"], PROTOCOL["dim"]["pre_ms"] + t_end + a),
            "recede_onset": (s0, r0), "recede_motion": (r0, r0 + t_end + a),
            "prey": (PROTOCOL["prey"]["pre_ms"], PROTOCOL["prey"]["pre_ms"] + PROTOCOL["prey"]["duration_ms"]),
            "blank": (PROTOCOL["blank"]["pre_ms"], PROTOCOL["blank"]["duration_ms"]),
            "loom_expansion_ms": t_end}


def loom_angle_at(t_ms: float) -> float:
    """Loom disc angular size (deg) at time t_ms from stimulus start (0 before onset)."""
    L = PROTOCOL["loom"]
    rel = t_ms - L["pre_ms"]
    if rel < 0:
        return 0.0
    return float(ST.loom_angle_deg(rel, L["l_over_v_ms"], L["start_deg"], L["end_deg"]))


# =============================================================================================
# 7. trials and decisions
# =============================================================================================

@dataclass
class Trial:
    condition: str
    seed: str
    spike_hash: str
    rates_sha256: str
    n_spikes: int
    tectal_spikes: int
    readout_spikes: Dict[int, np.ndarray]   # seg id -> spike times (ms from stimulus start)
    duration_ms: float
    wall_s: float
    cpu_s: float
    n_steps: int


def run_trial(bundle: Bundle, rates: np.ndarray, seed: str, condition: str = "", frame_ms: float = RETINA.frame_ms,
              lesion_ids: Optional[Sequence[int]] = None, params: S.LIFParams = S.LIFParams(),
              rates_sha: str = "") -> Trial:
    net = bundle.net if not lesion_ids else bundle.net.lesion(np.asarray(lesion_ids, np.uint64))
    sim = S.Simulator(net, seed=seed, params=params)
    sim.add_poisson(bundle.stim_idx, rates, frame_ms=frame_ms, name="retina")
    dur = rates.shape[0] * frame_ms
    c0, w0 = time.process_time(), time.perf_counter()
    r = sim.run(dur)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    sel = np.isin(r.neurons, bundle.ro_idx)
    rn, rs = r.neurons[sel], r.steps[sel]
    ro = {}
    for sid, i in zip(bundle.ro_ids.tolist(), bundle.ro_idx.tolist()):
        ro[int(sid)] = rs[rn == i] * params.dt_ms
    is_stim = np.zeros(net.n, bool)
    is_stim[bundle.stim_idx] = True
    return Trial(condition, seed, S.spike_hash(r), rates_sha, int(len(r.neurons)), int(is_stim[r.neurons].sum()),
                 ro, dur, wall, cpu, r.n_steps)


def mauthner_ids(bundle: Bundle) -> Dict[str, int]:
    return {r["side"]: r["seg_id"] for r in bundle.readouts if r["group"] == "mauthner"}


def escape_decision(trial: Trial, bundle: Bundle, window: Tuple[float, float]) -> dict:
    """First Mauthner spike in [start, stop): side, time, spike counts per side."""
    M = mauthner_ids(bundle)
    lo, hi = window
    first, counts = {}, {}
    for side, sid in M.items():
        t = trial.readout_spikes.get(sid, np.zeros(0))
        t = t[(t >= lo) & (t < hi)]
        counts[side] = int(len(t))
        if len(t):
            first[side] = float(t.min())
    if not first:
        return {"escape": None, "t_ms": None, "counts": counts}
    tmin = min(first.values())
    sides = sorted(s for s, v in first.items() if v == tmin)
    return {"escape": sides[0] if len(sides) == 1 else "both", "t_ms": tmin, "counts": counts}


def turning_cells(bundle: Bundle) -> Dict[str, List[int]]:
    out = {"left": [], "right": []}
    for r in bundle.readouts:
        if r["group"] in ("spn", "spn_mirror") and r["class"] == "turning":
            out[r["side"]].append(r["seg_id"])
    return out


def turn_signal(trial: Trial, bundle: Bundle, window: Tuple[float, float]) -> dict:
    """Turn index (R - L) / (R + L) of mean spikes per turning SPN (RoV3/MiV1/MiV2) per side.
    Huang et al. 2013: these cells fire for turns to their own side. Positive = right turn."""
    lo, hi = window
    tc = turning_cells(bundle)
    per = {}
    mean = {}
    for side, ids in tc.items():
        n = [int(((trial.readout_spikes[s] >= lo) & (trial.readout_spikes[s] < hi)).sum()) for s in ids]
        per[side] = n
        mean[side] = float(np.mean(n)) if n else 0.0
    R, L = mean["right"], mean["left"]
    ti = 0.0 if R + L == 0 else (R - L) / (R + L)
    return {"turn": None if ti == 0 else ("right" if ti > 0 else "left"), "index": ti, "mean_right": R,
            "mean_left": L, "per_cell": per}


def seeds(n: Optional[int] = None) -> List[str]:
    return [GATE["seed_fmt"].format(k=k) for k in range(GATE["n_seeds"] if n is None else n)]


def majority(values: Sequence) -> Optional[str]:
    vals = [v for v in values]
    if not vals:
        return None
    u = sorted(set(map(str, vals)))
    c = [sum(str(v) == x for v in vals) for x in u]
    best = max(c)
    tops = [x for x, k in zip(u, c) if k == best]
    return tops[0] if len(tops) == 1 else "tie:" + "/".join(tops)


# =============================================================================================
# 8. graph checks (B1 section 8) and a wiring-level direction prediction
# =============================================================================================

def mauthner_graph_checks(P: Optional[Dict[str, np.ndarray]] = None, ident: Optional[dict] = None) -> dict:
    """B1 section 8.1-8.3 on the full parquet: input/output counts, input rank among segments with
    a hindbrain soma, inputs from spiral-fiber-cluster somas, and lateral-dendrite inputs."""
    import pandas as pd
    P = load_pairs() if P is None else P
    ident = identified() if ident is None else ident
    cells = C.load()
    ids = P["ids"]
    n = len(ids)
    nsyn = P["t1"].astype(np.int64) + P["t2"]
    indeg = np.bincount(P["post"], weights=nsyn, minlength=n).astype(np.int64)
    outdeg = np.bincount(P["pre"], weights=nsyn, minlength=n).astype(np.int64)
    in1 = np.bincount(P["post"], weights=P["t1"], minlength=n).astype(np.int64)
    in2 = np.bincount(P["post"], weights=P["t2"], minlength=n).astype(np.int64)
    out = {}

    def cnt(sid):
        i = int(index_of(ids, sid)[0])
        return {"in": int(indeg[i]), "in_type1": int(in1[i]), "in_type2": int(in2[i]), "out": int(outdeg[i])}

    for s in ("left", "right"):
        out[f"mauthner_{s}"] = dict(seg_id=int(ident["mauthner"][s]), **cnt(ident["mauthner"][s]))
    out["runner_ups"] = {str(s): cnt(s) for s in (17405827637, 16039136570)}
    out["spn_158660"] = cnt(20607607971)
    # 8.2 rank among segments with a hindbrain soma (atlas mece0 label 6 at 4096 nm)
    somas = C._somas()
    agg = C._agg_map()
    a0 = np.load(C.REF / "mece0_231218_4096.npy", mmap_mode="r")
    pos = np.vstack(somas.pt_position.values).astype(float)
    q = np.floor(pos * np.array(C.VOXEL_NM) / np.array([4096, 4096, 3840])).astype(int)
    q = np.clip(q, 0, np.array(a0.shape) - 1)
    lab = np.asarray(a0[q[:, 0], q[:, 1], q[:, 2]])
    hb_lore = somas.id.values[lab == 6]
    hb_seg = np.unique(agg.reindex(hb_lore).dropna().astype(np.int64).values.astype(np.uint64))
    hi = index_of(ids, hb_seg)
    hi = hi[hi >= 0]
    hb_in = np.sort(indeg[hi])[::-1]
    out["hindbrain_soma_segments"] = int(len(hi))
    for s in ("left", "right"):
        v = out[f"mauthner_{s}"]["in"]
        rank = int((hb_in > v).sum()) + 1
        out[f"mauthner_{s}"]["rank_among_hindbrain_soma_segments"] = rank
        out[f"mauthner_{s}"]["top_percent"] = round(100.0 * rank / len(hi), 4)
    # 8.3 spiral fiber neuron clusters (mece2 29, 32) -> Mauthner
    a2 = np.load(C.REF / "mece2_231218_4096.npy", mmap_mode="r")
    lab2 = np.asarray(a2[q[:, 0], q[:, 1], q[:, 2]])
    sfn_lore = somas.id.values[np.isin(lab2, [29, 32])]
    sfn_seg = np.unique(agg.reindex(sfn_lore).dropna().astype(np.int64).values.astype(np.uint64))
    si = index_of(ids, sfn_seg)
    si = si[si >= 0]
    is_sfn = np.zeros(n, bool)
    is_sfn[si] = True
    out["spiral_fiber_cluster_somas"] = int(len(sfn_lore))
    out["spiral_fiber_cluster_segments"] = int(len(si))
    for s in ("left", "right"):
        m = int(index_of(ids, ident["mauthner"][s])[0])
        sel = (P["post"] == m) & is_sfn[P["pre"]]
        out[f"mauthner_{s}"]["inputs_from_spiral_fiber_cluster_segments"] = {
            "segments": int(sel.sum()), "synapses": int(nsyn[sel].sum()),
            "type2": int(P["t2"][sel].sum()), "type1": int(P["t1"][sel].sum())}
    # lateral dendrite: synapse positions onto each Mauthner farther than 50 um from the midline
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    tb = pq.read_table(SYNAPSES, columns=["pre", "post", "x", "y", "z", "type"],
                       filters=[("post", "in", [int(ident["mauthner"]["left"]), int(ident["mauthner"]["right"])])])
    df = tb.to_pandas()
    ax = cells["axes"]
    ymid = C.midline_y(df.x.values, ax)
    lat_um = np.abs(df.y.values - ymid) * C.VOXEL_NM[1] / 1000.0
    has_soma = set(agg.values.astype(np.int64).tolist())
    for s in ("left", "right"):
        d = df[df.post == ident["mauthner"][s]]
        lu = lat_um[df.post.values == ident["mauthner"][s]]
        far = d[lu > 50.0]
        out[f"mauthner_{s}"]["inputs_beyond_50um_lateral"] = {
            "synapses": int(len(far)), "pre_segments": int(far.pre.nunique()),
            "pre_segments_with_a_soma": int(sum(int(p) in has_soma for p in far.pre.unique())),
            "type2": int((far.type == 2).sum()), "max_lateral_um": round(float(lu.max()), 1)}
        cap = d[(d.x >= 70144) & (d.x <= 71168) & (d.y >= 29696) & (d.y <= 36352) & (d.z >= 5248) & (d.z <= 5760)]
        out[f"mauthner_{s}"]["inputs_in_axon_cap_box"] = {"synapses": int(len(cap)),
                                                         "from_spiral_fiber_cluster": int(np.isin(cap.pre.values.astype(np.uint64), sfn_seg).sum())}
    return out


def lobe_prediction(rule: str = GATE_SIGN_RULE) -> dict:
    """Wiring-only prediction: stimulate one tectal lobe at a time and count, per Mauthner cell,
    the excitatory input synapses from cells that can fire, and the minimum hops."""
    P = load_pairs()
    ident = identified()
    ids = P["ids"]
    signed, _ = signed_counts(P, rule)
    stim, _ = stimulated_set(P, ident)
    side = dict(zip(ident["tectum"].tolist(), ident["tectum_side"].tolist()))
    stim_side = np.array([side[int(i)] for i in ids[stim]])
    pre = P["pre"].astype(np.int64)
    post = P["post"].astype(np.int64)
    pos = signed > 0
    ip, ix = _csr(len(ids), pre[pos], post[pos])
    res = {}
    for lobe in ("left", "right"):
        d = bfs_levels(len(ids), ip, ix, stim[stim_side == lobe])
        F = d >= 0
        row = {}
        for s in ("left", "right"):
            m = int(index_of(ids, ident["mauthner"][s])[0])
            sel = (post == m) & F[pre]
            row[f"mauthner_{s}"] = {"exc_synapses_from_fireable": int(signed[sel & pos].sum()),
                                    "inh_synapses_from_fireable": int(-signed[sel & (signed < 0)].sum()),
                                    "hops": int(d[m])}
        row["fireable_cells"] = int(F.sum())
        res[f"{lobe}_lobe_only"] = row
    return res


# =============================================================================================
# 9. the gate protocol: conditions, parallel runner, pre-registered evaluation
# =============================================================================================

CONDITIONS = {
    "loom_left": ("loom", "left"), "loom_right": ("loom", "right"),
    "recede_left": ("recede", "left"), "recede_right": ("recede", "right"),
    "dim": ("dim", None), "prey_left": ("prey", "left"), "prey_right": ("prey", "right"),
    "blank": ("blank", None),
}
GATE_RUNS = [  # (rule, variant, lesion, conditions, n_seeds[, LIF overrides])
    ("per_synapse", "intact", False, list(CONDITIONS), GATE["n_seeds"]),
    ("per_synapse", "intact", True, ["loom_left", "loom_right"], 4),
    ("per_synapse", "shuffled", False, ["loom_left", "loom_right", "prey_left", "prey_right"], GATE["n_seeds"]),
]
SENSITIVITY_RUNS = [
    ("per_synapse", "subshuffled", False, ["loom_left", "loom_right", "prey_left", "prey_right"], GATE["n_seeds"]),
    ("dale_majority", "intact", False, ["loom_left", "loom_right", "prey_left", "prey_right", "recede_right", "dim"], GATE["n_seeds"]),
]

# ---- the one logged change after run 1 (G1-gate.md s.5.2), fixed before it was simulated ----
# Fish1's automated agglomeration attaches far fewer synapses to each neuron than FlyWire's
# proofread neurons, for which Shiu calibrated w_syn. Scale w_syn by the ratio of synapses per
# neuron: FlyWire ~5e7 chemical synapses / ~130,000 neurons (bioRxiv 2023.06.27.546656 v2 abstract,
# read 2026-09-25) = 384.6; Fish1 mean input synapses per soma-bearing segment = 53.2 (all 180,782
# segments in agglomerated_segments_and_soma_ids.csv, zeros included; measured 2026-09-25).
FLYWIRE_SYNAPSES_PER_NEURON = 5e7 / 1.3e5
FISH1_INPUT_SYNAPSES_PER_SOMA_SEGMENT = 53.2
ANCHORED_W_SYN_MV = round(0.275 * FLYWIRE_SYNAPSES_PER_NEURON / FISH1_INPUT_SYNAPSES_PER_SOMA_SEGMENT, 4)
CHANGE_RUNS = [
    ("per_synapse", "intact", False, list(CONDITIONS), GATE["n_seeds"], {"w_syn_mV": ANCHORED_W_SYN_MV}),
]


def run_key(rule, variant, lesion=False, lif=None) -> str:
    k = f"{rule}/{variant}" + ("/lesion_mauthner" if lesion else "")
    if lif:
        k += "/" + ",".join(f"{a}={b}" for a, b in sorted(lif.items()))
    return k


def make_stim(cond: str, params: RetinaParams = RETINA):
    kind, side = CONDITIONS[cond]
    return {"loom": lambda: stim_loom(side, params), "recede": lambda: stim_recede(side, params),
            "dim": lambda: stim_dim(params), "prey": lambda: stim_prey(side, params),
            "blank": lambda: stim_blank(params)}[kind]()


def window_for(cond: str) -> Tuple[float, float]:
    w = windows_ms()
    kind = CONDITIONS[cond][0]
    return {"loom": w["loom"], "recede": w["recede_motion"], "dim": w["dim"], "prey": w["prey"], "blank": w["blank"]}[kind]


def cached_rates(retina: Retina, cond: str, use_cache: bool = True) -> Tuple[np.ndarray, dict]:
    """Retina rates for a condition. A pure function of (W, retina params, stimulus); cached on disk."""
    stim = make_stim(cond, retina.p)
    key = _sha(CODE_VERSION, retina.p.digest(), stim.digest(),
               hashlib.sha256(b"".join(retina.W[e].tobytes() for e in ST.EYES)).hexdigest(), retina.stim_ids)[:20]
    path = NET / "rates" / f"{cond}_{key}.npz"
    if use_cache and path.exists():
        d = np.load(path, allow_pickle=False)
        diag = json.loads(str(d["diag"]))
        if hashlib.sha256(d["rates"].astype("<f8").tobytes()).hexdigest() == diag["rates_sha256"]:
            return d["rates"], diag
    rates, diag = retina.rates(stim)
    dj = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in diag.items()}
    dj["stimulus"] = stim.describe()
    dj["stimulus_digest"] = stim.digest()
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, rates=rates, diag=np.array(json.dumps(dj, default=float)))
    return rates, dj


def summarize(trial: Trial, bundle: Bundle, cond: str) -> dict:
    kind = CONDITIONS[cond][0]
    w = window_for(cond)
    esc = escape_decision(trial, bundle, w)
    turn = turn_signal(trial, bundle, w)
    M = mauthner_ids(bundle)
    out = {"condition": cond, "seed": trial.seed, "spike_hash": trial.spike_hash, "rates_sha256": trial.rates_sha256,
           "n_spikes": trial.n_spikes, "tectal_spikes": trial.tectal_spikes, "cpu_s": round(trial.cpu_s, 3),
           "wall_s": round(trial.wall_s, 3), "sim_ms": trial.duration_ms, "window_ms": list(w),
           "escape": esc["escape"], "escape_t_ms": esc["t_ms"], "mauthner_counts_in_window": esc["counts"],
           "mauthner_spike_times_ms": {s: [round(float(x), 1) for x in trial.readout_spikes.get(sid, [])] for s, sid in M.items()},
           "turn": turn["turn"], "turn_index": turn["index"], "turn_mean_right": turn["mean_right"],
           "turn_mean_left": turn["mean_left"], "turn_per_cell": turn["per_cell"],
           "readout_spikes_total": {str(k): int(len(v)) for k, v in trial.readout_spikes.items() if len(v)}}
    if kind == "loom" and esc["t_ms"] is not None:
        out["escape_angle_deg"] = round(loom_angle_at(esc["t_ms"]), 2)
        out["escape_t_before_expansion_end_ms"] = round(windows_ms()["loom"][0] + windows_ms()["loom_expansion_ms"] - esc["t_ms"], 1)
    if kind == "recede":
        on = escape_decision(trial, bundle, windows_ms()["recede_onset"])
        out["onset_escape"] = on["escape"]
        out["onset_mauthner_counts"] = on["counts"]
    return out


def _work(job) -> dict:
    """One (rule, variant, lesion, condition, seeds, lif) unit; runs in a worker process."""
    rule, variant, lesion, cond, seed_list, lif = job
    params = S.LIFParams(**(lif or {}))
    b = build_bundle(rule, variant)
    ret = Retina(b.stim_ids)
    rates, diag = cached_rates(ret, cond)
    les = list(mauthner_ids(b).values()) if lesion else None
    trials = []
    for sd in seed_list:
        tr = run_trial(b, rates, sd, cond, lesion_ids=les, params=params, rates_sha=diag["rates_sha256"])
        trials.append(summarize(tr, b, cond))
    small = {k: diag[k] for k in ("expected_tectal_input_spikes", "rates_sha256", "stimulus_digest")}
    small["peak_cells_driven"] = int(max(diag["cells_driven_per_frame"]))
    small["peak_tectal_drive_sum"] = float(max(diag["tectal_drive_sum_per_frame"]))
    small["retina_output_total"] = float(sum(diag["retina_output_per_frame"]))
    return {"rule": rule, "variant": variant, "lesion": bool(lesion), "lif": lif or {}, "condition": cond,
            "trials": trials, "diag": small, "lif_digest": params.digest()}


def run_protocol(runs=GATE_RUNS, workers: int = 4) -> Dict[str, dict]:
    """Every (run, condition) unit, in a process pool. Keys: 'rule/variant[/lesion]'."""
    import concurrent.futures as cf
    import multiprocessing as mp
    for rule, variant, *_ in runs:   # build caches once, serially (memory)
        b = build_bundle(rule, variant)
        Retina(b.stim_ids)
    jobs = [(r[0], r[1], r[2], cond, seeds(r[4]), (r[5] if len(r) > 5 else None)) for r in runs for cond in r[3]]
    out: Dict[str, dict] = {}
    if workers <= 1:
        res = [_work(j) for j in jobs]
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_work, jobs))
    for r in res:
        out.setdefault(run_key(r["rule"], r["variant"], r["lesion"], r["lif"]), {})[r["condition"]] = r
    return out


def decision_vector(conds: Dict[str, dict]) -> Dict[str, Optional[str]]:
    d = {}
    for c in ("loom_left", "loom_right"):
        if c in conds:
            d[c] = majority([t["escape"] for t in conds[c]["trials"]])
    for c in ("prey_left", "prey_right"):
        if c in conds:
            d[c] = majority([t["turn"] for t in conds[c]["trials"]])
    return d


def _m_total(t):
    return sum(t["mauthner_counts_in_window"].values())


def evaluate(results: Dict[str, dict], lif: Optional[dict] = None) -> dict:
    """Pre-registered gate criteria (GATE, s.3 of G1-gate.md) applied to run_protocol() output.
    lif: the LIF overrides of the run to evaluate (None = Shiu defaults)."""
    g = GATE
    I = results[run_key(GATE_SIGN_RULE, "intact", False, lif)]
    ev = {}
    # (a) looming
    for side in ("left", "right"):
        tr = I[f"loom_{side}"]["trials"]
        n = len(tr)
        esc = [t for t in tr if t["escape"] is not None]
        pred = side if MAUTHNER_PREDICTION == "ipsilateral_to_stimulus" else ("left" if side == "right" else "right")
        side_ok = [t for t in esc if t["escape"] == pred]
        timing_ok = [t for t in esc if t.get("escape_angle_deg", 0) >= g["crit_angle_min_deg"]]
        p = len(esc) / n
        sf = len(side_ok) / len(esc) if esc else 0.0
        tf = len(timing_ok) / len(esc) if esc else 0.0
        ev[f"a_loom_{side}"] = {
            "pass": bool(p >= g["p_escape_min"] and sf >= g["side_frac_min"] and tf >= g["timing_frac_min"]),
            "p_escape": p, "predicted_mauthner": pred, "first_mauthner": [t["escape"] for t in tr],
            "side_frac": sf, "timing_frac": tf, "angles_deg": [t.get("escape_angle_deg") for t in tr],
            "mauthner_spikes_per_trial": [t["mauthner_counts_in_window"] for t in tr],
            "mean_mauthner_spikes": float(np.mean([_m_total(t) for t in tr]))}
    # controls: receding (per side) and dimming (vs mean of both looms)
    loom_mean = {s: ev[f"a_loom_{s}"]["mean_mauthner_spikes"] for s in ("left", "right")}
    loom_p = {s: ev[f"a_loom_{s}"]["p_escape"] for s in ("left", "right")}
    for side in ("left", "right"):
        tr = I[f"recede_{side}"]["trials"]
        m = float(np.mean([_m_total(t) for t in tr]))
        p = float(np.mean([t["escape"] is not None for t in tr]))
        ok = loom_mean[side] > 0 and m <= g["control_ratio_max"] * loom_mean[side] and p < loom_p[side]
        ev[f"control_recede_{side}"] = {"pass": bool(ok), "mean_mauthner_spikes": m, "p_escape": p,
                                        "loom_mean_mauthner_spikes": loom_mean[side], "loom_p_escape": loom_p[side],
                                        "onset_escapes": [t["onset_escape"] for t in tr],
                                        "onset_mauthner_counts": [t["onset_mauthner_counts"] for t in tr]}
    tr = I["dim"]["trials"]
    m = float(np.mean([_m_total(t) for t in tr]))
    p = float(np.mean([t["escape"] is not None for t in tr]))
    lm, lp = float(np.mean(list(loom_mean.values()))), float(np.mean(list(loom_p.values())))
    ev["control_dim"] = {"pass": bool(lm > 0 and m <= g["control_ratio_max"] * lm and p < lp),
                         "mean_mauthner_spikes": m, "p_escape": p, "loom_mean_mauthner_spikes": lm, "loom_p_escape": lp}
    # (b) prey turn
    for side in ("left", "right"):
        tr = I[f"prey_{side}"]["trials"]
        good = [t for t in tr if (t["turn_index"] > 0 if side == "right" else t["turn_index"] < 0)]
        f = len(good) / len(tr)
        ev[f"b_prey_{side}"] = {"pass": bool(f >= g["turn_frac_min"]), "frac_predicted_sign": f,
                                "turn_index": [round(t["turn_index"], 4) for t in tr],
                                "mean_right_cell_spikes": [t["turn_mean_right"] for t in tr],
                                "mean_left_cell_spikes": [t["turn_mean_left"] for t in tr],
                                "mauthner_spikes": [t["mauthner_counts_in_window"] for t in tr]}
    # paired within-cell (reported, not gating)
    pr, pl = I["prey_right"]["trials"], I["prey_left"]["trials"]
    rR = [sum(a["turn_per_cell"]["right"]) - sum(b["turn_per_cell"]["right"]) for a, b in zip(pr, pl)]
    rL = [sum(b["turn_per_cell"]["left"]) - sum(a["turn_per_cell"]["left"]) for a, b in zip(pr, pl)]
    ev["b_paired_within_cell"] = {"right_cells_preyR_minus_preyL": rR, "left_cells_preyL_minus_preyR": rL,
                                  "frac_seeds_right_cells_prefer_right": float(np.mean([x > 0 for x in rR])),
                                  "frac_seeds_left_cells_prefer_left": float(np.mean([x > 0 for x in rL]))}
    # no stimulus
    tr = I["blank"]["trials"]
    turns = [t["turn"] for t in tr]
    fr = max(sum(x == "right" for x in turns), sum(x == "left" for x in turns)) / len(tr)
    vac = all(t["n_spikes"] == 0 for t in tr)
    ev["control_blank"] = {"pass": bool(fr < g["turn_frac_min"]), "turns": turns, "vacuous_no_spikes": vac,
                           "total_spikes": [t["n_spikes"] for t in tr],
                           "mauthner_spikes": [_m_total(t) for t in tr]}
    # (c) lesion
    L = results.get(run_key(GATE_SIGN_RULE, "intact", True, lif), {})
    lt = [t for c in L.values() for t in c["trials"]]
    ev["c_lesion"] = {"pass": bool(lt) and all(_m_total(t) == 0 and t["escape"] is None for t in lt),
                      "trials": len(lt), "mauthner_spikes": [_m_total(t) for t in lt],
                      "note": "true by construction: a lesioned cell loses its synapses and may not fire"}
    # (d) shuffle
    Sh = results.get(run_key(GATE_SIGN_RULE, "shuffled", False, lif), {})
    dv_i = decision_vector(I)
    dv_s = decision_vector(Sh)
    ev["d_shuffle"] = {"pass": bool(Sh) and dv_s != {k: dv_i[k] for k in dv_s}, "intact": dv_i, "shuffled": dv_s,
                       "shuffled_mean_mauthner_spikes": {c: float(np.mean([_m_total(t) for t in Sh[c]["trials"]]))
                                                         for c in Sh if c.startswith("loom")}}
    ev["downstream_spikes"] = {c: [t["n_spikes"] - t["tectal_spikes"] for t in I[c]["trials"]] for c in I}
    ev["gate_pass"] = bool(all(v["pass"] for k, v in ev.items() if isinstance(v, dict) and "pass" in v))
    return ev


def gate_run(workers: int = 4, write: bool = True, runs=GATE_RUNS) -> dict:
    t0 = time.time()
    res = run_protocol(runs, workers=workers)
    ev = evaluate(res) if runs is GATE_RUNS else None
    if runs is CHANGE_RUNS:
        ev = evaluate(res, lif=CHANGE_RUNS[0][5])
    out = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
           "loadavg_end": os.getloadavg(), "retina": dataclasses.asdict(RETINA), "protocol": PROTOCOL, "gate": GATE,
           "windows_ms": windows_ms(), "results": res, "evaluation": ev}
    if write:
        name = {id(GATE_RUNS): "g1_gate_results.json", id(CHANGE_RUNS): "g1_change_results.json"}.get(id(runs), "g1_sensitivity_results.json")
        json.dump(out, open(NET / name, "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FISHBRAIN G1 gate")
    ap.add_argument("what", choices=["gate", "change", "sensitivity", "build", "checks"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.what == "build":
        for r, v in (("per_synapse", "intact"), ("per_synapse", "shuffled"), ("per_synapse", "subshuffled"),
                     ("dale_majority", "intact")):
            b = build_bundle(r, v)
            _log(r, v, b.net.n, b.net.nnz, len(b.stim_ids))
    elif a.what == "checks":
        P = load_pairs()
        out = {"graph_checks": mauthner_graph_checks(P), "lobe_prediction_per_synapse": lobe_prediction("per_synapse"),
               "lobe_prediction_dale_majority": lobe_prediction("dale_majority"), "ei_polarity_check": ei_polarity_check(P)}
        json.dump(out, open(NET / "g1_graph_checks.json", "w"), indent=1, default=int)
    elif a.what == "gate":
        r = gate_run(a.workers)
        print(json.dumps({k: v.get("pass") for k, v in r["evaluation"].items() if isinstance(v, dict)}, indent=1))
        print("gate_pass", r["evaluation"]["gate_pass"])
    elif a.what == "change":
        r = gate_run(a.workers, runs=CHANGE_RUNS)
        print(json.dumps({k: v.get("pass") for k, v in r["evaluation"].items() if isinstance(v, dict)}, indent=1))
    else:
        r = gate_run(a.workers, runs=SENSITIVITY_RUNS)
        print(json.dumps({k: decision_vector(v) for k, v in r["results"].items()}, indent=1))
