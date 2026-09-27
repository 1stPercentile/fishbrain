"""FISHBRAIN G1b repair probe (task P): can Fish1's cut axons be reattached to their somas?

Ground truth is the Fish1 authors' proofread hindbrain neurons (``hindbrain_reconstructions``,
labelled by lore id). Every number this module writes goes into ``evidence/G1b-repair-probe.md``;
the pre-registration there (section 0) fixes the sample, the truth definitions, the three rules,
the abstention sweep, the calibration/test split and the verdict before anything here ran.

Stages (``python -m fishbrain.repair_probe <stage>``; each caches its output under ``data/repair/``):

1. ``select``        50 fully reconstructed neurons (all 5 fully reconstructed SPNs + 45 seeded draws),
                     their LOD-0 proofread meshes, and the mip-0 chunks those meshes touch.
2. ``region``        every synapse of the brain of record inside the sampled neurons' box + 60 um, with
                     q_pre (p0 pushed 40 nm back into the pre side) and q_post (p1 pushed 20 nm on).
3. ``validate``      the points read against the agglomeration itself on 200 random synapses;
   ``validate80``    the same for the 80 nm pre point that the truth uses (``qpre80``).
4. ``label_mp``      the proofread label at every q point inside the touched chunks (process pool,
                     resumable per block; ``label`` is the slower threaded original, kept for reference;
                     ``assemble`` rebuilds the label arrays from the block files).
5. ``truth``         per neuron: clean outputs, member fragments, damage, anchor purity, and the positive
                     controls (anchor is N's, HMI hand-annotated outputs, absent chunks).
6. ``rules_compute`` R1 (nearest soma body), R2 (nearest soma-bearing arbor), R3 (arbor + direction)
                     over every output-bearing fragment in the region.
7. ``evaluate``      the abstention sweep, the calibration/test split, the nulls and the verdict.
8. ``report``        the markdown tables quoted in the evidence note (``data/repair/report.md``).

Units: synapse-table voxels are 8 x 8 x 30 nm; segmentation mip 0 is 16 x 16 x 30 nm; every
distance here is in nm unless named otherwise.
"""
from __future__ import annotations

import ast
import json
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
REF = DATA / "ref"
OUT = DATA / "repair"
MESH_DIR = OUT / "meshes"
PUBLIC = "precomputed://gs://fish1-public/"
PROOF = "hindbrain_reconstructions"
AGG = "seg_241003_agg241003"

VOX8 = np.array([8.0, 8.0, 30.0])      # synapse-table voxel (nm)
MIP0 = np.array([16.0, 16.0, 30.0])    # mip-0 voxel of both segmentation layers (nm)
CHUNK = np.array([128, 128, 64])       # mip-0 chunk (voxels), both layers
SEG_SIZE = np.array([140000, 32500, 8689])

SEED = 20260926
N_SPN_EXPECTED = 5
N_OTHERS = 45
FULL_STATUS = "soma, dendrite(reconstructed), axon(reconstructed)"
PRE_SHIFT_NM = 40.0
POST_SHIFT_NM = 20.0
REGION_MARGIN_NM = 60_000.0
CAP_NM = 50_000.0
T_GRID = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
D1_FLOOR_NM = 100.0
R3_MIN_POINTS = 3
R3_MIN_EXPLAINED = 0.80
R3_MIN_SPAN_NM = 2_000.0
R3_CONE_DEG = 45.0
KNN = 16            # neighbours per point for the nearest / runner-up distinct segment
KNN_R3 = 64         # neighbours per fragment end for the R3 cone search
PRECISION_BAR = 0.80
UNREAD = np.uint64(np.iinfo(np.uint64).max)
# The HMI dataframe's synapse coordinates are y-mirrored against the volume: y_volume = 65000 - y_hmi
# (8 nm voxels; 65,000 = the volume's y extent). Fitted 2026-09-26 02:27 EDT on the 707 hand-annotated
# outputs of the 11 sampled neurons that have them, against the proofread meshes (not against any rule
# output): unflipped, 0/707 lie within 1 um of their own neuron's mesh; flipped, 706/707 (median 0.07 um).
HMI_Y_FLIP = 65000.0
LABEL_THREADS = 12  # per worker process; chunk reads are latency-bound (measured 2026-09-26)


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _cv(layer, mip=0, fill_missing=True):
    from cloudvolume import CloudVolume
    return CloudVolume(PUBLIC + layer, mip=mip, use_https=True, progress=False, fill_missing=fill_missing)


def _save_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def _load_json(name):
    return json.load(open(OUT / name))


def chunk_key(c: np.ndarray) -> np.ndarray:
    """(cx, cy, cz) mip-0 chunk indices -> one int64 key."""
    c = np.asarray(c, dtype=np.int64)
    return (c[..., 0] << 32) | (c[..., 1] << 16) | c[..., 2]


def key_chunk(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.int64)
    return np.stack([k >> 32, (k >> 16) & 0xFFFF, k & 0xFFFF], -1)


def nm_to_chunk(p_nm: np.ndarray) -> np.ndarray:
    return (np.floor(p_nm / MIP0) // CHUNK).astype(np.int64)


# ---------------------------------------------------------------------------------------------
# 1. select
# ---------------------------------------------------------------------------------------------

def _agg_map():
    import pandas as pd
    ag = pd.read_csv(REF / "agglomerated_segments_and_soma_ids.csv")
    return dict(zip(ag.lores_id.astype(int), ag.hires_id_agglo.astype(np.uint64)))


def select() -> dict:
    import pandas as pd
    df = pd.read_excel(REF / "em_zfish1_dataframe.xlsx")
    df["lore"] = df["Cell ID"].astype(int)
    full = df[df.reconstruction_status == FULL_STATUS]
    agg = _agg_map()
    is_spn = full.classifier.astype(str).str.startswith("spn")
    spn = sorted(full.loc[is_spn, "lore"].tolist())
    others = np.array(sorted(full.loc[~is_spn, "lore"].tolist()))
    order = np.random.default_rng(SEED).permutation(others)
    cv = _cv(PROOF)
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    info = df.set_index("lore")
    chosen, skipped = [], []

    def take(lore, group):
        a = int(agg.get(int(lore), 0))
        if a == 0:
            skipped.append({"lore": int(lore), "why": "no agg segment in csv"})
            return False
        try:
            m = cv.mesh.get(int(lore), lod=0)
            m = m[int(lore)] if isinstance(m, dict) else m
            v = np.asarray(m.vertices, dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            skipped.append({"lore": int(lore), "why": f"mesh error {type(e).__name__}"})
            return False
        if len(v) == 0:
            skipped.append({"lore": int(lore), "why": "empty mesh"})
            return False
        np.savez_compressed(MESH_DIR / f"{int(lore)}.npz", v=v)
        ch = nm_to_chunk(v.astype(np.float64))
        u, c = np.unique(chunk_key(ch), return_counts=True)
        row = info.loc[int(lore)]
        chosen.append({"lore": int(lore), "anchor": a, "group": group, "classifier": str(row.classifier),
                       "network_level": str(row.network_level), "nt": str(row.final_neurotransmitter_ID),
                       "n_vertices": int(len(v)), "bbox_nm_lo": v.min(0).tolist(), "bbox_nm_hi": v.max(0).tolist(),
                       "n_chunks": int(len(u)), "chunk_keys": u.tolist(), "chunk_vertex_counts": c.tolist(),
                       "has_hmi_outputs": bool(isinstance(row.outputs, str))})
        _log("selected", lore, group, len(v), "vertices", len(u), "chunks")
        return True

    for lore in spn:
        take(lore, "spn")
    n_other = 0
    draws = []
    for lore in order:
        if n_other >= N_OTHERS:
            break
        draws.append(int(lore))
        if take(lore, "other"):
            n_other += 1
    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "seed": SEED, "status": FULL_STATUS,
           "n_full": int(len(full)), "n_spn_full": len(spn), "spn": spn, "draw_order_used": draws,
           "skipped": skipped, "neurons": chosen,
           "status_counts": df.reconstruction_status.value_counts().to_dict()}
    _save_json("selection.json", out)
    return out


# ---------------------------------------------------------------------------------------------
# 2. region
# ---------------------------------------------------------------------------------------------

def region_box(sel: dict):
    lo = np.min([n["bbox_nm_lo"] for n in sel["neurons"]], 0) - REGION_MARGIN_NM
    hi = np.max([n["bbox_nm_hi"] for n in sel["neurons"]], 0) + REGION_MARGIN_NM
    return lo, hi


def region(sel: dict | None = None) -> dict:
    """Every synapse whose p0 lies in the region box, with q_pre and q_post (nm, float32)."""
    import pyarrow.parquet as pq
    sel = sel or _load_json("selection.json")
    lo, hi = region_box(sel)
    f = pq.ParquetFile(DATA / "synapses_raw.parquet")
    keep = {k: [] for k in ("syn_id", "pre", "post", "qpre", "qpost")}
    n_zero = 0
    for g in range(f.metadata.num_row_groups):
        t = f.read_row_group(g, columns=["syn_id", "pre", "post", "x0", "y0", "z0", "x1", "y1", "z1"])
        p0 = np.stack([t.column(c).to_numpy() for c in ("x0", "y0", "z0")], 1).astype(np.float64) * VOX8
        m = np.all((p0 >= lo) & (p0 <= hi), 1)
        if not m.any():
            continue
        p0 = p0[m]
        p1 = np.stack([t.column(c).to_numpy()[m] for c in ("x1", "y1", "z1")], 1).astype(np.float64) * VOX8
        d = p0 - p1
        L = np.linalg.norm(d, axis=1)
        u = np.zeros_like(d)
        nz = L > 0
        u[nz] = d[nz] / L[nz, None]
        n_zero += int((~nz).sum())
        keep["qpre"].append((p0 + PRE_SHIFT_NM * u).astype(np.float32))
        keep["qpost"].append((p1 - POST_SHIFT_NM * u).astype(np.float32))
        keep["syn_id"].append(t.column("syn_id").to_numpy()[m].astype(np.uint32))
        keep["pre"].append(t.column("pre").to_numpy()[m])
        keep["post"].append(t.column("post").to_numpy()[m])
        _log("row group", g, "kept", int(m.sum()))
        del t
    for k, v in keep.items():
        np.save(OUT / f"region_{k}.npy", np.concatenate(v))
    stats = {"box_nm_lo": lo.tolist(), "box_nm_hi": hi.tolist(), "n_synapses": int(sum(len(x) for x in keep["pre"])),
             "zero_length_lines": n_zero, "pre_shift_nm": PRE_SHIFT_NM, "post_shift_nm": POST_SHIFT_NM}
    _save_json("region.json", stats)
    return stats


def load_region() -> Dict[str, np.ndarray]:
    return {k: np.load(OUT / f"region_{k}.npy", mmap_mode="r") for k in ("syn_id", "pre", "post", "qpre", "qpost")}


SIDES = ("qpre", "qpost", "qpre80")


def qpre80(R) -> np.ndarray:
    """q_pre pushed a further 40 nm back along the line (80 nm behind p0 in total).

    Added 2026-09-26 01:55 EDT after the agglomeration check (validate.json) showed q_pre reads the
    post cell 27/200 times; logged in G1b section 7. q_pre - q_post = (|p0 - p1| + 60 nm) x unit, so
    its direction is the line's own; zero-length lines stay put."""
    a = np.asarray(R["qpre"], dtype=np.float64)
    d = a - np.asarray(R["qpost"], dtype=np.float64)
    L = np.linalg.norm(d, axis=1)
    out = a.copy()
    nz = L > 0
    out[nz] += 40.0 * d[nz] / L[nz, None]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------------------------
# 3. validate the point mapping against the agglomeration
# ---------------------------------------------------------------------------------------------

def validate(n: int = 200, names=("qpre", "qpost", "p0", "p1")) -> dict:
    R = load_region()
    rng = np.random.default_rng(SEED + 2)
    idx = np.sort(rng.choice(len(R["pre"]), n, replace=False))
    import pyarrow.parquet as pq  # raw p0/p1 for the same synapses, for comparison
    sid = np.asarray(R["syn_id"][idx]).astype(np.int64)
    raw = pq.read_table(DATA / "synapses_raw.parquet", columns=["x0", "y0", "z0", "x1", "y1", "z1"])
    p0 = np.stack([raw.column(c).to_numpy()[sid] for c in ("x0", "y0", "z0")], 1) * VOX8
    p1 = np.stack([raw.column(c).to_numpy()[sid] for c in ("x1", "y1", "z1")], 1) * VOX8
    del raw
    cv = _cv(AGG)
    allpts = {"qpre": np.asarray(R["qpre"][idx], float), "qpost": np.asarray(R["qpost"][idx], float), "p0": p0, "p1": p1,
              "qpre80": qpre80({"qpre": R["qpre"][idx], "qpost": R["qpost"][idx]}).astype(float)}
    pts = {k: allpts[k] for k in names}
    res = {}
    for name, P in pts.items():
        vox = np.floor(P / MIP0).astype(int)
        lab = cv.scattered_points([tuple(v) for v in vox])
        res[name] = np.array([int(lab[tuple(v)]) for v in vox], dtype=np.uint64)
        _log("validate", name, "done")
    pre = np.asarray(R["pre"][idx])
    post = np.asarray(R["post"][idx])
    out = {"n": n, "seed": SEED + 2}
    for name in pts:
        out[name] = {"reads_pre": int((res[name] == pre).sum()), "reads_post": int((res[name] == post).sum()),
                     "reads_zero": int((res[name] == 0).sum()),
                     "reads_other": int(((res[name] != pre) & (res[name] != post) & (res[name] != 0)).sum())}
    _save_json("validate.json" if tuple(names) == ("qpre", "qpost", "p0", "p1") else f"validate_{'_'.join(names)}.json", out)
    return out


# ---------------------------------------------------------------------------------------------
# 4. label every synapse point in the touched chunks with the proofread segmentation
# ---------------------------------------------------------------------------------------------

def label(workers: int = 16, fill_missing: bool = True) -> dict:
    sel = _load_json("selection.json")
    R = load_region()
    keys = np.unique(np.concatenate([np.array(n["chunk_keys"], dtype=np.int64) for n in sel["neurons"]]))
    _log("chunks to read", len(keys))
    labs = {}
    groups = {}
    P = {"qpre": R["qpre"], "qpost": R["qpost"], "qpre80": qpre80(R)}
    for side in SIDES:
        ck = chunk_key(nm_to_chunk(np.asarray(P[side], dtype=np.float64)))
        inset = np.isin(ck, keys)
        ii = np.nonzero(inset)[0]
        o = np.argsort(ck[ii], kind="stable")
        ii = ii[o]
        kk = ck[ii]
        starts = np.searchsorted(kk, keys, "left")
        ends = np.searchsorted(kk, keys, "right")
        groups[side] = (ii, starts, ends)
        labs[side] = np.full(len(ck), UNREAD, dtype=np.uint64)
        _log(side, "points in touched chunks", len(ii))
    local = threading.local()
    chunk_labels: List = [None] * len(keys)

    def work(j):
        if not hasattr(local, "cv"):
            local.cv = _cv(PROOF, fill_missing=fill_missing)
        c = key_chunk(keys[j])
        lo = c * CHUNK
        hi = np.minimum(lo + CHUNK, SEG_SIZE)
        for attempt in range(4):
            try:
                a = np.asarray(local.cv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])[..., 0]
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    return j, None, type(e).__name__
                time.sleep(2 * (attempt + 1))
        for side in SIDES:
            ii, s, e = groups[side]
            sub = ii[s[j]:e[j]]
            if len(sub) == 0:
                continue
            v = np.floor(np.asarray(P[side][sub], dtype=np.float64) / MIP0).astype(np.int64) - lo
            v = np.clip(v, 0, np.array(a.shape) - 1)
            labs[side][sub] = a[v[:, 0], v[:, 1], v[:, 2]]
        u, cnt = np.unique(a, return_counts=True)
        return j, dict(zip(u.tolist(), cnt.tolist())), None

    t0 = time.time()
    errors = []
    with ThreadPoolExecutor(workers) as ex:
        for k, (j, lab, err) in enumerate(ex.map(work, range(len(keys)))):
            chunk_labels[j] = lab
            if err:
                errors.append((int(keys[j]), err))
            if k % 1000 == 0:
                _log("chunk", k, "of", len(keys), f"{time.time() - t0:.0f} s")
    for side in SIDES:
        np.save(OUT / f"label_{side}.npy", labs[side])
    json.dump({"keys": keys.tolist(), "labels": [None if c is None else {str(a): b for a, b in c.items()}
                                                   for c in chunk_labels]}, open(OUT / "chunk_labels.json", "w"))
    stats = {"n_chunks": int(len(keys)), "seconds": round(time.time() - t0, 1), "workers": workers,
             "errors": errors, "points_labelled": {s: int(len(groups[s][0])) for s in groups},
             "fill_missing": fill_missing}
    _save_json("label.json", stats)
    return stats


def _label_part(job):
    """Worker process: read one block of mip-0 chunks and label the points that fall in them."""
    part, keys, pts, fill_missing = job
    path = OUT / "label_parts" / f"part_{part:04d}.npz"
    if path.exists():
        return part, len(keys), "cached"
    import fastremap
    cv = _cv(PROOF, fill_missing=fill_missing)
    res = {side: np.full(len(idx), UNREAD, dtype=np.uint64) for side, (idx, vox, ck) in pts.items()}
    order = {side: np.argsort(ck, kind="stable") for side, (idx, vox, ck) in pts.items()}
    bounds = {side: (np.searchsorted(ck[order[side]], keys, "left"), np.searchsorted(ck[order[side]], keys, "right"))
              for side, (idx, vox, ck) in pts.items()}
    ch_lab, errors = [], []

    def one(j):
        c = key_chunk(keys[j])
        lo = c * CHUNK
        hi = np.minimum(lo + CHUNK, SEG_SIZE)
        for attempt in range(4):
            try:
                return j, np.asarray(cv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])[..., 0], lo
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    return j, type(e).__name__, lo
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(LABEL_THREADS) as ex:
        for j, a, lo in ex.map(one, range(len(keys))):
            if isinstance(a, str):
                errors.append((int(keys[j]), a))
                ch_lab.append("")
                continue
            for side, (idx, vox, ck) in pts.items():
                s0, s1 = bounds[side][0][j], bounds[side][1][j]
                sub = order[side][s0:s1]
                if len(sub) == 0:
                    continue
                v = np.clip(vox[sub].astype(np.int64) - lo, 0, np.array(a.shape) - 1)
                res[side][sub] = a[v[:, 0], v[:, 1], v[:, 2]]
            ch_lab.append(",".join(str(x) for x in fastremap.unique(a).tolist()))
    np.savez(path, keys=keys, chunk_labels=np.array(ch_lab), errors=np.array([str(e) for e in errors]),
             **{f"{side}_idx": pts[side][0] for side in pts}, **{f"{side}_lab": res[side] for side in pts})
    return part, len(keys), f"{len(errors)} errors"


def label_mp(procs: int = 12, parts: int = 96, fill_missing: bool = True) -> dict:
    """Process-parallel version of ``label`` (the threaded one was GIL-bound at 6.4 chunks/s)."""
    from multiprocessing import get_context
    sel = _load_json("selection.json")
    R = load_region()
    keys = np.unique(np.concatenate([np.array(n["chunk_keys"], dtype=np.int64) for n in sel["neurons"]]))
    (OUT / "label_parts").mkdir(parents=True, exist_ok=True)
    P = {"qpre": R["qpre"], "qpost": R["qpost"], "qpre80": qpre80(R)}
    side_pts = {}
    for side in SIDES:
        q = np.asarray(P[side], dtype=np.float64)
        ck = chunk_key(nm_to_chunk(q))
        ii = np.nonzero(np.isin(ck, keys))[0]
        side_pts[side] = (ii, np.floor(q[ii] / MIP0).astype(np.int32), ck[ii])
        _log(side, "points in touched chunks", len(ii))
    blocks = np.array_split(keys, parts)
    jobs = []
    for b, kb in enumerate(blocks):
        pts = {}
        for side, (ii, vox, ck) in side_pts.items():
            m = np.isin(ck, kb)
            pts[side] = (ii[m], vox[m], ck[m])
        jobs.append((b, kb, pts, fill_missing))
    t0 = time.time()
    done = 0
    with get_context("spawn").Pool(procs) as pool:
        for part, n, msg in pool.imap_unordered(_label_part, jobs):
            done += n
            _log("part", part, n, "chunks", msg, "| total", done, "of", len(keys), f"{time.time() - t0:.0f} s")
    return assemble_labels(keys, t0, procs, parts, fill_missing)


def assemble_labels(keys=None, t0=None, procs=None, parts=None, fill_missing=True) -> dict:
    R = load_region()
    n = len(R["pre"])
    labs = {side: np.full(n, UNREAD, dtype=np.uint64) for side in SIDES}
    all_keys, all_lab, errors = [], [], []
    for f in sorted((OUT / "label_parts").glob("part_*.npz")):
        d = np.load(f)
        for side in SIDES:
            labs[side][d[f"{side}_idx"]] = d[f"{side}_lab"]
        all_keys.append(d["keys"])
        all_lab += d["chunk_labels"].tolist()
        errors += d["errors"].tolist()
    for side in SIDES:
        np.save(OUT / f"label_{side}.npy", labs[side])
    ak = np.concatenate(all_keys)
    json.dump({"keys": ak.tolist(), "labels": all_lab}, open(OUT / "chunk_labels.json", "w"))
    stats = {"n_chunks": int(len(ak)), "seconds": None if t0 is None else round(time.time() - t0, 1),
             "procs": procs, "parts": parts, "errors": errors, "fill_missing": fill_missing,
             "points_labelled": {s: int((labs[s] != UNREAD).sum()) for s in SIDES},
             "unique_keys": int(len(np.unique(ak)))}
    _save_json("label.json", stats)
    return stats


# ---------------------------------------------------------------------------------------------
# 5. truth and damage
# ---------------------------------------------------------------------------------------------

def _seg_totals(segs: np.ndarray) -> np.ndarray:
    """n_in + n_out from cells.parquet for each segment id (0 when absent)."""
    import pyarrow.parquet as pq
    t = pq.read_table(DATA / "cells.parquet", columns=["seg_id", "n_in", "n_out"])
    sid = t.column("seg_id").to_numpy()
    tot = t.column("n_in").to_numpy() + t.column("n_out").to_numpy()
    o = np.argsort(sid)
    sid, tot = sid[o], tot[o]
    i = np.clip(np.searchsorted(sid, segs), 0, len(sid) - 1)
    return np.where(sid[i] == segs, tot[i], 0)


def soma_segments() -> np.ndarray:
    a = np.array(list(_agg_map().values()), dtype=np.uint64)
    return np.unique(a[a != 0])


def truth() -> dict:
    sel = _load_json("selection.json")
    R = load_region()
    pre = np.asarray(R["pre"])
    post = np.asarray(R["post"])
    # Clean reads (G1b section 7, 01:57 EDT): the pre side is q_pre pushed 80 nm back (85% reads the
    # pre segment in the agglomeration check), and a point only votes when the two ends of its line
    # read different labels. An input onto N whose pre point strays into N therefore does not count.
    lp_raw = np.load(OUT / "label_qpre80.npy")
    lq_raw = np.load(OUT / "label_qpost.npy")
    same = lp_raw == lq_raw
    lp = np.where(same, np.uint64(0), lp_raw)
    lq = np.where(same, np.uint64(0), lq_raw)
    lp[lp == UNREAD] = 0
    lq[lq == UNREAD] = 0
    somas = soma_segments()
    qpre = qpre80(R)
    import pandas as pd
    sdf = pd.read_parquet(REF / "cave_somas_full.parquet", columns=["id", "pt_position"])
    soma_pos = dict(zip(sdf.id.astype(int), [np.asarray(v, float) * VOX8 for v in sdf.pt_position.values]))
    rows_out = {}
    lores = np.array([n["lore"] for n in sel["neurons"]], dtype=np.uint64)
    # every (segment, label) vote once: outputs vote through q_pre, inputs through q_post
    seg_all = np.concatenate([pre, post])
    lab_all = np.concatenate([lp, lq])
    m = np.isin(lab_all, lores)
    seg_v, lab_v = seg_all[m], lab_all[m]
    pair = np.stack([seg_v, lab_v], 1)
    upair, cnt = np.unique(pair, axis=0, return_counts=True)
    tot = _seg_totals(upair[:, 0])
    frac = cnt / np.maximum(tot, 1)
    member = frac > 0.5
    chunk_info = json.load(open(OUT / "chunk_labels.json"))
    ck_index = {k: i for i, k in enumerate(chunk_info["keys"])}
    per = []
    fragments = {}
    import pandas as pd
    df = pd.read_excel(REF / "em_zfish1_dataframe.xlsx")
    df["lore"] = df["Cell ID"].astype(int)
    hmi = df.set_index("lore")
    for n in sel["neurons"]:
        L = np.uint64(n["lore"])
        A = np.uint64(n["anchor"])
        mine = upair[:, 1] == L
        segs, segc, segtot, segf = upair[mine, 0], cnt[mine], tot[mine], frac[mine]
        is_mem = segf > 0.5
        mem = segs[is_mem]
        mem_soma = np.isin(mem, somas)
        out_rows = np.nonzero(lp == L)[0]
        out_pre = pre[out_rows]
        on_anchor = out_pre == A
        on_member_frag = np.isin(out_pre, mem[~mem_soma])
        on_other_soma = np.isin(out_pre, somas) & ~on_anchor
        in_rows = np.nonzero(lq == L)[0]
        # anchor purity: the anchor's own outputs that read somebody else / nothing
        a_rows = np.nonzero(pre == A)[0]
        a_read = lp[a_rows]
        a_read = a_read[a_read != UNREAD]
        fA = float(segf[segs == A][0]) if np.any(segs == A) else 0.0
        # absent-chunk check: chunks with >= 20 of N's vertices whose labels lack N
        sus, unread_chunks = [], 0
        for k, c in zip(n["chunk_keys"], n["chunk_vertex_counts"]):
            if k not in ck_index:
                unread_chunks += 1
                continue
            if c >= 20:
                lab = chunk_info["labels"][ck_index[k]]
                labset = set(lab.split(",")) if isinstance(lab, str) else set(map(str, (lab or {}).keys()))
                if str(int(L)) not in labset:
                    sus.append(int(k))
        # HMI hand-annotated outputs (x, y, z in 8 nm voxels) vs label-derived output points
        hmi_ctrl = None
        row = hmi.loc[int(L)]
        if isinstance(row.outputs, str):
            ann = np.array([[float(t[1]), HMI_Y_FLIP - float(t[2]), float(t[3])]
                            for t in ast.literal_eval(row.outputs)]) * VOX8
            from scipy.spatial import cKDTree
            q = np.asarray(qpre[out_rows], float)
            if len(q):
                d, _ = cKDTree(q).query(ann)
            else:
                d = np.full(len(ann), np.inf)
            hmi_ctrl = {"n_annotated": int(len(ann)), "within_1um": int((d <= 1000).sum()),
                        "median_nm": float(np.median(d))}
        rows_out[str(int(L))] = out_pre
        sp = soma_pos.get(int(L))
        geo = None
        if sp is not None and len(out_rows):
            q = np.asarray(qpre[out_rows], float)
            dist = np.linalg.norm(q - sp, axis=1)
            geo = {"soma_nm": sp.tolist(), "median_out_dist_um": float(np.median(dist) / 1000),
                   "median_out_dist_on_anchor_um": float(np.median(dist[on_anchor]) / 1000) if on_anchor.any() else None,
                   "median_out_dist_off_anchor_um": float(np.median(dist[~on_anchor]) / 1000) if (~on_anchor).any() else None,
                   "share_out_anterior_of_soma": float(np.mean(q[:, 0] < sp[0])),
                   "share_out_within_20um": float(np.mean(dist <= 20_000)),
                   "on_anchor_within_20um": int((on_anchor & (dist <= 20_000)).sum()),
                   "off_anchor_within_20um": int((~on_anchor & (dist <= 20_000)).sum())}
        rec = {"lore": int(L), "anchor": int(A), "group": n["group"], "classifier": n["classifier"], "geometry": geo,
               "n_outputs": int(len(out_rows)), "n_inputs": int(len(in_rows)),
               "outputs_on_anchor": int(on_anchor.sum()),
               "outputs_on_member_fragments": int(on_member_frag.sum()),
               "outputs_on_other_soma_segments": int(on_other_soma.sum()),
               "outputs_on_nonmember_orphans": int((~on_anchor & ~on_member_frag & ~on_other_soma).sum()),
               "n_member_segments": int(is_mem.sum()), "n_member_fragments": int((~mem_soma).sum()),
               "n_member_output_fragments": int(len(np.unique(out_pre[on_member_frag]))),
               "n_member_other_soma_segments": int((mem_soma & (mem != A)).sum()),
               "n_segments_with_any_N_point": int(len(segs)),
               "anchor_f": fA, "anchor_is_member": bool(fA > 0.5),
               "anchor_outputs_read": int(len(a_read)), "anchor_outputs_reading_N": int((a_read == L).sum()),
               "suspect_chunks": sus, "unread_chunks": unread_chunks, "hmi_outputs_control": hmi_ctrl}
        # distance of outputs from the soma (nm), on / off anchor
        per.append(rec)
        fragments[str(int(L))] = {"member_fragments": mem[~mem_soma].tolist(),
                                  "member_other_soma": mem[mem_soma & (mem != A)].tolist(),
                                  "f_hist": np.histogram(segf, bins=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001])[0].tolist()}
        _log("truth", int(L), rec["n_outputs"], "outputs", rec["outputs_on_anchor"], "on anchor",
             rec["n_member_fragments"], "fragments", "anchor_f", round(fA, 3))
    tot_out = sum(r["n_outputs"] for r in per)
    summary = {"n_neurons": len(per), "pooled_outputs": tot_out,
               "pooled_on_anchor": sum(r["outputs_on_anchor"] for r in per),
               "pooled_on_member_fragments": sum(r["outputs_on_member_fragments"] for r in per),
               "pooled_on_other_soma": sum(r["outputs_on_other_soma_segments"] for r in per),
               "pooled_on_nonmember_orphans": sum(r["outputs_on_nonmember_orphans"] for r in per),
               "anchors_member": sum(r["anchor_is_member"] for r in per),
               "f_hist_bins": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]}
    summary["point_rule"] = "q_pre 80 nm behind p0, q_post 20 nm past p1; a point votes only if its line's two ends differ"
    np.savez(OUT / "truth_rows.npz", **{f"out_pre_{k}": v for k, v in rows_out.items()})
    _save_json("truth.json", {"summary": summary, "neurons": per})
    _save_json("fragments.json", fragments)
    return summary


# ---------------------------------------------------------------------------------------------
# 6. rules (no ground truth used here: only geometry, the soma table and the lore-id csv)
# ---------------------------------------------------------------------------------------------

def _first_per_group(group: np.ndarray, value: np.ndarray):
    """Index of the smallest value in each group (group ids 0..G-1, all present)."""
    o = np.lexsort((value, group))
    g = group[o]
    first = np.ones(len(g), bool)
    first[1:] = g[1:] != g[:-1]
    return o[first]


def _cave_somas(box_lo, box_hi):
    import pandas as pd
    df = pd.read_parquet(REF / "cave_somas_full.parquet", columns=["id", "pt_position"])
    p = np.vstack(df.pt_position.values).astype(np.float64) * VOX8
    agg = _agg_map()
    seg = np.array([agg.get(int(i), 0) for i in df.id.values], dtype=np.uint64)
    m = (seg != 0) & np.all((p >= box_lo) & (p <= box_hi), 1)
    return p[m], seg[m], df.id.values[m]


def _distinct_runner(nn_seg: np.ndarray, nn_d: np.ndarray, best: np.ndarray):
    """Per row: distance to the first neighbour whose segment differs from ``best`` (else the last
    neighbour's distance, a lower bound) and whether it was found."""
    diff = nn_seg != best[:, None]
    has = diff.any(1)
    j = np.argmax(diff, 1)
    d2 = np.where(has, nn_d[np.arange(len(j)), j], nn_d[:, -1])
    s2 = np.where(has, nn_seg[np.arange(len(j)), j], 0)
    return d2, s2, has


def rules_compute(workers: int = 8) -> dict:
    from scipy.spatial import cKDTree
    sel = _load_json("selection.json")
    R = load_region()
    lo, hi = region_box(sel)
    pre = np.asarray(R["pre"])
    post = np.asarray(R["post"])
    qpre = qpre80(R)
    qpost = np.asarray(R["qpost"])
    somas = soma_segments()
    anchors = np.array([n["anchor"] for n in sel["neurons"]], dtype=np.uint64)
    pre_s = np.isin(pre, somas)
    post_s = np.isin(post, somas)
    # soma-bearing geometry: soma centres + q_pre of their outputs + q_post of their inputs
    cpos, cseg, _ = _cave_somas(lo, hi)
    G = np.concatenate([cpos.astype(np.float32), qpre[pre_s], qpost[post_s]])
    Gseg = np.concatenate([cseg, pre[pre_s], post[post_s]])
    _log("soma-bearing geometry points", len(G), "somas", len(cpos))
    tree = cKDTree(G)
    _log("tree built")
    # fragments: output-bearing agg segments without a soma; geometry = outputs + inputs
    fo = ~pre_s
    frag_ids = np.unique(pre[fo])
    fi = ~post_s & np.isin(post, frag_ids)
    P = np.concatenate([qpre[fo], qpost[fi]])
    Pid = np.concatenate([pre[fo], post[fi]])
    del qpre, qpost
    inv = np.searchsorted(frag_ids, Pid)
    del Pid
    nF = len(frag_ids)
    _log("fragments", nF, "fragment points", len(P))
    # stage 1: nearest soma-bearing point for every fragment point (R2 winner for every fragment)
    d1p = np.empty(len(P), np.float32)
    s1p = np.empty(len(P), np.uint64)
    B = 4_000_000
    for b in range(0, len(P), B):
        d, i = tree.query(P[b:b + B], k=1, workers=workers)
        d1p[b:b + B] = d
        s1p[b:b + B] = Gseg[i]
        _log("stage 1", b + B)
    fb = _first_per_group(inv, d1p)
    r2_seg = s1p[fb]
    r2_d1 = d1p[fb].astype(np.float64)
    # R1: fragment centroid -> nearest soma centre (distinct segments for the runner-up)
    cnt = np.bincount(inv, minlength=nF).astype(np.float64)
    cen = np.stack([np.bincount(inv, weights=P[:, k].astype(np.float64), minlength=nF) for k in range(3)], 1) / cnt[:, None]
    ctree = cKDTree(cpos)
    cd, ci = ctree.query(cen, k=8, workers=workers)
    cs = cseg[ci]
    r1_seg = cs[:, 0]
    r1_d1 = cd[:, 0]
    r1_d2, r1_s2, _ = _distinct_runner(cs, cd, r1_seg)
    _log("R1 done")
    # scope: fragments with a point within 20 um of a sampled anchor's geometry, or R2 winner = anchor
    am = np.isin(Gseg, anchors)
    atree = cKDTree(G[am])
    near = np.zeros(nF, bool)
    for b in range(0, len(P), B):
        d, _ = atree.query(P[b:b + B], k=1, distance_upper_bound=20_000.0, workers=workers)
        near[np.unique(inv[b:b + B][np.isfinite(d)])] = True
    won = np.isin(r2_seg, anchors)
    scope = near | won
    sidx = np.nonzero(scope)[0]
    _log("scope fragments", len(sidx), "of which R2 won by an anchor", int(won.sum()),
         "won but not within 20 um", int((won & ~near).sum()))
    # R2 runner-up inside the scope: per point, the 16 nearest; fragment d2 = min over its points
    pm = scope[inv]
    pidx = np.nonzero(pm)[0]
    r2_d2 = np.full(nF, np.inf)
    r2_s2 = np.zeros(nF, np.uint64)
    r2_found = np.zeros(nF, bool)
    kmax_r = 0.0
    B2 = 500_000
    best_of_point = r2_seg[inv]
    for b in range(0, len(pidx), B2):
        ii = pidx[b:b + B2]
        d, i = tree.query(P[ii], k=KNN, workers=workers)
        sg = Gseg[i]
        dd, ss, has = _distinct_runner(sg, d, best_of_point[ii])
        kmax_r = max(kmax_r, float(d[:, -1].max()))
        f = inv[ii]
        o = np.lexsort((dd, f))
        f_o = f[o]
        first = np.ones(len(o), bool)
        first[1:] = f_o[1:] != f_o[:-1]
        sel_i = o[first]
        ff = f[sel_i]
        better = dd[sel_i] < r2_d2[ff]
        r2_d2[ff[better]] = dd[sel_i][better]
        r2_s2[ff[better]] = ss[sel_i][better]
        r2_found[ff[better]] = has[sel_i][better]
        _log("R2 runner-up", b + B2, "of", len(pidx))
    # R3: principal axis per scope fragment; cone search from both ends
    sums = {}
    for k in range(3):
        sums[k] = np.bincount(inv, weights=P[:, k].astype(np.float64), minlength=nF)
    C = np.zeros((nF, 3, 3))
    for a in range(3):
        for c in range(a, 3):
            v = np.bincount(inv, weights=P[:, a].astype(np.float64) * P[:, c], minlength=nF) / np.maximum(cnt, 1) \
                - (sums[a] / np.maximum(cnt, 1)) * (sums[c] / np.maximum(cnt, 1))
            C[:, a, c] = v
            C[:, c, a] = v
    elig = scope & (cnt >= R3_MIN_POINTS)
    eidx = np.nonzero(elig)[0]
    w, V = np.linalg.eigh(C[eidx])
    w = np.clip(w, 0, None)
    expl = w[:, 2] / np.maximum(w.sum(1), 1e-9)
    axis = V[:, :, 2]
    ax_full = np.zeros((nF, 3))
    ax_full[eidx] = axis
    del C
    # projections -> extreme points
    pe = np.nonzero(elig[inv])[0]
    fe = inv[pe]
    t = np.einsum("ij,ij->i", P[pe].astype(np.float64) - (cen[fe]), ax_full[fe])
    imin = pe[_first_per_group(np.searchsorted(eidx, fe), t)]
    imax = pe[_first_per_group(np.searchsorted(eidx, fe), -t)]
    span = np.zeros(nF)
    tmin = np.full(nF, np.inf)
    tmax = np.full(nF, -np.inf)
    np.minimum.at(tmin, fe, t)
    np.maximum.at(tmax, fe, t)
    span[eidx] = (tmax - tmin)[eidx]
    use3 = np.zeros(nF, bool)
    use3[eidx] = (expl >= R3_MIN_EXPLAINED) & (span[eidx] >= R3_MIN_SPAN_NM)
    uidx = np.nonzero(use3)[0]
    pos_u = np.searchsorted(eidx, uidx)
    ends = np.concatenate([P[imin[pos_u]].astype(np.float64), P[imax[pos_u]].astype(np.float64)])
    outward = np.concatenate([-axis[pos_u], axis[pos_u]])
    owner = np.concatenate([uidx, uidx])
    cos_min = np.cos(np.radians(R3_CONE_DEG))
    r3_seg = r2_seg.copy()
    r3_d1 = r2_d1.copy()
    r3_d2 = r2_d2.copy()
    r3_axis_used = np.zeros(nF, bool)
    r3_cone_found = np.zeros(nF, bool)
    kmax_3 = 0.0
    best_end = {}
    cand = []  # (fragment, seg, dist)
    for b in range(0, len(ends), B2):
        E = ends[b:b + B2]
        d, i = tree.query(E, k=KNN_R3, workers=workers)
        kmax_3 = max(kmax_3, float(d[:, -1].max()))
        vec = G[i].astype(np.float64) - E[:, None, :]
        nrm = np.linalg.norm(vec, axis=2)
        cosang = np.einsum("ijk,ik->ij", vec, outward[b:b + B2]) / np.maximum(nrm, 1e-9)
        ok = cosang >= cos_min
        r, c = np.nonzero(ok)
        cand.append(np.stack([owner[b:b + B2][r].astype(np.float64), Gseg[i[r, c]].astype(np.float64),
                              d[r, c]], 1) if len(r) else np.zeros((0, 3)))
        _log("R3 cone", b + B2, "of", len(ends))
    if cand:
        Cd = np.concatenate(cand)
        f = Cd[:, 0].astype(np.int64)
        sg = Cd[:, 1].astype(np.uint64)  # float64 holds agg ids exactly below 2**53
        dd = Cd[:, 2]
        # min distance per (fragment, segment), then best and runner-up per fragment
        o = np.lexsort((dd, sg, f))
        f, sg, dd = f[o], sg[o], dd[o]
        firstfs = np.ones(len(f), bool)
        firstfs[1:] = (f[1:] != f[:-1]) | (sg[1:] != sg[:-1])
        f, sg, dd = f[firstfs], sg[firstfs], dd[firstfs]
        o = np.lexsort((dd, f))
        f, sg, dd = f[o], sg[o], dd[o]
        firstf = np.ones(len(f), bool)
        firstf[1:] = f[1:] != f[:-1]
        fi_ = np.nonzero(firstf)[0]
        r3_seg[f[fi_]] = sg[fi_]
        r3_d1[f[fi_]] = dd[fi_]
        r3_cone_found[f[fi_]] = True
        r3_axis_used[f[fi_]] = True
        second = fi_ + 1
        has2 = (second < len(f))
        has2[has2] = f[second[has2]] == f[fi_[has2]]
        r3_d2[f[fi_]] = np.where(has2, dd[np.minimum(second, len(f) - 1)], np.inf)
    _log("R3 done: axis fragments", len(uidx), "with a cone candidate", int(r3_cone_found.sum()))
    np.savez(OUT / "rules.npz", frag_ids=frag_ids, n_points=cnt, scope=scope, near20=near,
             r1_seg=r1_seg, r1_d1=r1_d1, r1_d2=r1_d2, r1_s2=r1_s2,
             r2_seg=r2_seg, r2_d1=r2_d1, r2_d2=r2_d2, r2_s2=r2_s2, r2_found=r2_found,
             r3_seg=r3_seg, r3_d1=r3_d1, r3_d2=r3_d2, r3_axis_used=r3_axis_used, use3=use3)
    stats = {"soma_geometry_points": int(len(G)), "somas_in_region": int(len(cpos)), "fragments": int(nF),
             "fragment_points": int(len(P)), "scope_fragments": int(len(sidx)),
             "r2_won_by_anchor": int(won.sum()), "r2_won_by_anchor_beyond_20um": int((won & ~near).sum()),
             "r2_runner_up_not_found_in_16nn": int((scope & ~r2_found).sum()),
             "max_16nn_radius_nm": kmax_r, "max_64nn_radius_nm_r3": kmax_3,
             "r3_axis_fragments": int(len(uidx)), "r3_cone_found": int(r3_cone_found.sum()),
             "r3_eligible_ge3_points_in_scope": int(len(eidx))}
    _save_json("rules_compute.json", stats)
    return stats


# ---------------------------------------------------------------------------------------------
# 7. evaluate against the proofread truth: sweep, split, nulls, verdict
# ---------------------------------------------------------------------------------------------

def _score(neurons, frag_ids, pred_seg, assign, members, out_pre, nout_of, perm=None):
    """Pooled and per-neuron scores for one rule at one threshold.

    assign[f] = the rule assigns fragment f (index into frag_ids) to pred_seg[f].
    members[L] = N's member fragments (truth); out_pre[L] = pre ids of N's clean output synapses.
    perm: optional {L: L'} to score N's anchor against another neuron's truth (label-permutation null)."""
    rec_num = rec_den = base_num = pr_num = 0
    p_num = p_den = 0.0
    fc_num = fc_den = fr_num = fr_den = 0
    per = []
    a_seg = pred_seg[assign]
    a_fid = frag_ids[assign]
    o = np.argsort(a_seg)
    a_seg, a_fid = a_seg[o], a_fid[o]
    for n in neurons:
        L = str(n["lore"])
        A = np.uint64(n["anchor"])
        T = str(perm[n["lore"]]) if perm else L
        lo_, hi_ = np.searchsorted(a_seg, A, "left"), np.searchsorted(a_seg, A, "right")
        got = a_fid[lo_:hi_]
        mem = members[T]
        ok = np.isin(got, mem)
        w = nout_of(got)
        p_num += float(w[ok].sum())
        p_den += float(w.sum())
        fc_num += int(ok.sum())
        fc_den += int(len(got))
        op = out_pre[T]
        on_a = op == A
        on_got = np.isin(op, got)
        pr_num += int(on_got.sum())
        rec = int(on_a.sum() + on_got.sum())
        rec_num += rec
        rec_den += len(op)
        base_num += int(on_a.sum())
        mo = mem[np.isin(mem, op)]  # member fragments that carry at least one clean output
        fr_num += int(np.isin(mo, got).sum())
        fr_den += int(len(mo))
        per.append({"lore": n["lore"], "recovered": rec / max(len(op), 1), "baseline": float(on_a.mean()) if len(op) else 0.0,
                    "n_out": int(len(op)), "assigned_fragments": int(len(got)), "correct_fragments": int(ok.sum())})
    return {"recovered": rec_num / max(rec_den, 1), "recovered_repair_only": (rec_num - base_num) / max(rec_den, 1),
            "baseline_soma_segment": base_num / max(rec_den, 1), "precision_syn": p_num / p_den if p_den else None,
            "precision_syn_reads_preregistered": pr_num / p_den if p_den else None,
            "precision_frag": fc_num / fc_den if fc_den else None, "recall_frag": fr_num / max(fr_den, 1),
            "n_out": rec_den, "assigned_fragments": fc_den, "assigned_output_weight": p_den,
            "median_recovered_per_neuron": float(np.median([x["recovered"] for x in per])) if per else None,
            "per_neuron": per}


def evaluate() -> dict:
    import pyarrow.parquet as pq
    sel = _load_json("selection.json")
    neurons = sel["neurons"]
    Z = np.load(OUT / "rules.npz")
    frag_ids = Z["frag_ids"]
    fr = _load_json("fragments.json")
    members = {k: np.array(v["member_fragments"], dtype=np.uint64) for k, v in fr.items()}
    TR = np.load(OUT / "truth_rows.npz")
    out_pre = {k[len("out_pre_"):]: TR[k] for k in TR.files}
    t = pq.read_table(DATA / "cells.parquet", columns=["seg_id", "n_out"])
    sid = t.column("seg_id").to_numpy()
    nout = t.column("n_out").to_numpy()
    o = np.argsort(sid)
    sid, nout = sid[o], nout[o]
    del t

    def nout_of(segs):
        i = np.clip(np.searchsorted(sid, segs), 0, len(sid) - 1)
        return np.where(sid[i] == segs, nout[i], 0).astype(np.float64)

    order = np.random.default_rng(SEED + 1).permutation(len(neurons))
    halves = {"calibration": [neurons[i] for i in order[:len(neurons) // 2]],
              "test": [neurons[i] for i in order[len(neurons) // 2:]]}
    rules = {}
    for r in ("r1", "r2", "r3"):
        seg, d1, d2 = Z[f"{r}_seg"], Z[f"{r}_d1"], Z[f"{r}_d2"]
        conf = d2 / np.maximum(d1, D1_FLOOR_NM)
        rules[r] = (seg, d1, conf)
    curves = {}
    for r, (seg, d1, conf) in rules.items():
        for h, ns in halves.items():
            for thr in T_GRID:
                assign = (d1 <= CAP_NM) & (conf >= thr)
                sc = _score(ns, frag_ids, seg, assign, members, out_pre, nout_of)
                sc.pop("per_neuron")
                curves.setdefault(r, {}).setdefault(h, {})[str(thr)] = sc
            _log("curve", r, h)
    chosen = {}
    for r in rules:
        cal = curves[r]["calibration"]
        ok = [thr for thr in T_GRID if cal[str(thr)]["precision_syn"] is not None and cal[str(thr)]["precision_syn"] >= PRECISION_BAR]
        chosen[r] = min(ok) if ok else None
    reach = {r: curves[r]["calibration"][str(chosen[r])]["recovered"] for r in rules if chosen[r] is not None}
    best = max(reach, key=reach.get) if reach else None
    test = {}
    for r, (seg, d1, conf) in rules.items():
        thr = chosen[r] if chosen[r] is not None else max(T_GRID)
        assign = (d1 <= CAP_NM) & (conf >= thr)
        test[r] = {"t": thr, "t_reached_bar_on_calibration": chosen[r] is not None,
                   **_score(halves["test"], frag_ids, seg, assign, members, out_pre, nout_of)}
    # nulls (whole sample, no threshold): R2's runner-up, and R2 scored against the wrong neurons
    seg2, d1_2, conf2 = rules["r2"]
    scope = Z["scope"]
    run_seg = np.where(scope, Z["r2_s2"], np.uint64(0))
    null_runner = _score(neurons, frag_ids, run_seg, scope & (Z["r2_d2"] <= CAP_NM), members, out_pre, nout_of)
    null_runner.pop("per_neuron")
    lores = [n["lore"] for n in neurons]
    rot = {lores[i]: lores[(i + 1) % len(lores)] for i in range(len(lores))}
    null_perm = _score(neurons, frag_ids, seg2, (d1_2 <= CAP_NM) & (conf2 >= 1.0), members, out_pre, nout_of, perm=rot)
    null_perm.pop("per_neuron")
    r2_all = _score(neurons, frag_ids, seg2, (d1_2 <= CAP_NM) & (conf2 >= 1.0), members, out_pre, nout_of)
    r2_all.pop("per_neuron")
    # oracle ceiling: every member fragment reattached, nothing else
    ceil_num = sum(int(np.isin(out_pre[str(n["lore"])], np.append(members[str(n["lore"])], np.uint64(n["anchor"]))).sum())
                   for n in neurons)
    ceil_den = sum(len(out_pre[str(n["lore"])]) for n in neurons)
    # verdict on the test half
    verdict, why = "NOT_VIABLE", "no rule reached 80% precision on calibration"
    if best is not None:
        tb = test[best]
        prec = tb["precision_syn"] if tb["precision_syn"] is not None else 0.0
        if prec >= PRECISION_BAR and tb["recovered"] >= 0.5:
            verdict, why = "VIABLE", "test recovery >= 50% at precision >= 80%"
        elif prec >= PRECISION_BAR and tb["recovered"] >= 0.25 and tb["recovered"] >= 2 * tb["baseline_soma_segment"]:
            verdict, why = "PARTIAL", "test recovery >= 25% and >= 2x the soma segment, at precision >= 80%"
        else:
            why = (f"best rule {best} on test: recovered {tb['recovered']:.3f} at precision {prec:.3f} "
                   f"(baseline {tb['baseline_soma_segment']:.3f})")
    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "halves": {h: [n["lore"] for n in ns] for h, ns in halves.items()},
           "t_grid": T_GRID, "cap_nm": CAP_NM, "precision_bar": PRECISION_BAR, "curves": curves, "chosen_t": chosen,
           "best_rule": best, "test": test, "null_runner_up_r2": null_runner, "null_label_permutation_r2": null_perm,
           "r2_t1_all_neurons": r2_all, "oracle_ceiling_recovered": ceil_num / max(ceil_den, 1),
           "verdict": verdict, "verdict_reason": why}
    _save_json("eval.json", out)
    return {k: v for k, v in out.items() if k not in ("curves",)}


def report() -> str:
    """Markdown tables for the evidence note, from truth.json and eval.json."""
    T = _load_json("truth.json")
    E = _load_json("eval.json")
    L = []
    per = T["neurons"]
    f = lambda x: "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    outs = np.array([r["n_outputs"] for r in per])
    base = np.array([r["outputs_on_anchor"] / r["n_outputs"] if r["n_outputs"] else np.nan for r in per])
    frags = np.array([r["n_member_fragments"] for r in per])
    ofr = np.array([r["n_member_output_fragments"] for r in per])
    L.append(f"neurons {len(per)}; clean output synapses per neuron: median {np.median(outs):.0f}, "
             f"range {outs.min()}-{outs.max()}, pooled {outs.sum()}; neurons with 0 outputs: {(outs == 0).sum()}")
    L.append(f"member fragments per neuron (synapse-bearing): median {np.median(frags):.0f}, IQR "
             f"{np.percentile(frags, 25):.0f}-{np.percentile(frags, 75):.0f}, max {frags.max()}; with >=1 clean output: "
             f"median {np.median(ofr):.0f}, max {ofr.max()}")
    L.append(f"share of outputs on the soma segment per neuron (n with outputs = {np.isfinite(base).sum()}): median "
             f"{np.nanmedian(base):.3f}, IQR {np.nanpercentile(base, 25):.3f}-{np.nanpercentile(base, 75):.3f}")
    TR = np.load(OUT / "truth_rows.npz")
    nseg = np.array([len(np.unique(TR[k])) for k in TR.files if len(TR[k])])
    L.append(f"distinct agg segments carrying a neuron's clean outputs (neurons with outputs, n = {len(nseg)}): "
             f"median {np.median(nseg):.0f}, IQR {np.percentile(nseg, 25):.0f}-{np.percentile(nseg, 75):.0f}, max {nseg.max()}")
    fh = np.sum([json.load(open(OUT / "fragments.json"))[str(r["lore"])]["f_hist"] for r in per], 0)
    L.append(f"f_N over every segment with a clean N point, pooled histogram [0,.1,.25,.5,.75,.9,1]: {fh.tolist()}")
    pur = [(r["anchor_outputs_reading_N"], r["anchor_outputs_read"]) for r in per]
    L.append(f"anchor purity: anchor outputs in region reading N cleanly, pooled {sum(a for a, b in pur)}/{sum(b for a, b in pur)}; "
             f"anchor f_N > 0.5 for {sum(r['anchor_is_member'] for r in per)}/{len(per)}; anchor f_N median "
             f"{np.median([r['anchor_f'] for r in per]):.3f}")
    hm = [r["hmi_outputs_control"] for r in per if r["hmi_outputs_control"]]
    if hm:
        L.append(f"HMI control: {len(hm)} neurons, annotated outputs within 1 um of a label-derived output "
                 f"{sum(h['within_1um'] for h in hm)}/{sum(h['n_annotated'] for h in hm)}; per-neuron median distance "
                 f"{', '.join(f'{h[chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)+chr(95)+chr(110)+chr(109)]/1000:.2f}' for h in hm)} um")
    L.append(f"suspect chunks (>= 20 LOD-0 vertices of N, no N voxel): {sum(len(r['suspect_chunks']) for r in per)}; "
             f"unread chunks: {sum(r['unread_chunks'] for r in per)}")
    S = T["summary"]
    L.append("pooled outputs: on anchor {pooled_on_anchor}, on member fragments {pooled_on_member_fragments}, "
             "on other soma segments {pooled_on_other_soma}, on non-member orphans {pooled_on_nonmember_orphans} "
             "of {pooled_outputs}; anchors that are members {anchors_member}/{n_neurons}".format(**S))
    L.append("")
    L.append("| rule | half | t | recovered | repair only | soma seg | precision (membership) | precision (reads, pre-reg) | "
             "frag precision | frag recall | fragments assigned |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ("r1", "r2", "r3"):
        for h in ("calibration", "test"):
            for t, sc in E["curves"][r][h].items():
                L.append(f"| {r.upper()} | {h} | {t} | {f(sc['recovered'])} | {f(sc['recovered_repair_only'])} | "
                         f"{f(sc['baseline_soma_segment'])} | {f(sc['precision_syn'])} | "
                         f"{f(sc.get('precision_syn_reads_preregistered'))} | {f(sc['precision_frag'])} | "
                         f"{f(sc['recall_frag'])} | {sc['assigned_fragments']} |")
    L.append("")
    L.append(f"chosen t (calibration, precision >= 0.80): {E['chosen_t']}; best rule: {E['best_rule']}")
    for r, sc in E["test"].items():
        L.append(f"TEST {r.upper()} at t={sc['t']} (reached bar on calibration: {sc['t_reached_bar_on_calibration']}): "
                 f"recovered {f(sc['recovered'])} (repair only {f(sc['recovered_repair_only'])}, soma segment "
                 f"{f(sc['baseline_soma_segment'])}), precision {f(sc['precision_syn'])} (reads {f(sc.get('precision_syn_reads_preregistered'))}), "
                 f"frag precision {f(sc['precision_frag'])}, frag recall {f(sc['recall_frag'])}, "
                 f"median per neuron {f(sc['median_recovered_per_neuron'])}, n_out {sc['n_out']}")
    for k in ("null_runner_up_r2", "null_label_permutation_r2", "r2_t1_all_neurons"):
        sc = E[k]
        L.append(f"{k}: recovered {f(sc['recovered'])} (repair only {f(sc['recovered_repair_only'])}), precision "
                 f"{f(sc['precision_syn'])}, frag precision {f(sc['precision_frag'])}, assigned {sc['assigned_fragments']}")
    L.append(f"oracle ceiling (every member fragment reattached): {E['oracle_ceiling_recovered']:.3f}")
    L.append(f"VERDICT: {E['verdict']} ({E['verdict_reason']})")
    txt = "\n".join(L)
    (OUT / "report.md").write_text(txt)
    return txt


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "help"
    fn = {"select": select, "region": region, "validate": validate, "label": label, "truth": truth,
          "validate80": lambda: validate(names=("qpre80",)), "label_mp": label_mp,
          "assemble": assemble_labels, "rules_compute": rules_compute, "evaluate": evaluate,
          "report": report}.get(stage)
    if fn is None:
        print(__doc__)
        sys.exit(1)
    r = fn()
    if isinstance(r, str):
        print(r)
        sys.exit(0)
    print(json.dumps(r if not isinstance(r, dict) or "neurons" not in r else
                     {k: v for k, v in r.items() if k != "neurons"}, indent=1, default=str)[:4000])
