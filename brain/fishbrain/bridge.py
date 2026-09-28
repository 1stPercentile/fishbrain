"""FISHBRAIN G1c task B: build the bridged fish from the frozen spec, tune its two gains on the
training stimuli only, and gate it on held-out stimuli.

Everything this file does follows ``data/g1c/bridge_spec.json`` (frozen 2026-09-26 05:34:57 EDT,
sha256 185b3780...5e111499; companion note evidence/G1c-bridge-spec.md). It never edits or
regenerates that file.

1. ``load_spec()`` checks the spec's sha256 and recomputes its edge-list hash from its own edges.
2. ``rederive()`` rebuilds every list the spec names (stimulated set, E1 and P1 sources, targets,
   the APN relays from a fresh read of the public mece3 atlas layer, the class-(a) fragments and
   their columns, the 1,780 edges) from the rules written in the spec, in this file's own code, and
   asserts equality with the JSON. Any mismatch raises: the build stops.
3. ``build_topology()`` adds the bridge edges (base count, class, pathway) to the measured Fish1
   pairs (Shiu weights, per-synapse signs as G1) and reduces the brain exactly as G1 did
   (``brain.reduce_graph``: cells that can fire and can reach a deciding readout). Gains only scale
   counts, so one node set serves every gain setting and every ablation (removing edges only
   shrinks reachability, so simulating an ablation on the intact node set is exact and keeps the
   eye's target list, hence every Poisson draw, paired).
4. ``network()`` gives the Network for (g_a, g_c, ablation) plus its ``provenance.EdgeProvenance``.
   ``bridged_digest()`` folds the bridge version, the spec's edge-list hash, the gains, the ablation
   and the edge-provenance digest into ``Network.digest()`` (which hashes arrays only).
5. ``tune()`` evaluates the full 6 x 6 gain grid on the training set only (seeds G1c-tune-*), logs
   every step with its time, and picks the point by the spec's lexicographic objective.
6. ``gate()`` runs the held-out stimuli, controls, lesions, determinism, provenance (the
   decision-path share), the measured-wiring shuffles (M1) and the bridge ablations (M2).

Run from brain:  PYTHONPATH=. .venv/bin/python -m fishbrain.bridge <step>
Outputs go to data/g1c/gate/ (git-ignored). Nothing here writes into data/net/.
"""
from __future__ import annotations

import concurrent.futures as cf
import copy
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from fishbrain import brain as B
from fishbrain import cells as C
from fishbrain import readouts as RO
from fishbrain import sim as S
from fishbrain import stimuli as ST

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
G1C = DATA / "g1c"
SPEC_PATH = G1C / "bridge_spec.json"
OUT = G1C / "gate"
NETDIR = OUT / "net"
RATES = OUT / "rates"
REFDIR = OUT / "ref"

SPEC_SHA256 = "185b378031a3f3d730ed90bfe9cd56aede6aba6ad0c0984032e03a2c5e111499"
EDGE_LIST_SHA256 = "f7f810d56059860dc81d7fa7f064cb248779acec83e00ce88efc6facc8100bf0"
MECE3_SHA256 = "795292769933e360a3bce8188a773b81f4ea5f8e388ce0717f2750da5ca48e49"
BRIDGE_VERSION = "g1c-bridge-v1"
CODE_VERSION = "g1c-bridge-build-v1"
RULE = B.GATE_SIGN_RULE            # per_synapse, as G1
CLASSES = ("a", "b", "c")          # provenance.BRIDGE_CLASSES order (b is empty in the primary build)
UM = np.array([0.008, 0.008, 0.030])   # um per 8 x 8 x 30 nm voxel
ATLAS_MIP_NM = np.array([4096, 4096, 3840])
VOXEL_NM = np.array(C.VOXEL_NM)
FRAME_MS = B.RETINA.frame_ms

M_KEYS = [f"{c}_{s}" for c in ("Mauthner", "MiD2cm", "MiD3cm") for s in ("left", "right")]
DECIDING_KEYS = M_KEYS + ["nIII_dorsal_left", "nIII_dorsal_right", "nMLF_left", "nMLF_right",
                          "turning_left", "turning_right"]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_json(obj) -> str:
    return _sha_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float).encode())


def _jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))


# =============================================================================================
# 1. the frozen spec
# =============================================================================================

def edge_list_sha(edges: Sequence[Mapping]) -> str:
    """The spec's recipe: json.dumps([[pre, post, base, sign, cls, pathway] for edges sorted by
    (cls, pathway, pre, post)], separators=(',', ':')) as UTF-8."""
    rows = sorted(([int(e["pre"]), int(e["post"]), int(e["base"]), int(e["sign"]), str(e["cls"]), str(e["pathway"])]
                   for e in edges), key=lambda r: (r[4], r[5], r[0], r[1]))
    return _sha_bytes(json.dumps(rows, separators=(",", ":")).encode("utf-8"))


def load_spec(path: Path = SPEC_PATH) -> dict:
    raw = open(path, "rb").read()
    got = _sha_bytes(raw)
    if got != SPEC_SHA256:
        raise AssertionError(f"bridge_spec.json sha256 is {got}, the frozen spec is {SPEC_SHA256}: stop")
    spec = json.loads(raw)
    if spec["bridge_version"] != BRIDGE_VERSION:
        raise AssertionError("bridge_version differs from the frozen spec")
    if spec["edge_list_sha256"] != EDGE_LIST_SHA256 or edge_list_sha(spec["edges"]) != EDGE_LIST_SHA256:
        raise AssertionError("the spec's edge list does not hash to the frozen edge_list_sha256: stop")
    if len(spec["edges"]) != spec["n_edges"]:
        raise AssertionError("n_edges disagrees with the edge list")
    return spec


# =============================================================================================
# 2. independent re-derivation of every list (spec s.2-s.5, Appendix B), then equality with the JSON
# =============================================================================================

def g1c_stimulated_ids(P: Mapping[str, np.ndarray], ident: Mapping) -> np.ndarray:
    """Spec s.2: B1's tectal segments present in the brain of record minus B1's never-stimulate
    exclusions (G1's removal of direct readout partners is dropped). Sorted uint64."""
    t = np.asarray(ident["tectum"], np.uint64)
    present = t[B.index_of(P["ids"], t) >= 0]
    excl = set(int(x) for x in ident["exclusions"])
    return np.array(sorted({int(x) for x in present} - excl), np.uint64)


def _nearest_with_tie(pts: np.ndarray, cand_xy: np.ndarray, cand_ids: np.ndarray) -> List[int]:
    """For each point, the candidate at the smallest Euclidean distance; exact ties go to the lower id."""
    out = []
    for p in pts:
        d2 = ((cand_xy - p) ** 2).sum(1)
        m = d2.min()
        out.append(int(cand_ids[d2 == m].min()))
    return out


def _k_nearest_with_tie(p: np.ndarray, cand_xyz: np.ndarray, cand_ids: np.ndarray, k: int) -> Tuple[List[int], float, float]:
    d = np.sqrt(((cand_xyz - p) ** 2).sum(1))
    order = np.lexsort((cand_ids, d))       # distance, then lower seg id
    sel = order[:k]
    return [int(x) for x in cand_ids[sel]], float(d[sel].max()), float(d[sel].min())


def fetch_mece3(cache: bool = True) -> np.ndarray:
    """The public atlas layer mece3_231218 at mip [4096, 4096, 3840] (no login), hash-checked
    against the spec (sha256 of the C-order uint64 array bytes). Cached under data/g1c/gate/ref/."""
    path = REFDIR / "mece3_231218_4096.npy"
    if cache and path.exists():
        a = np.load(path)
    else:
        a = np.asarray(C._cv("mece3_231218", [4096, 4096, 3840])[:, :, :])[..., 0]
    got = _sha_bytes(np.ascontiguousarray(a).tobytes())
    if got != MECE3_SHA256:
        raise AssertionError(f"mece3 sha256 {got} != spec {MECE3_SHA256}: stop")
    if cache and not path.exists():
        REFDIR.mkdir(parents=True, exist_ok=True)
        np.save(path, a)
    return a


def rederive(spec: Optional[dict] = None, P: Optional[dict] = None, mece3: Optional[np.ndarray] = None) -> dict:
    """Rebuild every list from the spec's rules and compare with the spec. Raises on any mismatch."""
    spec = load_spec() if spec is None else spec
    P = B.load_pairs() if P is None else P
    cells = C.load()
    ident = B.identified(cells)
    ids = P["ids"]
    rep = {"started": now(), "checks": {}}

    def check(name, ours, theirs):
        ok = ours == theirs
        rep["checks"][name] = bool(ok)
        if not ok:
            raise AssertionError(f"re-derivation mismatch in {name}: ours {str(ours)[:300]} vs spec {str(theirs)[:300]}")

    # --- stimulated set ---
    stim = g1c_stimulated_ids(P, ident)
    tm = C.tectum_map(cells)
    side_of_tect = dict(zip(tm.seg_ids.tolist(), tm.side.tolist()))
    st = spec["stimulated_set"]
    check("stimulated_n", len(stim), st["n"])
    check("stimulated_sha256", _sha_bytes(stim.astype(np.uint64).tobytes()), st["sha256_sorted_ids"])
    g1_stim, _ = B.stimulated_set(P, ident)
    readmitted = sorted(set(stim.tolist()) - set(ids[g1_stim].tolist()))
    check("readmitted", readmitted, sorted(int(x) for x in st["readmitted"]))
    n_side = {s: int(sum(side_of_tect[int(x)] == s for x in stim)) for s in ("left", "right")}
    check("stimulated_left", n_side["left"], st["n_left"])
    check("stimulated_right", n_side["right"], st["n_right"])

    row_of = {int(s): i for i, s in enumerate(tm.seg_ids.tolist())}
    rows = np.array([row_of[int(x)] for x in stim])
    lobe = tm.side[rows]

    # --- E1 sources: nearest stimulated cell (u, v) to ((i + .5)/6, (j + .5)/6), per lobe ---
    grid_uv = [((i + 0.5) / 6.0, (j + 0.5) / 6.0) for i in range(6) for j in range(6)]
    check("E1_grid_uv", [list(g) for g in grid_uv], [list(g) for g in spec["sources"]["E1_grid_uv"]])
    E1 = {}
    for s in ("left", "right"):
        m = lobe == s
        uv = np.stack([tm.u[rows[m]], tm.v[rows[m]]], 1)
        E1[s] = sorted(set(_nearest_with_tie(np.array(grid_uv), uv, stim[m])))
        check(f"E1_sources_{s}", E1[s], sorted(int(x) for x in spec["sources"]["E1"][s]))

    # --- P1 sources: 6-deg grid in (theta, elev), theta centres -17..55, elev 67..-65 ---
    th = [-17.0 + 6.0 * i for i in range(13)]
    el = [67.0 - 6.0 * j for j in range(23)]
    check("P1_grid_theta", th, spec["sources"]["P1_grid"]["theta_deg"])
    check("P1_grid_elev", el, spec["sources"]["P1_grid"]["elev_deg"])
    pts = np.array([(a, b) for a in th for b in el])
    P1 = {}
    for s in ("left", "right"):
        m = lobe == s
        te = np.stack([tm.theta[rows[m]], tm.elev[rows[m]]], 1)
        P1[s] = sorted(set(_nearest_with_tie(pts, te, stim[m])))
        check(f"P1_sources_{s}", P1[s], sorted(int(x) for x in spec["sources"]["P1"][s]))

    # --- targets (readouts.json, strict; M-system task_criteria) ---
    ro = RO.load()
    sets = RO.readout_sets(ro, "strict", "task_criteria")
    in_graph = lambda seg: B.index_of(ids, [seg])[0] >= 0
    tg = {}
    tg["m_system"] = {s: [int(sets[f"{c}_{s}"][0]) for c in ("Mauthner", "MiD2cm", "MiD3cm")] for s in ("left", "right")}
    pops = ro["populations"]["strict"]
    tg["nIII_dorsal_lateral"], not_in = {}, {}
    for s in ("left", "right"):
        dors = [m for m in pops["strike"]["nIII"][s] if m["nIII_dorsal"]]
        med = float(np.median([m["lat_um"] for m in dors]))
        lat = [int(m["seg_id"]) for m in dors if m["lat_um"] >= med]
        tg["nIII_dorsal_lateral"][s] = [x for x in lat if in_graph(x)]
        not_in[f"nIII_dorsal_lateral_{s}"] = [x for x in lat if not in_graph(x)]
    tg["nMLF"] = {s: [int(x) for x in sets[f"nMLF_{s}"] if in_graph(int(x))] for s in ("left", "right")}
    tg["turning"] = {s: [int(x) for x in sets[f"turning_{s}"] if in_graph(int(x))] for s in ("left", "right")}
    for k in tg:
        for s in ("left", "right"):
            check(f"targets_{k}_{s}", sorted(tg[k][s]), sorted(int(x) for x in spec["targets"][k][s]))
    for k, v in not_in.items():
        check(f"not_in_graph_{k}", sorted(v), sorted(int(x) for x in spec["readout_members_not_in_graph"][k]))
    for s in ("left", "right"):
        dors_all = [int(m["seg_id"]) for m in pops["strike"]["nIII"][s] if m["nIII_dorsal"]]
        check(f"nIII_dorsal_not_in_graph_{s}", sorted(x for x in dors_all if not in_graph(x)),
              sorted(int(x) for x in spec["nIII_dorsal_members_not_in_graph"][s]))

    # --- APN relay: CAVE v709 soma nearest to the same-side AF7 voxels (mece3 label 3) ---
    m3 = fetch_mece3() if mece3 is None else mece3
    rep["mece3_sha256"] = _sha_bytes(np.ascontiguousarray(m3).tobytes())
    check("mece3_sha256", rep["mece3_sha256"], spec["mece3_sha256"])
    m2 = np.load(C.REF / "mece2_231218_4096.npy", mmap_mode="r")
    somas = C._somas()
    agg = C._agg_map()
    axes = cells["axes"]
    f = ATLAS_MIP_NM / VOXEL_NM                       # 8-nm voxels per atlas voxel
    af7 = np.argwhere(m3 == 3)
    af7_c8 = (af7 + 0.5) * f
    af7_side = C.side_of(af7_c8, axes)
    pos8 = np.vstack(somas.pt_position.values).astype(float)
    q = np.clip(np.floor(pos8 / f).astype(int), 0, np.array(m2.shape) - 1)
    lab2 = np.asarray(m2[q[:, 0], q[:, 1], q[:, 2]])
    cand = np.isin(lab2, [14, 4])
    soma_side = C.side_of(pos8, axes)
    seg_of = agg.reindex(somas.id.values)
    lore_per_seg = agg.value_counts()
    apn = {}
    for s in ("left", "right"):
        A = af7_c8[af7_side == s] * UM
        sel = np.flatnonzero(cand & (soma_side == s))
        rows_ = []
        for i in sel:
            d = float(np.sqrt((((A - pos8[i] * UM)) ** 2).sum(1)).min())
            rows_.append((d, int(somas.id.values[i]), i))
        rows_.sort()
        pick = None
        for d, lore, i in rows_:
            seg = seg_of.iloc[i]
            if seg != seg or seg is None:
                continue
            seg = int(seg)
            if not in_graph(seg) or int(lore_per_seg.get(seg, 0)) != 1:
                continue
            pick = {"lore_id": lore, "seg_id": seg, "dist_to_af7_um": round(d, 2)}
            break
        apn[s] = pick
        want = spec["classes"]["c"]["apn_relay"][s]
        check(f"apn_relay_{s}", (pick["seg_id"], pick["lore_id"], pick["dist_to_af7_um"]),
              (int(want["seg_id"]), int(want["lore_id"]), float(want["dist_to_af7_um"])))
        check(f"af7_voxels_{s}", int((af7_side == s).sum()), int(want["af7_voxels_side"]))
        check(f"pretectal_somas_{s}", int((cand & (soma_side == s)).sum()), int(want["n_pretectal_somas_side"]))

    # --- class (a): eligible O-a1 fragments, not B1 tectal, >= 1 input synapse in the tectum ---
    frags = json.load(open(G1C / "oa_fragments.json"))["fragments"]
    tect_set = set(int(x) for x in ident["tectum"])
    elig = []
    for fr in frags:
        if not any(r["cls"] == "O-a1" for r in fr["readouts"]):
            continue
        if int(fr["seg_id"]) in tect_set or fr["b1_tectal"]:
            continue
        if fr["roles"]["in"]["in_tectum"] < 1:
            continue
        elig.append(fr)
    check("class_a_eligible", sorted(int(fr["seg_id"]) for fr in elig),
          sorted(int(e["seg_id"]) for e in spec["classes"]["a"]["eligible"]))
    stim_pos_um = np.array([cells["tectum"][row_of[int(x)]]["pos"] for x in stim], float) * UM
    cols = {}
    spec_a = {int(e["seg_id"]): e for e in spec["classes"]["a"]["eligible"]}
    for fr in elig:
        s = fr["tectal_end_side"]
        m = lobe == s
        col, rad, near = _k_nearest_with_tie(np.asarray(fr["tectal_end_8nm"], float) * UM, stim_pos_um[m], stim[m],
                                             spec["unit"]["U"])
        cols[int(fr["seg_id"])] = col
        e = spec_a[int(fr["seg_id"])]
        check(f"class_a_column_{fr['seg_id']}", sorted(col), sorted(int(x) for x in e["column"]))
        check(f"class_a_radius_{fr['seg_id']}", round(rad, 2), float(e["column_radius_um"]))

    # --- the edges ---
    edges = []
    for seg, col in cols.items():
        edges += [dict(pre=c, post=seg, base=1, sign=1, cls="a", pathway="A1_column_to_observed_origin_fragment")
                  for c in col]
    other = {"left": "right", "right": "left"}
    for s in ("left", "right"):
        edges += [dict(pre=c, post=t, base=1, sign=1, cls="c", pathway="E1_tectum_to_contra_M_system")
                  for c in E1[s] for t in tg["m_system"][other[s]]]
        edges += [dict(pre=c, post=apn[s]["seg_id"], base=1, sign=1, cls="c", pathway="P1_frontal_tectum_to_APN")
                  for c in P1[s]]
        r = apn[s]["seg_id"]
        edges += [dict(pre=r, post=t, base=17, sign=1, cls="c", pathway="P2_APN_to_nIII_dorsal_lateral_bilateral")
                  for ss in ("left", "right") for t in tg["nIII_dorsal_lateral"][ss]]
        edges += [dict(pre=r, post=t, base=17, sign=1, cls="c", pathway="P3_APN_to_nMLF_bilateral")
                  for ss in ("left", "right") for t in tg["nMLF"][ss]]
        edges += [dict(pre=r, post=t, base=34, sign=1, cls="c", pathway="P4_APN_to_contra_turning_SPNs")
                  for t in tg["turning"][other[s]]]
    check("n_edges", len(edges), spec["n_edges"])
    check("edge_list_sha256", edge_list_sha(edges), spec["edge_list_sha256"])
    # --- structural checks: duplicates, self loops, endpoints, collisions with measured pairs ---
    keys = [(e["pre"], e["post"]) for e in edges]
    check("no_duplicate_bridge_pairs", len(set(keys)), len(keys))
    check("no_self_loops", sum(a == b for a, b in keys), 0)
    pre_i = B.index_of(ids, [k[0] for k in keys])
    post_i = B.index_of(ids, [k[1] for k in keys])
    check("endpoints_in_graph", int(((pre_i < 0) | (post_i < 0)).sum()), 0)
    n = len(ids)
    kb = pre_i.astype(np.int64) * n + post_i.astype(np.int64)
    km = P["pre"].astype(np.int64) * n + P["post"].astype(np.int64)
    coll = np.isin(kb, km)
    rep["collisions"] = [dict(pre=keys[i][0], post=keys[i][1]) for i in np.flatnonzero(coll)]
    check("collisions_with_measured_pairs", len(rep["collisions"]), len(spec["weights"]["measured_pair_collisions_found"]))
    rep["measured_pairs_checked"] = int(len(km))
    rep["n_edges"] = len(edges)
    rep["edge_list_sha256"] = edge_list_sha(edges)
    rep["apn_relay"] = apn
    rep["finished"] = now()
    rep["all_equal"] = all(rep["checks"].values())
    return rep


# =============================================================================================
# 3. the bridged topology (gain-invariant node set) and networks per gain / ablation
# =============================================================================================

def deciding_sets() -> Dict[str, List[int]]:
    """Segment ids per deciding readout key (readouts.readout_sets strict, M-system task_criteria)."""
    s = RO.readout_sets(RO.load(), "strict", "task_criteria")
    return {k: [int(x) for x in s[k]] for k in DECIDING_KEYS}


def lesion_sets() -> Dict[str, List[int]]:
    ro = RO.load()
    sets = deciding_sets()
    out = {"bilateral": sorted(int(x) for x in ro["lesion_sets"]["task_criteria"])}
    for s in ("left", "right"):
        out[f"unilateral_{s}"] = sorted(sets[f"{c}_{s}"][0] for c in ("Mauthner", "MiD2cm", "MiD3cm"))
    union = sorted(int(x) for x in ro["lesion_sets"]["union"])
    out["union_bilateral"] = union
    # which side each union member sits on: from readouts.json m_system picks
    side = {}
    for c in ("Mauthner", "MiD2cm", "MiD3cm"):
        for s in ("left", "right"):
            v = ro["m_system"][c][s]
            for k in ("seg_id", "task_criteria", "preregistered_crossing"):
                if k in v:
                    side[int(v[k])] = s
    for s in ("left", "right"):
        out[f"union_unilateral_{s}"] = sorted(x for x in union if side.get(x) == s)
    if sorted(out["union_unilateral_left"] + out["union_unilateral_right"]) != union:
        raise AssertionError("a union lesion cell has no side")
    if sorted(out["unilateral_left"] + out["unilateral_right"]) != out["bilateral"]:
        raise AssertionError("unilateral task_criteria lesion sets do not add up to the bilateral set")
    return out


def _topo_path(tag: str) -> Path:
    return NETDIR / f"topology_{tag}.npz"


def build_topology(spec: Optional[dict] = None, tag: str = "intact", shuffle_seed: Optional[str] = None,
                   P: Optional[dict] = None, rebuild: bool = False) -> dict:
    """Reduce (measured pairs [shuffled if shuffle_seed] + bridge edges at base count) exactly as G1.

    Saved: kept ids, measured local pairs, bridge local pairs with base/class/pathway, the kept
    stimulated ids, stats. Bridge edges are re-added by segment id after any shuffle."""
    spec = load_spec() if spec is None else spec
    path = _topo_path(tag)
    if path.exists() and not rebuild:
        return load_topology(tag)
    t0 = time.time()
    P = B.load_pairs() if P is None else P
    ident = B.identified()
    ids = P["ids"]
    n = len(ids)
    signed, sinfo = B.signed_counts(P, RULE)
    nz = signed != 0
    mpre, mpost, mcnt = P["pre"][nz].astype(np.int64), P["post"][nz].astype(np.int64), signed[nz].astype(np.int64)
    extra = {"sign_info": {k: v for k, v in sinfo.items()}}
    if shuffle_seed is not None:
        full = S.Network.from_pairs(mpre, mpost, mcnt, n, ids=ids)
        del mpre, mpost, mcnt
        t1 = time.time()
        sh = full.shuffled_copy(shuffle_seed)
        del full
        mpre, mpost, mcnt = sh.rows(), sh.indices.astype(np.int64), sh.counts.astype(np.int64)
        extra["shuffle"] = dict(sh.meta.get("shuffle", {}), seed=shuffle_seed, seconds=round(time.time() - t1, 1),
                                pairs=int(sh.nnz))
        del sh
    E = spec["edges"]
    bpre = B.index_of(ids, [e["pre"] for e in E])
    bpost = B.index_of(ids, [e["post"] for e in E])
    if (bpre < 0).any() or (bpost < 0).any():
        raise AssertionError("a bridge endpoint is not in the brain of record")
    base = np.array([e["base"] * e["sign"] for e in E], np.int64)
    bcls = np.array([e["cls"] for e in E])
    bpath = np.array([e["pathway"] for e in E])
    # collisions with (possibly shuffled) measured pairs: listed, never dropped (the provenance
    # table keeps the parts apart; Network.from_pairs sums them)
    km = mpre * n + mpost
    kb = bpre * n + bpost
    coll = np.flatnonzero(np.isin(kb, km))
    extra["bridge_collisions_with_measured"] = [dict(pre=int(ids[bpre[i]]), post=int(ids[bpost[i]]),
                                                     pathway=str(bpath[i])) for i in coll]
    del km
    stim_ids = g1c_stimulated_ids(P, ident)
    stim = B.index_of(ids, stim_ids)
    sets = deciding_sets()
    ro_idx = {k: B.index_of(ids, v) for k, v in sets.items()}
    pre = np.concatenate([mpre, bpre])
    post = np.concatenate([mpost, bpost])
    cnt = np.concatenate([mcnt, base])
    nsyn = np.abs(cnt)
    if shuffle_seed is None:
        nsyn = np.concatenate([(P["t1"].astype(np.int64) + P["t2"])[nz], np.abs(base)])
    R = B.reduce_graph(n, pre, post, cnt, nsyn, stim, ro_idx)
    keep, pk = R["keep_idx"], R["pair_keep"]
    nm = len(mpre)
    loc = np.full(n, -1, np.int64)
    loc[keep] = np.arange(len(keep))
    kids = ids[keep]
    mk = pk[:nm]
    bk = pk[nm:]
    stats = dict(R["stats"])
    stats.update(extra)
    stats["bridge_edges_kept"] = int(bk.sum())
    stats["bridge_edges_dropped"] = [dict(pre=int(ids[bpre[i]]), post=int(ids[bpost[i]]), pathway=str(bpath[i]))
                                     for i in np.flatnonzero(~bk)]
    stats["stimulated_g1c"] = int(len(stim_ids))
    kstim = np.sort(ids[stim[np.isin(stim, keep)]])
    stats["stimulated_kept"] = int(len(kstim))
    stats["build_seconds"] = round(time.time() - t0, 1)
    stats["tag"] = tag
    stats["built"] = now()
    NETDIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=kids, mpre=loc[mpre[mk]].astype(np.int32), mpost=loc[mpost[mk]].astype(np.int32),
             mcnt=mcnt[mk].astype(np.int32), bpre=loc[bpre[bk]].astype(np.int32), bpost=loc[bpost[bk]].astype(np.int32),
             bbase=base[bk].astype(np.int32), bcls=bcls[bk], bpath=bpath[bk], stim_ids=kstim)
    _jdump({"key": topology_key(tag, shuffle_seed), "stats": stats}, path.with_suffix(".json"))
    _log("topology", tag, "neurons", len(kids), "measured pairs", int(mk.sum()), "bridge edges", int(bk.sum()),
         "stim kept", len(kstim), f"{stats['build_seconds']} s")
    return load_topology(tag)


def topology_key(tag: str, shuffle_seed: Optional[str]) -> str:
    return _sha_json([CODE_VERSION, BRIDGE_VERSION, EDGE_LIST_SHA256, B.WIRING_CONTENT_HASH, RULE, tag,
                      shuffle_seed, B._file_sha(C.CELLS_JSON), B._file_sha(RO.READOUTS_JSON)])[:16]


def load_topology(tag: str = "intact") -> dict:
    path = _topo_path(tag)
    d = np.load(path, allow_pickle=False)
    meta = json.load(open(path.with_suffix(".json")))
    out = {k: d[k] for k in d.files}
    out["stats"] = meta["stats"]
    out["key"] = meta["key"]
    out["tag"] = tag
    return out


GAIN_OF = {"a": "g_a", "c": "g_c"}


@dataclasses.dataclass
class Bridged:
    """A simulated network with its bridge provenance, its eye targets and its digests."""
    net: S.Network
    prov: object                  # provenance.EdgeProvenance
    stim_ids: np.ndarray
    g_a: int
    g_c: int
    ablate: Tuple[str, ...]
    digest: str
    topo_tag: str
    rule: str = RULE

    def __post_init__(self):
        self.stim_idx = self.net.index_of(self.stim_ids)


def bridged_digest(net: S.Network, g_a: int, g_c: int, ablate: Sequence[str], prov_digest: str, topo_tag: str) -> str:
    """Network.digest() (arrays only; the JS kernel reproduces it) folded with the bridge version, the
    spec's edge-list hash, the gains, the ablation, the topology tag and the edge-provenance digest."""
    return _sha_json({"network_digest": net.digest(), "bridge_version": BRIDGE_VERSION,
                      "edge_list_sha256": EDGE_LIST_SHA256, "g_a": int(g_a), "g_c": int(g_c),
                      "ablate": sorted(ablate), "topology": topo_tag, "edge_provenance": prov_digest})


def network(topo: dict, g_a: int, g_c: int, ablate: Sequence[str] = ()) -> Bridged:
    """The Network for gains (g_a, g_c) with the named bridge classes ('a', 'c') or pathway prefixes
    ('E1', 'P1', ...; 'P' = P1-P4; 'all') removed. Measured pairs are never touched."""
    from fishbrain import provenance as PV
    for g in (g_a, g_c):
        if int(g) != g or not (1 <= g <= 32):
            raise ValueError("gains are integers in [1, 32] (spec s.6)")
    cls = topo["bcls"].astype(str)
    path = topo["bpath"].astype(str)
    drop = np.zeros(len(cls), bool)
    for a in ablate:
        if a == "all":
            drop[:] = True
        elif a in ("a", "b", "c"):
            drop |= cls == a
        elif a == "P":
            drop |= np.char.startswith(path, "P")
        else:
            drop |= np.char.startswith(path, a + "_")
    gain = np.where(cls == "a", g_a, np.where(cls == "c", g_c, 0)).astype(np.int64)
    bc = topo["bbase"].astype(np.int64) * gain
    sel = ~drop
    n = len(topo["ids"])
    pre = np.concatenate([topo["mpre"].astype(np.int64), topo["bpre"][sel].astype(np.int64)])
    post = np.concatenate([topo["mpost"].astype(np.int64), topo["bpost"][sel].astype(np.int64)])
    cnt = np.concatenate([topo["mcnt"].astype(np.int64), bc[sel]])
    net = S.Network.from_pairs(pre, post, cnt, n, ids=topo["ids"],
                               meta={"rule": RULE, "bridge_version": BRIDGE_VERSION, "g_a": int(g_a), "g_c": int(g_c),
                                     "ablate": sorted(ablate), "topology": topo["tag"]})
    prov = PV.EdgeProvenance(CLASSES, topo["bpre"][sel], topo["bpost"][sel], bc[sel], cls[sel],
                             meta={"bridge_version": BRIDGE_VERSION, "g_a": int(g_a), "g_c": int(g_c)})
    dg = bridged_digest(net, g_a, g_c, ablate, prov.digest(), topo["tag"])
    return Bridged(net, prov, topo["stim_ids"], int(g_a), int(g_c), tuple(sorted(ablate)), dg, topo["tag"])


# =============================================================================================
# 4. the eye (brain.Retina unchanged; its caches redirected under data/g1c/gate/) and the stimuli
# =============================================================================================

class GateRetina(B.Retina):
    """brain.Retina with RetinaParams() unchanged. Only the W cache path moves to data/g1c/gate/net
    (brain.Retina would write into data/net/, which this task does not own). Same key recipe."""

    def _load_or_build_W(self, cache):
        key = B._sha("W", B._file_sha(C.CELLS_JSON), self.stim_ids, self.p.res_deg, dataclasses.asdict(self.eye))[:16]
        path = NETDIR / f"retina_W_{key}.npz"
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
            NETDIR.mkdir(parents=True, exist_ok=True)
            np.savez(path, **W)
        return W


def _canon(d: Mapping) -> dict:
    return {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in d.items()}


def stim_hash(kind: str, params: Mapping) -> str:
    """sha256 of a stimulus's parameter dict (spec s.7's held-out hash check)."""
    return _sha_json({"kind": kind, "params": _canon(params)})


def make_condition(kind: str, params: Optional[Mapping] = None, side: Optional[str] = None) -> dict:
    """A gate condition: name, stimulus builder inputs, window, stimulus side, parameter hash.

    kind: 'loom' | 'prey' (params from the spec's stimulus dicts), 'recede' | 'dim' | 'blank'
    (G1's own stimuli, brain.stim_recede / stim_dim / stim_blank)."""
    if kind == "loom":
        p = dict(params)
        _, t_end = ST.loom_schedule(p["l_over_v_ms"], p["start_deg"], p["end_deg"])
        if abs(t_end - p["expansion_ms"]) > 0.051:
            raise AssertionError(f"expansion {t_end} differs from the spec's {p['expansion_ms']}")
        name = f"loom_az{p['azimuth_deg']:+.0f}_lv{p['l_over_v_ms']:.0f}"
        win = (p["pre_ms"], p["pre_ms"] + t_end + B.GATE["sensorimotor_allow_ms"])
        sside = "right" if p["azimuth_deg"] > 0 else "left"
        return {"name": name, "kind": kind, "params": p, "window": win, "side": sside, "expansion_ms": t_end,
                "hash": stim_hash(kind, {k: v for k, v in p.items() if k != "expansion_ms"})}
    if kind == "prey":
        p = dict(params)
        name = f"prey_az{p['azimuth_deg']:+.0f}_v{p['speed_deg_s']:.0f}"
        win = (p["pre_ms"], p["pre_ms"] + p["duration_ms"])
        sside = "right" if p["azimuth_deg"] > 0 else "left"
        q = {k: v for k, v in p.items() if k != "direction"}
        q["direction"] = 1 if p["azimuth_deg"] > 0 else -1
        return {"name": name, "kind": kind, "params": p, "window": win, "side": sside,
                "hash": stim_hash(kind, q)}
    w = B.windows_ms()
    if kind == "recede":
        return {"name": f"recede_{side}", "kind": kind, "side": side, "window": tuple(w["recede_motion"]),
                "onset_window": tuple(w["recede_onset"]), "params": {"side": side, **B.PROTOCOL["loom"], **B.PROTOCOL["recede"]},
                "hash": stim_hash("recede", {"side": side, **B.PROTOCOL["loom"], **B.PROTOCOL["recede"]})}
    if kind == "dim":
        return {"name": "dim", "kind": kind, "side": None, "window": tuple(w["dim"]),
                "params": dict(B.PROTOCOL["dim"]), "hash": stim_hash("dim", {**B.PROTOCOL["loom"], **B.PROTOCOL["dim"]})}
    if kind == "blank":
        return {"name": "blank", "kind": kind, "side": None, "window": tuple(w["blank"]),
                "params": dict(B.PROTOCOL["blank"]), "hash": stim_hash("blank", B.PROTOCOL["blank"])}
    raise ValueError(kind)


def render(cond: Mapping):
    """The Stimulus object of a condition (5 ms frames, 2 deg lattice, as G1)."""
    k, p = cond["kind"], cond["params"]
    dt, res = FRAME_MS, B.RETINA.res_deg
    if k == "loom":
        return ST.looming(p["azimuth_deg"], p["elevation_deg"], p["l_over_v_ms"], p["start_deg"], p["end_deg"],
                          p["contrast"], pre_ms=p["pre_ms"], hold_ms=p["hold_ms"], dt_ms=dt, res_deg=res)
    if k == "prey":
        return ST.prey_dot(p["azimuth_deg"], p["elevation_deg"], p["size_deg"], p["speed_deg_s"], p["sweep_deg"],
                           p["contrast"], p["duration_ms"], direction=1 if p["azimuth_deg"] > 0 else -1,
                           pre_ms=p["pre_ms"], dt_ms=dt, res_deg=res)
    if k == "recede":
        return B.stim_recede(cond["side"])
    if k == "dim":
        return B.stim_dim()
    if k == "blank":
        return B.stim_blank()
    raise ValueError(k)


def conditions(spec: Mapping, which: str) -> List[dict]:
    """'training' (tuning only), 'held_out_loom', 'held_out_prey', 'controls', 'c1_baseline'
    (the training looms, run with the gate seeds after tuning as C1/C2's baseline)."""
    st = spec["stimuli"]
    if which == "training":
        return [make_condition("loom", p) for p in st["training"]["loom"]] + \
               [make_condition("prey", p) for p in st["training"]["prey"]]
    if which == "held_out_loom":
        return [make_condition("loom", p) for p in st["held_out"]["loom"]]
    if which == "held_out_prey":
        return [make_condition("prey", p) for p in st["held_out"]["prey"]]
    if which == "controls":
        return [make_condition("recede", side="left"), make_condition("recede", side="right"),
                make_condition("dim"), make_condition("blank")]
    if which == "c1_baseline":
        return [dict(make_condition("loom", p), role="c1_baseline") for p in st["training"]["loom"]]
    raise ValueError(which)


def rates_for(retina: GateRetina, cond: Mapping) -> Tuple[np.ndarray, dict]:
    """Retina rates for a condition, cached under data/g1c/gate/rates/ (pure function of W, the
    retina parameters and the stimulus)."""
    stim = render(cond)
    key = B._sha(CODE_VERSION, retina.p.digest(), stim.digest(),
                 hashlib.sha256(b"".join(retina.W[e].tobytes() for e in ST.EYES)).hexdigest(), retina.stim_ids)[:20]
    path = RATES / f"{cond['name']}_{key}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=False)
        diag = json.loads(str(d["diag"]))
        if hashlib.sha256(d["rates"].astype("<f8").tobytes()).hexdigest() == diag["rates_sha256"]:
            return d["rates"], diag
    rates, diag = retina.rates(stim)
    dj = {"rates_sha256": diag["rates_sha256"], "expected_tectal_input_spikes": diag["expected_tectal_input_spikes"],
          "peak_cells_driven": int(diag["cells_driven_per_frame"].max()),
          "retina_output_total": float(diag["retina_output_per_frame"].sum()),
          "stimulus_digest": stim.digest(), "stimulus": stim.describe(), "path": str(path.name)}
    RATES.mkdir(parents=True, exist_ok=True)
    np.savez(path, rates=rates, diag=np.array(json.dumps(dj, default=float)))
    return rates, dj


def rates_path(retina: GateRetina, cond: Mapping) -> Path:
    _, dj = rates_for(retina, cond)
    return RATES / dj["path"]


# =============================================================================================
# 5. one trial: run, read out, score
# =============================================================================================

def _group_idx(net: S.Network, sets: Mapping[str, Sequence[int]]) -> Dict[str, List[Tuple[int, int]]]:
    """Per key: (segment id, internal index or -1 when the member is not in the network)."""
    out = {}
    order = np.argsort(net.ids, kind="stable")
    sid = net.ids[order]
    for k, v in sets.items():
        q = np.asarray(v, np.uint64)
        pos = np.searchsorted(sid, q)
        pc = np.minimum(pos, len(sid) - 1)
        ok = (pos < len(sid)) & (sid[pc] == q)
        out[k] = [(int(s), int(order[pc[i]]) if ok[i] else -1) for i, s in enumerate(v)]
    return out


def readout_counts(raster: S.Raster, net: S.Network, sets: Mapping, window: Tuple[float, float]) -> dict:
    """Per deciding key: spike count per member in the window (0 for members not in the network) and
    each member's first spike time in the window."""
    t = raster.steps * raster.dt_ms
    sel = (t >= window[0]) & (t < window[1])
    nn, tt = raster.neurons[sel], t[sel]
    cnt = np.bincount(nn, minlength=net.n)
    first = np.full(net.n, np.inf)
    np.minimum.at(first, nn, tt)
    g = _group_idx(net, sets)
    out = {}
    for k, mem in g.items():
        c = [int(cnt[i]) if i >= 0 else 0 for _, i in mem]
        f = [float(first[i]) if (i >= 0 and np.isfinite(first[i])) else None for _, i in mem]
        out[k] = {"counts": c, "first": f}
    return out


def score_trial(rc: Mapping, cond: Mapping) -> dict:
    """The spec's readouts (s.9) on one trial's counts."""
    first = {k: (min(x for x in rc[k]["first"] if x is not None) if any(x is not None for x in rc[k]["first"]) else None)
             for k in M_KEYS}
    spikes = {k: int(sum(rc[k]["counts"])) for k in M_KEYS}
    esc = RO.escape_decision(first, spikes)
    strike = RO.strike_decision({k: rc[k]["counts"] for k in ("nMLF_left", "nMLF_right", "nIII_dorsal_left", "nIII_dorsal_right")})
    turn = RO.turn_index({k: rc[k]["counts"] for k in ("turning_left", "turning_right")})
    out = {"escape": bool(esc["escape"]), "initiator": esc["initiator"], "cell_side": esc["cell_side"],
           "escape_to": esc["escape_to"], "escape_t_ms": esc.get("first_spike_ms"),
           "m_spikes": spikes, "mauthner_spikes": spikes["Mauthner_left"] + spikes["Mauthner_right"],
           "strike": bool(strike["strike"]), "strike_side": strike["side"], "converged": strike["converged"],
           "drive": strike["drive"], "nMLF_side_index": strike["side_index"],
           "active_frac": {"nIII_dorsal_left": strike["nIII_dorsal"]["left"]["active_frac"],
                           "nIII_dorsal_right": strike["nIII_dorsal"]["right"]["active_frac"],
                           "nMLF_left": strike["nMLF"]["left"]["active_frac"], "nMLF_right": strike["nMLF"]["right"]["active_frac"]},
           "turn": turn["turn"], "turn_index": turn["turn_index"],
           "turn_mean": {"left": turn["left"]["mean_spikes"], "right": turn["right"]["mean_spikes"]}}
    if cond["kind"] == "loom" and esc["escape"]:
        p = cond["params"]
        rel = esc["first_spike_ms"] - p["pre_ms"]
        out["escape_angle_deg"] = round(float(ST.loom_angle_deg(max(rel, 0.0), p["l_over_v_ms"], p["start_deg"], p["end_deg"])), 3) if rel >= 0 else 0.0
        out["escape_after_expansion_end_ms"] = round(esc["first_spike_ms"] - (p["pre_ms"] + cond["expansion_ms"]), 1)
    if cond.get("side") in ("left", "right"):
        out["initiator_on_stimulus_side"] = bool(esc["escape"] and esc["cell_side"] == cond["side"])
        out["turn_toward"] = bool(turn["turn"] == cond["side"])
    return out


_WCACHE: Dict[str, object] = {}


def _worker_state(topo_tag: str, g_a: int, g_c: int, ablate: Tuple[str, ...]) -> Bridged:
    key = f"{topo_tag}|{g_a}|{g_c}|{','.join(ablate)}"
    if key not in _WCACHE:
        if len(_WCACHE) > 6:
            _WCACHE.clear()
        _WCACHE[key] = network(load_topology(topo_tag), g_a, g_c, ablate)
    return _WCACHE[key]


def run_job(job: Mapping) -> dict:
    """One condition x seed list on one network (runs in a worker process).

    job: topo, g_a, g_c, ablate, lesion (segment ids or []), cond, seeds, rates_path, and optional
    'trace' (True: the run is provenance.trial_traced's, so every hash doubles as a replay check),
    'l1_tmed' (intact median escape time for L1's informative row)."""
    from fishbrain import provenance as PV
    bw = _worker_state(job["topo"], job["g_a"], job["g_c"], tuple(job["ablate"]))
    d = np.load(job["rates_path"], allow_pickle=False)
    rates = d["rates"]
    rsha = hashlib.sha256(rates.astype("<f8").tobytes()).hexdigest()
    cond = job["cond"]
    win = tuple(cond["window"])
    sets = deciding_sets()
    lesion_req = [int(x) for x in job.get("lesion") or []]
    # a cell outside the reduced network cannot reach a deciding readout, so lesioning it is a no-op
    lesion = [x for x in lesion_req if x in set(bw.net.ids.tolist())]
    lesion_absent = sorted(set(lesion_req) - set(lesion))
    net = bw.net if not lesion else bw.net.lesion(np.asarray(lesion, np.uint64))
    out = []
    for sd in job["seeds"]:
        w0, c0 = time.perf_counter(), time.process_time()
        tr = None
        if job.get("trace"):
            windows = {"decision": win}
            raster, tr = PV.trial_traced(bw, rates, sd, prov=bw.prov, windows=windows, lesion_ids=lesion or None,
                                         eye_segments=job["eye_segments"], check=True)
        else:
            sim = S.Simulator(net, seed=sd)
            sim.add_poisson(bw.stim_idx, rates, frame_ms=FRAME_MS, name="retina")
            raster = sim.run(rates.shape[0] * FRAME_MS)
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        rc = readout_counts(raster, net, sets, win)
        row = {"cond": cond["name"], "seed": sd, "spike_hash": S.spike_hash(raster), "n_spikes": int(len(raster.neurons)),
               "tectal_spikes": int(np.isin(raster.neurons, bw.stim_idx).sum()), "rates_sha256": rsha,
               "wall_s": round(wall, 2), "cpu_s": round(cpu, 2), "digest": bw.digest, "lesion": job.get("lesion_name")}
        if lesion_req:
            row["lesioned_ids"] = lesion
            row["lesion_ids_not_in_network"] = lesion_absent
        row.update(score_trial(rc, cond))
        row["readout_first_ms"] = {k: min([x for x in v["first"] if x is not None], default=None) for k, v in rc.items()}
        if cond["kind"] == "recede":
            on = readout_counts(raster, net, {k: sets[k] for k in M_KEYS}, tuple(cond["onset_window"]))
            row["onset_m_spikes"] = {k: int(sum(v["counts"])) for k, v in on.items()}
        if "l1_tmed" in job and job["l1_tmed"] is None:
            row["l1"] = {"t_med_ms": None, "m_system_spikes_within_20ms": int(sum(row["m_spikes"].values()) > 0),
                         "other_deciding_spikes_within_20ms": 0, "note": "intact fish never escaped in this condition"}
        if job.get("l1_tmed") is not None:
            tm_ = job["l1_tmed"]
            lo, hi = tm_ - 20.0, tm_ + 20.0
            t = raster.steps * raster.dt_ms
            rest = [i for k in DECIDING_KEYS if k not in M_KEYS for (_, i) in _group_idx(net, {k: sets[k]})[k] if i >= 0]
            m = np.isin(raster.neurons, rest) & (t >= lo) & (t <= hi)
            mm = np.isin(raster.neurons, [i for k in M_KEYS for (_, i) in _group_idx(net, {k: sets[k]})[k] if i >= 0]) \
                & (t >= lo) & (t <= hi)
            row["l1"] = {"t_med_ms": tm_, "m_system_spikes_within_20ms": int(mm.sum()),
                         "other_deciding_spikes_within_20ms": int(m.sum())}
        if tr is not None:
            row["provenance"] = provenance_row(tr, net, sets, cond, row)
        out.append(row)
    return {"job": {k: v for k, v in job.items() if k not in ("cond",)}, "cond": cond["name"], "trials": out}


def provenance_row(tr, net: S.Network, sets: Mapping, cond: Mapping, row: Mapping) -> dict:
    """The pre-registered decision-path share of one trial (brief G1c; spec s.9 deciding cells):
    loom -> the initiating M-system cell; prey -> every nIII_dorsal/nMLF cell and every turning SPN
    that spiked in the window. Population-level shares are a labelled secondary row."""
    from fishbrain import provenance as PV
    groups, missing = PV.readout_groups(net, sets)
    t = tr.raster.steps * tr.raster.dt_ms
    lo, hi = tr.windows["decision"]
    inwin = (t >= lo) & (t < hi)
    spiked = set(tr.raster.neurons[inwin].tolist())
    out = {"checks": {k: v for k, v in tr.checks.items()}, "digests": tr.digests,
           "synapse_count": tr.synapse_count}
    esc = PV.escape_initiator(tr, groups, "decision")
    heads = {}
    if esc.get("initiator_cells"):
        ei = np.array(sorted(esc["initiator_cells"].values()), np.int64)
        heads["escape"] = ei
        fs = {}
        for key, i in esc["initiator_cells"].items():
            pos = tr.spikes_of(i, "decision")
            fs[key] = tr.spike_share(int(pos[0])) if len(pos) else None
        out["escape_first_spike"] = fs
    st = [i for k in ("nIII_dorsal_left", "nIII_dorsal_right", "nMLF_left", "nMLF_right") for i in groups[k].tolist() if i in spiked]
    tu = [i for k in ("turning_left", "turning_right") for i in groups[k].tolist() if i in spiked]
    if st:
        heads["strike"] = np.array(sorted(st), np.int64)
    if tu:
        heads["turn"] = np.array(sorted(tu), np.int64)
    out["deciding"] = {g: tr.share(v, "decision") for g, v in heads.items()}
    use = ["escape"] if cond["kind"] == "loom" else (["strike", "turn"] if cond["kind"] == "prey" else ["escape", "strike", "turn"])
    idx = [heads[g] for g in use if g in heads]
    out["headline_groups"] = [g for g in use if g in heads]
    out["headline"] = tr.share(np.unique(np.concatenate(idx)), "decision") if idx else None
    out["population"] = {k: tr.share(v, "decision") for k, v in groups.items() if len(v)}
    return out


# =============================================================================================
# 6. choices fixed before tuning (written to data/g1c/gate/choices.json with the time)
# =============================================================================================

CHOICES = {
    "written_before": "any tuning or gate simulation of the bridged network",
    "M1_comparison": "primary: the two booleans (H1 pass, H2 pass) of each shuffled fish (held-out H1/H2 conditions, gate "
                     "seeds G1-seed-0..3) against the intact fish scored on the SAME seeds 0-3 (like for like); the page "
                     "line applies when they are equal in >= 8 of 10 shuffles. Secondary rows: the same against the intact "
                     "8-seed verdict, and the per-side four booleans.",
    "L1_nonvacuity": "L1 is scored on the bilateral-lesion trials paired with intact trials in which the intact fish "
                     "escaped. t_med = the intact median first M-system spike time per held-out loom condition. Pass = 0 "
                     "M-system (escape readout) spikes within +/-20 ms of t_med in those trials AND the intact fish escapes "
                     "with P >= 0.5 per stimulus side on the same trials (H1's threshold). If the intact fish does not, "
                     "L1 is 'not evaluable (intact did not escape)', which is not a pass.",
    "L2_scoring": "per lesioned side X (task_criteria left or right M-system): escapes initiated on side X == 0, and the "
                  "held-out looms on the other stimulus side keep P(escape) >= 0.5 and initiator-on-stimulus-side >= 0.75 "
                  "(pooled over that side's 4 conditions, 32 trials). Both lesions must pass. Union sets: sensitivity rows.",
    "H1_zero_escapes": "side and angle fractions of a side with no escapes are 0 (fail)",
    "C1_C2_zero_loom": "a control fails when the baseline loom's mean Mauthner spikes are 0 (as G1)",
    "deciding_cells": "headline = provenance share over the spec's deciding cells: loom -> the initiating M-system cell(s); "
                      "prey -> every nIII_dorsal and nMLF cell and every turning SPN that spiked in the window. Pooled over "
                      "trials as sum(measured drive) / sum(link drive). Population-level shares are a labelled secondary row.",
    "eye_root_set": "provenance root targets = the G1c stimulated set (24,920 ids), pinned in every trace's digests",
    "ablations_on_intact_node_set": "M2 ablations remove bridge edges from the intact bridged node set (exact; keeps the "
                                    "eye's target list, so the gate seeds stay paired)",
    "workers": "process pool, spawn context; results do not depend on the worker count (every trial is seeded by its "
               "own seed string)",
}


def write_choices():
    p = OUT / "choices.json"
    if p.exists():
        return json.load(open(p))
    d = {"written_at": now(), "choices": CHOICES}
    _jdump(d, p)
    return d


# =============================================================================================
# 7. process pool
# =============================================================================================

def _pool(workers: int):
    return cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))


def run_jobs(jobs: List[dict], workers: int, on_done=None) -> List[dict]:
    """Run jobs in a spawn pool; results come back in job order. on_done(i, result) as each finishes."""
    res: List[Optional[dict]] = [None] * len(jobs)
    if workers <= 1:
        for i, j in enumerate(jobs):
            res[i] = run_job(j)
            if on_done:
                on_done(i, res[i])
        return res
    with _pool(workers) as ex:
        futs = {ex.submit(run_job, j): i for i, j in enumerate(jobs)}
        for f in cf.as_completed(futs):
            i = futs[f]
            res[i] = f.result()
            if on_done:
                on_done(i, res[i])
    return res


# =============================================================================================
# 8. tuning: the two class gains on the TRAINING set only
# =============================================================================================

def training_metrics(trials_by_cond: Mapping[str, List[dict]], conds: Sequence[Mapping]) -> dict:
    """The spec's 10 training checks, cross-talk, and every number behind them."""
    thr = {"p": 0.5, "side": 0.75, "angle": 0.75, "angle_deg": 12.0, "strike": 0.5, "turn": 0.75}
    m, checks = {}, {}
    xtalk = 0
    for c in conds:
        tr = trials_by_cond[c["name"]]
        s = c["side"]
        if c["kind"] == "loom":
            esc = [t for t in tr if t["escape"]]
            p = len(esc) / len(tr)
            sf = sum(t["cell_side"] == s for t in esc) / len(esc) if esc else 0.0
            af = sum(t.get("escape_angle_deg", 0) >= thr["angle_deg"] for t in esc) / len(esc) if esc else 0.0
            m[c["name"]] = {"p_escape": p, "side_frac": sf, "angle_frac": af,
                            "strikes": sum(t["strike"] for t in tr), "mean_m_spikes": float(np.mean([sum(t["m_spikes"].values()) for t in tr]))}
            checks[f"loom_{s}_p_escape"] = p >= thr["p"]
            checks[f"loom_{s}_side"] = sf >= thr["side"]
            checks[f"loom_{s}_angle"] = af >= thr["angle"]
            xtalk += sum(t["strike"] for t in tr)
        else:
            sf = sum(t["strike"] for t in tr) / len(tr)
            tf = sum(t["turn"] == s for t in tr) / len(tr)
            m[c["name"]] = {"strike_frac": sf, "turn_toward_frac": tf, "escapes": sum(t["escape"] for t in tr)}
            checks[f"prey_{s}_strike"] = sf >= thr["strike"]
            checks[f"prey_{s}_turn"] = tf >= thr["turn"]
            xtalk += sum(t["escape"] for t in tr)
    return {"checks": checks, "n_pass": int(sum(checks.values())), "crosstalk": int(xtalk), "metrics": m}


def tune(spec: Optional[dict] = None, workers: int = 5, topo_tag: str = "intact") -> dict:
    """The full 6 x 6 grid of (g_a, g_c) on the TRAINING stimuli with the tuning seeds. Nothing else
    moves. Every finished job and every finished grid point is appended to tuning_log.jsonl with its
    time; the pick follows the spec's objective (max checks, then min cross-talk, then min g_c, min g_a)."""
    spec = load_spec() if spec is None else spec
    log_p = OUT / "tuning_log.jsonl"
    if log_p.exists() and any(json.loads(l).get("event") == "start" for l in open(log_p) if l.strip()):
        raise RuntimeError(f"{log_p} already holds a tuning run; the log is append-only evidence. Move it aside "
                           "(and tuning.json) before re-tuning.")
    write_choices()
    tuning = spec["tuning"]
    seeds = list(tuning["seeds"])
    grid_a = list(spec["gains"]["g_a"]["grid"])
    grid_c = list(spec["gains"]["g_c"]["grid"])
    conds = conditions(spec, "training")          # the ONLY stimuli this function renders
    topo = load_topology(topo_tag)
    ret = GateRetina(topo["stim_ids"])
    rp = {c["name"]: str(rates_path(ret, c)) for c in conds}
    log_p = OUT / "tuning_log.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)

    def log(entry):
        entry = {"time": now(), **entry}
        with open(log_p, "a") as f:
            f.write(json.dumps(entry, default=_json_default) + "\n")

    log({"event": "start", "grid_a": grid_a, "grid_c": grid_c, "seeds": seeds, "workers": workers,
         "stimuli": [{"name": c["name"], "param_sha256": c["hash"], "rates": Path(rp[c["name"]]).name} for c in conds],
         "topology": topo["key"], "what_moves": "g_a, g_c only"})
    jobs = []
    for ga in grid_a:
        for gc in grid_c:
            for c in conds:
                jobs.append(dict(topo=topo_tag, g_a=ga, g_c=gc, ablate=[], cond=c, seeds=seeds, rates_path=rp[c["name"]]))
    done: Dict[Tuple[int, int], Dict[str, List[dict]]] = {}
    table = []

    def on_done(i, r):
        j = jobs[i]
        key = (j["g_a"], j["g_c"])
        done.setdefault(key, {})[r["cond"]] = r["trials"]
        log({"event": "job", "g_a": j["g_a"], "g_c": j["g_c"], "cond": r["cond"], "param_sha256": j["cond"]["hash"],
             "digest": r["trials"][0]["digest"], "hashes": [t["spike_hash"] for t in r["trials"]]})
        if len(done[key]) == len(conds):
            tm = training_metrics(done[key], conds)
            row = {"g_a": key[0], "g_c": key[1], "digest": r["trials"][0]["digest"], **tm}
            table.append(row)
            log({"event": "grid_point", **row})
            _log("grid", key, "checks", tm["n_pass"], "xtalk", tm["crosstalk"])

    run_jobs(jobs, workers, on_done)
    table.sort(key=lambda r: (-r["n_pass"], r["crosstalk"], r["g_c"], r["g_a"]))
    best = table[0]
    rule = ("objective: max checks passed (10), then min cross-talk (strikes on loom + escapes on prey trials), "
            "then min g_c, then min g_a")
    log({"event": "chosen", "g_a": best["g_a"], "g_c": best["g_c"], "n_pass": best["n_pass"],
         "crosstalk": best["crosstalk"], "rule": rule})
    out = {"when": now(), "chosen": {"g_a": best["g_a"], "g_c": best["g_c"]}, "rule": rule, "table": table,
           "training_stimulus_hashes": {c["name"]: c["hash"] for c in conds}, "seeds": seeds}
    _jdump(out, OUT / "tuning.json")
    return out



# =============================================================================================
# 9. the gate (held-out stimuli, controls, lesions, determinism, provenance)
# =============================================================================================

def _by_cond(results: Sequence[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in results:
        out.setdefault(r["cond"], []).extend(r["trials"])
    return out


def score_H1(trials_by_cond: Mapping[str, List[dict]], loom_conds: Sequence[Mapping], thr: Mapping) -> dict:
    out = {}
    for s in ("left", "right"):
        tr = [t for c in loom_conds if c["side"] == s for t in trials_by_cond[c["name"]]]
        esc = [t for t in tr if t["escape"]]
        p = len(esc) / len(tr) if tr else 0.0
        sf = sum(t["cell_side"] == s for t in esc) / len(esc) if esc else 0.0
        af = sum(t.get("escape_angle_deg", 0.0) >= thr["angle_min_deg"] for t in esc) / len(esc) if esc else 0.0
        late = [t for t in esc if t.get("escape_after_expansion_end_ms", -1e9) > thr["after_expansion_max_ms"]]
        ok = p >= thr["p_escape_min"] and sf >= thr["side_frac_min"] and af >= thr["angle_frac_min"] and not late
        out[s] = {"pass": bool(ok), "trials": len(tr), "escapes": len(esc), "p_escape": p, "side_frac": sf,
                  "angle_frac": af, "late_escapes": len(late),
                  "per_condition": {c["name"]: _loom_row(trials_by_cond[c["name"]], s, thr) for c in loom_conds if c["side"] == s}}
    return {"pass": all(v["pass"] for v in out.values()), "per_side": out}


def _loom_row(tr: List[dict], s: str, thr: Mapping) -> dict:
    esc = [t for t in tr if t["escape"]]
    return {"p_escape": len(esc) / len(tr), "side_frac": (sum(t["cell_side"] == s for t in esc) / len(esc)) if esc else None,
            "angle_frac": (sum(t.get("escape_angle_deg", 0) >= thr["angle_min_deg"] for t in esc) / len(esc)) if esc else None,
            "initiator_sides": [t["cell_side"] for t in tr], "escape_t_ms": [t["escape_t_ms"] for t in tr],
            "angles_deg": [t.get("escape_angle_deg") for t in tr], "strikes": sum(t["strike"] for t in tr),
            "mean_mauthner_spikes": float(np.mean([t["mauthner_spikes"] for t in tr])),
            "spike_hashes": [t["spike_hash"] for t in tr]}


def score_H2(trials_by_cond: Mapping[str, List[dict]], prey_conds: Sequence[Mapping], thr: Mapping) -> dict:
    out = {}
    for s in ("left", "right"):
        tr = [t for c in prey_conds if c["side"] == s for t in trials_by_cond[c["name"]]]
        sf = sum(t["strike"] for t in tr) / len(tr) if tr else 0.0
        tf = sum(t["turn"] == s for t in tr) / len(tr) if tr else 0.0
        out[s] = {"pass": bool(sf >= thr["strike_frac_min"] and tf >= thr["turn_toward_frac_min"]), "trials": len(tr),
                  "strike_frac": sf, "turn_toward_frac": tf,
                  "nMLF_side_index_reported": [round(t["nMLF_side_index"], 4) for t in tr],
                  "per_condition": {c["name"]: {"strike_frac": sum(t["strike"] for t in trials_by_cond[c["name"]]) / len(trials_by_cond[c["name"]]),
                                                "turn_toward_frac": sum(t["turn"] == s for t in trials_by_cond[c["name"]]) / len(trials_by_cond[c["name"]]),
                                                "turn_index": [round(t["turn_index"], 4) for t in trials_by_cond[c["name"]]],
                                                "escapes": sum(t["escape"] for t in trials_by_cond[c["name"]]),
                                                "spike_hashes": [t["spike_hash"] for t in trials_by_cond[c["name"]]]}
                                  for c in prey_conds if c["side"] == s}}
    return {"pass": all(v["pass"] for v in out.values()), "per_side": out}


def tuning_hashes() -> set:
    hs = set()
    p = OUT / "tuning_log.jsonl"
    for line in open(p):
        e = json.loads(line)
        if "param_sha256" in e:
            hs.add(e["param_sha256"])
        for st in e.get("stimuli", []) or []:
            if isinstance(st, dict) and "param_sha256" in st:
                hs.add(st["param_sha256"])
    return hs


def held_out_hash_check(spec: Mapping) -> dict:
    tuned = tuning_hashes()
    ho = conditions(spec, "held_out_loom") + conditions(spec, "held_out_prey") + conditions(spec, "controls")
    hit = [c["name"] for c in ho if c["hash"] in tuned]
    train = {c["hash"] for c in conditions(spec, "training")}
    return {"pass": not hit and tuned <= train, "tuning_hashes": sorted(tuned), "training_hashes": sorted(train),
            "held_out_and_control_hashes": {c["name"]: c["hash"] for c in ho}, "held_out_or_control_in_tuning": hit,
            "tuning_only_training": sorted(tuned) == sorted(train)}


def gate(spec: Optional[dict] = None, workers: int = 5) -> dict:
    spec = load_spec() if spec is None else spec
    t0 = time.time()
    T = json.load(open(OUT / "tuning.json"))
    ga, gc = T["chosen"]["g_a"], T["chosen"]["g_c"]
    hc = held_out_hash_check(spec)
    if not hc["pass"]:
        raise AssertionError(f"held-out hash check failed: {hc['held_out_or_control_in_tuning']}")
    topo = load_topology("intact")
    bw = network(topo, ga, gc)
    ret = GateRetina(topo["stim_ids"])
    seeds = list(spec["stimuli"]["gate_seeds"])
    HL, HP = conditions(spec, "held_out_loom"), conditions(spec, "held_out_prey")
    CT, C1 = conditions(spec, "controls"), conditions(spec, "c1_baseline")
    rp = {c["name"]: str(rates_path(ret, c)) for c in HL + HP + CT + C1}
    stim_all = g1c_stimulated_ids(B.load_pairs(), B.identified())
    eye = [int(x) for x in stim_all]
    base = dict(topo="intact", g_a=ga, g_c=gc, ablate=[], seeds=seeds)
    _log("gate: intact held-out + controls + C1 baseline", ga, gc)
    J = [dict(base, cond=c, rates_path=rp[c["name"]]) for c in HL + HP + CT + C1]
    R = run_jobs(J, workers)
    I = _by_cond(R)
    _jdump({"when": now(), "gains": {"g_a": ga, "g_c": gc}, "I": I}, OUT / "gate_raw_phase1.json")
    # lesions (paired: same net, dead cells, same eye targets)
    tmed = {}
    for c in HL:
        ts = [t["escape_t_ms"] for t in I[c["name"]] if t["escape"]]
        tmed[c["name"]] = float(np.median(ts)) if ts else None
    LS = lesion_sets()
    LJ = []
    for lname, ids_ in LS.items():
        for c in HL:
            j = dict(base, cond=c, rates_path=rp[c["name"]], lesion=ids_, lesion_name=lname)
            if lname in ("bilateral", "union_bilateral"):
                j["l1_tmed"] = tmed[c["name"]]
            LJ.append(j)
    _log("gate: lesions", len(LJ), "jobs")
    LR = run_jobs(LJ, workers)
    L = {}
    for j, r in zip(LJ, LR):
        L.setdefault(j["lesion_name"], {}).setdefault(r["cond"], []).extend(r["trials"])
    # provenance: every held-out intact trial traced (its raster must equal the gate run's: determinism)
    _log("gate: traced held-out runs (provenance + replay)")
    TJ = [dict(base, cond=c, rates_path=rp[c["name"]], trace=True, eye_segments=eye) for c in HL + HP]
    TR = run_jobs(TJ, workers)
    TI = _by_cond(TR)
    _jdump({"when": now(), "gains": {"g_a": ga, "g_c": gc}, "I": I, "L": L, "TI": TI, "tmed": tmed, "rp": rp},
           OUT / "gate_raw.json")
    ev = evaluate_gate(spec, I, L, TI, HL, HP, CT, C1, tmed, rp)
    out = {"when": now(), "seconds": round(time.time() - t0, 1), "gains": {"g_a": ga, "g_c": gc},
           "network_digest": bw.net.digest(), "bridged_digest": bw.digest, "edge_provenance_digest": bw.prov.digest(),
           "topology": {k: topo["stats"][k] for k in ("kept_neurons", "kept_pairs", "stimulated_kept", "bridge_edges_kept")},
           "held_out_hash_check": hc, "evaluation": ev, "t_med_ms": tmed,
           "trials": {"intact": I, "lesion": L, "traced": {k: [_strip_prov(t) for t in v] for k, v in TI.items()}},
           "provenance_trials": {k: [{"seed": t["seed"], "spike_hash": t["spike_hash"], **t["provenance"]} for t in v]
                                 for k, v in TI.items()},
           "rates": {k: Path(v).name for k, v in rp.items()}, "workers": workers, "loadavg_end": os.getloadavg()}
    _jdump(out, OUT / "gate_results.json")
    _log("gate verdict", ev["verdict"])
    return out


def _strip_prov(t: Mapping) -> dict:
    return {k: v for k, v in t.items() if k != "provenance"}


def evaluate_gate(spec, I, L, TI, HL, HP, CT, C1, tmed, rp) -> dict:
    G = spec["gate"]
    ev = {}
    ev["H1"] = score_H1(I, HL, G["H1"]["thresholds"])
    ev["H2"] = score_H2(I, HP, G["H2"]["thresholds"])
    # C1: receding motion window vs the same-side training loom at the gate seeds
    base_by_side = {c["side"]: I[c["name"]] for c in C1}
    lm = {s: float(np.mean([t["mauthner_spikes"] for t in base_by_side[s]])) for s in ("left", "right")}
    lp = {s: float(np.mean([t["escape"] for t in base_by_side[s]])) for s in ("left", "right")}
    c1 = {}
    for s in ("left", "right"):
        tr = I[f"recede_{s}"]
        m = float(np.mean([t["mauthner_spikes"] for t in tr]))
        p = float(np.mean([t["escape"] for t in tr]))
        ok = lm[s] > 0 and m <= G["C1"]["thresholds"]["ratio_max"] * lm[s] and p < lp[s]
        c1[s] = {"pass": bool(ok), "mean_mauthner_spikes": m, "p_escape": p, "loom_mean_mauthner_spikes": lm[s],
                 "loom_p_escape": lp[s], "onset_window_escapes_reported": [sum(t["onset_m_spikes"].values()) > 0 for t in tr],
                 "strikes": sum(t["strike"] for t in tr)}
    ev["C1"] = {"pass": all(v["pass"] for v in c1.values()), "per_side": c1}
    tr = I["dim"]
    m = float(np.mean([t["mauthner_spikes"] for t in tr]))
    p = float(np.mean([t["escape"] for t in tr]))
    lmm, lpm = float(np.mean(list(lm.values()))), float(np.mean(list(lp.values())))
    dim_diag = json.loads(str(np.load(rp["dim"], allow_pickle=False)["diag"]))
    eye_zero = dim_diag["expected_tectal_input_spikes"] == 0.0
    ev["C2"] = {"pass": bool(lmm > 0 and m <= G["C2"]["thresholds"]["ratio_max"] * lmm and p < lpm),
                "mean_mauthner_spikes": m, "p_escape": p, "loom_mean_mauthner_spikes": lmm, "loom_p_escape": lpm,
                "eye_model_zeroes_tectal_input": bool(eye_zero),
                "label": "passed by the eye, not by the brain" if eye_zero else "scored on the brain's output",
                "tectal_spikes": [t["tectal_spikes"] for t in tr]}
    tr = I["blank"]
    quiet = sum((not t["escape"]) and (not t["strike"]) for t in tr)
    turns = [t["turn"] for t in tr]
    fr = max(sum(x == "right" for x in turns), sum(x == "left" for x in turns)) / len(tr)
    vac = all(t["n_spikes"] == 0 for t in tr)
    ev["C3"] = {"pass": bool(quiet >= 7 and fr < 0.75), "quiet_seeds": quiet, "max_turn_direction_frac": fr, "turns": turns,
                "total_spikes": [t["n_spikes"] for t in tr],
                "label": "passed by construction (no background input: 0 spikes)" if vac else "scored"}
    # L1: bilateral M-system lesion; non-vacuous only when the intact fish escaped
    bl = L["bilateral"]
    rows, short, pairs_esc = [], 0, 0
    intact_side = {}
    for c in HL:
        for ti, tl in zip(I[c["name"]], bl[c["name"]]):
            assert ti["seed"] == tl["seed"]
            if ti["escape"]:
                pairs_esc += 1
                short += int((tl.get("l1") or {}).get("m_system_spikes_within_20ms", 0) > 0 or tl["escape"])
            l1 = tl.get("l1") or {}
            rows.append({"cond": c["name"], "seed": ti["seed"], "intact_escape": ti["escape"], "lesioned_escape": tl["escape"],
                         "t_med_ms": l1.get("t_med_ms"), "other_deciding_spikes_within_20ms": l1.get("other_deciding_spikes_within_20ms", 0)})
        intact_side.setdefault(c["side"], []).extend(t["escape"] for t in I[c["name"]])
    nonvac = {s: float(np.mean(v)) >= G["H1"]["thresholds"]["p_escape_min"] for s, v in intact_side.items()}
    informative = sum(r["other_deciding_spikes_within_20ms"] > 0 for r in rows if r["intact_escape"])
    ev["L1"] = {"pass": bool(all(nonvac.values()) and short == 0 and pairs_esc > 0),
                "evaluable": bool(all(nonvac.values()) and pairs_esc > 0),
                "status": ("pass" if (all(nonvac.values()) and short == 0 and pairs_esc > 0) else
                           ("not evaluable (intact did not escape)" if not all(nonvac.values()) else "fail")),
                "paired_trials_intact_escaped": pairs_esc, "short_latency_escapes_lesioned": short,
                "intact_p_escape_by_side": {s: float(np.mean(v)) for s, v in intact_side.items()},
                "label": "passes by construction when evaluable (the escape readout is the lesioned set)",
                "informative_row_other_deciding_readouts_within_20ms": informative, "rows": rows}
    # L2: unilateral task_criteria lesions
    l2 = {}
    for x in ("left", "right"):
        les = L[f"unilateral_{x}"]
        init_x = sum(t["escape"] and t["cell_side"] == x for c in HL for t in les[c["name"]])
        other = "right" if x == "left" else "left"
        H = score_H1(les, [c for c in HL], G["H1"]["thresholds"])["per_side"]
        keep = H[other]
        ok = init_x == 0 and keep["p_escape"] >= G["L2"]["thresholds"]["p_escape_min"] and \
            keep["side_frac"] >= G["L2"]["thresholds"]["side_frac_min"]
        l2[x] = {"pass": bool(ok), "escapes_initiated_on_lesioned_side": init_x,
                 "non_lesioned_side_looms": {k: keep[k] for k in ("p_escape", "side_frac", "angle_frac", "trials", "escapes")},
                 "lesioned_side_looms": {k: H[x][k] for k in ("p_escape", "side_frac", "trials", "escapes")}}
    ev["L2"] = {"pass": all(v["pass"] for v in l2.values()), "per_lesion": l2}
    # union lesion sets: sensitivity rows (never scored)
    sens = {}
    for x in ("left", "right"):
        les = L[f"union_unilateral_{x}"]
        H = score_H1(les, HL, G["H1"]["thresholds"])["per_side"]
        sens[f"union_unilateral_{x}"] = {"escapes_initiated_on_lesioned_side": sum(t["escape"] and t["cell_side"] == x
                                                                                   for c in HL for t in les[c["name"]]),
                                         "per_side": {s: {k: H[s][k] for k in ("p_escape", "side_frac", "escapes")} for s in H}}
    ub = L["union_bilateral"]
    sens["union_bilateral"] = {"escapes": sum(t["escape"] for c in HL for t in ub[c["name"]])}
    ev["lesion_sensitivity_union"] = sens
    # cross-talk (reported, not scored)
    xt = {}
    for c in HL + C1:
        xt[c["name"]] = {"strikes_on_loom": sum(t["strike"] for t in I[c["name"]]), "trials": len(I[c["name"]])}
    for c in HP:
        xt[c["name"]] = {"escapes_on_prey": sum(t["escape"] for t in I[c["name"]]), "trials": len(I[c["name"]])}
    for c in CT:
        xt[c["name"]] = {"escapes": sum(t["escape"] for t in I[c["name"]]), "strikes": sum(t["strike"] for t in I[c["name"]]),
                         "trials": len(I[c["name"]])}
    ev["crosstalk"] = xt
    # determinism: every traced run replays the gate run's raster exactly
    mism = []
    for c in HL + HP:
        for a, b in zip(I[c["name"]], TI[c["name"]]):
            if a["seed"] != b["seed"] or a["spike_hash"] != b["spike_hash"]:
                mism.append((c["name"], a["seed"]))
    ev["determinism"] = {"pass": not mism, "trials_compared": sum(len(TI[c["name"]]) for c in HL + HP), "mismatches": mism,
                         "how": "each held-out trial run twice (plain Simulator, then provenance.trial_traced's replay run) in separate worker processes; spike hashes compared"}
    prov_ok = all(t["provenance"]["checks"].get("mirror_ok") and
                  all(p["ok"] for p in t["provenance"]["checks"].get("parts_sum_to_total", []))
                  for v in TI.values() for t in v)
    ev["provenance_self_checks_ok"] = bool(prov_ok)
    ev["headline"] = pooled_headline(TI, HL, HP)
    H, Cc = ev["H1"]["pass"] and ev["H2"]["pass"], ev["C1"]["pass"] and ev["C2"]["pass"] and ev["C3"]["pass"]
    ev["kill"] = not (H and Cc)
    ev["verdict"] = "PASS" if (H and Cc and ev["L1"]["pass"] and ev["L2"]["pass"]) else ("KILL" if ev["kill"] else "FAIL_NOT_KILL")
    return ev


def pooled_headline(TI: Mapping[str, List[dict]], HL, HP) -> dict:
    """Decision-path share pooled over trials: sum(m * link drive) / sum(link drive), per condition,
    per stimulus kind, and overall; per-class 'through' shares pooled the same way."""
    def pool(rows):
        link = sum(r["link_drive_mV_ms"] for r in rows if r and r.get("m") is not None)
        if link <= 0:
            return {"m": None, "link_drive_mV_ms": 0.0, "trials_with_drive": 0}
        meas = sum(r["m"] * r["link_drive_mV_ms"] for r in rows if r and r.get("m") is not None)
        thr = {c: sum(r["bridge_through"].get(c, 0.0) * r["link_drive_mV_ms"] for r in rows if r and r.get("m") is not None) / link
               for c in CLASSES}
        only = {c: sum(r["bridge_only"].get(c, 0.0) * r["link_drive_mV_ms"] for r in rows if r and r.get("m") is not None) / link
                for c in CLASSES}
        eye = sum((r.get("chosen_share") or {}).get("chosen (eye model)") or 0.0 for r in rows if r) / max(1, len([r for r in rows if r]))
        return {"m": meas / link, "measured_drive_mV_ms": meas, "link_drive_mV_ms": link, "through": thr, "only": only,
                "trials_with_drive": sum(1 for r in rows if r and r.get("m") is not None),
                "mean_direct_eye_share": eye}
    out = {"per_condition": {}, "by_group": {}}
    for c in HL + HP:
        out["per_condition"][c["name"]] = pool([t["provenance"]["headline"] for t in TI[c["name"]]])
    out["loom_escape"] = pool([t["provenance"]["headline"] for c in HL for t in TI[c["name"]]])
    out["prey_strike_turn"] = pool([t["provenance"]["headline"] for c in HP for t in TI[c["name"]]])
    out["all_held_out"] = pool([t["provenance"]["headline"] for c in HL + HP for t in TI[c["name"]]])
    for g in ("escape", "strike", "turn"):
        out["by_group"][g] = pool([t["provenance"]["deciding"].get(g) for c in HL + HP for t in TI[c["name"]]])
    keys = sorted({k for v in TI.values() for t in v for k in t["provenance"]["population"]})
    out["population_secondary"] = {k: pool([t["provenance"]["population"].get(k) for c in HL + HP for t in TI[c["name"]]])
                                   for k in keys}
    return out



# =============================================================================================
# 10. mandatory publications: M1 shuffles (bridge fixed), M2 ablations (+ M3 metrics per class)
# =============================================================================================

def _h_outcomes(I: Mapping[str, List[dict]], HL, HP, spec, seeds: Optional[Sequence[str]] = None) -> dict:
    sub = I if seeds is None else {k: [t for t in v if t["seed"] in seeds] for k, v in I.items()}
    h1 = score_H1(sub, HL, spec["gate"]["H1"]["thresholds"])
    h2 = score_H2(sub, HP, spec["gate"]["H2"]["thresholds"])
    return {"H1": h1["pass"], "H2": h2["pass"],
            "per_side": {"H1_left": h1["per_side"]["left"]["pass"], "H1_right": h1["per_side"]["right"]["pass"],
                         "H2_left": h2["per_side"]["left"]["pass"], "H2_right": h2["per_side"]["right"]["pass"]},
            "numbers": {"H1": {s: {k: h1["per_side"][s][k] for k in ("p_escape", "side_frac", "angle_frac", "escapes", "trials")}
                               for s in ("left", "right")},
                        "H2": {s: {k: h2["per_side"][s][k] for k in ("strike_frac", "turn_toward_frac", "trials")}
                               for s in ("left", "right")}}}


def shuffles(spec: Optional[dict] = None, workers: int = 5, n: int = 10, keep_cache: bool = False) -> dict:
    """M1: degree-preserving shuffle of the MEASURED pairs only (sim.shuffled_copy, as G1), bridge
    edges re-added unchanged by segment id at the tuned gains, stimulated set unchanged; H1/H2 held-out
    conditions with gate seeds 0-3. Each shuffle is re-reduced (exact), so it has its own eye-target
    list; its W and rates are deleted after use unless keep_cache."""
    spec = load_spec() if spec is None else spec
    T = json.load(open(OUT / "tuning.json"))
    ga, gc = T["chosen"]["g_a"], T["chosen"]["g_c"]
    G = json.load(open(OUT / "gate_results.json"))
    HL, HP = conditions(spec, "held_out_loom"), conditions(spec, "held_out_prey")
    seeds = list(spec["stimuli"]["gate_seeds"])[:4]
    intact_like = _h_outcomes(G["trials"]["intact"], HL, HP, spec, seeds)
    intact_8 = {"H1": G["evaluation"]["H1"]["pass"], "H2": G["evaluation"]["H2"]["pass"]}
    rows = []
    P = B.load_pairs()
    path = OUT / "m1_shuffles.json"
    prev = json.load(open(path)) if path.exists() else {"rows": []}
    have = {r["shuffle"]: r for r in prev.get("rows", []) if r.get("g") == [ga, gc]}
    for k in range(n):
        seed = f"G1c-measured-shuffle-{k}"
        if seed in have:
            rows.append(have[seed])
            continue
        t0 = time.time()
        tag = f"m1_{k}"
        topo = build_topology(spec, tag, shuffle_seed=seed, P=P, rebuild=True)
        ret = GateRetina(topo["stim_ids"], cache=False)
        rp = {c["name"]: str(rates_path(ret, c)) for c in HL + HP}
        del ret
        J = [dict(topo=tag, g_a=ga, g_c=gc, ablate=[], seeds=seeds, cond=c, rates_path=rp[c["name"]]) for c in HL + HP]
        I = _by_cond(run_jobs(J, workers))
        o = _h_outcomes(I, HL, HP, spec)
        row = {"shuffle": seed, "g": [ga, gc], "when": now(), "seconds": round(time.time() - t0, 1),
               "topology": {kk: topo["stats"][kk] for kk in ("kept_neurons", "kept_pairs", "stimulated_kept", "bridge_edges_kept")},
               "shuffle_meta": topo["stats"].get("shuffle"), "bridge_collisions": topo["stats"]["bridge_collisions_with_measured"],
               "outcomes": o, "equal_to_intact_like_for_like": o["H1"] == intact_like["H1"] and o["H2"] == intact_like["H2"],
               "equal_to_intact_8_seed": o["H1"] == intact_8["H1"] and o["H2"] == intact_8["H2"],
               "per_side_equal_like_for_like": o["per_side"] == intact_like["per_side"],
               "hashes": {c: [t["spike_hash"] for t in v] for c, v in I.items()},
               "crosstalk": {"strikes_on_loom": sum(t["strike"] for c in HL for t in I[c["name"]]),
                             "escapes_on_prey": sum(t["escape"] for c in HP for t in I[c["name"]])}}
        rows.append(row)
        if not keep_cache:
            for v in rp.values():
                Path(v).unlink(missing_ok=True)
        _log("M1", seed, o["H1"], o["H2"], "equal", row["equal_to_intact_like_for_like"], f"{row['seconds']} s")
        _jdump({"rows": rows}, path)
    eq = sum(r["equal_to_intact_like_for_like"] for r in rows)
    out = {"when": now(), "gains": [ga, gc], "seeds": seeds, "n_shuffles": len(rows), "intact_like_for_like": intact_like,
           "intact_8_seed": intact_8, "equal_like_for_like": eq, "equal_8_seed": sum(r["equal_to_intact_8_seed"] for r in rows),
           "rule": "if equal in >= 8 of 10 shuffles, the page says 'the anatomy is real; the decision runs through the model'",
           "page_line_applies": bool(len(rows) >= 10 and eq >= 8), "rows": rows}
    _jdump(out, path)
    return out


ABLATIONS = {"i_all_bridge_removed": ("all",), "ii_class_a_removed": ("a",), "iii_class_c_removed": ("c",),
             "iv_E1_removed": ("E1",), "v_P1_P4_removed": ("P",)}


def ablations(spec: Optional[dict] = None, workers: int = 5) -> dict:
    """M2 (and M3's per-class metrics): each ablation on the intact bridged node set, H1/H2 held-out
    conditions, 8 gate seeds, every trial traced (provenance) so both metrics come per class."""
    spec = load_spec() if spec is None else spec
    T = json.load(open(OUT / "tuning.json"))
    ga, gc = T["chosen"]["g_a"], T["chosen"]["g_c"]
    HL, HP = conditions(spec, "held_out_loom"), conditions(spec, "held_out_prey")
    topo = load_topology("intact")
    ret = GateRetina(topo["stim_ids"])
    rp = {c["name"]: str(rates_path(ret, c)) for c in HL + HP}
    eye = [int(x) for x in g1c_stimulated_ids(B.load_pairs(), B.identified())]
    seeds = list(spec["stimuli"]["gate_seeds"])
    out = {"when": now(), "gains": [ga, gc], "rows": {}}
    path = OUT / "m2_ablations.json"
    for name, abl in ABLATIONS.items():
        t0 = time.time()
        bw = network(topo, ga, gc, abl)
        J = [dict(topo="intact", g_a=ga, g_c=gc, ablate=list(abl), seeds=seeds, cond=c, rates_path=rp[c["name"]],
                  trace=True, eye_segments=eye) for c in HL + HP]
        TI = _by_cond(run_jobs(J, workers))
        o = _h_outcomes(TI, HL, HP, spec)
        from fishbrain import provenance as PV
        out["rows"][name] = {"ablate": list(abl), "digest": bw.digest, "network_digest": bw.net.digest(),
                             "outcomes": o, "headline": pooled_headline(TI, HL, HP),
                             "synapse_count_reduced": PV.synapse_count_share(bw.net, bw.prov),
                             "crosstalk": {"strikes_on_loom": sum(t["strike"] for c in HL for t in TI[c["name"]]),
                                           "escapes_on_prey": sum(t["escape"] for c in HP for t in TI[c["name"]])},
                             "per_condition": {c["name"]: {"escapes": sum(t["escape"] for t in TI[c["name"]]),
                                                           "initiator_sides": [t["cell_side"] for t in TI[c["name"]]],
                                                           "strikes": sum(t["strike"] for t in TI[c["name"]]),
                                                           "turns": [t["turn"] for t in TI[c["name"]]]}
                                               for c in HL + HP},
                             "provenance_self_checks_ok": all(t["provenance"]["checks"].get("mirror_ok") for v in TI.values() for t in v),
                             "seconds": round(time.time() - t0, 1)}
        _log("M2", name, o["H1"], o["H2"], f"{round(time.time() - t0, 1)} s")
        _jdump(out, path)
    return out


# =============================================================================================
# 11. JS parity on the bridged network, and the synapse-count shares
# =============================================================================================

def js_parity(spec: Optional[dict] = None, cond_name: str = "prey_az+15_v60", seed: str = "G1-seed-0") -> dict:
    """Export one bridged gate trial as a JS case (js/export_case.make_case: the bridge is plain extra
    (pre, post, count) pairs) and run js/run_case.mjs once. Records exit code and mismatches."""
    import subprocess
    sys.path.insert(0, str(ROOT / "js"))
    import export_case as XC
    spec = load_spec() if spec is None else spec
    T = json.load(open(OUT / "tuning.json"))
    ga, gc = T["chosen"]["g_a"], T["chosen"]["g_c"]
    topo = load_topology("intact")
    bw = network(topo, ga, gc)
    ret = GateRetina(topo["stim_ids"])
    cond = [c for c in conditions(spec, "held_out_prey") + conditions(spec, "held_out_loom") if c["name"] == cond_name][0]
    rates, diag = rates_for(ret, cond)
    total = rates.shape[0] * FRAME_MS
    # exact length: 500 ms chunks plus the remainder (rounding to whole chunks ran looms 105 ms long and broke the
    # comparison with the recorded trial; found 2026-09-26)
    full = int(total // 500.0)
    rem = total - 500.0 * full
    runs = [{"duration_ms": 500.0}] * full + ([{"duration_ms": rem}] if rem > 1e-9 else [])
    meta = {"condition": cond_name, "g_a": ga, "g_c": gc, "bridge_version": BRIDGE_VERSION,
            "edge_list_sha256": EDGE_LIST_SHA256, "bridged_digest": bw.digest, "edge_provenance_digest": bw.prov.digest(),
            "rates_sha256": diag["rates_sha256"]}
    case = XC.make_case(f"g1c-bridged-{cond_name}-{seed}", bw.net, {}, seed,
                        [XC.source("poisson", bw.stim_idx, rates, frame_ms=FRAME_MS, name="retina")], runs,
                        with_ids=True, meta=meta)
    G = json.load(open(OUT / "gate_results.json"))
    rec = [t for t in G["trials"]["intact"][cond_name] if t["seed"] == seed][0]
    case["expected"]["recorded_spike_hash_matches"] = case["expected"]["raster_hash"] == rec["spike_hash"]
    p = XC.write(case, OUT / "js")
    t0 = time.time()
    r = subprocess.run(["node", str(ROOT / "js" / "run_case.mjs"), str(p)], capture_output=True, text=True, timeout=3600)
    try:
        js = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        js = {"raw": r.stdout[-2000:], "stderr": r.stderr[-2000:]}
    out = {"when": now(), "case": p.name, "case_bytes": p.stat().st_size, "n": bw.net.n, "nnz": bw.net.nnz,
           "exit_code": r.returncode, "mismatches": js.get("mismatches"), "js_network_digest": js.get("network_digest"),
           "python_network_digest": bw.net.digest(), "js_raster_hash": js.get("raster_hash"),
           "python_raster_hash": case["expected"]["raster_hash"], "gate_recorded_spike_hash": rec["spike_hash"],
           "recorded_spike_hash_matches": case["expected"]["recorded_spike_hash_matches"],
           "active_simulator_identical": case["expected"].get("active_simulator_identical"),
           "bridged_digest_from_js_network_digest": _sha_json({"network_digest": js.get("network_digest"), "bridge_version": BRIDGE_VERSION,
                                                               "edge_list_sha256": EDGE_LIST_SHA256, "g_a": int(ga), "g_c": int(gc),
                                                               "ablate": [], "topology": "intact",
                                                               "edge_provenance": bw.prov.digest()}),
           "bridged_digest_python": bw.digest, "timing": js.get("timing"), "wall_s": round(time.time() - t0, 1)}
    out["bridged_digest_matches"] = out["bridged_digest_from_js_network_digest"] == bw.digest
    _jdump(out, OUT / "js_parity.json")
    return out


def synapse_count_whole_brain(ga: int, gc: int, spec: Optional[dict] = None) -> dict:
    """Secondary metric (1): the whole brain of record plus the bridge, two conventions."""
    spec = load_spec() if spec is None else spec
    P = B.load_pairs()
    signed, _ = B.signed_counts(P, RULE)
    raw = int(P["n_synapses"])
    by_cls = {"a": 0, "c": 0}
    for e in spec["edges"]:
        by_cls[e["cls"]] += int(e["base"]) * (ga if e["cls"] == "a" else gc)
    abs_signed = int(np.abs(signed.astype(np.int64)).sum())
    tot_raw = raw + sum(by_cls.values())
    tot_abs = abs_signed + sum(by_cls.values())
    return {"raw_synapses": {"measured": raw, "bridge_a": by_cls["a"], "bridge_c": by_cls["c"], "total": tot_raw,
                             "measured_share": raw / tot_raw, "class_share": {c: v / tot_raw for c, v in by_cls.items()}},
            "abs_signed_counts": {"measured": abs_signed, "bridge_a": by_cls["a"], "bridge_c": by_cls["c"], "total": tot_abs,
                                  "measured_share": abs_signed / tot_abs, "class_share": {c: v / tot_abs for c, v in by_cls.items()}},
            "formula": f"bridge synapse-equivalents = 646 x g_a + 7,397 x g_c = {by_cls['a'] + by_cls['c']}"}



def fireable_network(ga: int, gc: int, spec: Optional[dict] = None) -> Tuple[S.Network, np.ndarray]:
    """The whole fireable set F of the bridged brain (every cell reachable from the eye's targets
    through excitatory pairs, measured or bridged, with every nonzero pair inside F) at gains
    (ga, gc). Used only by the reduction-exactness test."""
    spec = load_spec() if spec is None else spec
    P = B.load_pairs()
    ids = P["ids"]
    n = len(ids)
    signed, _ = B.signed_counts(P, RULE)
    nz = signed != 0
    E = spec["edges"]
    bpre = B.index_of(ids, [e["pre"] for e in E])
    bpost = B.index_of(ids, [e["post"] for e in E])
    bc = np.array([e["base"] * (ga if e["cls"] == "a" else gc) for e in E], np.int64)
    pre = np.concatenate([P["pre"][nz].astype(np.int64), bpre])
    post = np.concatenate([P["post"][nz].astype(np.int64), bpost])
    cnt = np.concatenate([signed[nz].astype(np.int64), bc])
    stim = B.index_of(ids, g1c_stimulated_ids(P, B.identified()))
    pos = cnt > 0
    ip, ix = B._csr(n, pre[pos], post[pos])
    F = B.bfs_levels(n, ip, ix, stim) >= 0
    inF = F[pre] & F[post]
    fidx = np.flatnonzero(F)
    loc = np.full(n, -1, np.int64)
    loc[fidx] = np.arange(len(fidx))
    net = S.Network.from_pairs(loc[pre[inF]], loc[post[inF]], cnt[inF], len(fidx), ids=ids[fidx])
    return net, np.sort(ids[stim])


def report(spec: Optional[dict] = None) -> dict:
    """Assemble every mandatory publication into data/g1c/gate/g1c_summary.json."""
    from fishbrain import provenance as PV
    spec = load_spec() if spec is None else spec
    T = json.load(open(OUT / "tuning.json"))
    ga, gc = T["chosen"]["g_a"], T["chosen"]["g_c"]
    G = json.load(open(OUT / "gate_results.json"))
    topo = load_topology("intact")
    bw = network(topo, ga, gc)
    ev = G["evaluation"]
    m1 = json.load(open(OUT / "m1_shuffles.json")) if (OUT / "m1_shuffles.json").exists() else None
    m2 = json.load(open(OUT / "m2_ablations.json")) if (OUT / "m2_ablations.json").exists() else None
    js = json.load(open(OUT / "js_parity.json")) if (OUT / "js_parity.json").exists() else None
    red = PV.synapse_count_share(bw.net, bw.prov)
    ro_idx = [i for k in DECIDING_KEYS for i in PV.readout_groups(bw.net, {k: deciding_sets()[k]})[0][k].tolist()]
    red_onto = PV.synapse_count_share(bw.net, bw.prov, onto=ro_idx)
    whole = synapse_count_whole_brain(ga, gc, spec)
    log_lines = [json.loads(l) for l in open(OUT / "tuning_log.jsonl")]
    grid_pts = [e for e in log_lines if e.get("event") == "grid_point"]
    e1 = (m2 or {}).get("rows", {}).get("iv_E1_removed")
    disclosure = {"crossing_built": True,
                  "sentence": "Which way the fish flees is set by the published pathway model, not by the measured wiring.",
                  "applies": True,
                  "why": "E1 (the crossed tectum -> contralateral M-system edges) is built and kept in the gate network; "
                         "the brief pre-registers the sentence for that case.",
                  "E1_ablation_H1": None if e1 is None else e1["outcomes"]["H1"],
                  "E1_ablation_numbers": None if e1 is None else e1["outcomes"]["numbers"]["H1"]}
    out = {"when": now(), "spec_sha256": SPEC_SHA256, "edge_list_sha256": EDGE_LIST_SHA256, "bridge_version": BRIDGE_VERSION,
           "rederive": json.load(open(OUT / "rederive.json")) if (OUT / "rederive.json").exists() else None,
           "topology": {k: topo["stats"][k] for k in ("stimulated", "can_fire_F", "kept_neurons", "kept_pairs",
                                                     "stimulated_kept", "bridge_edges_kept", "build_seconds")},
           "gains": {"g_a": ga, "g_c": gc}, "tuning": {"rule": T["rule"], "grid_points_logged": len(grid_pts),
                                                     "first": log_lines[0].get("time"), "last": log_lines[-1].get("time"),
                                                     "chosen_row": [r for r in T["table"] if r["g_a"] == ga and r["g_c"] == gc][0]},
           "network_digest": bw.net.digest(), "bridged_digest": bw.digest, "edge_provenance_digest": bw.prov.digest(),
           "verdict": ev["verdict"], "kill": ev["kill"],
           "items": {k: ev[k]["pass"] for k in ("H1", "H2", "C1", "C2", "C3", "L1", "L2")},
           "L1_status": ev["L1"]["status"], "C2_label": ev["C2"]["label"], "C3_label": ev["C3"]["label"],
           "determinism": ev["determinism"], "held_out_hash_check": {k: G["held_out_hash_check"][k] for k in ("pass", "held_out_or_control_in_tuning", "tuning_only_training")},
           "provenance_self_checks_ok": ev["provenance_self_checks_ok"],
           "decision_path_share": ev["headline"],
           "synapse_count_share": {"whole_brain_plus_bridge": whole, "reduced_gate_network": red,
                                   "reduced_gate_network_onto_deciding_readouts": red_onto},
           "M1": None if m1 is None else {k: m1[k] for k in ("n_shuffles", "equal_like_for_like", "equal_8_seed", "page_line_applies",
                                                              "intact_like_for_like", "intact_8_seed", "rule")},
           "M1_rows": None if m1 is None else [{"shuffle": r["shuffle"], "H1": r["outcomes"]["H1"], "H2": r["outcomes"]["H2"],
                                                "numbers": r["outcomes"]["numbers"], "equal": r["equal_to_intact_like_for_like"]}
                                               for r in m1["rows"]],
           "M2": None if m2 is None else {k: {"H1": v["outcomes"]["H1"], "H2": v["outcomes"]["H2"], "numbers": v["outcomes"]["numbers"],
                                              "decision_path_share_loom": v["headline"]["loom_escape"],
                                              "decision_path_share_prey": v["headline"]["prey_strike_turn"],
                                              "synapse_count_reduced": v["synapse_count_reduced"], "crosstalk": v["crosstalk"]}
                                          for k, v in m2["rows"].items()},
           "M4_crosstalk": ev["crosstalk"], "disclosure": disclosure, "js_parity": js}
    _jdump(out, OUT / "g1c_summary.json")
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="FISHBRAIN G1c bridged fish")
    ap.add_argument("step", choices=["rederive", "topology", "tune", "gate", "shuffles", "ablations", "js", "report", "all"])
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    spec = load_spec()
    if a.step in ("rederive", "all"):
        rep = rederive(spec)
        _jdump(rep, OUT / "rederive.json")
        _log("rederive all_equal", rep["all_equal"], len(rep["checks"]))
    if a.step in ("topology", "all"):
        build_topology(spec, "intact", rebuild=True)
    if a.step in ("tune", "all"):
        t = tune(spec, workers=a.workers)
        _log("chosen", t["chosen"])
    if a.step in ("gate", "all"):
        gate(spec, workers=a.workers)
    if a.step in ("shuffles", "all"):
        shuffles(spec, workers=a.workers)
    if a.step in ("ablations", "all"):
        ablations(spec, workers=a.workers)
    if a.step in ("js", "all"):
        js_parity(spec)
    if a.step in ("report", "all"):
        report(spec)


if __name__ == "__main__":
    main()
