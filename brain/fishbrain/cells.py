"""FISHBRAIN identified cells: where vision enters the Fish1 brain and where behaviour is read out.

Two halves:

1. ``tectum_drive(stimulus)``: the retinotopic input. A visual scene, given per eye as lists of
   (azimuth_deg, elevation_deg, contrast), becomes one input value per optic-tectum cell listed in
   ``data/cells_identified.json``. Each eye drives only the CONTRALATERAL tectum.
2. ``build()``: the one-off, network-bound pipeline that produced ``data/cells_identified.json``
   (axes, tectum cells, both Mauthner cells, the 50 named spinal projection neurons and the cells
   that must never receive direct visual input). Run ``python -m fishbrain.cells`` to rebuild.

Every rule and parameter that is our choice (not measured) is listed in
``launches/fishbrain/evidence/G1-cells.md`` under "Choices we made". Coordinates are voxels of
8 x 8 x 30 nm (x, y, z) throughout: x anterior->posterior, y fish's right->left, z dorsal->ventral.

Stimulus conventions (ours):
    azimuth_deg   0 = straight ahead, positive = toward the fish's RIGHT, negative = LEFT, (-180, 180].
    elevation_deg 0 = horizon, positive = up (dorsal), negative = down.
    contrast      magnitude of the luminance change; the sign is ignored (tectal cells answer ON and
                  OFF; polarity is not modelled). One point of contrast 1 at a cell's preferred
                  location gives that cell exactly 1.0.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
REF = DATA / "ref"
CELLS_JSON = DATA / "cells_identified.json"

VOXEL_NM = (8, 8, 30)
PUBLIC = "precomputed://gs://fish1-public/"

# ---- Retinotopy parameters: OUR CHOICES, not fitted to data (see evidence note) ----
THETA_MIN_DEG = -20.0   # eye-lateral azimuth mapped to the anterior pole (20 deg into the frontal binocular zone)
THETA_MAX_DEG = 160.0   # eye-lateral azimuth mapped to the posterior pole (20 deg rear blind zone per side)
EL_MIN_DEG = -70.0      # elevation mapped to the ventrolateral edge of the tectum
EL_MAX_DEG = 70.0       # elevation mapped to the dorsomedial edge of the tectum
SIGMA_DEG = 15.0        # Gaussian receptive-field width (great-circle degrees)

CONTRA = {"left": "right", "right": "left"}  # eye -> tectal lobe it drives


# =============================================================================================
# Loading and geometry
# =============================================================================================

_CACHE: dict = {}


def load(path: os.PathLike = CELLS_JSON) -> dict:
    """Read cells_identified.json (cached per path)."""
    path = str(path)
    if path not in _CACHE:
        with open(path) as fh:
            _CACHE[path] = json.load(fh)
    return _CACHE[path]


def midline_y(x, axes: dict) -> np.ndarray:
    """Fitted midline y (8 nm voxels) at anterior-posterior position x."""
    m = axes["midline"]
    return m["a"] + m["b"] * np.asarray(x, dtype=float)


def side_of(pos, axes: dict) -> np.ndarray:
    """'left' / 'right' (the fish's own sides) for (N, 3) positions in 8 nm voxels."""
    pos = np.atleast_2d(np.asarray(pos, dtype=float))
    return np.where(pos[:, 1] > midline_y(pos[:, 0], axes), "left", "right")


def _rank01(a: np.ndarray) -> np.ndarray:
    """Rank-normalise to (0, 1): smallest -> ~0, largest -> ~1, ties broken by order."""
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = (np.arange(len(a)) + 0.5) / len(a)
    return r


def lobe_coords(pos: np.ndarray, side: np.ndarray, axes: dict):
    """Per tectal cell (u, v) within its own lobe, both in (0, 1).

    u: anterior (0) -> posterior (1), the rank of x inside the lobe.
    v: dorsomedial (0) -> ventrolateral (1), the rank of (rank(z) + rank(|y - midline|)) inside
       the lobe. In Fish1 the tectal cell layer curves so that dorsal cells are medial: the
       correlation of z with lateral distance is 0.74 (left lobe) and 0.77 (right lobe).
    Rank normalisation means every part of the visual field gets the same number of cells;
    the real map over-represents the frontal/upper field, which this does not model.
    """
    pos = np.asarray(pos, dtype=float)
    u = np.zeros(len(pos))
    v = np.zeros(len(pos))
    for s in ("left", "right"):
        m = side == s
        if not m.any():
            continue
        p = pos[m]
        lat = np.abs(p[:, 1] - midline_y(p[:, 0], axes))
        u[m] = _rank01(p[:, 0])
        v[m] = _rank01(_rank01(p[:, 2]) + _rank01(lat))
    return u, v


def preferred_field(u: np.ndarray, v: np.ndarray):
    """Map lobe coordinates to a preferred (eye-lateral azimuth, elevation) in degrees.

    Anterior tectum <- frontal field (temporal retina); posterior <- rear field (nasal retina).
    Dorsomedial tectum <- upper field (ventral retina); ventrolateral <- lower field (dorsal retina).
    """
    theta = THETA_MIN_DEG + np.asarray(u) * (THETA_MAX_DEG - THETA_MIN_DEG)
    elev = EL_MAX_DEG - np.asarray(v) * (EL_MAX_DEG - EL_MIN_DEG)
    return theta, elev


def _unit(theta_deg, elev_deg) -> np.ndarray:
    t = np.radians(np.asarray(theta_deg, dtype=float))
    e = np.radians(np.asarray(elev_deg, dtype=float))
    return np.stack([np.cos(e) * np.cos(t), np.cos(e) * np.sin(t), np.sin(e)], axis=-1)


class TectumMap:
    """Precomputed retinotopic map for the tectum cells of one cells_identified.json."""

    def __init__(self, cells: dict):
        tect = cells["tectum"]
        self.axes = cells["axes"]
        self.seg_ids = np.array([c["seg_id"] for c in tect], dtype=np.uint64)
        self.pos = np.array([c["pos"] for c in tect], dtype=float)
        self.side = np.array([c["side"] for c in tect])
        self.u, self.v = lobe_coords(self.pos, self.side, self.axes)
        self.theta, self.elev = preferred_field(self.u, self.v)
        self.vec = _unit(self.theta, self.elev)
        self.index = {s: np.nonzero(self.side == s)[0] for s in ("left", "right")}

    def drive(self, stimulus: dict, sigma_deg: float = SIGMA_DEG) -> np.ndarray:
        out = np.zeros(len(self.seg_ids))
        two_s2 = 2.0 * math.radians(sigma_deg) ** 2
        for eye, points in (stimulus or {}).items():
            if eye not in CONTRA:
                raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")
            if not points:
                continue
            p = np.asarray(points, dtype=float).reshape(-1, 3)
            az, el, con = p[:, 0], p[:, 1], np.abs(p[:, 2])
            # eye-lateral azimuth: 0 = straight ahead, positive = toward that eye's own side
            theta = np.where(eye == "right", az, -az)
            theta = (theta + 180.0) % 360.0 - 180.0
            seen = (theta >= THETA_MIN_DEG) & (theta <= THETA_MAX_DEG) & (np.abs(el) <= 90.0)
            if not seen.any():
                continue
            pv = _unit(theta[seen], el[seen])
            idx = self.index[CONTRA[eye]]
            # numpy 2.0 + macOS Accelerate raises spurious FP flags inside matmul; the values match
            # an einsum reference to 1e-13 (checked 2026-09-25), and we assert finiteness below.
            with np.errstate(all="ignore"):
                cosd = np.clip(self.vec[idx] @ pv.T, -1.0, 1.0)
                w = np.exp(-np.arccos(cosd) ** 2 / two_s2)
                out[idx] += w @ con[seen]
        if not np.isfinite(out).all():
            raise FloatingPointError("tectum_drive produced a non-finite value")
        return out


def tectum_map(cells: dict | None = None) -> TectumMap:
    cells = load() if cells is None else cells
    key = id(cells)
    if _CACHE.get(("map", key)) is None:
        _CACHE[("map", key)] = TectumMap(cells)
    return _CACHE[("map", key)]


def tectum_drive(stimulus: dict, cells: dict | None = None, sigma_deg: float = SIGMA_DEG) -> np.ndarray:
    """Per-tectal-cell input for a visual scene.

    stimulus: {"left": [(azimuth_deg, elevation_deg, contrast), ...], "right": [...]}
              (either key may be missing). Points outside an eye's field
              [THETA_MIN_DEG, THETA_MAX_DEG] eye-lateral azimuth are ignored for that eye.
    Returns a float array aligned with cells["tectum"] (same order as tectum_seg_ids()).
    The left eye drives only right-lobe cells and vice versa; every other cell gets 0.
    """
    return tectum_map(cells).drive(stimulus, sigma_deg)


def tectum_seg_ids(cells: dict | None = None) -> np.ndarray:
    return tectum_map(cells).seg_ids


def disk(azimuth_deg: float, elevation_deg: float, radius_deg: float,
         contrast: float = 1.0, step_deg: float = 2.0) -> list:
    """Sample a filled disk (e.g. a looming shadow or a prey dot) into stimulus points.

    Points lie on a grid of spacing step_deg in (azimuth, elevation) about the centre; a disk
    smaller than one step returns just its centre. Each point carries `contrast`, so a bigger
    disk drives more cells (that is how looming grows in this model).
    """
    if radius_deg <= step_deg / 2:
        return [(azimuth_deg, elevation_deg, contrast)]
    n = int(math.ceil(radius_deg / step_deg))
    pts = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            da, de = i * step_deg, j * step_deg
            if da * da + de * de <= radius_deg * radius_deg:
                pts.append((azimuth_deg + da, elevation_deg + de, contrast))
    return pts


def never_stimulate(cells: dict | None = None) -> set:
    """Segment ids that must never receive direct (visual or other sensory) input."""
    cells = load() if cells is None else cells
    return set(int(s) for s in cells["exclusions"]["seg_ids"])


# =============================================================================================
# Build pipeline (network; run once).  python -m fishbrain.cells
# =============================================================================================

SPN_NAMES = {"rov3": "RoV3", "miv1": "MiV1", "miv2": "MiV2", "rom1": "RoM1", "mim1": "MiM1",
             "mir1": "MiR1", "mir2": "MiR2", "rol_r1": "RoL-R1"}

# Unpaired midline structures (mece label layer, id) used to fit the midline.
MIDLINE_REGIONS = [("mece2_231218", k) for k in ("6", "40", "41", "20", "30", "50", "39", "59", "64", "65")] \
    + [("mece1_231218", "22")]

# mece2 labels of reticulospinal / escape-circuit / premotor hindbrain groups. Somas inside get no
# direct sensory input (they are readouts or intermediate stages, not sensory layers).
RETICULOSPINAL_LABELS = {16: "NucMLF", 21: "RoL2", 22: "RoM2", 24: "RoM1", 27: "RoM3", 28: "RoV3",
                         29: "Spiral Fiber Neuron Posterior cluster", 31: "RoL3",
                         32: "Spiral Fiber Neuron Anterior cluster", 34: "MiV2", 38: "MiD2",
                         44: "RoL-R1", 47: "MiV1", 48: "Mauthner", 49: "MiR1", 52: "Mauthner Cell Axon Cap",
                         53: "MiM1", 54: "MiR2", 55: "MiD3", 60: "MiT", 61: "CaD", 62: "CaV"}
TECTUM_LABELS = {17: "SPV", 18: "neuropil"}


def _cv(layer, mip, fill_missing=True):
    from cloudvolume import CloudVolume
    return CloudVolume(PUBLIC + layer, mip=mip, use_https=True, progress=False, fill_missing=fill_missing)


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _segprops(layer):
    d = json.load(open(REF / f"segprops_{layer.replace('/', '_')}.json"))["inline"]
    return dict(zip(map(int, d["ids"]), d["properties"][0]["values"]))


def _atlas():
    """Region centroids/bboxes of mece0/1/2 at the 4096 nm mip (cached in data/ref)."""
    out_path = REF / "atlas_regions_4096.json"
    if out_path.exists():
        return json.load(open(out_path))
    out = {}
    for layer in ("mece0_231218", "mece1_231218", "mece2_231218"):
        npy = REF / f"{layer}_4096.npy"
        if npy.exists():
            a = np.load(npy)
        else:
            a = np.asarray(_cv(layer, [4096, 4096, 3840])[:, :, :])[..., 0]
            np.save(npy, a)
        names = _segprops(layer)
        res = np.array([4096, 4096, 3840]) / np.array(VOXEL_NM)
        for lab in sorted(set(np.unique(a).tolist()) - {0}):
            idx = np.argwhere(a == lab)
            out.setdefault(layer, {})[str(lab)] = dict(
                name=names.get(lab), n=int(len(idx)), centroid_8nm=((idx.mean(0) + 0.5) * res).round().tolist(),
                bbox_lo=(idx.min(0) * res).round().tolist(), bbox_hi=((idx.max(0) + 1) * res).round().tolist())
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def _axes(atlas):
    pts = np.array([atlas[l][k]["centroid_8nm"][:2] for l, k in MIDLINE_REGIONS])
    b, a = np.polyfit(pts[:, 0], pts[:, 1], 1)
    resid = pts[:, 1] - (a + b * pts[:, 0])
    c1 = atlas["mece1_231218"]
    c2 = atlas["mece2_231218"]
    return {
        "voxel_nm": list(VOXEL_NM),
        "x": "anterior -> posterior (increasing x = posterior)",
        "y": "fish's right -> fish's left (increasing y = toward the fish's LEFT)",
        "z": "dorsal -> ventral (increasing z = ventral)",
        "left_side_rule": "left if y > midline.a + midline.b * x, else right",
        "midline": {"a": float(a), "b": float(b), "formula": "y_mid = a + b * x (8 nm voxels)",
                    "fit_regions": [atlas[l][k]["name"] for l, k in MIDLINE_REGIONS],
                    "rms_residual_um": float(np.sqrt((resid ** 2).mean()) * VOXEL_NM[1] / 1000)},
        "evidence": {
            "anterior_posterior_centroid_x": {c1[k]["name"]: c1[k]["centroid_8nm"][0] for k in ("1", "2", "3", "21")}
            | {"Hindbrain": atlas["mece0_231218"]["6"]["centroid_8nm"][0],
               "Spinal Cord": atlas["mece0_231218"]["4"]["centroid_8nm"][0]},
            "dorsal_ventral_centroid_z": {c2[k]["name"]: c2[k]["centroid_8nm"][2] for k in ("6", "3", "45", "40", "68")}
            | {c1[k]["name"]: c1[k]["centroid_8nm"][2] for k in ("22", "23")},
            "left_right": "Fish1 authors' LateralLineVisualization state: segments named "
                          "pre_prop_MONs_left sit at high y, pre_prop_MONs_right at low y (see mon_check)",
        },
    }


def _mon_check():
    """Mesh centroids of the authors' left/right MON segment groups (12 each)."""
    path = REF / "lr_mon_check.json"
    if path.exists():
        return json.load(open(path))
    import warnings
    warnings.filterwarnings("ignore")
    d = json.load(open(REF / "LateralLineVisualization.json"))
    layers = {l["name"]: l for l in d["layers"]}
    cv = _cv("seg_241003_agg241003", 0)
    out = {}
    for name in ("pre_prop_MONs_left", "pre_prop_MONs_right"):
        cents = []
        for s in [int(x) for x in layers[name]["segments"]][:12]:
            m = cv.mesh.get(s, lod=3)
            m = m[s] if isinstance(m, dict) else m
            if len(m.vertices):
                cents.append((m.vertices / np.array(VOXEL_NM)).mean(0))
        c = np.array(cents)
        out[name] = {"n_segments": len(c), "mean_centroid_8nm": c.mean(0).round().tolist(),
                     "y_min": float(c[:, 1].min().round()), "y_max": float(c[:, 1].max().round())}
    json.dump(out, open(path, "w"), indent=1)
    return out


def _somas():
    """CAVE somas table (v709), cached. pt_position is in 8 nm voxels (measured)."""
    import pandas as pd
    path = REF / "cave_somas_full.parquet"
    if not path.exists():
        import caveclient
        for attempt in range(8):
            try:
                c = caveclient.CAVEclient(datastack_name="fish1_full", server_address="https://global.brain-wire-test.org")
                c.materialize.version = c.materialize.most_recent_version()
                df = c.materialize.query_table("somas", limit=250000)
                df["materialization_version"] = c.materialize.version
                df.to_parquet(path)
                break
            except Exception as e:  # never echo the message: it could carry auth details
                _log("CAVE attempt", attempt, type(e).__name__)
                time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError("CAVE somas table unavailable and no cache in data/ref")
    df = pd.read_parquet(path)
    p = np.vstack(df.pt_position.values).astype(float)
    df["x"], df["y"], df["z"] = p[:, 0], p[:, 1], p[:, 2]
    return df


def _agg_map():
    import pandas as pd
    ag = pd.read_csv(REF / "agglomerated_segments_and_soma_ids.csv")
    return ag.set_index("lores_id")["hires_id_agglo"]


def _seg_votes(points, expect=None, workers=8):
    """Read seg_241003_agg241003 (16x16x30 nm) in a 9x9x3 neighbourhood around each 8 nm point.

    Returns per point: (plurality id, its voxel count, count of `expect`, total voxels)."""
    from concurrent.futures import ThreadPoolExecutor
    import warnings
    warnings.filterwarnings("ignore")
    cv = _cv("seg_241003_agg241003", 0)

    def one(i):
        x, y, z = int(points[i][0]) // 2, int(points[i][1]) // 2, int(points[i][2])
        a = np.asarray(cv[x - 4:x + 5, y - 4:y + 5, z - 1:z + 2])[..., 0]
        u, c = np.unique(a, return_counts=True)
        k = int(np.argmax(c))
        e = int(c[u == expect[i]].sum()) if expect is not None else -1
        return int(u[k]), int(c[k]), e, int(a.size)

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, range(len(points))))


def _label_votes(M, b0, f, pts):
    """Plurality mece label over a 3x3x3 neighbourhood of each point in a downloaded label box."""
    q = (np.asarray(pts, dtype=float) // f - b0).astype(int)
    hi = np.array(M.shape) - 1
    votes = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                qq = np.clip(q + np.array([dx, dy, dz]), 0, hi)
                votes.append(M[qq[:, 0], qq[:, 1], qq[:, 2]])
    V = np.stack(votes, 1)
    lab = np.empty(len(V), dtype=int)
    for i, row in enumerate(V):
        u, c = np.unique(row, return_counts=True)
        lab[i] = u[np.argmax(c)]
    return lab, V


def _mece2_box(name, lo8, hi8):
    """mece2 labels at the 1024 nm mip over an 8 nm-voxel box (cached as uint8)."""
    path = REF / f"mece2_1024_{name}.npz"
    f = np.array([128, 128, 32])
    if path.exists():
        d = np.load(path)
        return d["M"], d["b0"], f
    b0 = np.array(lo8) // f
    b1 = np.array(hi8) // f + 1
    M = np.asarray(_cv("mece2_231218", [1024, 1024, 960])[b0[0]:b1[0], b0[1]:b1[1], b0[2]:b1[2]])[..., 0]
    M = M.astype(np.uint8)
    np.savez_compressed(path, M=M, b0=b0)
    return M, b0, f


def _synapse_counts(seg):
    """Input / output synapse counts and output positions for one agg segment (brain of record)."""
    from fishbrain import precomputed as pc
    info = _CACHE.setdefault("syninfo", pc.info())
    rel = {r["id"]: r for r in info["relationships"]}
    res = {}
    for key, which in (("inputs", "post_synaptic_cell"), ("outputs", "pre_synaptic_cell")):
        raw = pc.get_sharded(rel[which]["key"], rel[which]["sharding"], int(seg))
        recs = pc.decode_multiple(raw)[0] if raw is not None else None
        res[key] = 0 if recs is None else int(len(recs))
        res[key + "_type_counts"] = {} if recs is None else {
            str(int(t)): int(n) for t, n in zip(*np.unique(recs["type"], return_counts=True))}
        if key == "outputs":
            res["_out_pos"] = None if recs is None else recs["p0"].astype(float)
    return res


def build_spn(somas, agg, axes, M, b0, f, names2):
    import pandas as pd
    hmi = json.load(open(REF / "hindbrain_motion_integrator.json"))
    hl = {l["name"]: l for l in hmi["layers"]}
    authors = {"forward": set(map(int, hl["spn_forward"]["segments"])),
               "turning": set(map(int, hl["spn_turning"]["segments"]))}
    df = pd.read_excel(REF / "em_zfish1_dataframe.xlsx")
    spn = df[df["classifier"].astype(str).str.contains("spn")].copy()
    rows = []
    s = somas.set_index("id")
    pts, exp = [], []
    for _, r in spn.iterrows():
        lore = int(r["Cell ID"])
        pts.append(s.loc[lore, ["x", "y", "z"]].values.astype(float))
        exp.append(int(agg.get(lore, 0)))
    votes = _seg_votes(pts, exp)
    labs, _ = _label_votes(M, b0, f, pts)
    for (_, r), p, e, v, lab in zip(spn.iterrows(), pts, exp, votes, labs):
        cl = str(r["classifier"])
        key = cl.replace("spn_turning_", "").replace("spn_forward_", "").replace("_prox", "")
        klass = "turning" if "turning" in cl else "forward"
        lore = int(r["Cell ID"])
        rows.append({
            "lore_id": lore, "seg_id": e, "side": str(side_of(p, axes)[0]), "class": klass,
            "name": SPN_NAMES[key], "classifier": cl, "prox": cl.endswith("_prox"),
            "in_authors_state": lore in authors[klass],
            "pos": [int(round(t)) for t in p],
            "atlas_region": names2.get(int(lab), "unlabelled") if lab else "unlabelled",
            "seg_check": {"expected_voxels": v[2], "of": v[3], "plurality_id": v[0]},
            "neurotransmitter": None if pd.isna(r["final_neurotransmitter_ID"]) else str(r["final_neurotransmitter_ID"]),
        })
    return rows


def build_spn_mirrors(spn, agg, axes):
    """Other-side counterparts for the named SPNs, which the release gives almost only on the right.

    For each named SPN we reflect its soma across the fitted midline and look for somas on the
    other side whose centroid lies within MIRROR_RADIUS_UM of that point, reading lores_cbs_231218
    at 512x512x60 nm in a +-16 um box (so candidate somas are whole). Score =
    distance_um / 5 + |ln(volume / named cell's volume)|; lowest score wins, one partner per cell
    (greedy, best scores first). These are CANDIDATES, not identifications.
    """
    from concurrent.futures import ThreadPoolExecutor
    import warnings
    warnings.filterwarnings("ignore")
    f = np.array([64, 64, 2])
    half = np.array([16000 / 512, 16000 / 512, 16000 / 60]).astype(int)
    um = np.array([0.008, 0.008, 0.030])
    pos = np.array([s["pos"] for s in spn], dtype=float)
    mir = pos.copy()
    mir[:, 1] = 2 * midline_y(pos[:, 0], axes) - pos[:, 1]
    cache = REF / "spn_mirror_boxes.npz"
    stored = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    cv = _cv("lores_cbs_231218", [512, 512, 60])

    def somas_in_box(p8):
        c = (np.asarray(p8) / f).astype(int)
        b0 = c - half
        L = np.asarray(cv[b0[0]:c[0] + half[0] + 1, b0[1]:c[1] + half[1] + 1, b0[2]:c[2] + half[2] + 1])[..., 0]
        ids, inv = np.unique(L.ravel(), return_inverse=True)
        inv = inv.ravel()
        cnt = np.bincount(inv)
        idx = np.indices(L.shape).reshape(3, -1)
        cen = np.stack([np.bincount(inv, weights=idx[k]) / cnt for k in range(3)], 1)
        return np.column_stack([ids.astype(float), cnt * 0.512 * 0.512 * 0.060, (cen + b0 + 0.5) * f])

    def job(i):
        key = str(spn[i]["lore_id"])
        if key + "_own" in stored:
            return stored[key + "_own"], stored[key + "_mir"]
        return somas_in_box(pos[i]), somas_in_box(mir[i])

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(job, range(len(spn))))
    np.savez_compressed(cache, **{f"{s['lore_id']}_own": r[0] for s, r in zip(spn, res)},
                        **{f"{s['lore_id']}_mir": r[1] for s, r in zip(spn, res)})
    options, own_vol = [], {}
    for i, (own_t, mir_t) in enumerate(res):
        me = own_t[own_t[:, 0] == spn[i]["lore_id"]]
        own = float(me[0, 1]) if len(me) else None
        own_vol[i] = own
        other = "left" if spn[i]["side"] == "right" else "right"
        cand = mir_t[mir_t[:, 0] > 0]
        d = np.sqrt((((cand[:, 2:5] - mir[i]) * um) ** 2).sum(1))
        sd = side_of(cand[:, 2:5], axes)
        for j in np.nonzero((d <= MIRROR_RADIUS_UM) & (sd == other))[0]:
            score = d[j] / 5.0 + (abs(math.log(cand[j, 1] / own)) if own else 0.0)
            options.append((score, i, int(cand[j, 0]), float(d[j]), float(cand[j, 1]), cand[j, 2:5].tolist()))
    options.sort(key=lambda t: t[0])
    taken_cell, taken_partner, best = set(), set(), {}
    for score, i, lore, dist, v, p in options:
        if i in taken_cell or lore in taken_partner:
            continue
        taken_cell.add(i)
        taken_partner.add(lore)
        best[i] = (score, lore, dist, v, p)
    rows = []
    for i, s in enumerate(spn):
        r = {"of_lore_id": s["lore_id"], "name": s["name"], "class": s["class"],
             "side": "left" if s["side"] == "right" else "right",
             "named_cell_volume_um3": None if own_vol[i] is None else round(own_vol[i], 1),
             "n_somas_within_radius": sum(1 for o in options if o[1] == i)}
        if i in best:
            score, lore, dist, v, p = best[i]
            r.update({"lore_id": lore, "seg_id": int(agg.get(lore, 0)), "distance_to_mirror_point_um": round(dist, 1),
                      "volume_um3": round(v, 1), "score": round(score, 2), "pos": [int(round(t)) for t in p]})
        else:
            r.update({"lore_id": None, "seg_id": None})
        rows.append(r)
    return rows


MIRROR_RADIUS_UM = 10.0


def build_mauthner(agg, axes, somas):
    """Rank somas by volume in and around the atlas 'Mauthner' region (mece2 48), per side."""
    import pandas as pd
    from scipy import ndimage
    import warnings
    warnings.filterwarnings("ignore")
    atlas = _atlas()["mece2_231218"]
    r48 = atlas["48"]
    mip = [512, 512, 60]
    f = np.array([64, 64, 2])
    margin = np.array([20000 / 8, 20000 / 8, 20000 / 30])  # 20 um
    b0 = ((np.array(r48["bbox_lo"]) - margin) / f).astype(int)
    b1 = ((np.array(r48["bbox_hi"]) + margin) / f).astype(int)
    path = REF / "r4_box_512x512x60.npz"
    if path.exists():
        d = np.load(path)
        L, Mm, b0 = d["L"], d["M"], d["b0"]
    else:
        L = np.asarray(_cv("lores_cbs_231218", mip)[b0[0]:b1[0], b0[1]:b1[1], b0[2]:b1[2]])[..., 0]
        Mm = np.asarray(_cv("mece2_231218", mip)[b0[0]:b1[0], b0[1]:b1[1], b0[2]:b1[2]])[..., 0].astype(np.uint8)
        np.savez_compressed(path, L=L, M=Mm, b0=b0)
    ids, inv = np.unique(L.ravel(), return_inverse=True)
    inv = inv.ravel()
    n = len(ids)
    cnt = np.bincount(inv, minlength=n)
    idx = np.indices(L.shape)
    cen = np.stack([np.bincount(inv, weights=idx[a].ravel(), minlength=n) / cnt for a in range(3)], 1)
    lab_inv = inv.reshape(L.shape)
    mn = np.stack([ndimage.minimum(idx[a], lab_inv, range(n)) for a in range(3)], 1)
    mx = np.stack([ndimage.maximum(idx[a], lab_inv, range(n)) for a in range(3)], 1)
    in48 = np.bincount(inv, weights=(Mm == 48).ravel(), minlength=n) / cnt
    vox_um3 = 0.512 * 0.512 * 0.060
    c8 = (cen + b0 + 0.5) * f
    df = pd.DataFrame({"lore": ids.astype(np.int64), "vox": cnt, "vol_um3": cnt * vox_um3,
                       "x": c8[:, 0], "y": c8[:, 1], "z": c8[:, 2],
                       "clipped": (mn == 0).any(1) | (mx == np.array(L.shape) - 1).any(1),
                       "frac_atlas_mauthner": in48})
    df = df[(df.lore > 0) & ~df.clipped].sort_values("vox", ascending=False)
    df["side"] = side_of(df[["x", "y", "z"]].values, axes)
    p99 = float(np.percentile(df.vol_um3, 99))
    med = float(np.median(df.vol_um3))
    out = {"method": {
        "box_8nm": [(b0 * f).tolist(), (np.array(L.shape) * f + b0 * f).tolist()],
        "soma_layer": "lores_cbs_231218 at 512x512x60 nm", "n_somas_unclipped": int(len(df)),
        "median_soma_um3": med, "p99_soma_um3": p99}}
    cvm = _cv("seg_241003_agg241003", 0)
    so = somas.set_index("id")
    for side in ("left", "right"):
        d = df[df.side == side].head(5)
        cands = []
        for _, r in d.iterrows():
            seg = int(agg.get(int(r.lore), 0))
            sc = _synapse_counts(seg) if seg else {"inputs": 0, "outputs": 0, "_out_pos": None}
            cp = so.loc[int(r.lore), ["x", "y", "z"]].values.astype(float) if int(r.lore) in so.index else None
            cands.append({"lore_id": int(r.lore), "seg_id": seg, "soma_volume_um3": round(float(r.vol_um3), 1),
                          "pos": [int(round(t)) for t in (cp if cp is not None else (r.x, r.y, r.z))],
                          "frac_in_atlas_mauthner_region": round(float(r.frac_atlas_mauthner), 2),
                          "input_synapses": sc["inputs"], "output_synapses": sc["outputs"],
                          "input_type_counts": sc.get("inputs_type_counts", {})})
        best = cands[0]
        # anatomy from the agglomerated mesh (lowest detail level)
        m = cvm.mesh.get(best["seg_id"], lod=3)
        m = m[best["seg_id"]] if isinstance(m, dict) else m
        v = m.vertices / np.array(VOXEL_NM)
        ymid = midline_y(v[:, 0], axes)
        own = 1 if side == "left" else -1
        contra = np.sign(v[:, 1] - ymid) != own
        sp = np.array(best["pos"], dtype=float)
        seg_check = _seg_votes([sp], [best["seg_id"]])[0]
        mesh = {"n_vertices_lod3": int(len(v)),
                "x_range": [float(v[:, 0].min()), float(v[:, 0].max())],
                "y_range": [float(v[:, 1].min()), float(v[:, 1].max())],
                "z_range": [float(v[:, 2].min()), float(v[:, 2].max())],
                "lateral_reach_um": float(np.abs(v[:, 1] - sp[1]).max() * 8 / 1000),
                "ventral_reach_um": float((v[:, 2].max() - sp[2]) * 30 / 1000),
                "fraction_vertices_past_midline": float(contra.mean()),
                "max_past_midline_um": float((np.abs(v[contra, 1] - ymid[contra]).max() * 8 / 1000) if contra.any() else 0.0),
                "posterior_reach_um": float((v[:, 0].max() - sp[0]) * 8 / 1000)}
        runner = cands[1]
        out[side] = {
            "seg_id": best["seg_id"], "lore_id": best["lore_id"], "pos": best["pos"],
            "soma_volume_um3": best["soma_volume_um3"],
            "candidates": cands,
            "evidence": {
                "largest_soma_on_side_in_r4_box": True,
                "volume_ratio_to_runner_up": round(best["soma_volume_um3"] / runner["soma_volume_um3"], 2),
                "volume_ratio_to_p99": round(best["soma_volume_um3"] / p99, 2),
                "frac_in_atlas_mauthner_region": best["frac_in_atlas_mauthner_region"],
                "input_synapses": best["input_synapses"],
                "input_ratio_to_most_connected_other_candidate": round(
                    best["input_synapses"] / max(1, max(c["input_synapses"] for c in cands[1:])), 2),
                "output_synapses": best["output_synapses"],
                "seg_check_at_soma": {"expected_voxels": seg_check[2], "of": seg_check[3]},
                "mesh": mesh,
            },
        }
    return out


def build_tectum(somas, agg, axes):
    atlas = _atlas()["mece2_231218"]
    lo = np.minimum(atlas["17"]["bbox_lo"], atlas["18"]["bbox_lo"]) - np.array([1500, 1500, 60])
    hi = np.maximum(atlas["17"]["bbox_hi"], atlas["18"]["bbox_hi"]) + np.array([1500, 1500, 60])
    lo = np.maximum(lo, 0)
    M, b0, f = _mece2_box("tectum", lo, hi)
    sel = somas[(somas.x >= b0[0] * f[0]) & (somas.x < (b0[0] + M.shape[0]) * f[0])].copy()
    lab, _ = _label_votes(M, b0, f, sel[["x", "y", "z"]].values)
    sel["label"] = lab
    T = sel[sel.label.isin(list(TECTUM_LABELS))].copy()
    T["side"] = side_of(T[["x", "y", "z"]].values, axes)
    T["lat_um"] = np.abs(T.y - midline_y(T.x, axes)) * 8 / 1000
    T["seg_id"] = T.id.map(lambda i: int(agg.get(int(i), 0)))
    stats = {"somas_in_tectum_labels": int(len(T)),
             "by_label": {TECTUM_LABELS[k]: int((T.label == k).sum()) for k in TECTUM_LABELS},
             "by_side": {s: int((T.side == s).sum()) for s in ("left", "right")},
             "cell_type_counts": {str(k): int(v) for k, v in T.cell_type.value_counts().items()},
             "no_agg_segment": int((T.seg_id == 0).sum()),
             "within_3um_of_midline": int((T.lat_um < 3).sum())}
    all_counts = agg[agg > 0].value_counts()
    T = T[T.seg_id > 0]
    shared = T.seg_id.map(all_counts) > 1
    stats["seg_shared_with_another_soma"] = int(shared.sum())
    # a segment holding several somas is a merge error; keep it only if every soma it holds is a
    # tectal soma on the same side (then it is one node inside one lobe), else drop it
    tect_by_seg = T.groupby("seg_id")
    allsoma_seg = agg[agg > 0]
    keep, dropped_mixed = [], 0
    for seg, g in tect_by_seg:
        n_all = int(all_counts.get(seg, 0))
        if n_all > len(g) or g.side.nunique() > 1:
            dropped_mixed += len(g)
            continue
        keep.append((seg, g))
    stats["dropped_somas_in_segments_merged_outside_tectum_or_across_sides"] = int(dropped_mixed)
    stats["dropped_somas_within_3um_of_midline"] = 0
    out = []
    for seg, g in keep:
        if (g.lat_um < 3).all():
            stats["dropped_somas_within_3um_of_midline"] += len(g)
            continue
        p = g[["x", "y", "z"]].values.mean(0)
        out.append({"seg_id": int(seg), "side": str(g.side.iloc[0]), "pos": [int(round(t)) for t in p],
                    "lore_ids": [int(i) for i in g.id], "layer": TECTUM_LABELS[int(g.label.mode().iloc[0])],
                    "cell_type": str(g.cell_type.iloc[0]) if len(g) == 1 else "merged"})
    out.sort(key=lambda c: (c["side"], c["pos"][0], c["pos"][1], c["pos"][2]))
    stats["tectum_cells_kept"] = len(out)
    stats["kept_by_side"] = {s: sum(1 for c in out if c["side"] == s) for s in ("left", "right")}
    stats["kept_segments_with_multiple_somas"] = sum(1 for c in out if len(c["lore_ids"]) > 1)
    # verify a seeded random sample of kept cells against the agglomeration at the soma
    rng = np.random.default_rng(0)
    pick = rng.choice(len(out), size=min(60, len(out)), replace=False)
    pos = somas.set_index("id")
    pts = [pos.loc[out[i]["lore_ids"][0], ["x", "y", "z"]].values.astype(float) for i in pick]
    votes = _seg_votes(pts, [out[i]["seg_id"] for i in pick])
    stats["seg_check_sample"] = {"n": len(pick),
                                 "plurality_equals_expected": int(sum(v[0] == out[i]["seg_id"] for v, i in zip(votes, pick))),
                                 "median_expected_fraction": float(np.median([v[2] / v[3] for v in votes]))}
    return out, stats


def _hindbrain_labels():
    """mece2 labels (1024 nm mip) over the box holding every reticulospinal atlas region."""
    atlas = _atlas()["mece2_231218"]
    keys = [str(k) for k in RETICULOSPINAL_LABELS if str(k) in atlas]
    lo = np.maximum(np.min([atlas[k]["bbox_lo"] for k in keys], 0) - np.array([1500, 1500, 60]), 0)
    hi = np.max([atlas[k]["bbox_hi"] for k in keys], 0) + np.array([1500, 1500, 60])
    return _mece2_box("hindbrain", lo, hi)


def build_exclusions(somas, agg, mauthner, spn, M, b0, f, mirrors=(), axes=None):
    sel = somas[(somas.x >= b0[0] * f[0]) & (somas.x < (b0[0] + M.shape[0]) * f[0])
                & (somas.y >= b0[1] * f[1]) & (somas.y < (b0[1] + M.shape[1]) * f[1])].copy()
    lab, _ = _label_votes(M, b0, f, sel[["x", "y", "z"]].values)
    sel["label"] = lab
    rs = sel[sel.label.isin(list(RETICULOSPINAL_LABELS))].copy()
    rs["seg_id"] = rs.id.map(lambda i: int(agg.get(int(i), 0)))
    by_region = {RETICULOSPINAL_LABELS[int(k)]: int(v) for k, v in rs.label.value_counts().items()}
    m_ids = [mauthner["left"]["seg_id"], mauthner["right"]["seg_id"]]
    spn_ids = sorted({c["seg_id"] for c in spn if c["seg_id"]})
    mirror_ids = sorted({int(m["seg_id"]) for m in mirrors if m.get("seg_id")})
    seg_ids = set(int(s) for s in rs.seg_id if s) | set(m_ids) | set(spn_ids) | set(mirror_ids)
    return {
        "rule": "Visual input enters the model only through tectum_drive() into the cells in 'tectum'. "
                "No segment in seg_ids may receive any direct sensory or injected current; they are "
                "read out (Mauthner, SPNs) or sit between the senses and the readout.",
        "mauthner_seg_ids": m_ids,
        "spn_seg_ids": spn_ids,
        "spn_mirror_candidate_seg_ids": mirror_ids,
        "reticulospinal_atlas_somas_by_region": by_region,
        "reticulospinal_atlas_somas_by_side": {s: int((side_of(rs[["x", "y", "z"]].values, axes) == s).sum())
                                               for s in ("left", "right")},
        "reticulospinal_atlas_somas_without_agg_segment": int((rs.seg_id == 0).sum()),
        "reticulospinal_atlas_labels": {str(k): v for k, v in RETICULOSPINAL_LABELS.items()},
        "seg_ids": sorted(seg_ids),
        "also_never": "Do not inject visual current into any non-tectal cell (retina, pretectum, "
                      "hindbrain). If a later model drives retinal afferents instead of tectal somata, "
                      "that replaces tectum_drive, it is not added on top.",
    }


SCHEMA = {
    "axes": "orientation of the 8x8x30 nm voxel grid, the fitted midline and the evidence",
    "tectum": "[{seg_id, side, pos, lore_ids, layer, cell_type}] one entry per agglomeration segment; "
              "pos = soma position (8 nm voxels, mean if a segment holds several tectal somas); "
              "side = fish's own side; layer = SPV (periventricular) or neuropil",
    "mauthner": "{left|right: {seg_id, lore_id, pos, soma_volume_um3, candidates[], evidence{}}, method}",
    "spn": "[{lore_id, seg_id, side, class (turning|forward), name, classifier, prox, in_authors_state, "
           "pos, atlas_region, seg_check, neurotransmitter}]",
    "spn_mirror_candidates": "[{of_lore_id, name, class, side, lore_id, seg_id, distance_to_mirror_point_um, "
                             "volume_um3, named_cell_volume_um3, score, pos}] UNCONFIRMED other-side partners",
    "exclusions": "{rule, mauthner_seg_ids, spn_seg_ids, spn_mirror_candidate_seg_ids, seg_ids (all never-stimulate), ...}",
    "stats": "counts measured while building",
    "sources": "every input with its URL or path",
}


def build(out_path: os.PathLike = CELLS_JSON) -> dict:
    _log("atlas + axes")
    atlas = _atlas()
    axes = _axes(atlas)
    axes["mon_check"] = _mon_check()
    somas = _somas()
    agg = _agg_map()
    _log("mauthner")
    mauthner = build_mauthner(agg, axes, somas)
    _log("hindbrain labels")
    M, b0, f = _hindbrain_labels()
    _log("spn")
    spn = build_spn(somas, agg, axes, M, b0, f, _segprops("mece2_231218"))
    _log("spn mirror candidates")
    mirrors = build_spn_mirrors(spn, agg, axes)
    _log("exclusions")
    exclusions = build_exclusions(somas, agg, mauthner, spn, M, b0, f, mirrors, axes)
    _log("tectum")
    tectum, tstats = build_tectum(somas, agg, axes)
    tect_ids = {c["seg_id"] for c in tectum}
    overlap = sorted(tect_ids & set(exclusions["seg_ids"]))
    if overlap:  # never let a readout or reticulospinal cell into the visual input set
        tectum = [c for c in tectum if c["seg_id"] not in set(exclusions["seg_ids"])]
    tstats["removed_because_excluded"] = len(overlap)
    cells = {
        "schema": SCHEMA,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "brain_of_record": "gs://fish1-public/syn_241003_agg241003_reorient_axde_ei_bayes_idx_pre_250410.precomputed",
        "segmentation": "gs://fish1-public/seg_241003_agg241003",
        "axes": axes,
        "tectum": tectum,
        "mauthner": mauthner,
        "spn": spn,
        "spn_mirror_candidates": mirrors,
        "exclusions": exclusions,
        "retinotopy": {"theta_min_deg": THETA_MIN_DEG, "theta_max_deg": THETA_MAX_DEG,
                       "el_min_deg": EL_MIN_DEG, "el_max_deg": EL_MAX_DEG, "sigma_deg": SIGMA_DEG,
                       "rule": "each eye -> contralateral lobe; frontal field -> anterior (low x); rear field -> "
                               "posterior; upper field -> dorsomedial; lower field -> ventrolateral; "
                               "position within a lobe by rank (see fishbrain/cells.py lobe_coords)"},
        "stats": {"tectum": tstats, "cave_somas_rows": int(len(somas)),
                  "cave_materialization_version": int(somas.materialization_version.iloc[0])},
        "sources": {
            "atlas": "precomputed://gs://fish1-public/mece0_231218 .. mece2_231218 (segment_properties labels)",
            "somas": "CAVE fish1_full table somas, materialization v709, cached data/ref/cave_somas_full.parquet",
            "soma_segmentation": "precomputed://gs://fish1-public/lores_cbs_231218 (labels = lore ids)",
            "lore_to_agg": "gs://fish1-release/paper_data/TEN_analysis.zip: agglomerated_segments_and_soma_ids.csv",
            "spn_names": "gs://fish1-release/paper_data/HMI_analysis.zip: data/em_zfish1_dataframe.xlsx",
            "authors_states": "gs://fish1-release/assets/neuroglancer_states/202504/"
                              "{hindbrain_motion_integrator,LateralLineVisualization}.json",
        },
    }
    with open(out_path, "w") as fh:
        json.dump(cells, fh, indent=1)
    _log("wrote", out_path, "tectum", len(tectum), "spn", len(spn))
    return cells


if __name__ == "__main__":
    build()
