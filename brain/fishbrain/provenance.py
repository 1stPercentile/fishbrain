"""FISHBRAIN G1c provenance: the pre-registered decision-path share, as bookkeeping over a run.

The metric (brief.md, "G1c: fill the gaps", pre-registered 2026-09-26 03:39 EDT)
---------------------------------------------------------------------------------
"In each gate trial's decision window, every neuron carries a provenance value m in [0, 1]: the
share of the synaptic drive it received that travelled through measured Fish1 synapses from
measured-driven sources. A measured synapse passes on its source's m; a bridged link passes on 0.
The eye model's input is labelled *chosen (eye model)* and scored separately, not counted as
measured wiring. The decision-path share is the drive-weighted mean m over the readout cells that
decide (the escaping M-system cell, and the strike and turn populations). It is reported per
readout and per bridge class." Secondary, never the headline: the synapse-count share.

This file is that sentence made exact. Nothing here touches the simulator: it reads a finished
run's raster, the network, the inputs' own random streams and a per-pair provenance table, and
replays sim.py's arithmetic on provenance channels. Spikes, rasters and every hash are unchanged.

Operational definition (each CHOICE is ours, fixed here before any bridged run)
--------------------------------------------------------------------------------
1. Paths, not a scalar. Each unit of drive carries a *path mask*: the set of bridge classes it
   has crossed on its way from the model's input. Mask 0 = every link so far was a measured Fish1
   synapse. A measured synapse keeps the mask; a bridged link of class y sets bit y. m is the
   share of drive with mask 0. Because masks are sets, "through class y" (any mask containing y)
   and "only class y" (mask == {y}) both come out of one bookkeeping; no first- or last-crossed
   rule is needed. With one bridge class and no eye input this reduces to the brief's recursion
   exactly: m_post = sum(w * m_pre over measured) / sum(w over all links).
2. Roots. The eye model is where every trial's activity starts (brain.run_trial's only input is
   the Poisson source "retina"). A neuron whose depolarization comes only from the eye model is
   labelled "chosen (eye model)": it has no synaptic drive, so its own m is undefined and is
   reported as that label, never as a number. What its spikes pass on counts as mask 0: the eye
   model is the root, and the measured share is counted from the first synapse after it.
   (Otherwise every readout would read 0 by construction.) Any other model input must be named:
   "chosen:<what>" is another root; "bridge:<class>" is an input that stands in for a missing
   link and carries that class's bit. An unlabelled source other than "retina" raises.
   A root is a pre-registered TARGET SET, not a name: trial_traced lets the eye model drive only
   the brain of record's stimulated set (brain.stimulated_set, rebuilt by eye_segments_of_record),
   and a root source that touches any other neuron raises. So a bridge cannot be hidden by adding
   its fragments to the "retina" targets; it has to be a "bridge:" input or a bridge edge. Each
   root source's target set is hashed into the trace's digests, beside the edge table's.
3. What a spike carries (CHOICE). The composition of the neuron's depolarizing membrane state at
   the threshold crossing, i.e. u after step 1 of the spike's step, split by channel. u is linear
   in its inputs between resets, so the split is exact: the excitatory synaptic parts (by mask),
   plus the direct input parts (roots -> mask 0, bridge inputs -> their bit). Inhibitory parts
   are negative and are not drive; in this LIF a spike is caused by depolarization only (no
   rebound). Drive delivered after the crossing does not count for that spike, and the reset
   wipes the membrane (u = v_reset, g = 0) exactly as the simulator does, so a later spike carries
   only what arrived after the last reset. That is the time order.
4. What a readout received (CHOICE). The drive delivered to the cell inside the decision window,
   summed with its carried masks: each arrival adds |w| times its source spike's composition.
   It is kernel-free and hand-computable. Units: mV*ms, the area of the membrane's linear response
   to that arrival at the threshold checks (1 / ((1 - a)(1 - b)) * dt per unit of h for a synapse;
   a / (1 - a) * dt per mV for a Poisson kick; 1 / (1 - a) * dt per mV for a current step). The
   units only matter where synaptic and direct drive meet at one cell; m itself is a ratio of
   synaptic drives. The at-spike composition of the escaping cell's first spike is reported beside
   it.
5. m of a cell or a population = measured-path excitatory link drive / all excitatory link drive
   (measured synapses + bridged links, including "bridge:" inputs). A population's value is the
   drive-weighted mean m of its cells (pooled sums). Direct eye drive is reported separately as
   the share of (link + direct) drive. Inhibitory drive keeps its own masks and is reported
   beside it; "m_abs" = (|measured exc| + |measured inh|) / (|exc| + |inh|) is a sensitivity row.
6. The synapse-count share = sum |measured part| / sum |all parts| over the simulated network's
   pairs, per class too. Parts are signed synapse counts as the simulator sees them (a pair's
   type-1 and type-2 synapses already net out in the Network), so the raw synapse total of the
   brain of record (e.g. the bundle's stats["kept_synapses"]) is larger and is quoted beside it.

Edge provenance
---------------
EdgeProvenance lists, per (pre, post) pair, the signed count of each bridge class on it. Whatever
is left of the pair's count is measured. Pairs are keyed by internal index, not CSR position, so the
table survives Network.lesion() (a pair with a lesioned endpoint is skipped). A bridge edge that
lands on an existing measured pair is legal: the Network sums them, and the parts keep them apart.
EdgeProvenance.digest() pins the table (Network.digest() does not see it).

Checks run on every trace (check=True)
--------------------------------------
- The bookkeeping mirrors the run: at every recorded spike the reconstructed u (sum of all
  channels) is above threshold, and no other free neuron is above it. Mismatches are counted.
- Parts sum to the total: the signed drive delivered in each window, summed over every channel,
  equals the same total computed straight from the raster and the Network's counts.
- With the simulator after the run: reconstructed u and h equal sim.u and sim.h, spike, synaptic
  event and Poisson event counts equal the simulator's stats.
"""
from __future__ import annotations

import copy
import hashlib
import json
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from fishbrain import sim as S

PROVENANCE_VERSION = "fishbrain-provenance-v1"
BRIDGE_CLASSES = ("a", "b", "c")   # brief: (a) observed-origin fragments, (b) tract topography, (c) literature relays + crossing
EYE = "eye"
EYE_LABEL = "chosen (eye model)"
MEASURED = "measured"
MAX_CLASSES = 8
MIRROR_TOL_MV = 1e-3               # float32 simulator vs float64 bookkeeping, absolute part
MIRROR_RTOL = 1e-5                 # relative part (u can hold many stacked kicks while refractory)


# =============================================================================================
# 1. edge provenance
# =============================================================================================

def _check_classes(classes) -> Tuple[str, ...]:
    cl = tuple(str(c) for c in classes)
    if len(cl) > MAX_CLASSES:
        raise ValueError(f"at most {MAX_CLASSES} bridge classes")
    if len(set(cl)) != len(cl) or any((not c) or "+" in c or c == MEASURED for c in cl):
        raise ValueError(f"bridge class names must be unique, non-empty, without '+', not {MEASURED!r}: {cl}")
    return cl


def mask_name(mask: int, classes: Sequence[str]) -> str:
    if mask == 0:
        return MEASURED
    return "+".join(c for i, c in enumerate(classes) if (mask >> i) & 1)


class EdgeProvenance:
    """Signed bridge counts per (pre, post) pair and class; the rest of every pair is measured.

    pre, post: internal neuron indices (the simulator's), one row per bridge record.
    counts:    signed synapse counts of those records (the same units as Network counts).
    klass:     the bridge class of each record (a name in `classes`).
    Records with the same (pre, post, class) are summed; all-zero pairs are dropped.
    """

    def __init__(self, classes: Sequence[str] = BRIDGE_CLASSES, pre=(), post=(), counts=(), klass=(),
                 meta: Optional[dict] = None):
        self.classes = _check_classes(classes)
        pre = np.asarray(pre, np.int64).ravel()
        post = np.asarray(post, np.int64).ravel()
        counts = np.asarray(counts, np.int64).ravel()
        klass = np.asarray(klass).ravel()
        if not (len(pre) == len(post) == len(counts) == len(klass)):
            raise ValueError("pre, post, counts and klass must have the same length")
        if len(pre) and (pre.min() < 0 or post.min() < 0):
            raise ValueError("negative neuron index")
        cidx = np.array([self.classes.index(str(k)) if str(k) in self.classes else -1 for k in klass.tolist()],
                        np.int64)
        if len(cidx) and cidx.min() < 0:
            bad = sorted({str(k) for k, c in zip(klass.tolist(), cidx.tolist()) if c < 0})
            raise ValueError(f"unknown bridge class {bad}; classes are {self.classes}")
        K = len(self.classes)
        if len(pre):
            pairs, inv = np.unique(np.stack([pre, post], 1), axis=0, return_inverse=True)
            inv = np.asarray(inv).ravel()
            parts = np.zeros((len(pairs), K), np.int64)
            np.add.at(parts, (inv, cidx), counts)
            keep = np.any(parts != 0, axis=1)
            self.pre, self.post, self.parts = pairs[keep, 0], pairs[keep, 1], parts[keep]
        else:
            self.pre = np.zeros(0, np.int64)
            self.post = np.zeros(0, np.int64)
            self.parts = np.zeros((0, K), np.int64)
        self.meta = dict(meta or {})

    @classmethod
    def measured_only(cls, classes: Sequence[str] = BRIDGE_CLASSES) -> "EdgeProvenance":
        return cls(classes)

    @classmethod
    def from_ids(cls, net: S.Network, pre_ids, post_ids, counts, klass, classes: Sequence[str] = BRIDGE_CLASSES,
                 meta: Optional[dict] = None) -> "EdgeProvenance":
        """Records given by external (segment) ids of `net`."""
        return cls(classes, net.index_of(pre_ids), net.index_of(post_ids), counts, klass, meta=meta)

    def digest(self) -> str:
        h = hashlib.sha256(PROVENANCE_VERSION.encode() + b"|edges")
        h.update(json.dumps(list(self.classes)).encode())
        h.update(struct.pack("<QQ", len(self.pre), len(self.classes)))
        for arr in (self.pre, self.post, self.parts):
            h.update(np.ascontiguousarray(arr, "<i8").tobytes())
        return h.hexdigest()

    def align(self, net: S.Network) -> "AlignedParts":
        """Per CSR entry of `net`: the measured signed count and the bridge parts."""
        n, nnz = net.n, net.nnz
        if len(self.pre) and (self.pre.max() >= n or self.post.max() >= n):
            raise ValueError("a bridge record's neuron index is outside the network")
        key_net = net.rows() * n + net.indices.astype(np.int64)
        if nnz > 1 and np.any(np.diff(key_net) <= 0):
            raise ValueError("network CSR is not sorted by (row, column)")
        key_b = self.pre * n + self.post
        pos = np.searchsorted(key_net, key_b)
        pos_c = np.minimum(pos, max(nnz - 1, 0))
        found = (pos < nnz) & (key_net[pos_c] == key_b) if nnz else np.zeros(len(key_b), bool)
        dead = net.dead[self.pre] | net.dead[self.post] if len(key_b) else np.zeros(0, bool)
        missing = ~found & ~dead
        if missing.any():
            k = np.flatnonzero(missing)[:5]
            raise ValueError(f"{int(missing.sum())} bridge pairs are not in the network (first: "
                             f"{list(zip(self.pre[k].tolist(), self.post[k].tolist()))}); a pair that netted "
                             "to zero or a wrong index cannot be attributed")
        epos = pos[found]
        bparts = self.parts[found]
        meas = net.counts.astype(np.int64).copy()
        if len(epos):
            meas[epos] -= bparts.sum(axis=1)
        brow = np.full(nnz, -1, np.int64)
        brow[epos] = np.arange(len(epos))
        return AlignedParts(self.classes, meas, brow, epos, bparts, int((~found & dead).sum()))


@dataclass
class AlignedParts:
    classes: Tuple[str, ...]
    measured: np.ndarray      # (nnz,) signed measured count per CSR entry
    brow: np.ndarray          # (nnz,) row into bparts, -1 = no bridge part
    epos: np.ndarray          # CSR entries that carry a bridge part
    bparts: np.ndarray        # (len(epos), K) signed bridge counts
    skipped_dead: int


def synapse_count_share(net: S.Network, prov: Optional[EdgeProvenance] = None, onto=None) -> dict:
    """Secondary metric: measured synapses / all synapses in the simulated network (|signed count|
    per pair part), overall and per class. `onto`: only pairs whose post neuron is in this set."""
    prov = prov or EdgeProvenance.measured_only(())
    al = prov.align(net)
    sel = np.ones(net.nnz, bool)
    if onto is not None:
        is_t = np.zeros(net.n, bool)
        is_t[np.asarray(onto, np.int64)] = True
        sel = is_t[net.indices]
    meas = int(np.abs(al.measured[sel]).sum())
    bsel = sel[al.epos] if len(al.epos) else np.zeros(0, bool)
    per = {c: int(np.abs(al.bparts[bsel, q]).sum()) if len(al.epos) else 0 for q, c in enumerate(al.classes)}
    tot = meas + sum(per.values())
    bridge_only = int(((al.measured[al.epos] == 0) & bsel).sum()) if len(al.epos) else 0
    return {"measured_share": (meas / tot) if tot else None,
            "class_share": {c: (v / tot) if tot else None for c, v in per.items()},
            "synapses": dict(measured=meas, total=tot, **per),
            "pairs": int(sel.sum()), "pairs_with_bridge_part": int(bsel.sum()) if len(al.epos) else 0,
            "pairs_bridge_only": bridge_only, "bridge_pairs_skipped_dead": al.skipped_dead,
            "unit": "|signed synapse count| per pair part, as the simulator sees it (mixed-type pairs net out)"}


# =============================================================================================
# 2. input sources
# =============================================================================================

def guard_roots(sources, labels: Optional[Mapping[str, str]], classes: Sequence[str],
                root_targets=None) -> dict:
    """A root is a pre-registered target set, not a name. With `root_targets` (internal indices),
    every root-labelled source may drive only those neurons; anything else it touches would enter
    the accounting as eye input, so a bridge could hide inside it. Raises; returns what was checked."""
    out = {"enforced": root_targets is not None, "allowed": None, "root_sources": {}}
    allowed = None
    if root_targets is not None:
        allowed = np.unique(np.asarray(root_targets, np.int64))
        out["allowed"] = int(len(allowed))
    for s in sources:
        kind, label, _ = source_label(s.name, labels, classes)
        if kind != "root":
            continue
        t = np.asarray(s.targets, np.int64)
        out["root_sources"][s.name] = int(len(t))
        if allowed is not None:
            outside = t[~np.isin(t, allowed)]
            if len(outside):
                raise ValueError(f"root source {s.name!r} ({label}) drives {len(outside)} neurons outside the "
                                 f"pre-registered root set (first: {outside[:5].tolist()}); a link into them must be "
                                 "a 'bridge:<class>' source or a bridge edge")
    return out


def eye_segments_of_record(rule: Optional[str] = None) -> np.ndarray:
    """The pre-registered stimulated set of the brain of record (brain.stimulated_set), as sorted segment
    ids: B1's tectal cells that have a synapse in the pinned pairs, minus the never-stimulate exclusions,
    minus the direct readout partners recorded in the intact bundle. Its size must equal the recorded
    stim_info["stimulated"], or this raises."""
    from fishbrain import brain as B
    rule = rule or B.GATE_SIGN_RULE
    info = json.load(open(B.NET / f"g1_{rule}_intact.json"))["stats"]["stim_info"]
    ident = B.identified()
    ids = np.load(B.NET / "pairs_ids.npy", mmap_mode="r")
    t = np.asarray(ident["tectum"], np.uint64)
    present = t[B.index_of(ids, t) >= 0]
    drop = {int(x) for x in ident["exclusions"]} | {int(d["tectal_seg"]) for d in info["direct_pairs"]}
    seg = np.array(sorted({int(x) for x in present} - drop), np.uint64)
    if len(seg) != int(info["stimulated"]):
        raise AssertionError(f"rebuilt stimulated set has {len(seg)} cells, the bundle records {info['stimulated']}")
    return seg


def _ids_hash(ids) -> str:
    return hashlib.sha256(np.sort(np.asarray(ids, np.uint64)).astype("<u8").tobytes()).hexdigest()


def source_label(name: str, labels: Optional[Mapping[str, str]], classes: Sequence[str]) -> Tuple[str, str, int]:
    """(kind, label, mask) of an input source. kind is "root" or "bridge"."""
    lab = (labels or {}).get(name)
    if lab is None:
        if name == "retina":
            lab = EYE
        else:
            raise ValueError(f"input source {name!r} has no provenance label; pass source_labels="
                             f"{{{name!r}: 'eye' | 'chosen:<what>' | 'bridge:<class>'}}")
    if lab == EYE:
        return "root", EYE_LABEL, 0
    if lab.startswith("chosen:") and len(lab) > 7:
        return "root", f"chosen ({lab[7:]})", 0
    if lab.startswith("bridge:"):
        c = lab[7:]
        if c not in classes:
            raise ValueError(f"source {name!r}: bridge class {c!r} is not in {tuple(classes)}")
        return "bridge", f"bridge {c}", 1 << list(classes).index(c)
    raise ValueError(f"source {name!r}: label {lab!r} is not 'eye', 'chosen:<what>' or 'bridge:<class>'")


# =============================================================================================
# 3. the trace
# =============================================================================================

@dataclass
class Trace:
    """Provenance of one run. Columns are the touched neurons (anything that ever received input
    or fired); every other neuron received nothing."""
    classes: Tuple[str, ...]
    touched: np.ndarray                       # global indices, sorted
    windows: Dict[str, Tuple[float, float]]   # name -> [lo, hi) ms of simulation time
    drive: Dict[str, np.ndarray]              # window -> (2M + S, T) delivered drive, mV*ms
    sources: List[dict]                       # per input source: name, kind, label, mask, events
    spike_comp: np.ndarray                    # (n_spikes, M) float32: carried mask composition, raster order
    spike_root: np.ndarray                    # (n_spikes,) share of the spike's depolarization from root inputs
    raster: S.Raster
    checks: dict
    digests: dict
    synapse_count: dict
    n_masks: int = field(init=False)

    def __post_init__(self):
        self.n_masks = 1 << len(self.classes)

    # -- indexing ------------------------------------------------------------------------------
    def columns(self, idx) -> Tuple[np.ndarray, int]:
        """Local columns of the tracked neurons among `idx` (global), and how many were not tracked."""
        idx = np.unique(np.atleast_1d(np.asarray(idx, np.int64)))
        pos = np.searchsorted(self.touched, idx)
        pos_c = np.minimum(pos, max(len(self.touched) - 1, 0))
        ok = (pos < len(self.touched)) & (self.touched[pos_c] == idx) if len(self.touched) else np.zeros(len(idx), bool)
        return pos[ok], int((~ok).sum())

    def mask_names(self) -> List[str]:
        return [mask_name(m, self.classes) for m in range(self.n_masks)]

    # -- shares --------------------------------------------------------------------------------
    def _parts(self, cols, window: str):
        D = self.drive[window]
        M = self.n_masks
        exc = D[:M][:, cols].sum(axis=1)
        inh = -D[M:2 * M][:, cols].sum(axis=1)
        root = {}
        for s_i, s in enumerate(self.sources):
            v = float(D[2 * M + s_i, cols].sum())
            if s["kind"] == "bridge":
                exc[s["mask"]] += max(v, 0.0)
            else:
                root[s["label"]] = root.get(s["label"], 0.0) + max(v, 0.0)
        return exc, inh, root

    def share(self, idx, window: str) -> dict:
        """The decision-path share of a cell or a population (pooled = drive-weighted mean m)."""
        cols, untracked = self.columns(idx)
        exc, inh, root = self._parts(cols, window)
        M, names = self.n_masks, self.mask_names()
        link = float(exc.sum())
        inh_tot = float(inh.sum())
        root_tot = float(sum(root.values()))
        out = {"cells": int(len(cols) + untracked), "cells_without_input": int(untracked),
               "window": window, "window_ms": list(self.windows[window]),
               "link_drive_mV_ms": link, "root_drive_mV_ms": root, "inhibitory_drive_mV_ms": inh_tot}
        if link > 0:
            out["m"] = float(exc[0] / link)
            out["label"] = "wired"
            out["by_path"] = {names[k]: float(exc[k] / link) for k in range(M) if exc[k] > 0}
            out["bridge_through"] = {c: float(sum(exc[k] for k in range(M) if (k >> q) & 1) / link)
                                     for q, c in enumerate(self.classes)}
            out["bridge_only"] = {c: float(exc[1 << q] / link) for q, c in enumerate(self.classes)}
        else:
            out["m"] = None
            out["label"] = (EYE_LABEL if root.get(EYE_LABEL, 0.0) > 0 else
                            (sorted(k for k, v in root.items() if v > 0)[0] if root_tot > 0 else "no drive"))
            out["by_path"], out["bridge_through"], out["bridge_only"] = {}, {}, {}
        tot_direct = link + root_tot
        out["chosen_share"] = {k: (v / tot_direct if tot_direct > 0 else None) for k, v in root.items()}
        if inh_tot > 0:
            out["inhibitory"] = {"m": float(inh[0] / inh_tot),
                                 "bridge_through": {c: float(sum(inh[k] for k in range(M) if (k >> q) & 1) / inh_tot)
                                                    for q, c in enumerate(self.classes)}}
        else:
            out["inhibitory"] = None
        out["m_abs"] = float((exc[0] + inh[0]) / (link + inh_tot)) if (link + inh_tot) > 0 else None
        return out

    def m(self, window: str) -> np.ndarray:
        """m of every tracked neuron (NaN where it received no link drive)."""
        D = self.drive[window]
        M = self.n_masks
        exc = D[:M].copy()
        for s_i, s in enumerate(self.sources):
            if s["kind"] == "bridge":
                exc[s["mask"]] += np.maximum(D[2 * M + s_i], 0.0)
        tot = exc.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, exc[0] / np.where(tot > 0, tot, 1.0), np.nan)

    # -- spikes --------------------------------------------------------------------------------
    def spikes_of(self, i: int, window: Optional[str] = None) -> np.ndarray:
        """Raster positions of neuron i's spikes (in the window, if given)."""
        sel = self.raster.neurons == int(i)
        if window is not None:
            lo, hi = self.windows[window]
            t = self.raster.steps * self.raster.dt_ms
            sel &= (t >= lo) & (t < hi)
        return np.flatnonzero(sel)

    def spike_share(self, pos: int) -> dict:
        """What one spike carried: its composition, its own synaptic m (None if all direct input)."""
        comp = self.spike_comp[pos].astype(np.float64)
        root = float(self.spike_root[pos])
        names = self.mask_names()
        syn_m = None if root >= 1.0 - 1e-12 else float((comp[0] - root) / (1.0 - root))
        return {"neuron": int(self.raster.neurons[pos]), "t_ms": float(self.raster.steps[pos] * self.raster.dt_ms),
                "carried": {names[k]: float(comp[k]) for k in range(len(comp)) if comp[k] > 0},
                "carried_measured": float(comp[0]), "direct_root_share": root,
                "m": syn_m, "label": EYE_LABEL if syn_m is None else "wired"}


def _pristine_sources(sources) -> list:
    if isinstance(sources, S.Simulator):
        if sources.step != 0:
            raise ValueError("pass the simulator's sources before it ran (deep-copy sim.sources first)")
        sources = sources.sources
    return copy.deepcopy(list(sources))


def trace(net: S.Network, raster: S.Raster, sources, params: S.LIFParams = S.LIFParams(),
          prov: Optional[EdgeProvenance] = None, source_labels: Optional[Mapping[str, str]] = None,
          windows: Optional[Mapping[str, Tuple[float, float]]] = None, check: bool = True,
          sim_after: Optional[S.Simulator] = None, keep_spikes: bool = True, root_targets=None) -> Trace:
    """Provenance bookkeeping over a finished run.

    net:      the Network that was simulated (after any lesion).
    raster:   the run's raster; it must start at step 0 of a fresh simulator.
    sources:  the input sources as they were BEFORE the run (copy.deepcopy(sim.sources) taken
              before sim.run), or an unrun Simulator configured the same way.
    prov:     EdgeProvenance; None = every pair measured.
    windows:  name -> (lo_ms, hi_ms) of simulation time; delivered drive is summed per window.
    sim_after: the simulator after the run, for the state and stats comparison.
    root_targets: internal indices the root inputs may drive (the pre-registered stimulated set);
              a root source that touches anything else raises. None = not enforced (recorded).
    """
    prov = prov or EdgeProvenance.measured_only(())
    classes = prov.classes
    K = len(classes)
    M = 1 << K
    if raster.start_step != 0:
        raise ValueError("provenance needs the run from a fresh simulator (raster.start_step == 0)")
    if raster.n_neurons != net.n:
        raise ValueError("raster and network sizes differ")
    if abs(raster.dt_ms - params.dt_ms) > 1e-12:
        raise ValueError("raster dt differs from params dt")
    d = params.derived()
    a, b = float(d["a"]), float(d["b"])
    theta, u_reset = float(d["theta"]), float(d["u_reset"])
    n_ref, n_delay = d["n_ref"], d["n_delay"]
    dt = params.dt_ms
    wscale = params.w_syn_mV * d["c64"]
    area_syn = dt / ((1.0 - a) * (1.0 - b))
    area_kick = dt * a / (1.0 - a)
    area_cur = dt / (1.0 - a)
    n_steps = int(raster.n_steps)
    windows = dict(windows or {"all": (0.0, n_steps * dt)})

    srcs = _pristine_sources(sources)
    root_guard = guard_roots(srcs, source_labels, classes, root_targets)
    S_n = len(srcs)
    sinfo = []
    for s in srcs:
        kind, label, mask = source_label(s.name, source_labels, classes)
        sinfo.append({"name": s.name, "kind": kind, "label": label, "mask": mask, "source_kind": s.kind,
                      "events": 0})
    currents = [(i, s) for i, s in enumerate(srcs) if s.kind == "current"]
    poissons = [(i, s) for i, s in enumerate(srcs) if s.kind == "poisson"]

    # ---- edges: measured and bridge weights in h units ----
    al = prov.align(net)
    indptr, indices = net.indptr, net.indices.astype(np.int64)
    w_meas = al.measured.astype(np.float64) * wscale
    w_br = al.bparts.astype(np.float64) * wscale                     # (P, K)
    brow = al.brow
    remap = [np.arange(M)] + [np.arange(M) | (1 << q) for q in range(K)]   # mask after a class-q link

    # ---- touched set: every neuron that fires, receives a spike, or is a source target ----
    steps_r = np.asarray(raster.steps, np.int64)
    neur_r = np.asarray(raster.neurons, np.int64)
    if len(steps_r) and np.any(np.diff(steps_r) < 0):
        raise ValueError("raster steps are not in order")
    firing = np.unique(neur_r)
    tgt_all = [firing]
    if len(firing):
        st, cnt = indptr[firing], indptr[firing + 1] - indptr[firing]
        tot = int(cnt.sum())
        if tot:
            cs = np.cumsum(cnt)
            g = np.arange(tot, dtype=np.int64) + np.repeat(st - (cs - cnt), cnt)
            tgt_all.append(indices[g])
    for s in srcs:
        tgt_all.append(np.asarray(s.targets, np.int64))
    touched = np.unique(np.concatenate(tgt_all)) if tgt_all else np.zeros(0, np.int64)
    T = len(touched)
    loc = np.full(net.n, -1, np.int64)
    loc[touched] = np.arange(T)

    C = 2 * M + S_n + 1                   # logical channels: exc masks, inh masks, sources, reset offset
    OFF = 2 * M + S_n
    # Physical rows are handed out on first use, so each step integrates only the live channels.
    # Source rows (and the offset row, if v_reset != v_rest) exist from the start, so a row never
    # appears between a refractory save and its restore.
    phys = np.full(C, -1, np.int64)
    n_live = [0]

    def rows_of(chs) -> np.ndarray:
        chs = np.asarray(chs, np.int64)
        new = np.unique(chs[phys[chs] < 0])
        for ch in new.tolist():
            phys[ch] = n_live[0]
            n_live[0] += 1
        return phys[chs]

    rows_of(np.arange(2 * M, 2 * M + S_n))
    if u_reset != 0.0:
        rows_of([OFF])
    U = np.zeros((C, T), np.float64)
    H = np.zeros((C, T), np.float64)      # rows of source and offset channels stay 0
    Hflat = H.reshape(-1)
    wnames = list(windows)
    wsteps = [(float(windows[w][0]), float(windows[w][1])) for w in wnames]
    Dw = [np.zeros((2 * M + S_n, T), np.float64) for _ in wnames]
    Dflat = [D.reshape(-1) for D in Dw]

    bounds = np.searchsorted(steps_r, np.arange(n_steps + 1), side="left")
    nslot = n_delay + 1
    slot_src: List[np.ndarray] = [np.zeros(0, np.int64) for _ in range(nslot)]
    slot_comp: List[np.ndarray] = [np.zeros((0, M)) for _ in range(nslot)]
    ref = deque(maxlen=max(n_ref - 1, 0)) if n_ref > 1 else None
    dead_loc = net.dead[touched] if T else np.zeros(0, bool)
    spike_comp = np.zeros((len(neur_r), M), np.float32) if keep_spikes else np.zeros((0, M), np.float32)
    spike_root = np.zeros(len(neur_r), np.float32) if keep_spikes else np.zeros(0, np.float32)
    root_rows = [int(phys[2 * M + i]) for i, s in enumerate(sinfo) if s["kind"] == "root"]
    bridge_rows = [(int(phys[2 * M + i]), s["mask"]) for i, s in enumerate(sinfo) if s["kind"] == "bridge"]
    mirror = {"missed_fire": 0, "extra_fire": 0, "examples": [], "min_margin_fired_mV": None,
              "max_free_minus_theta_mV": None}
    syn_events = 0
    kick_events = 0
    min_fired, max_free = np.inf, -np.inf

    for k in range(n_steps):
        tk = k * dt
        L = n_live[0]
        # 1. integrate; refractory neurons are frozen (their saved state is put back)
        R = np.concatenate(ref) if ref else None
        frozen_any = R is not None and len(R) > 0
        if frozen_any:
            UR, HR = U[:L, R].copy(), H[:L, R].copy()
        Ul, Hl = U[:L], H[:L]
        np.multiply(Ul, a, out=Ul)
        Ul += Hl
        np.multiply(Hl, b, out=Hl)
        for i, s in currents:
            if s.active(k):
                f = (k - s.start) // s.frame_steps
                if f < len(s.inc):
                    tl = loc[np.asarray(s.targets, np.int64)]
                    inc = np.broadcast_to(np.asarray(s.inc[f], np.float64), tl.shape)
                    U[phys[2 * M + i], tl] += inc
                    live = np.ones(len(tl), bool)
                    if frozen_any:
                        frozen = np.zeros(T, bool)
                        frozen[R] = True
                        live = ~frozen[tl]
                    for wi, (lo, hi) in enumerate(wsteps):
                        if lo <= tk < hi:
                            np.add.at(Dw[wi][2 * M + i], tl[live], inc[live] * area_cur)
        if frozen_any:
            U[:L, R] = UR
            H[:L, R] = HR
        # 2. the spikes this step (from the raster) and what each carries
        fg = neur_r[bounds[k]:bounds[k + 1]]
        fl = loc[fg]
        if check and T:
            recon = U[:L].sum(axis=0)
            if len(fl):
                marg = recon[fl] - theta
                tol = MIRROR_TOL_MV + MIRROR_RTOL * np.abs(recon[fl])
                bad = marg <= -tol
                min_fired = min(min_fired, float(marg.min()))
                if bad.any():
                    mirror["missed_fire"] += int(bad.sum())
                    if len(mirror["examples"]) < 5:
                        mirror["examples"].append({"step": k, "kind": "raster spike below threshold",
                                                   "neuron": int(fg[bad][0]), "margin_mV": float(marg[bad][0])})
            free = ~dead_loc.copy()
            free[fl] = False
            if frozen_any:
                free[R] = False
            if free.any():
                over = recon[free] - theta
                tol = MIRROR_TOL_MV + MIRROR_RTOL * np.abs(recon[free])
                max_free = max(max_free, float(over.max()))
                badf = over > tol
                if badf.any():
                    mirror["extra_fire"] += int(badf.sum())
                    if len(mirror["examples"]) < 5:
                        mirror["examples"].append({"step": k, "kind": "above threshold without a spike",
                                                   "neuron": int(touched[np.flatnonzero(free)[badf][0]]),
                                                   "over_mV": float(over[badf][0])})
        if len(fl):
            pos = np.zeros((M, len(fl)))
            live_e = np.flatnonzero(phys[:M] >= 0)
            if len(live_e):
                pos[live_e] = U[np.ix_(phys[live_e], fl)]
            rootp = np.zeros(len(fl))
            for r in root_rows:
                v = np.maximum(U[r, fl], 0.0)
                pos[0] += v
                rootp += v
            for r, mk in bridge_rows:
                pos[mk] += np.maximum(U[r, fl], 0.0)
            np.maximum(pos, 0.0, out=pos)
            tot = pos.sum(axis=0)
            if np.any(tot <= 0):
                j = int(fg[np.flatnonzero(tot <= 0)[0]])
                raise RuntimeError(f"step {k}: neuron {j} spiked with no depolarizing input in the bookkeeping; "
                                   "the replay does not match the run")
            comp = (pos / tot).T                                   # (nf, M)
            if keep_spikes:
                spike_comp[bounds[k]:bounds[k + 1]] = comp
                spike_root[bounds[k]:bounds[k + 1]] = rootp / tot
        else:
            comp = np.zeros((0, M))
        # 3. deliveries of the spikes fired n_delay steps ago, then the inputs' kicks
        slot_src[k % nslot] = fg
        slot_comp[k % nslot] = comp
        arr = slot_src[(k - n_delay) % nslot]
        if len(arr) and k >= n_delay:
            acomp = slot_comp[(k - n_delay) % nslot]
            st = indptr[arr]
            cnt = indptr[arr + 1] - st
            tot_e = int(cnt.sum())
            if tot_e:
                cs = np.cumsum(cnt)
                g = np.arange(tot_e, dtype=np.int64) + np.repeat(st - (cs - cnt), cnt)
                srcpos = np.repeat(np.arange(len(arr)), cnt)
                tl = loc[indices[g]]
                syn_events += tot_e
                in_w = [wi for wi, (lo, hi) in enumerate(wsteps) if lo <= tk < hi]
                br = brow[g]
                hasb = br >= 0
                for q in range(K + 1):
                    if q == 0:
                        wq = w_meas[g]
                    else:
                        if not hasb.any():
                            break
                        wq = np.zeros(len(g))
                        wq[hasb] = w_br[br[hasb], q - 1]
                    sel = np.flatnonzero(wq != 0)
                    if not len(sel):
                        continue
                    vals = (wq[sel, None] * acomp[srcpos[sel]]).ravel()          # (E, M) flattened
                    base = np.where(wq[sel] > 0, 0, M)
                    chan = (base[:, None] + remap[q][None, :]).ravel()          # logical channel
                    tcol = np.repeat(tl[sel], M)
                    nzv = vals != 0
                    if not nzv.all():
                        vals, chan, tcol = vals[nzv], chan[nzv], tcol[nzv]
                    if not len(vals):
                        continue
                    np.add.at(Hflat, rows_of(chan) * T + tcol, vals)
                    for wi in in_w:
                        np.add.at(Dflat[wi], chan * T + tcol, vals * area_syn)
        for i, s in poissons:
            if s.active(k):
                hit = s.hits(k)
                if hit is not None:
                    hl = loc[np.asarray(hit, np.int64)]
                    wgt = float(s.weight)
                    U[phys[2 * M + i], hl] += wgt
                    kick_events += len(hl)
                    sinfo[i]["events"] += len(hl)
                    for wi, (lo, hi) in enumerate(wsteps):
                        if lo <= tk < hi:
                            np.add.at(Dw[wi][2 * M + i], hl, wgt * area_kick)
        # 4. reset (v = v_reset, g = 0), as the simulator does after this step's deliveries
        if len(fl):
            L = n_live[0]
            U[:L, fl] = 0.0
            H[:L, fl] = 0.0
            if u_reset != 0.0:
                U[phys[OFF], fl] = u_reset
        if ref is not None:
            ref.append(fl)
    checks_live_channels = int(n_live[0])

    # ---- checks: parts against the total, and against the simulator ----
    checks = {"mirror": mirror, "n_steps": n_steps, "touched": int(T), "live_channels": checks_live_channels,
              "spikes": int(len(neur_r)),
              "syn_events": int(syn_events), "poisson_events": int(kick_events)}
    if np.isfinite(min_fired):
        mirror["min_margin_fired_mV"] = min_fired
    if np.isfinite(max_free):
        mirror["max_free_minus_theta_mV"] = max_free
    if check:
        # signed drive per window straight from the raster and the network counts (no provenance)
        arrive = steps_r + n_delay
        wsum = []
        w_all = net.counts.astype(np.float64) * wscale
        for wi, (lo, hi) in enumerate(wsteps):
            t_arr = arrive * dt
            sel = (arrive < n_steps) & (t_arr >= lo) & (t_arr < hi)
            src = neur_r[sel]
            direct = np.zeros(net.n)
            if len(src):
                st, cnt = indptr[src], indptr[src + 1] - indptr[src]
                tot_e = int(cnt.sum())
                if tot_e:
                    cs = np.cumsum(cnt)
                    g = np.arange(tot_e, dtype=np.int64) + np.repeat(st - (cs - cnt), cnt)
                    direct = np.bincount(indices[g], weights=w_all[g] * area_syn, minlength=net.n)
            books = Dw[wi][:2 * M].sum(axis=0)
            ref_t = direct[touched]
            diff = float(np.max(np.abs(books - ref_t))) if T else 0.0
            scale = float(np.max(np.abs(ref_t))) if T else 0.0
            outside = float(np.abs(np.delete(direct, touched)).max()) if net.n > T else 0.0
            wsum.append({"window": wnames[wi], "max_abs_diff_mV_ms": diff, "max_abs_drive_mV_ms": scale,
                         "drive_outside_tracked": outside,
                         "ok": bool(diff <= 1e-9 * max(1.0, scale) and outside == 0.0)})
        checks["parts_sum_to_total"] = wsum
        checks["mirror_ok"] = mirror["missed_fire"] == 0 and mirror["extra_fire"] == 0
    if sim_after is not None:
        su = sim_after.u.astype(np.float64)
        sh = sim_after.h.astype(np.float64)
        ru, rh = U.sum(axis=0), H.sum(axis=0)
        du = float(np.max(np.abs(su[touched] - ru) / (1.0 + np.abs(su[touched])))) if T else 0.0
        dh = float(np.max(np.abs(sh[touched] - rh) / (1.0 + np.abs(sh[touched])))) if T else 0.0
        rest = np.ones(net.n, bool)
        rest[touched] = False
        untouched_zero = bool(not np.any(su[rest]) and not np.any(sh[rest]))
        st = sim_after.stats
        checks["simulator"] = {
            "u_max_rel_diff": du, "h_max_rel_diff": dh, "untracked_state_zero": untouched_zero,
            "spikes_equal": int(st["spikes"]) == int(len(neur_r)),
            "syn_events_equal": int(st["syn_events"]) == int(syn_events),
            "poisson_events_equal": int(st["poisson_events"]) == int(kick_events),
            "ok": bool(du < 1e-4 and dh < 1e-4 and untouched_zero and int(st["spikes"]) == len(neur_r)
                       and int(st["syn_events"]) == syn_events and int(st["poisson_events"]) == kick_events)}
    digests = {"provenance_version": PROVENANCE_VERSION, "network": net.digest(), "edge_provenance": prov.digest(),
               "params": params.digest(), "raster": S.spike_hash(raster),
               "source_labels": hashlib.sha256(json.dumps([[s["name"], s["label"], s["mask"]] for s in sinfo]).encode()).hexdigest(),
               "root_targets": {s.name: _ids_hash(net.ids[np.asarray(s.targets, np.int64)])
                                for s, si in zip(srcs, sinfo) if si["kind"] == "root"},
               "root_allowed": None if root_targets is None else _ids_hash(net.ids[np.unique(np.asarray(root_targets, np.int64))])}
    checks["root_guard"] = root_guard
    return Trace(classes, touched, windows, {w: Dw[i] for i, w in enumerate(wnames)}, sinfo, spike_comp, spike_root,
                 raster, checks, digests, synapse_count_share(net, prov))


# =============================================================================================
# 4. traced runs (the simulator is used as is; the trace is computed afterwards)
# =============================================================================================

def run_traced(net: S.Network, seed: str, inputs: Callable[[S.Simulator], None], duration_ms: Optional[float] = None,
               n_steps: Optional[int] = None, params: S.LIFParams = S.LIFParams(),
               prov: Optional[EdgeProvenance] = None, source_labels: Optional[Mapping[str, str]] = None,
               windows: Optional[Mapping[str, Tuple[float, float]]] = None, check: bool = True,
               fast: bool = False, root_targets=None) -> Tuple[S.Raster, Trace]:
    """Build a simulator (sim.Simulator, or sim_fast's choice when fast=True: identical spikes),
    call inputs(sim) to add its sources, keep a copy of them, run, then trace."""
    if fast:
        from fishbrain.sim_fast import make_simulator
        sim = make_simulator(net, seed=seed, params=params)
    else:
        sim = S.Simulator(net, seed=seed, params=params)
    inputs(sim)
    guard_roots(sim.sources, source_labels, prov.classes if prov else (), root_targets)   # before spending the run
    pristine = copy.deepcopy(sim.sources)
    r = sim.run(duration_ms, n_steps)
    tr = trace(net, r, pristine, params, prov, source_labels, windows, check, sim_after=sim, root_targets=root_targets)
    return r, tr


def trial_traced(bundle, rates: np.ndarray, seed: str, prov: Optional[EdgeProvenance] = None,
                 windows: Optional[Mapping[str, Tuple[float, float]]] = None, lesion_ids=None,
                 params: S.LIFParams = S.LIFParams(), frame_ms: Optional[float] = None,
                 source_labels: Optional[Mapping[str, str]] = None, check: bool = True,
                 eye_segments: Union[str, Sequence[int], None] = "record") -> Tuple[S.Raster, Trace]:
    """brain.run_trial's run (one "retina" Poisson source on bundle.stim_idx), traced. Its raster
    hash equals run_trial's spike_hash (tests/test_provenance.py checks it on the gate bundle).
    eye_segments: "record" (default) = the eye may drive only the brain of record's stimulated set;
    an explicit segment-id list overrides it (and is pinned in the digests); None disables the guard."""
    from fishbrain import brain as B
    frame_ms = B.RETINA.frame_ms if frame_ms is None else frame_ms
    net = bundle.net if not lesion_ids else bundle.net.lesion(np.asarray(lesion_ids, np.uint64))

    def inputs(sim):
        sim.add_poisson(bundle.stim_idx, rates, frame_ms=frame_ms, name="retina")

    root_targets = None
    if eye_segments is not None:
        seg = eye_segments_of_record(bundle.rule) if isinstance(eye_segments, str) and eye_segments == "record" \
            else np.asarray(eye_segments, np.uint64)
        root_targets = net.index_of(seg[np.isin(seg, net.ids)]) if len(seg) else np.zeros(0, np.int64)
    return run_traced(net, seed, inputs, duration_ms=rates.shape[0] * frame_ms, params=params, prov=prov,
                      source_labels=source_labels, windows=windows, check=check, root_targets=root_targets)


# =============================================================================================
# 5. the report: per readout, per bridge class, the deciding cells
# =============================================================================================

def readout_groups(net: S.Network, sets: Mapping[str, Sequence[int]]) -> Tuple[Dict[str, np.ndarray], Dict[str, List[int]]]:
    """Segment ids per readout key (readouts.readout_sets) -> internal indices present in `net`,
    and the ids that are not in it."""
    present, missing = {}, {}
    ids = net.ids
    order = np.argsort(ids, kind="stable")
    sid = ids[order]
    for k, v in sets.items():
        q = np.asarray([int(x) for x in v], np.uint64)
        pos = np.searchsorted(sid, q)
        pc = np.minimum(pos, max(len(sid) - 1, 0))
        ok = (pos < len(sid)) & (sid[pc] == q) if len(sid) else np.zeros(len(q), bool)
        present[k] = np.sort(order[pc[ok]]).astype(np.int64)
        missing[k] = [int(x) for x in q[~ok]]
    return present, missing


M_SYSTEM = ("Mauthner", "MiD2cm", "MiD3cm")
DECIDERS = {   # CHOICE: the populations that decide, per the pre-registration and readouts.py's rules
    "turn": ("turning_left", "turning_right"),
    "strike": ("nMLF_left", "nMLF_right", "nIII_dorsal_left", "nIII_dorsal_right"),   # the strike rule reads both
    "strike_nMLF": ("nMLF_left", "nMLF_right"),                                        # G1c-measure's strike set
}


def escape_initiator(tr: Trace, groups: Mapping[str, np.ndarray], window: str) -> dict:
    """readouts.escape_decision over the M-system cells' spikes in the window; the initiating cell."""
    from fishbrain import readouts as RO
    first, counts, cell = {}, {}, {}
    lo, hi = tr.windows[window]
    t = tr.raster.steps * tr.raster.dt_ms
    for c in M_SYSTEM:
        for s in ("left", "right"):
            key = f"{c}_{s}"
            idx = groups.get(key, np.zeros(0, np.int64))
            if not len(idx):
                continue
            i = int(idx[0])
            sel = (tr.raster.neurons == i) & (t >= lo) & (t < hi)
            counts[key] = int(sel.sum())
            first[key] = float(t[sel].min()) if sel.any() else None
            cell[key] = i
    dec = RO.escape_decision(first, counts)
    init = dec.get("initiator")
    keys = [] if init is None else ([init] if isinstance(init, str) else list(init))
    dec["initiator_cells"] = {k: cell[k] for k in keys}
    return dec


def report(tr: Trace, groups: Mapping[str, np.ndarray], window: str,
           deciders: Mapping[str, Sequence[str]] = DECIDERS, pooled: Sequence[str] = ("escape", "turn", "strike")) -> dict:
    """Per readout key and per deciding group: m, bridge shares per class, direct eye share; the
    escaping M-system cell (window drive and its first spike); the pooled decision-path share
    over the groups in `pooled`; the synapse-count share.
    CHOICE: `pooled` defaults to every decider group present; per condition the caller should pass
    the groups that decide it (a loom: ("escape",); prey: ("strike", "turn")). "strike" is nMLF plus
    dorsal nIII because readouts.strike_decision reads both; "strike_nMLF" is G1c-measure's set."""
    per = {k: tr.share(v, window) for k, v in groups.items() if len(v)}
    absent = sorted(k for k, v in groups.items() if not len(v))
    esc = escape_initiator(tr, groups, window)
    out = {"window": window, "window_ms": list(tr.windows[window]), "classes": list(tr.classes),
           "per_readout": per, "readouts_absent_from_network": absent, "escape": esc, "groups": {}}
    grp_idx = {}
    if esc.get("initiator_cells"):
        ei = np.array(sorted(esc["initiator_cells"].values()), np.int64)
        grp_idx["escape"] = ei
        fs = {}
        for key, i in esc["initiator_cells"].items():
            pos = tr.spikes_of(i, window)
            fs[key] = tr.spike_share(int(pos[0])) if len(pos) else None
        out["escape"]["first_spike"] = fs
    for gname, keys in deciders.items():
        idx = [groups[k] for k in keys if k in groups and len(groups[k])]
        if idx:
            grp_idx[gname] = np.unique(np.concatenate(idx))
    for gname, idx in grp_idx.items():
        out["groups"][gname] = tr.share(idx, window)
    use = [grp_idx[g] for g in pooled if g in grp_idx]
    out["decision_path_share"] = tr.share(np.unique(np.concatenate(use)), window) if use else None
    out["decision_path_groups"] = [g for g in pooled if g in grp_idx]
    out["synapse_count_share"] = tr.synapse_count
    out["checks"] = tr.checks
    out["digests"] = tr.digests
    return out


def to_json(obj):
    """JSON-safe copy (numpy scalars and arrays to Python)."""
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
