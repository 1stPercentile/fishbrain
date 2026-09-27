"""FISHBRAIN spiking simulator: leaky integrate-and-fire on a sparse connectome (G1, task B2).

Model
-----
The neuron and synapse model is the whole-brain LIF model of Shiu et al. 2024 (Nature 634:210,
"A Drosophila computational brain model reveals sensorimotor processing"), as written in its code,
github.com/philshiu/Drosophila_brain_model `model.py` (commit 91bdd1e, 2024-09-14):

    dv/dt = (v_0 - v + g) / t_mbr     (unless refractory)
    dg/dt = -g / tau                  (unless refractory)
    spike when v > v_th; then v = v_rst and g = 0; refractory for t_rfc
    a presynaptic spike adds w = (signed synapse count) * w_syn to g, t_dly later
    a Poisson input adds f_poi * w_syn straight to v

with v_0 = v_rst = -52 mV, v_th = -45 mV, t_mbr = 20 ms, tau = 5 ms, t_rfc = 2.2 ms,
t_dly = 1.8 ms, w_syn = 0.275 mV, f_poi = 250. Shiu integrates it with Brian2 `method='linear'`
(exact) at Brian2's default clock of 0.1 ms; this file does the same closed-form update.
flycoinrh (github.com/fruitflydev/flycoinrh, MIT, Copyright (c) 2026 fruitflydev) follows the same
paper; its `flysim.py` ragged gather over the rows of the neurons that fired is the pattern reused
for spike propagation here. flycoinrh itself drops the synaptic time constant and the delay and
uses dt = 0.2 ms; this file keeps Shiu's. Every departure is listed in evidence/G1-sim.md.

Step order (fixed; it is Brian2's default schedule groups -> thresholds -> synapses -> resets):
  1. integrate every neuron that is not refractory (exact solution over one dt; currents too)
  2. threshold: fired = non-refractory, living neurons with v > v_th, ascending index
  3. synapses: deliver the spikes fired `delay` steps ago (rows ascending, targets ascending),
     then Poisson kicks, source by source in the order they were added
  4. reset: v = v_rst and g = 0 for the neurons that fired this step; they start refractory

Internally the state is u = v - v_rest and h = c * g (c is the exact-solution coupling), so one
step of step 1 is three array operations: u *= a; u += h; h *= b.

Determinism
-----------
* All randomness comes from PCG64 streams seeded with sha256 of the seed string (a block hash).
  Each input source gets its own stream, sha256(seed | "poisson" | source name), so splitting a
  run into pieces, or adding a source later, never shifts another source's draws.
* Only PCG64's raw 64-bit output is used (`random_raw`). numpy documents no stream guarantee for
  Generator methods (binomial, choice, random); the raw PCG64 stream is fixed by the algorithm.
  Bernoulli draws are an integer compare, raw < floor(p * 2**64). Poisson trains of a constant
  rate use geometric gaps computed with float64 multiplications only (no libm).
* Decay constants come from Python's `decimal` exp (software, correctly rounded), not libm, and
  their float32 bit patterns are part of `LIFParams.digest()`.
* State is float32; every per-step operation is an elementwise IEEE add or multiply, or an
  in-order `np.add.at`, so results do not depend on SIMD width or thread count. Pure numpy.
* `spike_hash()` hashes the raster; `Network.digest()` hashes the wiring; `LIFParams.digest()` the
  parameters; `Simulator.state_digest()` the full dynamic state. A replay receipt pins all four.
"""
from __future__ import annotations

import copy
import dataclasses
import decimal
import hashlib
import json
import struct
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp

MODEL_VERSION = "fishbrain-lif-v1"
_U64 = np.uint64


# ---------------------------------------------------------------------------------------------
# seeds

def derive_seed(seed: Union[str, bytes, int], *tags: str) -> int:
    """sha256 of the seed string (UTF-8, used verbatim), optionally joined with tags by '|'.

    derive_seed("0xabc") == int.from_bytes(sha256(b"0xabc").digest(), "big").
    """
    if isinstance(seed, bytes):
        seed = seed.decode("utf-8")
    parts = [str(seed)] + [str(t) for t in tags]
    return int.from_bytes(hashlib.sha256("|".join(parts).encode("utf-8")).digest(), "big")


def make_bitgen(seed: Union[str, bytes, int], *tags: str) -> np.random.PCG64:
    """A PCG64 bit generator seeded from derive_seed(seed, *tags). Use only its random_raw()."""
    return np.random.PCG64(derive_seed(seed, *tags))


# ---------------------------------------------------------------------------------------------
# parameters

def _dec(x) -> decimal.Decimal:
    return decimal.Decimal(repr(float(x)))


def _steps(ms: float, dt: float, what: str) -> int:
    n = float(_dec(ms) / _dec(dt))
    k = int(round(n))
    if abs(n - k) > 1e-9:
        raise ValueError(f"{what} = {ms} ms is not a whole number of dt = {dt} ms steps")
    return k


@dataclass(frozen=True)
class LIFParams:
    """Every model parameter, in one place. Defaults = Shiu et al. 2024 model.py (commit 91bdd1e).

    v_rest_mV       v_0     -52 mV   resting potential (Kakaria & de Bivort 2017, via Shiu)
    v_reset_mV      v_rst   -52 mV   reset potential
    v_thresh_mV     v_th    -45 mV   spike threshold (strict v > v_th)
    tau_m_ms        t_mbr    20 ms   membrane time constant
    tau_syn_ms      tau       5 ms   synaptic time constant (Jurgensen et al., via Shiu)
    refractory_ms   t_rfc   2.2 ms   refractory period (Lazar et al. 2021, via Shiu)
    delay_ms        t_dly   1.8 ms   transmission delay (Paul et al. 2015, via Shiu)
    w_syn_mV        w_syn 0.275 mV   weight per synapse (Shiu: a free parameter)
    poisson_weight_mV     68.75 mV   f_poi (250) * w_syn: one Poisson event = one kick into v
    dt_ms                   0.1 ms   Brian2's default clock (Shiu does not set dt)
    """
    v_rest_mV: float = -52.0
    v_reset_mV: float = -52.0
    v_thresh_mV: float = -45.0
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    w_syn_mV: float = 0.275
    poisson_weight_mV: float = 68.75
    dt_ms: float = 0.1

    def __post_init__(self):
        if not self.v_reset_mV < self.v_thresh_mV:
            raise ValueError("v_reset must be below v_thresh")
        if self.dt_ms <= 0 or self.tau_m_ms <= 0 or self.tau_syn_ms <= 0:
            raise ValueError("dt and time constants must be positive")
        _steps(self.refractory_ms, self.dt_ms, "refractory_ms")
        _steps(self.delay_ms, self.dt_ms, "delay_ms")

    # derived, platform-independent constants
    def derived(self) -> dict:
        with decimal.localcontext() as ctx:
            ctx.prec = 50
            dt, tm, ts = _dec(self.dt_ms), _dec(self.tau_m_ms), _dec(self.tau_syn_ms)
            a = (-(dt / tm)).exp()           # membrane decay per step
            b = (-(dt / ts)).exp()           # synaptic decay per step
            if tm != ts:                     # u(t+dt) = a u + c g for the exact linear solution
                c = ts / (ts - tm) * (b - a)
            else:
                c = (dt / tm) * a
            k_i = 1 - a                      # a constant drive I (mV) adds I * (1 - a) per step
        a64, b64, c64, ki64 = float(a), float(b), float(c), float(k_i)
        return {
            "a": np.float32(a64), "b": np.float32(b64), "c": np.float32(c64),
            "k_i": np.float32(ki64), "c64": c64, "k_i64": ki64,
            "theta": np.float32(self.v_thresh_mV - self.v_rest_mV),
            "u_reset": np.float32(self.v_reset_mV - self.v_rest_mV),
            "n_ref": _steps(self.refractory_ms, self.dt_ms, "refractory_ms"),
            "n_delay": _steps(self.delay_ms, self.dt_ms, "delay_ms"),
        }

    def steps(self, ms: float) -> int:
        """Number of dt steps in `ms` (must be a whole number)."""
        return _steps(ms, self.dt_ms, "duration")

    def digest(self) -> str:
        d = self.derived()
        bits = {k: struct.pack("<f", float(d[k])).hex() for k in ("a", "b", "c", "k_i", "theta", "u_reset")}
        payload = {"model": MODEL_VERSION, "params": {k: repr(v) for k, v in dataclasses.asdict(self).items()},
                   "float32_bits": bits, "n_ref": d["n_ref"], "n_delay": d["n_delay"]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# wiring

class Network:
    """A directed wiring diagram as CSR: row = presynaptic neuron, column = postsynaptic neuron.

    counts[k] is the *signed* synapse count of one (pre, post) pair: the sum over that pair's
    synapses of sign_map[type]. Mixed-type pairs net out; pairs that net to zero are dropped.
    `ids` are the external neuron ids (Fish1 segment ids, uint64); indices 0..n-1 are internal,
    stay fixed under lesion(), and are what rasters record.
    """

    def __init__(self, ids, indptr, indices, counts, dead=None, meta=None):
        self.ids = np.ascontiguousarray(ids, dtype=np.uint64)
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.counts = np.ascontiguousarray(counts, dtype=np.int32)
        n = len(self.ids)
        if len(self.indptr) != n + 1 or self.indptr[-1] != len(self.indices) or len(self.counts) != len(self.indices):
            raise ValueError("inconsistent CSR arrays")
        if len(np.unique(self.ids)) != n:
            raise ValueError("neuron ids must be unique")
        self.dead = np.zeros(n, bool) if dead is None else np.ascontiguousarray(dead, dtype=bool)
        self.meta = dict(meta or {})
        self._sorter = None

    # -- construction --------------------------------------------------------------------------
    @classmethod
    def from_pairs(cls, pre_idx, post_idx, signed_counts, n: int, ids=None, meta=None) -> "Network":
        """Build from index pairs; duplicate pairs are summed, zero pairs dropped."""
        pre_idx = np.asarray(pre_idx, dtype=np.int64)
        post_idx = np.asarray(post_idx, dtype=np.int64)
        data = np.asarray(signed_counts, dtype=np.int32)
        if len(pre_idx) and (pre_idx.min() < 0 or post_idx.min() < 0 or pre_idx.max() >= n or post_idx.max() >= n):
            raise ValueError("pair index out of range")
        m = sp.coo_matrix((data, (pre_idx, post_idx)), shape=(n, n)).tocsr()
        m.sum_duplicates()
        m.eliminate_zeros()
        m.sort_indices()
        if ids is None:
            ids = np.arange(n, dtype=np.uint64)
        return cls(ids, m.indptr, m.indices, m.data, meta=meta)

    @classmethod
    def from_synapses(cls, pre_ids, post_ids, types, sign_map: Mapping[int, int], counts=None,
                      ids=None, meta=None) -> "Network":
        """Build from synapse records (one row per synapse, or per pair with `counts`).

        pre_ids/post_ids: external ids (uint64). types: the E/I class value of each row.
        sign_map: {type_value: +1 or -1 (or 0 to drop)}. It is required and nothing here assumes
        which type value is excitatory. Unknown type values raise.
        ids: the neuron order; default sorted unique ids seen in pre/post.
        """
        pre_ids = np.asarray(pre_ids, dtype=np.uint64)
        post_ids = np.asarray(post_ids, dtype=np.uint64)
        types = np.asarray(types)
        counts = np.ones(len(pre_ids), np.int64) if counts is None else np.asarray(counts, np.int64)
        if not (len(pre_ids) == len(post_ids) == len(types) == len(counts)):
            raise ValueError("pre, post, types and counts must have the same length")
        keys = np.array(sorted(int(k) for k in sign_map), dtype=np.int64)
        vals = np.array([int(sign_map[int(k)]) for k in keys], dtype=np.int64)
        if not np.all(np.isin(vals, (-1, 0, 1))):
            raise ValueError("sign_map values must be +1, -1 or 0")
        present = np.unique(types).astype(np.int64)
        unknown = present[~np.isin(present, keys)]
        if len(unknown):
            raise ValueError(f"type values {unknown.tolist()} are not in sign_map")
        signs = vals[np.searchsorted(keys, types.astype(np.int64))]
        if ids is None:
            ids = np.unique(np.concatenate([pre_ids, post_ids]))
        ids = np.asarray(ids, dtype=np.uint64)
        net0 = cls(ids, np.zeros(len(ids) + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32))
        pre = net0.index_of(pre_ids)
        post = net0.index_of(post_ids)
        m = dict(meta or {})
        m["sign_map"] = {str(int(k)): int(v) for k, v in zip(keys, vals)}
        return cls.from_pairs(pre, post, signs * counts, len(ids), ids=ids, meta=m)

    # -- queries -------------------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def nnz(self) -> int:
        return len(self.indices)

    def index_of(self, ids) -> np.ndarray:
        """Internal indices of external ids; raises KeyError for unknown ids."""
        q = np.atleast_1d(np.asarray(ids, dtype=np.uint64))
        if self._sorter is None:
            self._sorter = np.argsort(self.ids, kind="stable")
        sid = self.ids[self._sorter]
        pos = np.searchsorted(sid, q)
        pos_c = np.minimum(pos, len(sid) - 1)
        bad = (pos >= len(sid)) | (sid[pos_c] != q)
        if bad.any():
            raise KeyError(f"unknown neuron ids: {q[bad][:5].tolist()}")
        return self._sorter[pos_c].astype(np.int64)

    def rows(self) -> np.ndarray:
        return np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.indptr))

    def out_degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.indices, minlength=self.n)

    def out_strength(self) -> np.ndarray:
        """Per neuron: sum of |signed count| over its outgoing pairs."""
        return np.bincount(self.rows(), weights=np.abs(self.counts).astype(np.float64), minlength=self.n).astype(np.int64)

    def digest(self) -> str:
        h = hashlib.sha256(b"fishbrain-net-v1")
        h.update(struct.pack("<QQ", self.n, self.nnz))
        for arr, dt in ((self.ids, "<u8"), (self.indptr, "<i8"), (self.indices, "<i8"),
                        (self.counts, "<i4"), (self.dead, "u1")):
            h.update(np.ascontiguousarray(arr.astype(dt)).tobytes())
        return h.hexdigest()

    def nbytes(self) -> int:
        return int(self.ids.nbytes + self.indptr.nbytes + self.indices.nbytes + self.counts.nbytes + self.dead.nbytes)

    # -- controls ------------------------------------------------------------------------------
    def lesion(self, neuron_ids) -> "Network":
        """A copy with these neurons (external ids) removed: every synapse in or out of them is
        dropped and the simulator never lets them fire. Indices of all other neurons are kept."""
        return self.lesion_indices(self.index_of(neuron_ids))

    def lesion_indices(self, idx) -> "Network":
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        dead = self.dead.copy()
        dead[idx] = True
        rows = self.rows()
        keep = ~dead[rows] & ~dead[self.indices]
        counts_per_row = np.bincount(rows[keep], minlength=self.n)
        indptr = np.concatenate([[0], np.cumsum(counts_per_row)]).astype(np.int64)
        meta = dict(self.meta)
        meta["lesioned_ids"] = sorted(set(meta.get("lesioned_ids", [])) | {int(i) for i in self.ids[idx]})
        return Network(self.ids, indptr, self.indices[keep], self.counts[keep], dead=dead, meta=meta)

    def shuffled_copy(self, seed, allow_self_loops: bool = False, max_rounds: int = 200) -> "Network":
        """Degree-preserving rewiring (a control for "does the wiring matter?").

        Each pair keeps its presynaptic neuron and its signed count; the postsynaptic ends are
        permuted uniformly across all pairs (PCG64 from sha256(seed | "shuffle")). Pairs that the
        permutation duplicates, and self-loops unless allowed, are repaired by swapping their
        postsynaptic end with another random pair until none remain. So every neuron keeps its
        out-degree, in-degree, out-strength and outgoing sign mix; in-strength is not preserved.
        meta["shuffle"] records rounds and any pairs left unrepaired (then merged, and degrees
        are no longer exact).
        """
        n, nnz = self.n, self.nnz
        bg = make_bitgen(seed, "shuffle")
        rows = self.rows()
        cols = self.indices.astype(np.int64)[np.argsort(bg.random_raw(nnz), kind="stable")]
        rounds, n_bad = 0, 0
        for rounds in range(1, max_rounds + 1):
            key = rows * n + cols
            order = np.argsort(key, kind="stable")
            ks = key[order]
            bad_mask = np.zeros(nnz, bool)
            bad_mask[order[1:][ks[1:] == ks[:-1]]] = True
            if not allow_self_loops:
                bad_mask |= rows == cols
            bad = np.flatnonzero(bad_mask)
            n_bad = len(bad)
            if n_bad == 0:
                break
            partners = (bg.random_raw(n_bad) % _U64(nnz)).astype(np.int64)
            ok = ~bad_mask[partners]
            first = np.zeros(n_bad, bool)
            first[np.unique(partners, return_index=True)[1]] = True
            ok &= first
            b, pp = bad[ok], partners[ok]
            tmp = cols[b].copy()
            cols[b] = cols[pp]
            cols[pp] = tmp
        meta = dict(self.meta)
        meta["shuffle"] = {"seed_tag": "shuffle", "rounds": rounds, "unrepaired_pairs": int(n_bad)}
        net = Network.from_pairs(rows, cols, self.counts, n, ids=self.ids, meta=meta)
        net.dead = self.dead.copy()
        return net


def lesion(network: Network, neuron_ids) -> Network:
    return network.lesion(neuron_ids)


def shuffled_copy(network: Network, seed, **kw) -> Network:
    return network.shuffled_copy(seed, **kw)


def synthetic_graph(n: int, n_synapses: int, seed: str = "synthetic", frac_type1: float = 0.8,
                    sign_map: Optional[Mapping[int, int]] = None, batch: int = 5_000_000) -> Network:
    """A random graph for tests and benchmarks: n_synapses synapses with uniform random pre and
    post (self-pairs allowed), each typed 1 with probability frac_type1, else 2. Repeated pairs are
    summed into one pair. sign_map defaults to {1: +1, 2: -1} for synthetic use only."""
    sign_map = {1: 1, 2: -1} if sign_map is None else dict(sign_map)
    bg = make_bitgen(seed, "synthetic")
    thr = _U64(min(int(frac_type1 * 2.0 ** 64), 2 ** 64 - 1))
    pre_l, post_l, sign_l = [], [], []
    s1, s2 = np.int8(sign_map[1]), np.int8(sign_map[2])
    left = n_synapses
    while left > 0:
        k = min(batch, left)
        pre_l.append((bg.random_raw(k) % _U64(n)).astype(np.int32))
        post_l.append((bg.random_raw(k) % _U64(n)).astype(np.int32))
        sign_l.append(np.where(bg.random_raw(k) < thr, s1, s2).astype(np.int8))
        left -= k
    pre = np.concatenate(pre_l); del pre_l
    post = np.concatenate(post_l); del post_l
    sign = np.concatenate(sign_l); del sign_l
    m = sp.coo_matrix((sign.astype(np.int32), (pre, post)), shape=(n, n))
    del pre, post, sign
    m = m.tocsr()
    m.sum_duplicates(); m.eliminate_zeros(); m.sort_indices()
    meta = {"synthetic": {"n": n, "n_synapses": n_synapses, "seed": seed, "frac_type1": frac_type1},
            "sign_map": {str(k): int(v) for k, v in sign_map.items()}}
    return Network(np.arange(n, dtype=np.uint64), m.indptr, m.indices, m.data, meta=meta)


# ---------------------------------------------------------------------------------------------
# rasters

class Raster(NamedTuple):
    steps: np.ndarray      # int64, absolute step of each spike
    neurons: np.ndarray    # int64, internal index of each spike
    start_step: int        # first step covered
    n_steps: int           # number of steps covered
    n_neurons: int
    dt_ms: float

    def counts(self) -> np.ndarray:
        return np.bincount(self.neurons, minlength=self.n_neurons)

    def rates_hz(self) -> np.ndarray:
        return self.counts() / (self.n_steps * self.dt_ms / 1000.0)


def spike_hash(spikes, n_neurons: Optional[int] = None) -> str:
    """sha256 of a spike raster.

    Accepts a Raster, or a list with one array of fired indices per step (step = list position).
    Encoding: b"fishbrain-raster-v1", then little-endian u64 n_neurons, start_step, n_steps,
    n_spikes, then all spike steps as int64 and all neuron indices as int64, sorted by (step, index).
    """
    if isinstance(spikes, Raster):
        steps, neurons = spikes.steps, spikes.neurons
        start, n_steps, nn = spikes.start_step, spikes.n_steps, spikes.n_neurons
    else:
        arrs = [np.asarray(a, dtype=np.int64) for a in spikes]
        steps = np.repeat(np.arange(len(arrs), dtype=np.int64), [len(a) for a in arrs])
        neurons = np.concatenate(arrs) if arrs else np.zeros(0, np.int64)
        start, n_steps = 0, len(arrs)
        nn = n_neurons if n_neurons is not None else (int(neurons.max()) + 1 if len(neurons) else 0)
    steps = np.asarray(steps, np.int64)
    neurons = np.asarray(neurons, np.int64)
    order = np.lexsort((neurons, steps))
    h = hashlib.sha256(b"fishbrain-raster-v1")
    h.update(struct.pack("<QQQQ", int(nn), int(start), int(n_steps), len(steps)))
    h.update(steps[order].astype("<i8").tobytes())
    h.update(neurons[order].astype("<i8").tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# inputs

def _geometric_tables(p: float):
    """Q[k] = (1-p)**(2**k) by repeated squaring (IEEE float64 multiplications only)."""
    q = 1.0 - p
    Q = [q]
    for _ in range(62):
        Q.append(Q[-1] * Q[-1])
    return [(k, Qk) for k, Qk in enumerate(Q) if Qk > 0.0][::-1]


def _geometric(u: np.ndarray, table) -> np.ndarray:
    """G = max{j >= 0 : (1-p)**j >= u} for u in (0, 1]: failures before the first success."""
    G = np.zeros(len(u), np.int64)
    prod = np.ones(len(u), np.float64)
    for k, Qk in table:
        cand = prod * Qk
        ok = cand >= u
        prod = np.where(ok, cand, prod)
        G += ok.astype(np.int64) << np.int64(k)
    return G


def _uniform_open0(raw: np.ndarray) -> np.ndarray:
    """Uniform in (0, 1] from raw uint64: ((raw >> 11) + 1) * 2**-53, exact in float64."""
    return ((raw >> _U64(11)).astype(np.float64) + 1.0) * (2.0 ** -53)


def _bernoulli_threshold(p: np.ndarray) -> np.ndarray:
    """uint64 thresholds so that P(raw < thr) = floor(p * 2**64) / 2**64."""
    p = np.asarray(p, dtype=np.float64)
    thr = np.zeros(p.shape, dtype=np.uint64)
    lt = (p < 1.0) & (p > 0.0)
    thr[lt] = np.floor(p[lt] * 2.0 ** 64).astype(np.uint64)
    thr[p >= 1.0] = np.iinfo(np.uint64).max
    return thr


class _Source:
    kind = "?"

    def __init__(self, name, targets, start, stop):
        self.name, self.targets, self.start, self.stop = name, targets, start, stop

    def active(self, step):
        return step >= self.start and (self.stop is None or step < self.stop)


class _PoissonScalar(_Source):
    """One constant rate for a group. Events are generated on the (step, neuron) lattice with
    geometric gaps, so cost scales with the number of events, not the group size."""
    kind = "poisson"
    BATCH = 4096

    def __init__(self, name, targets, p, weight, start, stop, seed):
        super().__init__(name, targets, start, stop)
        self.K = len(targets)
        self.p = float(p)
        self.weight = weight
        self.bg = make_bitgen(seed, "poisson", name)
        self.table = _geometric_tables(self.p) if 0.0 < self.p < 1.0 else None
        self.buf = np.zeros(0, np.int64)
        self.last = -1

    def hits(self, step):
        if self.p <= 0.0 or self.K == 0:
            return None
        if self.p >= 1.0:
            return self.targets
        rel = step - self.start
        lo, hi = rel * self.K, (rel + 1) * self.K
        while self.last < hi:
            G = _geometric(_uniform_open0(self.bg.random_raw(self.BATCH)), self.table)
            pos = self.last + np.cumsum(G + 1)
            self.last = int(pos[-1])
            self.buf = np.concatenate([self.buf, pos])
        j = int(np.searchsorted(self.buf, hi, "left"))
        ev = self.buf[:j]
        self.buf = self.buf[j:]
        ev = ev[ev >= lo]
        if not len(ev):
            return None
        return self.targets[ev - lo]


class _PoissonArray(_Source):
    """Per-neuron rates, constant (K,) or one row per frame (F, K). One raw draw per neuron per
    step, compared against an integer threshold."""
    kind = "poisson"

    def __init__(self, name, targets, rates_hz, dt_ms, frame_steps, weight, start, stop, seed):
        super().__init__(name, targets, start, stop)
        self.rates = rates_hz          # (F, K) float64
        self.dt_ms = dt_ms
        self.frame_steps = frame_steps
        self.weight = weight
        self.bg = make_bitgen(seed, "poisson", name)
        self._f, self._thr = -1, None

    def hits(self, step):
        f = (step - self.start) // self.frame_steps
        if f >= len(self.rates):
            return None
        if f != self._f:
            self._f, self._thr = f, _bernoulli_threshold(self.rates[f] * self.dt_ms / 1000.0)
        raw = self.bg.random_raw(len(self.targets))
        h = raw < self._thr
        return self.targets[h] if h.any() else None


class _Current(_Source):
    """A constant-per-frame drive I (mV, i.e. R*I) in dv/dt = (v0 - v + g + I)/tau_m."""
    kind = "current"

    def __init__(self, name, targets, amps_mV, frame_steps, k_i64, start, stop):
        super().__init__(name, targets, start, stop)
        self.inc = (np.asarray(amps_mV, np.float64) * k_i64).astype(np.float32)  # (F, K)
        self.frame_steps = frame_steps

    def add(self, step, u):
        f = (step - self.start) // self.frame_steps
        if f < len(self.inc):
            u[self.targets] += self.inc[f]


# ---------------------------------------------------------------------------------------------
# simulator

class Simulator:
    """Event-driven LIF network. Only the rows of neurons that spiked are propagated.

    sim = Simulator(net, seed="0x<block hash>")
    sim.add_poisson(idx, 40.0, start_ms=0, stop_ms=500)          # constant rate, sparse generator
    sim.add_poisson(idx, rates_FxK, frame_ms=1.0)                  # time-varying per-neuron rates
    sim.add_current(idx, 10.0, start_ms=100, stop_ms=200)          # mV of drive
    r = sim.run(1000.0)                                            # Raster of this window
    spike_hash(r)
    """

    def __init__(self, network: Network, seed: Union[str, bytes, int], params: LIFParams = LIFParams()):
        self.net = network
        self.p = params
        self.seed = seed if isinstance(seed, str) else (seed.decode() if isinstance(seed, bytes) else str(seed))
        d = params.derived()
        self._a, self._b, self._c = d["a"], d["b"], d["c"]
        self._k_i64 = d["k_i64"]
        self._theta, self._u_reset = d["theta"], d["u_reset"]
        self.n_ref, self.n_delay = d["n_ref"], d["n_delay"]
        n = network.n
        self._indptr = network.indptr
        self._indices = network.indices
        # weight in h units: signed count * w_syn * c, rounded once to float32
        self._w = (network.counts.astype(np.float64) * (params.w_syn_mV * d["c64"])).astype(np.float32)
        self._dead = network.dead
        self._any_dead = bool(network.dead.any())
        self._poisson_w = np.float32(params.poisson_weight_mV)
        self.u = np.zeros(n, np.float32)            # v - v_rest, mV
        self.h = np.zeros(n, np.float32)            # c * g
        self.last_spike = np.full(n, -(2 ** 62), np.int64)
        self.step = 0
        self._slots = [np.zeros(0, np.int64) for _ in range(self.n_delay + 1)]
        self._ref = deque(maxlen=max(self.n_ref - 1, 0)) if self.n_ref > 1 else None
        self._mask = np.empty(n, bool)
        self.sources: List[_Source] = []
        self._rec_steps: List[int] = []
        self._rec_idx: List[np.ndarray] = []
        self._rec_start = 0
        self.stats = {"spikes": 0, "syn_events": 0, "poisson_events": 0}

    # -- inputs --------------------------------------------------------------------------------
    def _targets(self, neurons) -> np.ndarray:
        t = np.atleast_1d(np.asarray(neurons, dtype=np.int64))
        if len(t) and (t.min() < 0 or t.max() >= self.net.n):
            raise IndexError("neuron index out of range")
        if len(np.unique(t)) != len(t):
            raise ValueError("duplicate neuron indices in one source")
        return t

    def _window(self, start_ms, stop_ms):
        start = self.p.steps(start_ms)
        stop = None if stop_ms is None else self.p.steps(stop_ms)
        return start, stop

    def _name(self, name, kind):
        name = name or f"{kind}{len(self.sources)}"
        if any(s.name == name for s in self.sources):
            raise ValueError(f"source name {name!r} already used")
        return name

    def add_poisson(self, neurons, rate_hz, start_ms: float = 0.0, stop_ms: Optional[float] = None,
                    frame_ms: Optional[float] = None, weight_mV: Optional[float] = None,
                    name: Optional[str] = None) -> str:
        """Poisson drive into internal indices `neurons`. Each event adds weight_mV (default
        params.poisson_weight_mV = 68.75) to v, as Shiu's PoissonInput does.

        rate_hz: a scalar (one rate for the group), a (K,) array (per neuron), or an (F, K) array
        of per-frame rates with frame_ms (the source ends after F frames unless stop_ms is sooner).
        The source draws from its own stream sha256(seed | "poisson" | name); name defaults to
        its position, "poisson<k>". Returns the name.
        """
        t = self._targets(neurons)
        start, stop = self._window(start_ms, stop_ms)
        name = self._name(name, "poisson")
        w = self._poisson_w if weight_mV is None else np.float32(weight_mV)
        r = np.asarray(rate_hz, dtype=np.float64)
        if np.any(r < 0):
            raise ValueError("rates must be >= 0")
        if r.ndim == 0:
            p = float(r) * self.p.dt_ms / 1000.0
            if 0.0 < p < 2.0 ** -40:
                p = 0.0
            src = _PoissonScalar(name, t, p, w, start, stop, self.seed)
        else:
            if r.ndim == 1:
                r = r[None, :]
                fs = 1 << 62
            else:
                if frame_ms is None:
                    raise ValueError("an (F, K) rate array needs frame_ms")
                fs = self.p.steps(frame_ms)
            if r.shape[1] != len(t):
                raise ValueError(f"rates have {r.shape[1]} columns for {len(t)} neurons")
            src = _PoissonArray(name, t, r, self.p.dt_ms, fs, w, start, stop, self.seed)
        self.sources.append(src)
        return name

    def add_current(self, neurons, amp_mV, start_ms: float = 0.0, stop_ms: Optional[float] = None,
                    frame_ms: Optional[float] = None, name: Optional[str] = None) -> str:
        """Constant or per-frame drive in mV (R*I) added to dv/dt: scalar, (K,), or (F, K) with
        frame_ms. Refractory neurons do not integrate it. Returns the source name."""
        t = self._targets(neurons)
        start, stop = self._window(start_ms, stop_ms)
        name = self._name(name, "current")
        a = np.asarray(amp_mV, dtype=np.float64)
        if a.ndim == 0:
            a = np.full((1, len(t)), float(a))
            fs = 1 << 62
        elif a.ndim == 1:
            a = a[None, :]
            fs = 1 << 62
        else:
            if frame_ms is None:
                raise ValueError("an (F, K) current array needs frame_ms")
            fs = self.p.steps(frame_ms)
        if a.shape[1] != len(t):
            raise ValueError(f"currents have {a.shape[1]} columns for {len(t)} neurons")
        self.sources.append(_Current(name, t, a, fs, self._k_i64, start, stop))
        return name

    def clear_inputs(self):
        self.sources = []

    # -- dynamics ------------------------------------------------------------------------------
    def _deliver(self, src: np.ndarray):
        indptr, h = self._indptr, self.h
        if len(src) == 1:
            s, e = indptr[src[0]], indptr[src[0] + 1]
            if e > s:
                h[self._indices[s:e]] += self._w[s:e]      # one row: targets are unique
                self.stats["syn_events"] += int(e - s)
            return
        starts = indptr[src]
        cnt = indptr[src + 1] - starts
        tot = int(cnt.sum())
        if not tot:
            return
        csum = np.cumsum(cnt)
        g = np.arange(tot, dtype=np.int64) + np.repeat(starts - (csum - cnt), cnt)
        np.add.at(h, self._indices[g], self._w[g])
        self.stats["syn_events"] += tot

    def run(self, duration_ms: Optional[float] = None, n_steps: Optional[int] = None) -> Raster:
        """Advance the network; returns the raster of this window."""
        if n_steps is None:
            n_steps = self.p.steps(duration_ms)
        u, h, last = self.u, self.h, self.last_spike
        a, b, theta, u_reset = self._a, self._b, self._theta, self._u_reset
        n_ref, n_delay, slots, ref = self.n_ref, self.n_delay, self._slots, self._ref
        mask, dead, any_dead = self._mask, self._dead, self._any_dead
        nslot = n_delay + 1
        w0_steps = len(self._rec_steps)
        first = self.step
        currents = [s for s in self.sources if s.kind == "current"]
        poissons = [s for s in self.sources if s.kind == "poisson"]
        for _ in range(n_steps):
            k = self.step
            # 1. integrate (refractory neurons are frozen: save, update all, restore)
            if ref:
                R = np.concatenate(ref)
                uR, hR = u[R], h[R]
            else:
                R = None
            np.multiply(u, a, out=u)
            np.add(u, h, out=u)
            np.multiply(h, b, out=h)
            for s in currents:
                if s.active(k):
                    s.add(k, u)
            if R is not None and len(R):
                u[R] = uR
                h[R] = hR
            # 2. threshold
            np.greater(u, theta, out=mask)
            fired = np.flatnonzero(mask)
            if len(fired):
                fired = fired[last[fired] <= k - n_ref]
                if any_dead and len(fired):
                    fired = fired[~dead[fired]]
            # 3. synapses: delayed delivery, then Poisson kicks
            slots[k % nslot] = fired
            arriving = slots[(k - n_delay) % nslot]
            if len(arriving):
                self._deliver(arriving)
            for s in poissons:
                if s.active(k):
                    hit = s.hits(k)
                    if hit is not None:
                        u[hit] += s.weight
                        self.stats["poisson_events"] += len(hit)
            # 4. reset
            if len(fired):
                u[fired] = u_reset
                h[fired] = 0.0
                last[fired] = k
                self._rec_steps.append(k)
                self._rec_idx.append(fired)
                self.stats["spikes"] += len(fired)
            if ref is not None:
                ref.append(fired)
            self.step += 1
        steps = self._rec_steps[w0_steps:]
        idx = self._rec_idx[w0_steps:]
        return self._raster(steps, idx, first, n_steps)

    def _raster(self, steps, idx, start, n_steps) -> Raster:
        if idx:
            st = np.repeat(np.asarray(steps, np.int64), [len(i) for i in idx])
            ne = np.concatenate(idx).astype(np.int64)
        else:
            st = np.zeros(0, np.int64)
            ne = np.zeros(0, np.int64)
        return Raster(st, ne, int(start), int(n_steps), self.net.n, self.p.dt_ms)

    def raster(self) -> Raster:
        """Every spike since construction (or the last restore())."""
        return self._raster(self._rec_steps, self._rec_idx, self._rec_start, self.step - self._rec_start)

    # -- state ---------------------------------------------------------------------------------
    def v_mV(self) -> np.ndarray:
        return self.u.astype(np.float64) + self.p.v_rest_mV

    def g_mV(self) -> np.ndarray:
        return self.h.astype(np.float64) / float(self._c)

    def snapshot(self) -> dict:
        """Full dynamic state (voltages, conductances, spike times, in-flight spikes, refractory
        queue, every source with its RNG state). restore() continues identically."""
        return copy.deepcopy({"step": self.step, "u": self.u, "h": self.h, "last": self.last_spike,
                              "slots": self._slots, "ref": list(self._ref) if self._ref is not None else None,
                              "sources": self.sources, "stats": self.stats})

    def restore(self, snap: dict):
        s = copy.deepcopy(snap)
        self.step, self.u, self.h, self.last_spike = s["step"], s["u"], s["h"], s["last"]
        self._slots, self.sources, self.stats = s["slots"], s["sources"], s["stats"]
        if self._ref is not None:
            self._ref = deque(s["ref"], maxlen=self._ref.maxlen)
        self._rec_steps, self._rec_idx, self._rec_start = [], [], self.step

    def state_digest(self) -> str:
        h = hashlib.sha256(b"fishbrain-state-v1")
        h.update(struct.pack("<Q", self.step))
        for arr in (self.u, self.h):
            h.update(arr.astype("<f4").tobytes())
        h.update(self.last_spike.astype("<i8").tobytes())
        for k in range(self.n_delay + 1):   # in-flight spikes, oldest first
            sl = self._slots[(self.step + k) % (self.n_delay + 1)]
            h.update(struct.pack("<Q", len(sl)))
            h.update(np.asarray(sl, "<i8").tobytes())
        for s in self.sources:
            h.update(s.name.encode())
            if hasattr(s, "bg"):
                h.update(json.dumps(s.bg.state, sort_keys=True, default=str).encode())
            if hasattr(s, "buf"):
                h.update(s.buf.astype("<i8").tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# benchmark

def _rss_bytes() -> int:
    import resource, sys
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)   # macOS reports bytes, Linux KiB


def _bench_one(g, label, rate, n, seconds, trials, params, seed) -> dict:
    """Wall time (what a viewer waits) and process CPU time (what one uncontended core would
    need; numpy here is single-threaded) for `trials` runs of `seconds` simulated."""
    import os, time
    walls, cpus, mean_hz, ev = [], [], None, None
    steps = params.steps(seconds * 1000.0)
    for tr in range(trials):
        sim = Simulator(g, seed=f"{seed}-{tr}", params=params)
        if rate > 0:
            sim.add_poisson(np.arange(n), rate, name="background")
        sim.run(50.0)  # warm-up, not timed
        s0 = dict(sim.stats)
        c, t = time.process_time(), time.perf_counter()
        r = sim.run(n_steps=steps)
        walls.append(time.perf_counter() - t)
        cpus.append(time.process_time() - c)
        mean_hz = len(r.neurons) / n / seconds
        ev = (sim.stats["syn_events"] - s0["syn_events"]) / seconds
    walls.sort()
    cpus.sort()
    med = len(walls) // 2
    return {"graph": label, "n": n, "nnz": g.nnz, "poisson_hz": rate, "mean_rate_hz": mean_hz,
            "syn_events_per_s": ev, "wall_s": walls, "cpu_s": cpus,
            "rtf_wall_best": seconds / walls[0], "rtf_wall_median": seconds / walls[med],
            "rtf_cpu_best": seconds / cpus[0], "rtf_cpu_median": seconds / cpus[med],
            "cpu_us_per_step_median": cpus[med] / steps * 1e6, "loadavg_1m": os.getloadavg()[0]}


def scaling(sizes=(20_000, 50_000, 100_000, 190_000), syn_per_neuron=158, rate=3.0, seconds=1.0,
            trials=2, dt_ms=0.1, seed="bench", out=None) -> dict:
    """RTF against population size at a fixed density and background rate."""
    import os, platform
    params = LIFParams(dt_ms=dt_ms)
    rows = []
    for n in sizes:
        net = synthetic_graph(n, n * syn_per_neuron, seed=f"{seed}-{n}")
        rows.append(_bench_one(net, "full", rate, n, seconds, trials, params, seed))
        print(json.dumps(rows[-1]), flush=True)
        del net
    res = {"sizes": list(sizes), "syn_per_neuron": syn_per_neuron, "rate": rate, "dt_ms": dt_ms,
           "machine": platform.platform(), "numpy": np.__version__, "rows": rows, "loadavg_end": os.getloadavg()}
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1, default=float)
    return res


def benchmark(n=190_000, n_synapses=30_000_000, rates=(0.0, 1.0, 3.0, 10.0), seconds=1.0,
              trials=3, dt_ms=0.1, seed="bench", out=None) -> dict:
    """Real-time factor (simulated ms per wall ms) on a random graph, with Poisson background on
    every neuron at each rate; plus the same drive on an edgeless graph of equal size to separate
    the network's cost from the drive's."""
    import os, platform, time
    res = {"n": n, "n_synapses_drawn": n_synapses, "dt_ms": dt_ms, "seconds_simulated": seconds,
           "trials": trials, "machine": platform.platform(), "processor": platform.processor(),
           "numpy": np.__version__, "loadavg_start": os.getloadavg()}
    t0 = time.perf_counter()
    net = synthetic_graph(n, n_synapses, seed=seed)
    res["build_s"] = time.perf_counter() - t0
    res["nnz_pairs"] = net.nnz
    res["network_bytes"] = net.nbytes()
    res["rss_after_build_bytes"] = _rss_bytes()
    empty = Network(net.ids, np.zeros(n + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32))
    params = LIFParams(dt_ms=dt_ms)
    rows = []
    for rate in rates:
        for label, g in (("full", net), ("edgeless", empty)):
            rows.append(_bench_one(g, label, rate, n, seconds, trials, params, seed))
            print(json.dumps(rows[-1]), flush=True)
    res["rows"] = rows
    res["rss_peak_bytes"] = _rss_bytes()
    res["sim_state_bytes"] = int(n * (4 + 4 + 8 + 1)) + int(net.nnz * 4)
    res["loadavg_end"] = os.getloadavg()
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1, default=float)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FISHBRAIN simulator benchmark")
    ap.add_argument("--n", type=int, default=190_000)
    ap.add_argument("--synapses", type=int, default=30_000_000)
    ap.add_argument("--rates", type=str, default="0,1,3,10")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--scaling", type=str, default=None, help="comma-separated sizes: run scaling() instead")
    a = ap.parse_args()
    if a.scaling:
        scaling(tuple(int(x) for x in a.scaling.split(",")), rate=float(a.rates.split(",")[-1]),
                seconds=a.seconds, trials=a.trials, dt_ms=a.dt, out=a.out)
        raise SystemExit(0)
    r = benchmark(a.n, a.synapses, tuple(float(x) for x in a.rates.split(",")), a.seconds, a.trials, a.dt, out=a.out)
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, default=float))
