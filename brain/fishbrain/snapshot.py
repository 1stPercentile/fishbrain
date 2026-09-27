"""Serialized checkpoints, canonical inputs and the seed rule: what Verify needs from the engine.

`sim.Simulator.snapshot()` / `restore()` are in-process deep copies with no wire format. This module
gives the engine's state a canonical byte format, so a browser can resume the brain at a trade's
window instead of replaying from step 0, and it pins the two things a receipt must bind that the
simulator itself does not: the complete input specification and the seed string.

The JS kernel (js/fishbrain-sim.mjs: encodeInputs, checkpointBytes, parseCheckpoint,
restoreCheckpoint) writes and reads the same bytes; tests/test_verify.py requires both sides to write
identical bytes for the same state. This file is the reference for the format.

Seed rule (fishbrain-seed-v1)
-----------------------------
A receipt's seed is `{"source": "solana blockhash", "value": <blockhash>}`. The simulation seed string
is `seed_string(value)` = "fishbrain-seed-v1:solana-blockhash:" + value, where value must be the
canonical base58 of exactly 32 bytes (decode, check the length, re-encode, compare) with at least 8
distinct byte values (all-zero "1"*32 and other degenerate values are refused). The engine is
constructed as `Simulator(net, seed=seed_string(blockhash), ...)`; every Poisson stream is then
sha256(seed_string | "poisson" | name) as sim.py already does. Verify derives the string from the
receipt and never takes a seed from a case file.

A format rule cannot stop seed grinding: any well-formed value could have been picked. The seed is
unpickable only when Verify checks, over a Solana RPC or a supplied proof, that it is the real
blockhash of the slot pinned by a run-commit Memo that landed before that slot
(`run_commit_memo`; js/verify.mjs `opts.chainLookup`; without one a result says seed_anchored: false).

fishbrain-inputs-v1 (the complete input specification; little-endian throughout)
--------------------------------------------------------------------------------
What the engine uses after normalisation, source by source in the order they were added:

    magic   b"fishbrain-inputs-v1\\n"                       20 bytes
    u32     number of sources
    per source:
      u32 + bytes   name (UTF-8)
      u8            kind: 1 Poisson scalar rate, 2 Poisson per-neuron/per-frame rates, 3 current
      i64           start step
      i64           stop step, or -1 for none (a negative stop is refused, so -1 is unambiguous)
      u64 K, K x i64   target neuron indices, in the source's order
      kind 1: f64 p (per-step event probability, after sim.py's 2^-40 flush), f32 weight (mV)
      kind 2: u64 F, u64 frame_steps (2^62 for a constant (K,) rate), f32 weight (mV),
              F*K x f64 rates (Hz, row-major (F, K))
      kind 3: u64 F, u64 frame_steps, F*K x f32 inc (mV added per step = float32(amp * k_i64))

`inputs_digest(sim)` = sha256 of those bytes. It covers every source's name, kind, targets, rates,
frame length, weight, start/stop and currents, so an extra hidden source, a changed rate byte or a
moved window all change it. (The receipt field for it is proposed as `inputs.spec`.)

fishbrain-checkpoint-v1 (one simulator state)
---------------------------------------------
    magic   b"fishbrain-checkpoint-v1\\n"                   24 bytes
    32      Network.digest()        (raw bytes)
    32      LIFParams.digest()
    32      inputs digest (fishbrain-inputs-v1)
    u32 + bytes   seed string (UTF-8; Simulator.seed)
    u64 n, u32 n_delay, u32 n_ref, u64 step   (step = the next step run() will compute)
    n x f32 u, n x f32 h, n x i64 last spike step (-2^62 = never)
    n_delay + 1 slots, oldest first (slot j holds the spikes of step - n_delay - 1 + j; it is the
            slot state_digest() reads j-th): u64 count, count x i64 neuron indices (ascending)
    u32 R, then R refractory-queue entries oldest first (u64 count, count x i64); R = min(step, n_ref - 1)
            when n_ref > 1, else 0
    u32 number of sources, per source in order:
      u32 + bytes name, u8 kind (as in the inputs)
      kind 1 and 2: u128 PCG64 state, u128 PCG64 increment (16 bytes each, little-endian)
      kind 1: i64 last generated event position (-1 before the first batch),
              u64 count, count x i64 pending event positions (ascending)
    u8      1 if the inputs are embedded, else 0; if 1: u64 length + the fishbrain-inputs-v1 bytes
    32      state_digest() of this state (raw)
    32      sha256 of every byte above (a corruption check, not a security boundary: anyone can
            recompute it; the receipt's window.snapshot is what binds a checkpoint)

Every byte is either covered by state_digest() or checked against something that is, so no field
can steer a replay while the digest stays the same:
  * the refractory queue is not in state_digest(); it must equal the queue derived from `last`
    (refractoriness means a neuron appears at most once in it, at its last spike);
  * a scalar source's `last` is not in state_digest(); it must equal the last pending position
    (or -1 with nothing pending);
  * each delay slot must hold every neuron whose last spike was that slot's step;
  * PCG64 `has_uint32` and `uinteger` must be 0 (the engine only uses random_raw);
  * `stats` and the per-frame threshold cache are not state and are not stored (restored as zero /
    recomputed);
  * the recorded digest must equal the digest recomputed from the parsed state, and no trailing
    bytes are allowed.

What a checkpoint cannot prove by itself is that its state is the one the run really reached: a
fabricated u/h with every hash resealed is self-consistent. js/verify.mjs therefore refuses a window
past step 0 (check "anchor") unless its checkpoint is anchored by a verified chain of links from
genesis or from a trusted checkpoint, or is itself a trusted checkpoint the page has checked; see
js/README.md, "Verify: the checkpoint trust chain". Note that state_digest() does not cover rates,
weights, targets or start/stop, so an anchor for a state must also name the inputs digest.
"""
from __future__ import annotations

import hashlib
import re
import struct
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np

from . import sim as S

INPUTS_MAGIC = b"fishbrain-inputs-v1\n"
CHECKPOINT_MAGIC = b"fishbrain-checkpoint-v1\n"
SEED_PREFIX = "fishbrain-seed-v1:solana-blockhash:"
SEED_SOURCE = "solana blockhash"
KIND_SCALAR, KIND_ARRAY, KIND_CURRENT = 1, 2, 3
NEVER = -(2 ** 62)
CONST_FRAME_STEPS = 1 << 62
KERNEL_PATH = Path(__file__).resolve().parents[1] / "js" / "fishbrain-sim.mjs"
_MAX_SAFE = 2 ** 53 - 1


class CheckpointError(ValueError):
    """A checkpoint or inputs blob that is malformed, inconsistent, or for another brain."""


# ---------------------------------------------------------------------------------------------
# seed rule

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58)}


def b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = _B58[r] + out
    pad = len(data) - len(data.lstrip(b"\x00"))
    return "1" * pad + out


def b58decode(s: str) -> bytes:
    n = 0
    for c in s:
        if c not in _B58_INDEX:
            raise ValueError(f"not base58: {c!r}")
        n = n * 58 + _B58_INDEX[c]
    pad = len(s) - len(s.lstrip("1"))
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    return b"\x00" * pad + body


MIN_DISTINCT_BYTES = 8


def seed_string(blockhash: str) -> str:
    """The simulation seed string for a Solana blockhash (fishbrain-seed-v1). The value must be the
    canonical base58 encoding of exactly 32 bytes, and not degenerate: a blockhash is a SHA-256
    output, so fewer than MIN_DISTINCT_BYTES distinct byte values (all zero, all 0xff, a short
    repeated pattern) is refused; a real one does that with probability below 1e-37.

    A well-formed value is still one the operator could have picked. Only a check that it is the
    real blockhash of the slot the run-commit memo pinned (js/verify.mjs, opts.chainLookup) makes
    the seed unpickable; no format rule can."""
    if not isinstance(blockhash, str) or not 32 <= len(blockhash) <= 44:
        raise ValueError("a blockhash is 32 to 44 base58 characters")
    raw = b58decode(blockhash)
    if len(raw) != 32 or b58encode(raw) != blockhash:
        raise ValueError("not the canonical base58 of 32 bytes")
    if len(set(raw)) < MIN_DISTINCT_BYTES:
        raise ValueError(f"a degenerate blockhash ({len(set(raw))} distinct byte values; a real one is a SHA-256 output)")
    return SEED_PREFIX + blockhash


RUN_MEMO_TAG = "fishbrain:run:v1"


def run_commit_memo(network: str, params: str, kernel_sha256: str, seed_slot: int) -> str:
    """The run-commit Memo text (fishbrain:run:v1): the brain and the slot whose blockhash will seed
    it, published before that slot exists. js/verify.mjs reads it through opts.chainLookup."""
    for k, v in (("network", network), ("params", params), ("kernel", kernel_sha256)):
        if not re.fullmatch(r"[0-9a-f]{64}", v or ""):
            raise ValueError(f"{k} must be a sha256 hex digest")
    if not (isinstance(seed_slot, int) and 0 <= seed_slot < 10 ** 16):
        raise ValueError("seed_slot must be a slot number")
    return f"{RUN_MEMO_TAG} network={network} params={params} kernel={kernel_sha256} seed_slot={seed_slot}"


# ---------------------------------------------------------------------------------------------
# canonical inputs

def _kind(src) -> int:
    if isinstance(src, S._PoissonScalar):
        return KIND_SCALAR
    if isinstance(src, S._PoissonArray):
        return KIND_ARRAY
    if isinstance(src, S._Current):
        return KIND_CURRENT
    raise TypeError(f"unknown source type {type(src).__name__}")


def _pack_str(out: bytearray, s: str):
    b = s.encode("utf-8")
    out += struct.pack("<I", len(b))
    out += b


def encode_inputs(sim: S.Simulator) -> bytes:
    """The fishbrain-inputs-v1 bytes of every source in `sim`, in order."""
    out = bytearray(INPUTS_MAGIC)
    out += struct.pack("<I", len(sim.sources))
    for s in sim.sources:
        k = _kind(s)
        if s.stop is not None and s.stop < 0:
            raise ValueError(f"source {s.name!r}: a negative stop step has no canonical encoding")
        _pack_str(out, s.name)
        out += struct.pack("<Bqq", k, int(s.start), -1 if s.stop is None else int(s.stop))
        t = np.ascontiguousarray(s.targets, "<i8")
        out += struct.pack("<Q", len(t))
        out += t.tobytes()
        if k == KIND_SCALAR:
            out += struct.pack("<d", float(s.p))
            out += np.asarray([s.weight], "<f4").tobytes()
        elif k == KIND_ARRAY:
            r = np.ascontiguousarray(s.rates, "<f8")
            if r.ndim != 2 or r.shape[1] != len(t):
                raise ValueError(f"source {s.name!r}: rates are not (F, K)")
            out += struct.pack("<QQ", r.shape[0], int(s.frame_steps))
            out += np.asarray([s.weight], "<f4").tobytes()
            out += r.tobytes()
        else:
            inc = np.ascontiguousarray(s.inc, "<f4")
            if inc.ndim != 2 or inc.shape[1] != len(t):
                raise ValueError(f"source {s.name!r}: currents are not (F, K)")
            out += struct.pack("<QQ", inc.shape[0], int(s.frame_steps))
            out += inc.tobytes()
    return bytes(out)


def inputs_digest(sim: S.Simulator) -> str:
    return hashlib.sha256(encode_inputs(sim)).hexdigest()


def frames_digest(sim: S.Simulator, name: str = "retina") -> Optional[str]:
    """sha256 of the named per-frame Poisson source's rates (float64 little-endian, (F, K)): what the
    fish saw. It equals the G1 gate's recorded `rates_sha256` for the retina. None if absent."""
    for s in sim.sources:
        if s.name == name and isinstance(s, S._PoissonArray):
            return hashlib.sha256(np.ascontiguousarray(s.rates, "<f8").tobytes()).hexdigest()
    return None


class _Reader:
    def __init__(self, data: bytes, what: str):
        self.b, self.o, self.what = memoryview(data), 0, what

    def need(self, k: int):
        if k < 0 or self.o + k > len(self.b):
            raise CheckpointError(f"{self.what}: truncated")

    def take(self, k: int) -> bytes:
        self.need(k)
        v = bytes(self.b[self.o:self.o + k])
        self.o += k
        return v

    def unpack(self, fmt: str):
        k = struct.calcsize(fmt)
        v = struct.unpack(fmt, self.take(k))
        return v if len(v) > 1 else v[0]

    def array(self, count: int, dtype: str) -> np.ndarray:
        size = np.dtype(dtype).itemsize
        if count < 0 or count > (len(self.b) - self.o) // size:
            raise CheckpointError(f"{self.what}: truncated")
        return np.frombuffer(self.take(count * size), dtype=dtype).copy()

    def text(self) -> str:
        n = self.unpack("<I")
        try:
            return self.take(n).decode("utf-8")
        except UnicodeDecodeError as e:
            raise CheckpointError(f"{self.what}: a name is not UTF-8") from e

    def done(self):
        if self.o != len(self.b):
            raise CheckpointError(f"{self.what}: {len(self.b) - self.o} trailing bytes")


def _frame_steps(x: int, what: str) -> int:
    if not (1 <= x <= _MAX_SAFE or x == CONST_FRAME_STEPS):
        raise CheckpointError(f"{what}: frame_steps {x} is out of range")
    return x


def decode_inputs(data: bytes) -> List[dict]:
    """Parse fishbrain-inputs-v1 bytes into source specs (no network needed)."""
    r = _Reader(data, "inputs")
    if r.take(len(INPUTS_MAGIC)) != INPUTS_MAGIC:
        raise CheckpointError("inputs: not fishbrain-inputs-v1")
    specs, names = [], set()
    for _ in range(r.unpack("<I")):
        name = r.text()
        if name in names:
            raise CheckpointError(f"inputs: source name {name!r} repeated")
        names.add(name)
        kind, start, stop = r.unpack("<Bqq")
        if kind not in (KIND_SCALAR, KIND_ARRAY, KIND_CURRENT):
            raise CheckpointError(f"inputs: unknown source kind {kind}")
        if stop < -1:
            raise CheckpointError(f"inputs: source {name!r} has a negative stop")
        if abs(start) > _MAX_SAFE or stop > _MAX_SAFE:
            raise CheckpointError(f"inputs: source {name!r} window out of range")
        K = r.unpack("<Q")
        t = r.array(K, "<i8").astype(np.int64)
        sp = {"name": name, "kind": kind, "start": start, "stop": None if stop == -1 else stop, "targets": t}
        if kind == KIND_SCALAR:
            p = r.unpack("<d")
            w = r.array(1, "<f4")[0]
            if not (np.isfinite(p) and p >= 0.0 and np.isfinite(w)):
                raise CheckpointError(f"inputs: source {name!r} has a bad rate or weight")
            sp.update(p=float(p), weight=np.float32(w))
        elif kind == KIND_ARRAY:
            F, fs = r.unpack("<QQ")
            w = r.array(1, "<f4")[0]
            rates = r.array(F * K, "<f8").reshape(F, K)
            if not (np.isfinite(w) and np.all(np.isfinite(rates)) and np.all(rates >= 0.0)):
                raise CheckpointError(f"inputs: source {name!r} has bad rates or weight")
            sp.update(frame_steps=_frame_steps(fs, "inputs"), weight=np.float32(w), rates=rates.astype(np.float64))
        else:
            F, fs = r.unpack("<QQ")
            inc = r.array(F * K, "<f4").reshape(F, K)
            if not np.all(np.isfinite(inc)):
                raise CheckpointError(f"inputs: source {name!r} has non-finite currents")
            sp.update(frame_steps=_frame_steps(fs, "inputs"), inc=inc.astype(np.float32))
        specs.append(sp)
    r.done()
    return specs


def _source_from_spec(sp: dict, seed: str, params: S.LIFParams, n: int):
    t = sp["targets"]
    if len(t) and (t.min() < 0 or t.max() >= n):
        raise CheckpointError(f"inputs: source {sp['name']!r} targets a neuron outside the network")
    if len(np.unique(t)) != len(t):
        raise CheckpointError(f"inputs: source {sp['name']!r} targets a neuron twice")
    k = sp["kind"]
    if k == KIND_SCALAR:
        return S._PoissonScalar(sp["name"], t, sp["p"], sp["weight"], sp["start"], sp["stop"], seed)
    if k == KIND_ARRAY:
        return S._PoissonArray(sp["name"], t, sp["rates"], params.dt_ms, sp["frame_steps"], sp["weight"],
                               sp["start"], sp["stop"], seed)
    src = S._Current(sp["name"], t, np.zeros(sp["inc"].shape), sp["frame_steps"], 0.0, sp["start"], sp["stop"])
    src.inc = sp["inc"].copy()
    return src


def sources_from_inputs(data: bytes, seed: str, params: S.LIFParams, n: int) -> list:
    """Fresh source objects (random streams at their seeded start) from fishbrain-inputs-v1 bytes."""
    return [_source_from_spec(sp, seed, params, n) for sp in decode_inputs(data)]


# ---------------------------------------------------------------------------------------------
# random streams without the network

def fast_forward_sources(sources: Sequence, step: int):
    """Advance fresh sources to where a run from step 0 leaves them at `step`, without the network.

    A source's random stream never depends on the neurons: a per-frame source draws exactly K raw
    numbers on every active step inside its frames, and a scalar source generates 4096-draw batches
    until its last event position passes the step's lattice end. So the stream position at `step`
    is a function of (seed, inputs, step) alone, and a checkpoint's streams can be checked against it.

    `Simulator.run` computes steps 0, 1, 2, ... only. A source whose start is negative (it began
    before the run) is active from step 0, so its draws are counted from max(start, 0): its active
    steps are [max(start, 0), min(stop, end of its frames)). Counting from `start` would demand
    draws the simulator never made (tests/test_verify.py checks this against real runs).
    """
    for s in sources:
        if isinstance(s, S._Current):
            continue
        end = step if s.stop is None else min(step, s.stop)
        first = max(int(s.start), 0)   # the first step a run computes with the source active
        if isinstance(s, S._PoissonScalar):
            if s.p <= 0.0 or s.p >= 1.0 or s.K == 0:
                continue
            if end - 1 >= first:
                s.hits(end - 1)        # the batch loop is monotone: one call equals the stepwise calls
        else:
            K = len(s.targets)
            lim = s.start + len(s.rates) * s.frame_steps
            count = max(0, min(end, lim) - first)
            if K and count:
                s.bg.advance(count * K)
                s._f = -1


def source_records(sources: Sequence) -> list:
    """(name, kind, PCG64 state, increment, last, pending positions) per source, for comparisons."""
    out = []
    for s in sources:
        k = _kind(s)
        st = s.bg.state["state"] if k != KIND_CURRENT else None
        out.append((s.name, k, None if st is None else int(st["state"]), None if st is None else int(st["inc"]),
                    int(s.last) if k == KIND_SCALAR else None,
                    tuple(int(x) for x in s.buf) if k == KIND_SCALAR else None))
    return out


# ---------------------------------------------------------------------------------------------
# checkpoints

@dataclass
class SourceState:
    name: str
    kind: int
    state: Optional[int] = None          # PCG64 128-bit state (kinds 1, 2)
    inc: Optional[int] = None            # PCG64 increment (kinds 1, 2)
    last: Optional[int] = None           # kind 1: last generated event position
    buf: Optional[np.ndarray] = None     # kind 1: pending event positions, int64


@dataclass
class Checkpoint:
    network_digest: str
    params_digest: str
    inputs_digest: str
    seed: str
    n: int
    n_delay: int
    n_ref: int
    step: int
    u: np.ndarray                        # float32
    h: np.ndarray                        # float32
    last: np.ndarray                     # int64
    slots: List[np.ndarray]              # n_delay + 1, oldest first
    ref: List[np.ndarray]                # refractory queue, oldest first
    sources: List[SourceState]
    inputs: Optional[bytes]              # embedded fishbrain-inputs-v1 bytes, or None
    state_digest: str = ""
    checksum: str = field(default="", repr=False)


def derived_ref(last: np.ndarray, step: int, n_ref: int) -> List[np.ndarray]:
    """The refractory queue a run leaves: the spikes of the last min(step, n_ref - 1) steps."""
    if n_ref <= 1:
        return []
    R = min(step, n_ref - 1)
    return [np.flatnonzero(last == s).astype(np.int64) for s in range(step - R, step)]


def checkpoint_of(sim: S.Simulator, embed_inputs: bool = True) -> Checkpoint:
    """The state of `sim` as a Checkpoint (nothing is copied lazily; the sim can keep running)."""
    inputs = encode_inputs(sim)
    nslot = sim.n_delay + 1
    srcs = []
    for s in sim.sources:
        k = _kind(s)
        ss = SourceState(s.name, k)
        if k != KIND_CURRENT:
            st = s.bg.state
            if st["bit_generator"] != "PCG64" or st["has_uint32"] or st["uinteger"]:
                raise ValueError(f"source {s.name!r}: PCG64 state carries a buffered uint32")
            ss.state, ss.inc = int(st["state"]["state"]), int(st["state"]["inc"])
        if k == KIND_SCALAR:
            ss.last, ss.buf = int(s.last), np.asarray(s.buf, np.int64).copy()
        srcs.append(ss)
    ref = [np.asarray(x, np.int64).copy() for x in sim._ref] if sim._ref is not None else []
    return Checkpoint(
        network_digest=sim.net.digest(), params_digest=sim.p.digest(),
        inputs_digest=hashlib.sha256(inputs).hexdigest(), seed=sim.seed, n=sim.net.n,
        n_delay=sim.n_delay, n_ref=sim.n_ref, step=int(sim.step),
        u=sim.u.astype(np.float32).copy(), h=sim.h.astype(np.float32).copy(),
        last=sim.last_spike.astype(np.int64).copy(),
        slots=[np.asarray(sim._slots[(sim.step + j) % nslot], np.int64).copy() for j in range(nslot)],
        ref=ref, sources=srcs, inputs=inputs if embed_inputs else None, state_digest=sim.state_digest())


def _digest_of(ck: Checkpoint) -> str:
    """Simulator.state_digest() of a parsed state, computed by sim.py's own code on a stand-in."""
    nslot = ck.n_delay + 1
    ring = [None] * nslot
    for j, sl in enumerate(ck.slots):
        ring[(ck.step + j) % nslot] = sl
    srcs = []
    for s in ck.sources:
        ns = SimpleNamespace(name=s.name)
        if s.kind != KIND_CURRENT:
            bg = np.random.PCG64()
            bg.state = {"bit_generator": "PCG64", "state": {"state": s.state, "inc": s.inc},
                        "has_uint32": 0, "uinteger": 0}
            ns.bg = bg
        if s.kind == KIND_SCALAR:
            ns.buf = s.buf
        srcs.append(ns)
    stand_in = SimpleNamespace(step=ck.step, u=ck.u, h=ck.h, last_spike=ck.last, n_delay=ck.n_delay,
                               _slots=ring, sources=srcs)
    return S.Simulator.state_digest(stand_in)


def encode(ck: Checkpoint, recompute_digest: bool = False) -> bytes:
    """fishbrain-checkpoint-v1 bytes of `ck`. recompute_digest=True recomputes state_digest from the
    fields first (what a forger would do; tests use it to build self-consistent forgeries)."""
    if recompute_digest:
        ck.state_digest = _digest_of(ck)
    out = bytearray(CHECKPOINT_MAGIC)
    for d in (ck.network_digest, ck.params_digest, ck.inputs_digest):
        out += bytes.fromhex(d)
    _pack_str(out, ck.seed)
    out += struct.pack("<QIIQ", ck.n, ck.n_delay, ck.n_ref, ck.step)
    out += np.ascontiguousarray(ck.u, "<f4").tobytes()
    out += np.ascontiguousarray(ck.h, "<f4").tobytes()
    out += np.ascontiguousarray(ck.last, "<i8").tobytes()
    for sl in ck.slots:
        out += struct.pack("<Q", len(sl)) + np.ascontiguousarray(sl, "<i8").tobytes()
    out += struct.pack("<I", len(ck.ref))
    for sl in ck.ref:
        out += struct.pack("<Q", len(sl)) + np.ascontiguousarray(sl, "<i8").tobytes()
    out += struct.pack("<I", len(ck.sources))
    for s in ck.sources:
        _pack_str(out, s.name)
        out += struct.pack("<B", s.kind)
        if s.kind != KIND_CURRENT:
            out += int(s.state).to_bytes(16, "little") + int(s.inc).to_bytes(16, "little")
        if s.kind == KIND_SCALAR:
            out += struct.pack("<qQ", s.last, len(s.buf)) + np.ascontiguousarray(s.buf, "<i8").tobytes()
    if ck.inputs is None:
        out += b"\x00"
    else:
        out += b"\x01" + struct.pack("<Q", len(ck.inputs)) + ck.inputs
    out += bytes.fromhex(ck.state_digest)
    out += hashlib.sha256(out).digest()
    return bytes(out)


def serialize(sim: S.Simulator, embed_inputs: bool = True) -> bytes:
    """The canonical fishbrain-checkpoint-v1 bytes of the simulator's state. With embed_inputs=False
    the checkpoint carries only the inputs digest and deserialize() needs `inputs=`."""
    return encode(checkpoint_of(sim, embed_inputs))


def _sorted_unique_in(a: np.ndarray, n: int) -> bool:
    return (len(a) == 0) or (a.min() >= 0 and a.max() < n and bool(np.all(np.diff(a) > 0)))


def parse(data: bytes) -> Checkpoint:
    """Parse and check a checkpoint's self-consistency (no network needed). Raises CheckpointError."""
    data = bytes(data)
    if len(data) < len(CHECKPOINT_MAGIC) + 32 * 5:
        raise CheckpointError("checkpoint: truncated")
    body, trailer = data[:-32], data[-32:]
    if hashlib.sha256(body).digest() != trailer:
        raise CheckpointError("checkpoint: checksum mismatch (corrupted or edited bytes)")
    r = _Reader(body, "checkpoint")
    if r.take(len(CHECKPOINT_MAGIC)) != CHECKPOINT_MAGIC:
        raise CheckpointError("checkpoint: not fishbrain-checkpoint-v1")
    net_d, par_d, inp_d = (r.take(32).hex() for _ in range(3))
    seed = r.text()
    n, n_delay, n_ref, step = r.unpack("<QIIQ")
    if step > _MAX_SAFE or n > 2 ** 31 - 1:
        raise CheckpointError("checkpoint: step or size out of range")
    u, h = r.array(n, "<f4").astype(np.float32), r.array(n, "<f4").astype(np.float32)
    last = r.array(n, "<i8").astype(np.int64)
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(h))):
        raise CheckpointError("checkpoint: non-finite voltages or conductances")
    if np.any((last != NEVER) & ((last < 0) | (last >= step))):
        raise CheckpointError("checkpoint: a last-spike step is outside [0, step)")
    slots = []
    for j in range(n_delay + 1):
        sl = r.array(r.unpack("<Q"), "<i8").astype(np.int64)
        t = step - n_delay - 1 + j
        if not _sorted_unique_in(sl, n) or (t < 0 and len(sl)) or np.any(last[sl] < t):
            raise CheckpointError(f"checkpoint: delay slot {j} is not a set of spikes at step {t}")
        slots.append(sl)
    lo = step - n_delay - 1
    recent = np.flatnonzero(last >= max(lo, 0))
    for i in recent:
        sl = slots[int(last[i]) - lo]
        if not len(sl) or sl[np.searchsorted(sl, i).clip(0, len(sl) - 1)] != i:
            raise CheckpointError(f"checkpoint: neuron {int(i)} spiked at step {int(last[i])} but is not in that delay slot")
    R = r.unpack("<I")
    ref = [r.array(r.unpack("<Q"), "<i8").astype(np.int64) for _ in range(R)]
    want = derived_ref(last, step, n_ref)
    if len(ref) != len(want) or any(not np.array_equal(a, b) for a, b in zip(ref, want)):
        raise CheckpointError("checkpoint: the refractory queue does not match the last-spike steps")
    sources, names = [], set()
    for _ in range(r.unpack("<I")):
        name = r.text()
        kind = r.unpack("<B")
        if name in names or kind not in (KIND_SCALAR, KIND_ARRAY, KIND_CURRENT):
            raise CheckpointError(f"checkpoint: bad source entry {name!r}")
        names.add(name)
        ss = SourceState(name, kind)
        if kind != KIND_CURRENT:
            ss.state = int.from_bytes(r.take(16), "little")
            ss.inc = int.from_bytes(r.take(16), "little")
            if not ss.inc & 1:
                raise CheckpointError(f"checkpoint: source {name!r} has an even PCG64 increment")
        if kind == KIND_SCALAR:
            ss.last = r.unpack("<q")
            ss.buf = r.array(r.unpack("<Q"), "<i8").astype(np.int64)
            b = ss.buf
            if (len(b) and (b[0] < 0 or b[-1] > _MAX_SAFE or np.any(np.diff(b) <= 0))) or \
                    ss.last != (int(b[-1]) if len(b) else -1):
                raise CheckpointError(f"checkpoint: source {name!r} pending events are inconsistent")
        sources.append(ss)
    flag = r.unpack("<B")
    if flag not in (0, 1):
        raise CheckpointError("checkpoint: bad inputs flag")
    inputs = r.take(r.unpack("<Q")) if flag else None
    recorded = r.take(32).hex()
    r.done()
    if inputs is not None:
        if hashlib.sha256(inputs).hexdigest() != inp_d:
            raise CheckpointError("checkpoint: embedded inputs do not match the inputs digest")
        specs = decode_inputs(inputs)
        if [(sp["name"], sp["kind"]) for sp in specs] != [(s.name, s.kind) for s in sources]:
            raise CheckpointError("checkpoint: sources differ from the embedded inputs")
    ck = Checkpoint(net_d, par_d, inp_d, seed, n, n_delay, n_ref, step, u, h, last, slots, ref, sources,
                    inputs, state_digest=recorded, checksum=trailer.hex())
    if _digest_of(ck) != recorded:
        raise CheckpointError("checkpoint: the recorded state digest does not match the state")
    return ck


def deserialize(data: bytes, network: S.Network, params: S.LIFParams, inputs: Optional[bytes] = None,
                cls=S.Simulator) -> S.Simulator:
    """A Simulator (or `cls`, e.g. sim_fast.ActiveSimulator) in exactly the checkpoint's state; its
    state_digest() equals the recorded one. `inputs` (fishbrain-inputs-v1 bytes) is required when the
    checkpoint does not embed them, and must equal them when it does."""
    ck = parse(data)
    if network.n != ck.n or network.digest() != ck.network_digest:
        raise CheckpointError("checkpoint: made for a different network")
    if params.digest() != ck.params_digest:
        raise CheckpointError("checkpoint: made for different parameters")
    d = params.derived()
    if (d["n_delay"], d["n_ref"]) != (ck.n_delay, ck.n_ref):
        raise CheckpointError("checkpoint: delay or refractory steps differ from the parameters")
    ib = ck.inputs
    if inputs is not None:
        inputs = bytes(inputs)
        if ib is not None and ib != inputs:
            raise CheckpointError("checkpoint: embedded inputs differ from the inputs given")
        ib = inputs
    if ib is None:
        raise CheckpointError("checkpoint: no inputs embedded and none given")
    if hashlib.sha256(ib).hexdigest() != ck.inputs_digest:
        raise CheckpointError("checkpoint: inputs do not match the inputs digest")
    specs = decode_inputs(ib)
    if [(sp["name"], sp["kind"]) for sp in specs] != [(s.name, s.kind) for s in ck.sources]:
        raise CheckpointError("checkpoint: sources differ from the inputs")
    dead = network.dead
    if dead.any():
        if np.any(ck.last[dead] != NEVER) or any(np.any(dead[sl]) for sl in ck.slots):
            raise CheckpointError("checkpoint: a lesioned neuron has spiked")
    sim = cls(network, seed=ck.seed, params=params)
    sim.sources = [_source_from_spec(sp, ck.seed, params, network.n) for sp in specs]
    for src, ss in zip(sim.sources, ck.sources):
        if ss.kind != KIND_CURRENT:
            src.bg.state = {"bit_generator": "PCG64", "state": {"state": ss.state, "inc": ss.inc},
                            "has_uint32": 0, "uinteger": 0}
        if ss.kind == KIND_SCALAR:
            src.buf, src.last = ss.buf.copy(), int(ss.last)
    sim.step = ck.step
    sim.u[:] = ck.u
    sim.h[:] = ck.h
    sim.last_spike[:] = ck.last
    nslot = ck.n_delay + 1
    sim._slots = [np.zeros(0, np.int64) for _ in range(nslot)]
    for j, sl in enumerate(ck.slots):
        sim._slots[(ck.step + j) % nslot] = sl.copy()
    if sim._ref is not None:
        sim._ref = deque([x.copy() for x in ck.ref], maxlen=sim._ref.maxlen)
    sim._rec_steps, sim._rec_idx, sim._rec_start = [], [], ck.step
    sim.stats = {"spikes": 0, "syn_events": 0, "poisson_events": 0}
    if sim.state_digest() != ck.state_digest:
        raise CheckpointError("checkpoint: the restored state does not reproduce its digest")
    return sim


# ---------------------------------------------------------------------------------------------
# the receipt fields a window binds (reference for the receipt writer)

def kernel_ref(path: Path = KERNEL_PATH) -> dict:
    """{name, version, sha256} of a JS kernel file, as receipt.kernel carries it."""
    raw = Path(path).read_bytes()
    m = re.search(rb'export const KERNEL_VERSION = "([^"]+)";', raw)
    if not m:
        raise ValueError(f"{path}: no KERNEL_VERSION")
    return {"name": "fishbrain-sim", "version": m.group(1).decode(), "sha256": hashlib.sha256(raw).hexdigest()}


def window_receipt(sim: S.Simulator, n_steps: int, blockhash: str, kernel: Optional[dict] = None,
                   embed_inputs: bool = True, frames_source: str = "retina", commit: Optional[str] = None,
                   bridge: Optional[bytes] = None):
    """Run one decision window and return (receipt fields, checkpoint bytes or None, raster).

    `sim` must have been built with seed=seed_string(blockhash) and sit at the window's first step.
    The checkpoint is the state at that step (None at step 0, where Verify builds the state itself);
    serve it content-addressed by its state digest, which is receipt.window.snapshot. For a window
    past step 0, Verify also needs that checkpoint anchored (a chain of links from genesis, or a
    trusted commitment): see js/README.md. `commit` (optional) is the signature of the run-commit
    Memo transaction (run_commit_memo), written as seed.commit.

    `bridge` is the network's fishbrain-bridge-table-v1 bytes (bridgetable.encode / from_bridged), or
    None for an unbridged network. receipt.brain.bridge is its sha256, or null; the table is served
    content-addressed by that hash. Refused before the run: a table that does not account for every
    count of sim.net exactly, a table with no bridged count (that network declares null), and None for
    a network whose meta says it was built with a bridge (bridge.network sets meta["bridge_version"]).
    """
    if sim.seed != seed_string(blockhash):
        raise ValueError("the simulator was not seeded with seed_string(blockhash)")
    from . import bridgetable as BT
    if bridge is None:
        if sim.net.meta.get("bridge_version"):
            raise ValueError(f"the network was built with bridge {sim.net.meta['bridge_version']}: pass its provenance "
                             "table (bridgetable.from_bridged), a bridged network cannot declare brain.bridge null")
        bridge_sha = None
    else:
        BT.check(bytes(bridge), sim.net)
        bridge_sha = BT.sha256(bytes(bridge))
    start = int(sim.step)
    snapshot = sim.state_digest()
    ck = serialize(sim, embed_inputs=embed_inputs) if start > 0 else None
    ras = sim.run(n_steps=n_steps)
    inputs = {"spec": inputs_digest(sim)}
    fr = frames_digest(sim, frames_source)
    if fr is not None:
        inputs["frames"] = fr
    receipt = {
        "type": "receipt",
        "brain": {"network": sim.net.digest(), "params": sim.p.digest(), "state": sim.state_digest(),
                  "spikes": S.spike_hash(ras), "bridge": bridge_sha},
        "inputs": inputs,
        "seed": {"source": SEED_SOURCE, "value": blockhash, **({"commit": commit} if commit else {})},
        "window": {"start_step": start, "n_steps": int(n_steps), "dt_ms": sim.p.dt_ms, "snapshot": snapshot},
        "kernel": kernel if kernel is not None else kernel_ref(),
    }
    return receipt, ck, ras
