"""fishbrain-bridge-table-v1: the bridge provenance of one network, as canonical bytes a receipt can pin.

`Network.digest()` hashes the wiring's arrays only. It cannot say which part of a pair's signed count is a
measured Fish1 synapse and which part a bridge link (G1c class a, b or c) added. That split lives in
`provenance.EdgeProvenance`. This file gives it a wire format, so a receipt can name it
(`receipt.brain.bridge` = sha256 of these bytes, or null for an unbridged network) and a browser can check
it against the replayed network (js/verify.mjs, check "bridge") and colour bridged links from it.

The table lists EVERY pair of the network, in the network's CSR order, with the pair's measured count and
its bridged count per class. So "every count is attributed, none missing, none double-counted" is a
per-pair equation a verifier can check: measured + sum(bridged) == the network's count, pair by pair.

fishbrain-bridge-table-v1 (little-endian throughout)
---------------------------------------------------
    magic   b"fishbrain-bridge-table-v1\\n"                  26 bytes
    str     bridge version (e.g. "g1c-bridge-v1")
    32      edge-list sha256 (raw bytes; the frozen spec's edge_list_sha256)
    str     topology tag (e.g. "intact")
    u32 K,  K x str      bridge class names, in bit order (provenance.BRIDGE_CLASSES: a, b, c)
    u32 G,  G x (str key, i64 value)   gains, keys strictly increasing (e.g. g_a, g_c)
    u32 A,  A x str      ablations, strictly increasing (empty for the intact network)
    u64 n   neurons of the network
    u64 P   pairs (= the network's nnz)
    P x i64 pre           internal neuron index
    P x i64 post          internal neuron index
    P x i64 measured      signed measured count of the pair
    P*K x i64 bridged     signed bridged count per class, row-major (pair, class)

`str` = u32 byte length + printable ASCII (0x20..0x7e). Class names are non-empty and unique; gain
keys match g_<name>; ablation names are non-empty. Rows are strictly increasing in (pre, post) with
0 <= pre, post < n, every value is within +-2^48 (so a browser sums them exactly as doubles), and no
trailing bytes are allowed. Each field has exactly one encoding, so equal tables are equal bytes.
Python `encode` and JS `encodeBridgeTable` (js/verify.mjs) write identical bytes
(tests/test_verify.py).

What the hash does and does not prove
-------------------------------------
`receipt.brain.bridge` fixes WHICH table the receipt claims. Verify then proves the table accounts for
every count of the replayed network. It does not prove the split is the frozen spec's: a table that
moves bridged counts into the measured column, with the receipt's hash recomputed, still sums to the
same counts. That claim is pinned by a page that vouches for (network, bridge) pairs it re-derived from
the frozen spec (verify's opts.trustedBrains), or by a commitment made before the seed (proposed: a
bridge=<hex|none> key in the run-commit memo). See js/README.md, "Bridge provenance".

Two digests the gate already recorded are recomputed from the table alone, so a page can compare
them with evidence/G1c-gate.md's run without trusting this file:
  * `edge_provenance_digest(table)` equals `EdgeProvenance.digest()` of the network's provenance;
  * `bridged_digest(table, network_digest)` equals `bridge.bridged_digest(...)` (gate_results.json's
    "bridged_digest").
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
from typing import Mapping, Optional, Sequence

import numpy as np

from fishbrain import sim as S

TABLE_MAGIC = b"fishbrain-bridge-table-v1\n"
TABLE_FORMAT = "fishbrain-bridge-table-v1"
PROVENANCE_VERSION = "fishbrain-provenance-v1"      # provenance.PROVENANCE_VERSION (the digest recipe)
MAX_ABS = 2 ** 48
MAX_CLASSES = 8
_CLASS_RE = re.compile(r"[a-z]")   # single lowercase letter (no '__proto__', no 'measured')
_GAIN_KEY = re.compile(r"^g_[A-Za-z0-9_]+$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class TableError(ValueError):
    """A provenance table that is malformed, non-canonical, or does not account for a network."""


def _ascii(s: str, what: str, empty_ok: bool = False) -> bytes:
    if not isinstance(s, str) or any(not (0x20 <= ord(ch) <= 0x7E) for ch in s):
        raise TableError(f"{what} must be printable ASCII: {s!r}")
    if not s and not empty_ok:
        raise TableError(f"{what} must not be empty")
    return s.encode("ascii")


def _pack_str(out: bytearray, s: str, what: str, empty_ok: bool = False):
    b = _ascii(s, what, empty_ok)
    out += struct.pack("<I", len(b))
    out += b


def _i64(arr, what: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.size == 0:
        return np.zeros(a.shape, np.int64)
    if a.dtype.kind not in "iu":
        raise TableError(f"{what} must be integers")
    if len(a) and (a.max() > MAX_ABS or a.min() < -MAX_ABS):
        raise TableError(f"{what}: a value is outside +-2^48")
    return a.astype(np.int64)


def _increasing(pre: np.ndarray, post: np.ndarray) -> bool:
    """(pre, post) strictly increasing, compared lexicographically (no pre * n + post overflow)."""
    if len(pre) < 2:
        return True
    dp, dq = np.diff(pre), np.diff(post)
    return bool(np.all((dp > 0) | ((dp == 0) & (dq > 0))))


def encode_parts(n: int, pre, post, measured, bridged, classes: Sequence[str], *, bridge_version: str,
                 edge_list_sha256: str, gains: Mapping[str, int], ablate: Sequence[str] = (),
                 topology: str = "") -> bytes:
    """The canonical bytes from columns. Raises TableError on anything the parser would refuse."""
    classes = [str(c) for c in classes]
    if len(classes) > MAX_CLASSES or len(set(classes)) != len(classes):
        raise TableError(f"classes must be unique, at most {MAX_CLASSES}: {classes}")
    for c in classes:
        if not _CLASS_RE.fullmatch(c):
            raise TableError(f"class name {c!r} is not a single lowercase letter")
    if not _HEX64.fullmatch(str(edge_list_sha256)):
        raise TableError("edge_list_sha256 must be 64 lowercase hex characters")
    keys = sorted(gains)
    if any(not _GAIN_KEY.fullmatch(k) for k in keys):
        raise TableError(f"gain keys must match g_<name>: {keys}")
    abl = sorted(set(str(a) for a in ablate))
    if len(abl) != len(list(ablate)):
        raise TableError("ablations must be unique")
    pre, post, meas = _i64(pre, "pre"), _i64(post, "post"), _i64(measured, "measured")
    K = len(classes)
    flat = np.asarray(bridged).reshape(-1)
    if len(flat) != len(pre) * K:
        raise TableError("bridged must hold one count per pair and class")
    br = _i64(flat, "bridged").reshape(len(pre), K)
    P = len(pre)
    if not (len(post) == len(meas) == P == br.shape[0]):
        raise TableError("pre, post, measured and bridged must have one row per pair")
    if P and (pre.min() < 0 or post.min() < 0 or pre.max() >= n or post.max() >= n):
        raise TableError("a pair's neuron index is outside the network")
    if not _increasing(pre, post):
        raise TableError("pairs must be strictly increasing in (pre, post)")
    out = bytearray(TABLE_MAGIC)
    _pack_str(out, bridge_version, "bridge version")
    out += bytes.fromhex(edge_list_sha256)
    _pack_str(out, topology, "topology", empty_ok=True)
    out += struct.pack("<I", K)
    for c in classes:
        _pack_str(out, c, "class name")
    out += struct.pack("<I", len(keys))
    for k in keys:
        v = gains[k]
        if isinstance(v, bool) or int(v) != v or abs(int(v)) > MAX_ABS:
            raise TableError(f"gain {k} must be an integer within +-2^48")
        _pack_str(out, k, "gain key")
        out += struct.pack("<q", int(v))
    out += struct.pack("<I", len(abl))
    for a in abl:
        _pack_str(out, a, "ablation")
    out += struct.pack("<QQ", int(n), P)
    for arr in (pre, post, meas, br.reshape(-1)):
        out += arr.astype("<i8").tobytes()
    return bytes(out)


def encode(net: S.Network, prov, *, bridge_version: str, edge_list_sha256: str, gains: Mapping[str, int],
           ablate: Sequence[str] = (), topology: str = "") -> bytes:
    """The table of `net` under `prov` (a provenance.EdgeProvenance): every CSR pair, its measured part
    and its bridge parts. Refuses a bridge record whose pair is not in the network (provenance.align
    raises) or was skipped as dead (the table could not hold it, and its digest would disagree)."""
    al = prov.align(net)
    if al.skipped_dead:
        raise TableError(f"{al.skipped_dead} bridge records sit on lesioned pairs; the table cannot attribute them")
    K = len(al.classes)
    br = np.zeros((net.nnz, K), np.int64)
    if len(al.epos):
        br[al.epos] = al.bparts
    return encode_parts(net.n, net.rows(), net.indices.astype(np.int64), al.measured, br, al.classes,
                        bridge_version=bridge_version, edge_list_sha256=edge_list_sha256, gains=gains,
                        ablate=ablate, topology=topology)


def from_bridged(bw) -> bytes:
    """The table of a bridge.Bridged network (bridge.network(topo, g_a, g_c, ablate)). Reads the Bridged
    fields only; bridge.py is not changed."""
    from fishbrain import bridge as BR
    return encode(bw.net, bw.prov, bridge_version=BR.BRIDGE_VERSION, edge_list_sha256=BR.EDGE_LIST_SHA256,
                  gains={"g_a": int(bw.g_a), "g_c": int(bw.g_c)}, ablate=bw.ablate, topology=bw.topo_tag)


class _Reader:
    def __init__(self, data: bytes):
        self.b, self.o = memoryview(data), 0

    def take(self, k: int) -> bytes:
        if k < 0 or self.o + k > len(self.b):
            raise TableError("truncated table")
        v = bytes(self.b[self.o:self.o + k])
        self.o += k
        return v

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def text(self, what: str, empty_ok: bool = False) -> str:
        n = self.u32()
        raw = self.take(n)
        try:
            s = raw.decode("ascii")
        except UnicodeDecodeError:
            raise TableError(f"{what} is not ASCII")
        _ascii(s, what, empty_ok)
        return s


def decode(data: bytes) -> dict:
    """Parse and check canonical form. Returns the header fields and numpy columns."""
    r = _Reader(data)
    if r.take(len(TABLE_MAGIC)) != TABLE_MAGIC:
        raise TableError(f"not a {TABLE_FORMAT} table")
    t = {"format": TABLE_FORMAT, "bridge_version": r.text("bridge version"), "edge_list_sha256": r.take(32).hex(),
         "topology": r.text("topology", empty_ok=True)}
    K = r.u32()
    if K > MAX_CLASSES:
        raise TableError(f"more than {MAX_CLASSES} classes")
    t["classes"] = [r.text("class name") for _ in range(K)]
    for c in t["classes"]:
        if not _CLASS_RE.fullmatch(c):
            raise TableError(f"class name {c!r} is not a single lowercase letter")
    if len(set(t["classes"])) != K:
        raise TableError("class names must be unique")
    G = r.u32()
    gains = []
    for _ in range(G):
        k = r.text("gain key")
        if not _GAIN_KEY.fullmatch(k):
            raise TableError(f"gain key {k!r} must match g_<name>")
        v = struct.unpack("<q", r.take(8))[0]
        if abs(v) > MAX_ABS:
            raise TableError(f"gain {k} is outside +-2^48")
        gains.append((k, v))
    if any(gains[i][0] >= gains[i + 1][0] for i in range(len(gains) - 1)):
        raise TableError("gain keys must be strictly increasing")
    t["gains"] = dict(gains)
    A = r.u32()
    t["ablate"] = [r.text("ablation") for _ in range(A)]
    if any(t["ablate"][i] >= t["ablate"][i + 1] for i in range(A - 1)):
        raise TableError("ablations must be strictly increasing")
    n, P = struct.unpack("<QQ", r.take(16))
    rest = len(data) - r.o
    if rest != 8 * P * (3 + K):
        raise TableError(f"the table holds {rest} bytes of rows, {P} pairs of {K} classes need {8 * P * (3 + K)}"
                         + (" (trailing bytes)" if rest > 8 * P * (3 + K) else ""))
    cols = np.frombuffer(r.take(8 * P * (3 + K)), "<i8")
    pre, post, meas = cols[:P].copy(), cols[P:2 * P].copy(), cols[2 * P:3 * P].copy()
    br = cols[3 * P:].reshape(P, K).copy()
    if P and (cols.max() > MAX_ABS or cols.min() < -MAX_ABS):     # (np.abs(-2**63) is still negative)
        raise TableError("a value is outside +-2^48")
    if P and (pre.min() < 0 or post.min() < 0 or pre.max() >= n or post.max() >= n):
        raise TableError("a pair's neuron index is outside the table's network")
    if not _increasing(pre, post):
        raise TableError("pairs must be strictly increasing in (pre, post)")
    t.update(n=int(n), pairs=int(P), pre=pre, post=post, measured=meas, bridged=br)
    return t


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def attribution_error(t: Mapping, net: S.Network) -> Optional[str]:
    """None if the table accounts for every count of `net` exactly once; else what is wrong (the same
    sentence verify.mjs gives, first difference only)."""
    if t["n"] != net.n:
        return f"the table is for a network of {t['n']} neurons, this one has {net.n}"
    if t["pairs"] != net.nnz:
        return (f"the table lists {t['pairs']} pairs, the network has {net.nnz}: every pair must be attributed "
                "exactly once")
    rows = net.rows()
    cols = net.indices.astype(np.int64)
    bad = np.flatnonzero((t["pre"] != rows) | (t["post"] != cols))
    if len(bad):
        k = int(bad[0])
        return (f"pair {k}: the table names ({int(t['pre'][k])}, {int(t['post'][k])}), the network has "
                f"({int(rows[k])}, {int(cols[k])})")
    tot = t["measured"] + t["bridged"].sum(axis=1)
    bad = np.flatnonzero(tot != net.counts.astype(np.int64))
    if len(bad):
        k = int(bad[0])
        return (f"pair ({int(rows[k])}, {int(cols[k])}): measured {int(t['measured'][k])} + bridged "
                f"{int(t['bridged'][k].sum())} = {int(tot[k])}, the network's count is {int(net.counts[k])}")
    return None


def bridged_count(t: Mapping) -> int:
    return int(np.abs(t["bridged"]).sum())


def check(data: bytes, net: S.Network) -> dict:
    """decode + attribution + the canonical-null rule (a table with no bridged count is refused: an
    unbridged network declares brain.bridge null). Returns the decoded table or raises TableError."""
    t = decode(data)
    err = attribution_error(t, net)
    if err:
        raise TableError(err)
    if bridged_count(t) == 0:
        raise TableError("the table attributes no count to a bridge class: an unbridged network declares "
                         "brain.bridge null instead of a table")
    return t


def summary(t: Mapping) -> dict:
    """The synapse-count share from the table: provenance.synapse_count_share(net, prov)'s numbers
    (|signed count| per pair part), overall and per class. Secondary, never the headline."""
    meas = int(np.abs(t["measured"]).sum())
    per = {c: int(np.abs(t["bridged"][:, q]).sum()) for q, c in enumerate(t["classes"])}
    tot = meas + sum(per.values())
    has = np.any(t["bridged"] != 0, axis=1)
    return {"measured_share": (meas / tot) if tot else None,
            "class_share": {c: (v / tot) if tot else None for c, v in per.items()},
            "synapses": dict(measured=meas, total=tot, **per),
            "pairs": int(t["pairs"]), "pairs_with_bridge_part": int(has.sum()),
            "pairs_bridge_only": int((has & (t["measured"] == 0)).sum())}


def edge_provenance_digest(t: Mapping) -> str:
    """EdgeProvenance.digest() recomputed from the table: the pairs with a bridge part, their parts."""
    has = np.any(t["bridged"] != 0, axis=1)
    h = hashlib.sha256(PROVENANCE_VERSION.encode() + b"|edges")
    h.update(json.dumps(list(t["classes"])).encode())
    h.update(struct.pack("<QQ", int(has.sum()), len(t["classes"])))
    for arr in (t["pre"][has], t["post"][has], t["bridged"][has]):
        h.update(np.ascontiguousarray(arr, "<i8").tobytes())
    return h.hexdigest()


def bridged_digest(t: Mapping, network_digest: str) -> str:
    """bridge.bridged_digest(...) recomputed from the table and Network.digest()."""
    obj = {"network_digest": network_digest, "bridge_version": t["bridge_version"],
           "edge_list_sha256": t["edge_list_sha256"], **{k: int(v) for k, v in t["gains"].items()},
           "ablate": list(t["ablate"]), "topology": t["topology"], "edge_provenance": edge_provenance_digest(t)}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
