"""Read Fish1's public synapse annotations (neuroglancer precomputed, no login).

Brain of record: gs://fish1-public/syn_241003_agg241003_reorient_axde_ei_bayes_idx_pre_250410.precomputed
(Petkova, Januszewski et al. 2025). Each synapse is a LINE annotation with properties
pre_synaptic_confidence (f32), post_synaptic_confidence (f32) and type (u32: 1 or 2, the E/I class),
and relationships pre_synaptic_cell / post_synaptic_cell (agglomeration 241003 segment ids).

Verified 2026-09-25: spatial0/0_0_0 decodes to exactly 10,039 records, and by_id lookups for ids
1043765 and 15832199 return one pre cell and one post cell each.
"""
import gzip
import json
import struct
import urllib.request

import mmh3
import numpy as np

BASE = ("https://storage.googleapis.com/fish1-public/"
        "syn_241003_agg241003_reorient_axde_ei_bayes_idx_pre_250410.precomputed")

# LINE = two rank-3 float32 points (24 bytes), then f32, f32, u32 properties (12 bytes).
RECORD = np.dtype([("p0", "<f4", 3), ("p1", "<f4", 3),
                   ("pre_conf", "<f4"), ("post_conf", "<f4"), ("type", "<u4")])


def fetch(url, start=None, end=None):
    headers = {"Range": f"bytes={start}-{end - 1}"} if start is not None else {}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120).read()


def info(base=BASE):
    return json.loads(fetch(base + "/info"))


def decode_multiple(raw):
    """Multiple-annotation encoding (spatial and relationship chunks): count, records, ids."""
    n = struct.unpack("<Q", raw[:8])[0]
    recs = np.frombuffer(raw[8:8 + n * RECORD.itemsize], dtype=RECORD)
    ids = np.frombuffer(raw[8 + n * RECORD.itemsize:8 + n * RECORD.itemsize + 8 * n], dtype="<u8")
    return recs, ids


def shard_location(sharding, chunk_id):
    pre = chunk_id >> sharding["preshift_bits"]
    if sharding["hash"] == "identity":
        h = pre
    else:  # murmurhash3_x86_128, low 64 bits, seed 0
        h = mmh3.hash64(struct.pack("<Q", pre), seed=0, x64arch=False, signed=False)[0]
    mb, sb = sharding["minishard_bits"], sharding["shard_bits"]
    return (h >> mb) & ((1 << sb) - 1), h & ((1 << mb) - 1)


def read_minishard_index(url, sharding, minishard):
    """Return (chunk_ids, byte_starts, byte_sizes) for one minishard, starts absolute in the file."""
    mb = sharding["minishard_bits"]
    s, e = struct.unpack("<QQ", fetch(url, 16 * minishard, 16 * minishard + 16))
    if e == s:
        return np.zeros(0, "<u8"), np.zeros(0, "<u8"), np.zeros(0, "<u8")
    base = 16 * (1 << mb)
    mi = fetch(url, base + s, base + e)
    if sharding.get("minishard_index_encoding") == "gzip":
        mi = gzip.decompress(mi)
    m = np.frombuffer(mi, dtype="<u8").reshape(3, -1)
    ids = np.cumsum(m[0])
    starts = np.empty_like(m[1])
    for i in range(m.shape[1]):
        starts[i] = m[1][0] if i == 0 else starts[i - 1] + m[2][i - 1] + m[1][i]
    return ids, starts + base, m[2]


def get_sharded(key, sharding, chunk_id, base=BASE):
    shard, minishard = shard_location(sharding, chunk_id)
    url = f"{base}/{key}/{shard:x}.shard"
    ids, starts, sizes = read_minishard_index(url, sharding, minishard)
    hit = np.nonzero(ids == chunk_id)[0]
    if not len(hit):
        return None
    i = int(hit[0])
    d = fetch(url, int(starts[i]), int(starts[i] + sizes[i]))
    return gzip.decompress(d) if sharding.get("data_encoding") == "gzip" else d


def decode_by_id(raw, relationships):
    """One annotation from by_id: record, then per relationship a u32 count and u64 ids."""
    rec = np.frombuffer(raw[:RECORD.itemsize], dtype=RECORD)[0]
    o, rels = RECORD.itemsize, {}
    for r in relationships:
        k = struct.unpack("<I", raw[o:o + 4])[0]
        o += 4
        rels[r["id"]] = np.frombuffer(raw[o:o + 8 * k], dtype="<u8").astype(np.uint64)
        o += 8 * k
    return rec, rels
