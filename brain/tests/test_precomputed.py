"""Offline checks for the Fish1 precomputed decoder (no network)."""
import gzip
import struct

import mmh3
import numpy as np

from fishbrain import precomputed as p


def _multiple_blob(records, ids):
    return struct.pack("<Q", len(records)) + records.tobytes() + np.asarray(ids, dtype="<u8").tobytes()


def test_decode_multiple_round_trip():
    recs = np.zeros(3, dtype=p.RECORD)
    recs["p0"] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    recs["type"] = [1, 2, 1]
    recs["pre_conf"] = [0.5, 0.9, -0.1]
    got, ids = p.decode_multiple(_multiple_blob(recs, [10, 20, 30]))
    assert ids.tolist() == [10, 20, 30]
    assert got["type"].tolist() == [1, 2, 1]
    assert got["p0"][2].tolist() == [7, 8, 9]


def test_decode_by_id_relationships():
    rec = np.zeros(1, dtype=p.RECORD)
    rec["type"] = 2
    rels = [{"id": "pre_synaptic_cell"}, {"id": "pre_synaptic_site"},
            {"id": "post_synaptic_cell"}, {"id": "post_synaptic_site"}]
    raw = rec.tobytes()
    for ids in ([111], [], [222], []):
        raw += struct.pack("<I", len(ids)) + np.asarray(ids, dtype="<u8").tobytes()
    r, out = p.decode_by_id(raw, rels)
    assert int(r["type"]) == 2
    assert out["pre_synaptic_cell"].tolist() == [111]
    assert out["post_synaptic_cell"].tolist() == [222]
    assert out["pre_synaptic_site"].tolist() == []


def test_shard_location_identity_and_murmur():
    ident = {"preshift_bits": 0, "hash": "identity", "minishard_bits": 2, "shard_bits": 1}
    # 13 = 0b1101 -> minishard = 0b01, shard = (13 >> 2) & 1 = 1
    assert p.shard_location(ident, 13) == (1, 1)
    mm = {"preshift_bits": 10, "hash": "murmurhash3_x86_128", "minishard_bits": 12, "shard_bits": 3}
    cid = 1043765
    h = mmh3.hash64(struct.pack("<Q", cid >> 10), seed=0, x64arch=False, signed=False)[0]
    assert p.shard_location(mm, cid) == ((h >> 12) & 7, h & 4095)


def test_record_is_36_bytes():
    # LINE: 2 x 3 float32 + f32 + f32 + u32, as in the Fish1 info file.
    assert p.RECORD.itemsize == 36
    assert gzip  # imported for parity with the module's encodings
