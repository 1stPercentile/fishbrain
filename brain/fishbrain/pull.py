"""Pull the whole Fish1 wiring (brain of record) and write the graph files.

Brain of record, public, no login:
  gs://fish1-public/syn_241003_agg241003_reorient_axde_ei_bayes_idx_pre_250410.precomputed

Strategy (justified in launches/fishbrain/evidence/G1-pull.md):
  * download every by_id shard whole (1.95 GB): each by_id entry carries the full annotation record
    (both line endpoints, pre/post confidence, type) plus its pre_synaptic_cell / post_synaptic_cell,
    so by_id alone is the complete wiring;
  * download every spatial level spatial0..spatial8 (0.59 GB) and decode it independently, as the
    integrity check (count, id set, record equality);
  * the pre/post_synaptic_cell relationship indexes (1.82 GB) are NOT downloaded whole; a random
    sample of segments is fetched by byte range and cross-checked against by_id instead.

Usage (from launches/fishbrain/brain):
  .venv/bin/python -m fishbrain.pull list       # bucket listing -> data/raw/listing.json
  .venv/bin/python -m fishbrain.pull download   # resumable, md5-verified -> data/raw/<key>/<file>
  .venv/bin/python -m fishbrain.pull parse      # -> data/synapses_raw.parquet, synapses.parquet, cells.parquet
  .venv/bin/python -m fishbrain.pull cells      # rebuild data/cells.parquet from synapses.parquet only
  .venv/bin/python -m fishbrain.pull check      # spatial + relationship + peer cross-checks -> data/integrity.json
  .venv/bin/python -m fishbrain.pull report     # markdown tables, content hash, graph shape
  .venv/bin/python -m fishbrain.pull all        # download, parse, cells, check

Measured 2026-09-25 (launches/fishbrain/evidence/G1-pull.md): 29,474,316 synapses, 13,458,709 segments.

Reuses fishbrain.precomputed (RECORD, decode_multiple, decode_by_id, shard_location, BASE);
only the local-file shard reader is new here.
"""
import base64
import gzip
import hashlib
import json
import os
import struct
import sys
import time
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np

from fishbrain import precomputed as pc

BUCKET = "fish1-public"
PREFIX = "syn_241003_agg241003_reorient_axde_ei_bayes_idx_pre_250410.precomputed/"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")
LISTING = os.path.join(RAW, "listing.json")
MANIFEST = os.path.join(RAW, "manifest.json")
PULL_KEYS = ["info"] + [f"spatial{i}" for i in range(9)] + ["by_id"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- listing / download

def list_bucket():
    items, tok = [], None
    while True:
        q = {"prefix": PREFIX, "maxResults": "1000",
             "fields": "items(name,size,md5Hash,crc32c,generation,metageneration,updated,contentEncoding),nextPageToken"}
        if tok:
            q["pageToken"] = tok
        u = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(q)
        d = json.loads(urllib.request.urlopen(u, timeout=60).read())
        items += d.get("items", [])
        tok = d.get("nextPageToken")
        if not tok:
            break
    os.makedirs(RAW, exist_ok=True)
    with open(LISTING, "w") as f:
        json.dump(items, f, indent=1)
    return items


def rel(name):
    return name[len(PREFIX):]


def local_path(name):
    return os.path.join(RAW, rel(name))


def file_digests(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            md5.update(b)
            sha.update(b)
    return base64.b64encode(md5.digest()).decode(), sha.hexdigest()


PART = 32 << 20  # byte-range part size; parts download in parallel to get a fair share of a busy link


def _url(item):
    # Pin the generation so a re-upload mid-download can never splice two versions.
    return (f"https://storage.googleapis.com/{BUCKET}/" + urllib.parse.quote(item["name"])
            + f"?generation={item['generation']}")


def fetch_part(job, retries=40):
    """Download bytes [start, end) of an object into its own part file, resuming on retry."""
    item, k, start, end = job
    part = local_path(item["name"]) + f".part{k:04d}"
    attempt = 0
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > end - start:
            os.remove(part)
            have = 0
        if have == end - start:
            return part
        attempt += 1
        if attempt > retries:
            raise RuntimeError(f"gave up on {part} after {retries} attempts")
        try:
            # Accept-Encoding: gzip stops GCS from transcoding objects stored with
            # Content-Encoding: gzip (spatial0/0_0_0 is one), so we keep the stored bytes and their md5.
            req = urllib.request.Request(_url(item), headers={
                "Range": f"bytes={start + have}-{end - 1}", "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                if r.status != 206 and not (start + have == 0 and end == int(item["size"])):
                    raise RuntimeError(f"server ignored Range (status {r.status})")
                # A connection that trickles (seen: a part stuck 8+ min on this link) is dropped and
                # reopened: under 256 KB in any 120 s window counts as stalled.
                win_t, win_b = time.time(), 0
                for blk in iter(lambda: r.read(1 << 16), b""):
                    f.write(blk)
                    win_b += len(blk)
                    if time.time() - win_t > 120:
                        if win_b < (1 << 18):
                            raise RuntimeError(f"stalled: {win_b} B in {time.time() - win_t:.0f} s")
                        win_t, win_b = time.time(), 0
        except Exception as e:  # noqa: BLE001  retry anything network-shaped, with backoff
            wait = min(60, 2 ** min(attempt, 6))
            log(f"  retry {attempt} {os.path.basename(part)} of {rel(item['name'])}: "
                f"{type(e).__name__}: {e} (sleep {wait}s)")
            time.sleep(wait)


def _parts(item):
    size = int(item["size"])
    return [(item, k, s, min(s + PART, size)) for k, s in enumerate(range(0, size, PART))]


def _adopt_legacy(item):
    """A whole-file partial from an older single-stream run becomes part files (no bytes refetched)."""
    path = local_path(item["name"])
    size = int(item["size"])
    if not os.path.exists(path) or os.path.getsize(path) >= size:
        return
    have = os.path.getsize(path)
    with open(path, "rb") as f:
        for _, k, s, e in _parts(item):
            if s >= have:
                break
            part = path + f".part{k:04d}"
            if not os.path.exists(part):
                f.seek(s)
                with open(part, "wb") as g:
                    g.write(f.read(min(e, have) - s))
    os.remove(path)


def _verified(item):
    path = local_path(item["name"])
    if not os.path.exists(path) or os.path.getsize(path) != int(item["size"]):
        return None
    md5, sha = file_digests(path)
    if md5 != item["md5Hash"]:
        log("MD5 MISMATCH, refetching", rel(item["name"]), md5, item["md5Hash"])
        os.remove(path)
        return None
    return {"path": rel(item["name"]), "bytes": int(item["size"]), "md5": md5, "sha256": sha,
            "generation": item["generation"], "metageneration": item.get("metageneration"),
            "updated": item.get("updated"), "content_encoding": item.get("contentEncoding")}


def download(workers=16, rounds=3):
    items = json.load(open(LISTING)) if os.path.exists(LISTING) else list_bucket()
    want = [it for it in items if rel(it["name"]).split("/")[0] in PULL_KEYS]
    order = {k: i for i, k in enumerate(PULL_KEYS)}
    want.sort(key=lambda it: (order[rel(it["name"]).split("/")[0]], rel(it["name"])))
    total = sum(int(it["size"]) for it in want)
    log(f"downloading {len(want)} files, {total / 1e9:.3f} GB, {workers} parallel range parts")
    t0 = time.time()
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    for rnd in range(rounds):
        todo = []
        for it in want:
            rec = _verified(it)
            if rec:
                manifest[rec["path"]] = {**manifest.get(rec["path"], {}), **rec}
            else:
                os.makedirs(os.path.dirname(local_path(it["name"])), exist_ok=True)
                _adopt_legacy(it)
                todo.append(it)
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
        if not todo:
            break
        jobs = [j for it in todo for j in _parts(it)]
        remaining = {rel(it["name"]): len(_parts(it)) for it in todo}
        log(f"round {rnd}: {len(todo)} files, {len(jobs)} parts")
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(fetch_part, j): j for j in jobs}
            for fu in as_completed(futs):
                it = futs[fu][0]
                fu.result()
                r = rel(it["name"])
                remaining[r] -= 1
                if remaining[r] == 0:  # all parts present: assemble, verify, drop the parts
                    path = local_path(it["name"])
                    parts = [path + f".part{k:04d}" for _, k, _, _ in _parts(it)]
                    with open(path + ".tmp", "wb") as g:
                        for p in parts:
                            with open(p, "rb") as f:
                                for blk in iter(lambda: f.read(1 << 22), b""):
                                    g.write(blk)
                    os.replace(path + ".tmp", path)
                    for p in parts:
                        os.remove(p)
                    rec = _verified(it)
                    if rec:
                        rec["done_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                        manifest[r] = rec
                        with open(MANIFEST, "w") as f:
                            json.dump(manifest, f, indent=1, sort_keys=True)
                        log(f"OK {r} {rec['bytes']:,} B md5 ok ({time.time() - t0:.0f} s into run)")
    missing = [rel(it["name"]) for it in want if rel(it["name"]) not in manifest]
    if missing:
        raise RuntimeError(f"not verified after {rounds} rounds: {missing}")
    log(f"download done: {total / 1e9:.3f} GB verified, this run {time.time() - t0:.0f} s")
    return manifest


# ----------------------------------------------------------------------------- local shard reader

def _read(f, start, n):
    f.seek(start)
    b = f.read(n)
    if len(b) != n:
        raise IOError(f"short read at {start}: {len(b)} of {n}")
    return b


def local_minishard_index(f, sharding, minishard):
    """Local twin of precomputed.read_minishard_index, vectorized offsets."""
    mb = sharding["minishard_bits"]
    s, e = struct.unpack("<QQ", _read(f, 16 * minishard, 16))
    if e == s:
        z = np.zeros(0, "<u8")
        return z, z, z
    base = 16 * (1 << mb)
    mi = _read(f, base + s, e - s)
    if sharding.get("minishard_index_encoding") == "gzip":
        mi = gzip.decompress(mi)
    m = np.frombuffer(mi, dtype="<u8").reshape(3, -1)
    ids = np.cumsum(m[0], dtype=np.uint64)
    sizes = m[2].astype(np.uint64)
    # start_i = start_{i-1} + size_{i-1} + delta_i, with start_0 = delta_0
    starts = np.cumsum(m[1], dtype=np.uint64) + np.cumsum(sizes, dtype=np.uint64) - sizes
    return ids, starts + np.uint64(base), sizes


def iter_shard(path, sharding):
    """Yield (minishard, ids, starts, sizes) for every minishard of a local shard file."""
    with open(path, "rb") as f:
        for ms in range(1 << sharding["minishard_bits"]):
            ids, starts, sizes = local_minishard_index(f, sharding, ms)
            if len(ids):
                yield ms, ids, starts, sizes


def shard_files(key):
    d = os.path.join(RAW, key)
    return sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith(".shard"))


def count_entries(key, sharding):
    """Number of chunk entries in every minishard index of every shard of `key` (no data decoded)."""
    n = 0
    for p in shard_files(key):
        for _, ids, _, _ in iter_shard(p, sharding):
            n += len(ids)
    return n


# ----------------------------------------------------------------------------- by_id decode

BYID_FAST = np.dtype([("p0", "<f4", 3), ("p1", "<f4", 3), ("pre_conf", "<f4"), ("post_conf", "<f4"),
                      ("type", "<u4"), ("k_pre", "<u4"), ("pre", "<u8"), ("k_pre_site", "<u4"),
                      ("k_post", "<u4"), ("post", "<u8"), ("k_post_site", "<u4")])
assert BYID_FAST.itemsize == 68


def _decompress_entries(region, lo, starts, sizes, gz):
    """Decompress each entry by its own (start, size) from the minishard index; no member guessing."""
    mv = memoryview(region)
    parts = []
    for s, z in zip(starts.tolist(), sizes.tolist()):
        p = mv[s - lo:s - lo + z]
        parts.append(zlib.decompress(p, 31) if gz else bytes(p))
    return b"".join(parts), np.fromiter((len(p) for p in parts), np.int64, len(parts))


def parse_byid_shard(path):
    """Decode every annotation of one local by_id shard. Returns dict of columns + anomaly list."""
    sharding = _info()["by_id"]["sharding"]
    rels = _info()["relationships"]
    cols = {k: [] for k in ("syn_id", "pre", "post", "type", "pre_conf", "post_conf",
                            "x0", "y0", "z0", "x1", "y1", "z1", "k_pre", "k_post", "k_site")}
    anomalies = []
    with open(path, "rb") as f:
        for ms in range(1 << sharding["minishard_bits"]):
            ids, starts, sizes = local_minishard_index(f, sharding, ms)
            if not len(ids):
                continue
            lo, hi = int(starts.min()), int((starts + sizes).max())
            region = _read(f, lo, hi - lo)
            raw, lens = _decompress_entries(region, lo, starts, sizes, sharding.get("data_encoding") == "gzip")
            if np.all(lens == 68):
                a = np.frombuffer(raw, dtype=BYID_FAST)
                ok = (a["k_pre"] == 1) & (a["k_post"] == 1) & (a["k_pre_site"] == 0) & (a["k_post_site"] == 0)
                if not ok.all():  # 68 bytes but a different split of counts: fall back below
                    a = None
            else:
                a = None
            if a is not None:
                cols["syn_id"].append(ids)
                cols["pre"].append(a["pre"])
                cols["post"].append(a["post"])
                cols["type"].append(a["type"])
                cols["pre_conf"].append(a["pre_conf"])
                cols["post_conf"].append(a["post_conf"])
                for j, c in enumerate("xyz"):
                    cols[c + "0"].append(a["p0"][:, j])
                    cols[c + "1"].append(a["p1"][:, j])
                one = np.ones(len(ids), np.uint8)
                cols["k_pre"].append(one)
                cols["k_post"].append(one)
                cols["k_site"].append(np.zeros(len(ids), np.uint8))
                continue
            # general path: one entry at a time through precomputed.decode_by_id
            offs = np.concatenate([[0], np.cumsum(lens)])
            n = len(ids)
            pre = np.zeros(n, np.uint64); post = np.zeros(n, np.uint64)
            kp = np.zeros(n, np.uint8); kq = np.zeros(n, np.uint8); ks = np.zeros(n, np.uint8)
            recs = np.zeros(n, pc.RECORD)
            for i in range(n):
                rec, r = pc.decode_by_id(raw[offs[i]:offs[i + 1]], rels)
                recs[i] = rec
                a_pre, a_post = r["pre_synaptic_cell"], r["post_synaptic_cell"]
                kp[i], kq[i] = min(len(a_pre), 255), min(len(a_post), 255)
                ks[i] = min(len(r["pre_synaptic_site"]) + len(r["post_synaptic_site"]), 255)
                pre[i] = a_pre[0] if len(a_pre) else 0
                post[i] = a_post[0] if len(a_post) else 0
                used = RECORD_SIZE + sum(4 + 8 * len(v) for v in r.values())
                if len(a_pre) != 1 or len(a_post) != 1 or ks[i] or used != lens[i]:
                    anomalies.append({"syn_id": int(ids[i]), "k_pre": len(a_pre), "k_post": len(a_post),
                                      "k_site": int(ks[i]), "bytes": int(lens[i]), "decoded_bytes": used,
                                      "pre_all": [int(v) for v in a_pre][:10],
                                      "post_all": [int(v) for v in a_post][:10]})
            cols["syn_id"].append(ids)
            cols["pre"].append(pre); cols["post"].append(post)
            cols["type"].append(recs["type"]); cols["pre_conf"].append(recs["pre_conf"])
            cols["post_conf"].append(recs["post_conf"])
            for j, c in enumerate("xyz"):
                cols[c + "0"].append(recs["p0"][:, j])
                cols[c + "1"].append(recs["p1"][:, j])
            cols["k_pre"].append(kp); cols["k_post"].append(kq); cols["k_site"].append(ks)
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    return out, anomalies


RECORD_SIZE = pc.RECORD.itemsize
_INFO = None


def _info():
    global _INFO
    if _INFO is None:
        p = os.path.join(RAW, "info")
        _INFO = json.load(open(p)) if os.path.exists(p) else pc.info()
    return _INFO


# ----------------------------------------------------------------------------- spatial decode

def parse_spatial(keys=None):
    """Decode every chunk of every spatial level. Returns (per-level counts, chunks, ids, records)."""
    inf = _info()
    per_level, all_ids, all_recs, chunks = {}, [], [], {}
    for lvl in inf["spatial"]:
        key = lvl["key"]
        if keys is not None and key not in keys:
            continue
        n_lvl, n_chunks = 0, 0
        if "sharding" not in lvl:  # unsharded: one file per grid cell
            d = os.path.join(RAW, key)
            for fn in sorted(os.listdir(d)):
                b = open(os.path.join(d, fn), "rb").read()
                if b[:2] == b"\x1f\x8b":  # stored with Content-Encoding: gzip (GCS object metadata)
                    b = gzip.decompress(b)
                recs, ids = pc.decode_multiple(b)
                all_ids.append(ids.copy()); all_recs.append(recs.copy())
                n_lvl += len(ids); n_chunks += 1
        else:
            sh = lvl["sharding"]
            for p in shard_files(key):
                with open(p, "rb") as f:
                    for ms in range(1 << sh["minishard_bits"]):
                        cids, starts, sizes = local_minishard_index(f, sh, ms)
                        for s, z in zip(starts, sizes):
                            b = _read(f, int(s), int(z))
                            if sh.get("data_encoding") == "gzip":
                                b = gzip.decompress(b)
                            recs, ids = pc.decode_multiple(b)
                            all_ids.append(ids.copy()); all_recs.append(recs.copy())
                            n_lvl += len(ids); n_chunks += 1
        per_level[key] = n_lvl
        chunks[key] = n_chunks
        log(f"  {key}: {n_chunks} chunks, {n_lvl:,} annotations")
    return per_level, chunks, np.concatenate(all_ids), np.concatenate(all_recs)


# ----------------------------------------------------------------------------- parse -> parquet

def parse(procs=4):
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.time()
    paths = shard_files("by_id")
    log(f"decoding {len(paths)} by_id shards with {procs} processes")
    with ProcessPoolExecutor(procs) as ex:
        results = list(ex.map(parse_byid_shard, paths))
    anomalies = [a for r in results for a in r[1]]
    keys = list(results[0][0])
    cols = {}
    for k in keys:  # concatenate column by column and drop the per-shard copies as we go
        cols[k] = np.concatenate([r[0][k] for r in results])
        for r in results:
            r[0][k] = None
    del results
    order = np.argsort(cols["syn_id"], kind="stable")
    for k in keys:  # reorder in place, one column at a time (peak = one extra column)
        cols[k] = cols[k][order]
    del order
    n = len(cols["syn_id"])
    dup_ids = int(n - len(np.unique(cols["syn_id"])))
    log(f"by_id decoded: {n:,} annotations, {dup_ids} duplicate ids, {len(anomalies)} anomalies, "
        f"{time.time() - t0:.0f} s")

    # Memory: this Mac swaps hard under other sessions, so every step below frees what it no longer
    # needs, and the per-cell table is a separate pass over synapses.parquet (build_cells).
    raw_cols = ["syn_id", "pre", "post", "type", "pre_conf", "post_conf",
                "x0", "y0", "z0", "x1", "y1", "z1", "k_pre", "k_post", "k_site"]
    cols["type"] = cols["type"].astype(np.uint32, copy=False)
    for k in ("k_pre", "k_post", "k_site"):
        cols[k] = cols[k].astype(np.uint8, copy=False)
    tbl = pa.Table.from_arrays([pa.array(cols[k]) for k in raw_cols], names=raw_cols)
    pq.write_table(tbl, os.path.join(DATA, "synapses_raw.parquet"), compression="zstd")
    del tbl

    stats = {"n_raw": n, "duplicate_ids": dup_ids, "anomalies": len(anomalies),
             "anomaly_examples": anomalies[:50],
             "raw_min_id": int(cols["syn_id"][0]), "raw_max_id": int(cols["syn_id"][-1])}
    t = cols["type"]
    stats["type_counts"] = {str(int(v)): int(c) for v, c in zip(*np.unique(t, return_counts=True))}
    if not set(stats["type_counts"]) <= {"1", "2"}:
        log("WARNING: type values outside {1,2}:", stats["type_counts"])
    for k in ("k_pre", "k_post", "k_site"):
        stats[k + "_counts"] = {str(int(v)): int(c) for v, c in zip(*np.unique(cols[k], return_counts=True))}
        del cols[k]
    # pre_conf / post_conf distributions and type x sign(pre_conf) (handed to the E/I task)
    pcf, qcf = cols["pre_conf"], cols["post_conf"]
    edges = [-1.0001, -0.99, -0.9, -0.5, 0.0, 0.5, 0.9, 0.99, 1.0001]
    stats["pre_conf_hist"] = {"edges": edges, "counts": np.histogram(pcf, edges)[0].tolist(),
                              "min": float(pcf.min()), "max": float(pcf.max()),
                              "n_zero": int((pcf == 0).sum()), "n_nan": int(np.isnan(pcf).sum())}
    stats["post_conf_summary"] = {"min": float(qcf.min()), "max": float(qcf.max()),
                                  "n_zero": int((qcf == 0).sum()), "n_nonzero": int((qcf != 0).sum()),
                                  "n_nan": int(np.isnan(qcf).sum())}
    stats["type_x_sign_pre_conf"] = {
        str(tv): {"neg": int(((t == tv) & (pcf < 0)).sum()), "zero": int(((t == tv) & (pcf == 0)).sum()),
                  "pos": int(((t == tv) & (pcf > 0)).sum())} for tv in (1, 2)}
    for k in ("x1", "y1", "z1"):
        del cols[k]

    # Filters, each counted on the raw set (overlaps reported too).
    pre0 = cols["pre"] == 0
    post0 = cols["post"] == 0
    self_loop = (cols["pre"] == cols["post"]) & ~pre0
    drop = pre0 | post0 | self_loop
    stats["filter"] = {"pre_eq_0": int(pre0.sum()), "post_eq_0": int(post0.sum()),
                       "pre_and_post_eq_0": int((pre0 & post0).sum()),
                       "self_loop_nonzero": int(self_loop.sum()), "removed_total": int(drop.sum()),
                       "kept": int((~drop).sum())}
    del pre0, post0, self_loop
    keep = ~drop
    del drop
    out_names = {"syn_id": "syn_id", "pre": "pre", "post": "post", "type": "type", "pre_conf": "pre_conf",
                 "post_conf": "post_conf", "x0": "x", "y0": "y", "z0": "z"}
    arrays = []
    for k in out_names:  # filter one column at a time, dropping the unfiltered copy
        v = cols.pop(k)
        v = v[keep] if not keep.all() else v
        arrays.append(pa.array(v.astype(np.uint8) if k == "type" else v))
        del v
    del cols, keep
    tbl = pa.Table.from_arrays(arrays, names=list(out_names.values()))
    del arrays
    pq.write_table(tbl, os.path.join(DATA, "synapses.parquet"), compression="zstd")
    stats["n_synapses"] = tbl.num_rows
    del tbl
    stats["parse_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(DATA, "parse_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    log("parse: wrote synapses_raw.parquet and synapses.parquet; "
        + json.dumps({k: v for k, v in stats.items() if k in ("n_raw", "n_synapses", "filter", "type_counts")}))
    return stats


def build_cells():
    """Per-segment table from data/synapses.parquet (the filtered wiring)."""
    import pandas as pd
    import pyarrow.parquet as pq

    t0 = time.time()
    tb = pq.read_table(os.path.join(DATA, "synapses.parquet"), columns=["pre", "post", "type"])
    pre = tb.column("pre").to_numpy()
    post = tb.column("post").to_numpy()
    ty = tb.column("type").to_numpy()
    del tb
    seg = np.unique(np.concatenate([pre, post]))
    m = len(seg)
    ip = np.searchsorted(seg, pre)
    uniq_pre = int(len(np.unique(pre)))
    del pre
    iq = np.searchsorted(seg, post)
    uniq_post = int(len(np.unique(post)))
    del post

    def bc(idx, w=None):
        return np.bincount(idx, weights=w, minlength=m)

    t1 = (ty == 1).astype(np.float64)
    t2 = (ty == 2).astype(np.float64)
    type_counts = {str(int(v)): int(c) for v, c in zip(*np.unique(ty, return_counts=True))}
    del ty
    n_out, n_in = bc(ip), bc(iq)
    cells = pd.DataFrame({
        "seg_id": seg.astype(np.uint64),
        "n_out": n_out.astype(np.int64), "n_in": n_in.astype(np.int64),
        "out_type1": bc(ip, t1).astype(np.int64), "out_type2": bc(ip, t2).astype(np.int64),
        "in_type1": bc(iq, t1).astype(np.int64), "in_type2": bc(iq, t2).astype(np.int64),
    })
    del t1, t2
    tot = n_out + n_in
    for c in "xyz":
        v = pq.read_table(os.path.join(DATA, "synapses.parquet"), columns=[c]).column(c).to_numpy().astype(np.float64)
        so, si = bc(ip, v), bc(iq, v)
        del v
        cells["c" + c] = ((so + si) / tot).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            cells["out_c" + c] = (so / n_out).astype(np.float32)
            cells["in_c" + c] = (si / n_in).astype(np.float32)
    cells = cells[["seg_id", "n_out", "n_in", "out_type1", "out_type2", "in_type1", "in_type2",
                   "cx", "cy", "cz", "out_cx", "out_cy", "out_cz", "in_cx", "in_cy", "in_cz"]]
    cells.to_parquet(os.path.join(DATA, "cells.parquet"), compression="zstd", index=False)
    st = json.load(open(os.path.join(DATA, "parse_stats.json")))
    st.update({
        "unique_pre": uniq_pre, "unique_post": uniq_post, "unique_cells": int(m),
        "type_counts_filtered": type_counts,
        "cells_with_out": int((cells.n_out > 0).sum()), "cells_with_in": int((cells.n_in > 0).sum()),
        "cells_with_both": int(((cells.n_out > 0) & (cells.n_in > 0)).sum()),
        "cells_out_type_mixed": int(((cells.out_type1 > 0) & (cells.out_type2 > 0)).sum()),
        "cells_out_ge5": int((cells.n_out >= 5).sum()),
        "cells_out_ge5_type_mixed": int(((cells.n_out >= 5) & (cells.out_type1 > 0) & (cells.out_type2 > 0)).sum()),
        "max_n_in": int(cells.n_in.max()), "max_n_out": int(cells.n_out.max()),
        "cells_seconds": round(time.time() - t0, 1),
    })
    with open(os.path.join(DATA, "parse_stats.json"), "w") as f:
        json.dump(st, f, indent=1)
    log(f"cells: {m:,} segments, {uniq_pre:,} with outputs, {uniq_post:,} with inputs")
    return st


# ----------------------------------------------------------------------------- integrity checks

def check(n_rel_segments=60, seed=20260925):
    """Integrity: by_id vs spatial (count, id sets, bitwise records), peer counts, relationship sample.

    Lean on memory: spatial is decoded one level at a time and compared against the raw columns.
    """
    import pyarrow.parquet as pq

    t0 = time.time()
    inf = _info()
    rawp = os.path.join(DATA, "synapses_raw.parquet")
    out = {}
    out["by_id_index_entries"] = count_entries("by_id", inf["by_id"]["sharding"])
    log(f"by_id minishard-index entries: {out['by_id_index_entries']:,}")

    bid = pq.read_table(rawp, columns=["syn_id"]).column("syn_id").to_numpy()
    out["by_id_rows"] = int(len(bid))
    out["by_id_sorted_strictly"] = bool(np.all(bid[1:] > bid[:-1]))
    out["by_id_duplicate_ids"] = int(len(bid) - len(np.unique(bid))) if not out["by_id_sorted_strictly"] else 0
    if not out["by_id_sorted_strictly"]:
        log("WARNING: by_id ids not strictly increasing; duplicates:", out["by_id_duplicate_ids"])
        bid = np.unique(bid)
    out["by_id_min_id"], out["by_id_max_id"] = int(bid.min()), int(bid.max())

    rec_cols = ["x0", "y0", "z0", "x1", "y1", "z1", "type", "pre_conf", "post_conf"]
    rt = pq.read_table(rawp, columns=rec_cols)
    R = {k: rt.column(k).to_numpy() for k in rec_cols}
    del rt

    out["spatial_levels"], level_ids = {}, []
    n_cmp = n_eq = 0
    for lvl in inf["spatial"]:
        key = lvl["key"]
        pl, ch, sids, srecs = parse_spatial([key])
        uk, ck = np.unique(sids, return_counts=True)
        out["spatial_levels"][key] = {"chunks": ch[key], "entries": pl[key], "unique_ids": int(len(uk)),
                                      "ids_in_more_than_one_chunk": int((ck > 1).sum()),
                                      "max_copies": int(ck.max())}
        level_ids.append(uk)
        # bitwise record equality: every spatial entry (duplicates included) vs by_id's copy
        idx = np.searchsorted(bid, sids)
        ok = (idx < len(bid)) & (bid[np.minimum(idx, len(bid) - 1)] == sids)
        idx, sr = idx[ok], srecs[ok]
        del sids, srecs
        eq = np.ones(len(idx), bool)
        for j, c in enumerate("xyz"):
            eq &= R[c + "0"][idx].view(np.uint32) == np.ascontiguousarray(sr["p0"][:, j]).view(np.uint32)
            eq &= R[c + "1"][idx].view(np.uint32) == np.ascontiguousarray(sr["p1"][:, j]).view(np.uint32)
        eq &= R["type"][idx] == sr["type"]
        eq &= R["pre_conf"][idx].view(np.uint32) == np.ascontiguousarray(sr["pre_conf"]).view(np.uint32)
        eq &= R["post_conf"][idx].view(np.uint32) == np.ascontiguousarray(sr["post_conf"]).view(np.uint32)
        n_cmp += int(len(idx))
        n_eq += int(eq.sum())
        del idx, sr, eq
    out["spatial_sum_entries"] = int(sum(v["entries"] for v in out["spatial_levels"].values()))
    allu = np.concatenate(level_ids)
    u = np.unique(allu)
    out["spatial_sum_unique_per_level"] = int(len(allu))
    out["spatial_unique_ids"] = int(len(u))
    out["spatial_ids_on_more_than_one_level"] = int(len(allu) - len(u))
    out["spatial_ids_in_more_than_one_chunk"] = int(sum(v["ids_in_more_than_one_chunk"]
                                                        for v in out["spatial_levels"].values()))
    del allu, level_ids
    out["spatial_entries_compared"] = n_cmp
    out["spatial_entries_bitwise_equal_to_by_id"] = n_eq

    out["id_sets_equal"] = bool(len(u) == len(bid) and np.array_equal(u, bid))
    only_s = np.setdiff1d(u, bid, assume_unique=True)
    only_b = np.setdiff1d(bid, u, assume_unique=True)
    out["ids_in_spatial_not_by_id"] = int(only_s.size)
    out["ids_in_by_id_not_spatial"] = int(only_b.size)
    ib = np.searchsorted(bid, only_b)[:20]
    pp = pq.read_table(rawp, columns=["pre", "post"])
    P = {k: pp.column(k).to_numpy() for k in ("pre", "post")}
    del pp
    out["ids_in_by_id_not_spatial_rows"] = [
        {"syn_id": int(bid[i]), "pre": int(P["pre"][i]), "post": int(P["post"][i]), "type": int(R["type"][i]),
         "pre_conf": float(R["pre_conf"][i]),
         **{c: float(R[c][i]) for c in ("x0", "y0", "z0", "x1", "y1", "z1")}} for i in ib]
    del u
    typ_a = R["type"]
    del R

    # Counts measured earlier today from the relationship indexes by the cells task
    # (evidence/G1-cells.md section 8); matching them is a free pre/post cross-check.
    peer = {"15773771512": {"in": 5849, "in_type1": 1294, "in_type2": 4555, "out": 38},
            "16304202596": {"in": 4173, "in_type1": 629, "in_type2": 3544, "out": 25},
            "17405827637": {"in": 2603}, "16039136570": {"in": 2734},
            "20607607971": {"in": 117, "out": 35}}
    got = {}
    for sid, exp in peer.items():
        s_ = np.uint64(int(sid))
        mi, mo = P["post"] == s_, P["pre"] == s_
        g = {"in": int(mi.sum()), "in_type1": int((mi & (typ_a == 1)).sum()),
             "in_type2": int((mi & (typ_a == 2)).sum()), "out": int(mo.sum())}
        got[sid] = {"expected": exp, "measured_raw": {k: g[k] for k in exp},
                    "match": all(g[k] == v for k, v in exp.items())}
    out["peer_counts_vs_relationship_index"] = got
    log(f"  peer counts matching: {sum(v['match'] for v in got.values())}/{len(got)}")
    del typ_a
    out["cave_synapses_axde_v709"] = 29474316
    out["by_id_minus_cave"] = int(len(bid) - 29474316)
    with open(os.path.join(DATA, "integrity.json"), "w") as f:  # save before the network part
        json.dump(out, f, indent=1)

    # relationship index cross-check on a sample of segments, fetched by byte range: n_rel_segments
    # uniformly random segments plus the 10 highest-degree ones, per relationship.
    rng = np.random.default_rng(seed)
    rel_out = {}
    for key, col in (("pre_synaptic_cell", "pre"), ("post_synaptic_cell", "post")):
        sh = next(r for r in inf["relationships"] if r["id"] == key)["sharding"]
        segcol = P[col]
        o = np.argsort(segcol, kind="stable")
        sseg, sids = segcol[o], bid[o]
        del o
        uniq, first, cnt = np.unique(sseg, return_index=True, return_counts=True)
        del sseg
        nz = np.nonzero(uniq != 0)[0]
        pick = [int(i) for i in rng.choice(nz, size=min(n_rel_segments, len(nz)), replace=False)]
        pick += [int(i) for i in nz[np.argsort(cnt[nz])[-10:]] if int(i) not in pick]

        def one(i):
            seg = int(uniq[i])
            err = None
            for attempt in range(8):  # the link drops TLS handshakes; retry, then report
                try:
                    b = pc.get_sharded(key, sh, seg)
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    time.sleep(min(30, 2 ** attempt))
            else:
                return seg, -1, int(cnt[i]), None, err
            ids_rel = np.sort(pc.decode_multiple(b)[1]) if b is not None else np.zeros(0, np.uint64)
            ids_byid = np.sort(sids[first[i]:first[i] + cnt[i]])
            return seg, int(len(ids_rel)), int(len(ids_byid)), bool(np.array_equal(ids_rel, ids_byid)), None

        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(one, pick))
        failed = [{"seg": r[0], "error": r[4]} for r in res if r[3] is None]
        res = [r for r in res if r[3] is not None]
        agree = sum(r[3] for r in res)
        rel_out[key] = {"segments_sampled": len(res), "agree": agree, "fetch_failed": failed,
                        "disagree": [{"seg": r[0], "rel_n": r[1], "by_id_n": r[2]} for r in res if not r[3]][:20],
                        "synapses_covered": int(sum(r[2] for r in res)),
                        "largest_segment_synapses": int(max([r[2] for r in res] or [0]))}
        log(f"  {key}: {agree}/{len(res)} sampled segments identical, {len(failed)} fetch failures")
    out["relationship_sample"] = rel_out
    out["check_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(DATA, "integrity.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(json.dumps({k: v for k, v in out.items() if k not in ("spatial_levels",)}, indent=1))
    return out


def content_hash(path=None):
    """sha256 over the wiring itself, independent of parquet encoding: little-endian bytes of
    syn_id (u64), pre (u64), post (u64), type (u8), each column whole, in that order, rows in
    syn_id order. Same wiring -> same hash on any machine and any pyarrow version."""
    import pyarrow.parquet as pq

    path = path or os.path.join(DATA, "synapses.parquet")
    h = hashlib.sha256()
    for c, dt in (("syn_id", "<u8"), ("pre", "<u8"), ("post", "<u8"), ("type", "u1")):
        v = pq.read_table(path, columns=[c]).column(c).to_numpy().astype(dt, copy=False)
        if c == "syn_id" and not np.all(v[1:] > v[:-1]):
            raise ValueError("synapses not in strictly increasing syn_id order")
        h.update(np.ascontiguousarray(v).tobytes())
        del v
    return h.hexdigest()


def graph_shape():
    """Facts about the pulled wiring that downstream tasks rely on (printed by `report`)."""
    import pyarrow.parquet as pq

    out = {}
    t = pq.read_table(os.path.join(DATA, "synapses.parquet"), columns=["syn_id", "type"])
    ids, ty = t.column("syn_id").to_numpy(), t.column("type").to_numpy()
    del t
    sw = np.nonzero(ty[1:] != ty[:-1])[0]
    out["type_runs_by_syn_id"] = [[int(ids[a]), int(ids[b]), int(ty[a])] for a, b in
                                  zip(np.r_[0, sw + 1], np.r_[sw, len(ids) - 1])]
    blk = ids // 1024
    first = np.r_[True, blk[1:] != blk[:-1]]
    g = np.cumsum(first) - 1
    n1, n = np.bincount(g, weights=(ty == 1)), np.bincount(g)
    out["blocks_of_1024_ids"] = int(len(n))
    out["blocks_of_1024_ids_mixed_type"] = int(((n1 > 0) & (n1 < n)).sum())
    del ids, blk, first, g
    pcf = pq.read_table(os.path.join(DATA, "synapses.parquet"), columns=["pre_conf"]).column("pre_conf").to_numpy()
    out["pre_conf_p5_p25_p50_p75_p95_by_type"] = {
        str(tv): [round(float(x), 4) for x in np.percentile(pcf[ty == tv], [5, 25, 50, 75, 95])] for tv in (1, 2)}
    del pcf, ty
    r = pq.read_table(os.path.join(DATA, "synapses_raw.parquet"), columns=["x0", "y0", "z0", "x1", "y1", "z1"])
    d2 = np.zeros(r.num_rows, np.float64)
    for c, nm in (("x", 8.0), ("y", 8.0), ("z", 30.0)):
        d2 += ((r.column(c + "0").to_numpy().astype(np.float64) - r.column(c + "1").to_numpy()) * nm) ** 2
    del r
    d = np.sqrt(d2)
    out["line_length_nm_p1_p50_p99_max"] = [round(float(x), 1) for x in np.percentile(d, [1, 50, 99, 100])]
    out["zero_length_lines"] = int((d == 0).sum())
    del d, d2
    c = pq.read_table(os.path.join(DATA, "cells.parquet"), columns=["seg_id", "n_out", "n_in"])
    sid, o, i = (c.column(k).to_numpy() for k in ("seg_id", "n_out", "n_in"))
    tot = o + i
    out["segments"] = int(len(sid))
    for th in (10, 100, 1000):
        out[f"segments_with_ge_{th}_synapses"] = int((tot >= th).sum())
    out["segments_with_inputs_and_outputs"] = int(((o > 0) & (i > 0)).sum())
    out["segments_in_ge_10_and_out_ge_10"] = int(((o >= 10) & (i >= 10)).sum())
    out["segments_in_ge_100_and_out_ge_100"] = int(((o >= 100) & (i >= 100)).sum())
    out["n_in_p50_p90_p99_p999"] = np.percentile(i[i > 0], [50, 90, 99, 99.9]).tolist()
    out["n_out_p50_p90_p99_p999"] = np.percentile(o[o > 0], [50, 90, 99, 99.9]).tolist()
    out["top5_by_inputs_seg_in_out"] = [[int(sid[k]), int(i[k]), int(o[k])] for k in np.argsort(i)[-5:][::-1]]
    out["top5_by_outputs_seg_in_out"] = [[int(sid[k]), int(i[k]), int(o[k])] for k in np.argsort(o)[-5:][::-1]]
    return out


def report():
    """Markdown tables for the evidence note, generated from manifest/parse_stats/integrity JSON."""
    man = json.load(open(MANIFEST))
    lines = ["| file (under the .precomputed/ prefix) | bytes | generation | md5 (GCS, b64) | sha256 |",
             "|---|---:|---|---|---|"]
    tot = 0
    for k in sorted(man, key=lambda r: (PULL_KEYS.index(r.split("/")[0]), r)):
        r = man[k]
        tot += r["bytes"]
        enc = f" (stored {r['content_encoding']})" if r.get("content_encoding") else ""
        lines.append(f"| `{k}`{enc} | {r['bytes']:,} | {r['generation']} | `{r['md5']}` | `{r['sha256']}` |")
    lines.append(f"| **total, {len(man)} files** | **{tot:,}** | | | |")
    print("\n".join(lines))
    for fn in ("parse_stats.json", "integrity.json"):
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            print(f"\n{fn}:")
            d = json.load(open(p))
            d.pop("anomaly_examples", None)
            print(json.dumps(d, indent=1))
    import pyarrow
    print(f"\npyarrow {pyarrow.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}")
    for fn in ("synapses_raw.parquet", "synapses.parquet", "cells.parquet"):
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            print(f"{fn}: {os.path.getsize(p):,} bytes, sha256 {file_digests(p)[1]}")
    if os.path.exists(os.path.join(DATA, "synapses.parquet")):
        print(f"wiring content hash (syn_id|pre|post|type): {content_hash()}")
    if os.path.exists(os.path.join(DATA, "cells.parquet")):
        print("\ngraph shape:\n" + json.dumps(graph_shape(), indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "list":
        its = list_bucket()
        log(f"{len(its)} objects, {sum(int(i['size']) for i in its) / 1e9:.3f} GB")
    elif cmd == "download":
        download()
    elif cmd == "parse":
        parse(); build_cells()
    elif cmd == "cells":
        build_cells()
    elif cmd == "check":
        check()
    elif cmd == "report":
        report()
    elif cmd == "all":
        download(); parse(); build_cells(); check()
    else:
        sys.exit(__doc__)
