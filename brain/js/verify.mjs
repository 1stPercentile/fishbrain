// FISHBRAIN Verify: re-run one decision window in the viewer's browser and check its receipt.
//
//   import { verify } from "./verify.mjs";
//   const r = await verify(receipt, fetcher, { trustedKernels: ["<sha256 of the published kernel>"] });
//   r.ok ? show(r.raster) : explain(r.check, r.detail);
//
// This module has NO static imports. The brain kernel (fishbrain-sim.mjs) is fetched once, copied
// into memory this module alone holds, hashed with WebCrypto, and imported from exactly those bytes
// (a blob: URL in browsers, a data: URL where blob: modules are unsupported, e.g. Node). Nothing
// else is ever executed as the kernel, so a file swapped on the server, a second fetch, or a buffer
// the fetcher edits after returning it cannot change what runs. The kernel and the case are fetched
// once per call, and every link of a checkpoint chain reuses them.
//
// verify() FAILS unless every check below holds, and says which one failed (result.check):
//   kernel    the receipt names a kernel this verifier trusts; the fetched bytes hash to it; the code
//             executed is those bytes; name and version match the receipt.
//   seed      receipt.seed is a canonical 32-byte Solana blockhash, not degenerate (at least 8
//             distinct byte values); the simulation seed string is "fishbrain-seed-v1:solana-blockhash:"
//             + value (never a case file's seed). At start_step 0 the state built from that seed must
//             equal window.snapshot (network, params and inputs are already verified, so only the seed
//             is left to differ). From a checkpoint, its recorded seed must be that string and every
//             random stream must sit exactly where that seed puts it at start_step (fast-forwarded
//             without the network). With opts.chainLookup, a chain answer that contradicts the
//             receipt's seed (see "seed anchor" below) also fails here.
//   network   Network.digest() of the served wiring equals receipt.brain.network.
//   bridge    which part of each pair's count is a measured Fish1 synapse and which a bridge link
//             (G1c class a, b, c). receipt.brain.bridge is the sha256 of the network's provenance table
//             (fishbrain-bridge-table-v1), or null for an unbridged network. Non-null: the fetched table
//             hashes to it, parses canonically, lists exactly the network's pairs in CSR order, and on
//             every pair measured + bridged per class equals the network's count (every count
//             attributed once: none missing, none double-counted); it must attribute at least one count
//             to a bridge class (an unbridged network declares null). Null: nothing is fetched and every
//             count is measured. With opts.trustedBrains ([{network, bridge}], the pairs the page
//             re-derived from the frozen spec), the receipt's (network, bridge) must be one of them.
//             result.bridge carries the verified table, a per-link class mask for colouring, the
//             synapse-count share and the recomputed edge-provenance and bridged digests. Chain links
//             skip it (a link checks a state transition; provenance does not change dynamics).
//   params    LIFParams.digest() equals receipt.brain.params, and dt equals receipt.window.dt_ms.
//   inputs    sha256 of the complete input specification (fishbrain-inputs-v1: every source's name,
//             kind, targets, rates, frame length, weight, start/stop, currents) equals
//             receipt.inputs.spec; and receipt.inputs.frames, when present, equals the sha256 of the
//             "retina" source's rates (what the fish saw).
//   snapshot  (start_step > 0) the checkpoint parses, is self-consistent, and its state digest,
//             computed from its bytes before anything is restored, equals receipt.window.snapshot;
//             it is for this step, network, parameters and inputs.
//   anchor    the start state is one the run really reached. Step 0 is anchored by construction
//             (genesis). Past step 0 a self-consistent checkpoint proves nothing by itself (a
//             fabricated u/h resealed is self-consistent), so verify() FAILS CLOSED unless either
//               (a) opts.trustedCheckpoints has an entry {step, snapshot, network, params, inputs,
//                   seed} equal to this window's start, which the page has itself checked (e.g. the
//                   checkpoint's on-chain Memo commitment), or
//               (b) a chain of links (opts.chain, or fetcher("chain")) runs from genesis, or from a
//                   trusted checkpoint, to exactly (start_step, window.snapshot), and every link
//                   replays: restore its `from`, run to its `to`, reach its `to.snapshot`.
//   spikes    the replayed window's spike hash equals receipt.brain.spikes.
//   state     the state digest after the window equals receipt.brain.state.
//   receipt   the receipt is missing a field or has one of the wrong type.
//
// Seed anchor (result.seed_anchored). A well-formed blockhash is still one the operator could have
// picked (ground). It is unpickable only when it is the real blockhash of the slot a run-commit Memo
// pinned before that slot existed. With opts.chainLookup and receipt.seed.commit (the run-commit's
// transaction signature), verify reads the memo "fishbrain:run:v1 network=<hex> params=<hex>
// kernel=<hex> seed_slot=<n>", requires it to name this brain and to have landed in a slot before
// seed_slot, and requires seed_slot's blockhash to equal receipt.seed.value. Then seed_anchored is
// true. Without a lookup, or when the lookup cannot answer, seed_anchored is false (with a reason) and
// ok is unaffected; a chain answer that contradicts the receipt fails at "seed".
//   chainLookup({kind: "transaction", signature}) -> {slot, memo} | null   (getTransaction)
//   chainLookup({kind: "block", slot})            -> {blockhash} | null    (getBlock; null = skipped)
//   It should throw when it cannot answer (RPC down, history pruned): that is not a forgery.
//
// fetcher(name, receipt) returns the bytes (Uint8Array, ArrayBuffer, or a fetch Response) of:
//   "kernel"      the kernel file the receipt names (e.g. by receipt.kernel.sha256)
//   "case"        a fishbrain-js-case-v1 JSON (network, params, sources; its seed is ignored)
//   "checkpoint"  the fishbrain-checkpoint-v1 state at receipt.window.start_step, content-addressed
//                 by receipt.window.snapshot; only fetched when start_step > 0. Chain links call it
//                 with a receipt whose window names the link's `from`.
//   "chain"       (only when no trusted checkpoint anchors the window and opts.chain is absent) a
//                 JSON list of links [{from: {step, snapshot}, to: {step, snapshot}}, ...], or
//                 {links: [...]}. Throwing or returning nothing means "no chain".
//   "bridge"      the fishbrain-bridge-table-v1 bytes, content-addressed by receipt.brain.bridge; only
//                 fetched when that is not null.
//
// Needs WebCrypto (crypto.subtle: https pages and Node >= 19), BigInt and dynamic import().

export const SEED_PREFIX = "fishbrain-seed-v1:solana-blockhash:";
export const SEED_SOURCE = "solana blockhash";
export const KERNEL_NAME = "fishbrain-sim";
export const CASE_FORMAT = "fishbrain-js-case-v1";
export const RUN_MEMO_TAG = "fishbrain:run:v1";
export const MIN_DISTINCT_BYTES = 8;
export const CHECK_ORDER = ["receipt", "kernel", "seed", "network", "bridge", "params", "inputs", "snapshot", "anchor", "spikes", "state"];
export const BRIDGE_TABLE_FORMAT = "fishbrain-bridge-table-v1";
export const BRIDGE_TABLE_MAGIC = BRIDGE_TABLE_FORMAT + "\n";
export const BRIDGE_MAX_ABS = 2 ** 48;
export const BRIDGE_MAX_CLASSES = 8;
const BRIDGE_CLASS_RE = /^[a-z]$/;   // class names are single lowercase letters: no '__proto__', no 'measured'
export const PROVENANCE_VERSION = "fishbrain-provenance-v1";
export const DECISION_PATH_SHARE_SOURCE = "server-computed: fishbrain/provenance.py traces every gate trial (evidence/G1c-gate.md s.5); verify returns the verified table, not the share";

// ---------------------------------------------------------------------------------------------
// seed rule

const B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

function b58decode(s) {
  let n = 0n;
  for (const ch of s) {
    const v = B58.indexOf(ch);
    if (v < 0) return null;
    n = n * 58n + BigInt(v);
  }
  let pad = 0;
  while (pad < s.length && s[pad] === "1") pad++;
  const body = [];
  while (n > 0n) { body.unshift(Number(n & 0xffn)); n >>= 8n; }
  return Uint8Array.from([...new Array(pad).fill(0), ...body]);
}

function b58encode(bytes) {
  let n = 0n;
  for (const b of bytes) n = (n << 8n) | BigInt(b);
  let out = "";
  while (n > 0n) { out = B58[Number(n % 58n)] + out; n /= 58n; }
  let pad = 0;
  while (pad < bytes.length && bytes[pad] === 0) pad++;
  return "1".repeat(pad) + out;
}

/**
 * The simulation seed string for receipt.seed (fishbrain-seed-v1). Throws unless canonical and not
 * degenerate. A blockhash is a SHA-256 output: fewer than MIN_DISTINCT_BYTES distinct byte values
 * (all zero, all 0xff, a short repeated pattern) happens by chance with probability below 1e-37.
 * Passing this rule does NOT make a seed unpickable; see the seed anchor (opts.chainLookup).
 */
export function seedString(seed) {
  if (!seed || seed.source !== SEED_SOURCE) throw new Error(`seed.source must be "${SEED_SOURCE}"`);
  const v = seed.value;
  if (typeof v !== "string" || v.length < 32 || v.length > 44) throw new Error("seed.value must be 32 to 44 base58 characters");
  const raw = b58decode(v);
  if (!raw || raw.length !== 32 || b58encode(raw) !== v) throw new Error("seed.value is not the canonical base58 of a 32-byte blockhash");
  const distinct = new Set(raw).size;
  if (distinct < MIN_DISTINCT_BYTES) {
    throw new Error(`seed.value is a degenerate blockhash (${distinct} distinct byte values; a real one is a SHA-256 output)`);
  }
  return SEED_PREFIX + v;
}

/** Parse a run-commit memo "fishbrain:run:v1 key=value ..." into {key: value}; null if it is not one. */
export function parseRunMemo(text) {
  if (typeof text !== "string") return null;
  const tok = text.trim().split(/\s+/);
  if (tok[0] !== RUN_MEMO_TAG) return null;
  const kv = {};
  for (const t of tok.slice(1)) {
    const i = t.indexOf("=");
    if (i <= 0) return null;
    const k = t.slice(0, i);
    if (Object.prototype.hasOwnProperty.call(kv, k)) return null;      // a repeated key is ambiguous
    kv[k] = t.slice(i + 1);
  }
  return kv;
}

/**
 * Is receipt.seed.value the real blockhash of the slot a run-commit pinned before that slot existed?
 * Returns {anchored: true, slot, commit_slot, commit} or {anchored: false, reason[, contradiction]}.
 * A contradiction is the chain saying something the receipt denies; verify() fails at "seed" on it.
 */
async function seedAnchor(receipt, lookup) {
  const no = (reason) => ({ anchored: false, reason });
  const bad = (reason) => ({ anchored: false, reason, contradiction: true });
  if (typeof lookup !== "function") {
    return no("no chainLookup was given: the blockhash is well-formed, but nothing checked it is the real blockhash of the slot a run-commit pinned, so it could have been picked");
  }
  const sig = receipt.seed.commit;
  if (typeof sig !== "string" || !sig.length) return no("the receipt names no run-commit transaction (seed.commit)");
  let tx;
  try { tx = await lookup({ kind: "transaction", signature: sig }); } catch (e) { return no(`chainLookup could not read the run-commit ${sig}: ${e && e.message}`); }
  if (tx == null) return no(`chainLookup found no transaction ${sig}`);
  if (!isInt(tx.slot)) return no(`chainLookup returned no slot for the run-commit ${sig}`);
  const kv = parseRunMemo(tx.memo);
  if (!kv) return bad(`transaction ${sig} is not a ${RUN_MEMO_TAG} run-commit memo`);
  const want = [["network", receipt.brain.network], ["params", receipt.brain.params], ["kernel", receipt.kernel.sha256]];
  for (const [k, v] of want) {
    if (kv[k] !== v) return bad(`the run-commit ${sig} commits to ${k} ${kv[k] === undefined ? "(none)" : kv[k]}, the receipt's is ${v}`);
  }
  if (!/^[0-9]{1,16}$/.test(kv.seed_slot || "")) {
    return bad(`the run-commit ${sig} does not pin a seed slot (seed_slot=): any later blockhash could have been chosen`);
  }
  const seedSlot = Number(kv.seed_slot);
  if (!(seedSlot > tx.slot)) {
    return bad(`the run-commit landed in slot ${tx.slot}, not before the seed slot ${seedSlot} it pins: the blockhash could have been known when the brain was committed`);
  }
  let block;
  try { block = await lookup({ kind: "block", slot: seedSlot }); } catch (e) { return no(`chainLookup could not read slot ${seedSlot}: ${e && e.message}`); }
  if (block == null) return bad(`slot ${seedSlot} has no block (skipped), so no blockhash of it can seed this run`);
  if (block.blockhash !== receipt.seed.value) {
    return bad(`slot ${seedSlot}'s blockhash is ${block.blockhash}, the receipt's seed is ${receipt.seed.value}`);
  }
  return { anchored: true, slot: seedSlot, commit_slot: tx.slot, commit: sig };
}

// ---------------------------------------------------------------------------------------------
// bytes, hashing, and importing the kernel from verified bytes

/** A private copy of whatever the fetcher returned: nobody else holds a reference to it. */
async function privateBytes(x) {
  if (x && typeof x.arrayBuffer === "function" && !ArrayBuffer.isView(x)) {
    if (x.ok === false) throw new Error(`the server answered ${x.status}`);
    x = await x.arrayBuffer();
  }
  if (x instanceof ArrayBuffer) return new Uint8Array(x.slice(0));
  if (ArrayBuffer.isView(x)) return new Uint8Array(new Uint8Array(x.buffer, x.byteOffset, x.byteLength));   // copies
  throw new Error("the fetcher must return bytes (Uint8Array, ArrayBuffer or a Response)");
}

async function webSha256Hex(bytes) {
  const subtle = globalThis.crypto && globalThis.crypto.subtle;
  if (!subtle) throw new Error("WebCrypto (crypto.subtle) is unavailable: Verify needs https or Node >= 19");
  const d = new Uint8Array(await subtle.digest("SHA-256", bytes));
  let s = "";
  for (const b of d) s += (b < 16 ? "0" : "") + b.toString(16);
  return s;
}

function base64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

let importCount = 0;

/** Evaluate `bytes` as an ES module and return its namespace; a fresh module instance every call. */
async function importFromBytes(bytes) {
  if (typeof Blob === "function" && typeof URL !== "undefined" && typeof URL.createObjectURL === "function") {
    let url = null;
    try {
      url = URL.createObjectURL(new Blob([bytes], { type: "text/javascript" }));
      return await import(url);
    } catch (e) {
      // blob: modules are unsupported here (Node) or blocked by policy: fall back to data:
      if (!(e && (e.code === "ERR_UNSUPPORTED_ESM_URL_SCHEME" || e instanceof TypeError))) throw e;
    } finally {
      if (url) URL.revokeObjectURL(url);
    }
  }
  // a distinct fragment gives a distinct module record; the executed source is the base64 payload
  return await import(`data:text/javascript;base64,${base64(bytes)}#fishbrain-verify-${++importCount}`);
}

/** Fetch, hash and import the kernel a receipt names. Returns {F} or {fail: detail}. */
async function loadKernel(kernel, fetcher, ctx, trustedKernels) {
  if (!Array.isArray(trustedKernels) || !trustedKernels.length) {
    return { fail: "no trusted kernel list was given (opts.trustedKernels): a kernel hash alone says which code ran, not that it is honest" };
  }
  if (!trustedKernels.includes(kernel.sha256)) return { fail: `the receipt names kernel ${kernel.sha256}, which this verifier does not trust` };
  if (kernel.name !== KERNEL_NAME) return { fail: `kernel name ${kernel.name} is not ${KERNEL_NAME}` };
  const bytes = await privateBytes(await fetcher("kernel", ctx));
  const sha = await webSha256Hex(bytes);
  if (sha !== kernel.sha256) return { fail: `the fetched kernel hashes to ${sha}, not ${kernel.sha256}` };
  let F;
  try { F = await importFromBytes(bytes); } catch (e) { return { fail: `the kernel did not load: ${e && e.message}` }; }
  if (F.KERNEL_VERSION !== kernel.version) return { fail: `the kernel is ${F.KERNEL_VERSION}, the receipt says ${kernel.version}` };
  return { F, sha };
}

// ---------------------------------------------------------------------------------------------
// bridge provenance: fishbrain-bridge-table-v1 (the byte layout is in fishbrain/bridgetable.py)

const GAIN_KEY = /^g_[A-Za-z0-9_]+$/;
const TWO32 = 4294967296;

function asciiBytes(s, what, emptyOk = false) {
  if (typeof s !== "string") throw new Error(`${what} must be a string`);
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c < 0x20 || c > 0x7e) throw new Error(`${what} must be printable ASCII`);
  }
  if (!s.length && !emptyOk) throw new Error(`${what} must not be empty`);
  const b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i);
  return b;
}

function boundedInt(v, what) {
  if (!Number.isInteger(v) || v > BRIDGE_MAX_ABS || v < -BRIDGE_MAX_ABS) throw new Error(`${what}: a value is not an integer within +-2^48`);
  return v;
}

/**
 * The canonical fishbrain-bridge-table-v1 bytes. t = {n, pre, post, measured, bridged (P*K row-major or
 * P rows of K), classes, bridge_version, edge_list_sha256, gains: {g_x: int}, ablate, topology}. Throws on
 * anything parseBridgeTable would refuse. Python bridgetable.encode_parts writes the same bytes.
 */
export function encodeBridgeTable(t) {
  const classes = [...t.classes].map(String);
  if (classes.length > BRIDGE_MAX_CLASSES || new Set(classes).size !== classes.length) throw new Error("classes must be unique, at most 8");
  for (const c of classes) if (!BRIDGE_CLASS_RE.test(c)) throw new Error(`class name ${JSON.stringify(c)} is not a single lowercase letter`);
  if (!HEX64.test(t.edge_list_sha256 || "")) throw new Error("edge_list_sha256 must be 64 lowercase hex characters");
  const keys = Object.keys(t.gains || {}).sort();
  for (const k of keys) if (!GAIN_KEY.test(k)) throw new Error(`gain key ${k} must match g_<name>`);
  const abl = [...new Set((t.ablate || []).map(String))].sort();
  if (abl.length !== (t.ablate || []).length) throw new Error("ablations must be unique");
  const K = classes.length, P = t.pre.length, n = t.n;
  const flat = [];
  if (P && Array.isArray(t.bridged) && Array.isArray(t.bridged[0])) for (const row of t.bridged) flat.push(...row);
  else for (const v of t.bridged) flat.push(v);
  if (t.post.length !== P || t.measured.length !== P || flat.length !== P * K) throw new Error("pre, post, measured and bridged must have one row per pair");
  for (let k = 0; k < P; k++) {
    boundedInt(t.pre[k], "pre"); boundedInt(t.post[k], "post"); boundedInt(t.measured[k], "measured");
    if (t.pre[k] < 0 || t.post[k] < 0 || t.pre[k] >= n || t.post[k] >= n) throw new Error("a pair's neuron index is outside the network");
    if (k && (t.pre[k] < t.pre[k - 1] || (t.pre[k] === t.pre[k - 1] && t.post[k] <= t.post[k - 1]))) throw new Error("pairs must be strictly increasing in (pre, post)");
  }
  for (const v of flat) boundedInt(v, "bridged");
  const strs = [];
  const str = (s, what, emptyOk) => { const b = asciiBytes(s, what, emptyOk); strs.push(b); return 4 + b.length; };
  let head = BRIDGE_TABLE_MAGIC.length;
  head += str(t.bridge_version, "bridge version") + 32 + str(t.topology == null ? "" : t.topology, "topology", true) + 4;
  for (const c of classes) head += str(c, "class name");
  head += 4;
  for (const k of keys) head += str(k, "gain key") + 8;
  head += 4;
  for (const a of abl) head += str(a, "ablation");
  head += 16;
  const out = new Uint8Array(head + 8 * P * (3 + K));
  const dv = new DataView(out.buffer);
  let o = 0, si = 0;
  const putStr = () => { const b = strs[si++]; dv.setUint32(o, b.length, true); o += 4; out.set(b, o); o += b.length; };
  const putI64 = (v) => { dv.setBigInt64(o, BigInt(v), true); o += 8; };
  for (let i = 0; i < BRIDGE_TABLE_MAGIC.length; i++) out[o++] = BRIDGE_TABLE_MAGIC.charCodeAt(i);   // ends in "\n"
  putStr();
  for (let i = 0; i < 32; i++) out[o++] = parseInt(t.edge_list_sha256.slice(2 * i, 2 * i + 2), 16);
  putStr();
  dv.setUint32(o, K, true); o += 4;
  for (let q = 0; q < K; q++) putStr();
  dv.setUint32(o, keys.length, true); o += 4;
  for (const k of keys) { putStr(); putI64(boundedInt(t.gains[k], `gain ${k}`)); }
  dv.setUint32(o, abl.length, true); o += 4;
  for (let a = 0; a < abl.length; a++) putStr();
  dv.setBigUint64(o, BigInt(n), true); o += 8;
  dv.setBigUint64(o, BigInt(P), true); o += 8;
  for (const col of [t.pre, t.post, t.measured, flat]) for (let k = 0; k < col.length; k++) putI64(col[k]);
  if (o !== out.length) throw new Error("bridge table size mismatch");
  return out;
}

/**
 * Parse fishbrain-bridge-table-v1 and check canonical form. Returns {format, bridge_version,
 * edge_list_sha256, topology, classes, gains, ablate, n, pairs, pre, post, measured (Float64Array, P),
 * bridged (Float64Array, P*K row-major)}. Throws on truncation, trailing bytes, non-ASCII text, unsorted
 * gain keys, ablations or pairs, an index outside n, or a value outside +-2^48.
 */
export function parseBridgeTable(bytes) {
  const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let o = 0;
  const need = (k) => { if (k < 0 || o + k > b.length) throw new Error("truncated table"); };
  const u32 = () => { need(4); const v = dv.getUint32(o, true); o += 4; return v; };
  const i64 = (what) => {
    need(8);
    const v = dv.getInt32(o + 4, true) * TWO32 + dv.getUint32(o, true);
    o += 8;
    if (v > BRIDGE_MAX_ABS || v < -BRIDGE_MAX_ABS) throw new Error(`${what}: a value is outside +-2^48`);
    return v;
  };
  const text = (what, emptyOk = false) => {
    const len = u32(); need(len);
    let s = "";
    for (let i = 0; i < len; i++) s += String.fromCharCode(b[o + i]);
    o += len;
    asciiBytes(s, what, emptyOk);
    return s;
  };
  need(BRIDGE_TABLE_MAGIC.length);
  for (let i = 0; i < BRIDGE_TABLE_MAGIC.length; i++) if (b[i] !== BRIDGE_TABLE_MAGIC.charCodeAt(i)) throw new Error(`not a ${BRIDGE_TABLE_FORMAT} table`);
  o = BRIDGE_TABLE_MAGIC.length;
  const t = { format: BRIDGE_TABLE_FORMAT, bridge_version: text("bridge version") };
  need(32);
  let sha = "";
  for (let i = 0; i < 32; i++) sha += (b[o + i] < 16 ? "0" : "") + b[o + i].toString(16);
  o += 32;
  t.edge_list_sha256 = sha;
  t.topology = text("topology", true);
  const K = u32();
  if (K > BRIDGE_MAX_CLASSES) throw new Error("more than 8 classes");
  t.classes = [];
  for (let q = 0; q < K; q++) t.classes.push(text("class name"));
  if (new Set(t.classes).size !== K) throw new Error("class names must be unique");
  for (const c of t.classes) if (!BRIDGE_CLASS_RE.test(c)) throw new Error(`class name ${JSON.stringify(c)} is not a single lowercase letter`);
  const G = u32();
  t.gains = {};
  let prev = null;
  for (let g = 0; g < G; g++) {
    const k = text("gain key");
    if (!GAIN_KEY.test(k)) throw new Error(`gain key ${k} must match g_<name>`);
    if (prev !== null && !(k > prev)) throw new Error("gain keys must be strictly increasing");
    t.gains[k] = i64(`gain ${k}`);
    prev = k;
  }
  const A = u32();
  t.ablate = [];
  for (let a = 0; a < A; a++) {
    const s = text("ablation");
    if (a && !(s > t.ablate[a - 1])) throw new Error("ablations must be strictly increasing");
    t.ablate.push(s);
  }
  need(16);
  const n = Number(dv.getBigUint64(o, true)), P = Number(dv.getBigUint64(o + 8, true));
  o += 16;
  const rest = b.length - o;
  if (!Number.isSafeInteger(P) || rest !== 8 * P * (3 + K)) {
    throw new Error(`the table holds ${rest} bytes of rows, ${P} pairs of ${K} classes need ${8 * P * (3 + K)}${rest > 8 * P * (3 + K) ? " (trailing bytes)" : ""}`);
  }
  const col = (len, what) => { const a = new Float64Array(len); for (let k = 0; k < len; k++) a[k] = i64(what); return a; };
  t.n = n; t.pairs = P;
  t.pre = col(P, "pre"); t.post = col(P, "post"); t.measured = col(P, "measured"); t.bridged = col(P * K, "bridged");
  for (let k = 0; k < P; k++) {
    if (t.pre[k] < 0 || t.post[k] < 0 || t.pre[k] >= n || t.post[k] >= n) throw new Error("a pair's neuron index is outside the table's network");
    if (k && (t.pre[k] < t.pre[k - 1] || (t.pre[k] === t.pre[k - 1] && t.post[k] <= t.post[k - 1]))) throw new Error("pairs must be strictly increasing in (pre, post)");
  }
  return t;
}

/**
 * Does table `t` account for every count of `net` exactly once? null if so, else the first difference.
 * Same pairs as the network's CSR, in its order; on every pair measured + bridged per class = the count.
 */
export function bridgeAttributionError(t, net) {
  if (t.n !== net.n) return `the table is for a network of ${t.n} neurons, this one has ${net.n}`;
  if (t.pairs !== net.nnz) return `the table lists ${t.pairs} pairs, the network has ${net.nnz}: every pair must be attributed exactly once`;
  const K = t.classes.length;
  for (let i = 0, k = 0; i < net.n; i++) {
    for (; k < net.indptr[i + 1]; k++) {
      if (t.pre[k] !== i || t.post[k] !== net.indices[k]) return `pair ${k}: the table names (${t.pre[k]}, ${t.post[k]}), the network has (${i}, ${net.indices[k]})`;
      let sum = t.measured[k];
      for (let q = 0; q < K; q++) sum += t.bridged[k * K + q];
      if (sum !== net.counts[k]) return `pair (${i}, ${net.indices[k]}): measured ${t.measured[k]} + bridged ${sum - t.measured[k]} = ${sum}, the network's count is ${net.counts[k]}`;
    }
  }
  return null;
}

/** provenance.synapse_count_share's numbers from a verified table (|signed count| per pair part). */
export function bridgeSummary(t) {
  const K = t.classes.length;
  let meas = 0, withPart = 0, only = 0;
  const per = new Array(K).fill(0);
  for (let k = 0; k < t.pairs; k++) {
    meas += Math.abs(t.measured[k]);
    let has = false;
    for (let q = 0; q < K; q++) { const v = t.bridged[k * K + q]; per[q] += Math.abs(v); if (v !== 0) has = true; }
    if (has) { withPart++; if (t.measured[k] === 0) only++; }
  }
  const tot = meas + per.reduce((a, v) => a + v, 0);
  const synapses = { measured: meas, total: tot }, classShare = {};
  t.classes.forEach((c, q) => { synapses[c] = per[q]; classShare[c] = tot ? per[q] / tot : null; });
  return { measured_share: tot ? meas / tot : null, class_share: classShare, synapses, pairs: t.pairs,
           pairs_with_bridge_part: withPart, pairs_bridge_only: only };
}

function i64Bytes(values) {
  const out = new Uint8Array(8 * values.length);
  const dv = new DataView(out.buffer);
  for (let k = 0; k < values.length; k++) dv.setBigInt64(8 * k, BigInt(values[k]), true);
  return out;
}

function concatBytes(parts) {
  const out = new Uint8Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/** EdgeProvenance.digest() recomputed from a table: json.dumps(classes) keeps Python's ", " separator. */
export async function edgeProvenanceDigest(t) {
  const K = t.classes.length, pre = [], post = [], parts = [];
  for (let k = 0; k < t.pairs; k++) {
    let has = false;
    for (let q = 0; q < K; q++) if (t.bridged[k * K + q] !== 0) has = true;
    if (!has) continue;
    pre.push(t.pre[k]); post.push(t.post[k]);
    for (let q = 0; q < K; q++) parts.push(t.bridged[k * K + q]);
  }
  const head = asciiBytes(`${PROVENANCE_VERSION}|edges[${t.classes.map((c) => JSON.stringify(c)).join(", ")}]`, "provenance header");
  const counts = new Uint8Array(16);
  new DataView(counts.buffer).setBigUint64(0, BigInt(pre.length), true);
  new DataView(counts.buffer).setBigUint64(8, BigInt(K), true);
  return webSha256Hex(concatBytes([head, counts, i64Bytes(pre), i64Bytes(post), i64Bytes(parts)]));
}

/** bridge.bridged_digest(...) recomputed: sha256 of json.dumps(obj, sort_keys=True, separators=(",", ":")). */
export async function bridgedDigest(t, networkDigest) {
  const obj = { network_digest: networkDigest, bridge_version: t.bridge_version, edge_list_sha256: t.edge_list_sha256,
                ...t.gains, ablate: [...t.ablate], topology: t.topology, edge_provenance: await edgeProvenanceDigest(t) };
  const text = "{" + Object.keys(obj).sort().map((k) => `${JSON.stringify(k)}:${JSON.stringify(obj[k])}`).join(",") + "}";
  return webSha256Hex(asciiBytes(text, "bridged digest payload"));
}

/** opts.trustedBrains as a list of {network, bridge} (bridge: sha256 or null), or {error}; null when absent. */
function trustedBrainList(x) {
  if (x == null) return { list: null };
  const list = x instanceof Set ? [...x] : x;
  if (!Array.isArray(list)) return { error: "opts.trustedBrains must be a list" };
  for (const e of list) {
    if (!(e && typeof e === "object" && HEX64.test(e.network || "") && (e.bridge === null || HEX64.test(e.bridge || "")))) {
      return { error: "each opts.trustedBrains entry must be {network, bridge} (bridge: the sha256 of its provenance table, or null for an unbridged network): a bare network digest does not say which of its links are measured" };
    }
  }
  return { list };
}

/**
 * The bridge check. Returns {bridge: result.bridge} or {fail: detail}. result.bridge = {sha256, bridged,
 * trusted, trust_reason?, classes, gains, ablate, topology, bridge_version, edge_list_sha256, edge_provenance,
 * bridged_digest, synapse_count_share, decision_path_share, mask, table}. mask[k] has bit q set when CSR
 * entry k carries a class-q bridge part (colour it with --bridge); table is the verified table.
 */
async function checkBridge(receipt, fetcher, net, netDigest, opts) {
  const tb = trustedBrainList(opts.trustedBrains);
  if (tb.error) return { fail: tb.error };
  const claim = receipt.brain.bridge;
  const mask = new Uint8Array(net.nnz);
  let out;
  if (claim === null) {
    let meas = 0;
    for (let k = 0; k < net.nnz; k++) meas += Math.abs(net.counts[k]);
    out = { sha256: null, bridged: false, classes: [], gains: {}, ablate: [], topology: null, bridge_version: null,
            edge_list_sha256: null, edge_provenance: null, bridged_digest: null,
            synapse_count_share: { measured_share: meas ? 1 : null, class_share: {}, synapses: { measured: meas, total: meas },
                                   pairs: net.nnz, pairs_with_bridge_part: 0, pairs_bridge_only: 0 },
            mask, table: null };
  } else {
    const bytes = await privateBytes(await fetcher("bridge", receipt));
    const sha = await webSha256Hex(bytes);
    if (sha !== claim) return { fail: `the served provenance table hashes to ${sha}, the receipt says ${claim}` };
    let t;
    try { t = parseBridgeTable(bytes); } catch (e) { return { fail: `the provenance table is malformed: ${e.message}` }; }
    const bad = bridgeAttributionError(t, net);
    if (bad) return { fail: bad };
    const s = bridgeSummary(t);
    if (!s.pairs_with_bridge_part) {
      return { fail: "the provenance table attributes no count to a bridge class: an unbridged network declares brain.bridge null, not a table" };
    }
    const K = t.classes.length;
    for (let k = 0; k < t.pairs; k++) for (let q = 0; q < K; q++) if (t.bridged[k * K + q] !== 0) mask[k] |= 1 << q;
    out = { sha256: sha, bridged: true, classes: t.classes, gains: t.gains, ablate: t.ablate, topology: t.topology,
            bridge_version: t.bridge_version, edge_list_sha256: t.edge_list_sha256,
            edge_provenance: await edgeProvenanceDigest(t), bridged_digest: await bridgedDigest(t, netDigest),
            synapse_count_share: s, mask, table: t };
  }
  // Fail closed like trustedKernels (2026-09-26 bridge forge review): without a trust list a count-swap table
  // passes with every summary number identical to the honest one, so "untrusted but ok" hides the lie.
  if (!tb.list) {
    return { fail: "opts.trustedBrains is required: without it nothing ties this network's measured/bridged split to the published build (a relabelled table can keep every summary number identical)" };
  }
  if (tb.list) {
    if (!tb.list.some((e) => e.network === netDigest && e.bridge === claim)) {
      return { fail: `no opts.trustedBrains entry vouches for network ${netDigest} with ${claim === null ? "no bridge (brain.bridge null)" : `provenance table ${claim}`}` };
    }
    out.trusted = true;
  } else {
    out.trusted = false;
    out.trust_reason = "no opts.trustedBrains was given: the table accounts for every count of this network, but nothing checked that its measured/bridged split is the published one (a relabelled table with the receipt's hash recomputed also accounts for every count), nor that a null bridge is true";
  }
  out.decision_path_share = { computed: false, source: DECISION_PATH_SHARE_SOURCE };
  return { bridge: out };
}

// ---------------------------------------------------------------------------------------------
// receipts

const HEX64 = /^[0-9a-f]{64}$/;
const isInt = (x) => Number.isSafeInteger(x) && x >= 0;
const isPoint = (p) => !!p && isInt(p.step) && HEX64.test(p.snapshot || "");

function shapeError(r, { link = false } = {}) {
  const need = (cond, what) => (cond ? null : what);
  if (!r || typeof r !== "object") return "the receipt is not an object";
  return need(r.kernel && typeof r.kernel.name === "string" && typeof r.kernel.version === "string" && HEX64.test(r.kernel.sha256 || ""), "kernel {name, version, sha256}")
    || need(r.seed && typeof r.seed === "object", "seed {source, value}")
    || need(r.brain && HEX64.test(r.brain.network || "") && HEX64.test(r.brain.params || "") && HEX64.test(r.brain.state || "") && (link || HEX64.test(r.brain.spikes || "")), "brain {network, params, state, spikes}")
    || need(link || r.brain.bridge === null || HEX64.test(r.brain.bridge || ""), "brain.bridge (the sha256 of the network's provenance table, or null for an unbridged network; absent is not null)")
    || need(r.inputs && HEX64.test(r.inputs.spec || "") && (r.inputs.frames == null || HEX64.test(r.inputs.frames)), "inputs {spec, frames?}")
    || need(r.window && isInt(r.window.start_step) && isInt(r.window.n_steps) && (link || typeof r.window.dt_ms === "number") && HEX64.test(r.window.snapshot || ""), "window {start_step, n_steps, dt_ms, snapshot}")
    || null;
}

function decodeValues(F, x) {
  return x && typeof x === "object" && "f64_b64" in x ? { data: F.decodeF64(x.f64_b64), shape: x.shape } : x;
}

/** The receipt the fetcher sees for a state at `point` of this run (content-addressed by its snapshot). */
function receiptAt(receipt, point) {
  return { ...receipt, window: { ...receipt.window, start_step: point.step, snapshot: point.snapshot } };
}

/**
 * The state at (step, snapshot), checked: at step 0 the genesis built from the seed; past it the
 * fetched checkpoint, bound to this run and its streams to the seed. Returns {sim} or {check, detail}.
 * onPass(check) is told each check that held (the window's own start reports them; chain links do not).
 */
async function startState(ctx, step, snapshot, fetchReceipt, onPass = () => {}) {
  const { F } = ctx;
  if (step === 0) {
    const sim0 = ctx.buildSim();
    const d0 = sim0.stateDigest();
    if (d0 !== snapshot) {
      return { check: "seed", detail: `the step-0 state built from seed ${ctx.seedValue} hashes to ${d0}, not ${snapshot}; wiring, parameters and inputs already match, so the seed is what differs` };
    }
    onPass("snapshot"); onPass("seed");
    return { sim: sim0 };
  }
  const ckBytes = await privateBytes(await ctx.fetcher("checkpoint", fetchReceipt));
  let ck;
  try { ck = F.parseCheckpoint(ckBytes); } catch (e) { return { check: "snapshot", detail: e.message }; }
  if (ck.stateDigest !== snapshot) return { check: "snapshot", detail: `the checkpoint's state hashes to ${ck.stateDigest}, the expected snapshot is ${snapshot}` };
  if (ck.step !== step) return { check: "snapshot", detail: `the checkpoint is at step ${ck.step}, not ${step}` };
  if (ck.networkDigest !== ctx.netDigest || ck.paramsDigest !== ctx.parDigest || ck.inputsDigest !== ctx.spec) {
    return { check: "snapshot", detail: "the checkpoint was made for other wiring, parameters or inputs" };
  }
  onPass("snapshot");
  if (ck.seed !== ctx.seed) return { check: "seed", detail: `the checkpoint's run was seeded with "${ck.seed}", the receipt's seed gives "${ctx.seed}"` };
  const fresh = ctx.buildSim();
  F.fastForwardSources(fresh.sources, step);
  const want = F.sourceRecords(fresh.sources), got = F.sourceRecords(ck.sources);
  for (let q = 0; q < want.length; q++) {
    if (JSON.stringify(want[q]) !== JSON.stringify(got[q])) {
      return { check: "seed", detail: `source ${want[q].name}'s random stream is not where seed ${ctx.seedValue} puts it at step ${step}` };
    }
  }
  onPass("seed");
  try { return { sim: F.restoreCheckpoint(ckBytes, { net: ctx.net, params: ctx.params, inputs: ctx.inputs, gated: ctx.gated }) }; }
  catch (e) { return { check: "snapshot", detail: e.message }; }
}

/** opts.trustedCheckpoints as a list of full anchor tuples, or {error}. */
function trustedList(x) {
  if (x == null) return { list: [] };
  const list = x instanceof Set ? [...x] : x;
  if (!Array.isArray(list)) return { error: "opts.trustedCheckpoints must be a list" };
  for (const t of list) {
    const ok = t && typeof t === "object" && isPoint(t) && HEX64.test(t.network || "") && HEX64.test(t.params || "")
      && HEX64.test(t.inputs || "") && typeof t.seed === "string";
    if (!ok) {
      return { error: "each opts.trustedCheckpoints entry must be {step, snapshot, network, params, inputs, seed}: a state digest does not cover the inputs (rates, weights, targets), so a bare digest would vouch for the honest state under forged later inputs" };
    }
  }
  return { list };
}

/** Which of a link's run fields (if it carries them) differ from the receipt's; null if none. */
function otherRun(l, R) {
  if (l.kernel && (l.kernel.sha256 !== R.kernel.sha256 || l.kernel.version !== R.kernel.version || l.kernel.name !== R.kernel.name)) return "kernel";
  if (l.seed && (l.seed.value !== R.seed.value || l.seed.source !== R.seed.source)) return "seed";
  if (l.brain && l.brain.network !== undefined && l.brain.network !== R.brain.network) return "network";
  if (l.brain && l.brain.params !== undefined && l.brain.params !== R.brain.params) return "params";
  if (l.inputs && l.inputs.spec !== undefined && l.inputs.spec !== R.inputs.spec) return "inputs";
  return null;
}

/**
 * Anchor the window's start state (start_step, window.snapshot). Returns {by: "genesis" | "trusted" |
 * "chain", ...}, {by: null} (a link check whose start is unanchored), or {fail: detail}.
 */
async function anchorStart(ctx, W, opts, link) {
  const tl = trustedList(opts.trustedCheckpoints);         // validated on every call, so a malformed list fails at step 0 too
  if (tl.error) return { fail: tl.error };
  if (W.start_step === 0) return { by: "genesis" };
  const R = ctx.receipt;
  const isTrusted = (p) => tl.list.some((t) => t.step === p.step && t.snapshot === p.snapshot && t.network === R.brain.network
    && t.params === R.brain.params && t.inputs === R.inputs.spec && t.seed === R.seed.value);
  const start = { step: W.start_step, snapshot: W.snapshot };
  if (isTrusted(start)) return { by: "trusted" };

  let links = opts.chain, fetchNote = "";
  if (links == null && !link) {
    try {
      const raw = await ctx.fetcher("chain", R);
      if (raw != null) links = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(await privateBytes(raw)));
      else fetchNote = " (the fetcher served no chain)";
    } catch (e) { fetchNote = ` (fetching one failed: ${e && e.message})`; }
  }
  if (links == null) {
    if (link) return { by: null };
    return { fail: `the checkpoint at step ${W.start_step} is not anchored: it is not genesis, no opts.trustedCheckpoints entry names it for this run, and no chain of links from genesis or a trusted checkpoint reaches it${fetchNote}. A self-consistent checkpoint proves nothing about the run by itself.` };
  }
  if (!Array.isArray(links) && links && Array.isArray(links.links)) links = links.links;
  if (!Array.isArray(links) || !links.length) return { fail: "the chain is not a non-empty list of links" };
  for (let i = 0; i < links.length; i++) {
    const l = links[i];
    if (!l || !isPoint(l.from) || !isPoint(l.to) || !(l.to.step > l.from.step)) {
      return { fail: `chain link ${i} needs from/to {step, snapshot} with to.step > from.step` };
    }
    const other = otherRun(l, R);
    if (other) return { fail: `chain link ${i} is for another run: its ${other} differs from the receipt's` };
    if (i && (l.from.step !== links[i - 1].to.step || l.from.snapshot !== links[i - 1].to.snapshot)) {
      return { fail: `chain link ${i} starts at step ${l.from.step} (${l.from.snapshot}), not where link ${i - 1} ended (step ${links[i - 1].to.step}, ${links[i - 1].to.snapshot})` };
    }
  }
  const first = links[0].from, last = links[links.length - 1].to;
  if (last.step !== start.step || last.snapshot !== start.snapshot) {
    return { fail: `the chain ends at step ${last.step} with state ${last.snapshot}, not at this window's start (step ${start.step}, window.snapshot ${start.snapshot})` };
  }
  const root = first.step === 0 ? "genesis" : isTrusted(first) ? "trusted" : null;
  if (!root) return { fail: `the chain starts at step ${first.step}, which is neither genesis nor an opts.trustedCheckpoints entry for this run` };
  for (let i = 0; i < links.length; i++) {
    const l = links[i];
    const st = await startState(ctx, l.from.step, l.from.snapshot, receiptAt(R, l.from));
    if (st.check) return { fail: `chain link ${i} (step ${l.from.step} to ${l.to.step}) failed its ${st.check} check: ${st.detail}` };
    st.sim.run(null, l.to.step - l.from.step);
    const d = st.sim.stateDigest();
    if (d !== l.to.snapshot) {
      return { fail: `chain link ${i}: from step ${l.from.step} the brain reaches ${d} at step ${l.to.step}, the link says ${l.to.snapshot}` };
    }
  }
  return { by: "chain", root, links: links.length };
}

async function run(receipt, fetcher, opts, link) {
  const passed = [];
  let stage = "receipt";
  let sa = { anchored: false, reason: "not checked" };
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
  const fail = (check, detail) => ({ ok: false, check, detail, passed: passed.slice(), seed_anchored: sa.anchored });
  const pass = (check) => { if (!passed.includes(check)) passed.push(check); };
  try {
    const bad = shapeError(receipt, { link });
    if (bad) return fail("receipt", `missing or malformed: ${bad}`);
    const gated = opts.gated !== false;
    const W = receipt.window;

    // (a) the kernel: trusted, hashed, and executed from exactly the hashed bytes
    stage = "kernel";
    const k = await loadKernel(receipt.kernel, fetcher, receipt, opts.trustedKernels);
    if (k.fail) return fail("kernel", k.fail);
    const F = k.F;
    pass("kernel");

    // (b, format) the seed string comes from the receipt, by rule; (b, chain) the optional seed anchor
    stage = "seed";
    let seed;
    try { seed = seedString(receipt.seed); } catch (e) { return fail("seed", e.message); }
    const anchorOfSeed = await seedAnchor(receipt, opts.chainLookup);
    if (anchorOfSeed.contradiction) { sa = { anchored: false, reason: anchorOfSeed.reason }; return fail("seed", anchorOfSeed.reason); }
    sa = anchorOfSeed;

    // (d) wiring and parameters, from the served case (its seed and any stored answers are ignored)
    stage = "network";
    const caseBytes = await privateBytes(await fetcher("case", receipt));
    const c = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(caseBytes));
    if (!c || c.format !== CASE_FORMAT) return fail("network", `the served case is not ${CASE_FORMAT}`);
    const N = c.network;
    const net = new F.Network({ n: N.n, ids: N.ids, indptr: N.indptr, indices: N.indices, counts: N.counts, deadIndices: N.dead_indices });
    const netDigest = net.digest();
    if (netDigest !== receipt.brain.network) return fail("network", `the served wiring hashes to ${netDigest}, the receipt says ${receipt.brain.network}`);
    pass("network");

    // (d') which part of every pair's count is measured and which bridged (not for a chain link)
    let bridge = null;
    if (!link) {
      stage = "bridge";
      const bc = await checkBridge(receipt, fetcher, net, netDigest, opts);
      if (bc.fail) return fail("bridge", bc.fail);
      bridge = bc.bridge;
      pass("bridge");
    }
    stage = "params";
    const params = new F.LIFParams(c.params || {});
    const parDigest = params.digest();
    if (parDigest !== receipt.brain.params) return fail("params", `the served parameters hash to ${parDigest}, the receipt says ${receipt.brain.params}`);
    if (!link && params.dt_ms !== W.dt_ms) return fail("params", `dt is ${params.dt_ms} ms, the receipt says ${W.dt_ms} ms`);
    pass("params");

    // (c) the complete input specification (decoded once; every fresh simulator reuses it)
    stage = "inputs";
    const defs = [];
    for (const s of c.sources || []) {
      if (s.type !== "poisson" && s.type !== "current") return fail("inputs", `unknown source type ${s.type}`);
      defs.push({ s, values: decodeValues(F, s.type === "poisson" ? s.rate_hz : s.amp_mV) });
    }
    const buildSim = () => {
      const sim = new F.Simulator(net, seed, params, { gated });
      for (const { s, values } of defs) {
        const common = { startMs: s.start_ms, stopMs: s.stop_ms, frameMs: s.frame_ms, name: s.name };
        if (s.type === "poisson") sim.addPoisson(s.neurons, values, { ...common, weightMv: s.weight_mV });
        else sim.addCurrent(s.neurons, values, common);
      }
      return sim;
    };
    const sim0 = buildSim();
    const inputs = F.encodeInputs(sim0);
    const spec = F.sha256Hex(inputs);
    if (spec !== receipt.inputs.spec) return fail("inputs", `the served inputs (${sim0.sources.length} sources: ${sim0.sources.map((s) => s.name).join(", ")}) hash to ${spec}, the receipt says ${receipt.inputs.spec}`);
    if (receipt.inputs.frames != null) {
      const retina = sim0.sources.find((s) => s.name === "retina" && s.kind === "poisson" && !s.scalar);
      if (!retina) return fail("inputs", "the receipt names retina frames but the inputs have no retina source");
      const fr = F.sha256Hex(retina.rates);
      if (fr !== receipt.inputs.frames) return fail("inputs", `the retina frames hash to ${fr}, the receipt says ${receipt.inputs.frames}`);
    }
    pass("inputs");
    const tBuilt = now();
    const ctx = { F, net, netDigest, params, parDigest, spec, inputs, seed, seedValue: receipt.seed.value, gated, fetcher, receipt, buildSim };

    // (e) the starting state, and (b, binding) the random streams in it
    stage = "snapshot";
    const st = await startState(ctx, W.start_step, W.snapshot, receipt, pass);
    if (st.check) return fail(st.check, st.detail);
    const sim = st.sim;
    const tStart = now();

    // (g) the starting state is one the run really reached
    stage = "anchor";
    const an = await anchorStart(ctx, W, opts, link);
    if (an.fail) return fail("anchor", an.fail);
    if (an.by) pass("anchor");
    const tAnchor = now();

    // (f) the window itself
    stage = "spikes";
    const win = sim.run(null, W.n_steps);
    const tRun = now();
    if (!link) {
      const sh = F.spikeHash(win);
      if (sh !== receipt.brain.spikes) return fail("spikes", `the replayed window's spikes hash to ${sh}, the receipt says ${receipt.brain.spikes}`);
      pass("spikes");
    }
    stage = "state";
    const end = sim.stateDigest();
    if (end !== receipt.brain.state) return fail("state", `the state after the window hashes to ${end}, the receipt says ${receipt.brain.state}`);
    pass("state");
    return {
      ok: true, passed, raster: win, seed, bridge,
      anchor: an.by ? { by: an.by, ...(an.by === "chain" ? { root: an.root, links: an.links } : {}) } : null,
      seed_anchored: sa.anchored,
      seed_anchor: sa.anchored ? { slot: sa.slot, commit_slot: sa.commit_slot, commit: sa.commit } : { reason: sa.reason },
      kernel: { name: receipt.kernel.name, version: F.KERNEL_VERSION, sha256: k.sha },
      sources: sim.sources.map((s) => ({ name: s.name, kind: s.kind === "current" ? "current" : (s.scalar ? "poisson-scalar" : "poisson-frames") })),
      timing_ms: { build: +tBuilt.toFixed(1), start_state: +(tStart - tBuilt).toFixed(1), anchor: +(tAnchor - tStart).toFixed(1), replay: +(tRun - tAnchor).toFixed(1) },
    };
  } catch (e) {
    return fail(stage, `error: ${e && e.message ? e.message : e}`);
  }
}

/**
 * Verify one decision receipt. opts: {trustedKernels: [sha256, ...] (required), gated = true,
 * trustedCheckpoints: [{step, snapshot, network, params, inputs, seed}, ...], chain: [link, ...],
 * chainLookup: async (query) => answer, trustedBrains: [{network, bridge}, ...]}. A window past step 0
 * FAILS at "anchor" unless a trusted checkpoint or a verified chain anchors its start. Returns {ok: true,
 * passed, raster, bridge, anchor: {by}, seed_anchored, seed_anchor, ...} or {ok: false, check, detail,
 * passed, seed_anchored}. result.bridge.trusted is true only when opts.trustedBrains vouched for the
 * receipt's (network, bridge); without the list it is false with a trust_reason, and ok is unaffected.
 */
export async function verify(receipt, fetcher, opts = {}) {
  return run(receipt, fetcher, opts, false);
}

/**
 * Verify one link of the checkpoint chain: that the state committed at `to` is what the brain reaches
 * by running from the state committed at `from`. link = {kernel, seed, brain: {network, params},
 * inputs: {spec}, from: {step, snapshot}, to: {step, snapshot}}. The fetcher's "checkpoint" is the
 * checkpoint at from.step (none is needed when from.step is 0). Same result shape as verify(); the
 * failing check for a wrong `to` is "state".
 *
 * A link is a conditional check ("from `from`, the brain reaches `to`"), so links can be checked
 * independently and in parallel. ok: true does NOT anchor `from`: result.anchor is {by: "genesis"}
 * when from.step is 0, {by: "trusted" | "chain"} when opts.trustedCheckpoints or opts.chain anchor
 * it, and null otherwise ("anchor" is then absent from passed). A link never fetches a chain.
 */
export async function verifyLink(link, fetcher, opts = {}) {
  if (!link || !link.from || !link.to || !isInt(link.from.step) || !isInt(link.to.step) || link.to.step < link.from.step) {
    return { ok: false, check: "receipt", detail: "link needs from/to {step, snapshot} with to.step >= from.step", passed: [], seed_anchored: false };
  }
  const receipt = {
    kernel: link.kernel, seed: link.seed, inputs: link.inputs,
    brain: { network: link.brain && link.brain.network, params: link.brain && link.brain.params, state: link.to.snapshot, spikes: null },
    window: { start_step: link.from.step, n_steps: link.to.step - link.from.step, dt_ms: null, snapshot: link.from.snapshot },
  };
  return run(receipt, fetcher, opts, true);
}
