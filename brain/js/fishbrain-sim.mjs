// FISHBRAIN spiking kernel in JavaScript: a bit-exact port of fishbrain/sim.py (Simulator).
//
// Runs unchanged in Node and in browsers: no imports, only BigInt, typed arrays and Math.fround.
// Given the same network, params, seed string and inputs, run() returns the same raster as the
// Python engine, and spikeHash() / Simulator.stateDigest() return the same hex strings, byte for byte.
//
// How exactness is kept (see evidence/G1-sim.md for the Python side of the contract):
//  * Seeds: sha256(seed | tags) -> numpy SeedSequence (entropy as little-endian uint32 words, pool of 4,
//    mix_entropy, generate_state(4, uint64)) -> pcg64_set_seed (val[0] is the high word) ->
//    pcg_setseq_128_srandom_r. The browser derives every stream from the seed string itself.
//  * PCG64 (XSL-RR 128/64, step then output) on 16-bit limbs held in doubles: every partial product
//    is < 2^32 and every column sum < 2^36, so the 128-bit LCG is exact without BigInt.
//  * float32 state: IEEE float32 add and multiply computed in float64 and rounded once (Math.fround or
//    a Float32Array store) are correctly rounded (53 >= 2*24 + 2). Each numpy ufunc call is one
//    rounded operation; the order of operations is Simulator.run's.
//  * Synaptic delivery adds rows in the order of the fired list (ascending), targets in CSR order:
//    the order np.add.at uses.
//  * Constants a, b, c, k_i are derived from the parameters with a BigInt decimal (50 digits,
//    ROUND_HALF_EVEN, correctly rounded exp), matching Python's decimal, then rounded to float64/32.
//  * Hashes: a pure-JS SHA-256 over exactly the bytes Python hashes (same prefixes and packing, the
//    same json.dumps text for the PCG64 state).
//
// Activity gating (opts.gated) applies the identical arithmetic only to neurons whose u or h is not
// +0.0 (as fishbrain/sim_fast.py does); it is bit-identical because +0 is a fixed point of the
// update and cannot cross a non-negative threshold.

export const MODEL_VERSION = "fishbrain-lif-v1";
// v2: checkpoints, canonical inputs, source fast-forward. v3: fastForwardSources counts a source that
// starts before step 0 from step 0, as the run draws it (v2 counted from its negative start).
export const KERNEL_VERSION = "fishbrain-js-kernel-v3";

if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
  throw new Error("fishbrain-sim: big-endian platforms are not supported");
}

const TWO32 = 4294967296;
const TWO53 = 9007199254740992;
const ENC = new TextEncoder();
const utf8 = (s) => ENC.encode(s);

// =============================================================================================
// SHA-256 (FIPS 180-4), streaming, synchronous

const K256 = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

export class Sha256 {
  constructor() {
    this.H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
    this.W = new Uint32Array(64);
    this.buf = new Uint8Array(64);
    this.bufLen = 0;
    this.total = 0;
    this.done = false;
  }

  update(data) {
    if (this.done) throw new Error("sha256: update after digest");
    if (typeof data === "string") data = utf8(data);
    else if (!(data instanceof Uint8Array)) data = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const n = data.length;
    let off = 0;
    if (this.bufLen) {
      const take = Math.min(64 - this.bufLen, n);
      this.buf.set(data.subarray(0, take), this.bufLen);
      this.bufLen += take;
      off = take;
      if (this.bufLen === 64) { this._block(this.buf, 0); this.bufLen = 0; }
    }
    while (off + 64 <= n) { this._block(data, off); off += 64; }
    if (off < n) { this.buf.set(data.subarray(off), 0); this.bufLen = n - off; }
    this.total += n;
    return this;
  }

  _block(d, o) {
    const W = this.W, H = this.H;
    for (let i = 0; i < 16; i++) {
      const j = o + 4 * i;
      W[i] = (d[j] << 24) | (d[j + 1] << 16) | (d[j + 2] << 8) | d[j + 3];
    }
    for (let i = 16; i < 64; i++) {
      const w15 = W[i - 15], w2 = W[i - 2];
      const s0 = ((w15 >>> 7) | (w15 << 25)) ^ ((w15 >>> 18) | (w15 << 14)) ^ (w15 >>> 3);
      const s1 = ((w2 >>> 17) | (w2 << 15)) ^ ((w2 >>> 19) | (w2 << 13)) ^ (w2 >>> 10);
      W[i] = (W[i - 16] + s0 + W[i - 7] + s1) | 0;
    }
    let a = H[0], b = H[1], c = H[2], dd = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + S1 + ch + K256[i] + W[i]) | 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const mj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + mj) | 0;
      h = g; g = f; f = e; e = (dd + t1) | 0; dd = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    H[0] = (H[0] + a) | 0; H[1] = (H[1] + b) | 0; H[2] = (H[2] + c) | 0; H[3] = (H[3] + dd) | 0;
    H[4] = (H[4] + e) | 0; H[5] = (H[5] + f) | 0; H[6] = (H[6] + g) | 0; H[7] = (H[7] + h) | 0;
  }

  digest() {
    if (this.done) throw new Error("sha256: digest called twice");
    const bits = this.total * 8;                      // exact below 2^53
    const hi = Math.floor(bits / TWO32), lo = bits >>> 0;
    const padLen = (this.bufLen < 56 ? 56 : 120) - this.bufLen;
    const pad = new Uint8Array(padLen + 8);
    pad[0] = 0x80;
    const dv = new DataView(pad.buffer);
    dv.setUint32(padLen, hi, false);
    dv.setUint32(padLen + 4, lo, false);
    const total = this.total;
    this.update(pad);
    this.total = total;
    this.done = true;
    const out = new Uint8Array(32);
    const ov = new DataView(out.buffer);
    for (let i = 0; i < 8; i++) ov.setUint32(4 * i, this.H[i], false);
    return out;
  }

  hex() { return toHex(this.digest()); }
}

export function toHex(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += (bytes[i] < 16 ? "0" : "") + bytes[i].toString(16);
  return s;
}

export function sha256Hex(data) { return new Sha256().update(data).hex(); }

// little-endian packers into a reusable scratch buffer
class Packer {
  constructor(hash) { this.hash = hash; this.buf = new ArrayBuffer(1 << 16); this.dv = new DataView(this.buf); this.u8 = new Uint8Array(this.buf); this.n = 0; }
  _room(k) { if (this.n + k > this.u8.length) this.flush(); }
  flush() { if (this.n) { this.hash.update(this.u8.subarray(0, this.n)); this.n = 0; } }
  i64(x) {                                          // x: an integer-valued double (exact, |x| < 2^63) or a BigInt
    this._room(8);
    if (typeof x === "bigint") { this.dv.setBigInt64(this.n, x, true); }
    else {
      // every integer-valued double below 2^63 is packed exactly (e.g. last_spike's -2^62 sentinel)
      if (!Number.isInteger(x) || Math.abs(x) >= 9223372036854775808) throw new Error(`int64 pack: ${x} is not an int64`);
      const hi = Math.floor(x / TWO32);
      this.dv.setUint32(this.n, x - hi * TWO32, true);
      this.dv.setInt32(this.n + 4, hi, true);
    }
    this.n += 8;
  }
  u64(x) { if (x < 0) throw new Error("u64 pack of a negative"); this.i64(x); }
  bytes(b) { this.flush(); this.hash.update(b); }
}

// =============================================================================================
// seeds: derive_seed + numpy SeedSequence + PCG64 seeding

/** int.from_bytes(sha256("|".join([seed, *tags]).encode()), "big") as a BigInt. */
export function deriveSeed(seed, ...tags) {
  const parts = [String(seed), ...tags.map(String)];
  return BigInt("0x" + sha256Hex(parts.join("|")));
}

const SS_INIT_A = 0x43b0d7e5, SS_MULT_A = 0x931e8875, SS_INIT_B = 0x8b51f9dd, SS_MULT_B = 0x58f38ded;
const SS_MIX_L = 0xca01f9dd, SS_MIX_R = 0x4973f715;

/** numpy _int_to_uint32_array: little-endian 32-bit words, [0] for 0. */
export function intToUint32Words(n) {
  n = BigInt(n);
  if (n < 0n) throw new Error("expected non-negative integer");
  if (n === 0n) return [0];
  const out = [];
  while (n > 0n) { out.push(Number(n & 0xffffffffn)); n >>= 32n; }
  return out;
}

/** numpy SeedSequence(entropy).pool (pool_size 4, no spawn key). */
export function seedSequencePool(entropy, poolSize = 4) {
  const ent = intToUint32Words(entropy);
  const pool = new Array(poolSize).fill(0);
  let hc = SS_INIT_A;
  const hashmix = (v) => {
    v = (v ^ hc) >>> 0;
    hc = Math.imul(hc, SS_MULT_A) >>> 0;
    v = Math.imul(v, hc) >>> 0;
    return (v ^ (v >>> 16)) >>> 0;
  };
  const mix = (x, y) => {
    let r = ((Math.imul(SS_MIX_L, x) >>> 0) - (Math.imul(SS_MIX_R, y) >>> 0)) >>> 0;
    return (r ^ (r >>> 16)) >>> 0;
  };
  for (let i = 0; i < poolSize; i++) pool[i] = hashmix(i < ent.length ? ent[i] : 0);
  for (let s = 0; s < poolSize; s++) {
    for (let d = 0; d < poolSize; d++) if (s !== d) pool[d] = mix(pool[d], hashmix(pool[s]));
  }
  for (let s = poolSize; s < ent.length; s++) {
    for (let d = 0; d < poolSize; d++) pool[d] = mix(pool[d], hashmix(ent[s]));
  }
  return pool;
}

/** numpy SeedSequence(entropy).generate_state(nWords64, np.uint64) as BigInts. */
export function seedSequenceState64(entropy, nWords64 = 4) {
  const pool = seedSequencePool(entropy);
  let hc = SS_INIT_B;
  const w = [];
  for (let i = 0; i < 2 * nWords64; i++) {
    let v = pool[i % pool.length];
    v = (v ^ hc) >>> 0;
    hc = Math.imul(hc, SS_MULT_B) >>> 0;
    v = Math.imul(v, hc) >>> 0;
    w.push((v ^ (v >>> 16)) >>> 0);
  }
  const out = [];
  for (let i = 0; i < nWords64; i++) out.push(BigInt(w[2 * i]) | (BigInt(w[2 * i + 1]) << 32n));
  return out;
}

const M128 = (1n << 128n) - 1n;
export const PCG_MULT_128 = 0x2360ED051FC65DA44385DF649FCCF645n;
const MUL16 = [0xF645, 0x9FCC, 0xDF64, 0x4385, 0x5DA4, 0x1FC6, 0xED05, 0x2360];   // little-endian limbs

/** numpy.random.PCG64: only the raw 64-bit output (random_raw) is provided, as the engine uses. */
export class PCG64 {
  /** seed: a non-negative integer (BigInt or safe Number), seeded through SeedSequence like numpy. */
  constructor(seed) {
    const [s0, s1, i0, i1] = seedSequenceState64(BigInt(seed), 4);
    const initstate = (s0 << 64n) | s1;             // pcg64_set_seed: seed[0] is the high word
    const initseq = (i0 << 64n) | i1;
    const inc = ((initseq << 1n) | 1n) & M128;
    let st = 0n;
    st = (st * PCG_MULT_128 + inc) & M128;
    st = (st + initstate) & M128;
    st = (st * PCG_MULT_128 + inc) & M128;
    this._setState(st, inc);
  }

  static fromState(state, inc) {
    const g = Object.create(PCG64.prototype);
    g._setState(BigInt(state) & M128, BigInt(inc) & M128);
    return g;
  }

  _setState(st, inc) {
    this.s = new Float64Array(8);
    this.i = new Float64Array(8);
    for (let k = 0; k < 8; k++) {
      this.s[k] = Number((st >> BigInt(16 * k)) & 0xffffn);
      this.i[k] = Number((inc >> BigInt(16 * k)) & 0xffffn);
    }
    // jump-4 constants for fillRaw's four interleaved lanes: x -> A4 x + C4 is four steps
    const m2 = (PCG_MULT_128 * PCG_MULT_128) & M128;
    const m3 = (m2 * PCG_MULT_128) & M128;
    const A4 = (m2 * m2) & M128;
    const C4 = (inc * (1n + PCG_MULT_128 + m2 + m3)) & M128;
    this.j = new Float64Array(16);                   // [A4 limbs, C4 limbs]
    for (let k = 0; k < 8; k++) {
      this.j[k] = Number((A4 >> BigInt(16 * k)) & 0xffffn);
      this.j[8 + k] = Number((C4 >> BigInt(16 * k)) & 0xffffn);
    }
    this.lanes = new Float64Array(32);
    this.hi = 0; this.lo = 0;                        // the last output, as two uint32 halves
    this.draws = 0;
  }

  /**
   * Jump table for draws 1..K from a base state S: state after draw j = A_j S + C_j (mod 2^128),
   * A_j = M^j, C_j = inc (M^(j-1) + ... + 1). Entry j occupies [16 j, 16 j + 16): A limbs, C limbs.
   */
  jumpTable(K) {
    if (this._jt && this._jtK === K) return this._jt;
    const T = new Float64Array(16 * (K + 1));
    const inc = this.incBigInt;
    let A = 1n, C = 0n;
    for (let j = 1; j <= K; j++) {
      A = (A * PCG_MULT_128) & M128;
      C = (C * PCG_MULT_128 + inc) & M128;
      for (let k = 0; k < 8; k++) {
        T[16 * j + k] = Number((A >> BigInt(16 * k)) & 0xffffn);
        T[16 * j + 8 + k] = Number((C >> BigInt(16 * k)) & 0xffffn);
      }
    }
    this._jt = T; this._jtK = K;
    return T;
  }

  /**
   * K draws from the current state where only the draws at positions sel[0..m) (ascending, < K) are
   * needed: their outputs go to H[q], L[q] for q < m, and the state ends K draws later, exactly as
   * K calls of next(). Each needed state is computed straight from the base state through the jump
   * table, so the multiplies are independent of each other.
   */
  drawSelected(K, sel, m, H, L) {
    const T = this.jumpTable(K), s = this.s;
    const s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3], s4 = s[4], s5 = s[5], s6 = s[6], s7 = s[7];
    for (let q = 0; q <= m; q++) {
      const o = 16 * (q < m ? sel[q] + 1 : K);
      const a0 = T[o], a1 = T[o + 1], a2 = T[o + 2], a3 = T[o + 3], a4 = T[o + 4], a5 = T[o + 5], a6 = T[o + 6], a7 = T[o + 7];
      let c = s0 * a0 + T[o + 8];
      const r0 = c & 0xffff; c = (c - r0) * 1.52587890625e-05;
      c += s0 * a1 + s1 * a0 + T[o + 9];
      const r1 = c & 0xffff; c = (c - r1) * 1.52587890625e-05;
      c += s0 * a2 + s1 * a1 + s2 * a0 + T[o + 10];
      const r2 = c & 0xffff; c = (c - r2) * 1.52587890625e-05;
      c += s0 * a3 + s1 * a2 + s2 * a1 + s3 * a0 + T[o + 11];
      const r3 = c & 0xffff; c = (c - r3) * 1.52587890625e-05;
      c += s0 * a4 + s1 * a3 + s2 * a2 + s3 * a1 + s4 * a0 + T[o + 12];
      const r4 = c & 0xffff; c = (c - r4) * 1.52587890625e-05;
      c += s0 * a5 + s1 * a4 + s2 * a3 + s3 * a2 + s4 * a1 + s5 * a0 + T[o + 13];
      const r5 = c & 0xffff; c = (c - r5) * 1.52587890625e-05;
      c += s0 * a6 + s1 * a5 + s2 * a4 + s3 * a3 + s4 * a2 + s5 * a1 + s6 * a0 + T[o + 14];
      const r6 = c & 0xffff; c = (c - r6) * 1.52587890625e-05;
      c += s0 * a7 + s1 * a6 + s2 * a5 + s3 * a4 + s4 * a3 + s5 * a2 + s6 * a1 + s7 * a0 + T[o + 15];
      const r7 = c & 0xffff;
      if (q === m) {                                  // the state after all K draws
        s[0] = r0; s[1] = r1; s[2] = r2; s[3] = r3; s[4] = r4; s[5] = r5; s[6] = r6; s[7] = r7;
        break;
      }
      let xh = (((r7 << 16) | r6) ^ ((r3 << 16) | r2)) >>> 0;
      let xl = (((r5 << 16) | r4) ^ ((r1 << 16) | r0)) >>> 0;
      const rot = r7 >>> 10;
      if (rot & 32) { const t = xh; xh = xl; xl = t; }
      const r = rot & 31;
      if (r) {
        const nh = ((xh >>> r) | (xl << (32 - r))) >>> 0;
        xl = ((xl >>> r) | (xh << (32 - r))) >>> 0;
        xh = nh;
      }
      H[q] = xh; L[q] = xl;
    }
    this.draws += K;
  }

  /**
   * The next K raw outputs into H[0..K) (high 32 bits) and L[0..K) (low 32 bits), exactly as K calls
   * of next() would, leaving the same state. Four lanes hold the states of draws i..i+3 and each
   * jumps four steps at a time, so the four 128-bit multiplies are independent (the single-step
   * LCG is one long carry chain; this is ~3x faster, same arithmetic).
   */
  fillRaw(K, H, L) {
    if (K < 8) { for (let q = 0; q < K; q++) { this.next(); H[q] = this.hi; L[q] = this.lo; } return; }
    const ln = this.lanes, s = this.s, J = this.j;
    // lane l = state after draw l+1 (single steps)
    for (let l = 0; l < 4; l++) { this.next(); for (let k = 0; k < 8; k++) ln[8 * l + k] = s[k]; }
    const a0 = J[0], a1 = J[1], a2 = J[2], a3 = J[3], a4 = J[4], a5 = J[5], a6 = J[6], a7 = J[7];
    const c0 = J[8], c1 = J[9], c2 = J[10], c3 = J[11], c4 = J[12], c5 = J[13], c6 = J[14], c7 = J[15];
    let i = 0;
    for (;;) {
      for (let l = 0; l < 4 && i + l < K; l++) {
        const o = 8 * l;
        const r0 = ln[o], r1 = ln[o + 1], r2 = ln[o + 2], r3 = ln[o + 3], r4 = ln[o + 4], r5 = ln[o + 5], r6 = ln[o + 6], r7 = ln[o + 7];
        let xh = (((r7 << 16) | r6) ^ ((r3 << 16) | r2)) >>> 0;
        let xl = (((r5 << 16) | r4) ^ ((r1 << 16) | r0)) >>> 0;
        const rot = r7 >>> 10;
        if (rot & 32) { const t = xh; xh = xl; xl = t; }
        const r = rot & 31;
        if (r) {
          const nh = ((xh >>> r) | (xl << (32 - r))) >>> 0;
          xl = ((xl >>> r) | (xh << (32 - r))) >>> 0;
          xh = nh;
        }
        H[i + l] = xh; L[i + l] = xl;
      }
      if (i + 4 >= K) {                               // draw K-1 came from lane (K-1) & 3
        const o = 8 * ((K - 1) & 3);
        for (let k = 0; k < 8; k++) s[k] = ln[o + k];
        break;
      }
      i += 4;
      for (let o = 0; o < 32; o += 8) {               // each lane jumps four steps: x = A4 x + C4
        const s0 = ln[o], s1 = ln[o + 1], s2 = ln[o + 2], s3 = ln[o + 3], s4 = ln[o + 4], s5 = ln[o + 5], s6 = ln[o + 6], s7 = ln[o + 7];
        let c = s0 * a0 + c0;
        const r0 = c & 0xffff; c = (c - r0) * 1.52587890625e-05;
        c += s0 * a1 + s1 * a0 + c1;
        const r1 = c & 0xffff; c = (c - r1) * 1.52587890625e-05;
        c += s0 * a2 + s1 * a1 + s2 * a0 + c2;
        const r2 = c & 0xffff; c = (c - r2) * 1.52587890625e-05;
        c += s0 * a3 + s1 * a2 + s2 * a1 + s3 * a0 + c3;
        const r3 = c & 0xffff; c = (c - r3) * 1.52587890625e-05;
        c += s0 * a4 + s1 * a3 + s2 * a2 + s3 * a1 + s4 * a0 + c4;
        const r4 = c & 0xffff; c = (c - r4) * 1.52587890625e-05;
        c += s0 * a5 + s1 * a4 + s2 * a3 + s3 * a2 + s4 * a1 + s5 * a0 + c5;
        const r5 = c & 0xffff; c = (c - r5) * 1.52587890625e-05;
        c += s0 * a6 + s1 * a5 + s2 * a4 + s3 * a3 + s4 * a2 + s5 * a1 + s6 * a0 + c6;
        const r6 = c & 0xffff; c = (c - r6) * 1.52587890625e-05;
        c += s0 * a7 + s1 * a6 + s2 * a5 + s3 * a4 + s4 * a3 + s5 * a2 + s6 * a1 + s7 * a0 + c7;
        ln[o] = r0; ln[o + 1] = r1; ln[o + 2] = r2; ln[o + 3] = r3; ln[o + 4] = r4; ln[o + 5] = r5; ln[o + 6] = r6; ln[o + 7] = c & 0xffff;
      }
    }
    this.hi = H[K - 1]; this.lo = L[K - 1];
    this.draws += K - 4;                              // next() counted the first four
  }

  get stateBigInt() { let x = 0n; for (let k = 7; k >= 0; k--) x = (x << 16n) | BigInt(this.s[k]); return x; }
  get incBigInt() { let x = 0n; for (let k = 7; k >= 0; k--) x = (x << 16n) | BigInt(this.i[k]); return x; }

  /** json.dumps(bit_generator.state, sort_keys=True, default=str), as Python writes it. */
  stateJSON() {
    return `{"bit_generator": "PCG64", "has_uint32": 0, "state": {"inc": ${this.incBigInt.toString()}, "state": ${this.stateBigInt.toString()}}, "uinteger": 0}`;
  }

  /** Advance n draws; each output lands in this.hi / this.lo. Returns nothing (hot path). */
  next() {
    const s = this.s, inc = this.i;
    const s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3], s4 = s[4], s5 = s[5], s6 = s[6], s7 = s[7];
    let c = s0 * 0xF645 + inc[0];
    const r0 = c & 0xffff; c = (c - r0) * 1.52587890625e-05;
    c += s0 * 0x9FCC + s1 * 0xF645 + inc[1];
    const r1 = c & 0xffff; c = (c - r1) * 1.52587890625e-05;
    c += s0 * 0xDF64 + s1 * 0x9FCC + s2 * 0xF645 + inc[2];
    const r2 = c & 0xffff; c = (c - r2) * 1.52587890625e-05;
    c += s0 * 0x4385 + s1 * 0xDF64 + s2 * 0x9FCC + s3 * 0xF645 + inc[3];
    const r3 = c & 0xffff; c = (c - r3) * 1.52587890625e-05;
    c += s0 * 0x5DA4 + s1 * 0x4385 + s2 * 0xDF64 + s3 * 0x9FCC + s4 * 0xF645 + inc[4];
    const r4 = c & 0xffff; c = (c - r4) * 1.52587890625e-05;
    c += s0 * 0x1FC6 + s1 * 0x5DA4 + s2 * 0x4385 + s3 * 0xDF64 + s4 * 0x9FCC + s5 * 0xF645 + inc[5];
    const r5 = c & 0xffff; c = (c - r5) * 1.52587890625e-05;
    c += s0 * 0xED05 + s1 * 0x1FC6 + s2 * 0x5DA4 + s3 * 0x4385 + s4 * 0xDF64 + s5 * 0x9FCC + s6 * 0xF645 + inc[6];
    const r6 = c & 0xffff; c = (c - r6) * 1.52587890625e-05;
    c += s0 * 0x2360 + s1 * 0xED05 + s2 * 0x1FC6 + s3 * 0x5DA4 + s4 * 0x4385 + s5 * 0xDF64 + s6 * 0x9FCC + s7 * 0xF645 + inc[7];
    const r7 = c & 0xffff;
    s[0] = r0; s[1] = r1; s[2] = r2; s[3] = r3; s[4] = r4; s[5] = r5; s[6] = r6; s[7] = r7;
    // XSL-RR: rotr64(hi64 ^ lo64, state >> 122)
    let xh = (((r7 << 16) | r6) ^ ((r3 << 16) | r2)) >>> 0;
    let xl = (((r5 << 16) | r4) ^ ((r1 << 16) | r0)) >>> 0;
    const rot = r7 >>> 10;
    if (rot & 32) { const t = xh; xh = xl; xl = t; }
    const r = rot & 31;
    if (r) {
      const nh = ((xh >>> r) | (xl << (32 - r))) >>> 0;
      xl = ((xl >>> r) | (xh << (32 - r))) >>> 0;
      xh = nh;
    }
    this.hi = xh; this.lo = xl;
    this.draws++;
  }

  /** random_raw(n) as a BigUint64Array (for tests; the kernel uses next()). */
  randomRaw(n) {
    const out = new BigUint64Array(n);
    for (let k = 0; k < n; k++) { this.next(); out[k] = (BigInt(this.hi) << 32n) | BigInt(this.lo); }
    return out;
  }
}

/** make_bitgen(seed, *tags): PCG64 seeded from derive_seed. */
export function makeBitgen(seed, ...tags) { return new PCG64(deriveSeed(seed, ...tags)); }

// =============================================================================================
// decimal arithmetic (Python decimal semantics for the few operations LIFParams.derived uses)
// A decimal is {c: BigInt (signed coefficient), e: Number}: value = c * 10^e.

const P10 = [1n];
const pow10 = (k) => { while (P10.length <= k) P10.push(P10[P10.length - 1] * 10n); return P10[k]; };
const ndigits = (n) => (n === 0n ? 1 : n.toString().length);   // n >= 0
const bitlen = (n) => (n === 0n ? 0 : n.toString(2).length);

/** Round value = num/den * 10^e10 (den > 0) to P significant digits, ROUND_HALF_EVEN. */
function roundRat(num, den, e10, P) {
  if (num === 0n) return { c: 0n, e: 0 };
  const neg = num < 0n;
  const n = neg ? -num : num;
  let k = P - (ndigits(n) - ndigits(den)) - 1;
  const lowQ = pow10(P - 1), highQ = pow10(P);
  let q, r, D;
  for (;;) {
    let N = n; D = den;
    if (k >= 0) N = n * pow10(k); else D = den * pow10(-k);
    q = N / D; r = N - q * D;
    if (q >= highQ) { k -= 1; continue; }
    if (q < lowQ) { k += 1; continue; }
    break;
  }
  const twice = 2n * r;
  if (twice > D || (twice === D && (q & 1n) === 1n)) q += 1n;
  if (q === highQ) { q = lowQ; k -= 1; }
  return { c: neg ? -q : q, e: e10 - k };
}

const decCtx = (P) => ({
  add: (x, y) => { const e = Math.min(x.e, y.e); return roundRat(x.c * pow10(x.e - e) + y.c * pow10(y.e - e), 1n, e, P); },
  sub: (x, y) => { const e = Math.min(x.e, y.e); return roundRat(x.c * pow10(x.e - e) - y.c * pow10(y.e - e), 1n, e, P); },
  mul: (x, y) => roundRat(x.c * y.c, 1n, x.e + y.e, P),
  div: (x, y) => {
    if (y.c === 0n) throw new Error("decimal division by zero");
    const neg = y.c < 0n;
    return roundRat(neg ? -x.c : x.c, neg ? -y.c : y.c, x.e - y.e, P);
  },
  neg: (x) => roundRat(-x.c, 1n, x.e, P),
  exp: (x) => decExp(x, P),
});

/** exp(x) correctly rounded to P digits (half-even); throws if the working precision cannot decide. */
function decExp(x, P) {
  if (x.c === 0n) return { c: 1n, e: 0 };
  // reduce: |x| / 2^red < 1/2, i.e. 2A < B 2^red with |x| = A/B
  let red = 0;
  const A = (x.c < 0n ? -x.c : x.c) * pow10(Math.max(0, x.e));
  const B = pow10(Math.max(0, -x.e));
  while (2n * A >= B * (1n << BigInt(red))) {
    red += 1;
    if (red > 64) throw new Error("decimal exp: argument too large");
  }
  const W = P + 40 + red + Math.max(0, -x.e);        // working digits after the point
  const S = pow10(W);
  // X = x * 10^W (exact: W >= -e)
  let X = x.e + W >= 0 ? x.c * pow10(x.e + W) : x.c / pow10(-(x.e + W));
  const two = 2n ** BigInt(red);
  X = X / two;                                       // truncation error <= 1 unit
  let term = S, sum = S, k = 1n, nterms = 0;
  for (;;) {
    term = (term * X) / (k * S);
    if (term === 0n) break;
    sum += term;
    k += 1n; nterms++;
    if (nterms > 10000) throw new Error("decimal exp: series did not converge");
  }
  for (let i = 0; i < red; i++) sum = (sum * sum) / S;
  // error bound in units of 10^-W: series and truncation, doubled (relative) per squaring
  const err = BigInt(nterms + 4) * (2n ** BigInt(2 * red + 2)) * 4n;
  const lo = roundRat(sum - err, 1n, -W, P), hi = roundRat(sum + err, 1n, -W, P);
  if (lo.c * pow10(lo.e - Math.min(lo.e, hi.e)) !== hi.c * pow10(hi.e - Math.min(lo.e, hi.e))) {
    throw new Error("decimal exp: rounding undecidable at the working precision");
  }
  return lo;
}

/** The exact decimal value of the shortest round-trip repr of a float64 (Python Decimal(repr(x))). */
export function decFromNumber(x) {
  if (!Number.isFinite(x)) throw new Error(`not a finite number: ${x}`);
  if (x === 0) return { c: 0n, e: 0 };
  const { neg, digits, decpt } = shortestDigits(x);
  return { c: (neg ? -1n : 1n) * BigInt(digits), e: decpt - digits.length };
}

/** Shortest round-trip digits of x: value = 0.<digits> * 10^decpt. */
function shortestDigits(x) {
  const neg = x < 0 || Object.is(x, -0);
  const s = Math.abs(x).toExponential();               // shortest digits, e.g. "2.75e-1"
  const m = /^(\d)(?:\.(\d+))?e([+-]\d+)$/.exec(s);
  if (!m) throw new Error(`unexpected toExponential output ${s}`);
  let digits = m[1] + (m[2] || "");
  digits = digits.replace(/0+$/, "") || "0";
  return { neg, digits, decpt: parseInt(m[3], 10) + 1 };
}

/** Python repr(float) (float_repr_style 'short'). */
export function pyFloatRepr(x) {
  if (Number.isNaN(x)) return "nan";
  if (x === Infinity) return "inf";
  if (x === -Infinity) return "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const { neg, digits, decpt } = shortestDigits(x);
  let out;
  if (decpt <= -4 || decpt > 16) {
    const e = decpt - 1;
    out = digits[0] + (digits.length > 1 ? "." + digits.slice(1) : "") + "e" + (e < 0 ? "-" : "+") + String(Math.abs(e)).padStart(2, "0");
  } else if (decpt <= 0) {
    out = "0." + "0".repeat(-decpt) + digits;
  } else if (decpt >= digits.length) {
    out = digits + "0".repeat(decpt - digits.length) + ".0";
  } else {
    out = digits.slice(0, decpt) + "." + digits.slice(decpt);
  }
  return (neg ? "-" : "") + out;
}

/** float(Decimal): exact decimal -> nearest float64 (half-even), normal range only. */
export function decToNumber(d) {
  if (d.c === 0n) return 0;
  const neg = d.c < 0n;
  const n = neg ? -d.c : d.c;
  let N = n, D = 1n;
  if (d.e >= 0) N = n * pow10(d.e); else D = pow10(-d.e);
  let E2 = bitlen(N) - bitlen(D) - 53;
  let q, r, Dp;
  const lo53 = 1n << 52n, hi53 = 1n << 53n;
  for (;;) {
    let Np = N; Dp = D;
    if (E2 >= 0) Dp = D << BigInt(E2); else Np = N << BigInt(-E2);
    q = Np / Dp; r = Np - q * Dp;
    if (q >= hi53) { E2 += 1; continue; }
    if (q < lo53) { E2 -= 1; continue; }
    break;
  }
  const twice = 2n * r;
  if (twice > Dp || (twice === Dp && (q & 1n) === 1n)) q += 1n;
  if (q === hi53) { q = lo53; E2 += 1; }
  if (E2 + 52 > 1023 || E2 + 52 < -1022) throw new Error("decToNumber: outside the normal float64 range");
  const v = Number(q) * 2 ** E2;
  return neg ? -v : v;
}

/** Python round() of a float: half to even. */
function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

const DEFAULT_CTX = decCtx(28);                        // Python's default decimal context

/** sim._steps: whole dt steps in ms, via float(Decimal(repr(ms)) / Decimal(repr(dt))). */
export function stepsOf(ms, dt, what = "duration") {
  const n = decToNumber(DEFAULT_CTX.div(decFromNumber(Number(ms)), decFromNumber(Number(dt))));
  const k = pyRound(n);
  if (Math.abs(n - k) > 1e-9) throw new Error(`${what} = ${ms} ms is not a whole number of dt = ${dt} ms steps`);
  return k;
}

// =============================================================================================
// parameters

const PARAM_DEFAULTS = {
  v_rest_mV: -52.0, v_reset_mV: -52.0, v_thresh_mV: -45.0, tau_m_ms: 20.0, tau_syn_ms: 5.0,
  refractory_ms: 2.2, delay_ms: 1.8, w_syn_mV: 0.275, poisson_weight_mV: 68.75, dt_ms: 0.1,
};
export const PARAM_FIELDS = Object.keys(PARAM_DEFAULTS);   // dataclass field order

function f32Hex(x) {
  const b = new Uint8Array(4);
  new DataView(b.buffer).setFloat32(0, x, true);
  return toHex(b);
}
function f64Hex(x) {
  const b = new Uint8Array(8);
  new DataView(b.buffer).setFloat64(0, x, true);
  return toHex(b);
}
export { f32Hex, f64Hex };

/** json.dumps(obj, sort_keys=True) for dicts of str/int/str-valued dicts (what the digests use). */
function pyJson(v) {
  if (typeof v === "string") {
    if (!/^[\x20-\x7e]*$/.test(v) || /["\\]/.test(v)) throw new Error("pyJson: only plain ASCII strings");
    return `"${v}"`;
  }
  if (typeof v === "number") {
    if (!Number.isSafeInteger(v)) throw new Error("pyJson: only integers");
    return String(v);
  }
  if (typeof v === "bigint") return v.toString();
  const keys = Object.keys(v).sort();
  return "{" + keys.map((k) => `${pyJson(k)}: ${pyJson(v[k])}`).join(", ") + "}";
}

export class LIFParams {
  constructor(opts = {}) {
    for (const k of Object.keys(opts)) if (!(k in PARAM_DEFAULTS)) throw new Error(`unknown LIF parameter ${k}`);
    for (const k of PARAM_FIELDS) {
      const v = k in opts ? Number(opts[k]) : PARAM_DEFAULTS[k];
      if (!Number.isFinite(v)) throw new Error(`${k} must be a finite number`);
      this[k] = v;
    }
    if (!(this.v_reset_mV < this.v_thresh_mV)) throw new Error("v_reset must be below v_thresh");
    if (this.dt_ms <= 0 || this.tau_m_ms <= 0 || this.tau_syn_ms <= 0) throw new Error("dt and time constants must be positive");
    stepsOf(this.refractory_ms, this.dt_ms, "refractory_ms");
    stepsOf(this.delay_ms, this.dt_ms, "delay_ms");
    Object.freeze(this);
  }

  /** LIFParams.derived(): the same decimal computation at 50 digits, then float64 / float32. */
  derived() {
    const X = decCtx(50);
    const dt = decFromNumber(this.dt_ms), tm = decFromNumber(this.tau_m_ms), ts = decFromNumber(this.tau_syn_ms);
    const a = X.exp(X.neg(X.div(dt, tm)));
    const b = X.exp(X.neg(X.div(dt, ts)));
    const same = tm.c * pow10(tm.e - Math.min(tm.e, ts.e)) === ts.c * pow10(ts.e - Math.min(tm.e, ts.e));
    const c = !same ? X.mul(X.div(ts, X.sub(ts, tm)), X.sub(b, a)) : X.mul(X.div(dt, tm), a);
    const k_i = X.sub({ c: 1n, e: 0 }, a);
    const a64 = decToNumber(a), b64 = decToNumber(b), c64 = decToNumber(c), ki64 = decToNumber(k_i);
    return {
      a: Math.fround(a64), b: Math.fround(b64), c: Math.fround(c64), k_i: Math.fround(ki64),
      c64, k_i64: ki64,
      theta: Math.fround(this.v_thresh_mV - this.v_rest_mV),
      u_reset: Math.fround(this.v_reset_mV - this.v_rest_mV),
      n_ref: stepsOf(this.refractory_ms, this.dt_ms, "refractory_ms"),
      n_delay: stepsOf(this.delay_ms, this.dt_ms, "delay_ms"),
    };
  }

  steps(ms) { return stepsOf(ms, this.dt_ms, "duration"); }

  /** LIFParams.digest() */
  digest() {
    const d = this.derived();
    const bits = {};
    for (const k of ["a", "b", "c", "k_i", "theta", "u_reset"]) bits[k] = f32Hex(d[k]);
    const params = {};
    for (const k of PARAM_FIELDS) params[k] = pyFloatRepr(this[k]);
    const payload = { model: MODEL_VERSION, params, float32_bits: bits, n_ref: d.n_ref, n_delay: d.n_delay };
    return sha256Hex(pyJson(payload));
  }
}

// =============================================================================================
// wiring

const toI32 = (a, what) => {
  const out = a instanceof Int32Array ? a : Int32Array.from(a);
  if (!(a instanceof Int32Array)) for (let i = 0; i < a.length; i++) if (out[i] !== Number(a[i])) throw new Error(`${what}: value out of int32 range`);
  return out;
};

export class Network {
  /**
   * CSR wiring, row = presynaptic, column = postsynaptic (sim.Network).
   * ids: external ids (BigInt-able; default 0..n-1); indptr (n+1), indices (nnz), counts (nnz, signed
   * synapse counts); dead: a length-n 0/1 array or a list of dead internal indices.
   */
  constructor({ n, ids = null, indptr, indices, counts, dead = null, deadIndices = null }) {
    this.n = n;
    this.indptr = toI32(indptr, "indptr");
    this.indices = toI32(indices, "indices");
    this.counts = toI32(counts, "counts");
    if (this.indptr.length !== n + 1 || this.indptr[n] !== this.indices.length || this.counts.length !== this.indices.length || this.indptr[0] !== 0) {
      throw new Error("inconsistent CSR arrays");
    }
    for (let i = 0; i < n; i++) if (this.indptr[i + 1] < this.indptr[i]) throw new Error("indptr must be non-decreasing");
    for (let j = 0; j < this.indices.length; j++) if (this.indices[j] < 0 || this.indices[j] >= n) throw new Error("column index out of range");
    this.ids = new BigUint64Array(n);
    if (ids == null) for (let i = 0; i < n; i++) this.ids[i] = BigInt(i);
    else {
      if (ids.length !== n) throw new Error("ids length != n");
      for (let i = 0; i < n; i++) this.ids[i] = BigInt(ids[i]);
      const seen = new Set();
      for (let i = 0; i < n; i++) { const k = this.ids[i].toString(); if (seen.has(k)) throw new Error("neuron ids must be unique"); seen.add(k); }
    }
    this.dead = new Uint8Array(n);
    if (dead) { if (dead.length !== n) throw new Error("dead length != n"); for (let i = 0; i < n; i++) this.dead[i] = dead[i] ? 1 : 0; }
    if (deadIndices) for (const i of deadIndices) { if (i < 0 || i >= n) throw new Error("dead index out of range"); this.dead[i] = 1; }
  }

  get nnz() { return this.indices.length; }

  /** Network.digest() */
  digest() {
    const h = new Sha256();
    h.update("fishbrain-net-v1");
    const p = new Packer(h);
    p.u64(this.n); p.u64(this.nnz);
    p.flush();
    const idb = new Uint8Array(8 * this.n);
    const dv = new DataView(idb.buffer);
    for (let i = 0; i < this.n; i++) dv.setBigUint64(8 * i, this.ids[i], true);
    h.update(idb);
    for (let i = 0; i <= this.n; i++) p.i64(this.indptr[i]);
    for (let j = 0; j < this.nnz; j++) p.i64(this.indices[j]);
    p.flush();
    h.update(new Uint8Array(this.counts.buffer, this.counts.byteOffset, this.counts.byteLength));
    h.update(this.dead);
    return h.hex();
  }
}

// =============================================================================================
// rasters

/** spike_hash(raster): raster = {steps, neurons, startStep, nSteps, nNeurons}. */
export function spikeHash(r) {
  const n = r.neurons.length;
  let steps = r.steps, neurons = r.neurons;
  let sorted = true;
  for (let i = 1; i < n; i++) {
    if (steps[i] < steps[i - 1] || (steps[i] === steps[i - 1] && neurons[i] < neurons[i - 1])) { sorted = false; break; }
  }
  if (!sorted) {
    const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => (steps[x] - steps[y]) || (neurons[x] - neurons[y]));
    steps = Float64Array.from(order, (i) => steps[i]);
    neurons = Float64Array.from(order, (i) => neurons[i]);
  }
  const h = new Sha256();
  h.update("fishbrain-raster-v1");
  const p = new Packer(h);
  p.u64(r.nNeurons); p.u64(r.startStep); p.u64(r.nSteps); p.u64(n);
  for (let i = 0; i < n; i++) p.i64(steps[i]);
  for (let i = 0; i < n; i++) p.i64(neurons[i]);
  p.flush();
  return h.hex();
}

// =============================================================================================
// inputs

/** Q[k] = (1-p)^(2^k) by repeated squaring, kept where > 0, descending k (sim._geometric_tables). */
export function geometricTables(p) {
  const q = 1.0 - p;
  const Q = [q];
  for (let i = 0; i < 62; i++) Q.push(Q[Q.length - 1] * Q[Q.length - 1]);
  const ks = [], qs = [];
  for (let k = Q.length - 1; k >= 0; k--) if (Q[k] > 0.0) { ks.push(k); qs.push(Q[k]); }
  return { pow2: Float64Array.from(ks, (k) => 2 ** k), q: Float64Array.from(qs) };
}

/** uint64 threshold floor(p * 2^64) as [hi, lo] uint32 halves (sim._bernoulli_threshold). */
export function bernoulliThreshold(p) {
  if (p >= 1.0) return [0xffffffff, 0xffffffff];
  if (p > 0.0) {
    const t = Math.floor(p * 18446744073709551616);
    const hi = Math.floor(t / TWO32);
    return [hi, t - hi * TWO32];
  }
  return [0, 0];
}

class Source {
  constructor(name, targets, start, stop) { this.name = name; this.targets = targets; this.start = start; this.stop = stop; }
  active(step) { return step >= this.start && (this.stop === null || step < this.stop); }
}

const floorDiv = (a, b) => (b >= TWO53 ? 0 : Math.floor(a / b));   // a >= 0 integer, b > 0 integer

class PoissonScalar extends Source {
  constructor(name, targets, p, weight, start, stop, seed) {
    super(name, targets, start, stop);
    this.kind = "poisson"; this.scalar = true;
    this.K = targets.length; this.p = p; this.weight = weight;
    this.bg = makeBitgen(seed, "poisson", name);
    this.table = 0.0 < p && p < 1.0 ? geometricTables(p) : null;
    this.buf = new Float64Array(8192); this.head = 0; this.tail = 0;   // pending event positions
    this.last = -1;
    this.rawH = new Uint32Array(4096); this.rawL = new Uint32Array(4096);
  }

  _push(x) {
    if (this.tail === this.buf.length) {               // compact, or grow when more than half is live
      const live = this.tail - this.head;
      if (live > this.buf.length / 2) {
        const nb = new Float64Array(this.buf.length * 2);
        nb.set(this.buf.subarray(this.head, this.tail));
        this.buf = nb;
      } else {
        this.buf.copyWithin(0, this.head, this.tail);
      }
      this.head = 0; this.tail = live;
    }
    this.buf[this.tail++] = x;
  }

  hits(step, out) {
    if (this.p <= 0.0 || this.K === 0) return 0;
    if (this.p >= 1.0) { out.set(this.targets); return this.K; }
    const rel = step - this.start, lo = rel * this.K, hi = (rel + 1) * this.K;
    const bg = this.bg, P2 = this.table.pow2, Q = this.table.q, T = Q.length;
    const RH = this.rawH, RL = this.rawL;
    while (this.last < hi) {
      let pos = this.last;
      bg.fillRaw(4096, RH, RL);                      // BATCH = 4096 raw draws, as sim.py
      for (let d = 0; d < 4096; d++) {
        const u = (RH[d] * 2097152 + (RL[d] >>> 11) + 1.0) * 1.1102230246251565e-16;   // ((raw>>11)+1) * 2^-53
        let G = 0, prod = 1.0;
        for (let t = 0; t < T; t++) {
          const cand = prod * Q[t];
          if (cand >= u) { prod = cand; G += P2[t]; }
        }
        pos += G + 1;
        this._push(pos);
      }
      if (!(pos < TWO53)) throw new Error("Poisson event position exceeds 2^53");
      this.last = pos;
    }
    // j = searchsorted(buf, hi, 'left'); take buf[:j]; keep events >= lo
    let n = 0;
    const buf = this.buf, tg = this.targets;
    let j = this.head;
    while (j < this.tail && buf[j] < hi) {
      const e = buf[j];
      if (e >= lo) out[n++] = tg[e - lo];
      j++;
    }
    this.head = j;
    return n;
  }

  pendingPositions() { return this.buf.subarray(this.head, this.tail); }
}

class PoissonArray extends Source {
  constructor(name, targets, rates, F, dtMs, frameSteps, weight, start, stop, seed) {
    super(name, targets, start, stop);
    this.kind = "poisson"; this.scalar = false;
    this.rates = rates; this.F = F; this.K = targets.length;
    this.dtMs = dtMs; this.frameSteps = frameSteps; this.weight = weight;
    this.bg = makeBitgen(seed, "poisson", name);
    this.f = -1;
    this.thrHi = new Float64Array(this.K); this.thrLo = new Float64Array(this.K);
    this.rawH = new Uint32Array(this.K); this.rawL = new Uint32Array(this.K);
    this.sel = new Int32Array(this.K); this.m = 0;
  }

  hits(step, out) {
    const f = floorDiv(step - this.start, this.frameSteps);
    if (f >= this.F) return 0;
    const K = this.K;
    if (K === 0) return 0;                            // random_raw(0) draws nothing
    if (f !== this.f) {
      this.f = f;
      const base = f * K, dt = this.dtMs;
      let m = 0;
      const TH = this.thrHi, TL = this.thrLo, R = this.rates;
      for (let i = 0; i < K; i++) {
        // bernoulliThreshold inlined: floor(p 2^64) as uint32 halves; p >= 1 -> 2^64 - 1; p <= 0 -> 0
        const p = R[base + i] * dt / 1000.0;
        let h = 0, l = 0;
        if (p >= 1.0) { h = 0xffffffff; l = 0xffffffff; }
        else if (p > 0.0) { const t = Math.floor(p * 18446744073709551616); h = Math.floor(t / TWO32); l = t - h * TWO32; }
        TH[i] = h; TL[i] = l;
        if (h !== 0 || l !== 0) this.sel[m++] = i;    // raw < 0 never holds: those draws only advance
      }
      this.m = m;
    }
    // one raw draw per target per step (K draws); only the draws with a nonzero threshold are formed
    const RH = this.rawH, RL = this.rawL, TH = this.thrHi, TL = this.thrLo, tg = this.targets, sel = this.sel, m = this.m;
    this.bg.drawSelected(K, sel, m, RH, RL);
    let n = 0;
    for (let q = 0; q < m; q++) {
      const i = sel[q], h = RH[q], th = TH[i];
      if (h < th || (h === th && RL[q] < TL[i])) out[n++] = tg[i];   // raw < floor(p 2^64)
    }
    return n;
  }
}

class Current extends Source {
  constructor(name, targets, inc, F, frameSteps, start, stop) {
    super(name, targets, start, stop);
    this.kind = "current";
    this.inc = inc; this.F = F; this.K = targets.length; this.frameSteps = frameSteps;
  }
}

// =============================================================================================
// simulator

function asFloat64(x) {
  if (x instanceof Float64Array) return x;
  return Float64Array.from(x.flat ? x.flat(Infinity) : x);
}

/** Shape of a scalar / (K,) / (F, K) input: nested JS arrays or {data: Float64Array, shape}. */
function shapeOf(x) {
  if (typeof x === "number") return { ndim: 0, data: Float64Array.of(x), shape: [] };
  if (x && x.data && x.shape) return { ndim: x.shape.length, data: asFloat64(x.data), shape: x.shape.slice() };
  if (Array.isArray(x) && x.length && Array.isArray(x[0])) {
    const F = x.length, K = x[0].length;
    for (const row of x) if (row.length !== K) throw new Error("ragged (F, K) array");
    return { ndim: 2, data: Float64Array.from(x.flat()), shape: [F, K] };
  }
  if (Array.isArray(x) || ArrayBuffer.isView(x)) return { ndim: 1, data: asFloat64(Array.from(x)), shape: [x.length] };
  throw new Error("unsupported input array");
}

export class Simulator {
  /**
   * new Simulator(net, seed, params = new LIFParams(), {gated = false, weights = null})
   * weights: optional Float32Array (nnz) replacing the weights derived from counts (tests only).
   */
  constructor(net, seed, params = new LIFParams(), opts = {}) {
    this.net = net;
    this.p = params;
    this.seed = String(seed);
    const d = params.derived();
    this.d = d;
    this.a = d.a; this.b = d.b; this.theta = d.theta; this.uReset = d.u_reset;
    this.nRef = d.n_ref; this.nDelay = d.n_delay;
    const n = net.n, nnz = net.nnz;
    if (opts.weights) {
      if (opts.weights.length !== nnz) throw new Error("weights length != nnz");
      this.w = Float32Array.from(opts.weights);
    } else {
      this.w = new Float32Array(nnz);
      const scale = params.w_syn_mV * d.c64;             // float64, as Python: w_syn * c64
      for (let j = 0; j < nnz; j++) this.w[j] = net.counts[j] * scale;   // float64 product, stored as float32
    }
    this.poissonW = Math.fround(params.poisson_weight_mV);
    this.u = new Float32Array(n);
    this.h = new Float32Array(n);
    this.uBits = new Uint32Array(this.u.buffer);
    this.hBits = new Uint32Array(this.h.buffer);
    this.last = new Float64Array(n).fill(-(2 ** 62));
    this.step = 0;
    this.slots = Array.from({ length: this.nDelay + 1 }, () => new Int32Array(0));
    this.sources = [];
    this.stats = { spikes: 0, syn_events: 0, poisson_events: 0 };
    this._recSteps = new Float64Array(1024); this._recIdx = new Int32Array(1024); this._rec = 0;
    this._recStart = 0;                                // first step raster() covers (a restore moves it)
    this.gated = !!opts.gated && this.theta >= 0;      // gating needs +0 < theta
    this._inSet = new Uint8Array(n);
    this._active = new Int32Array(Math.max(16, n)); this._nAct = 0;
    this._fired = new Int32Array(n);
    this._hit = new Int32Array(n);
  }

  // -- inputs --------------------------------------------------------------------------------
  _targets(neurons) {
    const t = Int32Array.from(Array.from(neurons, Number));
    const seen = new Uint8Array(this.net.n);
    for (let i = 0; i < t.length; i++) {
      if (t[i] < 0 || t[i] >= this.net.n || t[i] !== Number(neurons[i])) throw new Error("neuron index out of range");
      if (seen[t[i]]) throw new Error("duplicate neuron indices in one source");
      seen[t[i]] = 1;
    }
    return t;
  }

  _window(startMs, stopMs) {
    return [this.p.steps(startMs), stopMs === null || stopMs === undefined ? null : this.p.steps(stopMs)];
  }

  _name(name, kind) {
    name = name || `${kind}${this.sources.length}`;
    if (this.sources.some((s) => s.name === name)) throw new Error(`source name ${name} already used`);
    return name;
  }

  /** Simulator.add_poisson(neurons, rate_hz, start_ms, stop_ms, frame_ms, weight_mV, name) */
  addPoisson(neurons, rateHz, { startMs = 0.0, stopMs = null, frameMs = null, weightMv = null, name = null } = {}) {
    const t = this._targets(neurons);
    const [start, stop] = this._window(startMs, stopMs);
    name = this._name(name, "poisson");
    const w = weightMv === null || weightMv === undefined ? this.poissonW : Math.fround(Number(weightMv));
    const r = shapeOf(rateHz);
    for (let i = 0; i < r.data.length; i++) if (r.data[i] < 0) throw new Error("rates must be >= 0");
    let src;
    if (r.ndim === 0) {
      let p = r.data[0] * this.p.dt_ms / 1000.0;
      if (0.0 < p && p < 2 ** -40) p = 0.0;
      src = new PoissonScalar(name, t, p, w, start, stop, this.seed);
    } else {
      let F, fs;
      if (r.ndim === 1) { F = 1; fs = 2 ** 62; if (r.shape[0] !== t.length) throw new Error(`rates have ${r.shape[0]} columns for ${t.length} neurons`); }
      else {
        if (frameMs === null || frameMs === undefined) throw new Error("an (F, K) rate array needs frame_ms");
        fs = this.p.steps(frameMs);
        F = r.shape[0];
        if (r.shape[1] !== t.length) throw new Error(`rates have ${r.shape[1]} columns for ${t.length} neurons`);
      }
      src = new PoissonArray(name, t, r.data, F, this.p.dt_ms, fs, w, start, stop, this.seed);
    }
    this.sources.push(src);
    return name;
  }

  /** Simulator.add_current(neurons, amp_mV, start_ms, stop_ms, frame_ms, name) */
  addCurrent(neurons, ampMv, { startMs = 0.0, stopMs = null, frameMs = null, name = null } = {}) {
    const t = this._targets(neurons);
    const [start, stop] = this._window(startMs, stopMs);
    name = this._name(name, "current");
    const a = shapeOf(ampMv);
    const K = t.length;
    let F, fs, data;
    if (a.ndim === 0) { F = 1; fs = 2 ** 62; data = new Float64Array(K).fill(a.data[0]); }
    else if (a.ndim === 1) { F = 1; fs = 2 ** 62; data = a.data; if (a.shape[0] !== K) throw new Error(`currents have ${a.shape[0]} columns for ${K} neurons`); }
    else {
      if (frameMs === null || frameMs === undefined) throw new Error("an (F, K) current array needs frame_ms");
      fs = this.p.steps(frameMs); F = a.shape[0]; data = a.data;
      if (a.shape[1] !== K) throw new Error(`currents have ${a.shape[1]} columns for ${K} neurons`);
    }
    const inc = new Float32Array(F * K);
    const ki = this.d.k_i64;
    for (let i = 0; i < F * K; i++) inc[i] = data[i] * ki;       // (amps float64 * k_i64) -> float32
    this.sources.push(new Current(name, t, inc, F, fs, start, stop));
    return name;
  }

  clearInputs() { this.sources = []; }

  // -- dynamics ------------------------------------------------------------------------------
  _touch(i) { if (!this._inSet[i]) { this._inSet[i] = 1; this._active[this._nAct++] = i; } }

  _touchNonzero() {
    const ub = this.uBits, hb = this.hBits;
    for (let i = 0; i < this.net.n; i++) if (ub[i] !== 0 || hb[i] !== 0) this._touch(i);
  }

  _record(k, fired, nf) {
    if (this._rec + nf > this._recIdx.length) {
      const cap = Math.max(this._recIdx.length * 2, this._rec + nf);
      const s = new Float64Array(cap); s.set(this._recSteps.subarray(0, this._rec)); this._recSteps = s;
      const x = new Int32Array(cap); x.set(this._recIdx.subarray(0, this._rec)); this._recIdx = x;
    }
    for (let q = 0; q < nf; q++) { this._recSteps[this._rec + q] = k; this._recIdx[this._rec + q] = fired[q]; }
    this._rec += nf;
  }

  /** Simulator.run(duration_ms) or run(null, nSteps): advance and return this window's raster. */
  run(durationMs = null, nSteps = null) {
    if (nSteps === null || nSteps === undefined) nSteps = this.p.steps(durationMs);
    if (!Number.isSafeInteger(nSteps) || nSteps < 0) throw new Error("n_steps must be a non-negative integer");
    const net = this.net, n = net.n, indptr = net.indptr, indices = net.indices, dead = net.dead;
    const u = this.u, h = this.h, last = this.last, w = this.w;
    const a = this.a, b = this.b, theta = this.theta, uReset = this.uReset;
    const nRef = this.nRef, nDelay = this.nDelay, nslot = nDelay + 1, slots = this.slots;
    const fired = this._fired, hit = this._hit;
    const currents = this.sources.filter((s) => s.kind === "current");
    const poissons = this.sources.filter((s) => s.kind === "poisson");
    const gated = this.gated;
    const rec0 = this._rec, first = this.step;
    const stats = this.stats;
    const inSet = this._inSet, ub = this.uBits, hb = this.hBits;
    if (gated) this._touchNonzero();
    for (let it = 0; it < nSteps; it++) {
      const k = this.step;
      const lim = k - nRef;                            // frozen (refractory) iff last > lim
      // does any current source add to u this step?
      let driven = false;
      for (let c = 0; c < currents.length; c++) {
        const s = currents[c];
        if (s.active(k) && floorDiv(k - s.start, s.frameSteps) < s.F) { driven = true; break; }
      }
      let nf = 0;
      if (!driven) {
        // 1+2 fused (no current this step, so u after integration is what the threshold sees):
        // integrate every non-refractory neuron, u = u*a; u = u + h; h = h*b (each rounded to
        // float32), then fire if u > theta and alive. Refractory neurons are frozen and cannot fire.
        if (!gated) {
          for (let i = 0; i < n; i++) {
            if (last[i] > lim) continue;
            const v = Math.fround(Math.fround(u[i] * a) + h[i]);
            u[i] = v;
            h[i] = h[i] * b;
            if (v > theta && !dead[i]) fired[nf++] = i;
          }
        } else {
          const act = this._active, na = this._nAct;
          for (let q = 0; q < na; q++) {
            const i = act[q];
            if (last[i] > lim) continue;
            const v = Math.fround(Math.fround(u[i] * a) + h[i]);
            u[i] = v;
            h[i] = h[i] * b;
            if (v > theta && !dead[i]) fired[nf++] = i;
          }
          if (nf > 1) fired.subarray(0, nf).sort();    // typed-array sort is numeric
        }
      } else {
        // 1. integrate every non-refractory neuron
        if (!gated) {
          for (let i = 0; i < n; i++) {
            if (last[i] > lim) continue;
            u[i] = Math.fround(u[i] * a) + h[i];
            h[i] = h[i] * b;
          }
        } else {
          const act = this._active, na = this._nAct;
          for (let q = 0; q < na; q++) {
            const i = act[q];
            if (last[i] > lim) continue;
            u[i] = Math.fround(u[i] * a) + h[i];
            h[i] = h[i] * b;
          }
        }
        //    currents, in source order; refractory targets are frozen (sim.py restores them)
        for (let c = 0; c < currents.length; c++) {
          const s = currents[c];
          if (!s.active(k)) continue;
          const f = floorDiv(k - s.start, s.frameSteps);
          if (f >= s.F) continue;
          const tg = s.targets, inc = s.inc, base = f * s.K;
          for (let q = 0; q < s.K; q++) {
            const t = tg[q];
            if (gated) this._touch(t);
            if (last[t] > lim) continue;
            u[t] = u[t] + inc[base + q];
          }
        }
        // 2. threshold: ascending index, non-refractory, alive
        if (!gated) {
          for (let i = 0; i < n; i++) if (u[i] > theta && last[i] <= lim && !dead[i]) fired[nf++] = i;
        } else {
          const act = this._active, na = this._nAct;
          for (let q = 0; q < na; q++) { const i = act[q]; if (u[i] > theta && last[i] <= lim && !dead[i]) fired[nf++] = i; }
          if (nf > 1) fired.subarray(0, nf).sort();    // typed-array sort is numeric
        }
      }
      // 3. synapses: the spikes fired nDelay steps ago, rows in fired order, targets in CSR order
      const firedNow = fired.slice(0, nf);
      slots[k % nslot] = firedNow;
      const arriving = slots[(((k - nDelay) % nslot) + nslot) % nslot];
      for (let q = 0; q < arriving.length; q++) {
        const src = arriving[q], s0 = indptr[src], s1 = indptr[src + 1];
        for (let j = s0; j < s1; j++) {
          const t = indices[j];
          h[t] = h[t] + w[j];
          if (gated && !inSet[t]) this._touch(t);
        }
        stats.syn_events += s1 - s0;
      }
      //    then Poisson kicks, source by source
      for (let c = 0; c < poissons.length; c++) {
        const s = poissons[c];
        if (!s.active(k)) continue;
        const nh = s.hits(k, hit);
        if (nh > 0) {
          const wt = s.weight;
          for (let q = 0; q < nh; q++) { const t = hit[q]; u[t] = u[t] + wt; if (gated && !inSet[t]) this._touch(t); }
          stats.poisson_events += nh;
        }
      }
      // 4. reset
      if (nf) {
        for (let q = 0; q < nf; q++) { const i = firedNow[q]; u[i] = uReset; h[i] = 0.0; last[i] = k; }
        this._record(k, firedNow, nf);
        stats.spikes += nf;
      }
      // 5. (gated) drop neurons that are exactly +0.0 in both u and h
      if (gated) {
        const act = this._active;
        let m = 0;
        for (let q = 0; q < this._nAct; q++) {
          const i = act[q];
          if (ub[i] !== 0 || hb[i] !== 0) act[m++] = i; else inSet[i] = 0;
        }
        this._nAct = m;
      }
      this.step++;
    }
    return {
      steps: this._recSteps.slice(rec0, this._rec), neurons: this._recIdx.slice(rec0, this._rec),
      startStep: first, nSteps, nNeurons: n, dtMs: this.p.dt_ms,
    };
  }

  /** Every spike since construction or restoreCheckpoint() (Simulator.raster()). */
  raster() {
    return { steps: this._recSteps.slice(0, this._rec), neurons: this._recIdx.slice(0, this._rec),
      startStep: this._recStart, nSteps: this.step - this._recStart, nNeurons: this.net.n, dtMs: this.p.dt_ms };
  }

  /** Simulator.state_digest() */
  stateDigest() {
    const hs = new Sha256();
    hs.update("fishbrain-state-v1");
    const p = new Packer(hs);
    p.u64(this.step);
    p.flush();
    hs.update(new Uint8Array(this.u.buffer, this.u.byteOffset, this.u.byteLength));
    hs.update(new Uint8Array(this.h.buffer, this.h.byteOffset, this.h.byteLength));
    for (let i = 0; i < this.net.n; i++) p.i64(this.last[i]);
    const nslot = this.nDelay + 1;
    for (let k = 0; k < nslot; k++) {
      const sl = this.slots[(this.step + k) % nslot];
      p.u64(sl.length);
      for (let q = 0; q < sl.length; q++) p.i64(sl[q]);
    }
    for (const s of this.sources) {
      p.flush();
      hs.update(s.name);
      if (s.bg) hs.update(s.bg.stateJSON());
      if (s.kind === "poisson" && s.scalar) {
        const b = s.pendingPositions();
        for (let q = 0; q < b.length; q++) p.i64(b[q]);
      }
    }
    p.flush();
    return hs.hex();
  }

  v_mV() { return Float64Array.from(this.u, (x) => x + this.p.v_rest_mV); }
}

// =============================================================================================
// case files (the JSON written by js/export_case.py)

export function decodeF64(b64) {
  let bytes;
  if (typeof atob === "function") {
    const s = atob(b64);
    bytes = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  } else {
    throw new Error("no base64 decoder available");
  }
  if (bytes.length % 8) throw new Error("f64 payload length is not a multiple of 8");
  return new Float64Array(bytes.buffer, bytes.byteOffset, bytes.length / 8);
}

function decodeArray(x) {
  if (x && typeof x === "object" && "f64_b64" in x) return { data: decodeF64(x.f64_b64), shape: x.shape };
  return x;
}

/** Build a Simulator from a case object and add its sources, in order. */
export function simulatorFromCase(c, opts = {}) {
  const N = c.network;
  const net = new Network({ n: N.n, ids: N.ids, indptr: N.indptr, indices: N.indices, counts: N.counts, deadIndices: N.dead_indices });
  const params = new LIFParams(c.params);
  let weights = null;
  if (opts.useExportedWeights) {
    const bits = Uint32Array.from(c.weights_f32_bits);
    weights = new Float32Array(bits.buffer);
  }
  const sim = new Simulator(net, c.seed, params, { gated: !!opts.gated, weights });
  for (const s of c.sources) {
    const common = { startMs: s.start_ms, stopMs: s.stop_ms, frameMs: s.frame_ms, name: s.name };
    if (s.type === "poisson") sim.addPoisson(s.neurons, decodeArray(s.rate_hz), { ...common, weightMv: s.weight_mV });
    else if (s.type === "current") sim.addCurrent(s.neurons, decodeArray(s.amp_mV), common);
    else throw new Error(`unknown source type ${s.type}`);
  }
  return { sim, net, params };
}

// =============================================================================================
// Verify support: canonical inputs (fishbrain-inputs-v1), checkpoints (fishbrain-checkpoint-v1),
// and a source's random stream fast-forwarded without the network.
//
// Both byte formats are specified in fishbrain/snapshot.py (the reference implementation). The
// Python engine and this kernel write identical bytes for the same state (tests/test_verify.py).

export const INPUTS_MAGIC = "fishbrain-inputs-v1\n";
export const CHECKPOINT_MAGIC = "fishbrain-checkpoint-v1\n";
export const KIND_SCALAR = 1, KIND_ARRAY = 2, KIND_CURRENT = 3;
const NEVER_SPIKED = -(2 ** 62);
const CONST_FRAME_STEPS = 2 ** 62;
const M64 = (1n << 64n) - 1n;

const u8Of = (x) => {
  if (x instanceof Uint8Array) return x;
  if (x instanceof ArrayBuffer) return new Uint8Array(x);
  if (ArrayBuffer.isView(x)) return new Uint8Array(x.buffer, x.byteOffset, x.byteLength);
  throw new Error("expected bytes (Uint8Array or ArrayBuffer)");
};
const rawBytes = (ta) => new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
function hexBytes32(hex) {
  if (!/^[0-9a-f]{64}$/.test(hex)) throw new Error(`expected a sha256 hex digest, got ${hex}`);
  const b = new Uint8Array(32);
  for (let i = 0; i < 32; i++) b[i] = parseInt(hex.slice(2 * i, 2 * i + 2), 16);
  return b;
}
const sameBytes = (a, b) => { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; };

class ByteWriter {
  constructor(cap = 1 << 16) { this.u8 = new Uint8Array(cap); this.dv = new DataView(this.u8.buffer); this.n = 0; }
  room(k) {
    if (this.n + k <= this.u8.length) return;
    let c = this.u8.length * 2;
    while (c < this.n + k) c *= 2;
    const nb = new Uint8Array(c); nb.set(this.u8.subarray(0, this.n));
    this.u8 = nb; this.dv = new DataView(nb.buffer);
  }
  byte(x) { this.room(1); this.u8[this.n++] = x; }
  u32(x) { if (!Number.isSafeInteger(x) || x < 0 || x > 0xffffffff) throw new Error(`u32 out of range: ${x}`); this.room(4); this.dv.setUint32(this.n, x, true); this.n += 4; }
  u64(x) { const v = BigInt(x); if (v < 0n || v > M64) throw new Error(`u64 out of range: ${x}`); this.room(8); this.dv.setBigUint64(this.n, v, true); this.n += 8; }
  i64(x) {                                          // an integer-valued double (|x| < 2^63), as Packer.i64
    if (!Number.isInteger(x) || Math.abs(x) >= 9223372036854775808) throw new Error(`int64 out of range: ${x}`);
    this.room(8);
    const hi = Math.floor(x / TWO32);
    this.dv.setUint32(this.n, x - hi * TWO32, true); this.dv.setInt32(this.n + 4, hi, true);
    this.n += 8;
  }
  u128(x) { this.u64(x & M64); this.u64(x >> 64n); }
  f32(x) { this.room(4); this.dv.setFloat32(this.n, x, true); this.n += 4; }
  f64(x) { this.room(8); this.dv.setFloat64(this.n, x, true); this.n += 8; }
  bytes(b) { this.room(b.length); this.u8.set(b, this.n); this.n += b.length; }
  str(s) { const b = utf8(s); this.u32(b.length); this.bytes(b); }
  ints(arr) { this.u64(arr.length); for (let i = 0; i < arr.length; i++) this.i64(arr[i]); }
  done() { return this.u8.slice(0, this.n); }
}

class ByteReader {
  constructor(u8, what) { this.u8 = u8; this.dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength); this.o = 0; this.what = what; }
  need(k) { if (!(k >= 0) || this.o + k > this.u8.length) throw new Error(`${this.what}: truncated`); }
  byte() { this.need(1); return this.u8[this.o++]; }
  u32() { this.need(4); const x = this.dv.getUint32(this.o, true); this.o += 4; return x; }
  u64big() { this.need(8); const x = this.dv.getBigUint64(this.o, true); this.o += 8; return x; }
  u64n() { const x = this.u64big(); if (x > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${this.what}: value out of range`); return Number(x); }
  i64n() {                                          // exact, or the -2^62 never-spiked sentinel
    this.need(8);
    const lo = this.dv.getUint32(this.o, true), hi = this.dv.getInt32(this.o + 4, true);
    this.o += 8;
    if (hi === -1073741824 && lo === 0) return NEVER_SPIKED;
    const x = hi * TWO32 + lo;
    if (!Number.isSafeInteger(x)) throw new Error(`${this.what}: int64 out of range`);
    return x;
  }
  u128() { const lo = this.u64big(), hi = this.u64big(); return lo | (hi << 64n); }
  f32() { this.need(4); const x = this.dv.getFloat32(this.o, true); this.o += 4; return x; }
  f64() { this.need(8); const x = this.dv.getFloat64(this.o, true); this.o += 8; return x; }
  take(k) { this.need(k); const b = this.u8.slice(this.o, this.o + k); this.o += k; return b; }   // an aligned copy
  str() {
    const b = this.take(this.u32());
    try { return new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(b); }
    catch (e) { throw new Error(`${this.what}: a name is not UTF-8`); }
  }
  ints(max) {                                       // u64 count, count x i64
    const k = this.u64n();
    if (k > max) throw new Error(`${this.what}: list too long`);
    this.need(8 * k);
    const out = new Float64Array(k);
    for (let i = 0; i < k; i++) out[i] = this.i64n();
    return out;
  }
  magic(m) { if (!sameBytes(this.take(m.length), utf8(m))) throw new Error(`${this.what}: not ${m.trim()}`); }
  done() { if (this.o !== this.u8.length) throw new Error(`${this.what}: ${this.u8.length - this.o} trailing bytes`); }
}

function sourceKind(s) {
  if (s.kind === "current") return KIND_CURRENT;
  if (s.kind === "poisson") return s.scalar ? KIND_SCALAR : KIND_ARRAY;
  throw new Error(`unknown source kind ${s.kind}`);
}

/** fishbrain-inputs-v1 bytes of every source of `sim`, in order (snapshot.encode_inputs). */
export function encodeInputs(sim) {
  const w = new ByteWriter();
  w.bytes(utf8(INPUTS_MAGIC));
  w.u32(sim.sources.length);
  for (const s of sim.sources) {
    const k = sourceKind(s);
    if (s.stop !== null && s.stop < 0) throw new Error(`source ${s.name}: a negative stop step has no canonical encoding`);
    w.str(s.name);
    w.byte(k);
    w.i64(s.start); w.i64(s.stop === null ? -1 : s.stop);
    w.ints(s.targets);
    if (k === KIND_SCALAR) { w.f64(s.p); w.f32(s.weight); }
    else if (k === KIND_ARRAY) {
      if (s.rates.length !== s.F * s.K) throw new Error(`source ${s.name}: rates are not (F, K)`);
      w.u64(s.F); w.u64(s.frameSteps); w.f32(s.weight); w.bytes(rawBytes(s.rates));
    } else {
      if (s.inc.length !== s.F * s.K) throw new Error(`source ${s.name}: currents are not (F, K)`);
      w.u64(s.F); w.u64(s.frameSteps); w.bytes(rawBytes(s.inc));
    }
  }
  return w.done();
}

/** sha256 of encodeInputs(sim): the complete input specification (receipt inputs.spec). */
export function inputsDigest(sim) { return sha256Hex(encodeInputs(sim)); }

function frameStepsOf(x, what) {
  if (x === BigInt(CONST_FRAME_STEPS)) return CONST_FRAME_STEPS;
  if (x < 1n || x > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${what}: frame_steps out of range`);
  return Number(x);
}

/** Parse fishbrain-inputs-v1 bytes into source specs (snapshot.decode_inputs). */
export function decodeInputs(bytes) {
  const r = new ByteReader(u8Of(bytes), "inputs");
  r.magic(INPUTS_MAGIC);
  const nsrc = r.u32(), specs = [], names = new Set();
  for (let q = 0; q < nsrc; q++) {
    const name = r.str();
    if (names.has(name)) throw new Error(`inputs: source name ${name} repeated`);
    names.add(name);
    const kind = r.byte();
    if (kind !== KIND_SCALAR && kind !== KIND_ARRAY && kind !== KIND_CURRENT) throw new Error(`inputs: unknown source kind ${kind}`);
    const start = r.i64n(), stop = r.i64n();
    if (stop < -1 || start === NEVER_SPIKED) throw new Error(`inputs: source ${name} has a bad window`);
    const t = r.ints(r.u8.length);
    const targets = new Int32Array(t.length);
    for (let i = 0; i < t.length; i++) { if (t[i] < 0 || t[i] > 2147483647) throw new Error(`inputs: source ${name} target out of range`); targets[i] = t[i]; }
    const K = targets.length;
    const sp = { name, kind, start, stop: stop === -1 ? null : stop, targets, K };
    if (kind === KIND_SCALAR) {
      sp.p = r.f64(); sp.weight = r.f32();
      if (!(Number.isFinite(sp.p) && sp.p >= 0 && Number.isFinite(sp.weight))) throw new Error(`inputs: source ${name} has a bad rate or weight`);
    } else if (kind === KIND_ARRAY) {
      sp.F = r.u64n(); sp.frameSteps = frameStepsOf(r.u64big(), "inputs"); sp.weight = r.f32();
      sp.rates = new Float64Array(r.take(8 * sp.F * K).buffer);
      if (!Number.isFinite(sp.weight)) throw new Error(`inputs: source ${name} has a bad weight`);
      for (let i = 0; i < sp.rates.length; i++) if (!(Number.isFinite(sp.rates[i]) && sp.rates[i] >= 0)) throw new Error(`inputs: source ${name} has a bad rate`);
    } else {
      sp.F = r.u64n(); sp.frameSteps = frameStepsOf(r.u64big(), "inputs");
      sp.inc = new Float32Array(r.take(4 * sp.F * K).buffer);
      for (let i = 0; i < sp.inc.length; i++) if (!Number.isFinite(sp.inc[i])) throw new Error(`inputs: source ${name} has a non-finite current`);
    }
    specs.push(sp);
  }
  r.done();
  return specs;
}

/** Fresh source objects (streams at their seeded start) from decoded specs, checked against n. */
export function sourcesFromSpecs(specs, seed, params, n) {
  const out = [];
  for (const sp of specs) {
    const seen = new Uint8Array(n);
    for (let i = 0; i < sp.K; i++) {
      const t = sp.targets[i];
      if (t >= n) throw new Error(`inputs: source ${sp.name} targets a neuron outside the network`);
      if (seen[t]) throw new Error(`inputs: source ${sp.name} targets a neuron twice`);
      seen[t] = 1;
    }
    if (sp.kind === KIND_SCALAR) out.push(new PoissonScalar(sp.name, sp.targets, sp.p, sp.weight, sp.start, sp.stop, seed));
    else if (sp.kind === KIND_ARRAY) out.push(new PoissonArray(sp.name, sp.targets, sp.rates, sp.F, params.dt_ms, sp.frameSteps, sp.weight, sp.start, sp.stop, seed));
    else out.push(new Current(sp.name, sp.targets, sp.inc, sp.F, sp.frameSteps, sp.start, sp.stop));
  }
  return out;
}

/** numpy PCG64.advance(delta): the state `delta` raw draws later (LCG jump-ahead, BigInt). */
export function pcgAdvance(g, delta) {
  const inc = g.incBigInt;
  let accMult = 1n, accPlus = 0n, curMult = PCG_MULT_128, curPlus = inc;
  let d = BigInt(delta) & M128;
  while (d > 0n) {
    if (d & 1n) { accMult = (accMult * curMult) & M128; accPlus = (accPlus * curMult + curPlus) & M128; }
    curPlus = ((curMult + 1n) * curPlus) & M128;
    curMult = (curMult * curMult) & M128;
    d >>= 1n;
  }
  g._setState((accMult * g.stateBigInt + accPlus) & M128, inc);
}

/**
 * Advance fresh sources to where a run from step 0 leaves them at `step`, without the network
 * (snapshot.fast_forward_sources). A per-frame source draws exactly K raw numbers on every active
 * step inside its frames; a scalar source generates batches until its last event position passes
 * the step's lattice end. Neither depends on the neurons, so this is exact.
 *
 * A run computes steps 0, 1, 2, ... only, so a source whose start is negative (it began before the
 * run) first draws at step 0, not at `start`: its active steps are [max(start, 0), min(stop, end of
 * its frames)). Counting from `start` instead would demand draws the simulator never made.
 */
export function fastForwardSources(sources, step) {
  for (const s of sources) {
    if (s.kind === "current") continue;
    const end = s.stop === null ? step : Math.min(step, s.stop);
    const first = Math.max(s.start, 0);                 // the first step the run computes with it active
    if (s.scalar) {
      if (s.p <= 0.0 || s.p >= 1.0 || s.K === 0) continue;
      if (end - 1 >= first) s.hits(end - 1, new Int32Array(s.K));   // monotone batch loop: one call = every step's
    } else {
      if (s.K === 0) continue;
      const lim = BigInt(s.start) + BigInt(s.F) * BigInt(s.frameSteps);
      const e = BigInt(end) < lim ? BigInt(end) : lim;
      const count = e - BigInt(first);
      if (count > 0n) { pcgAdvance(s.bg, count * BigInt(s.K)); s.f = -1; }
    }
  }
}

/** Per source: {name, kind, state, inc, last, pending} (decimal strings for the 128-bit values). */
export function sourceRecords(sources) {
  return sources.map((s) => {
    const kind = s.kind === "current" || s.kind === "poisson" ? sourceKind(s) : s.kind;
    const rec = { name: s.name, kind, state: null, inc: null, last: null, pending: null };
    if (kind !== KIND_CURRENT) {
      rec.state = (s.bg ? s.bg.stateBigInt : s.state).toString();
      rec.inc = (s.bg ? s.bg.incBigInt : s.inc).toString();
    }
    if (kind === KIND_SCALAR) {
      rec.last = s.last;
      rec.pending = Array.from(s.pendingPositions ? s.pendingPositions() : s.pending);
    }
    return rec;
  });
}

/** Refractory queue a run leaves: the spikes of the last min(step, nRef - 1) steps, oldest first. */
function derivedRef(last, step, nRef) {
  if (nRef <= 1) return [];
  const R = Math.min(step, nRef - 1), lo = step - R;
  const out = Array.from({ length: R }, () => []);
  for (let i = 0; i < last.length; i++) if (last[i] >= lo && last[i] !== NEVER_SPIKED) out[last[i] - lo].push(i);
  return out;
}

/** The canonical fishbrain-checkpoint-v1 bytes of the simulator's state (snapshot.serialize). */
export function checkpointBytes(sim, { embedInputs = true } = {}) {
  const inputs = encodeInputs(sim);
  const n = sim.net.n, nslot = sim.nDelay + 1;
  const w = new ByteWriter(1 << 20);
  w.bytes(utf8(CHECKPOINT_MAGIC));
  w.bytes(hexBytes32(sim.net.digest())); w.bytes(hexBytes32(sim.p.digest())); w.bytes(hexBytes32(sha256Hex(inputs)));
  w.str(sim.seed);
  w.u64(n); w.u32(sim.nDelay); w.u32(sim.nRef); w.u64(sim.step);
  w.bytes(rawBytes(sim.u)); w.bytes(rawBytes(sim.h));
  for (let i = 0; i < n; i++) w.i64(sim.last[i]);
  for (let j = 0; j < nslot; j++) w.ints(sim.slots[(sim.step + j) % nslot]);
  const ref = derivedRef(sim.last, sim.step, sim.nRef);
  w.u32(ref.length);
  for (const r of ref) w.ints(r);
  w.u32(sim.sources.length);
  for (const s of sim.sources) {
    const k = sourceKind(s);
    w.str(s.name); w.byte(k);
    if (k !== KIND_CURRENT) { w.u128(s.bg.stateBigInt); w.u128(s.bg.incBigInt); }
    if (k === KIND_SCALAR) { w.i64(s.last); w.ints(s.pendingPositions()); }
  }
  if (embedInputs) { w.byte(1); w.u64(inputs.length); w.bytes(inputs); } else w.byte(0);
  w.bytes(hexBytes32(sim.stateDigest()));
  const body = w.done();
  const out = new Uint8Array(body.length + 32);
  out.set(body);
  out.set(new Sha256().update(body).digest(), body.length);
  return out;
}

/**
 * Parse a checkpoint and check its self-consistency (snapshot.parse): checksum, structure, ranges,
 * the refractory queue and delay slots against the last-spike steps, each scalar source's last
 * position, and the recorded state digest against the digest recomputed from the parsed state.
 * Returns the parsed state with stateDigest (hex), before anything is restored. Throws on any defect.
 */
export function parseCheckpoint(bytes) {
  const all = u8Of(bytes);
  if (all.length < CHECKPOINT_MAGIC.length + 160) throw new Error("checkpoint: truncated");
  const body = all.subarray(0, all.length - 32);
  if (!sameBytes(new Sha256().update(body).digest(), all.subarray(all.length - 32))) {
    throw new Error("checkpoint: checksum mismatch (corrupted or edited bytes)");
  }
  const r = new ByteReader(body, "checkpoint");
  r.magic(CHECKPOINT_MAGIC);
  const networkDigest = toHex(r.take(32)), paramsDigest = toHex(r.take(32)), inputsDigest = toHex(r.take(32));
  const seed = r.str();
  const n = r.u64n(), nDelay = r.u32(), nRef = r.u32(), step = r.u64n();
  if (n > 2147483647) throw new Error("checkpoint: size out of range");
  const u = new Float32Array(r.take(4 * n).buffer), h = new Float32Array(r.take(4 * n).buffer);
  for (let i = 0; i < n; i++) if (!Number.isFinite(u[i]) || !Number.isFinite(h[i])) throw new Error("checkpoint: non-finite voltages or conductances");
  r.need(8 * n);
  const last = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = r.i64n();
    if (x !== NEVER_SPIKED && (x < 0 || x >= step)) throw new Error("checkpoint: a last-spike step is outside [0, step)");
    last[i] = x;
  }
  const lo = step - nDelay - 1, slots = [];
  for (let j = 0; j <= nDelay; j++) {
    const sl = r.ints(n), t = lo + j;
    if (t < 0 && sl.length) throw new Error(`checkpoint: delay slot ${j} is not a set of spikes at step ${t}`);
    for (let q = 0; q < sl.length; q++) {
      if (sl[q] < 0 || sl[q] >= n || (q && sl[q] <= sl[q - 1]) || last[sl[q]] < t) throw new Error(`checkpoint: delay slot ${j} is not a set of spikes at step ${t}`);
    }
    slots.push(Int32Array.from(sl));
  }
  for (let i = 0; i < n; i++) {
    if (last[i] === NEVER_SPIKED || last[i] < Math.max(lo, 0)) continue;
    const sl = slots[last[i] - lo];
    let a = 0, b = sl.length;
    while (a < b) { const m = (a + b) >> 1; if (sl[m] < i) a = m + 1; else b = m; }
    if (a >= sl.length || sl[a] !== i) throw new Error(`checkpoint: neuron ${i} spiked at step ${last[i]} but is not in that delay slot`);
  }
  const R = r.u32(), ref = [];
  for (let q = 0; q < R; q++) ref.push(r.ints(n));
  const want = derivedRef(last, step, nRef);
  if (ref.length !== want.length || ref.some((x, q) => x.length !== want[q].length || x.some((v, i) => v !== want[q][i]))) {
    throw new Error("checkpoint: the refractory queue does not match the last-spike steps");
  }
  const nsrc = r.u32(), sources = [], names = new Set();
  for (let q = 0; q < nsrc; q++) {
    const name = r.str(), kind = r.byte();
    if (names.has(name) || (kind !== KIND_SCALAR && kind !== KIND_ARRAY && kind !== KIND_CURRENT)) throw new Error(`checkpoint: bad source entry ${name}`);
    names.add(name);
    const s = { name, kind, state: null, inc: null, last: null, pending: null };
    if (kind !== KIND_CURRENT) {
      s.state = r.u128(); s.inc = r.u128();
      if ((s.inc & 1n) === 0n) throw new Error(`checkpoint: source ${name} has an even PCG64 increment`);
    }
    if (kind === KIND_SCALAR) {
      s.last = r.i64n();
      s.pending = r.ints(r.u8.length);
      const b = s.pending;
      let ok = s.last === (b.length ? b[b.length - 1] : -1);
      for (let i = 0; ok && i < b.length; i++) if (b[i] < 0 || b[i] >= TWO53 || (i && b[i] <= b[i - 1])) ok = false;
      if (!ok) throw new Error(`checkpoint: source ${name} pending events are inconsistent`);
    }
    sources.push(s);
  }
  const flag = r.byte();
  if (flag !== 0 && flag !== 1) throw new Error("checkpoint: bad inputs flag");
  const inputs = flag ? r.take(r.u64n()) : null;
  const recorded = toHex(r.take(32));
  r.done();
  if (inputs) {
    if (sha256Hex(inputs) !== inputsDigest) throw new Error("checkpoint: embedded inputs do not match the inputs digest");
    const specs = decodeInputs(inputs);
    if (specs.length !== sources.length || specs.some((sp, q) => sp.name !== sources[q].name || sp.kind !== sources[q].kind)) {
      throw new Error("checkpoint: sources differ from the embedded inputs");
    }
  }
  // the digest, recomputed by Simulator.stateDigest itself over the parsed state
  const nslot = nDelay + 1, ring = new Array(nslot);
  for (let j = 0; j < nslot; j++) ring[(step + j) % nslot] = slots[j];
  const standIn = {
    step, u, h, last, nDelay, net: { n }, slots: ring,
    sources: sources.map((s) => ({
      name: s.name, kind: s.kind === KIND_CURRENT ? "current" : "poisson", scalar: s.kind === KIND_SCALAR,
      bg: s.kind === KIND_CURRENT ? null : PCG64.fromState(s.state, s.inc), pendingPositions: () => s.pending,
    })),
  };
  const stateDigest = Simulator.prototype.stateDigest.call(standIn);
  if (stateDigest !== recorded) throw new Error("checkpoint: the recorded state digest does not match the state");
  return { networkDigest, paramsDigest, inputsDigest, seed, n, nDelay, nRef, step, u, h, last, slots, ref, sources, inputs, stateDigest };
}

/**
 * A Simulator in exactly the checkpoint's state (snapshot.deserialize). `inputs` (fishbrain-inputs-v1
 * bytes) is required when the checkpoint does not embed them and must equal them when it does.
 * The network and parameters must be the ones the checkpoint names (their digests are compared).
 */
export function restoreCheckpoint(bytes, { net, params, inputs = null, gated = false } = {}) {
  const ck = parseCheckpoint(bytes);
  if (!net || !params) throw new Error("restoreCheckpoint needs {net, params}");
  if (net.n !== ck.n || net.digest() !== ck.networkDigest) throw new Error("checkpoint: made for a different network");
  if (params.digest() !== ck.paramsDigest) throw new Error("checkpoint: made for different parameters");
  const d = params.derived();
  if (d.n_delay !== ck.nDelay || d.n_ref !== ck.nRef) throw new Error("checkpoint: delay or refractory steps differ from the parameters");
  let ib = ck.inputs;
  if (inputs) {
    const given = u8Of(inputs);
    if (ib && !sameBytes(ib, given)) throw new Error("checkpoint: embedded inputs differ from the inputs given");
    ib = given;
  }
  if (!ib) throw new Error("checkpoint: no inputs embedded and none given");
  if (sha256Hex(ib) !== ck.inputsDigest) throw new Error("checkpoint: inputs do not match the inputs digest");
  const specs = decodeInputs(ib);
  if (specs.length !== ck.sources.length || specs.some((sp, q) => sp.name !== ck.sources[q].name || sp.kind !== ck.sources[q].kind)) {
    throw new Error("checkpoint: sources differ from the inputs");
  }
  for (let i = 0; i < ck.n; i++) if (net.dead[i] && ck.last[i] !== NEVER_SPIKED) throw new Error("checkpoint: a lesioned neuron has spiked");
  for (const sl of ck.slots) for (let q = 0; q < sl.length; q++) if (net.dead[sl[q]]) throw new Error("checkpoint: a lesioned neuron has spiked");
  const sim = new Simulator(net, ck.seed, params, { gated });
  sim.sources = sourcesFromSpecs(specs, ck.seed, params, net.n);
  sim.sources.forEach((s, q) => {
    const c = ck.sources[q];
    if (c.kind !== KIND_CURRENT) { s.bg = PCG64.fromState(c.state, c.inc); if (c.kind === KIND_ARRAY) s.f = -1; }
    if (c.kind === KIND_SCALAR) {
      s.buf = new Float64Array(Math.max(8192, 2 * c.pending.length));
      s.buf.set(c.pending);
      s.head = 0; s.tail = c.pending.length; s.last = c.last;
    }
  });
  sim.step = ck.step;
  sim.u.set(ck.u); sim.h.set(ck.h); sim.last.set(ck.last);
  const nslot = ck.nDelay + 1;
  for (let j = 0; j < nslot; j++) sim.slots[(ck.step + j) % nslot] = ck.slots[j].slice();
  sim._recStart = ck.step;
  if (sim.stateDigest() !== ck.stateDigest) throw new Error("checkpoint: the restored state does not reproduce its digest");
  return sim;
}

Simulator.prototype.checkpoint = function (opts) { return checkpointBytes(this, opts); };
