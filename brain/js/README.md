# FISHBRAIN JS kernel

`fishbrain-sim.mjs` is a bit-exact JavaScript port of `fishbrain/sim.py` (`Simulator`). Given the
same wiring, parameters, seed string and inputs, a browser gets the same spike hash and state digest
as the Python engine, byte for byte. Verify rests on this. `tests/test_js_parity.py` proves it and
proves that the proof can fail.

| file | runs in | what it is |
|---|---|---|
| `fishbrain-sim.mjs` | **browser** and Node | the kernel: SHA-256, SeedSequence/PCG64, LIFParams, Network, Simulator, spikeHash, simulatorFromCase |
| `run_case.mjs` | Node only (`node:fs`, `node:os`, `node:path`, `node:url`) | CLI: runs one case, prints hashes, timing and mismatches; exit 0 = identical |
| `export_case.py` | Python (brain venv) | writes cases (`fishbrain-js-case-v1` JSON) with Python's answers |
| `verify.mjs` | **browser** and Node, no imports | `verify(receipt, fetcher, {trustedKernels, trustedCheckpoints, chain, chainLookup, trustedBrains})` and `verifyLink`: loads the kernel only from verified bytes, refuses an unanchored checkpoint |

## Browser-safe

The whole of `fishbrain-sim.mjs` is browser-safe. It has **zero imports**, no `node:` specifiers,
no `process`, `Buffer` or `require`, and `test_kernel_is_browser_safe` pins that. It needs ES2020:
BigInt and `BigUint64Array`, `DataView.setBigUint64`, typed arrays, `Math.fround`, `TextEncoder`
(`TextDecoder` for checkpoints), and `atob` (used only by `decodeF64` for case payloads). It throws on a big-endian platform. The
source is pure ASCII, so hashing the fetched text or its bytes gives the same `sha256`.

Checked in a real browser on 2026-09-26: Chrome 152 imported the
file unchanged as a module and re-ran s1, s5 and the real gate case, full and gated, with zero
mismatches. The kernel's own `sha256Hex` of the fetched source equaled `shasum -a 256`.

## Verify: how the front end calls it

`verify.mjs` checks one decision receipt by re-running the fish's brain for the decision window in
the viewer's browser. It has **no static imports**. It fetches the kernel once, copies the bytes
into memory only it holds, hashes them with WebCrypto, and imports the module from exactly those
bytes: a `blob:` URL in browsers, or a `data:` URL where `blob:` modules are unsupported (Node).
The page never loads `fishbrain-sim.mjs` itself. The kernel and the case are fetched once per call;
every link of a checkpoint chain reuses them.

```js
import { verify } from "./verify.mjs";

// sha256 of each published fishbrain-sim.mjs this page vouches for (`shasum -a 256 fishbrain-sim.mjs`).
// The receipt's kernel hash says which code ran. This list is what makes that code the honest one.
const TRUSTED_KERNELS = ["<sha256 of the published kernel>"];

const fetcher = async (name, receipt) => {
  if (name === "kernel") return fetch(`/kernels/${receipt.kernel.sha256}.mjs`);
  if (name === "case") return fetch(`/cases/${receipt.brain.network}-${receipt.inputs.spec}.json`);
  if (name === "checkpoint") return fetch(`/checkpoints/${receipt.window.snapshot}.fbck`);   // content-addressed
  if (name === "chain") return fetch(`/chains/${receipt.window.snapshot}.json`);            // links back to genesis
  if (name === "bridge") return fetch(`/bridges/${receipt.brain.bridge}.fbbt`);            // content-addressed; never asked when null
};

const r = await verify(receipt, fetcher, {
  trustedKernels: TRUSTED_KERNELS,
  // optional fast path: window starts whose on-chain commitment this page has already checked itself
  trustedCheckpoints: [{ step, snapshot, network, params, inputs, seed }],
  // optional: answers two chain questions; with it, seed_anchored means the commit predates the seed slot (see "The seed anchor": signer and uniqueness are the page's checks)
  chainLookup: async (q) => q.kind === "block" ? rpcBlock(q.slot) : rpcTransaction(q.signature),
  // optional: the (network, provenance table) pairs this page re-derived from the frozen bridge spec
  trustedBrains: [{ network: "<Network.digest()>", bridge: "<sha256 of its table>" | null }],
});                                  // gated replay by default
if (r.ok) light(r.raster);          // r.raster.steps / r.raster.neurons: the window's spikes
if (r.ok) paintBridge(r.bridge.mask);   // mask[k] != 0: CSR entry k carries a bridge part -> --bridge orange
else explain(r.check, r.detail);    // which check failed, and why, in words
// r.anchor.by: "genesis" | "trusted" | "chain"     r.seed_anchored: true only with a chainLookup that proved it
```

`verify()` fails unless every check holds. The checks run in the order below, and `r.check` names
the first one that failed. (`receipt` means a field is missing or has the wrong type.)

| check | holds when | what it stops |
|---|---|---|
| `kernel` | the receipt's `kernel.sha256` is in `trustedKernels`; the fetched bytes hash to it; the code executed is those bytes; `name` and `version` match | a tampered kernel byte; a kernel swapped between the hash and execution (the swap never runs); a self-consistent kernel nobody vouched for |
| `seed` | `seed.source` is `"solana blockhash"` and `seed.value` is the canonical base58 of 32 bytes with at least 8 distinct byte values. With a `chainLookup`, nothing the chain says contradicts it (see the seed anchor). At step 0, the state built from the rule's seed string equals `window.snapshot`. From a checkpoint: its recorded seed is the rule's string, **and** every random stream sits exactly where that seed puts it at `start_step` | a receipt naming seed 0 over a seed-3 run; noise picked by moving a stream to another offset, even with every hash recomputed; the all-zero `"1"*32` and other degenerate values; a blockhash that is not the pinned slot's |
| `network` | `Network.digest()` of the served wiring equals `brain.network` | one synapse changed |
| `bridge` | `brain.bridge` is present (absent is not null). **Non-null:** the fetched table hashes to it, parses as canonical fishbrain-bridge-table-v1, lists exactly the network's pairs in CSR order, and on every pair measured + bridged per class equals the network's count; at least one count is bridged. **Null:** nothing is fetched and every count is measured. With `trustedBrains`, the receipt's `(network, bridge)` is one of its entries. Skipped for chain links | a table that relabels bridged links measured, drops a bridge link, is for another gain, or is stale (against the receipt's hash); a dropped link, another gain, a stale build or a bridge part named on the wrong pair even with the receipt's hash recomputed; an all-measured table instead of null; with `trustedBrains`, a resealed relabel and a bridged network declaring null |
| `params` | `LIFParams.digest()` equals `brain.params`, and dt equals `window.dt_ms` | different parameters |
| `inputs` | sha256 of the **complete** input specification equals `inputs.spec`. If `inputs.frames` is present, it equals the sha256 of the `retina` source's rates. | an extra hidden source (a 30 mV current into Mauthner); a changed rate byte; a moved start/stop |
| `snapshot` | (start_step > 0) the checkpoint parses and is self-consistent. Its state digest, computed from its bytes before restoring, equals `window.snapshot`. It is at `start_step` and names this wiring, these parameters and these inputs | a flipped checkpoint byte; an edited state with every hash resealed but the receipt left as it was |
| `anchor` | the start state is one the run really reached: genesis at step 0; past it, a `trustedCheckpoints` entry for exactly this start, or a chain of links that replays from genesis (or a trusted checkpoint) to exactly `(start_step, window.snapshot)`. **Fails closed**: a checkpoint with neither is refused | a fabricated neural state (the left Mauthner cell's u/h nudged so it fires) with every hash, the snapshot, the spikes and the end state recomputed to match |
| `spikes` | the replayed window's `spikeHash` equals `brain.spikes` | a claimed decision the brain did not make |
| `state` | `stateDigest()` after the window equals `brain.state` | a claimed end state |

**Bridge provenance (fishbrain-bridge-table-v1; `fishbrain/bridgetable.py` is the reference).**
`Network.digest()` hashes arrays only, so it cannot say which part of a pair's signed count is a
measured Fish1 synapse and which part a G1c bridge link (class a, b or c) added. The table lists
**every** pair of the network, in CSR order, with its measured count and its bridged count per
class. Its header carries the bridge version, the frozen spec's edge-list sha256, the topology tag,
the class names, the gains (`g_a`, `g_c`) and the ablations:

    magic "fishbrain-bridge-table-v1\n"; str bridge version; 32 raw bytes edge-list sha256; str topology;
    u32 K + K str classes; u32 G + G (str key, i64 gain), keys ascending; u32 A + A str ablations, ascending;
    u64 n; u64 P; then P i64 pre, P i64 post, P i64 measured, P*K i64 bridged (row-major)

A `str` is a u32 length plus printable ASCII. Everything is little-endian. Pairs are strictly
increasing, values stay within +-2^48 (so a browser sums them exactly), and no trailing bytes are
allowed. Each table has exactly one encoding.
- `receipt.brain.bridge` is `sha256(table)`, or `null` for an unbridged network.
- `snapshot.window_receipt(..., bridge=bytes)` writes the field and refuses a table that does not
  account for `sim.net`. It also refuses an all-measured table, and `None` for a network whose meta
  names a bridge (`bridge.network` sets it). `bridgetable.from_bridged(bw)` builds the table of a
  `bridge.Bridged` network without touching `bridge.py`.
- Python `encode_parts` and JS `encodeBridgeTable` write identical bytes. Python `decode` and JS
  `parseBridgeTable` refuse the same malformed ones (14 byte-level edits, 8 bad column sets).

`r.bridge` (on `ok`) is what the UI colours from:
- `mask`: a `Uint8Array(nnz)` where bit q means CSR entry k carries a class-q part.
- `table`: the verified table.
- `synapse_count_share`: exactly `provenance.synapse_count_share`'s numbers. It is secondary and
  never the headline.
- `edge_provenance` and `bridged_digest`: recomputed from the table. For the G1c network they
  equal `gate_results.json`'s `edge_provenance_digest` (`662772ed...`) and `bridged_digest`
  (`279ae5fa...`), so a page can match the receipt to the network the gate measured.
- `trusted` (with `trust_reason` when it is false).
- `decision_path_share: {computed: false, source}`.

**The decision-path share stays server-computed.** `provenance.py` traces every spike's drive
composition through the run (float64 bookkeeping over the raster, the Poisson streams and the
table). Verify returns the verified table and the replayed raster, but does not port that trace.

On the real G1c network (g_a 1, g_c 4) the table is 2.03 MB (42,342 pairs). JS parses it in about
3 ms. In the test, a 7,000-step window of the bridged gate fish verifies end to end with it.

**`trustedBrains` is required (fails closed, 2026-09-26).** The table accounts for every count, so a table that moves
bridged counts into the measured column still sums to the same counts. The bridge forge review built a count-swap:
the 216 escape-crossing links labelled measured, and 216 measured pairs with equal counts labelled bridged. It kept
**every summary number identical** to the honest table. So an "ok but untrusted" result would hide the lie rather
than show it. Verify therefore refuses at `bridge` unless `opts.trustedBrains` lists the receipt's `(network, bridge)`
pair: the builds the page re-derived from the frozen spec (`bridge.network` + `bridgetable.from_bridged`, or the gate's
published digests), with `bridge: null` entries for unbridged networks it serves. Class names must be single lowercase
letters, so `__proto__` or `measured` can't hide a class or collide with a total. For the G1c fish the pair is
network `0b6f3369f21f5c626d9ba1ba4bc02dfefd1e51b945b3355fcd6730c9aa083b4a`, bridge
`2340789188ee7f8ac4aab405503800f863645e04b95ac255cbe3dc44f2cb1f4a`.

**The seed rule (fishbrain-seed-v1).** The simulation seed string is
`"fishbrain-seed-v1:solana-blockhash:" + seed.value`. The engine is built as
`Simulator(net, seed=snapshot.seed_string(blockhash))`, and each Poisson stream stays
`sha256(seed | "poisson" | name)`. Verify never reads a case file's `seed`. The served case in the
tests keeps `"seed": "G1-seed-0"` to prove it. A value must be canonical base58 of exactly 32 bytes
**and not degenerate**: a blockhash is a SHA-256 output, so fewer than 8 distinct byte values (all
zero, which is `"1"*32`; all 0xff; a short repeated pattern) is refused. A real blockhash does that
with probability below 1e-37. Python `snapshot.seed_string` and JS `seedString` apply the same rule,
and a parity test holds them to it at the 7/8 boundary.

**The seed anchor: format checks cannot stop grinding.** Any well-formed value could have been picked
by an operator who tried many and kept the one whose noise made the decision look good. The seed is
unpickable only when it is the real blockhash of a slot fixed *before* that slot existed. Verify
checks that when the page passes `chainLookup` and the receipt carries `seed.commit`, the signature
of the run-commit Memo:
1. `chainLookup({kind: "transaction", signature: seed.commit})` returns `{slot, memo}`. The memo must be
   `fishbrain:run:v1 network=<hex> params=<hex> kernel=<hex> seed_slot=<n>`
   (`snapshot.run_commit_memo` writes it), naming this receipt's network, params and kernel.
2. The memo pins its seed slot, and it landed before it: `slot < seed_slot`. Without the pin, "a slot
   after the memo" would still let the operator choose among every later slot (2.5 per second).
3. `chainLookup({kind: "block", slot: seed_slot})` returns `{blockhash}`, and it equals `seed.value`.

Then `r.seed_anchored` is `true` and `r.seed_anchor` is `{slot, commit_slot, commit}`. On Solana RPC the
lookups are `getTransaction(signature)` (slot, and the memo from the Memo program's log line) and
`getBlock(slot, {transactionDetails: "none", rewards: false})`. A skipped slot returns `null`.
Anything the lookup cannot answer (RPC down, pruned history, no transaction found) should throw or
return nothing. It leaves `ok` alone and gives `seed_anchored: false` with a reason, because a dead RPC
is not a forgery. An answer that **contradicts** the receipt fails at `seed`: the slot's blockhash
differs, the memo landed at or after its seed slot, the memo names another brain, it pins no slot,
the slot was skipped, or the transaction is not a run-commit. Without a `chainLookup`, every result
says `seed_anchored: false`. Verify does the comparisons itself. The lookup only reports chain
facts, so a page can back it with an RPC or with a supplied proof. Verify cannot check that the fish
posted exactly one run-commit for the run. A second commit is a second roll, and that check is the
page's, over the committing account's history.

**What `seed_anchored: true` does and does not prove (from the 2026-09-26 re-forge review).** It proves the commit
memo landed before `seed_slot` and that the slot's blockhash equals the seed, *as reported by `chainLookup`*. Verify
trusts `chainLookup` completely, and it does **not** check who signed the memo. So the page must: (1) back `chainLookup`
with an RPC or proof it trusts; (2) check that the run-commit transaction was signed by the run's published account;
(3) check that the account posted exactly one run-commit for the run. Without all three, show "seed committed" rather
than "seed proven unpickable".

**Trusted kernels:** list only the current kernel (`fishbrain-js-kernel-v3`, sha256
`95d03374475a83097f0e817652cfa6505b3dea5dceab1b9f24909cf408847786`). Trusting v2 reopens the negative-start hole
(a v2 forgery returns ok). A kernel upgrade also makes earlier runs' commit memos name an old kernel, so re-verify them
with that kernel only when it has no known holes.

**Why the seed check works mid-run.** A source's random stream never depends on the neurons. A
per-frame source draws exactly K raw numbers on every active step inside its frames. A scalar
source draws 4096-number batches until its last event position passes the step. So the
stream position at `start_step` is a function of (seed, inputs, step) alone. Verify rebuilds it
without the network (`fastForwardSources`: an O(log n) PCG64 jump for per-frame sources, one
batch loop for scalar ones) and compares it to the checkpoint. In the tests this catches a stream
moved one batch ahead even when the window's spikes come out identical. A run computes steps 0, 1,
2, ... only, so a source with a **negative start** (it began before the run) draws from step 0.
Kernel v3 counts its draws from `max(start, 0)`, and so does `snapshot.fast_forward_sources`. v2
counted from the negative start. It then rejected the honest receipt at `seed` and accepted a
forgery whose stream had been moved to where v2 wanted it. The tests check both functions against
real runs in seven negative-start variants, and a v2-count kernel copy must reopen the hole.

**Inputs (fishbrain-inputs-v1).** The input specification hashes what the engine actually uses.
For every source, in order, it hashes: name, kind, start/stop step, targets, and then the
per-step probability and weight (scalar Poisson), the rates, frame length and weight (per-frame
Poisson), or the per-step drive (current). Python `snapshot.encode_inputs` and JS
`encodeInputs` write identical bytes. The byte layout is in the `fishbrain/snapshot.py` docstring.

**Checkpoints (fishbrain-checkpoint-v1).** `snapshot.serialize(sim)` stores the state at a step:
u, h, last spike, the delay slots, the refractory queue, and each source's PCG64 state and
pending Poisson events. It also stores the network, params and inputs digests and the seed
string, the inputs themselves (optional), the state digest, and a checksum.
`snapshot.deserialize(bytes, network, params)` in Python and
`restoreCheckpoint(bytes, {net, params, inputs, gated})` in JS resume it. Python and JS write
byte-identical checkpoints for the same state. A JS resume of a Python checkpoint reaches
Python's spike hash and state digest (all 5 synthetic cases, full and gated, and the real gate
window). Every byte is either covered by `state_digest()` or checked against something that is:
- The refractory queue must match the last-spike steps.
- A scalar source's last position must match its pending events.
- Each delay slot must hold every neuron whose last spike was that slot's step.
- The recorded digest must match the parsed state.
- No trailing bytes are allowed.

A forger who reseals every hash still gets a state whose digest differs from `window.snapshot`.
That stops edits the receipt did not follow. It does not stop a forger who also rewrites the
receipt. The anchor does.

**The checkpoint trust chain (enforced by `anchor`).** A self-consistent checkpoint proves nothing
about the run. u and h are free fields, so a fabricated state with every hash resealed parses, and
its streams are honest. So `verify()` refuses any window past step 0 unless its start is anchored,
in one of three ways:
- **Genesis.** Checkpoint 0 is not a file. It is the state built from the wiring, parameters,
  inputs and seed, and its digest is the genesis snapshot. A step-0 window is anchored by building it.
- **A chain of links.** `opts.chain`, or the fetcher's `"chain"` (only asked when no trusted entry
  matches and `opts.chain` is absent), is a list of links `{from: {step, snapshot}, to: {step,
  snapshot}}`. It is also accepted as `{links: [...]}`. The links must be contiguous and start at
  step 0 or at a trusted checkpoint. They must end at exactly `(start_step, window.snapshot)`. Any
  `kernel`/`seed`/`brain`/`inputs` a link carries must be the receipt's. Verify replays every link.
  Each link's start state gets the same checks as the window's (genesis, or a content-addressed
  checkpoint with its snapshot, binding and streams checked). Verify runs it and requires the
  `to.snapshot` digest. A failure names the link and its inner check. `r.anchor` is
  `{by: "chain", root: "genesis" | "trusted", links}`.
- **A trusted checkpoint (the fast path).** `opts.trustedCheckpoints` lists window starts the page
  has itself verified. For example, it found a `fishbrain:ckpt:v1` memo from the run's account that
  predates the decision. Each entry is the full tuple `{step, snapshot, network, params, inputs, seed}`
  (seed = the blockhash value). Every field must equal the receipt's. **A bare digest is refused.**
  `state_digest()` covers u, h, spikes, slots, source names and stream positions, but not rates,
  weights, targets or start/stop. A bare digest would therefore vouch for the honest state at step S
  under inputs forged after S. The list is validated on every call, so a malformed entry fails at
  `anchor` even on a step-0 receipt.

`verifyLink({kernel, seed, brain: {network, params}, inputs, from: {step, snapshot}, to: {step,
snapshot}}, fetcher, opts)` checks one link. It fails at `state` when `to` is wrong. A link is a
**conditional** check: "from `from`, the brain reaches `to`". So different people can check
different links in parallel. Its `ok` does not anchor `from`. `r.anchor` is `{by: "genesis"}` when
`from.step` is 0, `{by: "trusted" | "chain"}` when `opts.trustedCheckpoints` or `opts.chain` anchor
it, and `null` otherwise, with `anchor` then absent from `passed`. A link never fetches a chain.

The tests anchor the synthetic window at step 1,100 through each path: one link from genesis, two
links through a content-addressed step-600 checkpoint, a chain from a trusted step-600 entry, and a
chain served by the fetcher. They anchor the real gate window at step 20,000 through a trusted entry
and through a full 20,000-step link from genesis, in the same call. They refuse, at `anchor`:
- an unanchored honest checkpoint;
- a bare digest;
- a trusted entry naming other inputs;
- a chain from an untrusted checkpoint;
- a broken chain;
- a link for another seed;
- the fabricated state, with no anchor, with the honest chain, and with a chain that claims the
  forged digest.

When the page trusts the forged tuple, the fabricated state passes. That control proves the anchor
is the check that stops it.

**Commit on-chain (proposal).** Commitments stop fabrication only if each one was published before
anyone knew the decision it would produce. One Solana Memo per commitment, from the run's account:
1. **Run start:** `fishbrain:run:v1 network=<hex> params=<hex> kernel=<hex> seed_slot=<n>`
   (`snapshot.run_commit_memo`). The seed is the blockhash of `seed_slot`, a slot after this memo
   lands, so the brain is fixed before the noise exists (commit, then reveal). Verify checks this
   with `chainLookup` (above).
2. **Checkpoints:** `fishbrain:ckpt:v1 <run> <step> <state digest> <inputs digest>` at a fixed
   cadence of simulated time, and at every decision window's `start_step`. A page that finds the memo
   for `window.snapshot`, predating the decision, passes that tuple in
   `trustedCheckpoints`. Otherwise it serves the chain of links.

**Receipt fields this needs (proposed):**
- `brain.bridge` (new, **required**): the sha256 of the network's fishbrain-bridge-table-v1, or `null` for an unbridged
  network. Absent fails at `receipt`, **before** `kernel`, so a simulated receipt without it now stops at `receipt`
  instead of `kernel`. Simulated receipts should carry `"bridge": null`.
- `inputs.spec` (new, required): sha256 of fishbrain-inputs-v1. `inputs.frames` stays optional,
  as the display hash of the retina rates. It equals the G1 gate's recorded `rates_sha256`.
- `seed.value`: a canonical, non-degenerate base58 32-byte blockhash, fed through the seed rule above.
- `seed.commit` (new, optional): the run-commit Memo's transaction signature, for the seed anchor.
- `window.snapshot`: `state_digest()` at `start_step`. At step 0 this is the genesis state. The
  checkpoint is served content-addressed by this digest, and so is its chain.
- `kernel.version`: `"fishbrain-js-kernel-v3"`. `kernel.sha256`: sha256 of the kernel file.
- Optional `window.snapshot_memo`: the signature of the memo that committed `window.snapshot`.
  Verify does not read it. The page checks it and passes the tuple in `trustedCheckpoints`.

`snapshot.window_receipt(sim, n_steps, blockhash, commit=...)` is the reference receipt writer. It
returns the fields above, the checkpoint bytes and the raster.

**Measured (Node v24.21.0, loaded machine, 2026-09-26).** A real gate window from its step-20,000
checkpoint, anchored by a trusted entry, verified end to end in about 0.48 s. That covers 3,000
steps, including fetching and parsing the 16 MB case and building the brain. The same window
anchored by a chain (a full 20,000-step replay from genesis, then the window) took about 1.1 s. A
synthetic window took 20 to 80 ms with any anchor. An unanchored or fabricated gate checkpoint is
refused in about 0.27 s. Chrome 152 (served from localhost) earlier
measured this verifier at 0.79 s warm and 1.9 s cold for the gate window, including the 16 MB fetch.
That was before the anchor check. This version ran in headless Playwright Chromium 148 from
localhost, on a synthetic window with a negative-start source. It passed through a trusted entry in
48 ms and through a chain the fetcher served as a `Response` in 54 ms. It refused the unanchored
honest checkpoint, the fabricated state, and a chain claiming the forged digest, all at `anchor`.
The gate window was not re-run in a browser.

**Limits.**
- The trusted kernel list is the root of trust. A viewer who doubts the page can run
  `verify.mjs` from the repo in Node with the same receipt.
- **Anchoring the start state does not anchor the inputs.** A receipt for other inputs, recomputed
  honestly, passes: the retina's weight x4, or its targets moved, with `inputs.spec` recomputed and
  `inputs.frames` unchanged. It is an honest receipt of a different brain input. A chain from
  genesis under those inputs holds too, when they differ only after the window start. The fix is a
  commitment the page checks: the run-commit (or each ckpt memo) naming the inputs digest or an
  inputs policy, compared against `inputs.spec`. Not built. The tests keep these cases as passing
  controls and assert their specs differ from the honest one.
- Verify reads only the fields above. Anything else a receipt carries is for display.
- A case carries whole-run input arrays. The gate retina is about 11 MB of float64, so a checkpoint
  can embed its inputs (the default) or leave them to the case (`embed_inputs=False`). Live inputs
  grow without bound, so production needs input sources re-based at each checkpoint. That is not
  built.
- Fast-forwarding a scalar Poisson source costs work in proportion to its events since step 0
  (per-frame sources jump in O(log n)). A chain from genesis replays every step. Hours of run would
  make both slow. Re-basing sources at checkpoints, and trusted ckpt memos, fix that.

Tests: `.venv/bin/python -m pytest -q tests/test_verify.py` (163 passed). Honest receipts pass
through every anchor path. Each forgery fails at its named check, including every forgery in the
adversary's 2026-09-26 harness:
- the fabricated Mauthner state, synthetic and on the real gate;
- the negative-start stream move;
- the all-zero seed;
- the spikes-flipped control.

Its honest-other-inputs cases are kept as documented limits. Twenty-three mutants of `verify.mjs` each
reopen one hole, and each must flip the scenario that guards it. They include: executing the
fetcher's buffer, importing a static kernel, skipping one check, skipping the anchor, not replaying
a chain link, not checking where a chain ends or starts, a trusted entry that ignores the inputs, the
degenerate-seed rule, and the seed anchor's slot order and blockhash comparison. A v2-count kernel
copy must reopen hole A. Five of the mutants are for the bridge check:
- not comparing the table's hash (a relabelled table passes);
- not checking per-pair totals (a dropped link, another gain or a stale build passes when resealed);
- not checking the pairs (a bridge part named on a pair the network lacks passes);
- ignoring `trustedBrains`;
- dropping the canonical-null rule.

Each must still pass the honest bridged receipts. `tests/test_g1c.py` runs the same check on the real
bridged gate fish: the table's digests equal the gate's recorded ones, and hiding the escape crossing
(every E1 link's bridged count dropped, the hash recomputed) fails at `bridge`.

## Speed (measured 2026-09-26, Apple M5, 10 cores, 16 GB)

This is the real gate case: 8,654 neurons, 38,304 connections (CSR entries), 35,000 steps (3.5 s simulated) in
7 windows, with a digest after each. RTF = simulated ms / wall ms, so above 1 means faster than real
time. Each figure is a single run with other sessions loading the machine (load average about 4 to
7).

| where | case | full | gated |
|---|---|---|---|
| Node v24.21.0 | real gate | 2.13 s (RTF 1.65) | 1.03 s (RTF 3.39) |
| Node v24.21.0 | real anchored-w | 1.56 s (RTF 2.24) | 1.07 s (RTF 3.28) |
| Chrome 152, hidden tab | real gate | 2.34 s (RTF 1.50) | 1.15 s (RTF 3.05) |

In the browser, the 16 MB case took 50 to 80 ms to fetch and parse, and 155 to 300 ms to build. An
earlier coordinator run (03:31) measured RTF 0.78 in Node under unrecorded load. Treat the figures
as load-dependent. The synthetic cases run in 20 to 300 ms each.

## Tests

```sh
cd launches/fishbrain/brain && .venv/bin/python -m pytest -q tests/test_js_parity.py   # 52 passed, ~19 s
```

- **Parity:** the 5 synthetic cases (exported fresh into a tmp dir by `export_case.py --out`) and the
  2 real cases in `data/js/` (git-ignored), each full and gated. Both `run_case.mjs` and Python
  compare every hash. A real case is also re-derived from the current `sim.py`, so a stale file
  fails. Its raster hash has to equal the G1 trial recorded in `data/net/`. The only skip is the
  real cases, when `data/js/` holds none. A missing Node is a failure.
- **Controls:**
  - 9 kernel mutants run on every case in both modes: integrate order, a `Math.fround` removed,
    delay off by one, refractory off by one, one PCG64 multiplier limb, the 128-bit multiplier,
    reversed delivery order, a 4095-draw Poisson batch, and a gating-only change. Each must fail
    wherever its path runs. Where a case cannot see a mutant, the test asserts that case passes and
    records the reason (for example, s4 has zero delay, so a delay off by one names the same slot).
  - An unmodified kernel copy passes.
  - A weight bit flip (bits 22 and 0) must change the state digest.
  - A synapse count changed by one must change the network digest.
- **Units:** SeedSequence/PCG64 against numpy (including the fast `fillRaw`/`drawSelected` paths),
  60 parameter sets of derived constants and params digests, Python float `repr`, step rounding,
  and frame hashing.

## Limits

- **Snapshot restore** now exists (fishbrain-checkpoint-v1, `restoreCheckpoint`); see Verify above for
  what a checkpoint proves and the chain that backs it.
- Speed is measured on the 8,654-neuron gate network only. The full brain is unmeasured in a browser.
- A one-ULP weight change can wash out by a run boundary: the target's h decays or resets. The
  weight check uses a case where it survives. In Verify, weights are fixed by the hashed counts and
  params, so this is not a gap there.
