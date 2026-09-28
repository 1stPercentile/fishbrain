---
publication_copy: edited-for-publication copy of launches/fishbrain/evidence/G1c-gate.md at commit 7fcdbd4e2. Local paths made repo-relative; machine-owner, internal coordination, account and credential details removed. No number, hash, time, method, result or correction was changed; publication notes are marked [Publication note].
title: G1c gate, the bridged fish (built from the frozen spec, gains tuned on training stimuli only, gated on held-out stimuli)
created: 2026-09-26
status: done (task B). Verdict under the frozen spec's rule = KILL (the receding control C1 fails). Every mandatory publication is below.
supersedes: the 09:42 EDT automatic snapshot commit a525a05fc of this file, a draft written while runs were in progress (it held placeholder M2 rows and conclusions written ahead of their runs). This version, committed in e2881d4be after every run finished, is the record. [Publication note: git history shows this note's record text was first committed in the automatic snapshot c9f0a536c at 2026-09-26 10:12:20 -0400, after the last run (the pytest run of record ended 10:01:39 EDT); e2881d4be (10:19:35 -0400) committed the gate data under evidence/g1c-data/; 7fcdbd4e2 (10:20:58 -0400) added this supersedes line. The a525a05fc draft differs from the record by 105 insertions and 26 deletions.]
evidence_status: measured unless marked "expected" or "choice". Every number here comes from a run on the analysis machine on 2026-09-26 between 08:57 and 10:02 EDT (times per section).
verification_scope: >
  Ran 2026-09-26 08:57-10:02 EDT on the analysis machine (10 cores, shared, load average 6-11). Inputs: the frozen spec
  brain/data/g1c/bridge_spec.json (sha256 185b3780...5e111499, checked before every step), the brain of record
  (data/net/pairs_*.npy, wiring hash 09a71038...25af2), cells_identified.json, readouts.json, g1c/oa_fragments.json,
  the cached CAVE v709 somas, mece2 at 4096 nm, and one public network read: precomputed://gs://fish1-public/mece3_231218
  at mip [4096,4096,3840] (no login, re-fetched 08:57 EDT, sha256 of the array 795292769933...a48e49, equal to the spec's).
  Engine: sim.py (unmodified), provenance.py (unmodified), js/fishbrain-sim.mjs (node v24.21.0). No literature was re-read;
  every citation goes through evidence/G1c-bridge-spec.md.
files: G1c task B. brain/fishbrain/bridge.py, brain/tests/test_g1c.py, brain/data/g1c/gate/ (git-ignored), this note.
reproduce: >
  cd launches/fishbrain/brain && PYTHONPATH=. .venv/bin/python -m fishbrain.bridge <rederive|topology|tune|gate|ablations|shuffles|js|report>
  (about 60 min at 3-4 workers here; `tune` refuses to run over an existing tuning log), then .venv/bin/python -m pytest tests/test_g1c.py
---
# G1c gate: the bridged fish

**TLDR.** With the bridge, the fish flees every held-out loom to the correct side and strikes and turns toward every
held-out prey. But it also flees a *receding* shadow, harder than a loom, so the pre-registered controls do not hold and
the frozen rule says **KILL**. **The real share: 0.09% of the drive that made the escape decisions, and 0.010% of the
drive that made the strike and turn decisions, travelled over measured Fish1 synapses.** The rest ran through the
published pathway model (class c). M1 and M2 show the behaviour does not depend on the measured wiring (s.7, s.8).

1. **Built exactly to the spec (s.1).** All 76 lists re-derived independently equal the frozen JSON. The bridge adds
   1,780 edges and 0 cells, with 0 collisions with the 24,921,936 measured pairs.
2. **Tuned on training stimuli only (s.2).** Chosen: g_a = 1, g_c = 4, which passes 10 of 10 training checks. g_a
   changes nothing anywhere on the grid.
3. **Gate on held-out stimuli (s.3).**
   - Passed: H1 (loom escape, correct side), H2 (strike, turn toward prey), C2 (dimming; the eye passes it, not the
     brain), C3 (blank; passes by construction), L1 and L2 (lesions).
   - **Failed: C1.** The receding disc fires the M-system in 8 of 8 trials per side, with 92.5-92.75 Mauthner spikes
     against the loom's 63.6-65.1.
4. **The headline decision-path share (s.5):**
   - Loom escape (the initiating M-system cells): **0.090%** (m = 0.000895).
   - Prey strike + turn (the cells that spiked): **0.010%** (m = 0.000101).
   - All 128 held-out trials pooled: **0.011%**.
   - The bridge carries the rest: class (c) 99.89% of the escape drive and 99.99% of the prey drive; class (a) 0.022%
     and 0.00001%.
5. **Synapse-count share (s.6):**
   - 99.90% of all synapses in the brain of record plus the bridge are measured.
   - In the network the gate simulates it is 63.3%, and 13.5% counting only synapses onto the deciding cells.
6. **The pre-registered disclosure sentence applies (s.9):** *"Which way the fish flees is set by the published pathway
   model, not by the measured wiring."*
7. **Determinism and parity.**
   - All 128 held-out trials were run twice, and the hashes are identical.
   - The JS kernel reproduces a bridged trial bit for bit (exit 0, no mismatches).
   - `pytest tests/test_g1c.py` (bare): 21 passed, exit 0 (s.10).

## 1. What was built (measured 08:57-08:58 EDT)

- **Spec checks.**
  - `bridge_spec.json` sha256 = `185b3780...5e111499`, equal to the frozen value.
  - The edge list re-hashed from the JSON's own edges gives `f7f810d5...c8100bf0`, equal to the spec's.
  - `load_spec()` checks both before every step; a one-byte change stops it (tested).
- **Independent re-derivation (`bridge.rederive`, 08:57:03-08:57:13 EDT): 76 of 76 checks equal the JSON.** It was
  written from the spec's rule text, not from task S's script, using the plain metrics the text names (Euclidean in
  (u, v) for E1, Euclidean in (theta, elev) for P1, 3-D um for the class-(a) columns). What it rebuilt:
  - **Stimulated set:** 24,920 cells (11,799 left, 13,121 right), sha256 `8910b216...bc2348`, and the 7 re-admitted cells.
  - **E1 grid:** 36 + 36 sources.
  - **P1 grid:** 299 points giving 292 + 293 sources.
  - **Targets:** the M-system, the nIII dorsal lateral half (58 + 48 in graph), the members not in graph
    (7 + 5 lateral; 13 + 12 dorsal), nMLF (15 + 18) and turning SPNs (32 + 23).
  - **APN relays:** left seg 14241661373 (lore 125577, 0.55 um from AF7), right 14833177305 (lore 170137, 0.81 um).
    Also 207 / 179 AF7 voxels and 3,973 / 3,527 pretectal somas per side.
  - **Class (a):** the 19 eligible fragments, their 34-soma columns and column radii.
  - **Edges:** all 1,780. Also checked: 0 duplicate pairs, 0 self-loops, every endpoint in the graph, and **0 collisions**
    with all 24,921,936 measured pairs.
  - **mece3 re-fetched** from the public bucket. Array sha256 `79529276...a48e49` equals the spec's. Cached at
    `data/g1c/gate/ref/`.
  - A negative control, one E1 source dropped from a copy of the spec, stops the build (tested).
- **The bridged network (`bridge.build_topology`, 08:57:27 EDT, 5.4 s).**
  - The measured pairs keep Shiu's weights and the per-synapse sign rule, exactly as G1. The bridge adds 1,780
    (pre, post, base) pairs, all excitatory.
  - The whole bridged brain was reduced exactly as G1 did (`brain.reduce_graph`: cells that can fire and can reach a
    deciding readout). The deciding readouts are the 6 M-system cells, the nIII_dorsal, nMLF and turning pools.
  - Result:
    - 128,683 cells can fire;
    - **10,528 neurons kept**;
    - **42,342 pairs**: 40,562 measured + 1,780 bridge; every bridge edge was kept;
    - 3,518 eye-driven tectal cells kept.
  - Gains only scale counts, so one node set serves every gain and every ablation.
  - **Exactness is tested:** on 0.5 s of a held-out prey at the chosen gains, every kept cell's spike train equals its
    train on the whole fireable set (128,683 cells), with the other eye targets driven at 100 Hz
    (`test_reduction_is_exact`).
- **Digests.**
  - `Network.digest()` hashes arrays only: `0b6f3369...0aa083b4a` at the chosen gains. G1's gate network has a
    different one.
  - The **bridged digest** = sha256 over {Network.digest(), `g1c-bridge-v1`, the edge-list hash, g_a, g_c, the
    ablation, the topology tag, the edge-provenance digest} = `279ae5fa...dbb5b36a`. It changes with the version
    string, the gains and any ablation (tested).
  - Edge-provenance digest: `662772ed...1b237b12`.
- **The eye.** `brain.Retina` with `RetinaParams()` unchanged. Only its W and rate caches were redirected to
  `data/g1c/gate/` so nothing was written into `data/net/`.
  - W for the 3,518 kept targets built in 9.9 s.
  - The provenance root is the G1c stimulated set (24,920 ids), passed explicitly on every trace and pinned in its
    digests (`root_allowed` = the hash of the 3,518 kept targets).

## 2. Tuning: the two class gains, training stimuli only (09:01:06-09:12:26 EDT)

- **Frozen:** stimuli, seeds (`G1c-tune-0..7`) and objective, all from the spec. My operationalisation choices were
  written to `data/g1c/gate/choices.json` at **09:00:40 EDT**, before the grid.
- **Every job and every grid point is in `data/g1c/gate/tuning_log.jsonl`** with its time:
  - 144 job entries, each with 8 spike hashes, the bridged digest and the stimulus parameter hash;
  - 36 grid-point entries with all 10 checks and the cross-talk;
  - the start entry and the "chosen" entry.
- **One smoke timing run came before the grid** (08:58-08:59 EDT): g = (8, 8), seed G1c-tune-0, one trial per training
  condition. It is logged as not entering the objective; the grid re-ran (8, 8) with all 8 seeds.
- **Result (measured):**

| g_c (any g_a) | checks passed | cross-talk | what fails |
|---|---|---|---|
| 1 | 6 / 10 | 0 | no strike and no turn toward the prey on either side |
| 2 | 8 / 10 | 16 | no strike on prey (turn toward the prey passes) |
| **4** | **10 / 10** | 16 | nothing (all 16 training looms also strike) |
| 8 | 8 / 10 | 30 | escapes start below 12 deg (angle fraction 0.25); 14 prey trials escape |
| 16, 32 | 8 / 10 | 32 | angle fraction 0; every prey trial escapes |

- **g_a is behaviourally inert:** every one of the 6 g_a values gives identical checks and cross-talk at each g_c.
- **Chosen by the objective** (most checks, then least cross-talk, then smallest g_c, then smallest g_a):
  **g_a = 1, g_c = 4**, logged at 09:12:26 EDT.
- **The one allowed global change was not used** (see s.12).

## 3. The gate on held-out stimuli (09:15:29-09:26:21 EDT, 769 s, 4 workers)

- **Held-out hash check: PASS.**
  - The tuning log holds exactly the 4 training stimulus hashes.
  - None of the 8 held-out looms, the 8 held-out prey or the 4 controls is in it (`held_out_hash_check`, also a test).
- **Seeds:** `G1-seed-0..7` in every condition, paired.
- **Scoring:** H1 and H2 are pooled per stimulus side over 4 conditions (32 trials).

| item | result | numbers (measured) |
|---|---|---|
| **H1** loom escape | **PASS** | both sides: P(escape) 1.00 (32 / 32); first M-system cell on the stimulus side 1.00; disc at first spike ≥ 12 deg 1.00 (16.5-31.0 deg); 0 escapes after expansion end + 100 ms |
| **H2** prey | **PASS** | both sides: strike rule 1.00 (32 / 32); turn index toward the prey 1.00 (32 / 32); nMLF side index reported, 0.000-0.003 (direction-insensitive, as Lau 2025 expects) |
| **C1** receding | **FAIL** | left: 92.5 Mauthner spikes per trial vs the l/v 240 loom's 65.1 (ratio 1.42, needs ≤ 0.5), P(escape) 1.00 vs 1.00 (needs lower). right: 92.75 vs 63.6 (1.46), 1.00 vs 1.00 |
| **C2** dimming | PASS, *passed by the eye, not by the brain* | the eye model gives 0 tectal input (expected tectal input spikes 0), so 0 spikes anywhere |
| **C3** blank | PASS, *passed by construction* | 0 spikes in all 8 trials (no background input) |
| **L1** bilateral M-system lesion | PASS, *by construction, and non-vacuous* | the intact fish escaped in 64 / 64 paired trials. The lesioned fish shows 0 M-system spikes within ±20 ms of the intact median escape time (0 short-latency escapes). Informative row: in 32 of those 64 trials another deciding readout (turning, nMLF or nIII) spiked inside that ±20 ms |
| **L2** unilateral lesions | PASS | left or right task_criteria M-system lesioned: 0 escapes initiated on the lesioned side. Looms on the other side keep P(escape) 1.00 and side 1.00 (32 / 32 each) |

**Verdict under the frozen rule: KILL.**
- The rule is KILL = not (H1 and H2 and C1 and C2 and C3). C1 fails, so "the controls holding" is false.
- H1, H2, L1 and L2 pass. The page may not say the gate passed.

**Per condition (measured, 8 seeds each):**

| condition | P(escape) | side | angle at first spike (deg) | first spike (ms) | Mauthner spikes |
|---|---|---|---|---|---|
| loom +45 l/v 120 | 1.00 | 1.00 | 18.1-23.9 | 3,182-3,369 | 46.1 |
| loom −45 l/v 120 | 1.00 | 1.00 | 16.5-22.9 | 3,107-3,344 | 47.0 |
| loom +135 l/v 120 | 1.00 | 1.00 | 19.2-26.7 | 3,228-3,431 | 27.4 |
| loom −135 l/v 120 | 1.00 | 1.00 | 18.7-25.4 | 3,207-3,403 | 26.4 |
| loom +45 l/v 480 | 1.00 | 1.00 | 23.4-27.7 | 11,931-12,298 | 120.5 |
| loom −45 l/v 480 | 1.00 | 1.00 | 24.6-27.7 | 12,046-12,297 | 122.8 |
| loom +135 l/v 480 | 1.00 | 1.00 | 26.2-31.0 | 12,183-12,516 | 73.3 |
| loom −135 l/v 480 | 1.00 | 1.00 | 25.2-31.0 | 12,102-12,515 | 70.0 |

- Every prey condition (±15 and ±45 deg, at 15 and 60 deg/s) has strike 1.00 and turn toward 1.00, with 0 escapes.
- The turn index is weakest at ±15 deg: minimum |TI| 0.58-0.68. The turning cells on the far side fire 27.9-54.1
  spikes per cell against 141-203 on the prey's side. Our reading: the binocular zone drives both lobes. At ±45 deg the
  index is 1.00.

**L2 detail (informative, not scored).**
- For looms on the lesioned side, 16 of 32 still escape: all 16 at ±45 deg and none at ±135 deg.
- All 16 are initiated by the surviving, wrong-side M-system. Our reading: the frontal disc reaches the other eye
  late, through the binocular zone.
- The union lesion sets are sensitivity rows only. The union sets name 9 cells, and 2 of them (16039139313, 20690356355)
  are not in the reduced network. They cannot fire or cannot reach a deciding readout, so lesioning them is a no-op,
  and that is exact.
  The union rows give the same escapes as the task_criteria rows.

## 4. Why C1 fails (measured, then what it means)

**Measured.**
- **The eye model answers a receding disc more strongly than a loom.** Expected tectal input spikes in the scoring
  window, left-side stimuli:
  - receding (motion window): 296,587;
  - loom (l/v 240): 226,960.
- Over the whole trial the receding disc gives 388,153 (left) / 354,390 (right), against the loom's 294,227 / 258,669.
- **The receding escapes come 17-21 ms after the disc starts to shrink** (first M-system spike at 1,517.0-1,521.0 ms,
  16 of 16 trials).
  That is where the shrinking edge moves fastest: the reversed loom schedule changes angle fastest near 140 deg.
- The static onset window (the 140 deg disc appearing; reported, not scored) also fires the M-system in 8 / 8 trials
  per side.
- G1's unbridged brain already had this tectal drive: 227,681 / 229,400 tectal spikes per receding trial
  (G1-gate s.5.1). Nothing downstream relayed it then.

**What it means.**
- `Retina` is an OFF-transient centre-surround front end. A shrinking dark disc brightens its rim, and the surround
  turns that into OFF responses just inside the edge. That is our reading of the code, not a separate measurement.
- E1 sums tectal activity over a sparse whole-lobe grid. It has no direction or expansion selectivity.
- So the fish flees anything that changes a large dark area: a property of the chosen eye plus a non-selective bridge,
  not of the measured wiring.
- In the literature, loom selectivity sits in retinal and tectal computations we do not model (Temizer 2015, via the
  spec). The spec lists the eye's missing feature channels as a limit (spec s.15).

## 5. Headline: the decision-path share (M3, intact; `provenance.py`, every held-out trial traced)

**How it was measured.**
- The brief's metric, as `provenance.py` implements it, on all 128 held-out trials.
- Drive in each trial's decision window, pooled over trials as Σ measured drive / Σ link drive.
- Deciding cells (spec s.9):
  - loom: the initiating M-system cell(s);
  - prey: every nIII_dorsal and nMLF cell and every turning SPN that spiked.
- The eye is the root, so a stimulated tectal cell's spikes pass on m = 1.
- All 128 traces passed provenance's self-checks: the replay mirrors every spike, and the parts sum to the total.

| decision | m (measured share) | through class (c) | through class (a) | trials |
|---|---|---|---|---|
| **loom escape** (initiating M-system cells) | **0.000895 (0.090%)** | 0.99889 | 0.00022 | 64 |
| **prey strike + turn** (cells that spiked) | **0.000101 (0.010%)** | 0.99990 | 0.0000001 | 64 |
| all 128 held-out trials | 0.000111 (0.011%) | 0.99989 | 0.0000029 | 128 |
| by group: escape / strike / turn | 0.090% / 0.018% / 0.0058% | | | |

**Per condition.**
- Looms at ±45 deg: 0.15-0.20%.
- **Looms at ±135 deg: exactly 0.**
- Prey: 0.0001-0.03%.

**Where the measured sliver comes from (measured).**
- The escaping cells' measured eye-driven inputs are the direct contacts of re-admitted O-a0 tectal cells:
  - 17057069410 → Mauthner-L, 2 synapses;
  - 16670127295 → Mauthner-R, 2 synapses;
  - 13181211444 → MiD3cm-R, 1 synapse.
- Longer measured chains may add the rest of the measured drive; they were not decomposed.
- **Every escape's first spike carries 100% class (c)** (E1): 192 of 192 initiating-cell first spikes over the 64
  held-out loom trials (all three M-system cells of a side fire on the same step), each with m = 0.
- Population-level shares are a secondary row, pooled over each whole population's drive:
  - Mauthner 0.63-0.67%, MiD3cm-R 0.85%, MiD2cm 0;
  - nIII_dorsal 0.016-0.021%, nMLF 0.007-0.056%, turning 0-0.013%.
- As the spec expected, the decision runs through the bridge.

## 6. Secondary: the synapse-count share (never the headline)

The metric is reported under two conventions and in two networks, at g_a = 1, g_c = 4:

| network | convention | measured | bridge (a) | bridge (c) | measured share |
|---|---|---|---|---|---|
| whole brain of record + bridge | raw synapses | 29,474,316 | 646 | 29,588 | **99.90%** |
| whole brain of record + bridge | abs(netted signed count) | 27,167,678 | 646 | 29,588 | 99.89% |
| the gate's reduced network (10,528 cells) | abs(netted signed count), provenance.synapse_count_share | 52,052 | 646 | 29,588 | **63.26%** |
| same, only pairs onto the deciding cells | same | 4,259 | 0 | 27,248 | 13.52% |

- The bridge's synapse-equivalents are 646 × g_a + 7,397 × g_c = 30,234.
- Per class in the reduced network: (a) 0.79%, (c) 35.96%.

## 7. M1: the measured wiring shuffled, the bridge fixed

Run 09:26:57-09:51:52 EDT, 3 workers, concurrently with M2.
- **What was shuffled:** the measured pairs only, degree-preserving (`sim.shuffled_copy`, as G1), over the whole brain of
  record (24.19 M nonzero pairs). Seeds `G1c-measured-shuffle-0..9`.
- **What was fixed:** the 1,780 bridge edges, re-added by segment id at g_a = 1, g_c = 4, and the stimulated set.
- Each shuffle was re-reduced exactly (6,909-7,430 neurons; 2,911-3,012 eye targets kept).
- **Scoring:** all 16 H1/H2 held-out conditions with gate seeds 0-3 (64 trials per shuffle). The comparison rule was
  fixed at 09:00:40 EDT: the two booleans (H1, H2) against the intact fish on the same seeds.

| shuffle | H1 | H2 | escape P / side / angle (both sides) | strike / turn toward (both sides) | equal to intact |
|---|---|---|---|---|---|
| 0-9 (each of the 10) | PASS | PASS | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 | yes |

- **10 of 10 shuffles are equal to the intact fish,** like for like (seeds 0-3; intact H1 and H2 also PASS on them).
  Equal against the intact 8-seed verdict too, and all four per-side booleans are equal.
- Every shuffle also strikes on all 32 of its loom trials and escapes on 0 prey trials, as intact does.
- Three shuffles put one measured pair on a bridge pair (listed in `m1_shuffles.json`). The Network sums them and
  provenance keeps the parts apart.
- **The pre-registered rule is met (≥ 8 of 10), so the page says: *"the anatomy is real; the decision runs through the
  model"*.** Per-shuffle metrics and hashes are in `data/g1c/gate/m1_shuffles.json`.

## 8. M2: bridge ablations, with M3's metrics per class

Run 09:26:57-09:54:17 EDT, 3 workers, concurrently with M1.
- Each ablation removes bridge edges from the intact bridged network. The node set is unchanged, and the eye's targets
  and Poisson draws are paired with the gate.
- H1/H2 held-out conditions, 8 gate seeds, every trial traced, so both metrics come per class. All provenance
  self-checks passed.

| ablation | H1 | H2 | escapes / strikes on held-out looms | strikes / turns toward on held-out prey | decision-path share, loom (escape) | decision-path share, prey | synapse-count share (reduced net) |
|---|---|---|---|---|---|---|---|
| intact (g_a 1, g_c 4) | PASS | PASS | 64 / 64 | 64 / 64 | 0.090% (c 99.89%, a 0.022%) | 0.010% (c 99.99%) | 63.3% (a 0.8%, c 36.0%) |
| (i) all bridge removed | FAIL | FAIL | 0 / 0 | 0 / 0 | undefined (no drive) | undefined | 100% |
| (ii) class (a) removed | PASS | PASS | 64 / 64 | 64 / 64 | 0.090% (c 99.91%) | 0.010% | 63.8% |
| (iii) class (c) removed | FAIL | FAIL | 0 / 0 | 0 / 0 | undefined | undefined | 98.8% (a 1.2%) |
| (iv) E1 removed | **FAIL** | PASS | **0** / 64 | 64 / 64 | undefined (no M-system spike) | 0.010% | 63.9% |
| (v) P1-P4 removed | PASS | **FAIL** | 64 / 0 | **0 / 0** | 0.090% | undefined | 97.2% |

**What M2 shows (measured).**
- **The measured wiring alone does nothing.** With the bridge removed, not one deciding cell spikes in 128 trials. The
  deciding populations do receive measured drive (population m = 1.0), but it stays below threshold.
- **Class (a) is inert.** Removing it changes no outcome, and the escape share only moves from 0.089506% to 0.089526%.
  This matches the tuning: g_a changed nothing at any grid point.
- **Class (c) is the whole behaviour.**
  - Without E1, no held-out loom fires the M-system at all (0 of 64).
  - Without P1-P4, no prey strikes or turns (0 of 64), and the loom-evoked strikes disappear.
- So the bridge sets both *whether* and *which way* the fish flees and strikes. The measured tectum supplies the eye's
  map, which the bridge reads out.

## 9. M4: cross-talk, the tuning log, the hash check, the disclosure

**Cross-talk (reported, not scored).**
- **Every loom trial also fires the strike rule:** 64 / 64 held-out and 16 / 16 baseline trials. So do 16 / 16
  receding trials.
- No prey trial escapes (0 / 64). Dimming and blank: 0.
- **This matters wherever a strike is read as a decision:** every shadow also triggers a strike.
- The strike pools light identically in every stimulated trial: dorsal nIII 58 / 130 and 48 / 106, nMLF 15 / 15 and
  18 / 18. That is exactly P2 and P3's bridged targets. Either APN relay firing is enough, and a large disc covers the
  frontal P1 zone.

**Tuning log and hash check.** See s.2 and s.3; both PASS.

**Disclosure decision.** The pre-registered sentence **applies**: *"Which way the fish flees is set by the published
pathway model, not by the measured wiring."*
- E1, the crossing, is built and kept in the gate network.
- M2 (s.8) shows what happens without it.
- The parallel strike sentence is proposed, **not pre-registered**, and the page owner decides: *"Whether and which
  way the fish strikes is set by the published pathway model (Antinucci 2019), not by the measured wiring."* M2 row (v)
  is its evidence.

## 10. Determinism, JS parity, tests

- **Determinism: PASS.** Each of the 128 held-out trials ran twice in separate worker processes: once on the plain
  Simulator, once in provenance's traced replay. All 128 spike hashes are identical.
- **JS parity (09:30:20 EDT): exit 0, mismatches [].**
  - Case: `g1c-bridged-prey_az+15_v60-G1-seed-0.json` (27.3 MB; 10,528 neurons, 42,342 pairs; the bridge is plain extra
    pairs). Run with `node js/run_case.mjs`, node v24.21.0.
  - The JS network digest equals Python's (`0b6f3369...`).
  - The JS raster hash `f1729839...dbf09be2` equals Python's and the gate's recorded trial.
  - ActiveSimulator is identical.
  - The bridged digest recomputed from the JS network digest equals Python's.
  - Speed: real-time factor 0.85 (loaded machine).
- **`pytest tests/test_g1c.py`, run bare: 21 passed in 65.92 s, exit code 0** (10:00:33-10:01:39 EDT; log at
  `data/g1c/gate/pytest_g1c.log`). An earlier run of the first 20 tests, 09:54:39-09:55:30 EDT, also exited 0. What the tests cover:
  - the spec hash, plus a negative control where one changed byte stops the load;
  - the full re-derivation, plus a negative control where a dropped E1 source stops the build;
  - every bridge edge in the network at base × gain, with 0 measured part;
  - measured pairs equal the brain of record's signed counts;
  - the digest folds in the version, the gains and the ablation;
  - **reduction exactness** against the whole 128,683-cell fireable set, with deciding readouts spiking as the positive
    control;
  - the held-out hash check;
  - 36 grid points and 144 jobs logged with time, and the pick recomputed from the objective;
  - a second `tune()` refuses to touch the append-only log;
  - the gate ran on this digest;
  - a recorded held-out trial replayed live, with the same spike hash;
  - 128 of 128 determinism pairs;
  - the verdict and every item recomputed from the recorded trials with the spec's thresholds;
  - the strike denominator keeps the silent members (58/130 and 48/106);
  - lesions paired and non-vacuous;
  - provenance self-checks and root pinning on every trace;
  - the publications exist;
  - the JS kernel re-run live on the bridged case, exit 0.

## 11. The spec's expectations against what was measured

| spec expectation (s.6, s.13; written before any run) | measured |
|---|---|
| decision-path share near 0 for escape, strike and turn | 0.090% escape, 0.018% strike, 0.0058% turn: **as expected** |
| feasible g_c window about 2-8; prey-evoked escapes likely above about 16 | only g_c = 4 passes all 10 training checks. At g_c = 8 escapes already start below 12 deg (angle fraction 0.25) and 14 of 16 prey trials escape. **The window is narrower and the prey escapes start lower than the mean-field arithmetic said.** |
| M1: behaviour unchanged by shuffling the measured wiring | 10 of 10 shuffles equal: **as expected** |
| looms cause strikes (cross-talk) | 64 / 64 held-out looms and 16 / 16 baseline looms strike: **as expected** |
| class (a) columns sit in the lower field and are reached late; uncrossed, so if strong they push toward the threat | class (a) is **inert** at every gain (g_a changes nothing; removing it changes no outcome) |
| the disclosure sentence applies (the crossing is built) | applies; without E1, 0 of 64 looms escape: **as expected** |
| C3 blank and L1 pass by construction | both pass by construction, and L1 is non-vacuous (the intact fish escaped in 64 / 64): **as expected** |
| **C1 receding** | **not predicted.** The spec discussed dimming ("passed by the eye") but never the eye's response to a receding disc. It is the one result the pre-registration did not see coming, and it decides the verdict. |

## 12. What I did not do, and why

- **The one allowed global change was left unspent.**
  - It could have fixed C1: an eye whose OFF channel ignores shrinking edges, or expansion selectivity on E1.
  - The spec freezes the eye model ("g_a and g_c only").
  - C1 was seen only in the held-out gate, so a change now could not produce a held-out result. It could only add a
    post hoc row.
  - The brief says "no tuning to hide".
  - Whether to reopen the eye model, and with what pre-registration, is a decision outside this note.
- **Not run:**
  - class (b)'s optional sensitivity row (spec s.4; optional, never in the verdict);
  - the tuning or gate at any gain other than the pick.

## 13. Choices (for the honesty page)

1. Operationalisations written to `choices.json` at 09:00:40 EDT, before tuning:
   - M1 compares the two booleans (H1, H2) like-for-like on seeds 0-3;
   - L1 is non-vacuous only when the intact fish escapes with P ≥ 0.5 per side;
   - L2 per lesioned side;
   - a side with no escapes scores 0 for side and angle;
   - a control fails when its baseline loom has 0 Mauthner spikes;
   - the headline cells as in s.5.
2. The deciding readouts that define the reduction: the M-system 6, nIII_dorsal, nMLF and turning, all strict. Forward
   and non-dorsal nIII cells are not in it.
3. M2 ablations remove edges from the intact node set, which is exact and keeps the eye's targets and Poisson draws
   paired. M1 shuffles are re-reduced, which is exact but not paired.
4. The C1/C2 baseline is the training loom run with the gate seeds after tuning, as the spec says.
5. Cache redirection (W and rates under `data/g1c/gate/`). This is not a model change.

## 14. Limits

- The verdict rests on the eye model's lack of loom/recede selectivity. That is a spec-level limit (s.15 there), now
  measured.
- The escape's side comes from E1 by construction. The measured direct contacts are uncrossed and would push the other
  way; they carry 0.09%.
- H2 is scored on RoV3/MiV1/MiV2. Lau 2025 says these are minimally recruited in hunting J-turns (spec disclosure). P4
  is ours.
- The strike fires on every stimulus that reaches the frontal zone, loom or prey (s.9).
- One machine, loaded. Hashes do not depend on load or worker count; timings do.

## Files (all under launches/fishbrain/brain/)

- `fishbrain/bridge.py`: the whole build, tuning, gate, publications and report.
- `tests/test_g1c.py`.
- `data/g1c/gate/`:
  - `rederive.json`, `choices.json`;
  - `tuning_log.jsonl`, `tuning.json`;
  - `gate_results.json`, `gate_raw*.json`;
  - `m1_shuffles.json`, `m2_ablations.json`, `js_parity.json`, `g1c_summary.json`;
  - `net/` (topologies, W), `rates/`, `ref/mece3_231218_4096.npy`, `js/` (the case);
  - `*.out` (run logs).
