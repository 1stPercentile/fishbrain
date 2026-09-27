---
publication_copy: edited-for-publication copy of launches/fishbrain/evidence/G1-gate.md at commit 9ef0eb2ab. Local paths made repo-relative; machine-owner, internal coordination, account and credential details removed. No number, hash, time, method, result or correction was changed; publication notes are marked [Publication note].
title: FISHBRAIN G1 gate, the brain on the real wiring
created: 2026-09-25
status: complete, G1 FAIL (2026-09-25)
evidence_status: measured unless marked
verification_scope: >
  Numbers below were measured on 2026-09-25 on the analysis machine (Apple M5, 10 cores, 16 GB) from brain/data/synapses.parquet (the
  pinned brain of record, content hash 09a71038...25af2) and brain/data/cells_identified.json.
  Literature facts were read live on 2026-09-25 (PMC4742414 full text via fetch; Temizer 2015 and
  Huang 2013 via search summaries; Shiu model.py raw source at commit 91bdd1e).
---
# G1 gate: does the Fish1 wiring flee a loom and turn toward prey?

**Result: FAIL.** On the real Fish1 wiring with Shiu's LIF model, a strong visual drive (up to
about 1,000 tectal cells at 150 Hz) produces **zero spikes beyond the stimulated tectal cells**.
No Mauthner spike and no SPN spike occurred in any of 104 trials. One logged, anchored weight
change (x7.23, 1.99 mV per synapse) fails the same way, as it was predicted to before it ran.
The cause is structural. Fish1's agglomeration mostly separates axons from their cell bodies:
only 1.61% of synapses come from a soma-bearing segment, and 81% of the functional pairs are
single synapses. A segment-level spiking model cannot relay a signal even one synapse deep at
Shiu's weights. What does work: the wiring loads and pins, the E/I sign is settled by ground truth,
Mauthner is confirmed, the eye model behaves, the reduction is exact, runs replay bit for bit, and
the exact gate network is real time on one core. `pytest -q tests`: exit 1 (8 failed, 52 passed).

## 1. Sections

1. Sections. 2. Network built from the brain of record. 3. Pre-registered protocol and criteria
(written 2026-09-25 21:05 EDT, before the first stimulus run). 4. Literature predictions.
5. Results. 6. Benchmark. 7. Fixes in other lanes' files. 8. Choices we made (honesty page).
9. Limits and open decisions.

## 2. Network built from the brain of record (measured)

- **Pairs.** 29,474,316 synapses aggregate into 24,921,936 (pre, post) pairs over 13,458,709
  segments (type 1: 15,883,209; type 2: 13,591,107). Built in 218.8 s, cached in `data/net/pairs_*.npy`.
- **E/I value, settled against ground truth.** The authors' neuroglancer state
  `evaluation_polarity_assignment.json` lists cells by true polarity (same agglomeration). Output
  synapse types in the brain of record:
  - `habenular_axons_classified_true_excitatory`: 70/70 cells majority type 2 (3,683 type 2 vs 1,456 type 1).
  - `habenular_axons_classified_false_inhibitory`: 4/4 majority type 1 (34 vs 10).
  - `inhibitory_cells_predicted_as_inhibitory`: 64 majority type 1, 7 type 2, 1 tied (1,588 vs 350).
  - `excitatory_cells_predicted_as_excitatory`: 63 majority type 2, 8 type 1, 3 tied (2,309 vs 623).
  **Caveat (G1 neuroscience reviewer):** the habenular sets are selected by the authors' own polarity
  index, so their 70/70 and 4/4 are circular and are not evidence. The unbiased tally over **all**
  ground-truth cells still supports the mapping: synapses of true-excitatory cells are 68.7% type 2
  (7,951 vs 3,623), and synapses of true-inhibitory cells are 67.6% type 1.
  **Type 2 = excitatory (+1), type 1 = inhibitory (-1).** This agrees with Fish1's programmatic page
  and the ei_predictions shader (both for the sibling layer). Raw file: `data/ei_polarity_check.json`.
- **Sign rule for the gate: per synapse** (the brain of record as published). A pair's weight is
  (type-2 count - type-1 count) x 0.275 mV. 730,320 pairs (1,534,474 synapses) net to zero and are
  dropped; 11,117,849 pairs are positive and 13,073,767 negative. Dale's-law majority per segment is a
  sensitivity row only (it re-signs synapses of the 1,262,762 mixed-output segments).
- **Stimulated set.** B1 lists 44,435 tectal segments. 24,920 have at least one synapse. None is in
  B1's exclusion list. **7 tectal cells have a direct synapse onto a readout and are not stimulated**
  (10 pairs, 1-4 synapses each):
  - left lobe 17057069410 -> Mauthner-left 15773771512 (2 type-2 synapses) and -> mirror MiM1 20465643738 (1);
  - right lobe 16670127295 -> Mauthner-right 16304202596 (2 type-2) and -> SPN MiV2 19241755427 (1 type-1);
  - left lobe 9570863290, 11019094176, 12569625141, 20239723569 -> left mirror candidates RoM1 20505750235,
    21994882595, 22015160379 (1-4 each); right lobe 13446669933 -> SPN RoM1 19220528442 (1).
  Stimulated: **24,913 tectal cells** (left lobe 11,794, right lobe 13,119).
- **Hops.** Minimum graph distance from any stimulated cell to Mauthner-left = **2** (through
  excitatory pairs only: 3); to Mauthner-right = **2** (excitatory: 2). Before the exclusion both were
  1, through the two 2-synapse contacts above (0.55 mV of weight each against a 7 mV threshold).
- **Exact reduction.** 126,844 segments can fire (reachable from a stimulated cell through
  excitatory pairs; 235,430 nonzero pairs among them). 8,645 of those can also reach a readout. Kept
  network: **8,654 neurons, 38,304 pairs, 54,404 synapses, 2,048 stimulated cells**, plus every
  readout present in the graph. In Shiu's LIF with no background input, nothing outside the fireable
  set can spike, and nothing outside the backward set can change a readout, so the readouts are
  exact. Test `test_reduction_is_exact` re-checks this by simulation.
- **Mauthner confirmation (B1 section 8, all on the full parquet):**
  - 8.1 counts reproduce exactly: left 15773771512 in 5,849 (type 1 1,294 / type 2 4,555), out 38;
    right 16304202596 in 4,173 (629 / 3,544), out 25; runner-ups 17405827637 = 2,603 and
    16039136570 = 2,734; SPN 158660 (20607607971) in 117 / out 35.
  - 8.2 rank by input count among 49,047 segments holding a hindbrain soma (atlas mece0 label 6):
    left **1st**, right **2nd**.
  - 8.3 lateral dendrite: 2,363 (left) and 1,738 (right) input synapses sit more than 50 um lateral
    of the midline, from 734 and 682 presynaptic segments of which only 57 and 21 hold a soma
    (consistent with afferents from outside the brain, e.g. the VIIIth nerve); 1,673 and 1,444 of
    them are type 2. Spiral-fiber clusters (mece2 29, 32; 15 somas / 15 segments): **0** synapses
    onto either Mauthner segment. 29 and 92 inputs fall in B1's axon-cap box, none from those
    segments. This check is **inconclusive**: B1 found the Mauthner segments hold soma, dendrites
    and only the first microns of axon, and spiral fibers terminate on the axon cap.
- **Wiring-only direction prediction (before simulating).** Stimulating one lobe at a time,
  each lobe alone makes about 106,300 cells fireable and reaches both Mauthner cells.
  Mauthner-left receives 543 excitatory synapses from fireable cells whichever lobe is driven;
  Mauthner-right receives 262 (left lobe) or 267 (right lobe). **Reachability does not
  discriminate the side; it predicts a left-Mauthner bias** (about 2:1 in fireable excitation).
  Dale-majority gives the same picture (706/701 vs 371/378). `data/net/g1_graph_checks.json`.

## 3. Pre-registered protocol and criteria

Written 2026-09-25 21:05 EDT, before any stimulus was rendered or simulated. After the first run,
at most one principled global change is allowed, and it must be logged here with every control
re-run. Retina parameter digest `eca7c6a1ca385933ea3d22b753605f709c0b727ae4950fdf2b28542ad3867ce6`;
LIF parameter digest `958a1404a0e6dfab7682b5f9c4419aa5d3e25963cc18f582cbd99a4d69284717`
(Shiu defaults, dt 0.1 ms).

**The eye (brain.py `Retina`; every value ours unless a source is named).**
- The stimulus is rendered by B2's `stimuli.py` on its 2-degree lattice in 5 ms frames. For each
  eye, only the lattice points inside that eye's field are used.
- Transient stage: hp = c - m, and m moves toward c with a 100 ms time constant (a high-pass per
  point).
- Centre-surround stage: y = hp - S hp. S is a Gaussian surround (sigma 10 deg, cut at 30 deg) whose
  rows are renormalised inside the eye's field, so any uniform change of the whole field gives y = 0.
- OFF channel only (w_off 1, w_on 0): output r = max(0, -y). The source is Temizer 2015: dark
  looms evoke escapes, while bright looms, dimming and receding stimuli rarely do.
- Tectal drive = `cells.tectum_drive` of the point values r. It is linear in r, so it is applied as
  a matrix built from 2 x 4,396 single-point calls; that matrix matches direct calls to 1.2e-14.
- Rate = 150 Hz x clip(drive, 0, 1), where 150 Hz is Shiu's `r_poi` (model.py line 39, read
  2026-09-25). Rates are rounded to 1e-6 Hz. No background input.

**Stimuli.**
- Loom: dark disc, l/v 240 ms, 4 -> 140 deg, at azimuth +90 (right) or -90 (left), elevation 0;
  500 ms blank before, 500 ms held after. Expansion lasts 6,785 ms.
- Receding: 500 ms blank, then the 140 deg disc appears and stands still for 1,000 ms (the onset
  window, reported separately and not scored), then shrinks along the reversed loom, then holds
  500 ms at 4 deg.
- Dimming: B2's matched whole-field dimming, both eyes.
- Prey: 3 deg bright dot, 30 deg/s, 20 deg sweep around azimuth +30 (right) or -30 (left); the left
  dot is the mirror image (direction -1). 500 ms blank, then 3 s.
- Blank: 3.5 s.
- Seeds: the same 8 seed strings `G1-seed-0..7` in every condition, so comparisons are paired.

**Readouts and decision.**
- Escape = the first Mauthner spike in the window. Its side is the escape decision.
- Turn index TI = (R - L) / (R + L), where R and L are the mean spikes per turning SPN
  (RoV3/MiV1/MiV2) on each side. Right: 30 confirmed + 1 mirror candidate. Left: 1 confirmed + 30
  unconfirmed mirror candidates. Positive TI = right turn.
- Windows: loom and dimming [500, 7,385) ms (the expansion plus 100 ms); receding motion
  [1,500, 8,385); prey and blank [500, 3,500).

**Pass criteria.**
- (a) Per loom side: P(escape) >= 0.5 over 8 seeds; the first Mauthner to fire is the
  predicted one (section 4) in >= 75% of escapes; and the first spike comes when the disc is
  >= 12 deg (Temizer's 21.7 +- 4.9 deg critical angle minus 2 SD) in >= 75% of escapes. The window
  end enforces "no later than 100 ms after expansion ends" (Dunn: 81 ms). Mauthner spike counts are
  reported; one to a few spikes is an escape, tonic firing is flagged.
- Receding control (per side): mean Mauthner spikes per trial in the motion window <= 0.5 x the
  same-side loom's, and P(escape) lower than the loom's.
- Dimming control: the same, against the mean of both looms. Because the surround cancels any
  uniform change, this control is expected to be satisfied by the eye model, not by the brain. The
  tectal input spikes are reported so the reader can see where the discrimination happened.
- (b) Prey right: TI > 0 in >= 75% of seeds. Prey left: TI < 0 in >= 75%. The task's literal
  claim is the gate. The paired within-cell comparison (the same cells, prey right vs left, matched
  seeds) is reported beside it because the two sides' cell populations differ (confirmed vs
  candidates).
- No-stimulus control: no single turn direction in >= 75% of seeds. With no background input
  nothing fires, so this is expected to be vacuous and is reported as such.
- (c) Lesion of both Mauthner cells (loom left and right, 4 seeds): zero Mauthner spikes. True by
  construction.
- (d) Whole-brain degree-preserving shuffle (`sim.Network.shuffled_copy`, seed
  `G1-whole-brain-shuffle`, over all 13.46M segments and 24.19M nonzero pairs), re-reduced with the
  same stimulated cells: the decision vector (majority escape for loom left and right, majority
  turn for prey left and right) must differ from intact. A shuffle inside the kept subnetwork and
  the Dale-majority sign rule are sensitivity rows, not gate items.
- (e) The same seed and inputs give the same spike hash twice, and the rate array hashes the same
  when recomputed.

## 4. Literature predictions (read 2026-09-25)

- **Which Mauthner fires.** Dunn et al. 2016, Neuron 89:613 (PMC4742414, full text): looms were shown
  in one monocular field; "stimuli in the right visual field consistently evoked escapes to the
  left", and after unilateral ablation "only escape responses contralateral to the ablated
  M-system ... were perturbed". A Mauthner cell drives escapes to the side opposite itself, so a
  right loom (escape left) needs the **right** Mauthner. Prediction: **the Mauthner ipsilateral to
  the loom fires first**, i.e. contralateral to the tectal lobe that sees it.
- **When.** Temizer et al. 2015 (Curr Biol): critical angle 21.7 +- 4.9 deg (search summary). Dunn
  2016: tectal population threshold about 66-72 deg, then 81 ms to the escape. Temizer 2015: dimming,
  receding and bright looms were "substantially less effective" (search summary); Dunn 2016 measured
  3.4 +- 1.1% escapes to dimming.
- **Turns.** Huang et al. 2013 Curr Biol 23:1566 (search summary): RoV3, MiV1 and MiV2 fire for
  turns to their own side; 76% of these vSPNs showed an ipsilateral bias and were silent for
  contralateral turns. Prey on the right should raise right-side turning-SPN activity.
- **Tectum -> Mauthner.** Dunn 2016 leaves open "a direct or indirect path from the OT".

## 5. Results

### 5.1 Run 1: the pre-registered model (Shiu's weights), 2026-09-25 21:14-21:27 EDT

`python -m fishbrain.brain gate`, 4 worker processes, 801.5 s wall (load average 38-73 on 10 cores).
Raw: `data/net/g1_gate_results_run1.json`. **Gate: FAIL.**

**Nothing downstream of the tectum fired in any trial** (104 trials: 64 intact, 8 lesion, 32 shuffled; 14 condition/variant cells).
Mean tectal input spikes per trial and downstream spikes (all cells beyond the stimulated set):

| run | condition | trials | tectal spikes / trial | downstream spikes | readout spikes | Mauthner spikes |
|---|---|---|---|---|---|---|
| intact | loom_left | 8 | 179,526 | 0 | 0 | 0 |
| intact | loom_right | 8 | 173,574 | 0 | 0 | 0 |
| intact | recede_left | 8 | 227,681 | 0 | 0 | 0 |
| intact | recede_right | 8 | 229,400 | 0 | 0 | 0 |
| intact | dim | 8 | 0 | 0 | 0 | 0 |
| intact | prey_left | 8 | 31,710 | 0 | 0 | 0 |
| intact | prey_right | 8 | 46,586 | 0 | 0 | 0 |
| intact | blank | 8 | 0 | 0 | 0 | 0 |
| Mauthner lesion | loom_left / right | 4 + 4 | 179,417 / 173,351 | 0 | 0 | 0 |
| whole-brain shuffle | loom_left / right | 8 + 8 | 168,669 / 133,352 | 0 | 0 | 0 |
| whole-brain shuffle | prey_left / right | 8 + 8 | 37,902 / 40,922 | 0 | 0 | 0 |

Verdicts under the pre-registered criteria: (a) loom left FAIL, loom right FAIL (P(escape) = 0);
receding and dimming controls FAIL (undefined: the loom never fired Mauthner); (b) prey left and
right FAIL (turn index 0 in every trial); no-stimulus control PASS but **vacuous** (0 spikes in
the whole network); (c) lesion PASS but **vacuous** (true by construction, and the intact
Mauthner never fired either); (d) shuffle FAIL (intact and shuffled decisions are both "none").

**What fired instead.**
- The eye did its job. By the end of the loom, 1,026 of the 1,045 kept cells in the
  contralateral lobe were driven at 75 Hz or more (sum of rates 154,245 Hz at 7,285 ms). Only the
  stimulated tectal cells spiked.
- In the last 2 s of the right loom, 151,325 tectal spikes produced **0 spikes at every hop from 1
  to 19**.
- Why, from the wiring alone (steady state v = sum of w x synapses x rate x tau_syn, at the loom's
  end):
  - the most depolarised hop-1 cell reaches **0.83 mV** (1.03 mV counting excitation only) against
    the 7 mV threshold;
  - hop-1 cells get a median of **1** excitatory input synapse (max 39); hop 3, where most of
    Mauthner-left's fireable input sits, a median of 5 (max 543);
  - **81%** of kept pairs are a single synapse (94.5% have 1 or 2).
  At Shiu's 0.275 mV a cell needs about 25 coincident synapses. The first hop-1 cell would cross
  threshold in steady state only at w_syn = 2.33 mV (loom) or 3.88 mV (prey).
- The cause is the reconstruction, not the eye. Only **1.61%** of the 29,474,316 synapses have a
  presynaptic segment that holds a soma. Soma-bearing segments have a median of **0** output
  synapses (mean 4.1), and 32.6% of synapses land on them. The agglomeration mostly leaves axons
  detached from their cell bodies. Most of the graph's presynaptic partners are orphan axon
  fragments, and nothing drives an orphan fragment.
- Structural reachability is not the problem. Through soma-bearing segments alone, the stimulated
  cells still reach both Mauthner cells in 2 hops, plus 88 of 100 readouts (136,951 pairs).
- Exactness-test positive control (not a gate result): at w_syn = 20 mV (73x Shiu), 0.5 s of the
  right loom's end fires Mauthner-left 117 times and Mauthner-right 82 times. At 45 mV: 211 and
  210. Both fire, left more, for a loom on the right. That is the wiring's left bias from
  section 2, and it is the opposite of Dunn's prediction.

### 5.2 The one logged change: w_syn scaled by synapse density (prediction written before running)

Written 2026-09-25 21:28 EDT, before run 2 was simulated (run 2 started 21:29:08).
- **Change.** Shiu calibrated w_syn = 0.275 mV on FlyWire's proofread neurons: about 5 x 10^7
  chemical synapses over about 130,000 neurons (bioRxiv 2023.06.27.546656 v2 abstract), i.e.
  384.6 synapses per neuron. Fish1's 180,782 soma-bearing segments carry a mean of **53.2** input
  synapses (zeros included; 82.8 over the 116,059 that have any synapse).
- **Scale.** 384.6 / 53.2 = 7.23, so **w_syn = 1.9881 mV**. It is global, applies to every
  synapse, and nothing else changes.
- **Linear prediction.** The strongest hop-1 cell reaches 0.825 x 7.23 = **5.97 mV** at the
  loom's end and 3.6 mV for prey, both below 7 mV. Poisson fluctuations may fire a few hop-1
  cells. Hop-2 and hop-3 cells get almost no input from them, so Mauthner and the turning SPNs are
  **predicted silent: (a) and (b) FAIL**.
- The probe for the exactness test already showed only 106 downstream spikes and no readout at
  5 mV per synapse.
- Run 2 re-runs every intact condition, 8 seeds each, at this weight.

**Run 2 result** (`python -m fishbrain.brain change`, 2026-09-25 21:29-21:31 EDT, 157.1 s wall;
raw `data/net/g1_change_results.json`). **As predicted: FAIL.** Mauthner spikes 0 and readout spikes
0 in all 64 trials. Downstream spikes per trial (fluctuation-driven first-hop cells only):

| condition | downstream spikes, seeds 0-7 | readout | Mauthner |
|---|---|---|---|
| loom_left | 11, 14, 12, 13, 10, 15, 14, 20 | 0 | 0 |
| loom_right | 30, 37, 29, 22, 36, 40, 27, 29 | 0 | 0 |
| recede_left | 14, 18, 16, 26, 19, 18, 14, 17 | 0 | 0 |
| recede_right | 40, 41, 46, 49, 35, 56, 33, 43 | 0 | 0 |
| dim | all 0 | 0 | 0 |
| prey_left | 0, 0, 1, 2, 1, 2, 1, 1 | 0 | 0 |
| prey_right | 5, 5, 6, 8, 7, 10, 7, 5 | 0 | 0 |
| blank | all 0 | 0 | 0 |

The lesion and shuffle items read "fail" in that file only because run 2 did not re-run them: they
are moot when the intact Mauthner never fires. **No further change was made.** The task allowed
principled changes but said not to tune until green, and every weight that reaches Mauthner is
far from any anchor:
- the first hop-1 cell crosses threshold in steady state only at 2.33 mV;
- readouts first fire somewhere between 7 and 20 mV per synapse, i.e. 25x to 73x Shiu's weight.

### 5.3 Sensitivity rows (not gate items)

- **Dale-majority sign rule** (steady state, same loom end): the kept network is 9,219 neurons,
  42,719 pairs and 2,107 stimulated cells, and 72.9% of pairs are a single synapse. The strongest
  hop-1 cell reaches 1.24 mV for the right loom and 0.83 mV for the left; w_syn would have to be
  1.56-2.33 mV for the first hop-1 cell to cross. **Same conclusion.** It was not simulated.
- **Subnetwork shuffle**: built (`data/net/g1_per_synapse_subshuffled.npz`, same 8,654 neurons
  and 38,304 pairs), not simulated: with the intact network silent past the tectum, a shuffle
  cannot change a decision that is "none" on both sides.
- **Whole-brain shuffle**: built for the gate. The shuffle of 24,191,616 pairs over 13,458,709
  segments took 44.6 s (2 repair rounds, 0 unrepaired pairs). In the shuffled wiring 124,359
  segments can fire; re-reduced, it keeps **5,290 neurons, 10,471 pairs and 1,679 stimulated
  cells**, and is equally silent (section 5.1). The stimulated cells stay the same, so in the
  shuffled wiring both Mauthner cells land 1 hop from a stimulated cell. The hops >= 2 rule is a
  property of the intact brain.

### 5.4 The pytest run of record

Command, bare, no pipes (2026-09-25, finished 21:51 EDT):

    cd launches/fishbrain/brain && .venv/bin/python -m pytest -q tests

**Exit code 1. 8 failed, 52 passed, 0 skipped, in 83.14 s.** The gate fixture re-ran the whole
gate protocol in 64.9 s (retina rates from cache, load average 9) and wrote
`data/net/g1_gate_results.json`.
- Failed, all gate items: `test_a_loom_fires_the_predicted_mauthner_in_window[left]` and
  `[right]`; `test_control_receding_fires_mauthner_less[left]` and `[right]`;
  `test_control_dimming_fires_mauthner_less`; `test_b_prey_turns_toward_the_dot[left]` and
  `[right]`; `test_d_shuffled_wiring_changes_the_decision`. Each failed because no Mauthner or
  SPN spike exists to score.
- Passed in `tests/test_gate.py` (10):
  - wiring = brain of record;
  - Mauthner counts = B1's;
  - type 2 excitatory by ground truth;
  - stimulated set tectal and >= 2 hops from Mauthner;
  - retina matrix = `tectum_drive`;
  - whole-field dimming gives 0 tectal rate;
  - reduction exact (every kept cell's spike train, at 20 mV, readouts firing as the positive
    control);
  - no-stimulus control (vacuous);
  - Mauthner lesion (vacuous, by construction);
  - (e) determinism: same seed gives the same raster twice, and the rates hash the same when
    recomputed. A different seed gives a different raster. The pool worker's hash equals the
    in-process hash.
- The other lanes' 42 tests (`test_cells`, `test_precomputed`, `test_sim_synthetic`) all pass.
- **Replay across processes:** the 104 trials of run 1 (CLI, 21:14) and of the pytest run (21:50)
  have identical spike hashes and rate hashes, 104/104. Same machine; cross-machine is not tested.
- **The exactness check can fail:** deleting one active pair (5 synapses, 12775206625 ->
  Mauthner-left, presynaptic cell firing 70 times) changed the spike trains of 171 kept cells, and
  Mauthner-left went from 117 to 97 spikes.

**Gate G1: FAIL.** Per the brief, nothing downstream should be built on this brain model.

## 6. Benchmark (this Mac, 2026-09-25 21:33-21:39 EDT; `brain/data/bench/g1_bench.json`)

Apple M5 (sysctl), 10 cores, 16 GB, macOS 26.6.2 arm64, Python 3.9.6, numpy 2.0.2, single thread.
The Mac was shared: load average 45-57 at the start of every row, falling to 17 by the end of the
last one. **CPU time is the honest number; wall time is what a viewer would wait today.**
Workload: 1,000 ms simulated at dt 0.1 ms (10,000 steps) of the real right loom's busiest second,
Shiu LIF.

| row | neurons | pairs | spikes | CPU us/step | RTF (CPU) | RTF (wall) | peak RSS |
|---|---|---|---|---|---|---|---|
| full brain of record (every segment) | 13,458,709 | 24,191,616 | 986,125 | 13,202 | **0.0076** | 0.0041 | 914 MB |
| fireable set F | 126,844 | 235,430 | 986,125 | 372 | 0.27 | 0.072 | 400 MB |
| reduced (the gate network) | 8,654 | 38,304 | 85,698 | 61 | **1.64** | 0.28 | 292 MB |
| reduced at w 20 mV (activity reaches Mauthner) | 8,654 | 38,304 | 97,711 | 52 | 1.93 | 0.85 | 505 MB |

RTF = simulated ms per elapsed ms. Build time: full 41.6 s, fireable set 7.4 s, reduced 3.9-20.4 s
(from cache). An earlier reduced run at load 46 gave CPU RTF 2.01 (50 us/step).
- The full and fireable rows produced the **same 986,125 spikes** from the same seed and inputs:
  nothing outside the fireable set fires, an independent check of the reduction's premise.
- **Full brain on this Mac: about 132x slower than real time on one core** (13.2 ms per 0.1 ms
  step). The cost is the per-step update of 13.46M segments, and it is memory-bound.
- **The gate network is real time already**, at about 1.6-2x on one core, and it is exact for
  every readout. The fireable set runs at 0.27x.

**Server estimate (extrapolation, not measured).** A full-brain step moves about 25 bytes per
segment (u and h read and written as float32, a mask and the refractory bookkeeping), about
340 MB per step.
- One numpy core anywhere: about 0.01x (this Mac: 0.0076x).
- A compiled multi-core CPU backend at about 200 GB/s: about 1.7 ms per step, **about 0.06x**.
- A GPU at 1.5-3 TB/s: **about 0.5-1x**.

Any such backend must reproduce B2's golden hashes. None of this is needed for the gate
network, which is exact for readouts and already real time on one core.

## 7. Other lanes' files

- **No file owned by another lane was edited.**
- Read-only uses of other lanes' private helpers: `stimuli._disc_trace`, `stimuli._n_frames` and
  `stimuli._times` (to build the receding stimulus with a static onset period; B2's `receding()`
  has a blank lead-in only), and `cells._somas` and `cells._agg_map` (for the B1 section-8 graph
  checks).
- `sim.py` was used as-is: `Network.from_pairs`, `shuffled_copy`, `lesion`, `Simulator`,
  `LIFParams(w_syn_mV=...)` for the change run and the exactness test.
- The earlier draft of `brain.py` in this lane (before this run) had `SIGN_RULE = "dale_majority"`.
  It was switched to per synapse **before any stimulus was simulated**. The previous caches
  `data/net/reduce_*.npz/json` are from that draft and are superseded by `data/net/g1_*`.

## 8. Choices we made (for the honesty page)

Everything here is ours unless a source is named.
1. **Sign.** Per synapse, type 2 = +1, type 1 = -1 (the value mapping is measured, section 2).
   Zero-net pairs are dropped.
2. **Stimulated set.** B1's tectal cells with at least one synapse, minus B1's exclusions, minus
   any tectal cell with a direct synapse onto a readout (7 cells).
3. **Reduction.** Exact for Shiu's LIF with no background: fireable set F, then a backward
   closure to the readouts. Readouts are kept even if they cannot fire.
4. **Eye model.**
   - Transient high-pass per lattice point, tau 100 ms.
   - Balanced Gaussian surround, sigma 10 deg, cut at 30 deg, renormalised inside each eye's
     field.
   - OFF channel only, following Temizer 2015's dark-loom finding.
   - `cells.tectum_drive` applied as a linear operator.
   - Rate = 150 Hz (Shiu `r_poi`) x clip(drive, 0, 1), rounded to 1e-6 Hz; 5 ms frames. No
     background.
5. **Protocol.**
   - B2's stimulus defaults, plus: loom at +/-90 deg azimuth; 500 ms pre-blank and 500 ms hold;
     receding with a 1,000 ms static onset period (not scored).
   - Prey at +/-30 deg, mirrored for the left.
   - 8 paired seeds per condition (4 for the lesion).
6. **Readouts and decisions.** The first Mauthner spike in the window decides the escape. Turn
   index = (R - L)/(R + L) over mean spikes per turning SPN, including B1's unconfirmed left
   mirror candidates. The decision vector is the majority over seeds.
7. **Criteria.** Section 3: P(escape) >= 0.5; side and timing >= 75%; 12 deg minimum angle; 100 ms
   sensorimotor allowance; controls <= 0.5 x the loom's Mauthner spikes; turn sign >= 75%.
8. **Shuffle.** `sim.shuffled_copy` of the whole brain (seed `G1-whole-brain-shuffle`) with the same
   stimulated cells. Subnetwork shuffle seed `G1-subnetwork-shuffle`.
9. **The logged change.** w_syn x 7.23 = 1.9881 mV, anchored on FlyWire synapses per neuron
   (about 5e7 / 130,000) over Fish1 input synapses per soma-bearing segment (53.2). It failed.
10. **Exactness test weight.** 20 mV per synapse (a test of the reduction only, not a model
    claim).
11. **Benchmark workload.** The last 1,000 ms of the right loom. In the full and fireable rows,
    stimulated cells outside the reduced network get their lobe's mean kept-cell rate per frame.

## 9. Limits and open decisions

- **G1 fails for a structural reason.** The agglomeration `seg_241003_agg241003` is a segment
  graph, not a neuron graph:
  - only 1.61% of synapses come from a soma-bearing segment;
  - soma-bearing segments have a median of 0 outputs;
  - 81% of the functional pairs are single synapses.
  A spiking model that treats each segment as a neuron cannot carry a signal two synapses deep
  at any weight near Shiu's. That is not a parameter problem this lane can fix honestly.
- **Options for the rethink** (none tried here):
  1. Proofread or attach axons to somas for the tectum-to-hindbrain path (the authors' HMI
     release traced hindbrain cells; B1 noted `hindbrain_reconstructions` as a public layer).
  2. Change the model level: a rate or graph-propagation model over the soma-bearing graph. That
     graph still reaches both Mauthner cells in 2 hops and 88 of 100 readouts, but it is a
     different model from Shiu's and must be disclosed as such.
  3. A weight near the relay regime (about 20 mV and up). That is not Shiu's model, and at
     20-45 mV a right loom fires both Mauthner cells, left more (117 vs 82 spikes in 0.5 s; 211
     vs 210 at 45 mV). That is the side Dunn's rule says should stay quiet, and it matches the
     wiring's 2:1 left bias from section 2.
- **Not tested.**
  - Cross-machine determinism (B2's golden hashes have not run on the server).
  - The eye model's parameters: chosen, not fitted.
  - Left/right rests on B1's axis call. A mirror error would swap every side claim.
  - The left turning population is 30 unconfirmed mirror candidates plus 1 confirmed cell.
- **Vacuous items in this run.** The no-stimulus control and the lesion "pass" only because
  nothing downstream fires. The dimming control would pass for the eye model alone: the balanced
  surround gives exactly 0 tectal rate for whole-field dimming (tested).
