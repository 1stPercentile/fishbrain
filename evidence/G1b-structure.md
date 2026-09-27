---
publication_copy: edited-for-publication copy of launches/fishbrain/evidence/G1b-structure.md at commit c601f956d. Local paths made repo-relative; machine-owner, internal coordination, account and credential details removed. No number, hash, time, method, result or correction was changed; publication notes are marked [Publication note].
title: G1b structure, does the Fish1 wiring carry side and stimulus information? (task S)
created: 2026-09-26
status: done (task S). Verdict ABSENT under the pre-registered rule; both positive controls pass.
evidence_status: measured unless marked "expected", "post hoc" or "choice". Graph analysis only; nothing was simulated.
verification_scope: >
  Measured 2026-09-26 02:25-03:15 EDT on the analysis machine from the pinned brain of record (the cached pair
  arrays data/net/pairs_*.npy, wiring content hash 09a71038...25af2; 24,921,936 pairs, 29,474,316 synapses),
  brain/data/cells_identified.json (B1), brain/data/readouts.json (task R, strict populations, task_criteria
  M-system picks) and brain/data/cells.parquet (synapse centroids, for Null B's sides). No network call was
  made for the analysis. Literature read 2026-09-26: Dunn et al. 2016 (PMC4742414, the two laterality quotes,
  through a summarising fetch that returned them verbatim); Helmbrecht et al. 2018 (search summary only,
  secondary). Everything else literature-side is cited through G1-gate.md, G1-cells.md and G1b-readouts.md.
files: task S. brain/fishbrain/structure.py, brain/data/structure/, brain/tests/test_structure.py, this note.
outputs: brain/fishbrain/structure.py, brain/data/structure/structure.json (896 KB) + shuffles/ (31 files) +
  replay_check.json, 4.0 MB in all, git-ignored (rebuild with `python -m fishbrain.structure run`, about 25 min),
  brain/tests/test_structure.py, this note
---
# G1b structure: does the wiring carry the information?

**Verdict: ABSENT** under the rule fixed at 02:27 EDT. Against degree-preserving shuffles, the real
wiring separates neither left from right (0 of 3 readouts) nor loom from prey (0 of 2 contrasts).
**But this null comparison could not have succeeded in the predicted direction.** The shuffled values
spread so widely that the largest |z| any value could reach in the biologically predicted direction
is 2.25 (M-system), 2.85 (turning), 2.95 (escape vs strike) and 2.74 (escape vs turning), all under
the bar of 3. What carries the "lookup table" conclusion is the size of the signal, not the null test:
- The tectum's share of a readout's input, summed over all walks of 1 to 3 steps, is 1.5e-5 to 9.2e-4
  per readout side (4.6e-3 for left nIII dorsal).
- Of each readout's input synapses, 0.35% to 3.1% come from cells within 2 steps of the tectum.
- 68% to 81% come from segments that have no input synapse at all (orphan axon fragments), which
  nothing can drive.

**What the small signal that exists looks like.**
- Its side preference is ipsilateral: each lobe reaches the readouts on its own side (M-system +0.98,
  turning +0.41, strike +0.34). The biology predicts contralateral for the M-system and turning.
- A null that keeps each cell's own-side vs other-side partner counts (Null B) predicts as much
  ipsilateral preference or more: M-system +0.99, turning +0.90, strike +0.72. So it is "cells connect
  within their own side", not routing. That turning and strike come out *less* ipsilateral than
  laterality alone is a lead only (section 4).
- Loom vs prey has the wrong sign. Relative to each other, the loom patch leans toward the strike and
  turning readouts, and the prey patch toward the escape readouts. That is not beyond the shuffles either.

**The instrument works.** Inside the tectum, the same indices find strong structure:
- Each lobe's output stays in its own lobe: PC1 = +1.00, against Null A's -0.003 +- 0.072. Null B keeps
  sides, so it reproduces PC1 by construction.
- Loom-zone and prey-zone cells feed their own zones: PC2 = +0.97, beyond both nulls (Null A z 4.8,
  Null B z 3.8).

**That structure is gone by the hindbrain.** The tectum's own output synapses land in the tectal neuropil
(median x 51,735, against the Mauthner cells at ~70,400). That fits G1's cause: in Fish1 the tectobulbar
axons are not attached to the tectal somas.

`pytest -q tests/test_structure.py`: **exit 0, 21 passed.** Five deliberate breakages each made it fail.

## 0. Pre-registration (written 2026-09-26 02:27 EDT, before any path weight, index or shuffle was computed)

Orientation reads done before this was written (no result of this task): the pair arrays' sizes, the
number of in-graph tectal cells per lobe (11,799 left, 13,121 right; 24,920 total), the number of
tectal cells with any output synapse (8,893; 95,169 output synapses), and the input-synapse totals of
the readout sets. Everything below was fixed before the first index was computed. Anything changed
later gets its own timestamped line in section 9.

**Graph.** All 24,921,936 (pre, post) pairs of the brain of record (`data/net/pairs_*.npy`, wiring hash
09a71038...25af2), with their type-1 and type-2 synapse counts t1, t2. Type 2 = excitatory (G1-gate s.2).
n_in(j) = all input synapses of segment j (t1 + t2 summed over its input pairs). No pair is dropped
(pairs whose per-synapse net is 0 still carry excitatory synapses).

**Edge weights (our choice): input fractions.**
- Excitatory: E[j, i] = t2(i -> j) / n_in(j).
- Signed: S[j, i] = (t2 - t1)(i -> j) / n_in(j).
- Any: A[j, i] = (t1 + t2)(i -> j) / n_in(j) (used for reachability sets only).

**Path weight.** For a source vector s over segments, x_k = W^k s (walk sums: every walk of exactly k
synaptic steps, product of the input fractions along it; cells may repeat). x_k[j] is the share of j's
input traceable to the source through exactly k steps; in a linear rate model with unit gains it is the
k-th term of j's steady-state response. Cumulative = x_1 + x_2 + x_3.
- **Primary statistic: cumulative excitatory (E) weight, hops 1-3.** Per-hop (1, 2, 3) and signed (S)
  values are secondary rows. This avoids choosing a hop after seeing the data.
- Lobe sources: s = 1 on every in-graph tectal segment of that lobe (B1's tectum, all 24,920; not
  `brain.stimulated_set`, whose 7 dropped cells are direct contacts, which is signal here).
- Population weight: the mean of x over the population's members that have at least one input synapse.

**Readout populations** (`data/readouts.json`, strict atlas sets, M-system picks = task_criteria):
- M-system L / R = Mauthner + MiD2cm + MiD3cm of that side (3 cells, mean per cell). **Side-verdict readout.**
- turning L / R = RoV3 + MiV1 + MiV2 (strict: 32 / 23). **Side-verdict readout.**
- strike L / R = nMLF (15 / 18), the strike rule's drive and side term. **Side-verdict readout.**
- Secondary (reported, not in the verdict): Mauthner alone, nIII dorsal (the convergence proxy), forward.

**Side index (our choice).** For a population pair (X_L, X_R): a = w(left lobe -> X_L), b = w(left lobe -> X_R),
c = w(right lobe -> X_L), d = w(right lobe -> X_R).

    SI = 1/2 [ (a - b) / (|a| + |b|)  +  (d - c) / (|c| + |d|) ]

+1 = purely ipsilateral routing (each lobe to its own side), -1 = purely contralateral, 0 = none.
Undefined when a lobe reaches neither member. Under a no-routing model w = f(lobe) x g(target) with
positive weights, SI = 0 exactly, whatever the lobe sizes (11,799 vs 13,121) and whatever a target's own
bias (Mauthner-left's 2:1 excess of input); tested in pytest.

**Sign predictions (literature, not measured here).**
- M-system: **SI < 0 (contralateral)**. `brain.MAUTHNER_PREDICTION = "ipsilateral_to_stimulus"` (Dunn et
  al. 2016, G1-gate s.4): a right-field loom is seen by the right eye, which drives the LEFT lobe, and
  must fire the RIGHT Mauthner.
- turning: **SI < 0**. Prey on the right drives the left lobe and should turn the fish right; RoV3 / MiV1 /
  MiV2 fire for turns to their own side (Huang et al. 2013, via G1-cells s.4 and G1-gate s.4).
- strike (nMLF): **two predictions conflict.** Thiele 2014 (nMLF bends the tail to its own side) plus a
  strike toward the prey needs SI < 0. Gahtan 2005's anatomy (nMLF dendrites in the ipsilateral tectum)
  and task R's direct contacts (3 ipsi / 1 contra) predict SI > 0. Both are written down; the data
  says which it matches.
- nIII dorsal: no side prediction (convergence needs both medial recti).

**Stimulus patches (our choice), through `cells.tectum_drive`.** Per eye e (azimuth sign + for the right eye):
- loom-like: `cells.disk(+-90, +30, 30)` = a 60-degree disk, lateral, upper field; contrast 1.
- prey-like: a 3-degree dot at (+-20, 0), frontal / nasal field (a single lattice point); contrast 1.
- The drive vector over tectal cells is the source s (only the contralateral lobe is driven).
- **Stated plainly:** each patch's weights are normalised per patch, so patch size cancels; only
  retinotopic position matters. The position -> field map is B1's rank-uniform model (`lobe_coords`,
  `preferred_field`), not a measurement. So this test asks "do the tectal cells the model assigns to the
  frontal field route differently from those it assigns to the upper-lateral field?"

**Selectivity index (our choice), per eye, targets pooled over both sides** (so it measures type, not side):

    SEL(E vs X) = 1/2 [ (w(loom, E) - w(loom, X)) / (|w(loom, E)| + |w(loom, X)|)
                      + (w(prey, X) - w(prey, E)) / (|w(prey, E)| + |w(prey, X)|) ]

E = M-system (both sides), X = strike (nMLF, both sides) or turning (both sides). **Prediction: SEL > 0**
(loom routes to escape, prey to strike / turn). The verdict uses the mean of the two eyes; per-eye
values are reported. A stimulus-side index (does a right-eye loom reach the RIGHT M-system?) is a
secondary row.

**Intermediate sets and Jaccard (per readout population and side, excitatory edges t2 > 0).**
- F1(lobe) = the lobe's out-neighbours; F2(lobe) = out-neighbours of F1 (step-2 walk support).
- B1(T) = in-neighbours of T; B2(T) = in-neighbours of B1(T).
- Hop-1 intermediates I1 = F1 & (B1 | B2); hop-2 intermediates I2 = F2 & B1; both minus every tectal
  source cell and minus T's own members.
- Binary Jaccard |I(L) & I(R)| / |I(L) | I(R)|. Weighted Jaccard (sum min / sum max) of the flow through each
  intermediate (forward weight from the lobe x backward weight into T, normalised per lobe).
- Global line: Jaccard of the whole F1 and F2 sets of the two lobes (tectal cells removed).
- No direction is predicted; reported against the nulls.

**Reachability (item 4), per readout population and side.** Share of its input synapses from
presynaptic segments that (i) have no input synapse at all (orphans: unreachable from anything, the
ceiling), (ii) are tectal (readout at hop 1), (iii) are within 1 step of the tectum (hop <= 2), (iv) within
2 steps (hop <= 3). Once over any pair, once over excitatory pairs (t2 > 0, counting only excitatory input
synapses). Plus the walk share (cumulative E weight from both lobes).

**Null A (the task's control, primary).** Degree-preserving whole-brain shuffle, the algorithm of
`sim.Network.shuffled_copy` applied to all 24,921,936 raw pairs: each pair keeps its pre segment and its
(t1, t2); post ends are permuted uniformly (`sim.make_bitgen(seed, "shuffle")`); duplicate pairs and
self-loops are repaired by swapping with random partners until none remain. Preserves every segment's
out-degree, in-degree, out-strength and outgoing sign mix; n_in is recomputed. Seeds
`G1b-structure-shuffle-{k}`, **k = 0..19 (20 shuffles; 5 is the floor if the machine cannot finish 20)**.
Sources and readouts keep their segment ids. Reported: z = (real - mean) / sd and the two-sided empirical
p = (1 + #{|null - null mean| >= |real - null mean|}) / (n + 1).

**Null B (sensitivity row, not a verdict item).** Side-class shuffle: post ends permuted only among pairs of
the same (pre side, post side) class, with repairs drawn from the same class. A segment's side is the sign of
its synapse centroid (`cells.parquet` cx, cy) against the fitted midline. It keeps each segment's number
of ipsilateral and contralateral partners, so it asks whether hops 2-3 carry side information beyond
"cells mostly connect within their own side". Seeds `G1b-structure-sideshuffle-{k}`, k = 0..9.

**Why Null B exists.** A degree-preserving shuffle destroys spatial embedding. Real tectal outputs stay
local, so anything ipsilateral will beat Null A at short range. That alone is not "the wiring routes".

**Positive controls (must succeed, or the verdict is INVALID: the instrument would be blind).**
- PC1 (real data): SI of lobe -> the two lobes' own tectal cells at hop 1 (targets = left-lobe and
  right-lobe tectal cells) must be >= 0.5 and above every Null-A shuffle.
- PC2 (real data): at hop 1, SEL between tectal target groups "loom zone" (the top 10% of loom-patch drive)
  and "prey zone" (the top 10% of prey-patch drive), same eye, must be >= 0.1, above every Null-A shuffle, z >= 3.
- PC3 (pytest, synthetic): a planted left-source -> relay -> right-target route gives SI = -1 exactly;
  a multiplicative no-routing graph with lobe and target size biases gives SI = 0; shuffles of the planted
  graph scatter around 0.
- Pipeline check: the hop-1 direct synapses from each lobe must reproduce task R's `readouts.json`
  counts (Mauthner 4 ipsi / 0 contra, nIII dorsal 16 / 3, nMLF 3 / 1, forward 18 / 0, turning 2 / 0).

**Verdict rule (fixed now).** On the primary statistic (cumulative excitatory, hops 1-3):
- A readout **separates sides** when |SI| >= 0.10, |SI| exceeds every Null-A shuffle's |SI|, and |z| >= 3.
- A contrast **separates stimuli** when the eye-mean |SEL| >= 0.10, exceeds every Null-A shuffle's |SEL|,
  and |z| >= 3.
- **CARRIES:** side separation for at least 2 of {M-system, turning, strike} AND stimulus separation for
  at least 1 of {E vs strike, E vs turning}.
- **WEAK:** at least one separation, but not CARRIES.
- **ABSENT:** no separation (no better than shuffled; learning on it would be a lookup table).
- A second, separate line says whether each separation has the predicted sign. A strong ipsilateral
  separation for the M-system is information carried in the wrong direction, not absence.
- Null B is reported beside it: if a separation beats Null A but not Null B, the note says the side
  information is first-order laterality.

## 1. What exists

| File | What |
|---|---|
| `brain/fishbrain/structure.py` | `Wiring` (input-fraction matrices, walks), `side_index`, `shuffle_post` (Null A and Null B), `build_context` (lobes, readouts, patches, zones), `measure` (every number for one wiring), `verdict`, `run` / `one_shuffle` / `replay_check` / `assemble`, `tectal_output_summary` |
| `brain/data/structure/structure.json` | real values, Null-A and Null-B summaries per value, verdict, power, tectal-output summary, replay check (896 KB) |
| `brain/data/structure/shuffles/` | `real.json`, `A-00..A-19.json`, `B-00..B-09.json` (one per wiring, resumable) |
| `brain/tests/test_structure.py` | 21 tests (section 12) |

Rebuild: `cd launches/fishbrain/brain && .venv/bin/python -m fishbrain.structure run` (real wiring 13-27 s,
each shuffle 25-101 s on a shared Mac at load 9-20; the run of record took 02:34-02:58 EDT), then
`python -m fishbrain.structure replay` and `python -m fishbrain.structure assemble`.
Other lanes' code was only read: `brain.load_pairs`, `brain.index_of`, `brain.bfs_levels`, `brain.WIRING_CONTENT_HASH`,
`cells.load`, `cells.tectum_seg_ids`, `cells.tectum_map`, `cells.tectum_drive`, `cells.disk`, `cells.midline_y`,
`cells.never_stimulate`, `sim.make_bitgen`; data files `readouts.json`, `cells_identified.json`,
`cells.parquet`, `net/pairs_*.npy`, `net/pairs_meta.json`. **No file owned by another lane was written.**

## 2. How much of each readout the tectum can reach at all (task item 4)

Per readout (both sides pooled), from the real wiring. "Orphan" = the presynaptic segment has no input
synapse, so nothing can ever drive it. "<= k hops" = the share of the readout's input synapses whose
presynaptic segment lies within k - 1 steps of any tectal cell (BFS over every pair; the "exc" column
uses only pairs with a type-2 synapse and counts only type-2 inputs). Walk share = cumulative excitatory
input fraction from both lobes, per side.

| readout | input synapses | orphan share | from tectal cells (hop 1) | <= 2 hops | <= 3 hops | exc, <= 3 hops | walk share, left / right cells | Null A: orphan, <= 3 hops |
|---|---|---|---|---|---|---|---|---|
| M-system (6 cells) | 18,882 | 0.703 | 0.00026 | 0.0049 | 0.0259 | 0.0179 | 1.27e-4 / 4.05e-4 | 0.860, 0.0170 |
| Mauthner (2) | 10,022 | 0.681 | 0.00040 | 0.0025 | 0.0284 | 0.0232 | 3.46e-4 / 5.14e-4 | 0.860, 0.0172 |
| turning (55) | 39,315 | 0.747 | 0.00005 | 0.0029 | 0.0312 | 0.0174 | 1.59e-5 / 1.46e-5 | 0.860, 0.0169 |
| strike, nMLF (33) | 18,666 | 0.803 | 0.00021 | 0.0022 | 0.0069 | 0.0040 | 6.36e-5 / 3.51e-4 | 0.860, 0.0163 |
| nIII dorsal (209 with inputs) | 33,526 | 0.810 | 0.00057 | 0.0014 | 0.0035 | 0.0015 | 4.58e-3 / 2.65e-4 | 0.860, 0.0167 |
| forward (51) | 34,409 | 0.742 | 0.00052 | 0.0042 | 0.0262 | 0.0171 | 9.17e-4 / 2.31e-4 | 0.860, 0.0164 |

- **The ceiling.** 68-81% of every readout's input comes from orphan segments. Within 3 hops the tectum
  reaches the presynaptic partners of 0.35-3.1% of a readout's input synapses. Walks carry 1e-5 to 1e-3 of
  its input. This is the fragmentation G1 found, now measured per readout.
- **Real vs shuffled.** The real readouts get *less* of their input from orphans than shuffled ones
  (0.68-0.81 vs 0.86) and, except nIII dorsal and nMLF, more within 3 hops. Part of this gap is the shuffle,
  not the wiring: neither null preserves in-strength (how many synapses a pair carries into a cell).
- **Whole-brain reach.** Within 1 step of the tectum: 78,097 segments (Null A mean 93,319); within 2 steps:
  125,604 (Null A 201,708). Over excitatory pairs only: 40,412 and 48,062. Orphan segments: 6,863,202 of
  13,458,709.
- **Where the tectum's own output goes** (post hoc, section 9; `tectal_output_summary()`):
  - 8,893 of the 24,920 in-graph tectal cells have any output synapse, 95,169 in all (25,028 type 2).
  - 29.1% land on tectal cells of the same lobe, **0.0% on the other lobe**, 70.9% on non-tectal segments.
  - Of those, 99.6% are on the lobe's own side of the midline.
  - Their x positions (5th-95th percentile 44,535-58,591, median 51,735, in 8 nm voxels) sit inside the
    tectal somas' own x range (45,056-62,894), far anterior of the Mauthner cells (x ~70,400). They are
    local neuropil partners. The tectobulbar axons that would carry the signal to the hindbrain are, in
    Fish1, not attached to these somas.

## 3. Controls that must hold

- **Pipeline check (other code, other author): passed exactly.** Hop-1 synapses from each lobe onto each
  readout reproduce task R's `readouts.json`: Mauthner 4 ipsilateral / 0 contralateral, nMLF 3 / 1, turning
  2 / 0, nIII dorsal 16 / 3, forward 18 / 0. Test `test_pipeline_direct_contacts_reproduce_task_R`.
- **Independent recomputation: passed.** Hop-1 and hop-2 excitatory weights onto the M-system, recomputed
  with plain bincounts from the pair arrays, equal the stored values to 1e-5 relative. Mauthner-left's
  hop-1 weight from the left lobe is exactly 2/5,849 (two type-2 synapses out of 5,849 inputs).
- **PC1, lobe -> own-lobe tectal cells, hop 1: +1.000** (Null A -0.003 +- 0.072, max |null| 0.154, z 13.9).
  Required >= 0.5 and beyond every shuffle: **pass.** Null B reproduces it (+1.000), as it must, since it keeps sides.
- **PC2, loom zone vs prey zone inside the driven lobe, hop 1: +0.968** (left eye +0.968, right eye +0.967;
  Null A -0.034 +- 0.207, z 4.8, beyond all 20; Null B +0.014 +- 0.248, z 3.8, beyond all 10). Required
  >= 0.1, beyond every shuffle, z >= 3: **pass.** Zones are the top 10% of each patch's drive among the lobe's
  cells with inputs (1,242 and 1,119 cells; no overlap between the two zones).
  - So the tectum's local wiring is topographic beyond side alone. The indices can see structure where
    it exists.
- **Signed, same controls:** PC1 signed = -1.000 and PC2 signed = -0.843. The tectum's recurrent input to
  its own lobe is net inhibitory: 73.7% of tectal output synapses are type 1.
- **Shuffles:** 20 Null-A and 10 Null-B wirings, every one with 0 unrepaired pairs (2-3 and 4-6 repair
  rounds), in- and out-degree identical to the real wiring, and for Null B every pair's (pre side, post side)
  class unchanged. **Determinism, measured:** shuffle A-00 rebuilt from its seed in a fresh process
  (03:08 EDT, after a refactor of the runner) gave all 630 values identically (`replay_check.json`). Same
  machine only.

## 4. Left vs right (task item 1)

Side index SI (section 0): +1 = each lobe reaches only its own side's readout, -1 = only the other side's.
Excitatory, cumulative hops 1-3 (the primary), then hops 2+3 (post hoc, section 9) and single hops.
"beyond" = |real| above every Null-A |value|. Null B is centred near +1 by construction, so only its mean and
z are shown.

| readout | hops | SI real | Null A mean +- sd | z | p | beyond | Null B mean +- sd | z vs B |
|---|---|---|---|---|---|---|---|---|
| **M-system** | **1-3** | **+0.976** | -0.079 +- 0.409 | +2.6 | 0.048 | yes | +0.987 +- 0.012 | -0.9 |
| M-system | 2+3 | +0.839 | +0.052 +- 0.442 | +1.8 | 0.143 | no | +0.430 +- 0.273 | +1.5 |
| M-system | 1 / 2 / 3 | +1.000 / +0.837 / +0.939 | | +2.5 / +1.6 / +1.8 | | | | |
| **turning** | **1-3** | **+0.410** | -0.029 +- 0.340 | +1.3 | 0.190 | no | +0.901 +- 0.102 | -4.8 |
| turning | 2+3 | -0.021 | -0.128 +- 0.450 | +0.2 | 0.810 | no | +0.409 +- 0.397 | -1.1 |
| turning | 1 / 2 / 3 | undefined / -0.848 / +0.250 | | n/a / -1.6 / +0.6 | | | | |
| **strike (nMLF)** | **1-3** | **+0.335** | -0.129 +- 0.328 | +1.4 | 0.190 | no | +0.723 +- 0.207 | -1.9 |
| strike | 2+3 | -0.173 | +0.116 +- 0.488 | -0.6 | 0.524 | no | +0.483 +- 0.313 | -2.1 |
| Mauthner alone | 1-3 | +0.999 | -0.007 +- 0.429 | +2.3 | 0.048 | yes | +0.957 +- 0.071 | +0.6 |
| nIII dorsal | 1-3 | +0.998 | +0.004 +- 0.204 | +4.9 | 0.048 | yes | +0.997 +- 0.002 | +0.2 |
| forward | 1-3 | +0.990 | -0.068 +- 0.287 | +3.7 | 0.048 | yes | +0.961 +- 0.039 | +0.7 |
| turning, 5 um dilated | 1-3 | +0.484 | +0.031 +- 0.152 | +2.98 | 0.048 | yes | +0.860 +- 0.066 | -5.7 |
| strike, 5 um dilated | 1-3 | +0.844 | -0.035 +- 0.331 | +2.7 | 0.048 | yes | +0.833 +- 0.111 | +0.1 |

Magnitudes behind the primary row (mean input fraction per member, excitatory, hops 1-3; LL = left lobe ->
left readout, and so on):

| readout | LL | LR | RL | RR | of which hop 1 (LL / RR) |
|---|---|---|---|---|---|
| M-system | 1.17e-4 | 2.4e-8 | 9.9e-6 | 4.05e-4 | 1.14e-4 / 3.51e-4 |
| Mauthner | 3.45e-4 | 5e-8 | 3.5e-7 | 5.14e-4 | 3.42e-4 / 4.79e-4 |
| turning | 1.5e-5 | 1.04e-5 | 9.4e-7 | 4.2e-6 | 1.3e-5 / 0 |
| strike | 5.9e-5 | 1.09e-4 | 4.3e-6 | 2.43e-4 | 5.6e-5 / 2.25e-4 |

What this says:
- **No verdict readout separates.** The M-system is beyond all 20 shuffles, but z = 2.6 < 3. Turning and
  strike are inside the shuffle range.
- **The primary is dominated by hop 1, which is 5 synapses.** Hop 1 is 97% (left) and 87% (right) of the
  M-system's own-side cumulative weight. Its hop-1 contacts are 5 type-2 synapses, all ipsilateral: two from
  one left-lobe cell onto Mauthner-left and two from one right-lobe cell onto Mauthner-right (G1-gate s.2
  names both cells), plus one from the right lobe onto a right-side MiD cell (task R's table: MiD3cm).
- **The sign is ipsilateral wherever it is large, and first-order laterality explains it.** Null B, which
  keeps every cell's own-side and other-side partner counts, gives the same or a more ipsilateral value:
  M-system +0.987 (real +0.976), turning +0.901 (real +0.410), strike +0.723 (real +0.335).
- **A lead, not a result (sensitivity row):** turning and strike are *less* ipsilateral than laterality alone
  predicts (z -4.8 and -1.9 against Null B; p = 0.091, which is the floor with 10 shuffles). That is the
  direction a crossed route to the turning cells would push. It rests on 10 shuffles and tiny weights (1e-6
  to 1e-5).
- **Secondary rows that clear the bar** (8 of the excitatory SI rows; no SEL or stimulus-side row does).
  All are ipsilateral, and none differs from Null B, i.e. from laterality alone (z vs Null B 0.0 to 1.02):
  - nIII dorsal, hops 1-3 (z 4.9) and hop 1 (z 4.3);
  - forward, hops 1-3 (z 3.7) and hop 1 (z 3.8);
  - strike, hop 1 (z 3.2), and dilated strike, hop 1 (z 3.1);
  - dilated turning, hop 2 (+0.780, z 3.5) and hops 2+3 (+0.686, z 3.2; Null B +0.482).
  - Dilated turning at hops 1-3 gets z 2.98, just under. This agrees with task R's binomial test on direct contacts (nIII dorsal
  16 / 3, forward 18 / 0), reached here with a different null.
- **Signed** (secondary): M-system +0.989 (Null A -0.050 +- 0.384, z 2.7), turning +0.024, strike +0.407.
  Null B's signed M-system value is -0.957. Random same-side contacts would be mostly inhibitory, because
  74% of tectal output synapses are type 1. The real direct contacts onto Mauthner are all type 2. That
  is 4 synapses, so it is descriptive only.
- **Direct contacts are far fewer than chance.** Real ipsi / contra vs the Null-A mean: M-system 5 / 0 vs
  15.3 / 17.4; turning 2 / 0 vs 53.0 / 54.8; nMLF 3 / 1 vs 22.4 / 25.6; nIII dorsal 16 / 3 vs 49.8 / 47.7;
  forward 18 / 0 vs 44.3 / 46.2. Real tectal output stays local (section 2).

**Stimulus-side index (secondary):** does an eye's patch reach the readout on the stimulus side
(+1 = yes)? Loom -> M-system **-0.945**: a right-eye loom reaches the LEFT M-system more (Null A -0.160 +-
0.586, z -1.3). Prey -> turning -0.106, prey -> nMLF -0.021. Dunn et al. 2016 predicts +1 for the loom
(section 10). It is the same ipsilateral wiring seen from the eye: the right eye drives the left lobe.

## 5. Loom vs prey (task item 2)

Patches (section 0): loom = a 60-degree disk at azimuth +-90, elevation +30 (709 lattice points); prey = one
point at +-20, 0. `tectum_drive`'s Gaussian (sigma 15 degrees) never reaches exactly 0, so each patch gives
every cell of the driven lobe some drive. The patches differ in where the drive peaks: the loom's drive sums
to 396,962-437,439 with a maximum of 338.9, the prey's to 834-955 with a maximum of 1.0. Each patch's
weights are normalised within the patch, so only the shape of the drive over the lobe matters.

SEL: +1 = loom reaches escape more and prey reaches strike (or turning) more. Targets pooled over both sides.

| contrast | hops | SEL, eye mean | left eye / right eye | Null A mean +- sd | z | p | beyond | Null B mean +- sd |
|---|---|---|---|---|---|---|---|---|
| **escape vs strike** | **1-3** | **-0.247** | -0.500 / +0.005 | +0.070 +- 0.315 | -1.0 | 0.333 | no | -0.019 +- 0.316 |
| **escape vs turning** | **1-3** | **-0.533** | -0.985 / -0.080 | +0.077 +- 0.336 | -1.8 | 0.143 | no | -0.051 +- 0.394 |
| escape vs nIII dorsal | 1-3 | -0.334 | -0.634 / -0.035 | +0.131 +- 0.391 | -1.2 | 0.381 | no | -0.009 +- 0.490 |
| Mauthner vs strike | 1-3 | -0.017 | -0.010 / -0.024 | +0.098 +- 0.299 | -0.4 | 0.905 | no | +0.011 +- 0.368 |
| escape vs strike | 2+3 | -0.400 | -0.816 / +0.015 | +0.012 +- 0.322 | -1.3 | 0.238 | no | -0.047 +- 0.197 |
| escape vs turning | 2+3 | -0.359 | -0.901 / +0.184 | +0.047 +- 0.339 | -1.2 | 0.238 | no | +0.054 +- 0.298 |
| escape vs strike | 1 | -0.406 | -0.478 / -0.334 | +0.083 +- 0.332 | -1.5 | 0.095 | no | +0.042 +- 0.326 |

- **No contrast separates.** Every mean is inside the shuffle range. Signed values are the same story
  (escape vs strike -0.231, escape vs turning -0.598, z -0.5 and -1.5).
- **The sign is the opposite of the prediction**, on both contrasts and at every hop, and it comes mostly
  from the left eye. The two eyes disagree (-0.985 vs -0.080 for escape vs turning). A real stimulus code
  should be mirror-symmetric, so this asymmetry reads as a few idiosyncratic paths, not a code.
- Compare PC2 (+0.968, section 3): the same patches and the same index separate loom from prey cleanly
  one synapse into the tectum, and the separation is gone by the readouts.

## 6. Intermediate sets: do the two lobes share relays? (task item 1)

Excitatory edges. Hop-1 intermediates = cells a lobe contacts directly that reach the readout within 2 more
steps; hop-2 intermediates = cells at a lobe's second step that contact the readout directly. Tectal cells
and the readout's own members are excluded. Binary Jaccard, then weighted Jaccard (flow through each cell,
normalised per lobe).

| readout, side | hop | intermediates, left lobe / right lobe | shared | Jaccard | weighted | Null A mean Jaccard |
|---|---|---|---|---|---|---|
| M-system left | 1 | 13 / 9 | 0 | 0.000 | 0.000 | 0.090 |
| M-system left | 2 | 20 / 12 | 2 | 0.067 | 0.014 | 0.230 |
| M-system right | 1 | 8 / 15 | 0 | 0.000 | 0.000 | 0.045 |
| M-system right | 2 | 11 / 19 | 1 | 0.034 | 0.002 | 0.072 |
| turning left | 1 | 23 / 14 | 0 | 0.000 | 0.000 | 0.045 |
| turning left | 2 | 42 / 28 | 5 | 0.077 | 0.016 | 0.109 |
| turning right | 1 | 24 / 12 | 0 | 0.000 | 0.000 | 0.063 |
| turning right | 2 | 47 / 17 | 0 | 0.000 | 0.000 | 0.146 |
| strike left / right | 1 | 2 / 2 and 3 / 2 | 0 | 0.000 | 0.000 | 0.064 / 0.046 |
| strike left / right | 2 | 1 / 1 and 2 / 1 | 0 | 0.000 | 0.000 | 0.118 / 0.109 |
| nIII dorsal left / right | 1 | 10 / 1 and 2 / 2 | 0 | 0.000 | 0.000 | 0.050 / 0.041 |
| nIII dorsal left / right | 2 | 4 / 1 and 1 / 1 | 0 | 0.000 | 0.000 | 0.123 / 0.113 |

- Whole brain: the two lobes' first-step sets (7,735 and 7,761 non-tectal cells) share **4 cells**
  (Jaccard 0.00026; Null A 0.0149 +- 0.001, Null B 0.0005). At the second step: 7,356 and 6,165, sharing
  230 (0.0173; Null A 0.0773, Null B 0.0195).
- **The two lobes never share a first relay into any readout,** and share at most 5 second relays. That is
  locality again: Null B's Jaccards are as low as the real ones.
- **The channels are also tiny:** 1 to 47 relay cells per lobe per readout. The strike and nIII readouts
  are reached through 1 to 10 cells. None of the per-readout Jaccards differs from Null A (every |z| < 2).

## 7. The nulls, and what this test could detect (task item 3)

- **Null A** (task's control): 20 whole-brain degree-preserving shuffles, `G1b-structure-shuffle-0..19`. The
  minimum empirical two-sided p is 1/21 = 0.048. **Null B** (sensitivity): 10 side-class shuffles, minimum
  p 1/11 = 0.091.
- **Power, computed after the run (section 9):** an index lives in [-1, 1], so the largest |z| it can
  reach is |(+-1 - null mean) / null sd|.

  | item | predicted sign | Null A mean +- sd | largest |z| in the predicted direction | largest |z| the other way |
  |---|---|---|---|---|
  | M-system SI | - | -0.079 +- 0.409 | 2.25 | 2.64 |
  | turning SI | - | -0.029 +- 0.340 | 2.85 | 3.02 |
  | strike SI | - (Thiele) | -0.129 +- 0.328 | 2.66 | 3.45 |
  | escape vs strike SEL | + | +0.070 +- 0.315 | 2.95 | 3.40 |
  | escape vs turning SEL | + | +0.077 +- 0.336 | 2.74 | 3.20 |

  **No item could have met z >= 3 in the predicted direction.** The shuffled values spread so widely
  because in any wiring, real or shuffled, the tectum reaches each readout through a handful of synapses,
  so each index swings on a few contacts. The M-system could not reach 3 in either direction. I did not
  foresee this when I set the bar.
- **Sensitivity of the call to the bar (run after the result):** with z >= 2 instead of 3 the call would be
  **WEAK**, with one separation: the M-system, ipsilateral (+0.976, the opposite of the prediction).
- **Sampling:** a standard deviation estimated from 20 shuffles carries about 16% relative error, so the
  M-system's z of 2.6 could land either side of 3 under other seeds. That would not change what it is:
  ipsilateral, and reproduced by Null B.
- **In-strength is not preserved** by either null (the shuffle moves pairs, each with its own synapse count).
  That affects the orphan-share and reach comparisons in section 2, not the indices' logic.

## 8. Verdict (task item 5)

**ABSENT.** Pre-registered rule, primary statistic (cumulative excitatory, hops 1-3): 0 of 3 readouts
separate sides, and 0 of 2 contrasts separate stimuli. Both positive controls pass, so the call is valid.

- **Direction, as a separate line:**
  - M-system: +0.976 measured, contralateral (-) predicted. No match.
  - Turning: +0.410 measured, - predicted. No match.
  - Strike: +0.335. Its sign matches Gahtan's anatomy (+) and not Thiele-plus-strike-toward-prey (-),
    but it is no more ipsilateral than laterality alone gives (Null B +0.72).
  - Escape vs strike -0.247 and escape vs turning -0.533 measured, + predicted. No match.
  - None of these separates from the shuffles.
- **What ABSENT means here.** The measured wiring from tectal somas to the readouts is indistinguishable from
  shuffled wiring, and the test could not have shown the predicted routing even if it existed (section 7).
  The load-bearing result is the magnitude.
  - The tectum can account for about 0.001-0.1% of a readout's input through walks of 3 steps or fewer.
  - 68-81% of that input comes from orphan fragments.
  - The only structure that survives is local and ipsilateral.
- **For learning:** a learned weight on these paths would be fitting a few idiosyncratic contacts,
  effectively a lookup table. The tectum does carry side and stimulus structure (PC1, PC2), but it stops
  at the tectum. The measured tectum -> hindbrain step is missing, not mis-signed.

## 9. What changed after the pre-registration (for the honesty page)

1. **02:34 EDT, hops 2+3 row added (post hoc).** After the first real run (02:32) showed the cumulative
   weight was almost all hop 1, I added "h23" (hops 2 + 3) to every index, before any shuffle ran. It is a
   secondary row and is not in the verdict.
2. **About 02:37 EDT, tectal-output breakdown (post hoc, descriptive):** where the tectum's own 95,169 output
   synapses land (section 2). It is now `tectal_output_summary()` and is written into `structure.json`.
3. **About 02:36-02:38 EDT, literature read after the pre-registration:**
   - Dunn 2016's laterality quotes, re-read to check the prediction. They confirm it.
   - Helmbrecht 2018, search summary only: tectal escape and approach signals leave through *uncrossed*
     (ipsilateral) tectobulbar tracts.
   - The reading "hop-1 ipsilateral is what an uncrossed tract predicts, so the crossing has to happen
     downstream" is post hoc. It was not a pre-registered prediction.
4. **About 03:00 EDT, power analysis:** section 7's table. It uses the stored Null-A mean and sd; the formula
   is in `structure.json` -> `power`.
5. **About 03:02 EDT, the z >= 2 sensitivity check** (the call would be WEAK). Not the rule.
6. **03:04-03:08 EDT, runner refactor plus replay:**
   - The shuffle step became `one_shuffle()`, and `replay_check()` was added.
   - Null B's rows (both the per-value `nullB` summaries and `nullB_sensitivity`) no longer carry a
     "separates" or "beyond all" flag, because Null B is centred near +1 and that flag assumes a centre
     at 0. Removed from the per-value rows at 03:15 EDT (re-assembled; the test checks both).
   - The stored shuffles were not recomputed. The replay of A-00 with the refactored code matched all 630 values.
7. **A test threshold loosened:** the synthetic "planted route beats its shuffles" test first asserted a
   real SI below -0.8. The measured value on that synthetic graph was -0.654, because the noise pairs dilute
   the route. The assertion is now below -0.5; beyond every shuffle and z below -3 are still required.

Nothing in section 0 was changed. No threshold, population, patch, null or seed was changed after any
real value was seen.

## 10. Literature, and what the data matches

- **Dunn et al. 2016** (Neuron 89:613, PMC4742414; quotes returned verbatim by a summarising fetch,
  2026-09-26):
  - "In 33 fish, stimuli in the right visual field consistently evoked escapes to the left, and vice versa."
  - "Only escape responses contralateral to the ablated M-system (M-cell, MiD2, and MiD3) were perturbed."
  - So a right-field loom needs the right M-system, which the left lobe must drive. The prediction is
    `brain.MAUTHNER_PREDICTION = "ipsilateral_to_stimulus"`. On the pathway, the paper leaves "the possibility
    of either a direct or indirect path from the OT", and cites a direct OT -> M-cell ventral-dendrite
    pathway in adult goldfish.
  - **Data:** the only direct tectum -> Mauthner contacts are ipsilateral (2 + 2 synapses). The loom's
    stimulus-side index is -0.945 (the wrong side). Neither is beyond the shuffles.
- **Helmbrecht et al. 2018** (Neuron 100:1429; search summary only, secondary): "two spatially segregated and
  uncrossed descending axon tracts selectively transmit approach and escape signals to the hindbrain". In the
  ipsilateral tectobulbar tract, medial axons were tuned to threats and lateral axons to prey.
  - **Reading (post hoc):** an ipsilateral first step is what real anatomy predicts, and the crossing to the
    contralateral M-cell must come later.
  - In real larvae, loom and prey travel in separate tract bundles. Fish1's tectal somas show that
    separation locally (PC2), but the tract axons are not attached to them, so it cannot reach the readouts.
- **Huang et al. 2013** (via G1-gate s.4) and **Gahtan 2005 / Thiele 2014 / Lau 2025** (via G1b-readouts s.5),
  not re-read today: the turning and strike predictions of section 0.

## 11. Choices we made (for the honesty page)

Everything here is ours unless a source is named.
1. **Path weight = input fraction:** a pair's synapses over the post cell's total input synapses, multiplied
   along walks, walks of exactly k steps summed (cells may repeat). It is a linear rate model with unit gains.
2. **Excitatory = type-2 synapses; signed = type 2 minus type 1** (the G1 mapping, measured). No pair is
   dropped.
3. **Primary = cumulative hops 1-3, excitatory.** Per hop, hops 2+3 (post hoc) and signed are secondary.
4. **Population weight = mean over members with any input.** M-system = Mauthner + MiD2cm + MiD3cm
   (task_criteria picks); strike = nMLF; turning = RoV3 + MiV1 + MiV2; strict atlas sets (dilated as rows).
5. **Sources = every in-graph tectal cell** (24,920), not `brain.stimulated_set` (which drops 7 cells whose
   direct contacts are signal here).
6. **Side index and selectivity index**, both 1/2 [(a - b)/(|a| + |b|) + (d - c)/(|c| + |d|)], exactly 0
   when the weights factorise (source x target).
7. **Patches:** loom `cells.disk(+-90, +30, 30)`, prey a single point at (+-20, 0). Drive is
   `cells.tectum_drive` unclipped (the eye model's 150 Hz clip is not applied). Retinotopy is B1's
   rank-uniform model.
8. **PC2 zones:** the top 10% of a patch's drive within the driven lobe (cells with inputs).
9. **Intermediates:** excitatory edges; tectal cells and the readout's members excluded; flow = forward
   weight x backward weight.
10. **Reachability:** BFS from all tectal cells, any pair / pairs with a type-2 synapse; orphan = no input
    synapse and not tectal.
11. **Null A:** `sim.shuffled_copy`'s algorithm on the raw pairs (pairs keep pre, t1, t2), 20 seeds.
    **Null B:** the same within (pre side, post side) classes. A segment's side is its synapse centroid
    against the fitted midline. 10 seeds.
12. **Verdict thresholds:** |index| >= 0.10, beyond every Null-A shuffle, |z| >= 3; CARRIES = 2 of 3 side
    readouts + 1 of 2 contrasts. PC1 >= 0.5, PC2 >= 0.1 with z >= 3. All fixed at 02:27 EDT.

## 12. Tests

`cd launches/fishbrain/brain && .venv/bin/python -m pytest -q tests/test_structure.py` (bare, no pipe):
**exit code 0, 21 passed in 1.62 s** (2026-09-26, run of record 03:15 EDT).

- Synthetic, no data (10):
  - SI = 0 under source x target for three bias settings, including 11,799 vs 13,121 lobes and a 543 vs 265
    target;
  - extremes, signs and undefined cases;
  - a planted crossed route reads exactly -1 and an ipsilateral one +1 (PC3), through the module's own walk;
  - the walk equals explicit input-fraction products;
  - a noisy planted route (-0.654) beats 20 shuffles that centre on 0 (|mean| < 0.3, z < -3);
  - SEL reads a planted type route;
  - the shuffle keeps in-degree, leaves no duplicates or self-loops and is seeded;
  - the side-class shuffle keeps every pair's class.
- Data (11):
  - the direct contacts reproduce task R;
  - PC1 and PC2 pass and the call is not INVALID;
  - 20 + 10 shuffles, each with preserved degrees and 0 unrepaired pairs;
  - the verdict and every z recompute from the stored numbers;
  - reach fractions are bounded and monotone, and the orphan share is above 0.5;
  - hop-1 and hop-2 M-system weights recomputed with bincounts (Mauthner-left hop 1 = 2/5,849);
  - the tectal output is local;
  - the A-00 replay is identical;
  - Null B never claims a separation;
  - the power limit is recorded;
  - 15 key numbers and the call "ABSENT" are pinned.
- **The tests fail when they should** (each change reverted after):
  - flipping the sign of the index's second term: 7 failed;
  - disabling the shuffle's duplicate repair: 2 failed;
  - normalising by the pre cell's input instead of the post cell's: 1 failed;
  - swapping the lobes in one stored M-system weight and adding a fake contralateral contact: 2 failed;
  - loosening the bar to z >= 2: 2 failed (the call becomes WEAK).

## 13. Limits and open decisions

- **The test's power, stated plainly (section 7):** the pre-registered null comparison could not have
  returned the predicted routing. Read the verdict through the magnitudes in section 2, not through the
  z-scores alone.
- **Tectal somata are the sources**, as in G1. Retinal axon terminals in the neuropil (B1 s.8.7) or
  reattached tectobulbar axons (task P, the repair probe) are different sources and could change everything
  here. That is the fork this result points to: **the missing piece is the tectum -> hindbrain step, which
  Fish1 has but does not attach.** A disclosed bridge there, or recovered axons, is what would make the
  readouts reachable. G1b's bridge rule says it must be minimal and carry its measured accuracy.
- **Retinotopy is a model** (B1's rank-uniform map). PC2 shows the model's zones are wired locally to
  themselves; it does not validate the map against real receptive fields.
- **Left/right rests on B1's axis call.** A mirror error would flip every side sign consistently, so
  "ipsilateral" would stay ipsilateral.
- **More shuffles** (for example 100) would tighten the z-scores, but they cannot lift the power ceiling
  in section 7, which comes from the width of the null. About 1 min each on this Mac. Not run.
- **Nothing here is simulated.** The signal claims are about the wiring's linear reach. A spiking model with
  thresholds would carry less, not more (G1: silent past hop 1 at Shiu's weights).
- **`data/structure/` is under the git-ignored `data/`** (4.0 MB). It reaches the repo only if it is
  force-added; a rebuild takes about 25 minutes on this Mac, from files already on disk.

## Sources

- Brain of record and cell identities: as in G1-pull.md, G1-cells.md, G1b-readouts.md.
- Dunn TW, Gebhardt C, Naumann EA, Riegler C, Ahrens MB, Engert F, Del Bene F (2016) Neuron 89:613-628,
  doi 10.1016/j.neuron.2015.12.021, PMC4742414 (pmc.ncbi.nlm.nih.gov, summarising fetch, quotes verbatim).
- Helmbrecht TO, Dal Maschio M, Donovan JC, Koutsouli S, Baier H (2018) Topography of a visuomotor
  transformation. Neuron 100:1429-1445 (search summary only; cell.com/neuron/fulltext/S0896-6273(18)30909-7).
- Citation required by the Fish1 data policy: Petkova, Januszewski et al. (2025), "A connectomic resource
  for neural cataloguing and circuit dissection of the larval zebrafish brain", bioRxiv.
