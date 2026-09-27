---
publication_copy: edited-for-publication copy of launches/fishbrain/evidence/G1c-measure.md at commit 74eb5f3f8. Local paths made repo-relative; machine-owner, internal coordination, account and credential details removed. No number, hash, time, method, result or correction was changed; publication notes are marked [Publication note].
title: G1c measure, what is real before bridging (fragment origins, reach from the retina's endings, the eye, predicted decision-path share)
created: 2026-09-26
status: done (measure task). Pre-registration (s.0) was written at 03:52 EDT before any number; results in s.1-6.
evidence_status: measured unless marked "expected", "estimate", "post hoc" or "choice". Graph and geometry analysis only; nothing was simulated.
verification_scope: >
  Measured 2026-09-26 03:54-04:35 EDT on the analysis machine from the pinned brain of record (data/net/pairs_*.npy, wiring hash
  09a71038...25af2; data/synapses.parquet, 29,474,316 synapses with positions), the Fish1 atlas mece0/1/2 at 4096 nm
  (data/ref), B1's cells_identified.json, task R's readouts.json, the soma csv and the cached CAVE somas v709. One
  read-only network use: 20 public agglomeration meshes (seg_241003_agg241003, lod 3, no login) for the mesh row. No
  literature was re-read; biology is cited through G1-cells, G1-gate, G1b-readouts and G1b-structure.
files: G1c measure task. brain/fishbrain/origins.py, brain/data/g1c/, brain/tests/test_origins.py, this note.
outputs: brain/data/g1c/origins.json (step 1, controls, eye), oa_fragments.json (the observed-origin fragments in detail),
  meshes.json, afferents_primary.npz / afferents_near_zero.npz, shuffles/ (real, real_near_zero, A-00..A-19, B-*),
  reach.json (step 2 assembled), share.json (step 4), scan_acc.npz, *.log. All under the git-ignored data/.
  Rebuild: `cd launches/fishbrain/brain && PYTHONPATH=. .venv/bin/python -m fishbrain.origins all` (about 25 min here),
  then `... origins shuffles 20 10` and `... origins assemble` for the Null-B rows.
---
# G1c measure: what is real before we bridge

**TLDR.** Under 1% of the drive that makes the escape decision can reach the escaping cell over measured Fish1 synapses
from the eye's input. The bridge has to supply the rest, and it has to supply the crossing. What the real map does hold is a
small, uncrossed tectum-to-escape-cell pathway. It is the right shape, and it is the bridge's best anchor.

1. **Fragment origins (s.1).**
   - **Observed tectal origin (O-a).** 24 fragments on the M-system (16 on the two Mauthner cells) have an observed end
     in the tectum. They carry 0.5-0.6% of the M-system's input synapses.
   - **On the Mauthner they land only on the ventral dendrite**, the visual input site: 16 of 16 fragments (post hoc,
     p = 1e-5 and 8e-7 against each side's base rate). That is 1.7% and 2.4% of the ventral dendrite's excitatory input.
   - **Uncrossed.** Every O-a synapse but one comes from the tectal lobe on the cell's own side. The one crossed synapse
     belongs to a soma-bearing, non-tectal cell.
   - **The rest has no observed tectal end.** On the M-system, 24-30% of the input runs at least 20 um toward the
     tectum and stops short of it (median 59-69 um; O-b); meshes agree for 19 of 20 of the heaviest. Caudal-hindbrain cells get 15%
     of such input too, so O-b is not specific. 51-61% is local or terminal crumbs (O-c), 6-10% comes from somas
     elsewhere in the hindbrain (O-d1), and 0 synapses reach a ganglion (the VIIIth nerve is never seen).
2. **Reach from the retina's endings (s.2).**
   - **The set.** 1,284,858 zero-input fragments put at least 80% of their outputs in the tectal neuropil (4.66 M synapses).
   - **They line up with the tectal map:** PC1 +1.00, PC2 +0.62 (z 7.9).
   - **They reach the decision cells less than the tectal somas did:** 0.42-1.12% of input within 3 hops, against
     G1b's 0.69-3.12%. They make 0 direct contacts, where shuffles give 16%.
   - Pre-registered call: **WEAK.** The M-system separates the sides, ipsilaterally (+0.97, z 18.9). Turning separates
     contralaterally (-0.26, z -6.6), the predicted sign, but only in the strict population, and the weights are 1e-7
     to 1e-6. Against a side-keeping null (Null B, s.6), every readout is less ipsilateral than laterality alone
     predicts. That is a lead, not a route.
3. **The eye (s.3).** 0 of 29,474,316 synapses and 0 CAVE somas lie in the atlas Retina. The eye is not in the
   synapse data, so the eye model's input is always *chosen*.
4. **Predicted decision-path share (s.4, estimates).**
   - **The escape's correct pairing** (a lobe drives the contralateral M-system): 0.00003%-0.6% by the pre-registered
     range. A right-eye threat has **zero** measured synapses within 3 steps onto the escaping (right) Mauthner.
   - **Without the crossing** (ipsilateral): up to 2.8%. Turning: up to 1.8%. Strike: up to 3.8%.
   - **Bridge-independent ceiling (expected):** at most 4 of the ~25 coincident synapses the escaping Mauthner needs could
     arrive over measured paths.
   - **The brief's pre-registered disclosure sentence applies:** "Which way the fish flees is set by the published
     pathway model, not by the measured wiring."
5. **For the bridge builder (s.5).**
   - The 24 O-a fragments are bridge (a)'s inventory, with their tectal ends and columns (`oa_fragments.json`).
   - Their ends sit in the ventral SPV. **B1's map puts that in the lower visual field**, where G1b's loom patch drives
     them at a median 4.5e-6 of its peak. Linking them by position means the loom will not reach them unless the map
     or the stimulus is revisited.

`pytest -q tests/test_origins.py`: **exit 0, 19 passed** (04:35 EDT, s.8). Six deliberate breakages each made it fail.

## 0. Pre-registration (written 2026-09-26 03:52 EDT, before any class, reach, afferent set or share was computed)

**Orientation reads done before this was written (no result of this task):**
- The G1b notes, the brief's G1c section, and the code of `brain.py`, `structure.py`, `cells.py`, `readouts.py`.
- The atlas region names (`segprops_mece0..3`) and region bounding boxes (`atlas_regions_4096.json`). There is **no
  tectobulbar-tract label** in any atlas level. The atlas `Retina` region (mece0 1) has 38,184 voxels at 4 um, and its box
  spans both eyes (x 102-389 um, y 16-496 um).
- The position histogram of the two Mauthner cells' input synapses relative to their somas (used only to set the
  compartment rule below). Left: 5,849 inputs, right: 4,173. Soma about 40 um from the midline. Inputs cluster at
  (i) 50-95 um lateral of the midline and within 20 um of the soma depth (the lateral dendrite), (ii) 20-45 um lateral
  and 20-50 um ventral, and (iii) 30-65 um lateral and 60-100 um ventral (the ventral dendrite, which B1's mesh showed
  reaching 85-91 um ventral).

**Names.** To avoid a clash with the brief's bridge classes (a)/(b)/(c), the measured origin classes here are
**O-a, O-b, O-c, O-d**. O-a is the candidate pool for bridge (a), O-b for bridge (b). O-c and O-d are not bridge classes.

**Graph and data.** The brain of record as in G1b: `data/net/pairs_*.npy` (24,921,936 pairs, wiring hash
09a71038...25af2), per-synapse positions from `data/synapses.parquet` (x, y, z in 8 x 8 x 30 nm voxels). Type 2 =
excitatory. Readouts from `data/readouts.json` strict sets: M-system per side = Mauthner + MiD2cm + MiD3cm
(task_criteria picks); turning = RoV3 + MiV1 + MiV2; strike = nMLF.

**Atlas lookup (choice).** mece0/1/2 at the 4096 nm mip, whole volume (547 x 127 x 68). A point (x, y, z) in 8 nm voxels
maps to voxel (floor(x / 512), floor(y / 512), floor(z / 128)), clipped to the array. **Validity check (must pass before
any class is used):** at least 80% of B1's tectal somas land in mece1 21 (Tectum), and the left Mauthner soma lands in
mece2 48 or within one voxel of it.

**Tectum mask and distance.** T = voxels with mece2 in {17 (SPV), 18 (Neuropil)}. d_T = Euclidean distance to T in um
(scipy `distance_transform_edt`, sampling 4.096 x 4.096 x 3.84 um), 0 inside T. Each synapse gets the d_T of its voxel.

**Compartments of the M-system cells (choice, from the orientation histogram above).** Per input synapse, relative to the
cell's soma (Mauthner: B1 `pos`; MiD2cm/MiD3cm: readouts.json candidate `pos`): dz = (z - z_soma) x 0.030 um,
lat = |y - midline(x)| x 0.008 um. Applied in this order:
1. **ventral dendrite**: dz >= 20 um;
2. **lateral dendrite (VIIIth nerve zone)**: lat >= 50 um for the Mauthner cells (B1's threshold), lat >= soma lat + 10 um for
   MiD2cm / MiD3cm;
3. **soma / proximal**: the rest.

**Fragments.** Every segment with at least one synapse onto a readout member. Its **cloud** = every synapse where it is
pre or post in synapses.parquet. Its **contacts** = its synapses onto that readout.

**Origin classes, first match wins (choice; thresholds fixed now):**
1. **O-a0, tectal cell**: the segment is one of B1's in-graph tectal (soma-bearing) segments.
2. **O-a1, observed tectal end**: at least 2 cloud synapses have d_T = 0. **Lobe tag** = the side (fitted midline) of the
   majority of those tectal synapses; reported as ipsilateral or contralateral to the readout.
3. **O-d1, soma elsewhere**: the segment holds a soma (`agglomerated_segments_and_soma_ids.csv`) that is not tectal. Its
   origin is that soma; the soma's atlas region is reported.
4. **O-d2, ganglion**: at least 2 cloud synapses in mece0 3 (Ganglia: statoacoustic, lateral line, trigeminal...).
5. **O-b, tract-only toward the tectum**: progress = median d_T of its contacts - min d_T of its cloud >= 20 um. The cloud
   runs at least 20 um from the readout toward the tectum, but no tectal end is observed. **This is an envelope:** other
   descending axons (for example nMLF axons in the MLF) also move toward the tectum and are counted here.
6. **O-d3, other far origin**: the cloud reaches >= 30 um from its contacts' centroid (max distance), not toward the
   tectum. The atlas region of its farthest synapse is reported.
7. **O-c, local or terminal**: everything else.
A flag (not a class) marks fragments with min d_T <= 8 um but fewer than 2 synapses inside T ("edge").
Any O-a fragment that is also in the step-2 afferent set is tagged **eye path hop 1** (a retinal ending onto a readout
dendrite in the neuropil is a measured eye path, not a bridge).

**Weights.** Per readout x side (and per M-system compartment): count of fragments, and **input-synapse weight** = the
class's input synapses onto the population as a share of all its input synapses, pooled over members. Excitatory (type 2)
and all types both. Per-member means in the JSON.

**Controls for step 1 (fixed now):**
- **Pipeline check:** O-a0 contacts must reproduce G1b/task R's direct tectal contacts exactly (M-system 5 ipsi / 0 contra;
  Mauthner 4 / 0; nMLF 3 / 1; turning 2 / 0).
- **Positive control, real data (the rule fires where it should):** 60 B1 tectal cells with inputs (seed `G1c-pos-control`)
  as a pseudo-readout. Expected: at least 50% of their input synapses come from O-a fragments.
- **Negative control, real data:** 60 soma-bearing segments whose soma sits in mece1 31 (Caudal Hindbrain), with at least 20
  input synapses (seed `G1c-neg-control`). Expected: O-a share below 1%.
- **Is there anything to find (brain-wide):** the number of segments with >= 2 synapses at d_T = 0 and >= 2 synapses in
  mece0 6 (Hindbrain) posterior of x = 540 um (x >= 67,500 voxels), their synapses, and how many touch any readout.
- **The instrument is blind to tracts (sensitivity row, not a reclassification):** an axon carries almost no synapses
  inside a tract, so a synapse cloud cannot see a tectobulbar run. For the 20 heaviest ventral-dendrite input fragments of
  the Mauthner cells (10 per side, by excitatory contacts on the ventral dendrite), the public agglomeration mesh (lod 3,
  `seg_241003_agg241003`) gives min d_T and extent over its vertices. Reported: how many meshes reach T where their cloud
  did not.
- Synthetic tests (pytest): a fragment spanning tectum -> hindbrain classifies O-a1 with the right lobe tag; one that stops
  20+ um short but runs toward the tectum is O-b; a local blob is O-c; a soma segment is O-d1.

**Step 2, the retinal-afferent set (rule fixed now).** A segment is a **retinal afferent** when:
n_in = 0 (cells.parquet), n_out >= 1, it holds no soma (csv) and is not a B1 tectal segment, and at least 80% of its
output synapses have mece2 18 (tectal neuropil). **Sensitivity set:** n_in <= 0.1 x n_out instead of n_in = 0.
It also captures non-retinal orphan axon endings in the neuropil; retinal ganglion cell axons are one part of it.
- **Placement:** side = its output centroid (cells.parquet out_cx, out_cy) against the fitted midline; dropped within 3 um
  of it (B1's rule). Its preferred field comes from B1's own code: `cells.TectumMap` built on the afferents' output
  centroids (rank-uniform lobe coordinates, sigma 15 degrees). An eye drives the opposite lobe's afferents.
- **Metrics:** `structure.measure` unchanged, with a Context whose sources are the afferents per lobe; the patches are the
  G1b loom and prey patches driving the afferents; PC1 targets and PC2 zones are B1's tectal cells, exactly as in G1b.
  PC1 and PC2 therefore test whether the afferent map lines up with B1's tectal-cell map.
- **Nulls:** 20 Null-A shuffles with G1b's seeds `G1b-structure-shuffle-{k}` (identical null wirings, so the two source
  sets are compared on the same shuffles). Null B (10 side-class shuffles, G1b's seeds) only if machine time allows; it is
  never in the verdict.
- **Verdict rule:** G1b's, unchanged (|index| >= 0.10, beyond every Null-A shuffle, |z| >= 3; CARRIES / WEAK / ABSENT;
  INVALID if PC1 or PC2 fails), so the two runs compare one to one.
- **Comparison:** reach <= 1/2/3 hops, orphan share and walk share per readout, beside G1b's tectal-soma numbers
  (0.35-3.1% within 3 hops).

**Step 3, the eye.** Counts in mece0 1 (Retina): synapses; segments with any synapse there; of those, segments with output
synapses at d_T = 0 (and their synapse counts); CAVE somas in the region; soma-bearing segments (csv) with the soma there,
and of those, the ones with outputs in T. Zero synapses would mean the eye was not synapse-detected.

**Step 4, predicted decision-path share (estimates from the graph; no simulation).** The brief's metric counts only drive
received in the decision window: silent inputs add nothing, and a bridged link passes m = 0. So the share is measured
drive / (measured drive + bridge drive), not measured / all inputs. For each deciding readout x side and each stimulated
lobe:
- **M** = the readout's presynaptic segments within 2 excitatory steps (pairs with t2 > 0) of that lobe's afferents
  (so the afferents themselves are hop 1). **A** = O-a fragments with that lobe's tag, not in M. **B** = all O-b fragments
  on the readout, not in M or A (their lobe is unknown, so all are counted: an envelope).
- **Synapse view (equal activity per synapse, generous to measured paths):** LB_syn = exc(M) / exc(M + A + B);
  UB_syn = exc(M + A) / exc(M + A + B).
- **Walk view (linear, unit gains, dilution; pessimistic for measured paths):** x_M = cumulative excitatory walk,
  hops 1-3, from the lobe's afferents (G1b's measure); q_A, q_B = excitatory input fractions from A and B at unit
  activity. LB_walk = x_M / (x_M + q_A + q_B); UB_walk = (x_M + q_A) / (x_M + q_A + q_B).
- **All-inputs view (G1b-comparable):** exc(M) / all excitatory inputs, and exc(M + A) / all excitatory inputs.
- **Under the strict pre-registered rule, the O-a link is itself a bridge** unless the fragment has measured inputs
  from the afferents. The UB credits it because its origin is observed; the LB does not.
- **Pairings:** the behaviourally correct one is primary. Escape: a lobe drives the **contralateral** M-system (Dunn 2016,
  `brain.MAUTHNER_PREDICTION`). Turning: contralateral (G1b's prediction). Strike (nMLF): the two predictions conflict,
  so both are reported. The ipsilateral pairing is reported for every readout (the "bridge without the crossing" case).
- Any bridge class (c) link (the literature relays and crossing) adds drive that none of these count, and lowers every
  share. The simulation measures it.
- **The headline line** uses the range LB_walk to UB_syn of the escape's correct pairing (the M-system cell that
  must fire), with turning and strike beside it.

**Tests.** `tests/test_origins.py`, synthetic and data, with each check broken on purpose once. Run bare; the exit
code is reported.

## 1. Fragment origins (task item 1)

**Validity check first (measured, passed).** Under the pre-registered lookup, 97.7% of B1's 44,435 tectal somas land in
mece1 Tectum (98.5% in the mask {17, 18}). The left Mauthner soma lands in mece2 48. The right one lands in label 0 with 48
one voxel away, as B1 found at 512 nm (70% of its soma in the region).

**Pipeline check (other code, other author): passed exactly.** The O-a0 contacts (B1 tectal cells) reproduce task R's
direct tectal contacts: M-system 5 ipsi / 0 contra, Mauthner 4 / 0, nMLF 3 / 1, turning 2 / 0. Every fragment's contact
rows in synapses.parquet equal its synapse count in the pair arrays (asserted for all 46,991 watched fragments).

**Classes per readout and side** (share of all input synapses / share of excitatory input synapses; measured):

| readout, side | fragments | input syn (exc) | O-a0 | O-a1 | O-b | O-d1 | O-d2 | O-d3 | O-c |
|---|---|---|---|---|---|---|---|---|---|
| M-system L | 4,069 | 10,232 (7,436) | 0.02% / 0.03% | 0.5% / 0.6% | 28.4% / 29.9% | 9.9% / 10.6% | 0 | 10.4% / 9.9% | 50.8% / 49.0% |
| M-system R | 3,875 | 8,650 (6,514) | 0.03% / 0.05% | 0.6% / 0.6% | 23.6% / 24.8% | 5.7% / 6.3% | 0 | 8.6% / 8.3% | 61.5% / 59.9% |
| Mauthner L | 2,047 | 5,849 (4,555) | 0.03% / 0.04% | 0.3% / 0.4% | 27.6% / 29.5% | 13.9% / 14.5% | 0 | 12.3% / 11.9% | 45.8% / 43.7% |
| Mauthner R | 1,675 | 4,173 (3,544) | 0.05% / 0.06% | 0.3% / 0.3% | 23.8% / 24.4% | 7.0% / 7.5% | 0 | 10.2% / 10.2% | 58.7% / 57.5% |
| turning L | 11,520 | 21,840 (12,393) | 0.01% / 0.01% | 0.03% / 0.03% | 21.6% / 21.5% | 5.8% / 5.5% | 0 | 9.1% / 9.1% | 63.4% / 63.9% |
| turning R | 9,514 | 17,475 (9,845) | 0 | 0.03% / 0.03% | 22.8% / 22.2% | 4.2% / 4.2% | 0 | 9.1% / 9.0% | 63.8% / 64.7% |
| strike (nMLF) L | 5,574 | 8,919 (4,727) | 0.02% / 0.02% | 0.01% / 0.02% | 5.8% / 5.3% | 1.2% / 1.1% | 0 | 11.9% / 11.4% | 81.1% / 82.1% |
| strike (nMLF) R | 6,009 | 9,747 (5,375) | 0.02% / 0.04% | 0.02% / 0.02% | 4.4% / 4.0% | 1.2% / 1.3% | 0 | 11.9% / 11.8% | 82.5% / 82.8% |

Counts behind the O-a columns (synapses, ipsi / contra to the readout by the lobe tag): M-system L 48 / 1 (11 fragments),
R 53 / 0 (13); Mauthner L 20 / 1 (8), R 15 / 0 (8); turning L 8 / 0, R 5 / 0; nMLF L 2 / 1, R 2 / 2. Single-synapse
fragments (cloud of one synapse, so no origin is observable by construction): 8.6-10.5% of M-system input, 8.5-8.7% of
turning, 7.4-8.3% of strike. Edge flags (min d_T <= 8 um, fewer than 2 synapses inside): 0.4-0.7% of M-system input.

**The Mauthner cells by compartment** (share of the compartment's excitatory input; measured):

| cell, compartment | input syn (exc) | O-a0 | O-a1 | O-b | O-d1 | O-d3 | O-c |
|---|---|---|---|---|---|---|---|
| Mauthner L, ventral dendrite | 1,440 (1,137) | 0.2% | 1.5% | 40.2% | 6.4% | 11.3% | 40.5% |
| Mauthner L, lateral dendrite (VIIIth zone) | 2,154 (1,513) | 0 | 0 | 19.6% | 16.5% | 10.4% | 53.5% |
| Mauthner L, soma / proximal | 2,255 (1,905) | 0 | 0 | 31.1% | 17.7% | 13.3% | 37.9% |
| Mauthner R, ventral dendrite | 716 (591) | 0.3% | 2.0% | 56.3% | 4.1% | 4.6% | 32.7% |
| Mauthner R, lateral dendrite (VIIIth zone) | 1,614 (1,345) | 0 | 0 | 12.6% | 4.5% | 14.6% | 68.2% |
| Mauthner R, soma / proximal | 1,843 (1,608) | 0 | 0 | 22.5% | 11.3% | 8.5% | 57.8% |

The M-system's six cells give the same picture (ventral dendrite O-a 1.6% L / 1.8% R of excitatory input; lateral and soma
compartments 0-0.1%). O-d2 is 0 in every compartment. No fragment on any readout reaches a ganglion label: the whole
volume holds only 436 synapses in mece0 Ganglia, so the VIIIth nerve's origin is never observed. The lateral-dendrite
input is O-c (54-68%), O-b, O-d1 and O-d3. The O-d1 somas on the Mauthner cells sit in r4 (478 and 146 synapses), r5,
r3 and r6-r7.

**What the O-a fragments are** (`oa_fragments.json`; measured):
- 34 fragments touch any deciding readout: 8 are B1 tectal cells (O-a0), 23 are somaless O-a1 fragments and 3 are
  soma-bearing non-tectal cells that pass the tectal-end rule.
- **The 24 on the M-system are long.** Their clouds run from x 357-450 um to 582-791 um (the Mauthner sits at 563 um)
  and z 89-259 um, with about 140-830 synapses each. They look like axons that leave the ventral posterior tectum and run
  along the ventral hindbrain past the M-cells.
- **Their tectal end sits in the SPV cell layer.** For 16 of the 24 it lies only in the SPV label, 1.4-8.7 um (median
  3.7) from the nearest B1 tectal soma, at z 121-229 um and x 374-463 um.
- **Mostly inputs there.** Of the 21 that are not B1 cells, 16 have inputs at their tectal end (3-17 each); 5 have only
  outputs there (2-54). That fits the proximal dendrite or initial segment of a periventricular projection neuron whose soma was cut off. This
  reading is ours, not measured.
- **Five have measured inputs from the retinal-afferent set:**
  - B1 tectal cell 17057069410: 297 afferent inputs, 2 excitatory synapses onto the left Mauthner's ventral dendrite.
    This is G1b's cell, and the one complete measured eye -> tectum -> Mauthner path in the data.
  - 13181211444 (9 afferent inputs, onto a right MiD cell).
  - Three O-a1 fragments with 1 afferent input each.
- **Brain-wide (measured):** 91 segments have >= 2 synapses in the tectum mask and >= 2 in the hindbrain posterior of
  540 um, and **27 of them touch a deciding readout.** Of the 91, 19 bear a soma and 6 are B1 tectal cells. The
  tectum-to-hindbrain spans that exist in synapse space are strongly enriched on the escape and turn cells. They are few.

**Ventral-dendrite specificity (post hoc, not pre-registered).**
- All 8 O-a fragments on each Mauthner contact only the ventral dendrite.
- The ventral dendrite holds 24.6% (left) and 17.2% (right) of each Mauthner's input synapses. Fragment-level binomial:
  p = 0.246^8 = 1.3e-5 and 0.172^8 = 7.5e-7.
- On the M-system, 10 of 11 (left) and 12 of 13 (right) O-a fragments are ventral-only (p 3e-4, 7e-5).
- Tectal input onto the ventral dendrite is what the goldfish anatomy that Dunn 2016 cites predicts (G1b-structure s.10).
  So O-a1 is catching real tectal input, not atlas noise.

**Controls (pre-registered expectations; measured):**
- **Positive control, 60 B1 tectal cells as pseudo-readouts** (2,479 input synapses, 1,956 fragments): O-a =
  **84.9%** (O-a1 82.8%, O-a0 2.1%). Expected >= 50%: **pass.** The rest is O-c (14.6%, mostly single-synapse
  crumbs) and O-d3.
- **Negative control, 60 caudal-hindbrain soma cells** (7,277 synapses, 5,809 fragments): O-a = **0 fragments**.
  Expected < 1%: **pass.** O-b = 15.5%, O-c 76.9%, O-d3 4.8%, O-d1 2.8%.
  - **O-b is not specific.** Cells about 150 um from the tectum get 15.5% of their input from fragments that "run toward
    the tectum", against 24-30% for the M-system. A post hoc tightening shows how little of O-b gets close: fragments
    whose cloud comes within 20 um of the tectum are 1.5-1.7% of M-system input (1.7% and 5.3% of the Mauthner ventral
    dendrites). The median O-b fragment on the M-system stops 66-69 um from the tectum.
- **The instrument and tracts (sensitivity row).** The 20 heaviest ventral-dendrite fragments of the Mauthner cells
  (10 per side; 19 distinct segments: 17 O-b, 1 O-c, 1 O-d1, 1 O-d3) were checked with their public meshes.
  - One mesh reaches the tectum mask where its synapse cloud did not: 22585578987, an O-b fragment whose cloud stopped
    5.6 um short (edge-flagged).
  - A second mesh comes within 9.9 um.
  - For the rest, mesh and cloud agree: the median difference in min d_T is 2.8 um (max 14.8).
  - **So the synapse cloud is not blind here.** The O-b pool does stop tens of microns short of the tectum. 20 meshes is
    a sample, not a census.

## 2. Reach from the retina's endings (task item 2)

**The set (rule as pre-registered; measured).**
- **Primary set: 1,284,858 segments** (634,465 left lobe, 650,393 right; 0 dropped at the midline). Rule: n_in = 0,
  >= 1 output, no soma, not tectal, >= 80% of outputs in mece2 18. They carry 4,659,632 output synapses. The volume holds
  5,610,304 synapses in the neuropil label and 6,365,059 in the tectum mask.
- **Sensitivity set (n_in <= 0.1 n_out):** 1,294,886 segments, 5,129,422 outputs, 16,522 inputs.
- The rule also takes non-retinal axon endings in the neuropil. It is an upper envelope of the retinal ganglion cell
  terminals, not an identification.
- cells.parquet's n_in and n_out equal the scan's counts for every segment (checked).

**Placement.** B1's `TectumMap` on the afferents' output centroids spans the full field (theta -20 to 160 degrees,
elevation -70 to +70). The loom patch drives every afferent of the lobe (the Gaussian never reaches 0), with drive sums
2.87e7 / 2.88e7 and a peak of 339. The prey patch has a peak of 1.0.

**Controls (G1b's rule; measured).**
- **PC1, afferents of a lobe -> tectal cells of each lobe, hop 1: +1.000** (Null A +0.001 +- 0.014, z 73.6, beyond
  all 20). Afferents supply 8.6% (left) and 9.0% (right) of their own lobe's tectal cells' input, and 0 of the other
  lobe's.
- **PC2, loom-zone vs prey-zone tectal cells: +0.618** (left eye +0.659, right +0.577; Null A +0.011 +- 0.077, z 7.9,
  beyond all). **Pass.** The afferent map and B1's tectal-cell map are independent placements, and they line up: loom-zone
  afferents feed B1's loom-zone cells.

**Reach, afferents vs G1b's tectal somas** (both sides pooled; "<= k hops" = share of the readout's input synapses whose
presynaptic segment is within k - 1 steps of a source; walk = cumulative excitatory walk share, hops 1-3, per side):

| readout | orphan share | <= 1 hop | <= 2 hops | <= 3 hops | exc, <= 3 | walk, L / R cells | G1b tectal: <= 1 / <= 2 / <= 3 | G1b walk L / R | Null A (afferents): <= 1 / <= 3 |
|---|---|---|---|---|---|---|---|---|---|
| M-system | 0.703 | **0** | 0.00079 | **0.0074** | 0.0032 | 3.1e-5 / 2.6e-6 | 0.00026 / 0.0049 / 0.0259 | 1.27e-4 / 4.05e-4 | 0.160 / 0.230 |
| Mauthner | 0.681 | 0 | 0.00080 | 0.0061 | 0.0031 | 5.5e-5 / 5.0e-7 | 0.00040 / 0.0025 / 0.0284 | 3.46e-4 / 5.14e-4 | 0.160 / 0.230 |
| turning | 0.747 | 0 | 0.000025 | 0.0112 | 0.0051 | 5.1e-7 / 1.6e-6 | 0.00005 / 0.0029 / 0.0312 | 1.59e-5 / 1.46e-5 | 0.159 / 0.229 |
| strike | 0.803 | 0 | 0.00011 | 0.0042 | 0.0023 | 1.1e-5 / 1.4e-7 | 0.00021 / 0.0022 / 0.0069 | 6.36e-5 / 3.51e-4 | 0.159 / 0.229 |

- **No afferent synapses onto any readout** (hop 1 = 0 everywhere). A shuffle gives 16%, because the afferents hold 16%
  of all synapses. Real afferent output stays in the tectum.
- **The orphan share is unchanged** (0.68-0.81), since no afferent contacts a readout.
- **Reach is lower from the endings than from the tectal somas**, because every path gains a hop. Within 3 hops,
  afferents reach 0.42-1.12% of a readout's input, against 0.69-3.12% from the somas. The walk shares fall to 1e-7 to
  5e-5.
- **Whole brain:** 2,966,658 segments within 1 step of the afferents and 3,157,318 within 2 (Null A 3.23 M and 3.72 M).
  G1b's tectal somas: 78,097 and 125,604.
- **Sensitivity set (near-zero inputs; real wiring only, no shuffles):** the same picture. M-system <= 3 hops 0.0082, Mauthner 0.0068, turning 0.0117,
  strike 0.0046. PC2 +0.615.

**Side and stimulus indices** (excitatory, cumulative hops 1-3, G1b's rule; Null A = the same 20 wirings as G1b):

| item | real | Null A mean +- sd | z | beyond all 20 | separates |
|---|---|---|---|---|---|
| SI M-system | **+0.966** | -0.014 +- 0.052 | +18.9 | yes | yes (ipsilateral; predicted contralateral) |
| SI turning | **-0.256** | -0.014 +- 0.037 | -6.6 | yes | yes (contralateral, the predicted sign) |
| SI strike | -0.013 | +0.027 +- 0.049 | -0.8 | no | no |
| SEL escape vs strike | -0.250 | -0.020 +- 0.089 | -2.6 | yes | no (z < 3; wrong sign) |
| SEL escape vs turning | +0.006 | +0.012 +- 0.071 | -0.1 | no | no |
| SI Mauthner (secondary) | +0.705 | -0.020 +- 0.067 | +10.8 | yes | (ipsilateral) |
| SI turning, 5 um dilated (secondary) | +0.765 | -0.030 +- 0.054 | +14.7 | yes | (ipsilateral) |

- **Pre-registered call: WEAK.** Two side separations and no stimulus separation. PC1 and PC2 pass, so the call is valid.
- **Unlike G1b, this null has power.** Its sd is 0.04-0.09 against G1b's 0.33-0.41, because 1.28 M sources reach the
  readouts through many paths in a shuffle.
- **The magnitudes carry the meaning, as in G1b.** Mean excitatory walk weight per member, left lobe -> left / right cells
  and right lobe -> left / right:
  - M-system: 3.1e-5 / 1.9e-7 and 7e-8 / 2.4e-6;
  - turning: 3.1e-7 / 1.35e-6 and 2.0e-7 / 2.5e-7.
- **The turning result is a lead only.**
  - The contralateral turning value rests on weights of about 1e-6.
  - It flips to ipsilateral (+0.77) in the dilated population.
  - Relays are few: no readout shares a hop-1 or hop-2 relay between the two lobes (Jaccard 0), with 0-15 relays per
    lobe per readout.
- Null B (side-class shuffle, sensitivity only): see s.6.

## 3. The eye itself (task item 3)

Measured from the full synapse scan and the CAVE somas table:
- The atlas Retina (mece0 1) has 38,184 voxels at 4 um and spans both eyes.
- It holds **0 synapses** (of 29,474,316), 0 segments with any synapse there, **0 CAVE somas** (of 187,052) and 0
  soma-bearing segments.
- So there is no segment in the eye whose outputs land in the tectum: 0 segments, 0 synapses.
- **The eye is not in the synapse data.** This cannot tell "the eye was not imaged" from "imaged but not segmented or
  synapse-detected"; either way nothing there is usable. The retinal ganglion cells enter the data only as their axon endings in the
  tectum (s.2). The eye model's input is therefore always *chosen (eye model)*, as the brief's metric already scores
  it, and the retinal-afferent set is the first measured stage.

## 4. Predicted decision-path share (task item 4; estimates from measured graph quantities)

Definitions are in s.0 (step 4). M = presynaptic segments within 2 excitatory steps of the stimulated lobe's afferents.
A = O-a with that lobe's tag, not in M. B = all O-b on the readout, not in M or A. Synapse view = excitatory synapse
counts; walk view = x_M (walk) against q_A, q_B (input fractions at unit activity). Share of the readout's decision drive
over measured synapses from the eye input (LB - UB):

| readout, stimulated lobe -> cell | pairing | exc M / A / B | synapse view LB - UB | walk view LB - UB | of all exc inputs: M, M + A |
|---|---|---|---|---|---|
| **M-system: left lobe -> right cells** (right-eye threat) | **contra (correct escape)** | 10 / 0 / 1,616 | **0.62% - 0.62%** | **0.0001% - 0.0001%** | 0.15%, 0.15% |
| **M-system: right lobe -> left cells** (left-eye threat) | **contra (correct escape)** | 4 / 1 / 2,221 | **0.18% - 0.22%** | **0.00003% - 0.027%** | 0.05%, 0.07% |
| Mauthner alone: left lobe -> right Mauthner | contra | **0 / 0 / 864** | 0% | 0% | 0, 0 |
| Mauthner alone: right lobe -> left Mauthner | contra | 4 / 1 / 1,345 | 0.30% - 0.37% | 0.0001% - 0.074% | 0.09%, 0.11% |
| M-system: left lobe -> left cells | ipsi (no crossing) | 23 / 38 / 2,221 | 1.0% - 2.7% | 0.015% - 1.8% | 0.31%, 0.82% |
| M-system: right lobe -> right cells | ipsi (no crossing) | 8 / 38 / 1,615 | 0.48% - 2.8% | 0.001% - 2.2% | 0.12%, 0.71% |
| turning: right lobe -> left cells | contra (correct) | 28 / 0 / 2,664 | 1.0% | 0.0002% | 0.23% |
| turning: left lobe -> right cells | contra (correct) | 40 / 0 / 2,179 | 1.8% | 0.001% | 0.41% |
| turning: ipsi, L / R | ipsi | 31 / 4 / 2,665 and 15 / 3 / 2,183 | 1.1% - 1.3% and 0.7% - 0.8% | 0.0003% - 0.13% and 0.0002% - 0.37% | 0.25-0.28%, 0.15-0.18% |
| strike (nMLF): contra, L cells / R cells | contra (Thiele) | 7 / 0 / 251 and 6 / 1 / 213 | 2.7% and 2.7% - 3.2% | 0.00005% and 0.0007% - 0.09% | 0.15%, 0.11-0.13% |
| strike (nMLF): ipsi, L / R | ipsi (Gahtan) | 10 / 0 / 251 and 0 / 2 / 217 | 3.8% and 0% - 0.9% | 0.04% and 0% - 1.0% | 0.21%, 0-0.04% |

- **The escape's correct pairing, the headline:** 0.00003% to 0.6% of the escaping cell's drive would travel on measured
  synapses from the eye input. For a right-eye threat, the escaping Mauthner has 0 measured synapses within 3 steps of the
  left lobe's afferents.
- **A ceiling that does not depend on the bridge's size (expected):** G1-gate puts the threshold at about 25 coincident
  synapses at Shiu's 0.275 mV, the weight G1c keeps for measured synapses.
  - The measured contralateral paths give the escaping Mauthner at most 4 excitatory synapses (left cell, 2
    presynaptic segments) or 0 (right cell).
  - Even if every one fired in sync, at least 21 of the 25 must come from the bridge. The measured share at firing is
    at most 16%, or 0%.
  - Those 4 segments are themselves 1-2 steps from the afferents, with walk weights of 1e-7, so the real figure sits far
    below the ceiling.
- **How robust the range is (post hoc, 04:20 EDT, `variants_post_hoc` in share.json).**
  - **Synapse view, near-tectum O-b only** (min d_T <= 20 um), a smaller bridge pool: the escape's correct pairing rises
    to 3.3% (left cells) and 8.7% (right cells). No crossing: 20-30%. Turning (contra): 51-75%.
  - The synapse view counts every measured-reachable input as fully active. The walk view shows those inputs carry
    1e-7 to 1e-5 of the readout's input. Its range stays at 0.0005-0.4% for the escape's correct pairing and at 0.01-0.1%
    for turning.
  - A bridge that drives no O-b at all makes the synapse view read 100% trivially: then nothing but 4-40 measured
    synapses drives the cell, it cannot fire, and bridge (c) must add the drive.
- **What this means for "mostly true".**
  - On the brief's pre-registered headline metric, the escape decision path is almost entirely bridge in every bridge
    design we can draw from this graph. The crossing is not in the data (1 crossed O-a synapse, from a non-tectal
    soma cell).
  - The brief's pre-registered sentence applies: *"Which way the fish flees is set by the published pathway model, not
    by the measured wiring."*
  - The synapse-count share (secondary) will still be near 100% of the simulated network. The brief already says it
    says little on its own.

## 5. For the bridge builder (descriptive; `oa_fragments.json`)

- **Bridge (a)'s inventory.** 24 fragments onto the M-system (the columns are listed per fragment).
  - They make 45 (left) and 41 (right) excitatory contacts on each side's three M-system cells; 38 of each are not
    already on a measured path (the A column in s.4).
  - They make 19 (left) and 14 (right) excitatory contacts on the two Mauthner ventral dendrites.
  - Every one is uncrossed.
  - **Expected (not simulated):** these counts are within a factor of 2 of the ~25 coincident synapses a cell needs. So
    bridge (a) alone might fire the *ipsilateral* Mauthner, which is the wrong escape direction.
  - The correct direction needs bridge (c), as the brief anticipated.
- **The columns sit where the loom is not (post hoc, 04:30 EDT).** Each fragment's column = the 20 B1 tectal somas nearest
  its tectal end (a choice).
  - Under B1's map those columns prefer elevation -35 to -66 degrees (median -46; one fragment -0.5), theta -8 to 88
    degrees. That is the lower field, because their ends sit in the ventral SPV.
  - G1b's loom patch (azimuth 90, elevation +30) drives them at a median 4.5e-6 of its peak in the lobe (max 8.8e-4). The
    prey patch drives them at a median 0.7% (max 33%).
  - **If bridge (a) links by position, the gate's loom will not reach it.** Either the loom moves into the lower field,
    or the map's dorsal-ventral axis is re-derived, or the link is made by tract topography (bridge b) and disclosed.
  - **Caveat:** a projection neuron's axon exit point need not be its receptive field; its dendrites may sample other
    neuropil rows. This is exactly why the brief separates (a) from (b).
- **The O-b pool is large and unspecific** (24-30% of M-system input; 15.5% even for caudal-hindbrain cells). Assigning
  it by tract topography is a model with a lot of freedom. The near-tectum subset (1.5-1.7% of M-system input) is the
  defensible part of it.

## 6. Null B (side-class shuffle; sensitivity, never in the verdict)

10 side-class shuffles (G1b's seeds `G1b-structure-sideshuffle-0..9`). Each keeps every pair's (pre side, post side)
class, with 0 unrepaired pairs and degrees preserved. The empirical p floor is 1/11 = 0.091. Null B keeps "cells connect
within their own side", so it predicts what laterality alone gives:

| item | real | Null B mean +- sd | z vs Null B |
|---|---|---|---|
| SI M-system | +0.966 | +0.988 +- 0.004 | -4.7 |
| SI Mauthner | +0.705 | +0.981 +- 0.008 | -33 |
| SI turning | -0.256 | +0.972 +- 0.006 | -205 |
| SI turning, dilated | +0.765 | +0.942 +- 0.006 | -29 |
| SI strike | -0.013 | +0.825 +- 0.032 | -26 |
| SEL escape vs strike | -0.250 | +0.068 +- 0.065 | -4.9 |
| PC2 | +0.618 | -0.022 +- 0.082 | +7.8 |

- **Laterality alone predicts near-total ipsilateral reach (+0.83 to +0.99)** from the retinal endings. Every real readout
  is less ipsilateral than that. The M-system barely (+0.966 against +0.988). Turning and strike reach both sides about
  equally (-0.26 and -0.01).
- So beyond laterality, the multi-hop paths from the endings do reach the other side's turning and strike cells more
  than a side-keeping random wiring would. **This is a lead, not a route:**
  - the weights involved are 1e-7 to 1e-6;
  - the turning value turns ipsilateral (+0.77) in the dilated population;
  - with 10 shuffles the p value sits at its floor;
  - a z from an sd of 0.006 is not stable.
- It does not rescue the escape: the M-system stays ipsilateral, and the escaping Mauthner for a right-eye threat has no
  measured path at all (s.4).
- PC2 beats Null B too (z 7.8). The afferent-to-tectal-cell topography is more than side.

## 7. What changed after the pre-registration (for the honesty page)

1. **04:08-04:13 EDT, O-a detail (`run_oa`, descriptive).** Added after the first class table showed long O-a1 fragments:
   tectal-end roles, afferent inputs, column. It changes no class.
2. **About 04:15, ventral-dendrite specificity** (the binomial in s.1). Post hoc.
3. **About 04:18, near-tectum O-b share** (min d_T <= 20 um) and O-b's median distance. Descriptive, post hoc.
4. **04:20, share variants** `near20` and `none` in share.json. Post hoc sensitivity rows; the pre-registered row is
   unchanged and is the headline.
5. **04:30, loom/prey drive at the O-a columns** (`patch_drive_rel_post_hoc`). Post hoc.
6. **The ~25-coincident-synapse ceiling** in s.4 uses G1-gate's figure. It was not in the pre-registration and is an
   expectation, not a measurement.
7. **Null B ran** because machine time allowed (the pre-registration made it optional). It is not in the verdict.
8. **Priority rule as registered, flagged here.** Three soma-bearing non-tectal segments pass the tectal-end rule and sit
   in O-a1 (O-a1 outranks O-d1). One carries the only crossed O-a synapse onto the M-system (9695179340, 1 synapse, 2 output
   synapses at the SPV edge). Reading it as O-d1 would make the crossed O-a count 0. No other class changes.
9. **A test fixed, not a threshold.** The synthetic lookup test first probed y = -5, which clips into the synthetic ganglion
   strip; the test now probes a hindbrain voxel and checks the ganglion clip separately.

Nothing in s.0 changed: no threshold, class order, population, seed or rule was altered after a result was seen.

## 8. Tests

`cd launches/fishbrain/brain && .venv/bin/python -m pytest -q tests/test_origins.py` (bare, no pipe): see the exit code
line below. 8 synthetic tests (no data) and 11 data tests.
- **Synthetic tests:**
  - the atlas lookup, distance and clipping;
  - **the task's positive control:** a synthetic fragment spanning tectum -> hindbrain classifies O-a1, and its lobe tag
    follows the side of its tectal synapses;
  - a fragment that runs toward the tectum and stops 25 um short is O-b and not O-a;
  - a local blob is O-c, a fragment running posterior is O-d3, and ganglion synapses give O-d2;
  - one tectal synapse is not enough and is edge-flagged;
  - the priority order (tectal cell > tectal end > soma);
  - the compartment order;
  - the afferent rule (primary and near-zero);
  - the share formulas, including "nothing visual reaches the cell" reading undefined rather than 100%.
- **Data tests:**
  - the atlas validity check;
  - the pipeline check against task R;
  - both real-data controls;
  - the parts sum to the total: every readout side's classes and compartments sum to its input synapses, which are
    counted independently from the pair arrays;
  - the ventral-only O-a observation;
  - the empty eye;
  - the afferent set satisfies its rule on data;
  - 20 shuffles, each with preserved degrees and 0 unrepaired pairs; the verdict recomputes from the stored numbers;
    reach is monotone, with 0 at hop 1;
  - every share row recomputes from its parts and satisfies LB <= UB within [0, 1];
  - the mesh row;
  - the pinned key numbers.
- **The tests fail when they should** (each change reverted, file compared byte for byte after):
  - O-a threshold 1 instead of 2: 1 failed;
  - compartment order swapped: 1 failed;
  - afferent neuropil share 0.5: 1 failed;
  - UB_syn without A: 2 failed;
  - O-b rule removed: 1 failed;
  - 3 synapses removed from one class in origins.json: 1 failed.

**Run of record, 2026-09-26 04:35 EDT: `19 passed in 1.81s`, exit code 0.** G1b's own suites (test_structure,
test_readouts, test_cells) still pass (43 passed, 04:36 EDT); no file of theirs was written.

## 9. Limits

- **Graph estimates, not a simulation.** Activity per synapse is assumed (synapse view) or linear with unit gains (walk
  view). Spiking thresholds will make measured multi-hop paths carry less, not more (G1).
- **O-b is an envelope** (the negative control shows it). **O-a1 depends on the atlas's SPV / neuropil boundary at 4 um.**
  The ventral-only specificity and the 1.4-8.7 um distance to tectal somas argue that it is real; a proofread check of a
  few O-a1 fragments would settle it.
- **The retinal-afferent set is an envelope** of RGC terminals plus other orphan neuropil endings. Its placement is B1's
  rank-uniform model. PC2 shows that the two placements agree, not that either matches real receptive fields.
- **Meshes were checked for 20 fragments only**, all non-O-a.
- **Left/right rests on B1's axis call** (a mirror error flips every side consistently).
- **The data are the public agglomeration** (G1b's CAVE v709 check found the same tectal detachment).

## Sources

Brain of record, identities and literature as in G1-cells.md, G1-gate.md, G1b-readouts.md and G1b-structure.md; the atlas
region names from the Fish1 segment properties (mece0-3_231218). Citation required by the Fish1 data policy: Petkova,
Januszewski et al. (2025), "A connectomic resource for neural cataloguing and circuit dissection of the larval zebrafish
brain", bioRxiv.

## Verifier correction (2026-09-26, adversarial review, REFUTED on the bounds; arithmetic reproduced exactly)
An independent re-implementation reproduced every number above. What does **not** hold is calling the synapse-view
figures upper bounds, and the "at most 4 of ~25 synapses" ceiling being bridge-independent. Both depend on the
3-hop cutoff:
- At 4 hops the escaping Mauthner cells get 47–48 measured excitatory synapses, not 0–4.
- The escape's synapse-view share is **0.62% at ≤3 hops, 4.9% at ≤4, 13% at ≤5 and 20% at ≤6**.
- With the near-tectum bridge-(b) pool it is 3.3–8.7%.
- The walk (probability-weighted) view barely moves: 1.9e-7 at 3 hops and 2.3e-7 at 6.

**Re-scoped headline:** the escape path is overwhelmingly bridge on the walk estimator, and the midline crossing is not in
the data. The exact decision-path share, as the brief defines it (provenance-weighted drive, no hop cutoff), can only
come from simulating the bridged fish. That is the number that will be published. "Under 1%" is withdrawn as a bound;
it is one estimate under one bridge design and a 3-hop cutoff.
