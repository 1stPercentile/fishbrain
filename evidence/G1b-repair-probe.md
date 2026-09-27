---
publication_copy: edited-for-publication copy of launches/fishbrain/evidence/G1b-repair-probe.md at commit 3604d7df5. Local paths made repo-relative; machine-owner, internal coordination, account and credential details removed. No number, hash, time, method, result or correction was changed; publication notes are marked [Publication note].
title: G1b repair probe, can the cut axons be reattached?
created: 2026-09-26
status: complete, verdict NOT_VIABLE (2026-09-26 02:45 EDT)
evidence_status: measured unless marked "expected" or "observed, not tested"
verification_scope: >
  Run on the analysis machine on 2026-09-26 01:35-02:40 EDT against public Fish1 layers read that night
  (hindbrain_reconstructions, seg_241003_agg241003) and the pinned brain of record
  (brain/data/synapses_raw.parquet, content hash 09a71038...25af2). Sample: 50 proofread neurons.
  Raw outputs: brain/data/repair/*.json, report.md, rules.npz, truth_rows.npz (git-ignored).
files: task P. brain/fishbrain/repair_probe.py, brain/data/repair/, this note.
---
# G1b: can we reconnect Fish1's cut axons?

**Result: NOT_VIABLE.** None of the three pre-registered proximity rules gets anywhere near 80%
precision. The best membership-weighted precision in any rule, half or threshold is **0.005**. The
best pre-registered synapse-read precision is **0.043**. On the held-out half no threshold is usable:
repair adds **0.000** to the **0.500** of outputs the soma segment already carries.
The reason is geometric, and it was measured. The nearest soma-bearing arbor to an orphan axon is
usually the dendrite that axon synapses onto. Of the read outputs on fragments that R2 gives to a
sampled neuron, **4,058** synapse onto that neuron and **603** are its own outputs. The ground truth
itself held up: 706/707 hand-annotated outputs sit on their own neuron's mesh, and 593/707 are
recovered by the label pipeline.
**A second finding changes how to read the first.** The authors' proofread hindbrain neurons are
not typical Fish1 cells. Their soma segments carry **63%** of their outputs (brain-wide: 1.61% of all
synapses come from a soma segment). In the pool sampled from, the median soma segment has **136** input synapses. The 743
unreconstructed cells of the same study have a median of **11**, and random somas in the same box
**2**; the median output count is 0 for both. The agglomeration looks already repaired around these
cells (proofreading folded in, inferred), so the damage on the typical neuron is worse than
measured here. Six of the 50 soma segments are bare soma pieces with 0-2 synapses. Three more
mostly belong to another cell.

## 0. Pre-registration (written 2026-09-26 01:35 EDT, before any ground truth or rule was computed)

Everything in this section was fixed before the first label read of the proofread layer and before
any rule ran. Anything changed later is logged in section 7 with its time.

**Ground truth.** `gs://fish1-public/hindbrain_reconstructions` (proofread hindbrain neurons, labelled
by lore id) read at mip 0 (16 x 16 x 30 nm).

**Sample.** Of the HMI dataframe's 999 cells, 177 have `reconstruction_status` = "soma,
dendrite(reconstructed), axon(reconstructed)". The 20 cells marked "soma, dendrite(c), axon(c)" are
left out because the release never defines "(c)". The sample is **all 5 SPNs** with the full status
(158660, 147392, 174608, 178639, 180791) plus **45 others** drawn with
`numpy.random.default_rng(20260926)` from the remaining 172. A cell is replaced by the next draw
only if its lore id has no agg241003 segment in `agglomerated_segments_and_soma_ids.csv` or no mesh
in the proofread layer. Target n = 50.

**Point mapping.** For a synapse line p0 -> p1 (p0 pre side, p1 post side, G1-pull), the pre-side
point is q_pre = p0 + 40 nm x unit(p0 - p1) and the post-side point is q_post = p1 + 20 nm x
unit(p1 - p0) (zero-length lines use p0 and p1 as they are). Before any truth is computed, this
mapping is checked against the agglomeration itself on a random sample of about 200 synapses: the
share of q_pre that reads the parquet's `pre` id and of q_post that reads `post`.

**Which chunks are read.** Every mip-0 chunk (128 x 128 x 64 voxels) that holds at least one LOD-0
mesh vertex of a sampled neuron. Every synapse endpoint inside those chunks is labelled.

**Truth definitions.**
- Output synapses of neuron N: synapses whose q_pre reads label N.
- Fragment membership: an agg segment S belongs to N when f_N(S) > 0.5, where f_N(S) = (S's
  synapse points that read N) / (S's total synapse count, n_in + n_out from cells.parquet). S's
  points are q_pre of its outputs and q_post of its inputs. Points never read count as not-N.
- Anchor A_N: N's agg segment from the lore-id csv (the soma-bearing segment).
- A fragment is a **synapse-bearing** agg segment with no soma. Cable with no synapse is invisible to
  this method.

**Damage measures (per neuron and pooled).** Synapse-bearing fragments per neuron; share of N's
output synapses whose `pre` is A_N; share on detached segments.

**The three rules (no ground truth used).** Soma-bearing segments are the 180,782 distinct
nonzero agg ids in the lore-id csv. Their geometry = the CAVE v709 soma centres (`pt_position`) plus
q_pre of their outputs plus q_post of their inputs. A fragment's geometry = q_pre of its outputs plus
q_post of its inputs. All geometry inside the region (union of the sampled neurons' mesh boxes plus a
60 um margin) is used, not only the labelled chunks. Distances in nm. Cap: 50 um (beyond that the
rule abstains).
- **R1, nearest soma body.** The fragment's centroid goes to the nearest soma centre; that soma's agg
  segment is the owner.
- **R2, nearest soma-bearing arbor.** The owner is the soma-bearing segment with the smallest
  point-to-point distance to any of the fragment's points.
- **R3, arbor plus direction.** For a fragment with >= 3 points whose principal axis explains
  >= 80% of its point variance and spans >= 2 um, the candidates are soma-bearing geometry points
  within 45 degrees of the axis, beyond either end of the fragment (outward from its extreme points);
  the nearest such candidate wins. Every other fragment falls back to R2.

**Abstention variable, one per rule.** Confidence c = d2 / d1: the distance to the nearest
*different* candidate segment divided by the winning distance (d1 floored at 0.1 um). A rule assigns
only when c >= t. t is swept over {1, 1.25, 1.5, 2, 3, 5, 10}.

**Split.** The 50 neurons are split into a calibration half and a test half with
`default_rng(20260926 + 1).permutation`. For each rule, t is the smallest value whose pooled
calibration precision is >= 0.80. The best rule is the one with the highest calibration recovery at
its t. Every rule is then scored on the test half at its calibration t. The full curves are reported
for both halves.

**Metrics.**
- Recovered (the verdict metric): N's true output synapses whose `pre` is A_N or a fragment the rule
  assigns to A_N, over all of N's true output synapses. Pooled over neurons (synapse-weighted), with
  the per-neuron median beside it. Reported with and without the soma segment's own outputs.
- Precision (the verdict metric): among output synapses on fragments the rule assigns to sampled
  anchors, the share whose q_pre reads that anchor's neuron. This excludes the anchor's own
  outputs, so it is the rule's precision alone. Fragment-count precision and recall reported beside it.

**Controls.**
- Positive, must pass: every anchor A_N comes out as N's (f_N(A_N) > 0.5). The HMI `outputs`
  column's hand-annotated output synapses of any sampled neuron should sit within 1 um of a
  label-derived output synapse of that neuron. Chunks holding >= 20 LOD-0 vertices of N that read
  all zero are counted (absent-chunk check).
- Null: the runner-up (second-nearest distinct segment) under R2, and a uniform random pick among
  the 10 nearest distinct soma-bearing segments. The real rule must beat both.

**Verdict (kill criterion as given, stated before running).** On the test half, at its calibration t:
- **VIABLE:** the best rule recovers >= 50% of true output synapses at precision >= 80%.
- **PARTIAL:** not VIABLE, but at precision >= 80% it recovers >= 25% of true output synapses and at
  least 2x what the soma segment alone carries.
- **NOT_VIABLE:** otherwise. In particular, recovery under 50% at >= 80% precision means repair is
  not viable at scale.

## 1. What ran (measured)

| stage | what | numbers | wall |
|---|---|---|---|
| select | 5 fully reconstructed SPNs + 45 seeded others; LOD-0 meshes; mip-0 chunks they touch | 46 draws used, 1 skipped (97584: no mesh); 27,177 distinct chunks (31,853 summed over neurons) | about 1.5 min |
| region | synapses with p0 inside the union of mesh boxes + 60 um | **27,585,933** of 29,474,316 (93.6%): the rules run on nearly the whole brain; 5,689 zero-length lines | about 1 min |
| validate | point mapping vs the agglomeration, 200 random synapses (seed 20260928) | table below | 10 min (slow GCS) |
| label | proofread label at every q point in the 27,177 chunks, mip 0 | 496,186 pre points (80 nm), 495,868 post points, 496,134 pre points (40 nm); **0 read errors**; 0 unread chunks | 993 s (second process pool) + about 6 min (first pool, 8 parts kept); an earlier threaded run was discarded |
| rules | R1, R2, R3 over every output-bearing fragment in the region | 6,495,011 fragments, 28,651,681 fragment points; 9,728,864 soma-bearing points incl. 159,601 soma centres | about 3 min |
| truth, evaluate | membership, damage, split, sweep, nulls | tables below | about 2 min |

Budget: the task allowed about 2 hours. The work ran 01:25-02:50 EDT, about 85 minutes, including
about 30 minutes of GCS reads.

**Point mapping against the agglomeration** (`data/repair/validate*.json`; each cell: reads out of 200)

| point | reads `pre` | reads `post` | reads 0 | reads another id |
|---|---|---|---|---|
| p0 raw | 116 | 58 | 20 | 6 |
| q_pre 40 nm behind p0 | 151 | 27 | 16 | 6 |
| **q_pre 80 nm behind p0 (used)** | **170** | **10** | 11 | 9 |
| p1 raw | 2 | 179 | 10 | 9 |
| **q_post 20 nm past p1 (used)** | 3 | **181** | 7 | 9 |

Clean reads (section 7 item 1) lower the chance that an input onto N counts as N's output to
about 5% x 9.5% (expected, assuming the two ends err independently).

**Sample** (`data/repair/selection.json`). 50 neurons, all with status "soma, dendrite(reconstructed),
axon(reconstructed)". SPNs: 147392, 158660, 174608, 178639 (MiR1), 180791 (RoM1 prox); three are RoV3
prox. Calibration half: 25 neurons incl. SPNs 147392 and 174608. Test half: 25 incl. 158660, 178639 and
180791. Eleven sampled neurons have hand-annotated outputs in the HMI dataframe (707 synapses).

## 2. Controls (measured)

| control | must | result |
|---|---|---|
| mapping vs agglomeration | pre point reads `pre` well above chance | 85% (170/200) at 80 nm; post point 90.5% |
| HMI hand-annotated outputs | lie near N's label-derived outputs | **593/707 within 1 um** (83.9%); per-neuron median distance 0.21-0.33 um for all 11 neurons. The misses match the read rate. Needed the y-mirror in section 7 item 5 |
| HMI annotations vs meshes | lie on N's own mesh | 706/707 within 1 um after the mirror (0/707 before) |
| anchor is N's (f_N(A_N) > 0.5) | every neuron | **FAILED for 9/50.** Six anchors are soma pieces with 0-2 synapses: 120685, 151567 and 164353 have 0; 145368, 104213 and 131049 have 2, with f = 0.5, which is not > 0.5. Three have 101-751 synapses that mostly read another label: 141444 (177 synapses, f 0.00), 160192 (101, 0.00) and 131622 (751, 0.13). The csv's soma segment is then not the neuron: a detached soma or a merge. Reported, not repaired. |
| anchor purity | anchor's outputs read N | 2,881 of 3,669 anchor output points (78.5%), close to the read rate. Anchors median f_N 0.847 |
| absent chunks | chunks with >= 20 LOD-0 vertices of N and no N voxel | **0** |
| nulls for the rules | real rule beats them | runner-up null precision 0.000; label-permutation null 0.000. The real rules are no better (section 4) |

## 3. Damage on the proofread sample (measured)

Pooled over 50 neurons, **4,550 clean output synapses** (median 60 per neuron, range 0-371; neuron
160192 has none) and 10,390 clean input synapses.

| where N's outputs sit | synapses | share |
|---|---|---|
| on the soma segment A_N | 2,881 | **63.3%** |
| on member fragments (no soma, f_N > 0.5) | 1,117 | 24.5% |
| on segments that are not N's by majority (merged into others' pieces, or single misreads) | 536 | 11.8% |
| on another soma segment | 16 | 0.4% |

- **Oracle ceiling:** reattaching every member fragment perfectly would reach **87.9%**.
- **Per neuron:** share on the soma segment median 0.600, IQR 0.000-0.887 (n = 49). It is bimodal:
  either the anchor carries the axon or it carries almost none. Calibration pooled 0.759 vs
  test 0.500.
- **Fragments:** a neuron's clean outputs sit on a median of **8** distinct agg segments (IQR
  5-15, max 44). Synapse-bearing member fragments per neuron: median 4 (IQR 2-8, max 22), 286 in
  total. Only 109 of them carry a clean output (median 1 per neuron, max 11). Two member segments
  hold another soma.
- **f_N histogram** over the 882 (segment, neuron) pairs with any clean N point, bins 0/.1/.25/.5/.75/.9/1:
  304, 112, 83, 104, 68, 211. Membership is mostly clear-cut. The 104 in (0.25, 0.5] are merged or
  thin cases.
- **SPNs:** 147392 (181 outputs, 153 on the anchor), 158660 (37, 32), 174608 (212, 187), 178639
  (232, 211). 180791 is the outlier: 371 outputs, 155 on the anchor and 169 on 22 member fragments.
- **Geometry:** outputs lie a median of 105 um from the soma (per-neuron IQR 88-126 um). Even the
  anchor's own outputs sit a median 99.6 um away, so these soma segments include long axons.
  Median share of outputs within 20 um of the soma: 0.0.

**Why this sample understates the damage** (`data/repair/sample_bias_check.json`; agg segment at the
soma, from the lore-id csv; counts from cells.parquet)

| group | n | outputs: median / mean | share with >= 10 outputs | inputs: median |
|---|---|---|---|---|
| HMI "axon(reconstructed)" (the pool sampled from) | 177 | 7 / 64.6 | 0.48 | 136 |
| our 50 | 50 | 11.5 / 73.4 | 0.52 | 166.5 |
| HMI "(c)" | 20 | 61.5 / 76.9 | 0.80 | 115.5 |
| HMI soma only, same study, same region | 743 | **0 / 8.9** | 0.12 | **11** |
| 2,000 random CAVE somas in the HMI soma box (seed 20260929) | 1,943 | **0 / 2.3** | 0.04 | **2** |

The proofread cells' soma segments are one to two orders larger than their unproofread neighbours'.
Two readings fit this table (neither tested): `seg_241003_agg241003` already includes the authors'
merges for these cells, or the authors chose cells the agglomeration already handled well. Under
either one, the measured 63% is an upper bound for typical cells. Their soma
segments hold a median of 0 outputs, so for them almost everything would have to be reattached.

## 4. The three rules (measured; `data/repair/eval.json`, `report.md`)

"recovered" = N's clean outputs on A_N or on a fragment assigned to A_N, over all N's clean outputs.
"repair" = the part of that on reattached fragments. "prec (memb)" = the verdict precision
(section 7 item 3). "prec (reads)" = the pre-registered synapse-read version.
"frag P / R" = fragment-count precision and recall. Pooled over each half; t = confidence threshold.

| rule | half | t | recovered | repair | soma seg | prec (memb) | prec (reads) | frag P | frag R | fragments assigned |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | calibration | 1.0 | 0.760 | 0.001 | 0.759 | 0.000 | 0.001 | 0.000 | 0.000 | 502 |
| R1 | calibration | 2.0 | 0.759 | 0.000 | 0.759 | 0.000 | 0.004 | 0.000 | 0.000 | 13 |
| R1 | test | 1.0 | 0.501 | 0.001 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 3,704 |
| R2 | calibration | 1.0 | 0.815 | 0.057 | 0.759 | 0.000 | 0.004 | 0.002 | 0.419 | 6,333 |
| R2 | calibration | 2.0 | 0.780 | 0.021 | 0.759 | 0.002 | 0.007 | 0.005 | 0.323 | 1,928 |
| R2 | calibration | 5.0 | 0.770 | 0.011 | 0.759 | 0.005 | 0.020 | 0.009 | 0.129 | 440 |
| R2 | calibration | 10.0 | 0.760 | 0.001 | 0.759 | 0.000 | 0.043 | 0.000 | 0.000 | 32 |
| R2 | test | 1.0 | 0.533 | 0.033 | 0.500 | 0.000 | 0.005 | 0.002 | 0.077 | 3,138 |
| R2 | test | 5.0 | 0.503 | 0.003 | 0.500 | 0.002 | 0.010 | 0.005 | 0.013 | 195 |
| R3 | calibration | 1.0 | 0.794 | 0.035 | 0.759 | 0.000 | 0.002 | 0.002 | 0.419 | 6,370 |
| R3 | calibration | 5.0 | 0.767 | 0.008 | 0.759 | 0.004 | 0.011 | 0.009 | 0.129 | 431 |
| R3 | test | 1.0 | 0.519 | 0.019 | 0.500 | 0.000 | 0.002 | 0.002 | 0.077 | 3,278 |
| R3 | test | 5.0 | 0.503 | 0.002 | 0.500 | 0.001 | 0.006 | 0.005 | 0.013 | 204 |

All 42 rows (3 rules x 2 halves x 7 thresholds) are in `data/repair/report.md`. The rows left out
here say the same thing.
- **No rule reaches 0.80 precision at any t on calibration**, so no threshold is chosen and there
  is no best rule. Scored anyway at t = 10 on the test half, every rule recovers 0.500, the
  soma segment alone.
- **R2 and R3 do find true fragments.** At t = 1 they give 42% of calibration member fragments
  (with outputs) to the right soma. They also hand the same soma about 250 foreign fragments per
  neuron.
- **What the foreign fragments are.** Take the 9,510 fragments R2 gives to sampled anchors: they
  have 49,990 output synapses and 28,673 of them were read. **4,058** have a post point that reads
  the assigned neuron; they are inputs onto it. **603** have a pre point that reads it; they are its
  own outputs. **4,710** synapse directly onto the anchor segment. Proximity between synapse
  clouds cannot tell "my axon continues here" from "an axon synapses on me here".
- **R1 (nearest soma body)** never assigns a true fragment: outputs sit about 100 um from the soma.
- **Nulls.** R2's runner-up (all 50 neurons, t = 1): precision 0.000, 10,228 fragments. R2 scored
  against the wrong neurons (a rotation): precision 0.000. R2 itself over all 50 neurons at t = 1:
  precision 0.000 (fragment precision 0.002, 9,471 fragments). The rule is at null level.
- R2 had no distinct runner-up within 16 neighbours for 605 scope fragments (the 16th neighbour's
  distance was used, a lower bound). R3 had no runner-up in its cone for 971 axis fragments, so their
  confidence is unbounded. Neither changes the verdict: precision is near 0 at every t.

## 5. Verdict

Pre-registered criterion, test half at the calibration threshold: **NOT_VIABLE**. No rule reached
80% precision on calibration, so none has a threshold to carry to the test half. The best precision
anywhere is 0.005 (membership) or 0.043 (reads). **Simple proximity reattachment is not a repair for
Fish1.** The kill criterion's other half (recovery >= 50%) is met only by the soma segment's own
share on this biased sample (0.500 test, 0.759 calibration), never by the repair.

What this means for the rethink (G1-gate section 9), stated as observations, not tested:
- Option 1 (attach axons to somas) needs real shape-aware agglomeration or human proofreading.
  Nearest-arbor rules attach the wrong cell's axons.
- **The first untested rule** is R2 with a partner filter: drop any candidate the fragment synapses
  onto (its `post` is the candidate, or its post points read the candidate's neuron). It was outside
  the three-rule cap and was not run. We expect it to remove most foreign attachments and leave R2's
  42% fragment recall intact (expected, not measured). Even then, member fragments carry only 24.5% of
  outputs on this sample, so the recovery it could add is bounded by that.
- The authors' proofread hindbrain cells already carry most of their outputs on their soma
  segments (63% pooled). A model restricted to the 177 + 20 proofread HMI cells would sit on
  wiring that is mostly attached. It would be small and hindbrain-only, and it has no tectum.

## 6. Flags (observed, not tested further)

1. **HMI dataframe coordinates are y-mirrored against the volume:** y_volume = 65000 - y_hmi
   (8 nm voxels). Anyone reading synapse positions from `em_zfish1_dataframe.xlsx` must mirror y.
   B1's left/right call rests on CAVE and neuroglancer states in the volume frame, not on these
   columns, but the mirror is worth one check by whoever owns sides.
2. **SPN 158660's whole proofread arbor lies anterior of its soma** (mesh x 30,673-65,944; soma x
   65,664 in 8 nm voxels). The other four fully reconstructed SPNs run posterior to x 83,298-88,350.
   No sampled mesh extends past x = 88,350 (the largest xmax of the 50), so the SPN axons stop there
   in the reconstructions. 38/50 sampled neurons have more than half their mesh vertices anterior of the
   soma (`data/repair/arbor_vs_soma_x.csv`).
3. **The Mauthner cells (lore 187052, 187053) have no mesh in `hindbrain_reconstructions`.** There is
   no proofread Mauthner axon to use.
4. For 9 of 50 proofread neurons, the lore-id csv's soma segment is not the neuron by majority
   (section 2). Any model keyed on csv soma segments inherits this.

## 7. Changes after pre-registration (each logged before the step it affects ran)

1. **01:57 EDT, point mapping (before any proofread label was read).** The agglomeration check on 200
   random synapses (`data/repair/validate.json`) found q_pre (40 nm) reads the parquet's `pre` id
   151/200 times and its `post` id 27/200 times. Raw p0 read pre 116 and post 58. A second pre point 80 nm
   behind p0 was then read on the same 200: pre 170, post 10, zero 11, other 9
   (`validate_qpre80.json`). q_post (20 nm) read post 181/200 and pre 3/200. **Change:** the pre-side
   point is the 80 nm one. A point votes only when the two ends of its line read different
   labels ("clean read"), so an input onto N whose pre point strays into N (about 5% x 9.5%) is not
   counted as N's output. All three points (40 nm, 80 nm, post) are labelled, so the 40 nm version can
   still be compared.
2. **02:08-02:15 EDT, engineering only.** The threaded chunk reader was GIL/latency-bound at 6.4
   chunks/s; it was replaced by a process pool (12 processes x 12 threads) that writes one checkpoint
   file per block of about 283 chunks. `cloudvolume.scattered_points` was measured at 0.5 chunks/s on
   293 chunks and not used. The reads are the same either way.
3. **02:19 EDT, precision (before `evaluate` ran; truth not yet computed).** A true fragment of N reads
   N on only about 75-85% of its outputs (item 1), so the pre-registered synapse-read precision is
   capped near 0.8 by read noise, not by the rule. **Change:** the verdict precision is
   membership-weighted: an assigned fragment counts as correct when it is a member of N (f_N > 0.5),
   weighted by its whole-brain output count (`n_out`, cells.parquet). The pre-registered
   synapse-read precision is still computed and reported beside it
   (`precision_syn_reads_preregistered`).
4. **Clarification of R3 (02:19 EDT, written into the code before it ran at 02:13).** When a
   fragment qualifies for an axis but no soma-bearing point lies inside the cone among its ends' 64
   nearest, R3 falls back to R2. When the cone holds only one distinct segment, R3's runner-up
   distance is infinite (confidence unbounded). R2 instead uses the 16th neighbour's distance as a
   lower bound when no distinct runner-up is found.
5. **02:27 EDT, HMI positive control's coordinate frame (before the final truth run).** The HMI
   dataframe's hand-annotated output synapses did not match their own neurons' proofread meshes as
   given: 0 of 707 fell within 1 um (median 9-121 um per neuron at 8 nm). No 4/8/16 nm scaling and
   no rigid shift fixed it. A y-mirror did: **y_volume = 65000 - y_hmi** (8 nm voxels; 65,000 is the
   volume's y extent) puts 706 of 707 within 1 um of their own mesh (median 0.07 um; a residual
   shift of about -112, 0, -60 nm was left unapplied). The control now uses the mirrored coordinates. The fit used
   only the meshes, never a rule output. **Flag:** anyone reading positions from
   `em_zfish1_dataframe.xlsx` must mirror y. B1's side calls came from CAVE and the neuroglancer
   states (volume frame), not from these columns.

6. **02:20-02:24 EDT, dry run on partial labels (disclosed).** To debug the code, `truth` and
   `evaluate` were run once on the 7,367 of 27,177 chunks read by then. After that run no rule,
   threshold, split, truth definition or verdict rule changed. The only later change was item 5 (the
   HMI mirror), fitted against the meshes. The code fixes made then (a chunk-label lookup that failed
   on unread chunks, a substring match replaced by a set) change no definition. The dry-run outputs
   were overwritten by the final run at 02:32.

## 8. Choices we made (for the honesty page)

Everything here is ours unless a source is named.
1. **Truth.** The authors' `hindbrain_reconstructions` labels at mip 0 are the neuron. Only status
   "axon(reconstructed)" cells; "(c)" is left out because it is undefined.
2. **Sample.** 5 SPNs + 45 draws, `default_rng(20260926)`, 50 in total. Split `default_rng(20260927)`
   into 25/25.
3. **Point mapping.** Pre point 80 nm behind p0 along the line; post point 20 nm past p1. A point votes
   only when its line's two ends read different labels.
4. **Chunks.** Every mip-0 chunk holding a LOD-0 mesh vertex of a sampled neuron. Chunks without
   vertices (slivers) are not read; the absent-chunk check found 0 suspect chunks.
5. **Membership.** f_N(S) > 0.5 over S's whole-brain synapse count. Points never read count as not-N.
6. **Fragments.** Output-bearing agg segments without a soma. Soma-bearing = any nonzero agg id in
   `agglomerated_segments_and_soma_ids.csv`.
7. **Geometry.** Soma-bearing: CAVE v709 soma centres + pre points of outputs + post points of
   inputs. Fragment: pre points of outputs + post points of inputs of segments with outputs.
8. **Rules.**
   - R1: centroid to nearest soma centre.
   - R2: nearest soma-bearing point.
   - R3: principal axis if >= 3 points, >= 80% variance and >= 2 um span; 45-degree cone from both
     ends over the 64 nearest points; otherwise R2.
   - Cap 50 um. Confidence d2/d1 with d1 floored at 100 nm. t in {1, 1.25, 1.5, 2, 3, 5, 10}.
9. **Scope.** R2 runner-ups and R3 were computed for the 1,620,013 fragments with a point within
   20 um of a sampled anchor or won by one. 0 anchor-won fragments lay beyond 20 um. The largest
   64-neighbour radius seen was 12.1 um, inside the 20 um scope for fragments in scope. Out-of-scope
   radii were not checked.
10. **Verdict precision.** Membership-weighted by whole-brain `n_out` (section 7 item 3); the
    pre-registered read precision is reported beside it.
11. **Nulls.** R2's runner-up (scope only); R2 scored against a one-step rotation of the neuron list.
12. **Sample-bias comparison groups.** The HMI soma-only cells, and 2,000 random CAVE somas
    (`default_rng(20260929)`) inside the box of the HMI cells' CAVE positions, excluding HMI cells.

## 9. Limits

- **The sample is not the population** (section 3). The damage measured here is an upper bound on
  health for typical cells, and the rules were tested where the anchors are big. With a median of 2
  inputs, a typical soma segment offers even less arbor to attach to. We expect the rules to do
  worse there (not tested).
- Truth reaches only as far as the reconstructions. They stop at about x = 88,350 (8 nm), so SPN
  spinal outputs are outside the test.
- About 15-20% of true outputs are lost to read noise (the control's 593/707 and the anchors'
  78.5%). The loss hits every category alike, so the shares hold (expected). Single-synapse
  fragments can be misjudged either way.
- Fragments without synapses are invisible here. "Fragments per neuron" means synapse-bearing
  fragments.
- Three simple rules were tried, as capped by the task. Shape-aware rules would need the
  agglomeration's meshes or skeletons of every fragment (heavy supervoxel-graph work). Examples:
  continuity of the fragment's own mesh with the cut face, or a learned agglomeration. They were
  not tried.
- The GCS link was slow on the night (first byte 4-10 s per request). The mip-0 reads took about 20
  minutes; a replay pays that again.
- Region-edge fragments (6.4% of synapses lie outside the region box) have part of their geometry
  missing. They are far from every sampled neuron.

## Reproduce

From `launches/fishbrain/brain` (about 40 min on this link):

    .venv/bin/python -m fishbrain.repair_probe select
    .venv/bin/python -m fishbrain.repair_probe region
    .venv/bin/python -m fishbrain.repair_probe validate      # and validate80
    .venv/bin/python -m fishbrain.repair_probe label_mp      # resumable: data/repair/label_parts/
    .venv/bin/python -m fishbrain.repair_probe rules_compute
    .venv/bin/python -m fishbrain.repair_probe truth
    .venv/bin/python -m fishbrain.repair_probe evaluate
    .venv/bin/python -m fishbrain.repair_probe report
