---
title: FISHBRAIN brain gate, the public record
created: 2026-09-26
status: complete. G1 FAILED (2026-09-25) -> G1b STOP (2026-09-26) -> G1c KILL (2026-09-26).
scope: >
  A plain-language summary of three pre-registered tests of whether a published zebrafish brain wiring map can
  drive a simulated fish's escape and prey-capture decisions, with links to the edited source notes and the
  sha256 of every data file.
---
# FISHBRAIN brain gate: the public record

**About these files.** The notes in this folder are edited-for-publication copies of the project's working
evidence notes (same filenames, one folder up). Local paths were made repo-relative, and machine-owner, internal
coordination, account and credential details were removed. **No number, hash, commit, time, method, test count,
failure, control, caveat or reviewer correction was changed.** Where a copy corrects its source, the change is
marked *[Publication note]*. Each copy names the commit it was taken from in its `publication_copy` line.

## The short version

We asked a simple question: **can the published wiring of a real larval zebrafish brain, run as a simulation, make
the fish's two basic visual decisions?** The two decisions are to flee a looming shadow toward the correct side,
and to strike and turn toward a small moving prey dot. Every test fixed its rules and pass marks before it saw a
result, as each note records. The pre-registration table below shows which of those times git confirms
independently, and it lists one disclosed deviation (a G1b readout rule set aside after the data were seen).

| test | date | what it asked | result |
|---|---|---|---|
| **G1** | 2026-09-25 | Does the measured wiring, run as published, flee a loom and turn toward prey? | **FAILED.** Nothing downstream of the first visual stage fired in any of 104 trials. |
| **G1b** | 2026-09-26, 01:25-03:25 EDT | Why? Does the wiring carry the signal at all, and can the cut axons be repaired? | **STOP.** The released map cannot carry vision to the decision cells, and simple repair does not work. |
| **G1c** | 2026-09-26, 03:39-10:02 EDT | If a published pathway model fills the gaps, does the fish behave, and how much of each decision runs on measured wiring? | **KILL.** The fish acts correctly on new stimuli but also flees a *receding* shadow, which fails a pre-registered control. Only **0.09%** of the escape decision and **0.01%** of the strike-and-turn decision ran on measured synapses. |

In plain words: **the anatomy is real, but the decisions run through the model.** The measured Fish1 wiring
supplies the fish's visual map. It does not carry that map to the cells that decide.

## The headline numbers (G1c)

The **decision-path share** is the metric fixed in advance to answer "is the fish's behaviour mostly true to the
real wiring?". In each trial, it follows the synaptic drive that reached the cells making the decision, and
measures the share that travelled only through measured Fish1 synapses from measured-driven sources. A link
supplied by the model passes on 0.

| decision | decision-path share (measured) | carried by the literature model, class (c) | trials |
|---|---|---|---|
| **Escape from a looming shadow** (the escape cell that fired first) | **0.090%** (m = 0.000895) | 99.89% | 64 |
| **Strike and turn toward prey** (every deciding cell that fired), pooled | **0.010%** (m = 0.000101) | 99.99% | 64 |
| of which strike / turn | 0.018% / 0.0058% | | |
| all 128 held-out trials | 0.011% | 99.99% | 128 |

- **Before the escape cell's first spike, the measured share is 0%.** In all 192 initiating-cell first spikes across
  the 64 held-out loom trials, the spike came entirely from the model's crossing pathway (m = 0).
- Looms from behind (±135 degrees) had a measured share of exactly 0. Looms at ±45 degrees: 0.15-0.20%.
- **The synapse-count share is secondary and was pre-registered as never the headline.** It counts synapses, not
  decisions, so it flatters the wiring:

  | network | measured share of synapses |
  |---|---|
  | whole brain of record plus the bridge | **99.90%** (29,474,316 measured, 30,234 bridge synapse-equivalents) |
  | the network the gate actually simulates (10,528 cells) | 63.26% |
  | only the synapses onto the deciding cells | 13.52% |

Source: [G1c-gate.md](G1c-gate.md) sections 5 and 6; data in `../g1c-data/gate/`.

## What each test asked, and what happened

### G1: does the wiring work as published? (2026-09-25, [G1-gate.md](G1-gate.md))
- **Setup.** The whole Fish1 wiring was downloaded and pinned: 29,474,316 synapses, equal to the authors' own
  database count ([G1-pull.md](G1-pull.md)). The visual entry (the optic tectum), the two Mauthner escape cells and
  50 named spinal projection cells (turning and forward swimming) were identified ([G1-cells.md](G1-cells.md)). The brain ran as a spiking network with the
  parameters of a published whole-brain fly model (Shiu et al. 2024), built and tested for exact replay
  ([G1-sim.md](G1-sim.md)). A simple model eye turned stimuli into input to the tectum.
- **Pre-registered** at 2026-09-25 21:05 EDT, before the first stimulus ran: pass marks for the escape's
  probability, side and timing; receding, dimming and blank controls; a Mauthner lesion; and a whole-brain shuffle.
- **Result: FAIL.** In 104 trials (64 intact, 8 lesioned, 32 shuffled), strong visual drive produced zero spikes
  beyond the stimulated tectal cells. The one allowed change (synapse weight x7.23, anchored on synapse density)
  was predicted in writing to fail before it ran, and it failed in all 64 trials.
- **Why.** Fish1 is an automated reconstruction that leaves most axons detached from their cell bodies. Only
  **1.61%** of synapses come from a segment that holds a cell body, and 81% of the functional connections are a
  single synapse. At the published weights a cell needs about 25 synapses firing together.
- **Positive control.** At a weight 73x the published one, the escape cells do fire, but more on the wrong side
  for the stimulus. That matches a 2:1 left bias in the wiring, and it is the opposite of what the literature
  predicts (Dunn et al. 2016).

### G1b: is the signal there at all, and can it be repaired? (2026-09-26)
- **Structure** ([G1b-structure.md](G1b-structure.md); pre-registered 02:27 EDT): compared with degree-preserving
  shuffled wiring, the real wiring separates neither left from right (0 of 3 readouts) nor loom from prey (0 of 2
  contrasts). The verdict is **ABSENT**. The positive controls pass: inside the tectum the same measures find strong
  structure (+1.000 and +0.968). **Disclosed:** the pre-registered null test could not have reached its bar in the
  predicted direction, so the result rests on magnitudes. Only 0.35-3.1% of a decision cell's input synapses come
  from cells within 3 hops of the tectum, and 68-81% come from orphan axon fragments that nothing can drive.
- **Repair** ([G1b-repair-probe.md](G1b-repair-probe.md); pre-registered 01:35 EDT): three nearest-neighbour rules
  for reattaching cut axons were scored against 50 neurons the authors proofread by hand. The needed precision was
  80%. The best reached **0.005** (0.043 by the pre-registered read-based measure), which is null level. The verdict
  is **NOT_VIABLE**. Nearby arbors are usually the cell the axon *synapses onto*, not the cell it came from.
- **Readouts** ([G1b-readouts.md](G1b-readouts.md)): bilateral decision-cell populations from the Fish1 atlas, the
  two further escape cells (MiD2cm, MiD3cm), and a strike rule fixed at 01:39:52 EDT. A pre-registered
  identification rule was set aside after seeing the data; the deviation and both pick sets are disclosed.
- **The newer, partly proofread database version (CAVE v709) is the same** ([G1b-cave-v709-tectum.md](G1b-cave-v709-tectum.md)):
  sampled tectal cells have a median of 0 outputs (76.7% have none), and none reach the hindbrain.

### G1c: fill the gaps, then measure how much is real (2026-09-26)
- **The metric** (decision-path share, above) was committed to git before any G1c measurement (see the next section).
- **Measure first** ([G1c-measure.md](G1c-measure.md)): about two dozen axon fragments with an observed tectal
  origin reach the escape cells. On the Mauthner cells they land only on the visual (ventral) dendrite, and every
  one of their synapses but one is uncrossed (the crossed one belongs to a non-tectal cell). The retina holds 0 synapses in the volume, so the eye is always a model. **Reviewer correction,
  kept:** an adversarial reviewer showed the note's "upper bounds" depended on a 3-hop cutoff (0.62% at 3 hops,
  4.9% at 4, 13% at 5, 20% at 6 in the synapse view), so "under 1%" was withdrawn as a bound, and the real number
  was left to the simulation.
- **The bridge** ([G1c-bridge-spec.md](G1c-bridge-spec.md), frozen 05:34:57 EDT): 1,780 added connections and 0
  added cells. Class (a) links the observed fragments to their tectal columns (646 edges). Class (b) was not used.
  Class (c) is the literature pathway (1,134 edges): the escape crossing, whose relay no paper identifies, is a
  labelled functional edge set; the prey pathway uses one real pretectal relay cell per side (Antinucci et al.
  2019).
- **The gate** ([G1c-gate.md](G1c-gate.md), 08:57-10:02 EDT): the two class gains were tuned on training stimuli
  only (chosen g_a = 1, g_c = 4, 10 of 10 training checks). A hash check shows no held-out or control stimulus
  entered tuning. On held-out stimuli:

  | item | result |
  |---|---|
  | H1, escape from new looms | PASS: 64/64 escapes, correct side 1.00 |
  | H2, strike and turn toward new prey | PASS: 64/64 strikes, turn toward the prey 1.00 |
  | **C1, receding shadow (control)** | **FAIL:** 92.5 and 92.75 Mauthner spikes per trial, against the loom's 65.1 and 63.6; needed at most 0.5x |
  | C2, dimming (control) | PASS, passed by the eye model, not by the brain |
  | C3, blank (control) | PASS, by construction (no background input) |
  | L1, L2, lesions | PASS |

  The frozen rule says KILL when H1, H2 and C1-C3 do not all hold. **C1 fails, so the verdict is KILL.** Why it
  fails: the model eye responds to a shrinking dark disc more strongly than to a looming one, and the bridge has
  no direction selectivity. That is a property of the chosen eye and bridge, not of the measured wiring. The one
  allowed global change was left unused, because a fix seen only on held-out data could not produce a held-out
  result.
- **Mandatory publications, all run.**
  - **M1:** the measured wiring shuffled with the bridge fixed. 10 of 10 shuffled fish behave exactly like the
    intact one, so, by the pre-registered rule: *"the anatomy is real; the decision runs through the model."*
  - **M2:** remove the bridge and not one deciding cell spikes in 128 trials. Class (a) is inert. Class (c) is the
    whole behaviour.
  - **Cross-talk (reported, not scored):** every loom trial also triggers the strike rule (64/64).
- **The pre-registered disclosure sentence applies:** *"Which way the fish flees is set by the published pathway
  model, not by the measured wiring."*

## Pre-registration: what was fixed when, and how to check it

| item | fixed (as stated) | independent evidence | ran |
|---|---|---|---|
| **G1c headline metric**, disclosure sentence, bridge classes, the only kill | 2026-09-26 03:39 EDT | **git commit `f9c6fe9eb`, 2026-09-26 03:39:02 -0400** (project brief, section "G1c: fill the gaps"; text below) | measurement from 03:54, gate from 08:57 |
| **G1c bridge spec** (every rule, list, edge, gain grid, gate threshold) | 2026-09-26 05:34:57 EDT | `bridge_spec.json` sha256 `185b378031a3f3d730ed90bfe9cd56aede6aba6ad0c0984032e03a2c5e111499`, committed in the frontmatter of G1c-bridge-spec.md at **`4861cf1c6`, 05:38:37 -0400**; the note is unchanged since, and the JSON committed later (`e2881d4be`) has this sha256 | 08:57-10:02 EDT |
| G1c measure plan (section 0 of G1c-measure) | 03:52 EDT | git `194971ed5`, 04:05:18 -0400, holds section 0 and no results. **This commit is later than the 03:54 start of measurement**, so the 03:52 time is self-reported. | 03:54-04:35 |
| G1b structure test (section 0) | 02:27 EDT | git `c7c91e826`, 02:27:13 -0400, holds section 0 only | first real run 02:32 |
| G1b repair probe (section 0) | 01:35 EDT | git `edc6b4f48`, 01:54:54 -0400, holds section 0 only | 01:35-02:40; the first check (01:57) and every truth and rule computation came after that commit |
| G1b strike rule, M-system rule | 01:39:52 EDT | self-reported in G1b-readouts; the note was first committed after its results | 01:25-02:15; the r5/r6 boxes the M-system rule governs were read from 01:48:30 |
| G1 protocol and criteria | 2026-09-25 21:05 EDT | self-reported in G1-gate; the note was first committed after its results (`31e9971b4`, 23:09:50) | 21:14 onward |

Commit shas can be checked only against the project's git history. File hashes can be checked against any copy
([hashes.md](hashes.md)).

**The G1c metric as committed in `f9c6fe9eb`** (verbatim; "the page" is the project's public page):

> **Metric that decides "mostly true" (the headline number, published either way):** the **decision-path share**.
> In each gate trial's decision window, every neuron carries a provenance value m in [0, 1]: the share of the
> synaptic drive it received that travelled through **measured** Fish1 synapses from measured-driven sources. A
> measured synapse passes on its source's m; a bridged link passes on 0. The eye model's input is labelled
> *chosen (eye model)* and scored separately, not counted as measured wiring. The decision-path share is the
> drive-weighted mean m over the readout cells that decide (the escaping M-system cell, and the strike and turn
> populations). It is reported per readout and per bridge class.
> **Secondary, never the headline:** the synapse-count share (measured synapses / all synapses in the simulated
> network). It will be near 100% and says little on its own.
> **Pre-registered disclosure sentence**, used if the bridge supplies the crossing Dunn 2016 requires: *"Which
> way the fish flees is set by the published pathway model, not by the measured wiring."* If the bridge is built
> without the crossing and the fish escapes toward the threat, the page says that instead. No tuning to hide
> either.
> **Bridge classes, each counted and coloured separately:**
> - (a) Fragments whose tectal origin is *observed* in the volume, linked to that tectal column's activity.
> - (b) Fragments with no observable origin, assigned by tract topography (model).
> - (c) Missing relays and the midline crossing, taken from the literature model.
>
> Tuning is allowed only on bridge gains (at most 1 gain per class, logged). Measured synapses keep Shiu's weights.
> **Mandatory publications, not kill criteria:**
> - Shuffle the measured portion with the bridge fixed. If behaviour is unchanged, the page says "the anatomy is
>   real; the decision runs through the model".
> - Bridge ablation.
> - Both metrics, per class.
>
> **The only kill:** the fish can't escape or strike correctly even with the bridge, on held-out stimuli with the
> controls holding.

## What is measured, what is inferred, what is model

| measured (from Fish1) | inferred from data | chosen (model) |
|---|---|---|
| Every synapse's pre- and post-synaptic segment, position and type (29,474,316) | Which synapse type is excitatory: type 2, from the authors' ground-truth cells (68.7% of true-excitatory synapses are type 2; a reviewer flagged the habenular subsets as circular, and the unbiased tally is the one used) | Neuron model and parameters (Shiu et al. 2024, fitted to the fly, not the fish) |
| Cell-body positions and the authors' atlas regions | The identity of the escape and steering cells (largest soma in the atlas region, input count, shape; the left MiD3cm pick is contested) | The eye model, the stimuli and the retinotopic map (placements by rank, not measured receptive fields) |
| Where fragments start and end in the volume | Left and right, from the authors' layer names (a mirror error would flip every side consistently) | The bridge: classes, edges, the unit U = 34, gains (tuned), and the pretectal relay choice |
| That the retina holds no synapses in the volume | | Readout rules and every pass mark |

The eye's input is always labelled *chosen (eye model)* and never counted as measured.

## Controls, failures and corrections kept in the record
- G1: the no-stimulus control and the lesion "passed" only because nothing fired; both are marked vacuous.
- G1b: the structure test's null comparison had no power in the predicted direction (disclosed with a table);
  one pre-registered identification rule was set aside after seeing the data (disclosed with times).
- G1c-measure: the reviewer correction above; several rows marked post hoc.
- G1c: the receding-shadow failure decides the verdict. The spec's authors did not foresee it, and the note says so.
- G1-pull: a reviewer showed `pre_synaptic_confidence` is a per-axon value, not a per-synapse score. Struck
  through and corrected in place.
- G1b-readouts corrects G1-cells: the Mauthner cells hold the last two ids because ids follow soma size, not
  because they were "added by hand".
- G1c-gate: its `supersedes` line names the wrong commit for the record text; a *[Publication note]* gives the
  git history.
- The project brief records that two independent reviewers confirmed the G1 failure, and three confirmed the G1c
  numbers and held-out discipline. Those reviews are not part of this set.
- One data file added after the verdict shows a failed JS replay check (a run-length rounding bug) and its fixed
  re-run; see [hashes.md](hashes.md), note 3.

## Limits
- One machine. Replay is bit-exact across processes and between the Python and JS kernels on that machine;
  cross-machine replay has not been run.
- The data are the public automated reconstruction (`seg_241003_agg241003`). CAVE v709 was checked for tectal
  attachment only.
- The left-side steering population rests partly on unconfirmed mirror candidates and on the atlas.
- The eye model has no loom or direction selectivity, which is what fails C1.
- Nothing here shows how a living zebrafish would behave. It shows what this wiring, with these model choices,
  does.

## Source data and credit

**Brain wiring: Fish1** (Petkova, Januszewski et al., 2025, *A connectomic resource for neural cataloguing and
circuit dissection of the larval zebrafish brain*, bioRxiv, doi [10.1101/2025.06.10.658982](https://www.biorxiv.org/content/10.1101/2025.06.10.658982v1)),
released by the Lichtman and Engert laboratories at Harvard University with the Connectomics team at Google.
**Used without endorsement.**

- Brain of record: `gs://fish1-public/syn_241003_agg241003_reorient_axde_ei_bayes_idx_pre_250410.precomputed`,
  wiring content hash `09a71038502546afd12082fbde9f538fdc0704e940de33e860db1f0962d25af2` (recipe in
  [hashes.md](hashes.md)).
- Licence: the Fish1 data carry no stated licence. The data policy asks for the citation above and addresses
  research use. The CC-BY-NC 4.0 licence on bioRxiv covers the manuscript, not the data
  ([G0-licence.md](G0-licence.md); not legal advice).

## Files

| note | what it holds | its data |
|---|---|---|
| [G0-licence.md](G0-licence.md) | licence and attribution check | none |
| [G1-pull.md](G1-pull.md) | the wiring download, pinning and integrity | `../g1-data/integrity.json` |
| [G1-cells.md](G1-cells.md) | axes, escape cells, spinal projection cells, tectum, retinotopy | none in this set |
| [G1-sim.md](G1-sim.md) | the spiking engine, determinism, stimuli | none |
| [G1-gate.md](G1-gate.md) | G1 protocol, results, benchmark | `../g1-data/` |
| [engine-speed.md](engine-speed.md) | the activity-gated engine, bit-identical | none |
| [G1b-readouts.md](G1b-readouts.md) | bilateral readouts, escape system, strike rule | `../g1b-data/readouts.json` |
| [G1b-structure.md](G1b-structure.md) | structure vs shuffled wiring | `../g1b-data/structure-*.json` |
| [G1b-repair-probe.md](G1b-repair-probe.md) | axon reattachment against proofread cells | `../g1b-data/eval.json`, `truth.json`, `validate.json` |
| [G1b-cave-v709-tectum.md](G1b-cave-v709-tectum.md) | the newer database version check | none |
| [G1c-measure.md](G1c-measure.md) | what is real before bridging | `../g1c-data/meshes.json`, `oa_fragments.json`, `reach.json`, `share.json` |
| [G1c-bridge-spec.md](G1c-bridge-spec.md) | the frozen bridge spec | `../g1c-data/bridge_spec.json` |
| [G1c-gate.md](G1c-gate.md) | the bridged fish, gate, share, publications | `../g1c-data/gate/` |
| [hashes.md](hashes.md) | sha256 of all 31 data files and the pinned identifiers | |

**Path conventions.** Paths inside the notes are relative to `launches/fishbrain/` (for example
`brain/data/...`). Everything under `brain/data/` is git-ignored working data. The checkable copies are the
`evidence/g1-data/`, `evidence/g1b-data/` and `evidence/g1c-data/` folders listed in [hashes.md](hashes.md). "The
brief" in the notes is the project's working brief, which is not part of this set; its G1c pre-registration is
quoted in full above.
