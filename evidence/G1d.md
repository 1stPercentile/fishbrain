---
title: G1d, a separate eye model. Not promoted.
created: 2026-09-27
status: complete. gate_pass = false, promotion = false.
scope: >
  A summary of the G1d record. The full record (pre-registration, frozen specifications, independent reviews and the
  sealed raw trial package) is in the next release of this repository. The pre-registered specification's sha256 is
  given below so that the file, when it is published, can be checked against this note.
---
# G1d: a separate eye model

**Question.** G1c's fish fled a *receding* shadow as hard as a looming one, and failed its control. Its eye model
subtracts the surround from signed transients, so a brightening point can drive OFF input in its neighbours. G1d
tests one change: split the OFF and ON transients *before* the surround. Same adaptation, surround, retinotopy, rate
cap and constants. ON gain stays 0. The measured wiring, the readouts and the bridge gains (1, 4) do not change.

**It is a modelling hypothesis.** Zebrafish work supports rectified bipolar terminals (Esposti et al. 2013) and
loom-selective responses (Temizer et al. 2015; Dunn et al. 2016). None of it proves this equation is the fish's own
computation. A pass would be a property of the eye model, not recovered Fish1 wiring.

**Pre-registered** on 2026-09-26, before any behavioural trial. Specification sha256
`5d9a960a2f215f424c251f97be392e7234a48a56b438a1b8f9fcefc95349c4e8`. It pins the code, the network, the data, the
conditions, the seeds and a stopping rule: a 16-trial development screen first; any failed check stops the candidate
with no tuning.

## Result

| stage | trials | result |
|---|---|---|
| development screen | 16 | all 14 frozen checks passed |
| confirmatory core, fresh conditions | 216 | passed its frozen checks: 64/64 looms escaped on the correct side in time; 64/64 prey trials struck and turned toward the prey; 0/64 escapes while the disc receded |
| supplement: lesions, measured-wiring shuffles, bridge ablations, diagnostics | 1,360 | every row agrees with independent scoring |

**Why it is not promoted.**
- **The same behaviour survives shuffled wiring.** All ten measured-wiring shuffles keep both behaviours, which
  crosses the pre-registered model-disclosure threshold. Measured-wiring specificity is unsupported.
- **It is not general expansion selectivity.** A static dark disc and a translated dark disc each cause 16/16
  escapes. All 64 receding presentations still fired the escape system at the static onset, before the disc moved.
- **Attribution.** Measured excitatory-link drive: **0.0450469%** (loom), **0.00823497%** (prey),
  **0.00882306%** pooled. The model's literature pathways (bridge class c: the escape crossing and the prey
  relays) provide **99.9910%** pooled. These are G1d's own numbers. G1c's 0.09% / 0.01% do not transfer to G1d,
  and G1d's do not transfer back.

Nulls and diagnostics were registered disclosures, not extra kill criteria; the candidate's registered behaviour and
lesion conditions pass. The fish in this repository, and the one `python -m fishbrain.demo` runs, remains the G1c fish.
