# Evidence

Start at **[index.md](index.md)**: the plain-language record of the brain gate (G1, G1b, G1c), then one note per
test. Every test fixed its rules and pass marks before it saw a result, and each note says when. One deviation is
disclosed: a G1b readout rule set aside after the data were seen ([G1b-readouts.md](G1b-readouts.md) s.4). Checks
added after a result are labelled post hoc.

| note | what it holds |
|---|---|
| [G0-licence.md](G0-licence.md) | what Fish1's data policy does and does not grant |
| [G1-pull.md](G1-pull.md), [G1-cells.md](G1-cells.md), [G1-sim.md](G1-sim.md) | the wiring download, the cells, the simulator |
| [G1-gate.md](G1-gate.md) | G1: the wiring as published. **FAILED** |
| [G1b-structure.md](G1b-structure.md), [G1b-repair-probe.md](G1b-repair-probe.md), [G1b-readouts.md](G1b-readouts.md), [G1b-cave-v709-tectum.md](G1b-cave-v709-tectum.md) | G1b: why, and can it be repaired. **STOP** |
| [G1c-bridge-spec.md](G1c-bridge-spec.md), [G1c-measure.md](G1c-measure.md), [G1c-gate.md](G1c-gate.md) | G1c: fill the gaps with a published model, measure the share. **KILL** |
| [G1d.md](G1d.md) | G1d: a separate eye model. **Not promoted** (summary; full record in the next release) |
| [engine-speed.md](engine-speed.md) | how fast the engine runs |
| [hashes.md](hashes.md) | sha256 of every data file |

**Data** is in [data/](data/). The notes cite it by its working path: `../g1c-data/gate/x` in a note is
`data/g1c-data/gate/x` here, and the `public/data/` that [hashes.md](hashes.md) describes is `data/` here
(check it with `cd evidence/data && shasum -a 256 -c SHA256SUMS`). `data/SHA256SUMS` pins every file.

**Names and paths in the record.** Each note's `publication_copy` line says what was removed for publication:
owner, account and credential details, and local machine paths. Internal role and work-item labels were kept as
recorded: review roles ("decision-brief", "verifier", "reviewer"), work items ("task B", "task S") and
"lanes" (parallel workstreams). Times are the analysis machine's clock (EDT). `launches/fishbrain/` in a note or
a pinned file is this repository's root, so `cd launches/fishbrain/brain` means `cd brain`. Commit hashes the
notes cite are in the project's working history, which is not public; [hashes.md](hashes.md) says so. The pinned
files cannot change without breaking their hashes, so all of this is left as recorded.
