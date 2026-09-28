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

**Names and paths in the record.** Every copied note (all but [index.md](index.md), [hashes.md](hashes.md) and
[G1d.md](G1d.md), which were written for publication) carries the same sentence in its `publication_copy` line, the
publication edit's own summary: "Local paths made repo-relative; machine-owner, internal coordination, account and
credential details removed." The notes still contain, as recorded: workstream ("lane") ownership statements such
as "No file owned by another lane was edited"; review notes and roles ("Verifier correction", "reviewer", "page
owner"); and work items ("task B", "task S"). The pinned `bridge_spec.json` also contains "decision-brief agent"
and "session scratchpad". Times are the analysis machine's clock (EDT). `launches/fishbrain/` in a note or a
pinned file is this repository's root, so `cd launches/fishbrain/brain` means `cd brain`. Commit hashes the notes
cite are in the project's working history, which is not public. The data and kit files are sha256-pinned and
cannot change without breaking their hashes; the notes are left as published, so every number in them stays
exactly as recorded.
