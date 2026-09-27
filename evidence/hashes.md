---
title: FISHBRAIN brain-gate evidence, hashes
created: 2026-09-26
scope: sha256 of every file in launches/fishbrain/evidence/g1-data/, g1b-data/ and g1c-data/ (31 files), plus the pinned identifiers of the brain of record and the frozen G1c bridge spec.
computed: 2026-09-26 at repository HEAD 548e95d78. `git status` and `git diff HEAD` showed no change under launches/fishbrain/evidence/ other than this public/ folder, so every file below equals its committed version.
---
# Hashes

Paths are relative to `launches/fishbrain/evidence/`. The commit column is the last commit that touched the
file. Commit shas can be checked only against the project's git history; the sha256 values can be checked
against any copy of the files.

## The brain of record

| what | value |
|---|---|
| public source (no login) | `gs://fish1-public/syn_241003_agg241003_reorient_axde_ei_bayes_idx_pre_250410.precomputed` (HTTPS: `https://storage.googleapis.com/fish1-public/syn_241003_agg241003_reorient_axde_ei_bayes_idx_pre_250410.precomputed/`) |
| wiring content hash | `09a71038502546afd12082fbde9f538fdc0704e940de33e860db1f0962d25af2` |
| synapses / (pre, post) pairs / segments | 29,474,316 / 24,921,936 / 13,458,709 |
| equals CAVE `synapses_axde` v709 | yes, difference 0 (G1-pull) |
| segmentation (agglomeration) | `gs://fish1-public/seg_241003_agg241003` |

**How the content hash is computed** (G1-pull, `pull.content_hash()`): sha256 over the little-endian column
bytes of `syn_id` (u64), `pre` (u64), `post` (u64) and `type` (u8), in that order, with rows in `syn_id`
order (0 to 29,474,315, no gaps). It does not depend on the parquet encoding. A reader who pulls the layer
with `fishbrain.pull` can recompute it.

**The 21 downloaded source files** are pinned in G1-pull.md ("Source files: bytes, generation, MD5,
sha256"): each with its bytes, GCS generation, GCS MD5 and sha256, 2,539,301,126 bytes in all. Every
download requested the listed generation. The derived parquet files (`synapses_raw.parquet`,
`synapses.parquet`, `cells.parquet`) are git-ignored; their sha256 values in G1-pull depend on the pyarrow
version, and the content hash does not.

## The frozen G1c bridge spec

| what | value |
|---|---|
| `g1c-data/bridge_spec.json` sha256 | `185b378031a3f3d730ed90bfe9cd56aede6aba6ad0c0984032e03a2c5e111499` |
| frozen at (stated in the spec) | 2026-09-26 05:34:57 EDT |
| first commit carrying this sha256 | `4861cf1c6`, 2026-09-26 05:38:37 -0400 (the frontmatter of G1c-bridge-spec.md; that note is unchanged since) |
| first commit of the JSON itself | `e2881d4be`, 2026-09-26 10:19:35 -0400 (with the gate data) |
| gate runs | 2026-09-26 08:57-10:02 EDT, after the freeze |
| bridge version | `g1c-bridge-v1` |
| edge-list sha256 (1,780 edges) | `f7f810d56059860dc81d7fa7f064cb248779acec83e00ce88efc6facc8100bf0` |

The edge-list hash recomputes from the JSON alone (checked while writing this file):

```python
import json, hashlib
d = json.load(open("g1c-data/bridge_spec.json"))
rows = [[e["pre"], e["post"], e["base"], e["sign"], e["cls"], e["pathway"]]
        for e in sorted(d["edges"], key=lambda e: (e["cls"], e["pathway"], e["pre"], e["post"]))]
print(hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest())
# f7f810d56059860dc81d7fa7f064cb248779acec83e00ce88efc6facc8100bf0
```

## Other pinned identifiers

| what | value | where stated |
|---|---|---|
| LIF parameter digest (Shiu defaults, dt 0.1 ms) | `958a1404a0e6dfab7682b5f9c4419aa5d3e25963cc18f582cbd99a4d69284717` | G1-sim s.2, G1-gate s.3 |
| G1 eye-model (retina) parameter digest | `eca7c6a1ca385933ea3d22b753605f709c0b727ae4950fdf2b28542ad3867ce6` | G1-gate s.3 |
| G1c stimulated set, sorted uint64 ids | `8910b216619f7c02310895c16ff67bcd61cc3f612c79f14f64b5fe4507bc2348` | bridge_spec.json, G1c-gate s.1 |
| mece3_231218 array at 4096 nm | `795292769933e360a3bce8188a773b81f4ea5f8e388ce0717f2750da5ca48e49` | bridge_spec.json, G1c-gate s.1 |
| G1c gate network digest (g_a 1, g_c 4) | `0b6f3369f21f5c626d9ba1ba4bc02dfefd1e51b945b3355fcd6730c9aa083b4a` | js_parity.json, G1c-gate s.1 |
| G1c bridged digest | `279ae5fa21453de8dc8b320b6c7c5e4527877b19db45d137a0e45575dbb5b36a` | js_parity.json, G1c-gate s.1 |
| G1c edge-provenance digest | `662772eda622a0ca0a3eac6b0667222cccc46b8e54a9a1a04f5a013b1b237b12` | g1c_summary.json, G1c-gate s.1 |

## Data files (31)

| file | bytes | sha256 | last commit | described in |
|---|---:|---|---|---|
| `g1-data/ei_polarity_check.json` | 2,081 | `47d3cfc3fa191afad697cbbeeccfba1773d48a773253a0c2541e12b61d73d2b3` | 31e9971b4 | G1-gate s.2 (E/I value vs ground-truth polarity) |
| `g1-data/g1_bench.json` | 3,204 | `1f9af28b12626f0dfb91838a7d5db3eb2fce72a03594e48ec773b221a1456dd1` | 31e9971b4 | G1-gate s.6 (benchmark) |
| `g1-data/g1_change_results.json` | 114,601 | `5fb0e906b76d814635ed5ed6e71cad7b1915f981c6283b37bb0550905a36de8a` | 31e9971b4 | G1-gate s.5.2 (run 2, the one logged weight change) |
| `g1-data/g1_gate_results.json` | 180,818 | `f9afe202442fda896be76b6ebe4ba4048460720b0c8aba241db0956dc6878926` | 31e9971b4 | G1-gate s.5.4 (gate protocol re-run by the pytest fixture) |
| `g1-data/g1_gate_results_run1.json` | 178,827 | `8306577532d178677f95f2f99f7cd4beae723d2f557436108be39db8243072c0` | 31e9971b4 | G1-gate s.5.1 (run 1, the pre-registered model) |
| `g1-data/g1_graph_checks.json` | 2,853 | `57ca5183731f75fdc51f34d7bdf56f93a0f4aee02d84c4a8528cc8638fe5fb66` | 31e9971b4 | G1-gate s.2 (Mauthner and reachability graph checks) |
| `g1-data/integrity.json` | 3,744 | `734273b8855a46262460837b0ecdea378eb8411eff56eb336fb0d44815be9b55` | 31e9971b4 | G1-pull (integrity checks of the pulled wiring) |
| `g1b-data/eval.json` | 36,786 | `f0be7d172684eb8796ca232e3f3eb9b037e4da441184d4f5517fec473bf45014` | c601f956d | G1b-repair-probe s.4 (rule evaluation) |
| `g1b-data/readouts.json` | 871,128 | `81ea2fc9df281964ef5f527adec6a8be8c619f6b66f076b44d3a025944c3dee6` | c601f956d | G1b-readouts (populations, M-system, rules, wiring checks) |
| `g1b-data/structure-replay_check.json` | 118 | `91e53133ef6dda6389bc234ecc32c3f0ac89f164c30140a34a4e6a1b73095c47` | c601f956d | G1b-structure s.3 (A-00 shuffle replay) |
| `g1b-data/structure-structure.json` | 896,316 | `41483e8dd772d5f1d3cc54987f6ce52813b305631bb07169771e4c8104774999` | c601f956d | G1b-structure (real values, nulls, verdict, power) |
| `g1b-data/truth.json` | 53,381 | `9905ffb0e4bda4b6a27eac4bbf3dc229b6b3f92148e73f919d54a3142c5eb4f1` | c601f956d | G1b-repair-probe s.3 (ground truth, damage) |
| `g1b-data/validate.json` | 402 | `aacfc4d25722c15ef74c2b37d9bb8fe23071e2a802eab3b41abfb14e93dc85bb` | c601f956d | G1b-repair-probe s.1 (point mapping check) |
| `g1c-data/bridge_spec.json` | 407,933 | `185b378031a3f3d730ed90bfe9cd56aede6aba6ad0c0984032e03a2c5e111499` | e2881d4be | G1c-bridge-spec (the frozen spec) |
| `g1c-data/gate/choices.json` | 2,257 | `44737284facf9d9cb086e65c37ddc2f802cbb2ac1b83969bade7aae172cc638d` | e2881d4be | G1c-gate s.2, s.13 (operationalisation choices, 09:00:40 EDT) |
| `g1c-data/gate/g1c_summary.json` | 39,608 | `49aab48614485eebc0f1991ac73727fa8eef573855751e12eba62618da42530e` | e2881d4be | G1c-gate (summary of every item) |
| `g1c-data/gate/gate_raw.json` | 3,353,703 | `ca7b8be162107b900c9cc36c4c3691950d994adcc3aa3b139bbcd6835ca68a77` | e2881d4be | G1c-gate s.3 (raw trials) |
| `g1c-data/gate/gate_raw_phase1.json` | 299,346 | `facd3e7dd648dd7ffeb18ca95b60760b6e69c03aa290eca8d2ec93991ff4980a` | e2881d4be | G1c-gate s.3 (raw trials, first phase) |
| `g1c-data/gate/gate_results.json` | 3,379,529 | `7bbb89104fba55945d324aacccc188b61092ad1f8c9aa3e212327037ff34980b` | e2881d4be | G1c-gate s.3, s.5, s.6 (gate items, shares) |
| `g1c-data/gate/js_parity.json` | 1,182 | `32a224ab51eaa125ce0c1f1969c355da39ba3378ac1c92364cdb4e41c18620c0` | e2881d4be | G1c-gate s.10 (JS parity, prey case, 09:30:20 EDT) |
| `g1c-data/gate/js_parity_loom.json` | 1,184 | `113d0b7a30ec6c4040146b7e339771ec6042cba17266aff61891d2d344a4f796` | 312198c68 | after the verdict: JS parity on a loom case (see note 3) |
| `g1c-data/gate/js_parity_loom_exact.json` | 430 | `e143af3225d2312212ae2173568568c7e70a6dcec0719adac7d585a6a395e132` | 312198c68 | after the verdict: the same loom case at its exact length (see note 3) |
| `g1c-data/gate/m1_shuffles.json` | 67,281 | `ffb229a2c97758d684d9f7bab8ad896bdcf5c5ea5eeb7afcba313fae0525f070` | e2881d4be | G1c-gate s.7 (M1, measured wiring shuffled) |
| `g1c-data/gate/m2_ablations.json` | 82,422 | `d1ed6d1ce83bd3c75f1d99a21d497fd5bf4ea8693b1868c23b98e23604b0e6eb` | e2881d4be | G1c-gate s.8 (M2, bridge ablations) |
| `g1c-data/gate/rederive.json` | 3,190 | `d121792525404a5df19f401bb52abfa1a6ae550cf86344dc5a66553873a841e0` | e2881d4be | G1c-gate s.1 (independent re-derivation, 76/76) |
| `g1c-data/gate/tuning.json` | 36,818 | `d49e041b708b33ebe74ea08a5e16d62aaf49a8ac77d53fc53e70e434fb7f9b6e` | e2881d4be | G1c-gate s.2 (tuning grid summary) |
| `g1c-data/gate/tuning_log.jsonl` | 150,405 | `f2ce49f7d5af8a3dbff2fe3b71014f655e8cc1438795ea48681fbe9ea1e74121` | e2881d4be | G1c-gate s.2 (append-only tuning log) |
| `g1c-data/meshes.json` | 8,507 | `9f7f697dbc09483f489cdb4b95665e0ebd651b9ea7e715599f51ab42deb866d8` | 74eb5f3f8 | G1c-measure s.1 (mesh sensitivity row) |
| `g1c-data/oa_fragments.json` | 41,085 | `8164859db2faa00b2cec5f784d47929a69786c1bbd525b495f3cedf8a15c2b16` | 74eb5f3f8 | G1c-measure s.1, s.5 (observed-origin fragments) |
| `g1c-data/reach.json` | 888,006 | `cc777a9b62307a73e4320368270a44c327eae2832831e332d63cac7d14d10f95` | 74eb5f3f8 | G1c-measure s.2 (reach from the retinal-afferent set) |
| `g1c-data/share.json` | 32,833 | `fae9bfac419c75543579c6a8a41d1e2d8dc32079157104b7abdce3ebbee44e87` | 74eb5f3f8 | G1c-measure s.4 (predicted decision-path share) |

**Notes.**
1. `g1c-data/bridge_spec.json` is the spec itself; its sha256 above equals the frozen value.
2. G1's gate data (`g1-data/`) was first committed at `31e9971b4` (2026-09-25 23:09:50 -0400), after the G1
   runs. G1b's (`g1b-data/`) at `c601f956d` (2026-09-26 03:25:51 -0400). G1c-measure's at `74eb5f3f8`
   (2026-09-26 04:50:25 -0400). The G1c gate's at `e2881d4be` (2026-09-26 10:19:35 -0400).
3. **Added after the G1c verdict** (commit `312198c68`, 2026-09-26 11:45:36 -0400), an engine check, not a gate
   item. `js_parity_loom.json` (11:42:10 EDT) ran a bridged loom trial (`loom_az+45_lv120`, `G1-seed-0`) in the
   JS kernel and in Python: both gave raster hash `94279d03...b87bb429`, which did **not** match the gate's
   recorded hash (`recorded_spike_hash_matches: false`). The parity runner rounded runs up to whole 500 ms
   chunks (4,500 ms against the trial's 4,395 ms). `js_parity_loom_exact.json` re-ran it at the exact length:
   Python, JS and the gate record all give `340840a6b114c909f3d8d8bc45cc72b869d4b3806ffda310fc4201e702b8120a`,
   342,164 spikes each.

## Checksum list

Run from `launches/fishbrain/evidence/` with `shasum -a 256 -c` (or `sha256sum -c`):

```
47d3cfc3fa191afad697cbbeeccfba1773d48a773253a0c2541e12b61d73d2b3  g1-data/ei_polarity_check.json
1f9af28b12626f0dfb91838a7d5db3eb2fce72a03594e48ec773b221a1456dd1  g1-data/g1_bench.json
5fb0e906b76d814635ed5ed6e71cad7b1915f981c6283b37bb0550905a36de8a  g1-data/g1_change_results.json
f9afe202442fda896be76b6ebe4ba4048460720b0c8aba241db0956dc6878926  g1-data/g1_gate_results.json
8306577532d178677f95f2f99f7cd4beae723d2f557436108be39db8243072c0  g1-data/g1_gate_results_run1.json
57ca5183731f75fdc51f34d7bdf56f93a0f4aee02d84c4a8528cc8638fe5fb66  g1-data/g1_graph_checks.json
734273b8855a46262460837b0ecdea378eb8411eff56eb336fb0d44815be9b55  g1-data/integrity.json
f0be7d172684eb8796ca232e3f3eb9b037e4da441184d4f5517fec473bf45014  g1b-data/eval.json
81ea2fc9df281964ef5f527adec6a8be8c619f6b66f076b44d3a025944c3dee6  g1b-data/readouts.json
91e53133ef6dda6389bc234ecc32c3f0ac89f164c30140a34a4e6a1b73095c47  g1b-data/structure-replay_check.json
41483e8dd772d5f1d3cc54987f6ce52813b305631bb07169771e4c8104774999  g1b-data/structure-structure.json
9905ffb0e4bda4b6a27eac4bbf3dc229b6b3f92148e73f919d54a3142c5eb4f1  g1b-data/truth.json
aacfc4d25722c15ef74c2b37d9bb8fe23071e2a802eab3b41abfb14e93dc85bb  g1b-data/validate.json
185b378031a3f3d730ed90bfe9cd56aede6aba6ad0c0984032e03a2c5e111499  g1c-data/bridge_spec.json
44737284facf9d9cb086e65c37ddc2f802cbb2ac1b83969bade7aae172cc638d  g1c-data/gate/choices.json
49aab48614485eebc0f1991ac73727fa8eef573855751e12eba62618da42530e  g1c-data/gate/g1c_summary.json
ca7b8be162107b900c9cc36c4c3691950d994adcc3aa3b139bbcd6835ca68a77  g1c-data/gate/gate_raw.json
facd3e7dd648dd7ffeb18ca95b60760b6e69c03aa290eca8d2ec93991ff4980a  g1c-data/gate/gate_raw_phase1.json
7bbb89104fba55945d324aacccc188b61092ad1f8c9aa3e212327037ff34980b  g1c-data/gate/gate_results.json
32a224ab51eaa125ce0c1f1969c355da39ba3378ac1c92364cdb4e41c18620c0  g1c-data/gate/js_parity.json
113d0b7a30ec6c4040146b7e339771ec6042cba17266aff61891d2d344a4f796  g1c-data/gate/js_parity_loom.json
e143af3225d2312212ae2173568568c7e70a6dcec0719adac7d585a6a395e132  g1c-data/gate/js_parity_loom_exact.json
ffb229a2c97758d684d9f7bab8ad896bdcf5c5ea5eeb7afcba313fae0525f070  g1c-data/gate/m1_shuffles.json
d1ed6d1ce83bd3c75f1d99a21d497fd5bf4ea8693b1868c23b98e23604b0e6eb  g1c-data/gate/m2_ablations.json
d121792525404a5df19f401bb52abfa1a6ae550cf86344dc5a66553873a841e0  g1c-data/gate/rederive.json
d49e041b708b33ebe74ea08a5e16d62aaf49a8ac77d53fc53e70e434fb7f9b6e  g1c-data/gate/tuning.json
f2ce49f7d5af8a3dbff2fe3b71014f655e8cc1438795ea48681fbe9ea1e74121  g1c-data/gate/tuning_log.jsonl
9f7f697dbc09483f489cdb4b95665e0ebd651b9ea7e715599f51ab42deb866d8  g1c-data/meshes.json
8164859db2faa00b2cec5f784d47929a69786c1bbd525b495f3cedf8a15c2b16  g1c-data/oa_fragments.json
cc777a9b62307a73e4320368270a44c327eae2832831e332d63cac7d14d10f95  g1c-data/reach.json
fae9bfac419c75543579c6a8a41d1e2d8dc32079157104b7abdce3ebbee44e87  g1c-data/share.json
```

## Published data copies (public/data/)
The data folders are republished under `public/data/` with local file paths removed. Two files
(`g1b-data/readouts.json` and `g1c-data/gate/gate_raw.json`) contained absolute paths from the analysis machine; in the
public copies those prefixes are replaced with repo-relative paths (and scratch paths with `scratch/`). No number
changed. So those two public copies have different sha256 values from the internal record listed above; every other
file is byte-identical. Check the public copies with `cd public/data && shasum -a 256 -c SHA256SUMS` (31 files).
