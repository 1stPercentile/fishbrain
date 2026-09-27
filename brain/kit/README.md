# The G1c fish, in five files

Everything `python -m fishbrain.demo` needs to rebuild the fish the G1c gate tested, without a login or the
2.5 GB wiring download. `SHA256SUMS` pins each file; the demo checks them before it copies anything into `data/`.

| file | what it is | made by |
|---|---|---|
| `cells_identified.json` | tectum cells with their retinotopic positions, the Mauthner cells, 50 named spinal projection cells, exclusions | `fishbrain.cells` (G1, [evidence/G1-cells.md](../../evidence/G1-cells.md)) |
| `readouts.json` | the deciding cells: M-system, nIII dorsal, nMLF, turning SPNs, and the lesion sets | `fishbrain.readouts` (G1b, [evidence/G1b-readouts.md](../../evidence/G1b-readouts.md)) |
| `bridge_spec.json` | the frozen bridge: every link the model adds, with its class and source. sha256 `185b3780…` is checked by `bridge.load_spec()` | G1c, [evidence/G1c-bridge-spec.md](../../evidence/G1c-bridge-spec.md) |
| `topology_intact.npz` / `.json` | the reduced network: 10,528 cells, 42,342 measured pairs plus the bridge edges, before gains | `bridge.build_topology("intact")` |

Built with gains `g_a = 1`, `g_c = 4`, the topology gives network digest
`0b6f3369f21f5c626d9ba1ba4bc02dfefd1e51b945b3355fcd6730c9aa083b4a`, the one recorded in
`evidence/data/g1c-data/gate/gate_results.json`. The demo refuses to call anything else the G1c fish.

To rebuild all of it from the source data instead: `pip install -r requirements-full.txt`, then
`python -m fishbrain.pull` (public bucket, no login, MD5-verified), `python -m fishbrain.bridge rederive` and
`python -m fishbrain.bridge topology`. `fishbrain.cells` also reads the CAVE soma table, which needs a Fish1 login.
