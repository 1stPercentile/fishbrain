---
publication_copy: edited-for-publication copy of launches/fishbrain/evidence/G1-sim.md at commit 31e9971b4. Local paths made repo-relative; machine-owner, internal coordination, account and credential details removed. No number, hash, time, method, result or correction was changed; publication notes are marked [Publication note].
title: G1 simulator, the spiking engine and its stimuli (task B2)
created: 2026-09-25
status: built and tested on synthetic graphs; not yet run on Fish1 wiring
evidence_status: measured (numbers below were measured on this Mac on 2026-09-25 unless marked otherwise)
verification_scope: "Read 2026-09-25: philshiu/Drosophila_brain_model model.py at commit 91bdd1e (2024-09-14, the code of Shiu et al. 2024); fruitflydev/flycoinrh flysim.py, LICENSE and NOTICE (HEAD); Brian2 source brian2/devices/device.py (default clock) and core_preferences.py (default float dtype); Fotowat & Engert 2023 eLife 12:e82916 full text via Europe PMC (PMC10014075); Bianco, Kampff & Engert 2011 via frontiersin.org (read through a summarising fetch, so treat its quotes as secondary); the BadenLab Zebrafish-visual-space-model README. Built and tested against synthetic graphs only. Not run: the real Fish1 graph, a second machine, the target server."
---
# G1 simulator: the spiking engine (task B2)

**TLDR.** `fishbrain/sim.py` runs Shiu et al.'s whole-brain LIF model on a sparse CSR connectome,
event-driven, deterministic from a block-hash string, with Poisson and current inputs, lesion and a
degree-preserving shuffle. `fishbrain/stimuli.py` makes the loom, receding, dimming and prey-dot
stimuli in the per-eye `(azimuth, elevation, contrast)` format. **All 30 tests in `test_sim_synthetic.py` pass (pytest exit 0).**
Speed on this Mac: a 190,000-neuron, 29,983,766-pair random graph runs at **0.29-0.36x real time
at 3 Hz mean firing and 0.68x at 1 Hz** (process CPU time, dt 0.1 ms). The neuron update alone
costs ~93 us per 0.1 ms step, so pure numpy on one core cannot reach real time for the whole
brain at Shiu's dt. It reaches real time up to ~50,000 neurons (1.35-1.6x), or at dt 0.2 ms with
little activity. Peak memory: 1.10 GB.

## 1. What exists

| File | What |
|---|---|
| `brain/fishbrain/sim.py` | `LIFParams` (every parameter, one frozen dataclass), `Network` (CSR wiring: `from_synapses`, `from_pairs`, `lesion`, `shuffled_copy`, `digest`), `Simulator` (`add_poisson`, `add_current`, `run`, `raster`, `snapshot`/`restore`, `state_digest`), `spike_hash`, `derive_seed`, `synthetic_graph`, `benchmark`, `scaling` |
| `brain/fishbrain/stimuli.py` | `looming`, `receding`, `dimming`, `prey_dot`, `blank`, each a lazy `Stimulus` sequence of frames `{"left": [(az, el, c), ...], "right": [...]}`, plus `arrays(i)`, `lattice`, `eye_masks`, `EyeModel` |
| `brain/tests/test_sim_synthetic.py` | 30 tests: dynamics against closed forms, the four named behaviours, determinism (including a subprocess and golden hashes), Poisson generators and their frame schedule, stimuli |

How the other lanes plug in:
- **Loader (B1 / pull):** `Network.from_synapses(pre_ids, post_ids, types, sign_map, counts=None, ids=None)`.
  One row per synapse (or per pair with `counts`). `sign_map` is required, e.g. `{1: +1, 2: -1}` once
  the E/I value is settled. Nothing in the code assumes which type value is excitatory; an unknown
  type value raises.
- **Cells (tectum_drive):** a `Stimulus` is a `Sequence`; `stim[i]` is the literal
  `{"left": [(az, el, contrast), ...], "right": [...]}`; `stim.dt_ms` is the frame period (default
  1 ms); `stim.arrays(i)` gives the same frame as two `(n, 3)` arrays. The resulting per-frame rates
  go in as `sim.add_poisson(idx, rates_FxK, frame_ms=stim.dt_ms)`.
- **Neuron ids:** inputs take internal indices; convert segment ids with `net.index_of(ids)`.
  `lesion()` takes segment ids. Rasters record internal indices, which never move under `lesion()`.

## 2. The model and where each number comes from

Equations and parameters are those in the code of Shiu et al. 2024 (Nature 634:210), `model.py`
at commit 91bdd1e, read 2026-09-25. flycoinrh says it follows the same paper.

| Parameter | Value | Source |
|---|---|---|
| resting potential v_0 | -52 mV | Shiu model.py (they cite Kakaria & de Bivort 2017) |
| reset potential v_rst | -52 mV | Shiu model.py |
| threshold v_th | -45 mV, strict `v > v_th` | Shiu model.py |
| membrane time constant | 20 ms | Shiu model.py |
| synaptic time constant | 5 ms | Shiu model.py (cite Jurgensen et al.) |
| refractory period | 2.2 ms = 22 steps | Shiu model.py (cite Lazar et al.) |
| transmission delay | 1.8 ms = 18 steps | Shiu model.py (cite Paul et al. 2015) |
| weight per synapse | 0.275 mV | Shiu model.py ("free parameter") |
| Poisson kick | 68.75 mV into v (f_poi 250 x w_syn) | Shiu model.py |
| dt | 0.1 ms | Brian2's default clock (`brian2/devices/device.py`: `Clock(dt=0.1 * ms)`); Shiu does not set dt |
| integration | exact solution of the linear system | Shiu uses Brian2 `method='linear'` |

Equations: `dv/dt = (v_0 - v + g)/20 ms`, `dg/dt = -g/5 ms`, both frozen while refractory; a
spike sets v = v_rst **and g = 0** (Shiu's reset); an arriving spike adds
`w = signed synapse count x 0.275 mV` to g, 1.8 ms after the presynaptic spike.

Derived constants are computed with Python's `decimal` (correctly rounded, no libm):
a = exp(-dt/20) = 0.99501246 (float32), b = exp(-dt/5) = 0.98019868,
c = 5/(5-20)(b - a) = 0.0049379352, and a current of I mV adds I(1 - a) = 0.0049875206 I per step.
`LIFParams().digest()` = `958a1404a0e6dfab7682b5f9c4419aa5d3e25963cc18f582cbd99a4d69284717`
(includes these constants' float32 bit patterns).

**Step order** (Brian2's default schedule, groups -> thresholds -> synapses -> resets):
1. integrate every non-refractory neuron one exact step (u = v - v_0 and h = c g, so it is
   u *= a; u += h; h *= b), plus any current input;
2. threshold: fired = non-refractory, non-lesioned neurons with v > v_th, ascending index;
3. synapses: deliver the spikes fired 18 steps ago (rows ascending, targets ascending, `np.add.at`),
   then Poisson kicks, source by source in the order added;
4. reset v and g of the neurons that fired; they are refractory for the next 21 steps and can fire
   again 22 steps later.

A spike at step s lands on g in step s + 18 and first moves v in step s + 19.

## 3. Where we differ, and why (for the honesty page)

From Shiu et al. 2024:
1. **Sign per synapse, not per neuron.** Shiu signs a pair by the presynaptic neuron's predicted
   transmitter. Fish1 labels each synapse with `type` 1 or 2, so a pair's weight is
   `sum over its synapses of sign_map[type] x 0.275 mV`. Mixed pairs net out, and pairs that net
   to zero are dropped.
2. **Poisson-driven neurons keep their refractory period.** Shiu sets it to 0 for neurons that
   receive Poisson input. Ours stay at 2.2 ms, because the tectal cells the eye drives are also
   network neurons, and the stimulus code should not change a cell's properties. A kick that
   arrives during refractoriness is **deferred, not lost**: it writes v, v stays frozen, and the
   cell fires on its first free step. That is what Brian2 does.
3. **Current input** (`add_current`, I in mV, `dv/dt = (v_0 - v + g + I)/tau_m`) is our addition;
   Shiu has none.
4. **Lesion removes the neuron:** every synapse in and out goes, and it can never fire. Shiu's
   `silence` zeroes only the outgoing weights. Downstream the effect is the same; ours also
   removes the neuron's own spikes from the raster.
5. **float32 state** (Brian2 defaults to float64). The closed-form PSP matches to 8.2e-6 mV (test below).

From flycoinrh (MIT, (c) 2026 fruitflydev): we reuse the idea of its `flysim.py` loop, propagating
only the outgoing rows of neurons that fired, with a ragged gather. Its model differs from Shiu's
and from ours: no synaptic time constant (a spike jumps v at once), no delay, dt 0.2 ms, Poisson
kicks set v to threshold + 1, a refractory neuron is clamped every step (kicks lost), and there are
per-cell-type trainable gains. We have none of those.

## 4. Determinism

- **Seed:** `derive_seed(s) = int(sha256(s as UTF-8, verbatim))` into `np.random.PCG64`. The string
  is used as given, so `0xABC` and `0xabc` are different seeds; pass the block hash as the chain
  prints it (lowercase 0x-hex). Each input source has its own stream,
  `sha256(seed | "poisson" | name)`; the shuffle uses `sha256(seed | "shuffle")`.
- **Only raw PCG64 output is used** (`random_raw`). numpy promises nothing about `Generator`
  methods across versions, but the PCG64 stream is fixed by the algorithm. Bernoulli draw =
  `raw < floor(p 2^64)`. Constant-rate groups use geometric gaps on the (step, neuron) lattice,
  computed by repeated squaring and float64 multiplication only, so cost scales with events,
  not group size.
- **Arithmetic:** elementwise IEEE float32 add/multiply, in-order `np.add.at`, no BLAS, no
  threads, no fused kernels.
- **Hashes:** `spike_hash(raster)` = sha256 of `b"fishbrain-raster-v1"`, then u64 n_neurons,
  start_step, n_steps and n_spikes, then int64 steps and int64 neuron indices sorted by (step, index).
  Trailing silent steps change the hash. `Network.digest()`, `LIFParams.digest()` and
  `Simulator.state_digest()` (voltages, conductances, spike times, in-flight spikes, source RNG
  states) complete a replay receipt.
- **Tested:** same seed gives the same hash twice in-process and in a fresh subprocess with a
  different `PYTHONHASHSEED`. A one-character seed change gives a different hash. `run(37)+run(63)`
  equals `run(100)`. `snapshot()` then `restore()` replays the same window and the same end state.
- **Golden hashes recorded here** (tests pin them): the scenario in the test file with seed
  `0x9f3c...3a1c` gives spike hash `459892940b54b9f2e103a9dbba5a7a5b5eb1e8011d9b663a5e5667867e610f20`,
  state digest `c918d0e111599e5efe20e8640c8f3a6093dd30d0f9a38d6e76b6514c7601928f`, and 23,141 spikes.
- **Not tested: any other machine.** The design aims at bit-identical results across machines and
  numpy versions, but no second machine has run it. The golden-hash test is the check to run on
  the server. If it fails there while the other determinism tests pass, cross-machine replay is
  broken; that is a finding to report, not a hash to re-record. The stimuli use numpy trig (libm),
  rounded to 1e-6. That makes a cross-platform ulp difference matter only when it straddles a
  rounding boundary; it does not eliminate it. The replay proof should commit the rendered stimulus
  (or the rates) it fed the brain, not just its parameters.

## 5. Controls

- `net.lesion(segment_ids)`: new Network, same indices, `dead` mask set, in and out pairs dropped.
  On the 190k/30M synthetic graph, lesioning 190 neurons took 0.66 s and removed 59,953 pairs.
- `net.shuffled_copy(seed)`: each pair keeps its presynaptic neuron and signed count; postsynaptic
  ends are permuted uniformly, then duplicate pairs and self-loops are repaired by swapping with
  random other pairs until none are left. **Preserved exactly:** every neuron's out-degree,
  in-degree, out-strength and outgoing sign mix. **Not preserved:** in-strength. On the 190k/30M
  synthetic graph it took 50.6 s and 4 rounds, with 0 unrepaired pairs, and both degree arrays
  were checked equal.

## 6. Stimuli (`stimuli.py`)

Conventions: fish-centred, head-fixed directions. Azimuth 0 = ahead, + = the fish's right.
Elevation 0 = horizon, + = up. Contrast is signed: -1 black, +1 brightest, 0 background. A point
is listed under every eye that sees it (the frontal binocular zone appears in both).

| Item | Value | Source or status |
|---|---|---|
| eye field | 163 deg cone per eye | Easter & Nicola 1996, via the BadenLab visual-space model README |
| eye rest angle | 18.5 deg to the body axis | BadenLab model parameter (Bianco 2011: vergence 36.0 deg at rest, 66.9-76.4 hunting) |
| optical axis | azimuth +/-(90 - 18.5) = +/-71.5 deg, elevation 0 | **our derivation**; gives a frontal binocular zone about 20 deg wide at the horizon (412 lattice points) and a rear blind zone |
| sampling | near-equal-area lattice, 2 deg, 10,312 points; 4,396 per eye | **our choice** |
| edge antialiasing | contrast x clip((r + res/2 - d)/res, 0, 1) | **our choice**; a 1 deg dot still gives 4 dim points |
| loom | theta = 2 atan(l/v / (t_c - t)), l/v = 240 ms, 4 -> 140 deg, dark (-1), default azimuth +90 | Fotowat & Engert 2023 methods (read in full text); formula = the classic l/v course |
| loom timing | t_c = 6,872.7 ms after onset; 140 deg at 6,785.3 ms | computed |
| receding | the loom's time course reversed | **our choice** of the standard control |
| dimming | contrast(t) = -Omega_disc(theta(t)) / Omega_eye (5.354 sr), reaching -0.771; both eyes by default | matching rule is **our choice**; Fotowat & Engert 2023 separate a loom into expansion and dimming |
| prey dot | 3 deg bright spot (+1), 30 deg/s, 20 deg back-and-forth sweep around the given azimuth, 3 s | Bianco et al. 2011 (bright spots at max contrast; 1-5 deg prey-like, 10 deg aversive; ~30 deg/s restrained; 3 s trials) read through a summarising fetch; **sweep is our choice** |

Frames are rendered on demand. A late loom frame lists 3,324 right-eye points; a dimming frame
lists 4,396 per eye. Rendering took about 1.5 ms per frame under load.

## 7. Tests

`cd launches/fishbrain/brain && .venv/bin/python -m pytest tests/test_sim_synthetic.py -q`
gives **30 passed, exit code 0** (final run, 2026-09-25). The whole `tests/` folder, which now also holds other lanes' tests (decoder, cells), gave 41 passed, 1 skipped, exit 0 on the same final run. Measured values behind the assertions:

| Test | Measured |
|---|---|
| PSP against the closed form (100 synapses) | max error 8.2e-6 mV over 58 ms; peak 9.2 ms after arrival (analytic 9.242); peak/w 0.157489 (analytic 0.15749) |
| refractory, dt 0.1 and 0.2 | ISI exactly 22 and 11 steps |
| refractory freeze | v held at reset for steps 1-22; g takes the arriving input and holds without decay; step 23 integrates |
| kick during refractory | spike at step 1, deferred kick fires at step 23 |
| delay | the post-synaptic spike comes at least 19 steps after the pre spike |
| chain 0->1->2->3->4 (1000 synapses per link), 40 Hz into 0 | counts [17, 17, 17, 17, 17], unconnected neuron 0; first spikes at steps 505, 529, 553, 577, 601 (24 steps per hop) |
| inhibition (A -400-> T, I -1500-> T, type 2 = -1) | T fires 33 with A alone; **0** with I at 200 Hz |
| sign from the map (flipped to {1: -1, 2: +1}) | A alone: T 0; I alone: T 178 |
| lesion (segment-id-like ids) | intact chain [30, 30, 30, 30, 30]; with cell 2 lesioned [30, 30, 0, 0, 0] even though cell 2 got 66 direct kicks; upstream spike times unchanged |
| shuffle (300 neurons, 2,353 pairs) | 3 rounds, 0 unrepaired; degrees equal; spikes 355 -> 5,110; active undriven cells 83 -> 255 |
| Poisson scalar, 2,000 cells x 20 Hz x 1 s | 39,884 events (expected 40,000); per-cell variance 20.5 (Poisson: 20) |
| Poisson per-neuron rates | 40,072 events (expected 40,000) |
| Poisson (F, K) frames, frame 2 of 5 at p >= 1, start 3 ms, 1 ms frames | exactly 10 kicks, on steps 50-59; spikes at 51 and 73 (deferred); no RNG draws after the last frame |
| current (F, K) frames, frames 1 and 7 saturating, start 2 ms, 0.5 ms frames | v exactly at rest before step 25; spikes at 25 and 55 only |

**The tests fail when they should.** Each mutation broke at least one test:
integrating with the decayed g; delivering one step early; refractory one step long; ignoring
the sign map; removing the refractory freeze; not resetting g on a spike; discarding kicks that
arrive during refractoriness; reading the frame index one step early (Poisson and current separately); drawing after the last frame.

## 8. Benchmark (this Mac, 2026-09-25)

Raw per-trial results (git-ignored): `brain/data/bench/sim_dt0.1.json`, `sim_dt0.2.json`,
`scaling_dt0.1.json`, `controls_190k.json`, written by `python -m fishbrain.sim` (see `benchmark()`
and `scaling()`).

Setup: Apple M5, 10 cores, 16 GB, macOS 26.6.2, Python 3.9.6, numpy 2.0.2, scipy 1.13.1.
`synthetic_graph(190000, 30000000)`: 30,000,000 uniform random synapses, 80/20 type 1/2, summed
into **29,983,766 pairs** (all count +/-1). That is the worst case: the real graph's multi-synapse
pairs mean fewer pairs. Every neuron gets Poisson drive at the rate shown; the edgeless graph gets
the same drive with no synapses. Timing covers 1 s simulated after a 50 ms warm-up. RTF means
simulated time over time spent; above 1 is faster than real time.

**The machine was heavily loaded by other workloads:** load average 23-41 on 10 cores during the runs.
In the 19k-neuron trial runs before these, wall time reached 20x the CPU time. In the runs
tabulated here, wall time stayed within 1.7x of CPU time (worst: edgeless, 1 Hz, 3.02 s vs 1.76 s). The table gives **process CPU time** (numpy here is single-threaded,
so this is what one uncontended core would take) and wall time for reference. Contention still
inflates CPU time through shared caches, so these numbers are conservative.

dt = 0.1 ms (Shiu / Brian2), 3 trials, median (best):

| Graph | Drive | Mean rate | Synaptic events/s | CPU us/step | RTF (CPU) | RTF (wall) |
|---|---|---|---|---|---|---|
| full | 0 | 0 Hz | 0 | 92.9 | 1.08 (1.14) | 1.04 |
| edgeless | 0 | 0 Hz | 0 | 98.0 | 1.02 (1.09) | 0.99 |
| full | 1 Hz | 1.001 Hz | 3.0e7 | 147.0 | **0.68** (0.70) | 0.66 |
| edgeless | 1 Hz | 1.001 Hz | 0 | 174.3 | 0.57 (0.75) | 0.44 |
| full | 3 Hz | 3.000 Hz | 9.0e7 | 299.1 | **0.33** (0.36) | 0.25 |
| edgeless | 3 Hz | 3.000 Hz | 0 | 199.3 | 0.50 (0.52) | 0.39 |
| full | 10 Hz | 10.007 Hz | 3.0e8 | 429.3 | **0.23** (0.27) | 0.18 |
| edgeless | 10 Hz | 10.007 Hz | 0 | 191.5 | 0.52 (0.55) | 0.43 |

dt = 0.2 ms (flycoinrh's step, **not** Shiu's), 2 trials: full, no drive: 1.53 (1.63) RTF;
**full, 3 Hz: 0.73 (0.80)**; edgeless, 3 Hz: 1.26.

Scaling at 3 Hz, 158 synapses per neuron, dt 0.1 ms, 2 trials, CPU RTF median (best):
20k neurons **3.07** (3.15); 50k **1.35** (1.63); 100k **0.67** (0.82); 190k **0.29** (0.33).

What the numbers say:
- **Fixed cost:** about 93-98 us per step to update 190k neurons, roughly 0.5 ns per neuron. That
  is five full-array numpy passes and is near what one core does. So 190k neurons at dt 0.1 ms top
  out around 1x real time with nothing firing.
- **Spikes:** propagation costs about 8-11 ns per synaptic event (full minus edgeless: 100 us at
  9,000 events per step, 238 us at 30,000). Handling spikes and drive adds 75-100 us per step at
  19-190 spikes per step. These overhead numbers are noisy under this load; the edgeless 1 Hz row
  costing more than the full one is noise.
- The synthetic net is input-driven: mean rate equals drive rate. With 0.275 mV per synapse and
  about 158 random inputs, recurrent input stays far below threshold. The real graph's
  multi-synapse pairs and hubs will behave differently. Its RTF follows from the per-step
  costs above plus its measured events per step.
- **Memory:** network arrays 243 MB (29.98M pairs x int32 index + int32 count); the simulator adds
  a float32 weight per pair plus state, 123 MB. **Peak RSS 1.10 GB** (dt 0.1 run, graph build
  included). Shuffling the 30M graph peaked at 1.63 GB.

For G1: to watch the whole 190k brain live at dt 0.1 ms we need either (a) the vision-to-motor
section (at or below ~50k neurons is real time on one core here), (b) dt 0.2 ms, which is not
Shiu's step, so it goes on the honesty page and needs its own validation, or (c) a compiled or
GPU backend, which must reproduce these exact bits (the golden-hash test) before the proof can
use it. None of these was chosen here; the choice was left to the G1 gate.

## 9. Choices we invented (all listed for the honesty page)

1. Per-synapse sign summed per pair; zero-net pairs dropped.
2. Refractory kept on Poisson-driven cells (Shiu: 0); deferred kicks.
3. Current input term.
4. Lesion = remove in and out synapses and block firing.
5. float32 state; closed-form update in the (u, h = c g) variables.
6. Seed string used verbatim; per-source streams keyed by source name; geometric-gap Poisson on
   the step-by-neuron lattice; Bernoulli by integer threshold; p below 2^-40 per step treated as 0.
7. Shuffle: keep the pre side and count, permute the post side, repair duplicates and self-loops.
8. Stimulus geometry: optical axis at +/-71.5 deg, cone field, 2 deg lattice, antialiased edges,
   1e-6 rounding, fish-centred coordinates.
9. Receding = reversed loom; dimming matched to the loom's solid angle over one eye's field,
   shown to both eyes by default.
10. Prey sweep of 20 deg around the given azimuth; defaults of 3 deg and 30 deg/s.
11. Default frame period 1 ms; each frame is sampled at its start time.

## 10. Seams with other lanes (read from `cells.py` on 2026-09-25, not changed)

- `cells.tectum_drive(stimulus)` takes one frame dict, i.e. `stim[i]`, and uses the same azimuth
  convention (0 ahead, + right). Checked.
- **Two eye-field models exist.** stimuli.py lists a point under an eye if it falls in a 163 deg cone
  around azimuth +/-71.5 deg: eye-lateral azimuth -10 to +153 deg at the horizon. cells.py then keeps
  eye-lateral azimuth -20 to +160 deg. The field that takes effect is the overlap, -10 to +153.
  One of the two should own the eye field.
- cells.py drives with |contrast|, so ON and OFF look the same to the tectum. A whole-field
  dimming frame (4,396 points per eye at up to 0.77) will drive every tectal cell. That matters for
  the dimming control.
- cells.py computes the drive with `@` (BLAS). BLAS is the one place where summation order and FMA
  can differ between machines, so the drive values may differ in their last bits. The golden-hash
  check should cover the drive as well as the simulator.

## 11. Open

- Run the golden-hash test on the target server (Linux x86 expected) before the replay proof relies
  on cross-machine equality.
- The E/I `type` value mapping is still open (task brief). This code takes it as input.
- The real graph: nothing here has touched Fish1 wiring yet.
