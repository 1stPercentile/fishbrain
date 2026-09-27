---
publication_copy: edited-for-publication copy of launches/fishbrain/evidence/engine-speed.md at commit 0d05865ac. Local paths made repo-relative; machine-owner, internal coordination, account and credential details removed. No number, hash, time, method, result or correction was changed; publication notes are marked [Publication note].
title: Engine speed, the activity-gated simulator
created: 2026-09-26
evidence_status: measured (Apple M5 Mac, one core, load average 18-29, Python 3.9.6, numpy 2.0.2)
---
# Engine speed: ActiveSimulator (fishbrain/sim_fast.py)

**Idea:** in the LIF update, a neuron whose u and h are both +0.0 stays +0.0 and can't cross threshold. So
updating only the active set, with the identical numpy operations in the identical order, gives the same
results bit for bit. Most of Fish1's 13.46M segments never receive input.

**Exactness, tested (tests/test_sim_fast.py, 8 tests pass):**
- Identical spike hashes, state digests and stats against sim.Simulator on synthetic graphs with every
  input kind (scalar Poisson, framed Poisson, currents), across several run() calls, snapshot/restore,
  manual state edits, lesions and dt 0.2.
- A planted state perturbation changes the digest (positive control).
- It is faster on a sparse 400K-neuron graph.

**On the real brain** (the loom window from fishbrain.bench, seed bench-0, 300 ms simulated, same process):

| Network | Neurons | Simulator RTF (CPU) | ActiveSimulator RTF (CPU) | Speed-up | Identical spikes + state |
|---|---|---|---|---|---|
| Full brain of record | 13,458,709 | 0.011 | **0.273** | **24.8x** | yes (spike hash 9234478e…, state 179b5fae…) |
| Gate network | 8,654 | 2.73 | 2.05 | 0.7x | yes (spike hash 3a48d7c3…, state 66759b14…) |

The active set ended at 21,026 neurons on the full brain (0.16% of segments). `make_simulator()` picks
ActiveSimulator at 200,000 neurons and above, and Simulator below that. RTF is simulated ms per CPU ms on
a shared, loaded machine, so a dedicated server core should do better. That hasn't been measured: the
cross-machine run has not been done.
