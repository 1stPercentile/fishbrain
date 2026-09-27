---
publication_copy: edited-for-publication copy of launches/fishbrain/evidence/G1b-cave-v709-tectum.md at commit c601f956d. Local paths made repo-relative; machine-owner, internal coordination, account and credential details removed. No number, hash, time, method, result or correction was changed; publication notes are marked [Publication note].
title: G1b scope check, does CAVE v709 attach tectal axons?
created: 2026-09-26
evidence_status: measured (CAVE fish1_full materialization v709, authenticated, read-only)
---
# Scope check: the newer CAVE version

The G1b decision flagged that its stop held for the public agglomeration and that CAVE's newer, partly proofread
materialization (v709, agglomeration v250915) had never been checked for tectal attachment. Checked 2026-09-26:
- 150 tectal somas were sampled (seed 20260926) from the 44,641 tectal lore ids that have CAVE roots.
- Their output synapses in `synapses_axde` (filtered by pre_pt_root_id): 343 in total.
- Outputs per tectal soma root: **median 0**, mean 2.3, max 103. **76.7% have zero outputs.**
- Share of those outputs posterior of x = 62,000 (hindbrain side): **0.0**.
- Share landing on soma-bearing roots: 0.257.

**Conclusion:** the newer version does not attach tectal axons to their somas either. The G1b stop holds for both
the public pinned agglomeration and CAVE v709. Script: `cave_tectum.py`, a scratch file that was not kept in the repo;
the check is reproducible with the pinned CAVE somas snapshot and authenticated (login-required) CAVE access.
