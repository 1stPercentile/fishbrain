---
publication_copy: edited-for-publication copy of launches/fishbrain/evidence/G1b-readouts.md at commit c7c91e826. Local paths made repo-relative; machine-owner, internal coordination, account and credential details removed. No number, hash, time, method, result or correction was changed; publication notes are marked [Publication note].
title: G1b readouts, bilateral populations, the M-system and the strike rule (task R)
created: 2026-09-26
status: done (task R). Readouts built and tested; nothing was simulated with them.
evidence_status: measured unless marked "expected", "choice" or "post hoc"
verification_scope: >
  Measured 2026-09-26 01:25-02:15 EDT on the analysis machine from the pinned brain of record
  (brain/data/synapses.parquet, wiring content hash 09a71038...25af2, via the cached pair arrays
  data/net/pairs_*.npy), brain/data/cells_identified.json (B1, built 2026-09-25), CAVE somas v709
  (cached parquet, no CAVE call made), the lore-id -> segment csv, and public Fish1 layers read
  without login: mece1_231218 and mece2_231218 at 1024 nm and 512 x 512 x 60 nm, lores_cbs_231218
  at 512 x 512 x 60 nm, and seg_241003_agg241003 meshes (lod 3) for 24 segments. Literature read
  2026-09-26: abstracts of Thiele 2014, Gahtan 2005, Liu & Fetcho 1999, Greaney 2017 and Fajardo
  2013 (Europe PMC REST, read directly); Lau, Fitzgerald & Bianco 2025 (full text, Europe PMC XML,
  read directly); Greaney 2017, Fajardo 2013 and Kohashi & Oda 2008 full texts through a summarising
  fetch (secondary, marked where used).
outputs: brain/fishbrain/readouts.py, brain/data/readouts.json (git-ignored, rebuild with
  `python -m fishbrain.readouts`), brain/tests/test_readouts.py, this note
---
# G1b readouts: who fires means what

**TLDR.**
- **Bilateral populations now come from the Fish1 atlas, not from mirror guesses.** The atlas labels
  every reticulospinal group the task named, on both sides. RoV3, MiV1 and MiV2 give **32 left and
  23 right** turning cells, and no region is significantly lopsided except nIII (see below).
- **The positive control reproduces B1 exactly:** 12 of the 50 named SPNs sit inside an atlas
  reticulospinal region, and 11 of those 12 are in their own-named region (name-permutation null
  mean 1.8, p < 1e-4).
- **But the strict atlas regions are small.** Only 6 of the 30 named right turning SPNs are in the
  23-cell right turning population. A 5 um dilation, chosen after seeing the data, holds 16 of 30.
- **M-system found:** MiD2cm = 17405827637 (left) / 16039136570 (right); MiD3cm = 17365353494 (left) /
  17610066079 (right). Each is the largest soma with the most inputs in its atlas region, the
  left/right picks mirror each other within 6 um, and each shares VIIIth-nerve-zone afferents with
  its own side's Mauthner cell. The same code finds both Mauthner cells in r4 (positive control).
  **My own pre-registered rule (which also required a midline-crossing axon) disagreed on 3 of 4
  sides, and I set it aside after seeing the data.** That deviation is disclosed in section 4.
  MiD3cm-left is contested.
- **Strike rule, fixed before any simulation:**
  - both dorsal-nIII pools (the medial rectus proxy) active = the eyes converge;
  - one nMLF active = drive;
  - side = the more active nMLF (Thiele 2014: nMLF bends the tail to its own side).
- **Two findings cut against the brief.**
  1. The premise that hunting J-turn neurons cluster in RoV3 and MiV2 is contradicted by the
     newest imaging (Lau et al. 2025). Those cells steer other turns and are barely recruited in
     hunting J-turns.
  2. The same paper finds the nMLF mostly encodes speed. The strike's side term is therefore its
     weakest part.
- **Wiring only:** the tectum's direct contacts onto nIII, nMLF and the forward cells are
  ipsilateral. Nothing in the reachable network tells the two sides apart (fireable-input index
  within the permutation null for every readout).

`pytest -q tests/test_readouts.py`: **exit 0, 14 passed** (0.58 s).

## 1. What exists

| File | What |
|---|---|
| `brain/fishbrain/readouts.py` | atlas box and populations (`build_populations`), SPN validation (`validate_spn`, `compare_mirror_picks`), M-system (`identify_largest`, `choose_m_homologs`, `shared_afferent_evidence`), decision rules (`strike_decision`, `escape_decision`, `turn_index`), `readout_sets`, `wiring_checks`, `build` |
| `brain/data/readouts.json` | 871,128 bytes (see section 8), built 2026-09-26 02:07:06 EDT in 371 s |
| `brain/tests/test_readouts.py` | 14 offline tests (section 9) |

Rebuild: `cd launches/fishbrain/brain && FISHBRAIN_READOUTS_CACHE=<dir> .venv/bin/python -m fishbrain.readouts`.
The build reads public layers through cloud-volume and caches them in `$FISHBRAIN_READOUTS_CACHE`
(default: a folder in the system temp dir; this run used a scratch folder that was not kept). **No file owned by
another lane was written.** Read-only uses of other lanes' code: `cells._label_votes`, `cells._somas`,
`cells._agg_map`, `cells._atlas`, `cells.side_of`, `cells.midline_y`; `brain.load_pairs`,
`brain.identified`, `brain.signed_counts`, `brain.stimulated_set`, `brain.bfs_levels`, `brain._csr`,
`brain.index_of`; B1's cached `data/ref/mece2_1024_hindbrain.npz` and `data/ref/r4_box_512x512x60.npz`.

## 2. Atlas populations (task item 1)

**Region names in the atlas (mece2_231218 segment properties).**
- Reticulospinal and oculomotor labels present: NucMLF (16), Oculomotor Nucleus nIII (15), nIV (43);
  RoL-R1 (44), RoL2 (21), RoM1 (24), RoM2 (22), RoM3 (27), RoV3 (28), RoL3 (31); Spiral Fiber Neuron
  Posterior (29) and Anterior (32) clusters; MiV1 (47), Mauthner (48), MiR1 (49), Mauthner Cell Axon
  Cap (52), MiM1 (53), MiR2 (54), MiV2 (34), MiD2 (38), MiD3 (55), MiT (60), CaD (61) and CaV (62).
- **Absent:** the abducens nucleus (nVI), any split of the nMLF into MeLr / MeLc / MeLm, and the
  hunting populations Lau 2025 names (RoM2r, "Mi3 lateral", "Ca1 lateral").
- mece3 holds no hindbrain reticulospinal region. mece0 / mece1 give divisions and rhombomeres (r5
  = 26, r6 = 29).
- Used: turning {RoV3, MiV1, MiV2}; forward {RoM1, MiM1, MiR1, MiR2, RoL-R1}; M-system regions
  {Mauthner, MiD2, MiD3}; strike {NucMLF, nIII}. Present but unused: RoL2, RoM2, RoM3, RoL3, both
  spiral-fiber clusters, the axon cap, MiT, CaD, CaV and nIV.

**Box.** mece2 and mece1 at 1024 x 1024 x 960 nm over every label above plus B1's margin: b0 = (388,
152, 114), shape 256 x 212 x 128, mece2 sha256 `b2d85eea...a6ba0a`, mece1 sha256 `3f4da82b...92bb14`.
It is **identical to B1's cached box on all 5,210,112 overlapping voxels.** B1's box started at z
voxel 146 and cut off the dorsal nIII: 74,677 of the region's 86,090 voxels were inside it. The new
box holds all of nIII.

**Rule (B1's, reused so the check is comparable).**
- A CAVE soma belongs to a region when the plurality mece2 label over its 3 x 3 x 3 neighbourhood
  (about +-1 um) is that region.
- Its side comes from the fitted midline. Somas within 3 um of the midline are dropped: NucMLF 1,
  nIII 27, none elsewhere.
- Each soma maps to its agg241003 segment through the lore-id csv. None mapped to 0.
- A segment that also holds a soma outside the region or side is dropped: 0 strict, 2 dilated (MiD3).
- 32,462 CAVE somas lie in the box. Every strict member is soma-bearing. Every member of the
  reticulospinal groups (hindbrain and NucMLF) has input synapses; in nIII, 237 / 260 left and
  198 / 212 right do.

**Per-side counts, strict (primary) and 5 um dilation (post hoc, see section 3):**

| region | strict L | strict R | binomial p | atlas voxels L / R | dilated L | dilated R | p |
|---|---|---|---|---|---|---|---|
| RoV3 | 13 | 8 | 0.38 | 4,742 / 5,005 | 31 | 33 | 0.90 |
| MiV1 | 10 | 8 | 0.81 | 5,125 / 5,122 | 33 | 29 | 0.70 |
| MiV2 | 9 | 7 | 0.80 | 4,021 / 3,818 | 50 | 47 | 0.84 |
| **turning total** | **32** | **23** | | | 114 | 109 | |
| RoM1 | 11 | 12 | 1.00 | 3,838 / 4,384 | 36 | 31 | 0.63 |
| MiM1 | 1 | 2 | 1.00 | 936 / 1,262 | 16 | 14 | 0.86 |
| MiR1 | 2 | 3 | 1.00 | 527 / 967 | 17 | 18 | 1.00 |
| MiR2 | 2 | 2 | 1.00 | 776 / 545 | 17 | 14 | 0.72 |
| RoL-R1 | 11 | 5 | 0.21 | 5,423 / 4,892 | 42 | 31 | 0.24 |
| Mauthner region | 1 | 3 | 0.63 | 5,470 / 4,657 | 13 | 18 | 0.47 |
| MiD2 | 10 | 8 | 0.81 | 6,365 / 4,311 | 38 | 38 | 1.00 |
| MiD3 | 13 | 14 | 1.00 | 4,647 / 5,912 | 43 | 59 | 0.14 |
| NucMLF | 15 | 18 | 0.73 | 14,857 / 17,255 | 26 | 25 | 1.00 |
| **nIII** | **260** | **212** | **0.030** | 48,462 / 37,628 | 473 | 359 | 0.0001 |
| nIII dorsal half (MR/IR proxy) | 130 | 106 | | | 237 | 180 | |

- Binomial p is two-sided against 50:50. Voxels are counted at 1024 nm per side of the midline.
- **Symmetry.** No reticulospinal region is significantly lopsided. nIII is: 10% more somas on the
  left (p = 0.03). The atlas's own nIII region has 29% more voxels on the left, so the soma asymmetry
  tracks how the region was registered, not the fish. A readout that averages per cell (section 5)
  is not biased by it.
- nIII holds about 236 somas per side, against Greaney 2017's 104 +- 20 nIII motoneurons per side and
  about 55 IR/MR motoneurons. The region contains non-motoneurons, so "dorsal half" is a proxy
  (section 5).

**Positive-control failure of the strict rule, reported:** the right Mauthner soma (lore 187052)
gets plurality label 0 at its CAVE point, although 70% of its soma voxels lie in the atlas Mauthner
region (B1). The strict Mauthner-region population (1 L / 3 R) therefore misses a known cell and
holds 3 other right-side somas. **The M-system readout uses the identified cells (section 4), never
the atlas Mauthner population.**

## 3. Validation against the named SPNs (the positive control) and against B1's mirror picks

Measured on the 50 named HMI SPNs (49 right, 1 left):

1. **Strict (B1's rule): 12 in any reticulospinal region, 11 in their own-named region.** The one miss
   is RoV3 186443, which lands in RoM3. All 50 labels equal B1's `atlas_region` field. That
   reproduces B1's 11 of 12 exactly.
2. **Name-permutation null** (10,000 permutations of the 50 names, seed `G1b-spn-name-permutation`):
   - strict own-name matches: mean 1.80, max 6, observed 11 (p < 1e-4);
   - "nearest region at any distance" own-name matches: observed 45 of 50, null mean 9.76, max 21
     (p < 1e-4).
3. **Overlap, stated bluntly: the strict populations are mostly NOT the named cells.**
   - 11 of 50 named SPNs are members of their own population.
   - Of the 30 named right turning SPNs, **6** are in the 23-member strict right turning
     population. The other 17 members are unnamed somas inside the regions: either reticulospinal
     cells the HMI release did not trace, or neighbours.
   - The atlas regions are drawn tighter than where Fish1's cells sit.
4. **Nearest-region curve** (distance from the soma point to the nearest region voxel centre; sensitivity only):

   | r (um) | 1 | 2 | 3 | 5 | 8 | 10 | 15 | 20 |
   |---|---|---|---|---|---|---|---|---|
   | named SPNs within r of a region | 14 | 25 | 30 | 37 | 45 | 49 | 50 | 50 |
   | of those, own-named region | 13 | 24 | 29 | 34 | 40 | 44 | 45 | 45 |

5. **Mirror test** (each named SPN reflected across the fitted midline, labelled again): all 50 land
   on the other side. 10 are strictly inside a region, and all 10 are in the same-named region (vs
   11 of 12 unreflected). The nearest region has the same name for 44 of 50 (vs 45). Mirror curve,
   own-name within r = 1 / 2 / 3 / 5 / 8 / 10 um: 13 / 20 / 25 / 33 / 40 / 41. **The atlas's left
   regions sit where the right cells' mirror images fall.** That is the direct evidence that the left
   atlas populations are valid counterparts of the named right cells.
6. **B1's unconfirmed mirror picks** (kept for comparison only):
   - strict: 13 of the 49 picks are in any atlas population and 12 in their own-named population on
     the same side; 8 of the 30 left turning picks are in their own left population;
   - dilated: 35 and 31 of the 49, and 19 of the 30 left turning picks.
   - The picks mostly sit outside the regions, like the named cells themselves, so this neither
     confirms nor rejects them.
7. **5 um dilated populations, post hoc.** r = 5 um was chosen after seeing the curve above, so its
   34 of 37 own-name matches are not independent validation.
   - Rule: a soma in no atlas region at all joins the nearest used region whose voxel centre is
     within 5 um. A soma inside any other atlas region is never moved.
   - It holds 34 of 50 named SPNs in their own population, 16 of 30 named right turning SPNs, and
     114 L / 109 R turning cells.
   - It is in `readouts.json` as `dilated_5um` as an alternative. **Strict is the
     default.**

## 4. The M-system: MiD2cm and MiD3cm (task item 2)

**Literature.**
- Liu & Fetcho 1999 (Neuron 23:325, abstract read directly): "Killing all three cells [Mauthner,
  MiD2cm, MiD3cm] eliminated short-latency, high-performance escape responses to both head- and
  tail-directed stimuli"; killing only the Mauthner cell affected tail-directed escapes only.
- Kohashi & Oda 2008 (J Neurosci 28:10641, summarising fetch, secondary): the M-series are
  "repeated in the middorsal region" of r4-r6, are smaller than the M-cell, have lateral dendrites,
  and have axons that cross the midline.
- Nakayama & Oda 2004 (J Neurosci 24:3199, title and search summary only): the homologs share
  sensory input with the M-cell.

**Method (as B1 did for Mauthner).**
- `lores_cbs_231218` somas plus mece2 and mece1, all at 512 x 512 x 60 nm, over the atlas region's
  bbox plus 20 um.
  - MiD2 box: b0 (1088, 416, 2162), 111 x 207 x 1371, 1,581 unclipped somas, median 41.8 um^3, 99th
    percentile 85.2.
  - MiD3 box: b0 (1152, 384, 2098), 111 x 271 x 1307, 2,107 unclipped somas, median 41.7, 99th
    percentile 78.8.
- Candidates are unclipped somas that overlap the region; the top 5 per side by volume are kept.
- Evidence recorded per candidate: input and output synapses (brain of record), rhombomere
  (mece1 plurality over the soma's voxels), and the agglomerated mesh (lateral reach, midline
  crossing, reach in each direction).
- **Positive control, same code on B1's r4 box: it returns 15773771512 (left, 271.1 um^3) and
  16304202596 (right, 247.6 um^3), both crossing the midline (4.8% / 4.7% of vertices, up to 2.1 /
  6.1 um).** That is B1's exact result.

**Pre-registration, and my deviation from it (for the honesty page).**
- Before any r5/r6 box was read I wrote down, at **2026-09-26 01:39:52 EDT** (the r5 and r6 boxes
  were written to disk at 01:48:30 and 01:50:58):
  > pick = the largest [overlapping soma] whose agglomerated segment mesh (lod 3) has vertices past
  > the fitted midline (contralateral axon: "cm"; Kohashi & Oda 2008). If none of the top 5 crosses,
  > pick the largest and flag "crossing not seen".
- The candidate table came back at 01:52; the shared-afferent table below was computed at 01:54.
- **At about 01:57 EDT I set that rule aside and used the task's own criteria instead** (atlas
  region, soma size and input count, as for Mauthner). Reasons:
  1. The crossing test has a known false-negative mode. The agglomeration cuts axons: even
     Mauthner's crossing survives only 2-6 um (B1).
  2. The crossing rule changed 3 of 4 picks and broke the bilateral pairing (MiD3 picks 20.5 um
     from mirror images).
  3. The size picks are also the input-count leaders on all four sides and pair within 6 um.
- Both pick sets are in `readouts.json`.

**Candidates (top by volume per side; "shared" = synapses from presynaptic segments that also
contact the same-side Mauthner more than 50 um laterally, i.e. its VIIIth-nerve lateral dendrite zone;
734 / 682 such segments for Mauthner left / right, which equals B1's count). The shared check was written after
the candidate table was seen, so it is post hoc.**

| homolog, side | lore | segment | soma um^3 | inputs | inputs > 50 um lateral | shared (fraction) | mesh crosses midline | lateral reach um |
|---|---|---|---|---|---|---|---|---|
| **MiD2cm L (task pick)** | 187031 | **17405827637** | 133.0 | 2,603 | 391 | **70** (0.18) | no | 68.9 |
| MiD2 L (crossing pick) | 186895 | 16039139313 | 93.1 | 1,484 | 95 | 21 (0.22) | 6.3%, to 11.9 um | 36.6 |
| **MiD2cm R (both rules)** | 187043 | **16039136570** | 149.8 | 2,734 | 548 | **58** (0.11) | 2.8%, to 2.0 um | 61.1 |
| MiD2 R, next | 186962 | 17609826269 | 110.4 | 1,555 | 350 | 13 (0.04) | no | 51.3 |
| **MiD3cm L (task pick)** | 187003 | **17365353494** | 118.2 | 1,780 | 219 | **35** (0.16) | no | 33.3 |
| MiD3 L (crossing pick) | 186535 | 17365352671 | 78.8 | 1,102 | 326 | 15 (0.05) | **30.0%, to 7.4 um; 372 um posterior reach** | 35.6 |
| **MiD3cm R (task pick)** | 186938 | **17610066079** | 105.3 | 1,743 | 265 | **23** (0.09) | no | 34.2 |
| MiD3 R, next | 186845 | 16202533183 | 98.4 | 1,141 | 334 | 3 (0.01) | no | 45.4 |
| MiD3 R (crossing pick) | 185739 | 20690356355 | 63.4 | 451 | 163 | 0 (0.00) | 6.1%, to 15.5 um | 41.7 |

Full top-5 tables, with position, frac-in-region, rhombomere and every mesh measure:
`readouts.json` -> `m_system.identification`.

**Evidence for the task picks.**
- Size: every pick is 1.34-1.76x its box's 99th percentile, and 1.43x / 1.36x / 1.50x / 1.07x the
  next candidate (MiD2 L / R, MiD3 L / R).
- Inputs: every pick ranks first by input count in its candidate set, 1.53-1.75x the next.
- Region and rhombomere: every pick is 91-100% inside its region. mece1 gives r5 for both MiD2
  picks; the MiD3 picks fall in a mece1 gap ("unlabelled"), while their neighbours are r6.
- Pairing: left and mirrored-right picks lie **5.8 um** apart (MiD2) and **6.1 um** apart (MiD3).
  The crossing picks lie 7.8 and 20.5 um apart.
- Shared afferents: the raw count favours the task picks on 4 of 4 sides. Normalised to lateral
  inputs, it favours them on 3 of 4; on MiD2-left, 187031 (0.18) and 186895 (0.22) are not
  separated. Bigger cells with longer lateral dendrites share more afferents simply by having more
  lateral input, so the raw count is not independent of size.
- Other-side Mauthner afferents onto the picks: 5, 23, 3 and 0 synapses (low, as expected).

**Confidence, per side (set by reading the table, not computed):**
- MiD2cm right: **high** (both rules agree; crosses; largest; most inputs; most shared).
- MiD2cm left: **moderate** (no crossing seen).
- MiD3cm right: **moderate** (no crossing seen; the crossing alternative is below the 99th
  percentile in size and has 0 shared afferents).
- MiD3cm left: **low, contested.** 186535 has the strongest "cm" morphology of any cell examined:
  30% of its mesh past the midline and a 372 um posterior reach. That looks like an axon crossing and
  descending in the contralateral MLF. The size pick shows no crossing at all. This data cannot
  separate MiD3cm from an ipsilateral MiD3 cell on the left.

**Lesion sets (Liu & Fetcho 1999: M-cell + MiD2cm + MiD3cm, both sides):**
- `task_criteria` (the default): 15773771512, 16304202596, 17405827637, 16039136570, 17365353494,
  17610066079.
- `preregistered_crossing` and `union` (9 segments) are sensitivity rows. If escapes survive the
  default lesion but not the union, the MiD3-left call matters.

**Side finding, correcting B1:** lore ids are assigned roughly in order of soma volume (Spearman 0.929
over the 1,581 unclipped somas of the MiD2 box, 0.913 over the 2,186 of B1's r4 box). The two
Mauthner cells hold the last two ids because they are the two largest somas, not, as B1 suggested,
because they were "added by hand". The M-series picks (187031, 187043, 187003, 186938) have high ids because
they are among the largest somas.

## 5. The strike readout (task item 3)

**Literature (read 2026-09-26).**
- Bianco, Kampff & Engert 2011 (search summary, secondary): prey capture starts with eye convergence.
  Small moving spots evoke convergent eye movements and J-turns.
- Gahtan, Tanger & Baier 2005 (J Neurosci 25:9294, abstract read directly): "MeLc and MeLr ... extend
  dendrites into the ipsilateral tectum and project axons into the spinal cord." Bilateral ablation
  impaired prey capture. Unilateral nMLF ablation plus the contralateral tectum "mostly abolished"
  it; with the ipsilateral tectum the effect was much smaller.
- Thiele, Donovan & Baier 2014 (Neuron 83:679, abstract read directly): "Optogenetic stimulation of
  neurons in the left or right nMLF ... produces a graded ipsilateral tail deflection." Unilateral
  ablation "biases the tail position to the intact side", so the nMLF "steers the direction of
  swimming".
- Greaney et al. 2017 (J Comp Neurol 525:65, abstract read directly): "inferior and medial rectus
  motoneurons occupy dorsal nIII". Full text (summarising fetch, secondary): MR, IR and IO innervate
  the ipsilateral eye and SR the contralateral one. 104 +- 20 nIII motoneurons per side, about 55 +- 14
  IR/MR per side, and IR cannot be told from MR in their fills.
- Fajardo, Zhu & Friedrich 2013 (Front Neural Circuits 7:67):
  - abstract, read directly: J-turns are triggered from the anterior-ventral tectum and/or the
    adjacent pretectum;
  - full text, summarising fetch (secondary): "illumination on the right side evoked tail bends
    exclusively to the left", often with "convergent eye movements".
- **Lau, Fitzgerald & Bianco 2025** (Curr Biol 35:4408, PMC7618495, full text read directly):
  > "surprisingly, there was minimal recruitment of the ventral RSNs (RoV3, MiV1, and MiV2) that are
  > involved in other types of turn. Instead, J-turns associated with asymmetric activity in
  > laterally located u508 populations ('Mi3 lateral' and 'Ca1 lateral'), ipsilateral to turn
  > direction."
  - Steering outside hunting (their ar4/5) is "mostly restricted to the RoV3 and MiV2 clusters".
  - nMLF cells fall mostly in the speed archetypes (ar1-3), which "showed minimal sensitivity to
    swim direction".

**What this does to the brief.**
1. **"Hunting J-turn neurons cluster in RoV3 and MiV2" is contradicted.** RoV3 and MiV2 hold the
   *general* steering cells, and they are minimally recruited in hunting J-turns. The hunting side
   signal sits in "Mi3 lateral", "Ca1 lateral" and RoM2r, none of which the Fish1 atlas labels.
   The turning populations therefore feed the G1 turn index (phototaxis / OMR-type turns) and are
   **not** part of the strike readout.
2. **The side term is the strike rule's weakest part.** Thiele 2014 shows that nMLF activity on one
   side deflects the tail to that side (causal, optogenetic). Lau 2025 finds nMLF activity mostly
   speed-related and direction-insensitive during natural swims. We use Thiele's causal result, and
   flag the conflict.
3. **Direction watch-point.**
   - Fajardo: right tectal stimulation bends the tail left, so a lobe steers toward the half of
     space it sees.
   - Gahtan's anatomy wires each nMLF to its ipsilateral tectum. Thiele's rule then turns the fish
     away from that half of space.
   - Both cannot hold for the nMLF alone. If Fish1's tectum -> nMLF route is ipsilateral (section 7
     says its direct contacts are), a simulation with this rule would strike *away* from the prey.
   - **Expected, not measured.** That is a result to report if it happens, not a threshold to
     tune.

**The rule** (`readouts.strike_decision`; thresholds fixed 2026-09-26 01:39:52 EDT, before any
simulation used these populations):
- Inputs: per-cell spike counts in one decision window for nMLF_left, nMLF_right, nIII_dorsal_left
  and nIII_dorsal_right (`readout_sets` gives the segment ids).
1. **Convergence:** both eyes turn nasally, so both medial-rectus pools must fire:
   min(active fraction of nIII_dorsal_left, of nIII_dorsal_right) >= **0.2**. An active fraction is
   the share of cells with at least one spike.
2. **Drive:** max(active fraction of nMLF_left, of nMLF_right) >= **0.2**.
3. **Side:** SI = (mean nMLF_right - mean nMLF_left) / (sum), in mean spikes per cell.
   - SI >= **0.2** (a 1.5 : 1 ratio): strike **right**.
   - SI <= -0.2: strike **left**.
   - otherwise: strike **ahead** (the frontal binocular zone).
- **Strike = convergence AND drive, toward the side given by rule 3.**
- Fractions and per-cell means make the rule blind to the left/right population-size difference.
  A test checks that pools of 4 and 8 cells with equal per-cell activity give SI = 0.
- **nIII dorsal half = MR/IR proxy:** within each side, the 50% of atlas nIII somas with the
  lowest z (dorsal). The z cut is 4,985 on the left and 4,772 on the right, in 8 nm voxels
  (strict). IR is in the same pool and cannot be removed; SR (contralateral) and IO sit ventrally
  and are excluded.

## 6. Escape and turn readouts

- **Escape** (`escape_decision`):
  - Any M-series cell (Mauthner, MiD2cm, MiD3cm, either side) with at least 1 spike in the window
    is an escape. The initiator is the earliest first spike.
  - M-series axons cross the midline, so the body bends away and the escape goes to the **other**
    side from the initiating cell. Dunn et al. 2016: a right-field loom evokes escapes to the left
    through the right M-system.
  - A same-time tie goes to the side with more M-series spikes; a full tie gives no side.
- **Turn index** (`turn_index`, G1's definition): TI = (mean right - mean left) / (sum) over the atlas
  turning populations (RoV3 + MiV1 + MiV2 per side, strict: 32 L / 23 R). TI > 0 = right. Per Huang
  2013 and Lau 2025, this reads general steering, not hunting J-turns.

## 7. Wiring-only checks (no simulation; `readouts.json` -> `wiring`)

**Method.**
- G1's stimulated tectal cells, one lobe at a time: 11,794 left and 13,119 right. Per-synapse sign.
- For each readout set: the excitatory synapses it receives from cells that can fire (reachable
  from the lobe through excitatory pairs: 106,283 cells for the left lobe, 106,325 for the right),
  per target cell; and the direct synapses from all of B1's tectal segments of that lobe.
- The routing index is (ipsi - contra) / (ipsi + contra), with ipsi = left lobe -> left target +
  right lobe -> right target. Each target counts once each way, so its own input bias cancels.
- Null: the pairs' postsynaptic ends were permuted 20 times, seed `G1b-wiring-post-permutation`.
  Pre-ends, counts and signs were kept; duplicates were not repaired, unlike `sim.shuffled_copy`.

| readout | fireable-input routing index (null mean +- SD, p) | direct synapses ipsi / contra | binomial p (ipsi vs 50%) |
|---|---|---|---|
| nIII dorsal | +0.024 (+0.001 +- 0.023, 0.35) | **16 / 3** | **0.004** |
| nMLF | +0.011 (0.000 +- 0.042, 0.85) | 3 / 1 | 0.63 |
| forward | +0.006 (-0.008 +- 0.023, 0.60) | **18 / 0** | **< 1e-4** |
| turning | +0.001 (+0.008 +- 0.036, 0.85) | 2 / 0 | 0.50 |
| Mauthner | +0.003 (-0.026 +- 0.076, 0.30) | 4 / 0 | 0.13 |
| MiD2cm | +0.018 (-0.010 +- 0.098, 0.80) | 0 / 0 | n/a |
| MiD3cm | +0.002 (+0.008 +- 0.123, 1.00) | 1 / 0 | 1.00 |

- **The reachable network does not route by side:** every routing index is within its null.
- **Each target's own bias dominates, whichever lobe is driven.** Per-cell target laterality
  (R - L) / (R + L), left lobe / right lobe: Mauthner -0.35 / -0.34 (G1's 2:1 left bias: 543 vs 262
  / 267 excitatory synapses), nMLF +0.18 / +0.21, nIII dorsal -0.20 / -0.16, turning +0.05 / +0.05.
- **Direct tectal contacts are ipsilateral:** significant for nIII dorsal and the forward cells, too
  few to test for nMLF.
- The permuted graph gives far more direct contacts (null means 23-56 per side for the multi-cell
  readouts, 3-7 for single cells) than the real one,
  because real tectal outputs stay local. The raw counts are therefore not compared with the null,
  and the permutation p for the ipsilateral fraction is not reported as evidence (the counts are too
  small). The binomial column is the test.
- Minimum hops from a lobe: 1 to the same-side nMLF and nIII dorsal; 2-3 to the other side.
- The fireable input per cell is tiny for nIII dorsal (0.6-0.9 excitatory synapses) and nMLF
  (2.6-4.0), against 19-26 for forward/turning, 66-115 for the MiD cells and 262-543 for Mauthner.
- **Expected consequence:** under G1's segment-level model these new readouts will be as silent as
  the old ones. The readouts do not fix G1's detached-axon problem.

## 8. `readouts.json` schema

Top-level keys (the file also carries this as `schema`):
- `populations.{strict|dilated_5um}.{turning|forward|m_system|strike}.{region}.{left|right}` =
  `[member]`.
  - member = {seg_id, lore_ids, pos (8 nm voxels, mean over the segment's somas), lat_um,
    rhombomere, in_graph, n_in, n_out, in_type2}.
  - nIII members also carry `nIII_dorsal` (bool).
- `population_stats`, `symmetry`, `atlas_region_symmetry`: counts and tests from section 2.
- `spn_validation.{strict|dilated_5um}` and `mirror_picks_vs_atlas`: section 3, with a per-cell table
  (strict region, nearest region and distance, and the same for the mirror point).
- `m_system`:
  - `Mauthner.{left|right}` = {seg_id, lore_id} (from B1);
  - `MiD2cm|MiD3cm.{left|right}` = {task_criteria, task_criteria_lore_id, task_criteria_input_rank,
    preregistered_crossing, preregistered_crossing_lore_id, agree};
  - `*_mirror_distance` for each pick set, and `confidence`;
  - `identification`: the full candidate tables, `positive_control_r4`, `positive_control_passed`
    and `shared_afferents`.
- `lesion_sets.{task_criteria|preregistered_crossing|union}` = [seg_id].
- `readout_sets.{strict|dilated_5um}` = {nMLF_*, nIII_*, nIII_dorsal_*, turning_*, forward_*,
  Mauthner_*, MiD2cm_*, MiD3cm_*} = [seg_id]. These are ready for the simulator; `readout_sets(ro,
  population_set, m_pick)` rebuilds them.
- `rules`: STRIKE_RULE, ESCAPE_RULE, TURN_RULE, NIII_DORSAL_FRACTION and the docstring text of each
  rule.
- `wiring`: section 7.
- `atlas_box`: b0 / b1, shape, mip, both sha256s, and the equality check with B1's box.
- `labels`: used and unused.
- `built`, `code_version` (`g1b-readouts-v1`), `build_seconds`.

## 9. Tests

`cd launches/fishbrain/brain && .venv/bin/python -m pytest -q tests/test_readouts.py`: **exit code 0,
14 passed in 0.58 s** (2026-09-26). The tests run offline. They read `readouts.json` plus B1's cached
box and the lore csv; they never read the network or the build cache. A missing `readouts.json` makes
them fail, not skip.

1. **Positive control.** The labels are recomputed from B1's box, independently of the build's
   fetched box. 12 named SPNs are in a region, 11 in their own; each of the 11 is a member of its
   own population on its own side.
2. The same check fails when left and right populations are swapped: 0 found.
3. Own-name matches beat a 2,000-permutation name null (p < 0.01).
4. The mirror test is recorded, and every region has at least 1 member on each side.
5. Every member lies on its declared side, at least 3 um from the midline. Its segment equals the
   csv mapping of its lore ids. Its region, recomputed from B1's box (single-soma members whose
   3 x 3 x 3 neighbourhood lies inside it), equals its population.
6. The dilated populations contain the strict ones.
7. Readout sets are disjoint, and nIII dorsal is the low-z half.
8. M-system:
   - the Mauthner ids equal B1's, and the r4 positive control returns them with 271.1 / 247.6 um^3;
   - each task pick is the largest candidate and ranks first by inputs, lies on its side and is more
     than 50% in its region;
   - there are 4 distinct picks, none of them Mauthner;
   - the lesion sets are well formed.
9. Synthetic rules:
   - strike right, and the mirrored input strikes left;
   - no strike with one eye only or with no nMLF drive;
   - symmetric nMLF strikes ahead, whatever the pool sizes;
   - the escape goes away from the first M-series cell, including a homolog, and a full tie gives no
     side;
   - the turn index sign is right.
10. The wiring checks are recorded with a null of at least 20 shuffles.

**The tests can fail (mutations run 2026-09-26):**
- Dropping one own-region named SPN (MiV2 185944) from its population fails test 1.
- Swapping MiV1's left and right members fails test 5.
- A strike rule with the convergence gate set to 0 returns strike = True for one-eye input, which
  test 9 asserts is False. This one was shown by calling the rule directly.

## 10. Choices we made (for the honesty page)

Everything here is ours unless a source is named.
1. **Population membership.** B1's plurality vote at 1024 nm (strict, the default). Side from the
   fitted midline. Somas within 3 um of the midline dropped. Segments that also hold a soma outside
   the region or side dropped. The 5 um dilation is post hoc (r chosen after seeing the SPN curve).
2. **Which regions mean what.**
   - turning = RoV3 + MiV1 + MiV2 (Huang 2013);
   - forward = RoM1, MiM1, MiR1, MiR2, RoL-R1 (the HMI classifier);
   - strike = NucMLF + nIII.
   - RoM2, RoM3, RoL2, RoL3, MiT, CaD, CaV, the spiral fibers and nIV are unused.
3. **nIII dorsal half = the MR proxy.** Greaney 2017 puts IR/MR in dorsal nIII; the 50% cut is ours.
   IR stays in.
4. **Strike rule.** Convergence = both dorsal-nIII pools at least 20% active. Drive = one nMLF at
   least 20% active. Side = nMLF SI with a 0.2 dead zone that means "ahead". The side comes from
   Thiele 2014's ipsilateral tail deflection. Every threshold is ours. The rule was fixed before any
   simulation.
5. **Escape rule.** One spike of any M-series cell. The initiator is the earliest; direction away
   from its side (Dunn 2016).
6. **M-homolog identification.**
   - The task's criteria: the largest overlapping soma, which here is also the input leader.
   - The midline-crossing rule was pre-registered and then set aside after seeing the data
     (section 4, with times).
   - Confidence labels are judgments.
   - The shared-afferent test (VIIIth-nerve zone = more than 50 um lateral, B1's threshold) is post
     hoc.
7. **Soma boxes:** region bbox + 20 um at 512 x 512 x 60 nm (B1's Mauthner recipe). Candidates =
   unclipped somas with any voxel in the region; top 5 per side.
8. **Validation nulls:** name permutation (10,000, seed `G1b-spn-name-permutation`); wiring
   post-permutation (20, seed `G1b-wiring-post-permutation`, duplicates not repaired).
9. **Distances:** to the nearest atlas voxel centre at 1024 nm, so a soma inside a voxel can read up
   to about 0.9 um.

## 11. Limits and open decisions

- **Nothing was simulated.** Every behavioural reading of these populations is expected, not
  measured. The readouts do not address G1's cause (detached axons). Section 7 predicts the new
  readouts receive as little fireable input as the old ones.
- **`brain.stimulated_set()` is not updated for the new readouts** (not my file). It removes tectal
  cells that directly contact the *old* readouts only. Tectal cells directly contact the new ones:
  16 + 3 synapses onto nIII dorsal, 4 onto nMLF, 18 onto forward and 2 onto turning cells. Before a
  gate uses these readouts, `stimulated_set` should exclude those tectal cells too, as it did for
  Mauthner and the SPNs.
- **The build caches** (atlas box 1024 nm; r5 and r6 soma boxes) sat in a scratch folder that was not
  kept, not in `data/ref/` (not my files). The atlas
  box is pinned in `readouts.json` by sha256, and its overlap with B1's box is checked on every build.
- **The strike side term is the weakest link** (section 5): Lau 2025 finds the nMLF direction-
  insensitive, and the hunting-specific lateral populations are not labelled in the Fish1 atlas.
  Mapping "Mi3 lateral" (r6, lateral) and "Ca1 lateral" onto Fish1 would need their coordinates from
  Lau 2025's figures. Not done.
- **MiD3cm-left is contested** (section 4). The lesion sensitivity rows exist for this.
- **MR vs IR** cannot be separated in nIII from the atlas. nIII's L/R asymmetry follows the atlas
  region's registration.
- **The left side still rests on B1's axis call.** A mirror error would flip every side claim
  consistently.
- The HMI release's "_prox" cells (22 of 50) are lower-confidence names; they are included in every
  count above.

## Sources

- Fish1 layers (public, no login, read 2026-09-26): `precomputed://gs://fish1-public/mece2_231218`,
  `mece1_231218` (segment_properties + labels), `lores_cbs_231218`, `seg_241003_agg241003` (meshes).
  Brain of record as in G1-pull.md. CAVE somas v709 from B1's cache (no CAVE call).
- Liu KS, Fetcho JR (1999) Neuron 23:325-335, PMID 10399938 (abstract, Europe PMC).
- Gahtan E, Tanger P, Baier H (2005) J Neurosci 25:9294-9303, PMID 16207889 (abstract, Europe PMC).
- Thiele TR, Donovan JC, Baier H (2014) Neuron 83:679-691, PMID 25066082 (abstract, Europe PMC).
- Greaney MR et al. (2017) J Comp Neurol 525:65-78, PMID 27197595 (abstract, Europe PMC; PMC5116274
  via summarising fetch).
- Fajardo O, Zhu P, Friedrich RW (2013) Front Neural Circuits 7:67, PMID 23641200 (abstract, Europe
  PMC; PMC3640207 via summarising fetch).
- Lau JYN, Fitzgerald JE, Bianco IH (2025) Curr Biol 35:4408-4425, doi 10.1016/j.cub.2025.07.066,
  PMC7618495 (full text, Europe PMC XML).
- Kohashi T, Oda Y (2008) J Neurosci 28:10641, PMC6671347 (summarising fetch).
- Nakayama H, Oda Y (2004) J Neurosci 24:3199 (title and search summary only).
- Bianco IH, Kampff AR, Engert F (2011) Front Syst Neurosci 5:101 (search summary only).
- Huang KH et al. (2013) Curr Biol 23:1566 and Dunn TW et al. (2016) Neuron 89:613 as cited in
  G1-gate.md s.4 (not re-read today).
- Citation required by the Fish1 data policy: Petkova, Januszewski et al. (2025), "A connectomic
  resource for neural cataloguing and circuit dissection of the larval zebrafish brain", bioRxiv.
