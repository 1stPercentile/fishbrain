---
publication_copy: edited-for-publication copy of launches/fishbrain/evidence/G1-cells.md at commit 9ef0eb2ab. Local paths made repo-relative; machine-owner, internal coordination, account and credential details removed. No number, hash, time, method, result or correction was changed; publication notes are marked [Publication note].
title: G1 cells, inputs and outputs of the fish brain (task B1)
created: 2026-09-25
status: done; the integration graph checks (section 8) ran in G1-gate (Mauthner counts reproduce exactly); left turning readout remains unconfirmed
evidence_status: measured. Numbers were read from public Fish1 data on 2026-09-25 unless marked
  "choice" or "expected".
outputs: brain/fishbrain/cells.py, brain/data/cells_identified.json (5.7 MB, rebuild with
  `python -m fishbrain.cells`), brain/data/ref/ (cached inputs), brain/tests/test_cells.py
---
# G1 cells: where vision enters and where behaviour leaves

**TLDR.** Axes settled: x runs anterior to posterior, z dorsal to ventral, and y from the fish's
right (low y) to its left (high y). **Both Mauthner cells found:** left = segment 15773771512
(lore 187053) and right = segment 16304202596 (lore 187052). Each is the largest soma on its side
of rhombomere 4, sits in the Fish1 atlas's own "Mauthner" region and has the most input synapses of
the candidates. **The 50 named SPNs are mapped (50/50 verified at the soma), but 49 of the 50 are on
the fish's RIGHT.** The release traced one side only, so left-side turning readouts are unconfirmed
mirror candidates. **Tectum:** 44,435 segments (21,402 left lobe, 23,033 right lobe). `tectum_drive`
maps each eye to the opposite lobe by the published retinotopic map. **The Mauthner segments hold soma, dendrites
and the first few microns of axon only.** The giant descending axon is not attached, so escape is
read from the soma segment's spikes, not from its outputs.

Coordinates everywhere: **voxels of 8 x 8 x 30 nm**, written (x, y, z). Segment ids are
`seg_241003_agg241003` agglomeration ids, the ids used by the brain of record's
`pre_synaptic_cell` / `post_synaptic_cell` relationships.

## 1. Orientation

| Axis | Meaning | Evidence (measured) |
|---|---|---|
| x | anterior (low) to posterior (high) | Atlas `mece0/1_231218` region centroids at the 4096 nm mip: Olfactory Bulb x = 17,989; Telencephalon 26,761; Diencephalon 42,610; Tectum 53,548; Hindbrain 72,904; Spinal Cord 107,864. The named reticulospinal cells follow the same order: RoL-R1/RoM1 (r1-r2) x = 60.9k-63.2k < RoV3 (r3) 65.1k-68.0k < MiV1 (r4) 68.0k-69.7k < MiV2 (r5) 72.1k-74.1k. |
| z | dorsal (low) to ventral (high) | Pineal z = 2,437 vs Intermediate Hypothalamus 8,002; Torus Longitudinalis 1,760 vs Torus Semicircularis 7,457; Cerebellum 3,097 vs Interpeduncular Nucleus 7,214 and Inferior Olive 7,307. |
| y | fish's RIGHT (low) to fish's LEFT (high) | The authors' state `LateralLineVisualization.json` names two segment groups `pre_prop_MONs_left` and `pre_prop_MONs_right`. The mean mesh centroid of 12 segments each (detail level 3) is y = 44,009 for left and y = 21,879 for right, both at x ~77-78k and z = 3,068 (dorsal hindbrain, where the MON is). The Mauthner pair (section 3) lies 42.3 um left and 39.7 um right of the fitted midline. |

**Midline:** fitted as the line `y_mid = 31251.9 + 0.021974 x` through 11 unpaired atlas structures:
Pineal, Interpeduncular Nucleus, Torus Longitudinalis, Area Postrema and the seven raphe
subdivisions. The RMS residual is 1.35 um. The fish is slightly yawed in the volume: the midline
moves from y = 32,047 at x = 36k to 33,344 at x = 95k. The HMI analysis code's `MIDLINE_Y = 32804`
agrees in the hindbrain (the fit gives 32,799 at the Mauthner x).

**Volume:** 280,000 x 65,000 x 8,689 voxels (2.24 x 0.52 x 0.26 mm). The atlas also labels
Retina, Ganglia and Spinal Cord. No CAVE soma falls in the atlas "Retina" label at the 4 um mip,
so the model has no retinal ganglion cell somata. Soma counts per division at that mip: Hindbrain
75,175; Midbrain 54,108; Forebrain 42,021; Spinal Cord 5,737; Ganglia 3,244; unlabelled 6,767.

**Left/right caveat.** The side comes from the authors' layer names. We did not test it against
an independent anatomical asymmetry such as the habenulae. A mirror error would flip the whole fish
consistently, because eyes, tectum, Mauthner and SPNs all use the same y rule. Behaviour would
then be a mirror image, not broken.

## 2. Unit and id findings that every consumer needs

- **CAVE `pt_position` is in 8 x 8 x 30 nm voxels, not the 16 x 16 x 30 nm stated in Fish1's
  programmatic docs.** Test: SPN somas 158660, 178639, 179680 and 186978 were each looked up in the
  16 nm agglomeration over a 9 x 9 x 3 neighbourhood. Halving x and y (the 8 nm reading) returned
  the expected segment in 243/243 voxels for all four. Reading the position as 16 nm returned
  background (0) in 243/243 voxels. Corroboration: the docs' own code passes
  `coord_resolution=(8, 8, 30)`; CAVE y reaches 63,488, which exceeds the 16 nm y extent (32,500);
  and HMI's midline 32,804 is about half the 8 nm y extent.
- **Soma segmentation labels are lore ids:** `lores_cbs_231218` returns the lore id itself at 6/6
  somas checked (158660, 178639, 179680, 186978, 199, 284). `hindbrain_reconstructions` is also
  keyed by lore id.
- **Lore id to segment id:** `agglomerated_segments_and_soma_ids.csv` from the paper's TEN
  analysis (187,052 rows) matches the public agglomeration at the soma for every soma checked: 50/50
  SPNs, 2/2 Mauthner and 60/60 random tectal cells (243/243 voxels each). 4,982 somas map to 0
  (no segment). 180,782 distinct segments cover the rest, so some segments hold several somas
  (merge errors).
- **CAVE somas table:** materialization **v709**, 187,052 rows, cached at
  `data/ref/cave_somas_full.parquet`. Five 503s came first (retried with backoff); the sixth
  attempt succeeded. Lore id 176154 is absent; ids run 1 to 187,053.

- **Absent-chunk check (positive control for the enumeration reads).** The enumeration reads use
  `fill_missing=True`, so a lost chunk would read as "unlabelled / no soma". We re-read every chunk
  of the four boxes with `fill_missing=False` (`data/ref/absent_chunk_scan.json`):
  - tectum atlas box (1024 nm): 46 of 200 chunks absent;
  - hindbrain atlas box: 2 of 40 absent;
  - r4 soma box: 2 of 300 absent;
  - r4 atlas box: 34 of 300 absent.
  **No absent chunk lies where the next-coarser mip has any label.** Of the absent chunks, 42/46,
  2/2, 2/2 and 18/34 are present and all-zero at the coarser mip; the rest are absent there too.
  The two absent soma-box chunks contain no CAVE soma. The absent atlas chunks that do contain
  CAVE somas are unlabelled at the coarser mip, so those somas are genuinely outside every atlas
  region. No Mauthner candidate sits in an absent chunk, and no absent chunk overlaps the atlas
  Mauthner region. We read the absent chunks as unwritten empty chunks; the counts above stand.

## 3. Mauthner cells

**Method.** We read `lores_cbs_231218` (soma segmentation) and `mece2_231218` (atlas) at 512 x 512 x
60 nm. The box covers the atlas region "Hindbrain/Rhombomere 4/Mauthner" (mece2 id 48) plus a 20 um
margin: x 67,072-74,688, y 23,552-42,944, z 4,324-6,810. It holds 2,186 whole (unclipped) somas
with a median volume of 40.3 um^3 (99th percentile 80.1 um^3). We ranked them by volume per side,
then checked the top 5 per side against the brain of record's synapse relationship index (inputs =
`post_synaptic_cell`, outputs = `pre_synaptic_cell`) and against the agglomeration meshes.

| | LEFT | RIGHT |
|---|---|---|
| segment id | **15773771512** | **16304202596** |
| lore id | 187053 | 187052 |
| soma (8 nm) | (70336, 38080, 5256) | (70464, 27840, 5309) |
| distance from midline | 42.3 um | 39.7 um |
| soma volume | 271.1 um^3 | 247.6 um^3 |
| vs next-largest soma on that side | 2.04x (lore 187031, 133.0) | 1.65x (lore 187043, 149.8) |
| vs 99th percentile of the box | 3.38x | 3.09x |
| share of soma voxels in atlas "Mauthner" region | 1.00 | 0.70 |
| input synapses (brain of record) | **5,849** (type 1: 1,294, type 2: 4,555) | **4,173** (629 / 3,544) |
| vs most-connected other top-5 candidate | 2.03x (2,883) | 1.53x (2,734) |
| output synapses | 38 | 25 |
| segment at soma | 243/243 voxels | 243/243 voxels |
| mesh: lateral reach from soma | 56.3 um | 56.9 um |
| mesh: ventral reach from soma | 90.8 um | 85.0 um |
| mesh: past the midline | 4.8% of vertices, up to 2.1 um | 4.7% of vertices, up to 6.1 um |
| mesh: posterior reach | 30.8 um | 24.1 um |

No other soma in either side's top 5 touches the atlas Mauthner region (share 0.00). The
segment's shape matches a Mauthner cell: a large lateral dendrite (~56 um, mirror-symmetric
between sides), a long ventral dendrite (85-91 um), and an axon that runs medially through the
atlas's "Mauthner Cell Axon Cap" region (24 and 143 level-3 vertices inside its box) and just
crosses the midline. The two lore ids are the last two in the table (187052, 187053), which
suggests they were added by hand. That was not checked.

**What is NOT there (plain):** the agglomeration cuts each Mauthner segment 2-6 um past the
midline. The giant axon that descends the contralateral hindbrain and spinal cord is not in the
segment; the mesh ends 24-31 um behind the soma. That is why each has only 25-38 output synapses,
and why none of those outputs is contralateral (left 0.00, 74% posterior; right 0.00, 4% posterior).
The segments touching each cut end (a 49 x 49 x 21 voxel box at 16 nm) are small fragments (4-370
level-3 vertices; the largest spans ~12 um in x and ~3.6 um in y). None runs toward the spinal
cord, so no descending continuation was found. **Consequence for the model:** read
the escape from the Mauthner soma segment's spikes. Do not expect it to drive downstream motor
neurons through the graph.

## 4. Spinal projection neurons (readouts)

All 50 `spn_*` rows of the HMI release's `em_zfish1_dataframe.xlsx` are in `cells_identified.json`
`spn[]`. Each has its lore id, segment id, side, class (turning / forward, from the classifier),
name (RoV3, MiV1, MiV2, RoM1, MiM1, MiR1, MiR2, RoL-R1), the `prox` flag, whether it is in the
authors' display state, its position, its atlas region and its segment check.

- Positions: all 50 come from CAVE v709. For 47 of them this is identical to the HMI cache
  `cave_somas_in_big_box.csv` (30,346/30,346 rows of that cache match exactly); 147392, 168158 and
  176311 lie just below that cache's z limit.
- **Segment check: 50/50** return the csv's segment in 243/243 voxels at the soma. No SPN maps to
  0, and no two share a segment.
- Classes: 31 turning (RoV3 19, MiV1 6, MiV2 6) and 19 forward (RoM1 8, MiM1 5, MiR1 3, RoL-R1 2,
  MiR2 1).
- **The authors' own state `hindbrain_motion_integrator.json` shows exactly the 28 cells without
  `_prox`** (15 `spn_turning` + 13 `spn_forward`, identical sets). The 22 `_prox` cells are not
  displayed there, and the release does not define the suffix. Treat `prox: true` as lower
  confidence.
- **Side: 49 right, 1 left.** The one "left" cell (184054, RoV3 `_prox`) is only 5.2 um past the
  midline; the other 49 are 2.3-71.9 um right of it. The HMI reconstruction is one-sided. **So there
  is no confirmed left-side turning readout.**
- Atlas cross-check: 12 SPN somas land inside an atlas region; 11 of them are in the region of
  their own name (for example, RoV3 cells in "Rhombomere 3/RoV3" and MiM1 cells in "Rhombomere
  4/MiM1"). The exception is 186443, a RoV3 that lands in "RoM3". The other 38 land outside the
  atlas's small reticulospinal regions.
- **Other-side candidates (`spn_mirror_candidates`, UNCONFIRMED):** each named cell was reflected
  across the midline, and the other-side soma within 10 um of that point with the best score
  (distance / 5 + |ln volume ratio|) was chosen, one partner per cell. Found for 49/50 (RoM1 180791
  had no candidate). The median distance to the mirror point is 3.4 um (max 8.9). **These picks are
  weak:** a median of 12.5 somas lie within 10 um of each mirror point. They are there so the
  integration step can test them. Graph test in section 8.

Readout sense: in larval zebrafish, RoV3, MiV1 and MiV2 are needed for turning, and their
activity is lateralised to the side of the turn (Huang, Severi, Orger et al. 2013, Curr Biol,
"Spinal projection neurons control turning behaviors in zebrafish"; recalled from literature,
not re-read today). Forward-class cells come from the release's classifier. **Expected, not
measured:** right-side turning SPN spikes mean a right turn.

## 5. Optic tectum

**Rule (choice):** a soma belongs to the tectum when the plurality atlas label in a 3 x 3 x 3
neighbourhood around its CAVE `pt_position` is "Midbrain/Tectum/Stratum Periventriculare" (mece2 17)
or "Midbrain/Tectum/Neuropil" (mece2 18). The vote uses the 1024 x 1024 x 960 nm mip, about +-1 um.
We checked what the region layers encode before relying on them. Their `segment_properties` hold
labels: `mece0` = 6 major divisions, `mece1` = 31 subdivisions (Tectum = 21), `mece2` = 70 nuclei
and layers (SPV = 17, neuropil = 18, Mauthner = 48, Mauthner Cell Axon Cap = 52, the reticulospinal
groups and more), `mece3` = 21 finer areas. `tissue_type_32nm` is a tissue-class mask
(neuropil / cell bodies / folds / hair cells / non-neuron / support / do-not-segment) with no
regions, so it was not used. mece2 17 + 18 (106,244 coarse voxels) almost exactly tile mece1
"Tectum" (105,899).

Counts (measured):
- 45,458 somas in the two labels: SPV 43,812, neuropil 1,646; left 21,913, right 23,545.
  CAVE `cell_type`: 10,808 exc, 8,389 inh, 26,261 na.
- Dropped: 57 with no segment; 32 whose segment also holds a non-tectal soma or a soma on the other
  side; 728 within 3 um of the midline (side ambiguous).
- 432 tectal somas share a segment with another soma. Where all of a segment's somas are tectal
  and on one side, it is kept as one entry with the mean position: 188 such segments.
- **Kept: 44,435 segments. Left lobe 21,402, right lobe 23,033; layer SPV 42,795, neuropil 1,640.**
  None of them is in the exclusion set.
- Segment check on a seeded random sample: 60/60 match at the soma (median fraction 1.00).
- Geometry: inside each lobe, depth (z) correlates with distance from the midline (left 0.74,
  right 0.77), so the dorsal tectum is the medial tectum. Correlation of x with z: -0.16 left,
  0.02 right.

## 6. Retinotopy: `tectum_drive(stimulus)` (fishbrain/cells.py)

**The documented larval zebrafish map:**
- Each eye projects to the **contralateral** tectum.
- **Anterior-posterior:** stimuli in front of the larva reach the temporal retina, which projects
  to the anterior tectum; stimuli behind the animal reach the nasal retina and posterior tectum
  (Förster, Helmbrecht, Mearns, Jordan, Mokayes & Baier 2020, eLife 9:e58596, "Retinotectal
  circuitry of larval zebrafish is adapted to detection and pursuit of prey"). Stuermer 1988
  (J Neurosci 8:4513, "Retinotopic organization of the developing retinotectal projection in the
  zebrafish embryo") is the classic source for the map (title and abstract listing seen, full text
  not read today).
- **Dorsal-ventral:** dorsal retinal axons go to the ventral hemitectum and ventral retinal axons
  to the dorsal hemitectum. This is the standard retinotectal result; the search summary we read
  stated it without naming a paper, and Stuermer 1988 was not re-read today. The eye's optics invert the image, so the upper visual
  field falls on the ventral retina: **upper field -> dorsal (in Fish1, dorsomedial) tectum; lower
  field -> ventral (ventrolateral)**.

**Model (code: `lobe_coords`, `preferred_field`, `TectumMap.drive`):**
- `u` = rank of x within the lobe (0 = anterior, 1 = posterior).
- `v` = rank of (rank(z) + rank(|y - y_mid|)) within the lobe (0 = dorsomedial, 1 = ventrolateral).
- Preferred eye-lateral azimuth `theta = -20 + 180 u` degrees; preferred elevation `el = 70 - 140 v`.
- Stimulus: `{"left": [(azimuth, elevation, contrast), ...], "right": [...]}`. Azimuth is
  body-centred (0 ahead, positive toward the fish's RIGHT). Eye-lateral theta = azimuth for the right
  eye and -azimuth for the left. The left eye drives only right-lobe cells, and the reverse.
- Drive of cell i = sum over that eye's points of `|contrast| * exp(-d^2 / (2 * 15deg^2))`, where d
  is the great-circle angle to the cell's preferred point. A point of contrast 1 exactly at a cell's
  preferred point gives 1.0. Output order = `cells["tectum"]` = `tectum_seg_ids()`. Units are
  "contrast units"; the integration step sets the gain to current.
- Points outside an eye's field (theta < -20 or > 160 degrees) are ignored by that eye.
- `disk(az, el, radius)` fills a disk on a 2-degree grid, so a looming disk recruits more cells
  as it grows.

**Tests** (`tests/test_cells.py`, 8 pass): a left-eye point at azimuth 0, elevation +30 drives only
right-lobe cells, peaking in the anterior third and dorsomedial half. A left-eye point at azimuth
-150 peaks in the posterior third. A right-eye point at (+90, -40) peaks in the ventrolateral half
of the left lobe. A point outside the field gives 0. Contrast -0.8 equals +0.8. A point at the
preferred location gives exactly 1.0. A 30-degree disk gives more than 5x the drive of a 3-degree
one. On the real file: tectum ids are unique, nonzero and disjoint from the exclusions; the
Mauthner pair falls on opposite sides; 50 SPNs; a left-eye disk drives only the right lobe.

Engineering note: numpy 2.0.2 with macOS Accelerate raises spurious "divide by zero / overflow in
matmul" warnings for these products. The values match an einsum reference to 1.7e-13, so
`tectum_drive` silences the flags and raises if any output is non-finite.

## 7. Cells that must never receive direct input (`exclusions`)

`exclusions.seg_ids` holds 390 segments, returned by `never_stimulate()`:
- both Mauthner segments;
- all 50 SPN segments;
- the 49 other-side SPN mirror candidates;
- 315 somas (174 left, 141 right) inside the atlas's reticulospinal and escape-circuit regions:
  NucMLF, RoL2, RoM1-3, RoV3, RoL3, RoL-R1, MiV1-2, MiM1, MiR1-2, MiD2-3, MiT, CaD, CaV, Mauthner,
  Mauthner Cell Axon Cap, and the spiral fiber neuron clusters (per-region counts in the file).

Rule: visual input enters only through `tectum_drive()` into `cells["tectum"]`. Nothing in
`seg_ids` receives injected current; those cells are readouts or sit between senses and readout.
No other non-tectal cell (pretectum, hindbrain) receives visual current either. If a later model
drives retinal afferents, that replaces `tectum_drive`; it is not added on top.

## 8. For the integration step (needs `data/synapses.parquet`)

Run these against the downloaded graph. The numbers in brackets were measured today from the
same layer's relationship index, so a complete download must reproduce them exactly.
1. **Completeness check:** inputs to 15773771512 = [5,849] (type 1 1,294 / type 2 4,555); inputs to
   16304202596 = [4,173] (629 / 3,544); outputs [38] and [25]. Inputs to runner-ups 17405827637 =
   [2,603] and 16039136570 = [2,734]; SPN 158660 (20607607971) = [117] in / [35] out. A mismatch
   means the parquet is incomplete or keyed differently.
2. **Mauthner rank:** among all segments with a hindbrain soma (atlas mece0 6), both Mauthner
   segments should be in the top 1% by input count. Expected, not measured over the whole hindbrain.
3. **Mauthner inputs:** segments of somas in the atlas "Spiral Fiber Neuron" clusters (mece2 29 and
   32, r3) should synapse onto the Mauthner segments near the axon cap (x 70,144-71,168, y
   29,696-36,352, z 5,248-5,760). Statoacoustic (VIIIth nerve) afferents should contact the lateral
   dendrite (y beyond +-50 um from the midline). The contralateral/posterior OUTPUT test will fail
   by construction (section 3), so do not use it.
4. **Vision reaches the readouts:** count synapses and 2-hop / 3-hop paths from `tectum` segments
   to (a) the Mauthner segments and (b) the SPNs. Expected: few or no direct tectum -> Mauthner
   synapses (in larvae the tectal loom pathway reaches Mauthner through intermediate neurons).
   Report the numbers. If there is no path within 4 hops, the escape test cannot pass.
5. **Mirror candidates:** a true left partner of a right SPN should have a similar input count and
   similar presynaptic partner classes (mirrored). Rank each candidate by cosine similarity of
   its input partners' side-flipped identities with its named cell. Keep only those that clearly
   beat the other somas within 10 um.
6. **E/I value check** (not this task's deliverable; recorded because found): the `ei_predictions`
   shader colours type 1 magenta ("e.g., inhibitory"), matching HMI's Inh = #FF00FF, and Fish1's
   programmatic page states "tag='1' = inhibitory, tag='2' = excitatory". Both describe the sibling
   `_ei` layer / CAVE table, not the brain of record. Graph check run today on outputs of 30 random
   CAVE-`inh` cells: 5 had outputs, 95 type 1 vs 6 type 2 (all 5 majority type 1). For 30 random
   CAVE-`exc` cells: 9 had outputs, 12 type 1 vs 4 type 2 (5 majority type 1, 3 majority type 2).
   **Inconclusive:** most soma segments have their axon cut off, so n is tiny, and the exc sample
   leans the wrong way. Redo it on the full parquet: output type fractions over all labelled cells
   (`data/ref/ei_type_check.json` has today's per-cell counts).
7. **Retinal-afferent upgrade path:** segments presynaptic inside the tectal neuropil (mece2 18) with
   no soma are candidate retinal ganglion cell axon terminals. Driving those instead of the tectal
   somata would be closer to real vision. It is not done here.

## Choices we made (for the honesty page)

Everything in this list is ours, not measured:
1. Tectum membership: the atlas plurality vote above. The atlas is a registered region map, so
   boundary cells can be off by a few um.
2. Side of every cell: which side of the fitted midline line it sits on. Tectal cells within
   3 um of the midline are dropped.
3. Merged segments: kept as one tectal input only if every soma they hold is tectal and on one
   side; otherwise dropped. Somas without a segment are dropped.
4. Retinotopic parameters: field -20 to 160 degrees eye-lateral azimuth, -70 to +70 degrees
   elevation, Gaussian sigma 15 degrees, a rank-uniform map (no extra weight for the frontal/upper
   "strike zone").
5. Contrast sign ignored. ON/OFF polarity and size/direction selectivity are not modelled; size
   enters only through how many points a stimulus covers.
6. Tectal somata stand in for retinal input (section 8.7).
7. Mauthner identity: the largest soma per side around the atlas "Mauthner" region, confirmed by
   atlas overlap, input count and mesh shape. The atlas label is the authors'; the pick is ours.
8. SPN side from the fitted midline; class and name verbatim from the release's `classifier`.
9. Mirror candidates: 10 um radius and score distance/5 + |ln volume ratio|, greedy one-to-one.
10. Exclusion set: as listed in section 7.

## Sources (all public unless noted; read 2026-09-25)
- Brain of record: `gs://fish1-public/syn_241003_agg241003_reorient_axde_ei_bayes_idx_pre_250410.precomputed`
  (relationship index read with `fishbrain/precomputed.py`).
- Agglomeration `gs://fish1-public/seg_241003_agg241003` (segmentation + meshes), soma segmentation
  `lores_cbs_231218`, atlas `mece0..3_231218` + `segment_properties`, `hindbrain_reconstructions`,
  `masking/tissue_type_32nm_20200427_50849075`.
- Paper data: `gs://fish1-release/paper_data/HMI_analysis.zip` (em_zfish1_dataframe.xlsx, voxel.py),
  `TEN_analysis.zip` (agglomerated_segments_and_soma_ids.csv).
- Authors' neuroglancer states `gs://fish1-release/assets/neuroglancer_states/202504/`:
  automatic_results_for_browsing, hindbrain_motion_integrator, LateralLineVisualization,
  evaluation_polarity_assignment (copies in `brain/data/ref/`).
- CAVE `fish1_full` `somas` v709 (authenticated, read-only). Cached; not a runtime
  dependency.
- Förster et al. 2020 eLife 9:e58596 (quoted via search summary; the full text failed to load);
  Stuermer 1988 J Neurosci 8:4513 (not re-read);
  Huang et al. 2013 Curr Biol (recalled, not re-read).
- Citation required by the Fish1 data policy: Petkova, Januszewski et al. (2025), "A connectomic
  resource for neural cataloguing and circuit dissection of the larval zebrafish brain", bioRxiv.
