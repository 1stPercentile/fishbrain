---
publication_copy: edited-for-publication copy of launches/fishbrain/evidence/G1c-bridge-spec.md at commit 4861cf1c6. Local paths made repo-relative; machine-owner, internal coordination, account and credential details removed. No number, hash, time, method, result or correction was changed; publication notes are marked [Publication note].
title: G1c bridge spec, frozen before any build or simulation (classes a/b/c, gains, tuning, gate thresholds, publications)
created: 2026-09-26
frozen_at: 2026-09-26 05:34:57 EDT
status: FROZEN (task S). Nothing in this note or in bridge_spec.json was simulated.
spec_file: launches/fishbrain/brain/data/g1c/bridge_spec.json
spec_sha256: 185b378031a3f3d730ed90bfe9cd56aede6aba6ad0c0984032e03a2c5e111499
bridge_version: g1c-bridge-v1
edge_list_sha256: f7f810d56059860dc81d7fa7f064cb248779acec83e00ce88efc6facc8100bf0
evidence_status: rules are choices (marked); counts and IDs are measured from the brain of record (graph and geometry reads); every behavioural consequence is marked "expected".
verification_scope: >
  Read 2026-09-26 04:55-05:35 EDT. Data (read-only): brain/data/net/pairs_*.npy (wiring hash 09a71038...25af2),
  cells_identified.json, readouts.json, g1c/oa_fragments.json, ref/cave_somas_full.parquet, the lore-id csv, ref/mece2 at
  4096 nm; one public network read, precomputed://gs://fish1-public/mece3_231218 at mip [4096,4096,3840] (no login), saved to
  a scratch folder only (sha256 795292769933e360a3bce8188a773b81f4ea5f8e388ce0717f2750da5ca48e49). Literature: Europe PMC REST abstracts and full-text XML read
  directly (Zottoli 1987, Helmbrecht 2018 abstract, Lacoste 2015, Koyama 2016, Marquart 2019, Bhattacharyya 2017, Sato 2007,
  Temizer 2015, Antinucci 2019 full text, Dowell 2025 full text, Bianco & Engert 2015 full text, Gahtan 2005, Lau 2025 full
  text, Förster 2020 full text, Barker & Baier 2015, a 2018 locomotion review full text); Dunn 2016 and Lacoste 2015 PMC
  pages through a summarising fetch (secondary); Helmbrecht 2018 full text was blocked (HTTP 403 at cell.com and
  sciencedirect), so its tract details beyond the abstract are search summaries only (secondary). Fotowat & Engert 2023 from
  the local copy G1-sim used. Not run: the brain, the eye model on any stimulus, any shuffle.
files: G1c task S. This note, brain/data/g1c/bridge_spec.json. No other file written (scratch work in a scratch folder, not kept).
---
# G1c bridge spec (frozen)

**TLDR.** The bridge adds **1,780 edges and 0 cells** to the 24.9 M measured pairs. It uses real Fish1 cells at both ends
of every edge, and one real Fish1 relay per side where the literature names one. The escape crossing is a functional
edge set because no paper identifies the relay. Expected: the fish's decisions run through the bridge, and the headline
decision-path share lands near 0.

1. **(a) Observed-origin fragments: 19 fragments, 646 edges.** Each one gets a link from the 34 nearest stimulated tectal
   somas at its observed tectal end. They are uncrossed, so if they are strong they push the escape toward the threat.
2. **(b) Tract topography: not used** (0 edges). It is non-specific (15.5% of the negative control's input). A
   near-tectum sensitivity row is specified, but it never enters the verdict.
3. **(c) The literature pathway: 1,134 edges.**
   - **Escape (E1), 216 edges:** a sparse whole-lobe grid of 36 tectal cells drives the contralateral M-system (Mauthner,
     MiD2cm, MiD3cm). Labelled "functional edge for an unidentified crossed pathway".
   - **Prey (P1 to P4), 918 edges:** frontal tectum drives one AF7-pretectal (APN) relay soma per side (Antinucci 2019).
     The relay drives nIII dorsal-lateral and the nMLF on both sides, and the turning SPNs on the opposite side.
4. **Gains:** one per class, g_a and g_c, integer multipliers on counts, bounds [1, 32], grid 1-2-4-8-16-32, tuned only on
   the training set with separate seeds. Every stage is normalised to one unit, U = 34 synapse-equivalents.
5. **Gate:** H1, H2, C1-C3, L1 and L2 verbatim from the G1b reopen gate, operationalised in s.10. The kill is the brief's
   only kill.
6. **Mandatory:** a measured-wiring shuffle with the bridge fixed (10 shuffles), bridge ablations, and both metrics per class.

## 0. What was done, and the rules followed

- **Frozen at 2026-09-26 05:34:57 EDT.** `bridge_spec.json` sha256 = `185b378031a3f3d730ed90bfe9cd56aede6aba6ad0c0984032e03a2c5e111499`.
  - The JSON holds every rule, every derived ID list, all 1,780 edges (pre, post, base count, class, pathway, citation
    key), the gate, the tuning procedure and the publications.
  - `edge_list_sha256` = `f7f810d5...c8100bf0`, computed over the canonical edge tuples (the recipe is in the JSON).
- **Nothing was simulated.** The brain and the `Retina` eye model were not run on any stimulus. The lists come from graph
  and geometry reads only. The derivation script is Appendix B.
- **No file outside my two was written.** mece3 went to a scratch folder, not to `data/ref`.
- **Task B, before building:**
  1. Check the JSON's sha256.
  2. Re-derive every list from the rules in its own code, and assert it equals the JSON's lists and edge-list hash.
  3. Re-fetch mece3 and check its sha256.
  4. Re-check collisions with measured pairs.
  - Any mismatch stops the build and is reported. The JSON is the artefact: B never edits or regenerates it, and a
    re-derivation that disagrees never overwrites it.

## 1. Literature (read 2026-09-26; "direct" = abstract or full text read from the source, "secondary" = summarising fetch or search summary)

| Source | How read | What it supports here |
|---|---|---|
| Dunn TW et al. 2016, Neuron 89:613, doi 10.1016/j.neuron.2015.12.021, https://pmc.ncbi.nlm.nih.gov/articles/PMC4742414/ | secondary (summarising fetch; quotes in G1b-structure s.10 returned verbatim) | A right-field loom evokes a leftward escape. M-system ablation perturbs only escapes contralateral to it. The OT-to-M-cell path is left open ("either a direct or indirect path"). Critical size about 72 deg, then about 81 ms to the escape. |
| Zottoli SJ, Hordes AR, Faber DS 1987, Brain Res 401:113, PMID 3815088, https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:3815088 | direct (abstract) | Goldfish: "Tectal stimulation elicits similar postsynaptic potentials (PSPs) in both M-cells". Only the ipsilateral response was localised, to the ventral dendrite, and uncrossed projections were confirmed morphologically. This matches Fish1's O-a finding. |
| Helmbrecht TO et al. 2018, Neuron 100:1429, doi 10.1016/j.neuron.2018.10.021, PMID 30392799 | abstract direct (Europe PMC); full text blocked (403); tract tuning secondary | The approach and escape signals leave the tectum in "two spatially segregated and uncrossed descending axon tracts". Secondary: medial axons are tuned to threats, lateral axons to prey. |
| Lacoste AM et al. 2015, Curr Biol, doi 10.1016/j.cub.2015.04.025, https://pmc.ncbi.nlm.nih.gov/articles/PMC4452389/ | abstract direct; PMC page secondary | Spiral fiber neurons: about 10, bilateral, in r3. Their axons go to the **contralateral** M-cell axon cap, and they are essential for M-cell escapes. They were tested with water puffs and taps only. |
| Koyama M et al. 2016, eLife 5:e16808, doi 10.7554/elife.16808 | abstract direct; review summary secondary (PMC6146226) | Hindbrain feedforward-inhibition motif for the left/right choice, acoustic. |
| Marquart GD et al. 2019, PLoS Biol, doi 10.1371/journal.pbio.3000480, PMC6793939 | direct (full-text XML) | "The M-cell ventral dendrite receives polysynaptic visual information via the optic tectum". Delayed escapes run through a separate prepontine cluster. |
| Bhattacharyya K, McLean DL, MacIver MA 2017, Curr Biol, doi 10.1016/j.cub.2017.08.012 | direct (abstract) | Loom recruits vSPNs and a Mauthner cell (unilaterally) as a function of approach rate. |
| Sato T et al. 2007, J Neurosci, doi 10.1523/jneurosci.0883-07.2007 | direct (abstract) | Tectal neurons projecting to r2 and r6 are distributed along the A-P axis. |
| Temizer I et al. 2015, Curr Biol, doi 10.1016/j.cub.2015.06.002 | direct (abstract) | Dimming, receding and bright looms are much less effective. Tectal neuropil lesions impair the behaviour. |
| Antinucci P, Folgueira M, Bianco IH 2019, eLife 8:e48114, doi 10.7554/elife.48114, PMC6783268 | direct (full-text XML) | AF7-pretectal (APN) hunting command neurons. Contra-projecting cells put terminals "bilaterally in the vicinity of nIII/nMLF and proximal to ventral reticulospinal neurons in the contralateral hindbrain", and single-cell stimulation evokes hunting. Hunting evoked by avOT stimulation requires these neurons. "AF7-pretectum encompasses the ZBB masks 'Optic tract - AF7' and 'Griseum tectale (AF7)'". |
| Dowell CK, Hawkins T, Bianco IH 2025, Curr Biol 35:554, doi 10.1016/j.cub.2024.12.010, PMC7618498 | direct (abstract + full-text XML) | A laterally located subset of medial-rectus motoneurons is active for hunting saccades and gets premotor input from pretectal hunting command neurons. |
| Bianco IH, Engert F 2015, Curr Biol 25:831, doi 10.1016/j.cub.2015.01.042, PMC4386024 | direct (full-text XML) | Prey is attended "approximately −60° (left) to +60° (right)", in the anterior tectum. Tectal assemblies precede convergence. |
| Gahtan E, Tanger P, Baier H 2005, J Neurosci 25:9294, PMID 16207889 | direct (abstract) | nMLF MeLr/MeLc are required for prey capture and work in series with the tectum. |
| Lau JYN, Fitzgerald JE, Bianco IH 2025, Curr Biol 35:4408, PMC7618495 | direct (full text; G1b-readouts s.5) | RoV3/MiV1/MiV2 are minimally recruited in hunting J-turns, and the nMLF is direction-insensitive. |
| Förster D et al. 2020, eLife 9:e58596, PMC7550190 | direct (full-text XML) | The tectum's feature map for prey size and position. Loom and prey RGCs end in different layers, which the eye model does not have. |
| Barker AJ, Baier H 2015, Curr Biol, doi 10.1016/j.cub.2015.09.055 | direct (abstract) | Size classification (small = approach, large = avoid) happens in the tectum. |
| Fotowat H, Engert F 2023, eLife 12:e82916 | direct (local full text) | Looming-selective neurons sit in the hemisphere contralateral to the stimulated eye and relay to the brainstem escape network. |
| Fajardo 2013, Huang 2013, Liu & Fetcho 1999, Greaney 2017, Thiele 2014 | via G1b-readouts / G1-gate (not re-read) | Anterior-ventral tectum stimulation evokes J-turns and convergence. vSPNs fire for their own side's turns. The M-system lesion set. The nIII dorsal MR/IR proxy. The nMLF side rule. |

**The escape relay search (a negative result, recorded as a finding).**
- **Question:** which hindbrain cells carry tectal loom signals to the **contralateral** M-cell?
- **Searched:** Europe PMC and the web for looming, tectum, Mauthner, relay, contralateral, crossing and spiral fiber
  neurons. Read Zottoli 1987, Lacoste 2015, Koyama 2016, Marquart 2019, Bhattacharyya 2017, Sato 2007, Temizer 2015,
  Fotowat & Engert 2023 and the Helmbrecht 2018 abstract.
- **Result: no source identifies the relay.**
  - Dunn 2016 leaves the path open.
  - Zottoli 1987 shows that both M-cells receive tectal PSPs. Only the uncrossed input was localised.
  - Marquart 2019 calls the visual input polysynaptic.
  - Spiral fiber neurons do cross to the contralateral M-cell. They were only shown to respond to puffs and taps.
- **Not used:** a search-engine summary attributed "presumed to initiate the C-bend through the contralateral Mauthner" to
  Helmbrecht 2018. I could not confirm it in any primary text.

## 2. The eye and the stimulated set (choice, pre-registered)

- **Rule:** B1's tectal segments present in the brain of record, minus B1's never-stimulate exclusions.
- **What changed from G1:** G1 also removed tectal cells with a direct synapse onto a readout. G1c drops that step.
  - G1 made the removal so that it could ask whether the wiring relays past hop 1.
  - G1c asks how much of the decision the measured map carries. A direct tectal contact is the most measured path there
    is.
  - G1b-readouts s.11 recommended the opposite, for G1b's question. That is recorded here.
- **Measured:** 24,920 cells (11,799 left lobe, 13,121 right), against G1's 24,913. The set's sha256 (sorted uint64 ids) is
  `8910b216...bc2348`.
- **Re-admitted, 7 cells:** 9570863290, 11019094176, 12569625141, 13446669933, 16670127295, 17057069410 and 20239723569.
  - 16670127295 and 17057069410 are the O-a0 tectal cells with 2 synapses each onto Mauthner-R and Mauthner-L.
  - The other 6 O-a0 cells touching G1b readouts (8183346363, 11426688881, 13181211444, 14650651033, 14670990755,
    17281899416) were already in G1's set.
- **The eye model is unchanged:** `Retina` with `RetinaParams()`, and the same 8 gate seed strings as G1.
  - The retina cache key and the W matrix change with the stimulated ids.
  - Eye input is labelled *chosen (eye model)* and is never counted as measured wiring.

## 3. Class (a): observed-origin fragments (646 edges, 19 targets)

**Eligible.** A segment qualifies when all three hold:
1. oa_fragments.json classes it O-a1 on at least one deciding readout;
2. it is **not** a B1 tectal cell. The 8 O-a0 cells are tectal cells, so the eye drives them directly;
3. it has **at least 1 input synapse inside the tectum mask**.
- **Why the input rule:** where a fragment has only outputs in the tectum, the signal direction there is unknown.
- **Excluded, 7 output-only fragments:** 9695179340 (the only crossed O-a synapse), 11182734458, 15710979012, 18016345121,
  19546036317, 21789873167 and 22687700860.
- **Measured result: 19 eligible.**
  - 17 have no soma.
  - 2 are soma-bearing, non-tectal cells with dendrites in the tectum: 21729058924 (turning-L) and 22545347363 (M-system R).
  - Every one is uncrossed.
  - 16 contact the M-system, 10 of them a Mauthner cell (all on the ventral dendrite, G1c-measure). 6 contact turning
    cells and 1 contacts the left nMLF.
  - Together they make 34 (left) and 28 (right) excitatory synapses on the M-system.

**Link.**
- **Sources:** the U = 34 stimulated B1 tectal somas nearest (3-D, um) to the fragment's `tectal_end_8nm`, in its
  tectal-end lobe. Ties go to the lower seg id.
- **Count:** base 1 per link, so 1 U at gain 1.
- **Measured geometry:** column radius 8.5-14.6 um; nearest soma 1.4-6.2 um.
- **The fragment's own measured synapses are unchanged**, on the readouts and everywhere else.
- G1c-measure's 20-soma column was descriptive. 34 is not a biological column size; it puts every stage on the same unit
  U (s.6).

**What (a) can and cannot do (expected).**
- **The headline:** under the pre-registered metric, a bridged link passes m = 0, so a fragment driven through (a) carries
  m = 0 downstream. (a) raises the synapse-count share, not the headline.
- **Direction:** the fragments are uncrossed. When they fire, they drive the M-system of the tectal lobe's own side, which
  sends the fish **toward** the threat.
- **Timing:** their columns prefer the lower field (G1c-measure s.5), so a loom at elevation 0 reaches them only late.

## 4. Class (b): tract topography (not used)

- **Primary build: 0 edges, no gain.**
- **Why:**
  - O-b is 24-30% of the M-system's input, but also 15.5% of the caudal-hindbrain negative control's. It is not specific.
  - The median O-b fragment stops 59-69 um short of the tectum.
  - The mesh check agreed with the synapse clouds for 19 of 20 fragments.
  - Assigning an unobserved origin is pure model with a lot of freedom, and under the metric it cannot raise the headline.
- **Sensitivity row (optional, never in the verdict):**
  - It runs only after the mandatory publications.
  - Pool: O-b fragments on a deciding readout whose synapse cloud comes within 20 um of the tectum mask. That is
    G1c-measure's post hoc near-tectum pool, 1.5-1.7% of M-system input.
  - Each is linked like (a): the 34 nearest stimulated tectal somas to its closest-approach synapse, in that synapse's lobe.
  - g_b := the tuned g_a. It is never tuned separately.

## 5. Class (c): the literature pathway (1,134 edges, 0 new cells)

| Pathway | Sources -> targets | Edges | Base | U at gain 1 | Basis |
|---|---|---|---|---|---|
| E1 escape crossing | 36 grid cells per lobe -> the 3 M-system cells on the **opposite** side | 216 | 1 | 1.06 | Dunn 2016; Zottoli 1987; functional edge, relay unidentified |
| P1 | frontal-zone grid (292 L / 293 R cells) -> the APN relay on the **same** side | 585 | 1 | 8.59 / 8.62 | Antinucci 2019 (avOT hunting needs APN); Bianco & Engert 2015; Fajardo 2013 |
| P2 | APN (each side) -> nIII dorsal, lateral half, **both** sides (58 L + 48 R in graph) | 212 | 17 | 1.0 (2 relays x 17) | Antinucci 2019; Dowell 2025 |
| P3 | APN (each side) -> nMLF **both** sides (15 + 18) | 66 | 17 | 1.0 | Antinucci 2019; Gahtan 2005 |
| P4 | APN -> turning SPNs on the **opposite** side (32 L, 23 R) | 55 | 34 | 1.0 | Antinucci 2019; Huang 2013 |

**E1 sources (choice).**
- On each lobe's (u, v) map (`cells.lobe_coords`), take the grid points ((i + 0.5)/6, (j + 0.5)/6). The source is the
  stimulated cell nearest each point. Measured: 36 distinct cells per lobe.
- **Why a sparse whole-lobe grid:** Dunn 2016 finds that tectal ensemble activity encodes a critical size (about 72 deg).
  At gain g, the M-system needs about 34/g grid sources at full rate, so it integrates stimulus size.
- **Targets:** all three M-system cells (the M-system Dunn ablated). The lesion sets L1/L2 match them.
- **Why no relay:** see the search in s.1. Spiral fiber neurons (atlas labels 29/32) would add an unsupported
  tectum -> SFN claim and more edges.

**P1 sources (choice).**
- The frontal zone is preferred theta ≤ 60 deg (Bianco & Engert 2015's ±60 deg), on a 6-deg grid in (theta, elev). That
  gives 299 points; measured, 292 and 293 distinct cells per lobe.
- **Why dense:** the eye model has no size tuning (Förster 2020 and Barker & Baier 2015 put size classification in RGC
  layers and tectal interneurons we do not model). Pooling density is the only size code available: the approach relay
  pools about 8.6 U, the escape about 1.06 U.
- The 6-deg spacing (2 x the 3-deg prey) is our choice.
- The tectum -> APN link is labelled "required by ablation (Antinucci 2019 Fig 8), anatomy not shown; ipsilateral assumed".

**The APN relay (rule, then measured).**
- **Rule:** per side, the CAVE v709 soma nearest (um) to any AF7 voxel (mece3 label 3) on that side, with all of:
  - its 4096-nm atlas voxel is mece2 14 (Pretectum) or mece2 4 (Optic Tract and Accessory Optic System). Antinucci
    defines "AF7-pretectum" by the ZBB optic-tract/AF7 masks;
  - its segment is in the graph;
  - its segment holds exactly one soma.
- **Measured:**
  - Left: seg **14241661373** (lore 125577), 0.55 um from AF7.
  - Right: seg **14833177305** (lore 170137), 0.81 um from AF7.
  - Both sit in mece2 label 4. 3,973 / 3,527 somas per side carry those labels.
  - The mece3 AF7 region is 207 / 179 voxels per side. Every one of its voxels sits in mece1 3 (Diencephalon) and
    mece2 0 or 4.
- **Why one relay per side:** relay cells would share identical inputs, so more of them only split the output. One is the
  minimum, and Antinucci found single-cell stimulation sufficient.
- **Fallback, not used:** the soma of those labels nearest the tectum's rostral pole.

**P2-P4 targets (in graph only).**
- **nIII.** The lateral half (lat_um ≥ the side's median) of the strict nIII_dorsal pool: 65 L / 53 R, of which 58 / 48
  are in the graph.
  - 13 L and 12 R dorsal-pool members have no synapses at all. They stay in the strike rule's pool and stay silent.
  - If every bridged target fires, the pool is at most 58/130 = 45% and 48/106 = 45% active. The rule needs 20%.
- **nMLF, strict:** 15 L and 18 R.
- **Turning SPNs, strict (RoV3 + MiV1 + MiV2):** 32 L and 23 R, on the side opposite each relay.
- **Direction check (expected):** a right-field prey reaches the left lobe, then APN-L, then the right turning SPNs. So
  TI > 0 and the fish turns toward the prey (Antinucci: contra-projecting APN cells evoke contralaterally directed hunting).
- **Conflict, disclosed:** Lau 2025 says RoV3/MiV1/MiV2 barely take part in hunting J-turns. H2 scores them anyway, so P4
  is what makes H2's turn index readable. P4 is ours.

**Checks (measured):**
- 0 duplicate pairs within the bridge.
- 0 self-loops.
- **0 collisions with the 24,921,936 measured pairs.** `Network.from_pairs` would sum a colliding pair. If B finds one, it
  lists it and the provenance module splits that pair's drive by count ratio. Colliding sources are never dropped.
- Every endpoint is in the graph.
- 0 cells added.

## 6. The unit U, the gains and the expected arithmetic

**U.** U = ceil((v_th − v_rest) / (w_syn · r_max · τ_syn)) = ceil(7 / (0.275 × 0.150 × 5)) = ceil(33.94) = **34**, from
`LIFParams()` and `RETINA.r_max_hz`. That is G1-gate s.5.1's steady-state formula: U synapse-equivalents at 150 Hz hold a
cell at threshold.
- **Every target at gain 1 gets 1 U** (E1: 1.06 U; P1: about 8.6 U, the deliberate density exception).
- **A bridge edge's count is base × g_class.** Gains are integers, because the kernel takes int32 counts × w_syn.

**Gains.**

| gain | class | grid | bounds |
|---|---|---|---|
| g_a | a | 1, 2, 4, 8, 16, 32 | [1, 32] |
| g_c | c | 1, 2, 4, 8, 16, 32 | [1, 32] |

- **The floor of 1 means no class can be switched off by tuning.**
- The ceiling of 32 < U means no single base-1 edge can hold its target at threshold alone at the eye's full rate.

**Expected arithmetic (mean field, steady state; not simulated).**
- **Eye rates are estimated from G1's measured numbers.**
  - At the loom's end, 1,026 of 1,045 kept contralateral cells were at ≥ 75 Hz, so nearly the whole lobe fires.
  - A prey drove 31,710-46,586 tectal spikes per trial in G1's kept set.
  - With B1's σ = 15 deg Gaussian map, a 3-deg dot drives cells within about 15 deg near full rate and cells about 30 deg
    away at roughly 20 Hz.
- **E1 on a prey:** the nearest grid sources (about 11-13 deg away) give Σr ≈ 280-330 Hz, so D ≈ 1.9-2.2 × g_c against
  34. The M-system is **not** expected to fire on a prey below g_c ≈ 16.
- **E1 on a loom:** threshold needs about 34/g_c grid sources at full rate.
  - At g_c = 8: about 4 sources, at a disc of roughly 55-70 deg (compare Dunn's 72 deg).
  - At g_c = 1-2: only near the loom's end, when the lobe saturates.
  - A 12-deg disc cannot cover 4 grid points (30 x 23 deg spacing), so the ≥ 12 deg timing rule is not expected to bind.
  - The two directly driven O-a0 cells with Mauthner contacts cannot fire it at loom onset either.
    - 17057069410's column field is at theta −7.5, elev −41.9, and 16670127295's at theta 48.5, elev −58.5. Both are more
      than 40 deg from the loom centre (±90, 0).
    - 2 synapses at 150 Hz give 2 × 0.275 × 0.15 × 5 = 0.41 mV at steady state, against 7 mV.
- **P1 on a prey:** about 20 grid sources within 15 deg and about 60 within 30 deg give D ≈ 33 × g_c. The APN is
  expected to fire from g_c ≈ 1-2. Then P2/P4 targets need an APN rate ≥ about 300/g_c Hz (P2, one active relay) or
  ≥ 150/g_c Hz (P4).
- **Expected feasible window: g_c = 2-8.**
  - Above about 16, prey-evoked escapes become likely.
  - Looms are expected to cause strikes (cross-talk): the P1 zone is covered once the disc passes about 60 deg at ±90 deg,
    and at onset for held-out ±45 deg looms.
- **Class (a) (expected):** columns in the lower field are reached only late in a loom. At the loom's end the eligible
  fragments could deliver up to 34 (left) or 28 (right) measured excitatory synapses to the **wrong** M-system, plus 2-3
  from the directly driven O-a0 tectal cells. That is about 1 U at full rate. The side rule then depends on E1 winning the
  race.

## 7. Tuning (bridge gains only)

- **What moves:** g_a and g_c only. Measured weights, LIF parameters, the eye model, every list and every threshold are
  fixed.
- **Stimuli: the G1b reopen-gate training set only.**
  - Loom: l/v 240 ms at ±90 deg.
  - Prey: 30 deg/s at ±30 deg.
  - Controls and held-out stimuli never enter tuning.
- **Seeds:** `G1c-tune-0` to `G1c-tune-7`, distinct from the gate seeds `G1-seed-0..7`.
- **Search:** the full 6 × 6 grid. Every point is evaluated and logged with ISO time, gains, every training metric and the
  network digest.
- **Objective (lexicographic):**
  1. Maximise the number of the 10 training checks passed. Per stimulus side:
     - escape P ≥ 0.5;
     - initiator on the stimulus side in ≥ 75% of escapes;
     - first-spike disc angle ≥ 12 deg in ≥ 75% of escapes (the window enforces ≤ 100 ms after expansion ends);
     - strike rule in ≥ 50% of prey trials;
     - turn index toward the prey in ≥ 75% of prey trials.
  2. On a tie, fewest cross-talk events on the training stimuli (strikes on loom trials plus escapes on prey trials).
  3. On a tie, the smallest g_c, then the smallest g_a.
- **Why side is in the objective:**
  - The brief pre-registers that when the crossing is built, flee direction comes from the published pathway model.
  - A side-blind objective with a smallest-bridge tie-break would pick the uncrossed measured path (a) by construction, and
    the fish would flee toward the threat. That is not what "no tuning to hide" means.
  - The G1b reopen gate's side-blind T1 served a different question (whether the wiring carries laterality). It is not
    among the items this spec takes.
- **What keeps it honest:**
  - the gain floor of 1;
  - separate tuning seeds;
  - the held-out hash check: every rendered stimulus's parameter dict is logged by sha256, and the gate asserts that no
    held-out or control hash is in the tuning log;
  - the per-class ablations are published.

## 8. Stimuli (full parameters in the JSON)

**Shared parameters.**
- **Loom:** dark disc 4 -> 140 deg, elevation 0, 500 ms pre-blank, 500 ms hold. Expansion lasts l/v × (cot 2° − cot 70°):
  3,392.7 ms (l/v 120), 6,785.4 ms (240) and 13,570.7 ms (480).
- **Prey:** 3-deg bright dot, elevation 0, 20-deg sweep, 3,000 ms, 500 ms pre. Mirrored on the left, as `brain.stim_prey`.

**Sets.**
- **Training:** loom l/v 240 at ±90; prey 30 deg/s at ±30.
- **Held-out (gate):**
  - loom l/v 120 and 480 at ±45 and ±135 (8 conditions);
  - prey 15 and 60 deg/s at ±15 and ±45 (8 conditions).
- **Controls:** G1's receding stimulus at ±90 (l/v 240; the 1,000 ms static onset window is reported, not scored), G1's
  matched dimming, and blank (3,500 ms).

**Windows.**
- Loom: [500, 500 + expansion + 100) ms.
- Prey and blank: [500, 3,500).
- Receding motion: [1,500, 1,500 + 6,785.4 + 100).
- Dimming: [500, 500 + 6,785.4 + 100).

**Gate seeds:** `G1-seed-0..7`, paired across conditions.

## 9. Readouts that decide

- **Escape:** `readouts.escape_decision` over the six task_criteria M-system cells, and nothing else (checked in the code).
  Any spike is an escape. The initiator is the earliest first spike, and the fish escapes away from the initiator's side.
  So under L2 the lesioned side initiates 0 escapes by construction.
- **Strike:** `readouts.strike_decision` on the strict nIII_dorsal and nMLF pools. The caller passes one spike count per
  `readout_sets` member. Members not in the graph pass 0: they stay silent and stay in the denominator. The nMLF side index
  is reported, not scored.
- **Turn:** `readouts.turn_index` over strict RoV3 + MiV1 + MiV2. An undefined TI counts as not toward the prey.
- **Headline cells:**
  - the initiating M-system cell of each escape;
  - every nIII_dorsal and nMLF cell that spiked (strike);
  - every turning SPN that spiked (turn).

## 10. Gate thresholds (G1b reopen gate, verbatim in the JSON; operationalised here)

| item | threshold (from the reopen gate) | how scored in G1c |
|---|---|---|
| H1 | P(escape) ≥ 0.5 per side; first M-system cell on the stimulus side in ≥ 75% of escapes; disc ≥ 12 deg; ≤ 100 ms after expansion ends | per stimulus side, pooled over its 4 held-out loom conditions (32 trials); every per-condition value published |
| H2 | strike rule in ≥ 50% of trials; turn index toward the prey in ≥ 75% of seeds; nMLF side reported, not scored | per prey side, pooled over its 4 held-out prey conditions (32 trials); per-condition published |
| C1 | receding, per side: mean Mauthner spikes ≤ 0.5 × the loom's, and P(escape) lower | motion window vs the same-side training loom (±90, l/v 240) run with the **gate** seeds as a control baseline (not tuning) |
| C2 | dimming: the same test against the mean of both looms; say whether the eye zeroes the tectal input | G1 measured 0 tectal rate for dimming, so expected "passed by the eye, not by the brain" |
| C3 | blank: no escape and no strike in ≥ 7 of 8 seeds; no single turn direction in ≥ 75% | expected "passed by construction (no background input)"; labelled so |
| L1 | bilateral M-system lesion: short-latency escapes (first readout spike within 20 ms of the intact median escape time) fall to 0; the intact fish escapes in the same trials | held-out looms, 8 gate seeds, paired. It passes by construction (the escape readout is the lesioned set). Informative row beside it: trials where any remaining deciding readout spikes in that 20 ms window |
| L2 | unilateral lesion: only escapes to the side opposite the lesion are removed; 'union' lesion set is a sensitivity row | left and right lesions separately on the held-out looms: the lesioned side initiates 0 escapes (by construction), and looms on the non-lesioned side keep P(escape) ≥ 0.5 with the side rule ≥ 75% (H1's own thresholds, not new ones) |

**Verdict.**
- **PASS:** every item above passes.
- **KILL (the brief's only kill):** not (H1 and H2 and C1 and C2 and C3). That means no correct escape or strike on
  held-out stimuli with the controls holding, even with the bridge.
- **FAIL, not a kill:** H and C pass but L1 or L2 fails. It is published, and the page may not say the gate passed.

## 11. Provenance labelling

- **Every bridge edge carries:**
  - `cls` (a or c; b appears only in the sensitivity row);
  - `pathway` (A1, E1, P1, P2, P3 or P4);
  - `base` count and a citation key.
- **Other labels:** measured pairs are `measured`, and the eye's Poisson input is `chosen (eye model)`.
- **The network digest folds in** `bridge_version` (`g1c-bridge-v1`), `edge_list_sha256` and the gains. So it differs from
  G1's digest and from every other gain setting.
- **The JS kernel** runs the bridge as plain extra (pre, post, count) pairs.
- **Headline:** the brief's decision-path share, per readout, per condition, with drive split into measured, bridge (a),
  bridge (c) and eye (chosen).
- **Secondary:** the synapse-count share, reported twice.
  1. Whole brain of record plus bridge. This is the network the replay pins. The bridge's synapse-equivalents are
     646·g_a + 7,397·g_c; at g_a = g_c = 8 that is 64,344 against 29,474,316 measured synapses, so the measured share is
     99.8% (expected).
  2. The exact reduced network the gate simulates. Expected to be much lower, because the reduction keeps only cells that
     can fire and reach a readout.

## 12. Mandatory publications

1. **M1, measured-wiring shuffle with the bridge fixed.**
   - Shuffle the **measured** pairs only, degree-preserving (`sim.shuffled_copy`'s algorithm, as G1).
   - Re-add the bridge edges unchanged by segment id. Gains stay at the tuned values, and the stimulated set is unchanged.
   - 10 shuffles, seeds `G1c-measured-shuffle-0..9`. At least 5 if machine time forces it, and that is disclosed.
   - Run the H1 and H2 held-out conditions with gate seeds 0-3.
   - **Rule, fixed now:** if the shuffled fish's H1 and H2 pass/fail outcomes equal the intact fish's in ≥ 8 of 10
     shuffles, the page says *"the anatomy is real; the decision runs through the model"*. Per-shuffle metrics are
     published either way.
2. **M2, bridge ablation.** On the H1/H2 held-out conditions with 8 gate seeds:
   - (i) all bridge edges removed;
   - (ii) class (a) removed;
   - (iii) class (c) removed;
   - (iv) E1 removed and (v) P1-P4 removed, as rows.
3. **M3, both metrics per class,** for the intact gate run and every ablation.
4. **M4, also published:**
   - the cross-talk table (strikes on loom trials and escapes on prey trials, per condition; reported, not scored);
   - the tuning log;
   - the held-out hash check;
   - the disclosure decision.

## 13. Expected results (not measured)

- **Decision-path share: near 0 for escape, strike and turn.**
  - Every route that reaches a deciding cell crosses a bridge edge, and the metric gives a bridged link m = 0.
  - Measured-only routes carry walk weights of 1e-7 to 1e-5 (G1c-measure s.4).
  - This holds for any bridge that lets the fish act. It follows from the metric's definition, not from this design.
- **The disclosure sentence applies,** because the crossing (E1) is built: *"Which way the fish flees is set by the
  published pathway model, not by the measured wiring."*
- **Proposed parallel line (NOT pre-registered; the page owner decides):** *"Whether and which way the fish strikes is set
  by the published pathway model (Antinucci 2019), not by the measured wiring."*
- **M1 is expected to find behaviour unchanged** by the shuffle, so the page line in M1 is expected to apply.
- **Cross-talk is expected:** looms will cause strikes. That is not a gate item, but it matters wherever a strike is read as a decision.

## 14. Choices (for the honesty page)

1. The stimulated set drops G1's direct-contact removal (s.2).
2. Class (a) eligibility and column: ≥ 1 tectal input synapse, not a B1 tectal cell, 34 nearest somas (s.3).
3. Class (b) is not built (s.4).
4. **E1 is a functional crossed edge set** from a 6 × 6 lobe grid onto all three contralateral M-system cells. No relay,
   because none is identified (s.1).
5. **The APN relay is 1 real soma per side,** nearest AF7 in the AF7-pretectum labels. The tectum -> APN link is
   ipsilateral by assumption.
6. P1 frontal zone ≤ 60 deg on a 6-deg grid; nIII lateral half (Dowell); P2/P3 bilateral, P4 contralateral (Antinucci).
7. **U = 34 normalisation for every stage,** with P1 deliberately dense. Integer gains in 1-32 on powers of two.
8. **The tuning objective includes side,** on training stimuli only (s.7). Tie-breaks are cross-talk, then the smallest
   bridge.
9. The gate operationalisations in s.10: pooling per side, C1/C2 baselines, the L1 informative row, and L2 reusing H1's
   thresholds.

## 15. Limits

- **The arithmetic in s.6 is mean-field and expected.** A real run may need a gain outside 2-8. If no grid point passes
  the training checks, the gate still runs at the objective's choice, and the kill decides.
- **Helmbrecht 2018's full text was not readable (403).** Its tract details beyond the abstract are secondary. Dunn 2016
  and Lacoste 2015 were read through a summarising fetch.
- **The crossing relay is unidentified in the literature.** E1 states a function, not an anatomy.
- **The APN relay's link from the tectum is functional** (ablation), not anatomical.
- **The eye model has no feature channels** (loom and prey RGC layers, UV acute zone). Size selectivity comes only from
  pooling density.
- **B1's map is rank-uniform.** Theta and elevation are model placements, so the frontal zone and the grids inherit them.
- **The M-system homolog picks inherit G1b's confidence labels.** MiD3cm-left is contested. L2's union set is the
  sensitivity row.
- **Left/right rests on B1's axis call.** A mirror error flips every side consistently, and crossed stays crossed.

## Sources

Listed in s.1 with URLs, DOIs and how each was read (all 2026-09-26). Brain of record and identities as in G1-pull.md,
G1-cells.md, G1b-readouts.md and G1c-measure.md. Fish1 atlas layers mece2_231218 (cached) and mece3_231218 (public, read
2026-09-26). Citation required by the Fish1 data policy: Petkova, Januszewski et al. (2025), "A connectomic resource for
neural cataloguing and circuit dissection of the larval zebrafish brain", bioRxiv.

## Appendix A. Key IDs (full lists in bridge_spec.json)

- **M-system (task_criteria):**
  - Left: Mauthner 15773771512, MiD2cm 17405827637, MiD3cm 17365353494.
  - Right: Mauthner 16304202596, MiD2cm 16039136570, MiD3cm 17610066079.
- **APN relay:** left 14241661373, right 14833177305.
- **Class (a) fragments (19):**
  - Left tectal end: 11203134117, 15629698215, 18648533550, 20219074049, 21688132282, 21688200235, 21728808901,
    21729058924, 21789748483.
  - Right tectal end: 18036992749, 18036999791, 19525883657, 19607736501, 21096682251, 21178473331, 22545347363,
    22606097961, 24258372909, 25341829035.

## Appendix B. Derivation (how the lists were computed; task B re-derives independently)

Run from `launches/fishbrain/brain` with `PYTHONPATH=. .venv/bin/python derive.py <out.json> <mece3_4096.npy>`, then
assemble. Only reads; 7.3 s. Steps:
1. `brain.load_pairs()`, `brain.identified()` and `cells.load()`. Stimulated set = tectal ids present in `ids` minus
   `exclusions`. Compared with `brain.stimulated_set` to list the re-admitted cells.
2. `cells.tectum_map` gives each stimulated cell's (u, v, theta, elev, pos).
   - E1: nearest cell in (u, v) to each of the 6 × 6 grid points, per lobe (ties to the lower id; duplicates merged).
   - P1: nearest cell in (theta, elev) to each point of the 6-deg frontal grid.
3. Targets from `readouts.json` (strict): the M-system task_criteria; nIII_dorsal lateral half by `lat_um` ≥ the side
   median; nMLF; RoV3 + MiV1 + MiV2. Each kept only if its segment is in `ids`, and asserted equal to `readout_sets`.
4. APN relay:
   - CAVE `pt_position` (8 × 8 × 30 nm voxels), atlas voxel = floor(x/512, y/512, z/128) into mece2.
   - Candidates are labels 14 or 4.
   - Side from B1's midline; distance to the same-side AF7 voxel centres (mece3 = 3).
   - The nearest soma whose lore id maps to an in-graph segment holding exactly one soma.
5. Class (a): the eligibility rule of s.3 on `oa_fragments.json`. Column = 34 nearest stimulated somas (3-D um) to
   `tectal_end_8nm` in `tectal_end_side`.
6. Edges are built, then checked for internal duplicates, self-loops, endpoints in `ids`, and collisions with measured
   (pre, post) keys. `edge_list_sha256` is taken over the sorted canonical tuples.
