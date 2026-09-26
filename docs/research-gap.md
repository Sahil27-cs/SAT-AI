# SAT-AI — Research gap analysis

**Status: COMPLETE (primary corpus).** Phase 0, 2026-09-22.
Based on the **seven papers provided**. A supplementary structured search (§5) is
still outstanding and **no claim below is stated as novel until it is done** —
the verdicts here are "supported as a gap by the provided corpus", which is a
weaker and more honest statement.

Tags used throughout: **`[provided]`** — from the supplied corpus ·
**`[additional]`** — supplementary search · **`[ours]`** — this project.

---

## 1. Corpus

| # | Citation | Type |
|---|---|---|
| **P1** | Shang, Y., Cheng, B., Zhang, Z., Wang, Q., Jin, P., Huang, L., Liu, C., Huang, L., Ding, X., Shen, T., Tang, B.-H., Gu, Y. (2026). *Artificial Intelligence for Remote Sensing: Progress, Challenges, and Perspectives.* IEEE JSTARS, 19, 6840–6866. | Review |
| **P2** | Aziz, M.A., Hassan, F.M., Patil, R. (2025). *Integration of ChatGPT and GeoAI in Change Detection of Landcover on Landsat Images 2000–2020: A Critical & Empirical Review.* Bulletin de la Société de Géographie d'Égypte, 98(1), 25–53. | Review + case study |
| **P3** | Kemarau, R.A., Muhamad, N., Shamsuddin, A.S., Sa'adi, Z., Sakawi, Z., Eboy, O.V., Md Nor, N.N.F. (2026). *Floods From Space: How Remote Sensing, AI, and Cloud Platforms Are Reshaping Disaster Risk Reduction.* Journal of Flood Risk Management, 2026(3), e70214. | PRISMA review, 176 studies |
| **P4** | Jones, A., Kuehnert, J., Fraccaro, P., Meuriot, O., Ishikawa, T., Edwards, B., Stoyanov, N., Remy, S.L., Weldemariam, K., Assefa, S. (2023). *AI for climate impacts: applications in flood risk.* npj Climate and Atmospheric Science, 6:63. | Perspective (IBM Research) |
| **P5** | Akhyar, A., Zulkifley, M.A., Lee, J., Song, T., Han, J., Cho, C., Hyun, S., Son, Y., Hong, B.-W. (2024). *Deep artificial intelligence applications for natural disaster management systems: A methodological review.* Ecological Indicators, 163, 112067. | Methodological review |
| **P6** | Swain, K.P., Nayak, S.R., Pattanaik, T.P., Singh, A. (2025). *An Explainable Multi-Agent Framework for Real-Time Tree Detection and Canopy Segmentation in Remote-Sensed Imagery.* JASTT, 6(SI), 63–75. | Empirical |
| **P7** | Kundu, S., Ninoria, S.Z., Chaturvedi, R.P., Mishra, A., Agrawal, A., Batra, R., Dubale, M., Hashmi, A. (2025). *Real-time deforestation anomaly detection using YOLO and LangChain agents for sustainable environmental monitoring.* Scientific Reports, 15:39961. | Empirical |

## 2. Extraction table

| # | Hazard / domain | Data | Method | **Evaluation protocol** | Reported result | Stated limitations | Bearing on SAT-AI |
|---|---|---|---|---|---|---|---|
| P1 | RS broadly; disaster monitoring | multimodal EO | Survey of single→multimodal multitask models; defines **"RS AI agent"** (autonomous tool use + multistep planning); surveys RS-Agent, Change-Agent, Tree-GPT, RS ChatGPT, MineAgent, GeoLLM-Squad | n/a (review); **critiques the field's evaluation practice** | n/a | Annotation scarcity; poor physical interpretability; **evaluation frameworks "fail to adequately address generalization across scales, regions, and temporal dimensions"** | Establishes that LLM-over-RS agents are an existing class, **and names the missing metrics** |
| P2 | Land-cover change | Landsat 2000–2020, Halayeb–Shalateen, Egypt | GeoAI supervised classification; **ChatGPT generates Python for ArcGIS** | Case study; no accuracy assessment of the LLM-generated output reported | Qualitative | Notes ChatGPT bias/limits generally | LLM used as **code generator**, not as an interpreter of numeric model output; no faithfulness evaluation |
| P3 | **Flood** | SAR, optical, precipitation, DEM, altimetry | PRISMA synthesis of 176 studies (2001–2024) | n/a (review) | **SAR-anchored fusion (SAR + precipitation + DEM) is the dominant architecture**; marked post-2018 rise in multi-sensor work | **Persistent:** optical cloud gaps; **radar saturation in urban/vegetated areas**; spatial–temporal resolution trade-offs; cross-country data access | **The single most important paper for SAT-AI.** Confirms the architecture is standard; names urban SAR as "a critical frontier" |
| P4 | Flood risk / climate impact | Sentinel-1, ERA5, soil, land use, DEM | ResNet-UNet flood detection; GP-Bayesian-optimisation UQ; ML surrogate flood maps | Validated against Copernicus EMS published extents | **0.98 / 0.99 accuracy** flood-water detection; ML map over-predicts least of three models | UQ a "grand challenge"; local ground truth scarce; ML black-box | Validates ResNet-UNet choice. **Explicitly:** AI flood extents "can be used as 'ground truth' data to derive ML flood risk models" |
| P5 | Flood, forest fire, earthquake damage | Multispectral, Sentinel-2, aerial | Review of SegNet, U-Net, FCN, FCDenseNet, PSPNet, HRNet, DeepLab for disaster segmentation | Per-study; heterogeneous | Active-fire IoU 0.73–0.99 across architectures | Spatial-information loss in CNNs; data availability | Benchmark context for Phase 5; confirms encoder–decoder as standard |
| P6 | Tree/canopy (vegetation) | Kaggle RGB satellite, 2,400 images, 0.5–1.5 m | YOLO11m + SAM2, zero-shot; vegetation/edge/colour **"agents"** fused by IoU; Grad-CAM + SHAP + LIME | Kaggle split; **no spatial holdout described; baseline comparison explicitly declined as "out of scope"** | 97.3 % acc, 0.92 IoU, <10 s/image | Dataset-specific | "Agents" are **CV modules, not LLM agents**. XAI is for "user trust", not evaluated |
| P7 | Deforestation | 3,133 annotated satellite/drone images | YOLOv8 + **LangChain agents** for threshold adjustment, RL feedback, GIS reporting | 73/21/**6 %** split (201 test images) | **mAP50 ≈ 0.07**, precision ≈ 0.1, recall ≈ 0.08; agent layer claimed to raise recall ~24 % rel. | No night/low-light evaluation | Closest architectural analogue to SAT-AI's agent plane — **and the clearest illustration of the evaluation gap** |

## 3. Thematic synthesis

**Established by the corpus — SAT-AI may not claim these as contributions:**

1. **SAR-anchored multi-sensor fusion is the dominant flood architecture.** P3, from 176 studies, finds SAR + precipitation + DEM to be the most common combination, with a marked post-2018 rise. SAT-AI's Phase 3–5 design is therefore *mainstream practice*, not innovation. This is good — it means the design is defensible — but it retires "multimodal fusion" as a headline contribution.
2. **Encoder–decoder segmentation of Sentinel-1 for flood extent is established.** P4 uses ResNet-UNet; P5 reviews the architecture family. SAT-AI's Phase 5 choice is validated and unoriginal.
3. **Deriving a flood-risk model from AI-generated flood extents is an established technique.** P4 states it directly. This *supports* SAT-AI's Phase 6 inventory design and removes any novelty claim for it. The circularity noted in `limitations.md` stands, but SAT-AI is now in published company.
4. **Cloud platforms (GEE, Sentinel Hub) are the recognised enabling compute layer.** P3. Validates ADR-002.
5. **LLM-driven agents over RS tools already exist as a class.** P1 names six. "An agentic AI interface for geospatial disaster analysis" is therefore **not** a contribution.
6. **Grad-CAM / SHAP / LIME on RS models is established.** P6; P4 notes XAI "beginning to be applied" in climate science.

**Named as open, urgent, or unresolved by the corpus:**

1. **Evaluation of RS AI agents.** P1 §VI.D is explicit: *"For agents performing complex multistep workflows, 'reasoning-consistency' and tool-invocation accuracy must be integrated into benchmarks to ensure logical reliability"*, alongside prompt-robustness and physical-consistency validation. It further states that *"the paradox of opaque reasoning versus trustworthy decision making limits their adoption in high-risk applications like territorial planning and emergency management."*
2. **Urban SAR saturation.** P3: *"urban complexity is a critical frontier in flood remote sensing, requiring dedicated methodological innovation"*, with an explicit call for "urban-specific algorithms".
3. **Cross-region / cross-temporal generalisation is under-evaluated.** P1 §VI.D.
4. **Susceptibility maps are static.** P4: ML hazard-susceptibility maps *"are usually derived for average climate conditions"* — i.e. not conditioned on the event state.
5. **Uncertainty quantification** is a "grand challenge" (P4).
6. **Local ground truth is scarce**, pushing the field toward transfer learning (P4) — with the transfer gap itself largely unquantified.

**A methodological observation about the corpus — itself a finding:**

Of the three papers that place a language model over remote-sensing outputs (P2, P6, P7), **none evaluates whether the generated language is faithful to the underlying model outputs.** P7 reports an agent layer as a success while its underlying detector scores mAP50 ≈ 0.07 with precision ≈ 0.1 — a detector that barely functions, wrapped in a reasoning layer whose output correctness is never tested. P4 reports 0.98–0.99 *accuracy* for flood-water segmentation, a metric that is close to uninformative on a task where water is typically 2–15 % of pixels. P6 reports 97.3 % accuracy with no spatial holdout and declines baseline comparison as out of scope.

**None of the seven reports a region-disjoint or spatially-blocked evaluation protocol.**

That turns out to run deeper than the corpus. Phase 2 verification
(`scripts/verify_sen1floods11.py`, 2026-09-22) established that **Sen1Floods11's
own official splits are not region-disjoint either**: every event region except
Bolivia appears in train, validation and test at ~60/20/20, so 512×512 chips from
the same flood — same acquisition, same orbit, adjacent ground — sit on both
sides of the train/test boundary.

The implication reaches every published Sen1Floods11 benchmark, including the
Prithvi-EO fine-tunes noted in the Phase 1 plan: those figures are
interpolation-within-event numbers carrying spatial autocorrelation, not
generalisation-to-new-event numbers. They are widely quoted as if they were the
latter.

SAT-AI therefore builds leave-one-region-out splits of its own (**ADR-009**) and
reports *both* protocols, so that `IoU_official − IoU_LORO` — the inflation the
standard protocol carries on this dataset — becomes a reported quantity rather
than an unexamined assumption. As far as this corpus shows, nobody reports it.
Metrics from P4, P6 and P7 are **not directly comparable** to SAT-AI's LORO
numbers, and the results chapter must say so before it says anything else.

## 4. Gap statement

> Across the provided corpus, the components SAT-AI assembles are individually
> established: SAR-anchored fusion of radar, precipitation and terrain is the
> dominant flood-mapping architecture **[P3]**; encoder–decoder segmentation of
> Sentinel-1 is standard and performs well **[P4, P5]**; deriving susceptibility
> labels from AI-generated flood extents is published practice **[P4]**; and
> LLM-driven agents over remote-sensing tools already form a recognised class
> **[P1]**. What is absent is **measurement of the language layer and of
> generalisation**. P1 states that reasoning-consistency and tool-invocation
> accuracy "must be integrated into benchmarks" for RS AI agents, and that opaque
> reasoning limits their adoption "in high-risk applications like … emergency
> management"; yet none of the corpus papers that place a language model over
> remote-sensing outputs **[P2, P6, P7]** measures whether the generated text is
> faithful to those outputs. In parallel, P3 identifies urban SAR saturation as "a
> critical frontier … requiring dedicated methodological innovation", P1 finds
> existing evaluation frameworks fail to address "generalization across scales,
> regions, and temporal dimensions", and no paper in the corpus reports a
> region-disjoint evaluation.
>
> **SAT-AI therefore locates its contribution in measurement rather than
> architecture.** It adopts the established architecture deliberately and without
> novelty claim, and reports three things the corpus does not: a grounding-violation
> rate for an agent layer over Earth-observation model outputs, a degradation curve
> under systematically induced modality loss, and a quantified rural→urban transfer
> gap for SAR flood segmentation in an Indian setting.

## 5. Verdicts on the candidate contributions

| # | Candidate (as stated in Phase 1) | Verdict | Reasoning |
|---|---|---|---|
| 1 | Quantified graceful degradation under missing modalities | **SURVIVES — narrowed** | Fusion itself is dominant **[P3]** and cannot be claimed. But no corpus paper evaluates fusion under *systematically induced* modality loss, and P3 names cloud gaps and resolution trade-offs — precisely the missing-modality regime — as persistent. Framed as **extension**, not novelty, pending §6. |
| 2 | Measured grounding-violation rate for an LLM over EO outputs | **SURVIVES — strongest** | Directly called for by **P1 §VI.D**. Absent from **P2, P6, P7**, which are the corpus papers that would need it. This is now SAT-AI's primary contribution. |
| 3 | Transparent, sensitivity-analysed multi-hazard risk with justified coupling | **DOWNGRADED** | P4 treats compound events as "rare and difficult to model", supporting the need — but multi-hazard risk frameworks are widespread outside this corpus and the corpus cannot establish novelty either way. Reclassified as a **reproducibility / engineering contribution**: a fully documented, configurable, sensitivity-analysed implementation. Novelty claim requires §6. |
| 4 | Modality-attribution XAI | **MERGED into #1** | Not a standalone contribution. It is the *instrument* that produces #1's degradation attribution. Retained as method, removed as claim. |
| 5 | India-specific evaluation of globally-trained flood models | **SURVIVES — strengthened** | **P3** names urban complexity as a critical frontier needing dedicated innovation; **P1** names cross-region generalisation as under-evaluated; **P4** names scarce local ground truth as the driver of transfer learning. The Mumbai transfer experiment (ADR-007) now addresses an explicitly named gap. |

**Revised contribution set:**

- **C1 — Measured grounding.** A reported grounding-violation rate, tool-invocation accuracy and refusal correctness for an agent layer over EO model outputs, on a fixed query benchmark. *Supported as a gap by P1; absent from P2, P6, P7.*
- **C2 — Quantified degradation under modality loss**, with modality-attribution XAI as the instrument. *Extension of the fusion practice established by P3.*
- **C3 — Quantified rural→urban domain-transfer gap** for SAR flood segmentation in an Indian setting, under region-disjoint evaluation. *Addresses the frontier named by P3 and the evaluation gap named by P1.*
- **C4 — Reproducible, sensitivity-analysed multi-hazard risk implementation.** Engineering/reproducibility contribution, not a novelty claim.

## 6. Supplementary search log

**Outstanding.** Required before any of C1–C3 is described as novel in the paper
or the viva. Target databases: Scopus, IEEE Xplore, Web of Science; window
2022–2026.

| Date | Query | Source | Hits | Screened | Included |
|---|---|---|---|---|---|
| | `(LLM OR "large language model" OR agent) AND ("remote sensing" OR "earth observation") AND (hallucination OR grounding OR faithfulness OR "tool use") AND evaluat*` | | | | |
| | `("missing modality" OR "modality dropout" OR "sensor dropout") AND (flood OR "remote sensing") AND fusion` | | | | |
| | `("domain adaptation" OR "transfer" OR generalization) AND SAR AND flood AND urban` | | | | |
| | `"multi-hazard" AND risk AND (sensitivity analysis) AND (exposure AND vulnerability)` | | | | |

## 7. What changes in the project as a result

1. **README contribution list rewritten** to C1–C4 above. The word "novel" appears nowhere until §6 is done.
2. **C1 is promoted to primary**, so the Phase 11 grounding benchmark is no longer an optional safety feature — it is the headline experiment and must be budgeted as such (a fixed, versioned query set with expected tool traces, built alongside the agents, not after).
3. **C3 is promoted**, which raises the Mumbai AOI from "nice contrast" to a required experiment. Phase 2 must confirm Mumbai has enough Sentinel-1 coverage and at least one reference extent to evaluate against.
4. **Phase 6 gains a citation** (P4) legitimising the extent→inventory→susceptibility chain, and **P4's finding that susceptibility maps are usually static** becomes the justification for SAT-AI's rainfall-modulated operational risk.
5. **Comparability caveat added to the results chapter:** metrics from P4, P6, P7 are not directly comparable to SAT-AI's, because none reports a region-disjoint protocol and P4/P6 report accuracy on heavily imbalanced tasks.
6. **A fifth candidate contribution emerged from Phase 2 verification** — the measured gap between Sen1Floods11's official-split and leave-one-region-out performance. It is cheap (11 extra training folds on a small dataset), it quantifies something the whole sub-field quotes without checking, and it strengthens C3 rather than competing with it. Folded into **C2/C3** as a reported quantity rather than listed as a separate claim, pending §6.
7. **India is now a LORO fold, not a training region** (ADR-009). The 68 Indian chips are tested on, never trained on, which is exactly what C3 requires and removes a leakage path that the original plan would have walked into.

---

## Completion checklist

- [x] Papers received and read (7/7)
- [x] Extraction table complete
- [ ] Supplementary search run and logged — **§6, outstanding**
- [x] Each candidate contribution given a verdict
- [x] Gap statement written
- [ ] Contribution claims in `README.md` and the abstract updated to match
- [x] Every claim tagged `[provided]` / `[additional]` / `[ours]`
