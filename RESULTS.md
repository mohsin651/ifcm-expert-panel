# Experiment results ledger — 25 October 2026

**Read this before quoting a number.** The primary section is the frozen A/B1/B2 comparison; the later source-uncertainty analysis is **post-hoc and exploratory**, not a new primary test. AUROC is area under the ROC curve; accuracy is the fraction correctly classified; Brier score is a probability error (smaller is better). Values below are rounded for reading; the linked CSV/JSON files retain full precision. The primary results were not changed by this analysis.

## Current leakage-aware comparison

The LIDC outer test has 588 nodules from 195 patients, but only **462 clear-label nodules** enter diagnosis metrics. Derm7pt has 395 test cases for melanoma AUROC and 355 eligible cases for four-class accuracy; there are no patient IDs for patient-level resampling. A, B1, and B2 use the same saved inputs and test cases. LIDC confidence intervals resample patients; Derm confidence intervals resample cases. Each interval below uses 5,000 paired bootstrap resamples.

| Dataset / model | Diagnosis AUROC ↑ | Accuracy ↑ | Brier ↓ | Wrong predictions | Perception / knowledge errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| LIDC A | 0.8556 | 0.8095 | 0.1397 | 88 | 32 / 56 |
| LIDC B1 | **0.8657** | 0.8074 | **0.1384** | 89 | 33 / 56 |
| LIDC B2 | 0.7729 | 0.6580 | 0.2185 | 158 | 15 / 143 |
| Derm A | **0.8220** | **0.7380** | **0.1462** | 93 | 37 / 56 |
| Derm B1 | 0.8017 | 0.7239 | 0.1552 | 98 | 49 / 49 |
| Derm B2 | 0.7753 | 0.6169 | 0.1865 | 136 | 15 / 121 |

Paired AUROC differences (left minus right; 95% CI):

| Dataset | A − B1 | A − B2 | B1 − B2 |
| --- | --- | --- | --- |
| LIDC | −0.0101 [−0.0220, +0.0021] | +0.0827 [+0.0557, +0.1128] | +0.0928 [+0.0607, +0.1268] |
| Derm | +0.0202 [+0.0066, +0.0348] | +0.0466 [+0.0237, +0.0717] | +0.0264 [+0.0091, +0.0450] |

**Interpretation:** B1's LIDC AUROC is numerically higher than A's, but its confidence interval crosses zero, so this is *not* evidence of a reliable gain. A leads Derm. B2 is worse on both. Do not say that B1 or B2 is universally best.

The strict LIDC split protects the CNN training boundary, but the method design had prior exposure to historical pilot results; this is not a pristine, prospectively untouched research program. Derm's case-level intervals assume independent cases because patient IDs are unavailable. Perception/knowledge categories are operational counterfactual classifications, not clinician-annotated causal explanations.

Exact sources: [LIDC metrics](fresh_runs/ifs_comparison/isbi-b1-lidc-v1/evaluation_v2/metrics.csv), [Derm metrics](fresh_runs/ifs_comparison/isbi-b1-derm-monet-v1/evaluation_v2/metrics.csv), each run's `paired_comparisons.csv` and `evaluation_manifest.json`. The corrected `evaluation_v2/` files are the only active evaluation; an earlier LIDC reader denominator was wrong.

### Finding, attribution, and uncertainty analyses

| Measure | LIDC | Derm |
| --- | ---: | ---: |
| Mean finding AUROC | 0.8082 (13 evaluable findings) | 0.8318 (15 findings) |
| A-error subset | 88 errors; 32 perception | 93 errors; 37 perception |
| A plain uncertainty (`1 − confidence`) attribution AUROC | 0.7126 | 0.5367 |
| A decision-flip attribution AUROC | 0.6473 | 0.5012 |
| B1 diagnosis-hesitancy attribution AUROC, on A errors | 0.5547 | 0.6713 |
| B2 diagnosis-hesitancy attribution AUROC, on A errors | 0.5011 | 0.4493 |
| A flip minus plain, paired 95% CI | −0.0653 [−0.2034, +0.0693] | −0.0355 [−0.1904, +0.1189] |

The attribution comparison fixes **A's error set** for a fair denominator. Other rows in `attribution.csv` use each model's own errors and answer a different question. The two A flip-minus-plain intervals cross zero. On LIDC, finding ambiguity and reader disagreement have Spearman correlation **0.5701** on 305 nodules from 165 patients (patient-bootstrap 95% CI **[0.4826, 0.6458]**); this is descriptive, not causal. Derm has no equivalent reader correlation.

Sources: each run's `evaluation_v2/{finding_auroc,attribution,attribution_paired,decision_flips,reader_correlations,calibration,ifs_diagnostics,edge_quality}.csv` and `supplementary_v1/reader_clustered_ci.csv`. The files contain all per-finding, per-edge, calibration, and diagnostic values, including exact denominators.

### Ablations and edge checks

| Test | LIDC | Derm |
| --- | ---: | ---: |
| B1 with finding hesitancy removed: AUROC / accuracy | 0.8539 / 0.8030 | 0.7962 / 0.7239 |
| B2 with edge hesitancy removed | Exactly B1: 0.8657 / 0.8074 | Exactly B1: 0.8017 / 0.7239 |
| Strong fit-set edges with sign disagreement | 0 / 8 | 3 / 28 |
| Edge-uncertainty AUROC for detecting sign disagreements | Undefined (no positive events) | 0.8000 |

The edge comparison is against a **fit-set association**, not clinical truth. Removing hesitancy includes a fresh calibration fit on training data (740 clear LIDC fit rows; 616 Derm fit rows), not test fitting. Each model has four LIDC or eight Derm fit-calibration parameters; no diagnosis-label-fitted reasoning weights. Runtime was not logged in the frozen manifests; no runtime claim is supported. Sources: `supplementary_v1/{ablations,edge_detection,parameters_compute}.csv` for each run.

### Input and output inventory

- Strict LIDC CNN boundary: **1,573 crops** total; 985 outer-fit nodules (740 clear-label) and 588 outer-test nodules (462 clear-label); 195 eligible test patients; 14 finding outputs, five seeds, 50 inner and 10 final tasks. The mean finding AUROC above is measured on **13 evaluable** findings. Stored in `fresh_runs/ifs_comparison/isbi-lidc-strict-v3/cnn/`; its manifest and task outputs are the proof, not the uploaded CNN notebook archive.
- Current comparison artifacts: `fresh_runs/ifs_comparison/isbi-b1-{lidc,derm-monet}-v1/{a,b1,b2,evaluation_v2,supplementary_v1,paper_v4}/`. The `paper_v4` directories hold four figures in PDF/SVG/PNG and tables in CSV/TeX. The CSVs, not rounded values in this ledger, are authoritative.
- Ten saved LLM-panel responses are reused. Only Qwen2.5-7B and Mistral-7B were independently spot-checked against archived answers; eight others were **not** rerun. `fresh_runs/models_manifest.json` records model revisions; locally cached model weights are machine-temporary. Do not call this a full ten-LLM replication.

## Exploratory source-of-hesitation analysis

This **new post-hoc analysis** compares four fixed IFS states for each case: point findings/point edges, interval findings/point edges, point findings/interval edges, and both intervals. An order-symmetric additive decomposition assigns incremental hesitation to the finding interval and the edge interval. It scores the diagnosis selected by the already-frozen Model A, so all four states refer to the same class. The implementation is `ifcm/source_uncertainty.py`; exact per-case and aggregate outputs are at `fresh_runs/ifs_comparison/isbi-b1-{lidc,derm-monet}-v1/source_uncertainty_v1/`. It neither retrains a model nor modifies A/B1/B2 predictions.

| Measure | LIDC | Derm |
| --- | ---: | ---: |
| Mean baseline IFS hesitation (neither interval) | 0.4533 | 0.4755 |
| Mean **increment** from image-finding interval | 0.0712 | 0.0506 |
| Mean **increment** from LLM-edge interval | 0.4620 | 0.4516 |
| Mean total IFS hesitation (both intervals) | 0.9864 | 0.9776 |
| Cases with total hesitation > 0.95 | 412/462 (89.2%) | 283/355 (79.7%) |
| AUROC for detecting A errors: plain `1-confidence` | 0.7272 | 0.7160 |
| AUROC for detecting A errors: total IFS hesitation | 0.4696 | 0.5019 |
| Total-minus-plain AUROC, exploratory paired 95% CI | −0.2575 [−0.3707,−0.1536] | −0.2141 [−0.3028,−0.1285] |
| AUROC for distinguishing perception from knowledge errors on A errors: image-minus-edge increment | 0.5257 | 0.5656 |
| Same task: existing A decision-flip score | 0.6473 | 0.5012 |
| Increment-balance minus flip AUROC, exploratory paired 95% CI | −0.1217 [−0.2799,+0.0294] | +0.0644 [−0.0466,+0.1772] |

Both perception-versus-knowledge comparison intervals cross zero. These categories are operational counterfactual labels, not adjudicated causes. The edge and image increments are on **different heuristic interval scales**; the larger edge increment does not establish that LLM knowledge is intrinsically more uncertain. The near-saturation of total hesitation is consistent with its poor error ranking, but does not prove a single causal failure mechanism.

For a simple review/referral test, each score's threshold was set to its **80th percentile on the fit cases**, without test labels. On LIDC, plain confidence referred 110 test nodules and caught 42 of A's 88 errors; total IFS hesitation referred 88 and caught 16. On Derm, plain confidence referred 62 and caught 34 of 93 errors; total hesitation referred 53 and caught 24. Referral counts differ, so those counts are not a matched-coverage superiority test; the AUROCs above are the cleaner ranking comparison. Per-score thresholds, coverage, and retained error rates are in `triage.csv`.

The two paired intervals used 2,000 resamples (LIDC patients; Derm cases). The primary test outcomes were known before this idea was proposed, so **all new analyses here are exploratory even when a bootstrap interval excludes zero**. A new independent cohort would be needed for a confirmatory clinical/referral claim.

## How to reproduce without overwriting the record

Read [README.md](README.md) for exact read-only verification and new-run downstream commands. The saved upstream votes/predictions and model outputs are preserved; a raw-data, from-scratch GPU recreation is **not yet guaranteed** because raw data, local crop files, weights, and all ten LLM reruns are not fully locked into this repository. The verified results are reproducible from the saved artifacts with new run IDs. Do not tune on the held-out test set or overwrite `evaluation_v2/` and `paper_v4/`.
