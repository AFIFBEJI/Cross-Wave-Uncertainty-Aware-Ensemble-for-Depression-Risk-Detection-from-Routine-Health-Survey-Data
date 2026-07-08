# MindScreen

**Cross-Wave Uncertainty-Aware Ensemble for Depression Risk Detection from Routine Health Survey Data**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)]()
[![Dataset](https://img.shields.io/badge/data-NHANES-orange.svg)](https://wwwn.cdc.gov/nchs/nhanes/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-red.svg)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/explainability-SHAP-9cf.svg)](https://github.com/shap/shap)
[![Reproducibility](https://img.shields.io/badge/reproducible-yes-brightgreen.svg)]()

> A research project investigating whether machine learning models trained on routine health survey data can flag depression risk earlier, and whether those predictions can be made trustworthy enough for clinical use.

---

## Table of Contents

- [Overview](#overview)
- [Project Personnel](#project-personnel)
- [Motivation](#motivation)
- [Relationship to Prior Work](#relationship-to-prior-work)
- [Research Gaps Addressed](#research-gaps-addressed)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Expected Results](#expected-results)
- [Evaluation Metrics](#evaluation-metrics)
- [Repository Structure](#repository-structure)
- [Risks and Limitations](#risks-and-limitations)
- [Data and Code Availability](#data-and-code-availability)
- [References](#references)
- [Status](#status)

---

## Overview

MindScreen predicts depression risk using the **National Health and Nutrition Examination Survey (NHANES)**, a nationally representative dataset combining physical exams, lab work, and questionnaires. The project reproduces a published baseline and extends it in three directions the original study left open: generalization across time (cross-wave testing), uncertainty quantification, and a cleaner separation between biological and demographic drivers of the model's predictions.

The end goal is not just a working classifier, but a model whose predictions can be trusted — or explicitly flagged as uncertain — in a way that's useful for clinical screening rather than just benchmark performance.

## Project Personnel

* **Supervisor:**
    * **Afif Beji, Eng., M.Sc.**
        * *Role:* Assistant Professor & Project Supervisor
        * *Institution:* ESPRIT, School of Engineering
* **Intern:**
    * **Fedi Mdaini**
        * *Role:* Fourth-Year Engineering Student (Data Scientist)
        * *Institution:* ESPRIT, School of Engineering

---

## Motivation

Depression affects roughly 280 million people worldwide and remains substantially underdiagnosed. Much of the data that could help flag at-risk individuals — bloodwork, physical measurements, lifestyle questionnaires — is already collected during routine checkups but rarely integrated into a single risk signal. MindScreen explores whether that untapped data can support earlier, more targeted intervention.

## Relationship to Prior Work

This project reproduces and extends:

> Vu et al. (2025). *Prediction of depressive disorder using machine learning approaches: findings from the NHANES.* BMC Medical Informatics and Decision Making, 25, 83.

The original work trained six classifiers on a single NHANES wave (2013–2014) and reported XGBoost as the top performer, using SHAP for feature importance without separating biomarker from demographic contributions. It did not test cross-wave generalization, quantify uncertainty, or assess the effect of survey weighting.

**Note on verification:** this README treats the cited study as reported in the project paper. Before publication or public release, the DOI and reported figures should be independently re-verified against the source, as is standard practice when building on external work.

## Research Gaps Addressed

| Gap | Original Study | This Project |
|---|---|---|
| **Cross-wave validation** | Not tested | Trains on 2013–2016, validates on 2017–2018, tests on the 2017–March 2020 file; disentangles demographic drift from concept drift |
| **Uncertainty quantification** | Point estimates only | Bootstrap ensemble (20–30 models) producing a mean risk score and a confidence interval |
| **Biomarker vs. demographic attribution** | Undifferentiated SHAP | Explicit categorization of SHAP contributions for interpretability and fairness auditing |
| **Survey weighting** | Disclosed as a limitation | Treated as a formal sensitivity analysis (weighted vs. unweighted training) |

## Dataset

NHANES is a nationally representative U.S. survey conducted by the CDC in two-year cycles.

| Wave | Years | Sample Size | Role |
|---|---|---|---|
| 1 | 2013–2014 | ~5,000 | Primary training (baseline replication) |
| 2 | 2015–2016 | ~5,000 | Extended training |
| 3 | 2017–2018 | ~5,000 | Validation |
| 4 | 2017–March 2020* | ~9,000 | Final external test set |

\* NHANES suspended field operations in March 2020 due to COVID-19. The CDC's combined "2017–March 2020 pre-pandemic" file is used as the final test set, with an explicit note on its partial year overlap with the validation wave.

**Target variable:** Derived from the PHQ-9 questionnaire; a score ≥ 10 indicates moderate-to-severe depressive symptoms. Fewer than 9% of respondents meet this threshold, so the project accounts for class imbalance via stratified splitting, class weighting, and AUPRC as a primary metric alongside AUROC.

## Methodology

**Phase 1 — Baseline Reproduction**
Replicate the original paper on 2013–2014 data using Logistic Regression, Random Forest, and XGBoost.

**Phase 2 — Cross-Wave Evaluation**
Train on 2013–2016, validate on 2017–2018, test on the 2017–March 2020 file. Run per-wave demographic analysis to separate population drift from genuine concept drift.

**Phase 3 — Bootstrap Uncertainty Ensemble**
Train 20–30 XGBoost models on bootstrap resamples of the training data. The ensemble mean gives the risk score; the standard deviation across models gives the uncertainty estimate.

```text
Patient A — Risk: 87% | Uncertainty: Low (σ = 0.03)  → high confidence, flag for screening
Patient B — Risk: 61% | Uncertainty: High (σ = 0.19) → borderline, refer for clinical interview
```

**Phase 4 — SHAP Explainability**
Apply SHAP for global and individual-level feature importance, split explicitly into biomarker and demographic groups.

**Phase 5 — Survey Weight Sensitivity Analysis**
Compare weighted (`sample_weight` with WTMEC2YR) vs. unweighted training on performance and calibration; report weighted prevalence statistics.

## Expected Results

| Model | Purpose | Expected AUROC |
|---|---|---|
| Logistic Regression (single-wave) | Minimum viable baseline | ~0.76 |
| Random Forest (single-wave) | Non-linear tabular baseline | ~0.80 |
| XGBoost (single-wave) | Reproduce original paper | ~0.84 |
| XGBoost (multi-wave) | Extended training | ~0.84–0.86 |
| Bootstrap Uncertainty Ensemble | Main contribution | ~0.85+ |

These are **target figures based on the original paper's reported performance**, not yet-observed results. They should be read as feasibility benchmarks, not guarantees — actual performance depends on cross-wave feature alignment and how well the ensemble calibrates.

## Evaluation Metrics

- AUROC
- AUPRC (primary metric given class imbalance)
- F1-Score
- Calibration (reliability curves)
- Sensitivity / Specificity at a clinically meaningful threshold
- Cross-wave AUROC degradation table

## Repository Structure

```
mindscreen/
├── data/
│   ├── raw/                  # Raw NHANES cycle files (not tracked in git)
│   └── processed/            # Cleaned, feature-aligned datasets
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_cross_wave_eval.ipynb
│   ├── 03_uncertainty_ensemble.ipynb
│   ├── 04_shap_explainability.ipynb
│   └── 05_survey_weight_analysis.ipynb
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── ensemble.py
│   └── evaluation.py
├── reports/
│   └── figures/
├── requirements.txt
├── README.md
└── LICENSE
```

## Risks and Limitations

- **Feature alignment across cycles** is the primary practical risk — NHANES variable names and coding occasionally change between waves. Mitigated by verifying variable availability before finalizing the feature set.
- **Phase 1 alone** is designed to stand on its own as a reproducible baseline, so the project has a fallback deliverable even if later phases run into obstacles.
- **A performance drop across waves is itself a valid finding** — the project is structured so that a negative or mixed result is still publishable, rather than requiring a specific outcome.
- **Clinical caveat:** this project is a research prototype for screening support, not a diagnostic tool, and should not be interpreted or deployed as one.

## Data and Code Availability

All data is publicly available through the [NHANES CDC portal](https://wwwn.cdc.gov/nchs/nhanes/), requiring no special credentials. All code will be released in this repository to support full reproducibility.

## References

- Vu et al. (2025). Prediction of depressive disorder using machine learning approaches: findings from the NHANES. *BMC Medical Informatics and Decision Making*, 25, 83.
- [NHANES Dataset — CDC](https://wwwn.cdc.gov/nchs/nhanes/)
- [SHAP Library](https://github.com/shap/shap)
- [scikit-learn](https://scikit-learn.org)
- [XGBoost](https://xgboost.readthedocs.io)

## Status

This repository is in active development. Baseline reproduction (Phase 1) is the current focus; subsequent phases will be added incrementally as milestones are completed.

---

<sub>Maintained as part of an academic research internship. Contributions, issues, and methodological critique are welcome via GitHub Issues.</sub>
