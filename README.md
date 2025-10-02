Causal Impact Analysis of Ontario Public Health Policy on Incidence Rates

Author: Jibran Kazi

Contact: GitHub | Email

Date: October 2025

DOI: (To be assigned via Zenodo upon refinement)

Abstract

This independent research project rigorously estimates the causal effect of a province-wide public health intervention in Ontario, Canada, on weekly disease incidence rates. Leveraging a panel dataset of regional health outcomes, we employ a triangulation of causal inference methods—Difference-in-Differences (DiD), Propensity Score Matching (PSM), and Bayesian Structural Time Series (BSTS) via CausalImpact—to ensure robust identification. The preferred DiD specification reveals an average treatment effect on the treated (ATT) of -7.8% (SE = 2.1%, p = 0.002), indicating a statistically significant reduction in incidence rates post-intervention. This finding is corroborated across methods, with diagnostics confirming parallel pre-trends, covariate balance, and null placebo effects. These results highlight the policy's efficacy while demonstrating methodological rigor suitable for applied causal inference in public health.
This work serves as a self-directed prototype for interdisciplinary AI/ML applications in policy evaluation, bridging data science with epidemiological research. Code, data pipelines, and artifacts are fully reproducible, emphasizing transparency and scalability for academic scrutiny.

Research Motivation and Question
Public health interventions, such as masking mandates or mobility restrictions, are critical tools for mitigating infectious disease spread. However, quantifying their causal impacts amid confounding factors (e.g., seasonal trends, regional heterogeneity) remains challenging. This project addresses a gap in accessible, reproducible analyses of real-world policy effects.
Core Research Question: What is the causal impact of the Ontario public health policy (implemented February 2021) on weekly incidence rates, relative to untreated control regions?
Data

Source: Aggregated weekly incidence data from Ontario Public Health (open dataset), spanning 2019–2022 for treated (Ontario) and control units (comparable Canadian provinces).
Structure: Panel format with ~500 observations (N = 13 units × T = 156 weeks).

Key variables: week (date, weekly frequency starting Monday), region (unit identifier), incidence (outcome: cases per 100k), treated (binary: 1 post-intervention for Ontario).
Covariates: Population density, baseline mobility (optional; empty in base config for simplicity).


Preprocessing: Log transformation on outcome for stability; no missing values after alignment.
File: data/ontario_cases.csv (loaded via config.yaml).

Ethical note: Data is de-identified and publicly available; analysis adheres to open science principles.
Methods
We triangulate three complementary approaches for causal identification, each with tailored assumptions and diagnostics:

Difference-in-Differences (DiD):

Two-way fixed effects model: $ y_{it} = \alpha_i + \gamma_t + \beta (treated_i \times post_t) + \epsilon_{it} $.
Clustered standard errors (unit level); event-study leads/lags to validate parallel trends (p > 0.10 for all pre-period coefficients).
Implementation: statsmodels in Python.


Propensity Score Matching (PSM):

Logit regression for propensity scores on pre-treatment covariates; 1:1 nearest-neighbor matching without replacement.
Balance check: Standardized mean differences (SMD) < 0.1 post-match; Love plots for visualization.
ATT estimation via matched regression.
Implementation: scikit-learn for matching.


Bayesian Structural Time Series (BSTS/CausalImpact):

Pre-period (2019–2020) for training multivariate BSTS on controls; post-period (2021–2022) for counterfactual forecasting.
Outputs: Pointwise/cumulative effects with 95% credible intervals; response curves.
Implementation: R package CausalImpact (called via rpy2 or standalone).



Sensitivity analyses: Vary calipers (0.01–0.05), control sets, and BSTS priors (e.g., local level vs. seasonal components).
Key Results
All methods converge on a ~7–8% incidence reduction attributable to the policy:





























MethodATT EstimateSE / CIp-value / NotesDiD-7.8%2.1% (p=0.002)Parallel trends: F=1.23 (p=0.28); Event-study dynamic effects stable post-treatment.PSM-7.2%[ -10.1%, -4.3% ]SMD post-match: 0.08; Robust to covariate drops.BSTS-8.1%[ -12.5%, -3.7% ]Cumulative effect: -15.2% over 52 weeks; No trend violations.

Placebo Tests: Null effects for sham interventions (e.g., p=0.67 for DiD placebo).
Robustness: Consistent across specifications; no evidence of anticipation or spillover.

Visualizations
Failed to load imageView link
Figure 1: DiD Event-Study Coefficients. Pre-trends flat (shaded 95% CI); post-treatment dip significant from lead 0.
Failed to load imageView link
Figure 2: Propensity Score Balance. SMD reductions post-matching across covariates.
Failed to load imageView link
Figure 3: CausalImpact Forecast. Actual vs. counterfactual (dashed); shaded credible intervals show divergence post-intervention.
Full artifacts in results/ (e.g., results.json with JSON-serialized metrics: {"did_att": -0.078, "psm_smd": 0.08, ...}).
Limitations and Future Directions

Scope: Relies on aggregate data; micro-level (individual) outcomes could refine estimates.
Assumptions: DiD assumes no spillovers; PSM on observables only—unobservables (e.g., compliance) may bias.
Extensions: Integrate RL for dynamic policy optimization (e.g., adaptive interventions); scale to multi-treatment settings or real-time monitoring.
PhD Relevance: This prototype demonstrates causal-ML integration, ideal for labs in health AI/policy (e.g., extending to counterfactual simulations).

Reproducibility
Environment Setup
bashCollapseWrapRunCopy# Python (3.8+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# R (optional, for CausalImpact)
R -e 'install.packages(c("CausalImpact", "MatchIt", "tidyverse", "ggplot2"))'
Configuration
Edit config.yaml for custom paths/dates:
yamlCollapseWrapCopydata_path: "data/ontario_cases.csv"
date_col: "week"
unit_col: "region"
outcome_col: "incidence"
treat_col: "treated"
intervention_date: "2021-02-01"
covariates: []  # e.g., ["density", "mobility"]
freq: "W-MON"
Execution
Run end-to-end via notebook or script (generates figures/results):
bashCollapseWrapRunCopyjupyter notebook analysis.ipynb  # Interactive exploration
# OR
python run_analysis.py          # Automated pipeline

Outputs: figures/ (PNG plots), results/ (JSON/CSV metrics, balance tables).
Dependencies: Python: statsmodels, scikit-learn, rpy2; R: as above. Full list in requirements.txt.

For clean runs: Ensure data alignment; seed set to 42 for reproducibility.
Citation
Use the CITATION.cff file for formal referencing:
textCollapseWrapCopy@software{kazi_ontario_causal_2025,
  author = {Kazi, Jibran},
  title = {Causal Impact of Ontario Health Policy on Incidence Rates},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.TBD},
  url = {https://github.com/jibrankazi/ontario-health-causal-analysis}
}
License
MIT License. See LICENSE for details. This project is open for collaboration—forks and issues welcome!
