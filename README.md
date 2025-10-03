# Causal Impact of Ontario's Public Health Policy on Incidence Rates

**Author**: Jibran Kazi  
**Contact**: [GitHub](https://github.com/jibrankazi) | [Email](mailto:jibrankazi@gmail.com)  
**Date**: October 2025  
**Repository**: [github.com/jibrankazi/ontario-health-causal-analysis](https://github.com/jibrankazi/ontario-health-causal-analysis)

## Abstract

This self-directed research evaluates the causal effect of Ontario’s 2021 public health policy on weekly disease incidence rates using a robust triangulation of methods: Difference-in-Differences (DiD), Propensity Score Matching (PSM), and Bayesian Structural Time Series (BSTS). Analyzing a panel dataset of 500+ observations (2019–2022), the DiD model estimates an average treatment effect on the treated (ATT) of -7.8% (p=0.002), corroborated by PSM (-7.2%) and BSTS (-8.1%). Diagnostics confirm parallel trends and covariate balance. This project showcases applied causal inference for policy evaluation, integrating Python and R workflows, and serves as a prototype for AI-driven public health research.

## Research Question

What is the causal impact of Ontario’s February 2021 public health policy on weekly disease incidence rates, compared to untreated Canadian provinces?

## Data

- **Source**: Publicly available weekly incidence data from Ontario Public Health and control provinces (2019–2022).
- **Structure**: Panel dataset (~500 observations; 13 regions × 156 weeks).
  - Variables: `week` (date), `region` (identifier), `incidence` (cases per 100k), `treated` (binary, post-intervention).
  - Covariates: Optional (e.g., population density, mobility).
- **Preprocessing**: Log-transformed outcomes; no missing data.
- **File**: `data/ontario_cases.csv` (configured via `config.yaml`).
- **Intervention cutoff**: Set in `config.yaml` with the `intervention_date` key (legacy `policy_date` is still supported for backwards compatibility).

*Note*: Data is anonymized, adhering to ethical standards.

## Methods

Three causal inference methods ensure robust estimation:

1. **Difference-in-Differences (DiD)**:
   - Model: $$  y_{it} = \alpha_i + \gamma_t + \beta (treated_i \times post_t) + \epsilon_{it}  $$.
   - Diagnostics: Parallel trends (F=1.23, p=0.28); clustered SEs.
   - Tools: Python (`statsmodels`).

2. **Propensity Score Matching (PSM)**:
   - 1:1 matching on pre-treatment covariates; balance via SMD (<0.1).
   - Tools: Python (`scikit-learn`).

3. **Bayesian Structural Time Series (BSTS)**:
   - Counterfactual forecasting using pre-intervention data.
   - Tools: R (`CausalImpact`) via `rpy2`.

**Sensitivity**: Varied control sets, calipers (0.01–0.05), and BSTS priors. Placebo tests confirm null effects (p=0.67).

## Results

| Method | ATT | SE/CI | Notes |
@@ -59,25 +60,35 @@ Three causal inference methods ensure robust estimation:
<image-card alt="Balance Plot" src="figures/fig2_smd_balance.png" ></image-card>  
*Figure 2: PSM covariate balance (SMD pre/post-matching).*

<image-card alt="Counterfactual Plot" src="figures/fig3_bsts_counterfactual.png" ></image-card>  
*Figure 3: BSTS actual vs. counterfactual incidence rates.*

Results are stored in `results/results.json` (e.g., `{"did_att": -0.078, ...}`).

## Limitations and Future Work

- **Limitations**: Aggregate data limits individual-level insights; assumes no spillovers or unobserved confounders.
- **Future Directions**: Integrate reinforcement learning for dynamic policy optimization; extend to multi-treatment settings.
- **PhD Relevance**: Demonstrates causal-ML integration for health policy, aligning with AI-driven epidemiology research.

## Reproducibility

### Setup
```bash
# Python (3.8+)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# R (for BSTS)
R -e 'install.packages(c("CausalImpact", "MatchIt", "tidyverse"))'

# Windows: expose Rscript for the BSTS step
R_BASE="/c/Program Files/R"  # adjust if installed elsewhere
R_HOME="$(ls -1d "$R_BASE"/R-* | sort -V | tail -n1)"
export R_HOME="$R_HOME"
export PATH="$R_HOME/bin/x64:$R_HOME/bin:$PATH"

# Optional: bake the path into config.yaml so the helpers find it automatically
# r_home: "C:/Program Files/R/R-4.5.1"
# rscript_path: "C:/Program Files/R/R-4.5.1/bin/Rscript.exe"
