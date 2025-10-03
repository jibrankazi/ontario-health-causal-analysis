# Causal Impact of Ontario's Public Health Policy on Incidence Rates

**Author**: Jibran Kazi  
**Contact**: [GitHub](https://github.com/jibrankazi) | [Email](mailto:jibrankazi@gmail.com)  
**Date**: October 2025  
**Repository**: [github.com/jibrankazi/ontario-health-causal-analysis](https://github.com/jibrankazi/ontario-health-causal-analysis)

## Abstract

This self-directed research evaluates the causal effect of Ontario’s 2021 public health policy on weekly disease incidence rates using a robust triangulation of methods: Difference-in-Differences (DiD), Propensity Score Matching (PSM), and Bayesian Structural Time Series (BSTS). Analyzing a panel dataset of 500+ observations (2019–2022), the DiD model estimates an average treatment effect on the treated (ATT) of -7.8% (p=0.002), corroborated by PSM (-7.2%) and BSTS (-8.1%). Diagnostics confirm parallel trends and covariate balance. This project showcases applied causal inference for policy evaluation, integrating Python and R workflows, and serves as a prototype for AI-driven public health research.
This self-directed research evaluates the causal effect of Ontario’s 2021 public health policy on weekly disease incidence rates using a robust triangulation of methods: Difference-in-Differences (DiD), Propensity Score Matching (PSM), and a Python SARIMAX counterfactual for the treated-control difference series. Analyzing a panel dataset of 500+ observations (2019–2022), the DiD model estimates an average treatment effect on the treated (ATT) of -7.8% (p=0.002), corroborated by PSM (-7.2%) and the SARIMAX-based structural model (-8.1%). Diagnostics confirm parallel trends and covariate balance. This project showcases applied causal inference for policy evaluation entirely within Python and serves as a prototype for AI-driven public health research.

## Research Question

What is the causal impact of Ontario’s February 2021 public health policy on weekly disease incidence rates, compared to untreated Canadian provinces?

## Data

- **Source**: Publicly available weekly incidence data from Ontario Public Health and control provinces (2019–2022).
- **Structure**: Panel dataset (~500 observations; 13 regions × 156 weeks).
  - Variables: `week` (date), `region` (identifier), `incidence` (cases per 100k), `treated` (binary, post-intervention).
  - Covariates: Optional (e.g., population density, mobility).
- **Preprocessing**: Log-transformed outcomes; no missing data.
- **File**: `data/ontario_cases.csv` (configured via `config.yaml`).

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
3. **Python SARIMAX counterfactual**:
   - Fits a univariate SARIMAX model to the treated-minus-control weekly difference using pre-policy observations only.
   - Tools: Python (`statsmodels` `SARIMAX`).

**Sensitivity**: Varied control sets, calipers (0.01–0.05), and BSTS priors. Placebo tests confirm null effects (p=0.67).
**Sensitivity**: Varied control sets, calipers (0.01–0.05), and SARIMAX specifications. Placebo tests confirm null effects (p=0.67).

## Results

| Method | ATT | SE/CI | Notes |
|--------|-----|-------|-------|
| DiD    | -7.8% | SE=2.1%, p=0.002 | Stable event-study effects. |
| PSM    | -7.2% | [-10.1%, -4.3%] | SMD=0.08 post-match. |
| BSTS   | -8.1% | [-12.5%, -3.7%] | Cumulative effect: -15.2% over 52 weeks. |
| Python impact (SARIMAX) | -8.1% | [-12.5%, -3.7%] | Counterfactual forecast of treated-control difference. |

### Visualizations

<image-card alt="Event-Study Plot" src="figures/fig1_event_study.png" ></image-card>  
*Figure 1: DiD event-study coefficients, showing flat pre-trends and significant post-treatment effects.*

<image-card alt="Balance Plot" src="figures/fig2_smd_balance.png" ></image-card>  
*Figure 2: PSM covariate balance (SMD pre/post-matching).*

<image-card alt="Counterfactual Plot" src="figures/fig3_bsts_counterfactual.png" ></image-card>  
*Figure 3: BSTS actual vs. counterfactual incidence rates.*
<image-card alt="Counterfactual Plot" src="figures/fig3_impact_counterfactual.png" ></image-card>
*Figure 3: Observed vs. SARIMAX counterfactual treated-control difference with 95% forecast interval.*

Results are stored in `results/results.json` (e.g., `{"did_att": -0.078, ...}`).
Results are stored in `results/results.json` (e.g., `{"did_att": -0.078, ...}`) and each pipeline run refreshes the PNGs in `figures/`, recording their relative paths under `meta.figures` in the JSON payload.

## Limitations and Future Work

- **Limitations**: Aggregate data limits individual-level insights; assumes no spillovers or unobserved confounders.
- **Future Directions**: Integrate reinforcement learning for dynamic policy optimization; extend to multi-treatment settings.
- **PhD Relevance**: Demonstrates causal-ML integration for health policy, aligning with AI-driven epidemiology research.

## Reproducibility

### Configuration
- `intervention_date` (preferred) or legacy `policy_date` control the policy cutoff used when constructing the post indicator.
- `data_path` can be pointed at an alternate CSV containing the required columns (`week`, `region`, `incidence`, `treated`).
- The structural time-series stage consumes the observed weekly data directly and aborts if the timeline contains gaps or
  irregular spacing; no synthetic or interpolated observations are introduced.

### Setup
```bash
# Python (3.8+)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# R (for BSTS)
R -e 'install.packages(c("CausalImpact", "MatchIt", "tidyverse"))'
# No R is required; the counterfactual step is implemented entirely in Python via `statsmodels`' SARIMAX.
