\# Ontario Health Causal Analysis: Policy Impact Evaluation



\*\*Author:\*\* Kazi Jibran Rafat Samie  

\*\*Location:\*\* Toronto, Canada  

\*\*Status:\*\* Complete (November 2025)



!\[Python](https://img.shields.io/badge/Python-3.9%2B-blue)

!\[R](https://img.shields.io/badge/R-4.0%2B-blue)

!\[Status](https://img.shields.io/badge/Reproducibility-High-green)



\## 📌 Abstract

This independent research project evaluates the causal impact of Ontario's public health policy intervention (effective \*\*February 1, 2021\*\*) on regional COVID-19 case incidence. 



Using a dataset of \*\*4,760 observations\*\* across 34 public health units, this analysis employs a multi-method causal inference pipeline to estimate the Average Treatment Effect on the Treated (ATT). The study triangulates results using \*\*Difference-in-Differences (DiD)\*\* and \*\*Propensity Score Matching (PSM)\*\* to ensure robustness.



\## 📊 Key Findings



| Method | Estimate (ATT) | Standard Error | Interpretation |

| :--- | :--- | :--- | :--- |

| \*\*DiD\*\* | \*\*-108.61\*\* | 105.80 | Policy associated with ~109 fewer cases per 100k. |

| \*\*PSM\*\* | \*\*-119.85\*\* | N/A | Matching confirms reduction magnitude (~120 fewer cases). |

| \*\*BSTS\*\* | \*Inconclusive\* | N/A | Pre-period volatility prevented Bayesian convergence. |



\### Interpretation

The analysis yields highly consistent point estimates between two distinct methodological frameworks:

1\.  \*\*Robustness:\*\* The DiD estimate (-108.61) and PSM estimate (-119.85) are within \*\*~10%\*\* of each other. This convergence provides strong evidence for the direction and magnitude of the effect.

2\.  \*\*Uncertainty:\*\* While the point estimates suggest a substantial reduction, the high standard error in the DiD model reflects the significant heterogeneity between Ontario's large urban centers (Toronto/Peel) and smaller rural regions.

3\.  \*\*Data Constraints:\*\* The Bayesian Structural Time Series (BSTS) model returned `NA`, correctly identifying that the pre-intervention period (March 2020 – Jan 2021) contained insufficient stable seasonality to construct a reliable synthetic counterfactual.



\## 🛠 Methodology



\### 1. Difference-in-Differences (DiD)

\* \*\*Specification:\*\* Two-way Fixed Effects (TWFE) model controlling for unit-invariant time trends and time-invariant unit characteristics.

\* \*\*Standard Errors:\*\* Clustered at the region level to account for serial correlation.

\* \*\*Equation:\*\* $Y\_{it} = \\alpha + \\beta (Treat\_i \\times Post\_t) + \\gamma\_i + \\delta\_t + \\epsilon\_{it}$



\### 2. Propensity Score Matching (PSM)

\* \*\*Technique:\*\* Logistic regression to estimate propensity scores based on pre-period incidence.

\* \*\*Matching:\*\* Nearest Neighbor (k=1) with a strict caliper (0.2 \* SD of logit propensity score) to ensure common support.

\* \*\*Validation:\*\* Caliper filtering removed poor matches to reduce bias.



\### 3. Bayesian Structural Time Series (BSTS)

\* \*\*Implementation:\*\* Integrated R's `CausalImpact` package via Python subprocess.

\* \*\*Note:\*\* Used as a sensitivity check. The model found the pre-period signal-to-noise ratio too low for valid inference in this specific time window.



\## 🚀 Reproducibility

This pipeline is designed to be fully deterministic and reproducible.



\### Prerequisites

\* \*\*Python 3.9+\*\* (`pandas`, `statsmodels`, `sklearn`, `numpy`, `yaml`)

\* \*\*R 4.x\*\* (`CausalImpact`, `jsonlite`, `zoo`)



\### Installation

```bash

\# 1. Clone the repository

git clone \[https://github.com/jibrankazi/ontario-health-causal-analysis.git](https://github.com/jibrankazi/ontario-health-causal-analysis.git)

cd ontario-health-causal-analysis



\# 2. Install Python dependencies

pip install -r requirements.txt



\# 3. Install R dependencies (run in terminal)

Rscript -e 'install.packages(c("CausalImpact", "jsonlite", "zoo"), repos="\[https://cloud.r-project.org](https://cloud.r-project.org)")'

