# Causal Impact of Ontario Health Policy on Incidence Rates

**Author:** Jibran Kazi

## Abstract
We estimate the causal effect of a province‑wide public‑health intervention in Ontario on weekly incidence using DiD (two‑way FE),
propensity score matching, and CausalImpact. Our preferred DiD model reports ATT, SE, and p‑value; matched comparisons and
counterfactual forecasts corroborate the findings.

## 1. Introduction
Motivates the policy question, data constraints, and why triangulation improves robustness.

## 2. Related Work
Card & Krueger (DiD), Rosenbaum & Rubin (PSM), Brodersen et al. (BSTS/CausalImpact).

## 3. Methods
- Data, pre/post, units, covariates
- DiD spec, identification, clustering
- PSM, balance checks (SMD)
- CausalImpact setup

## 4. Results
Insert key numbers from `results/results.json`. Include Figures 1–3 from `figures/`.

## 5. Conclusion & Future Work
Limitations (unobserved confounding), heterogeneity, staggered adoption, RL for adaptive policy.
