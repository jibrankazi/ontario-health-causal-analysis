# 🧠 Ontario Health Causal Analysis

A reproducible Python pipeline for estimating the causal impact of health policy interventions in Ontario using **Difference-in-Differences (DiD)**, **Propensity Score Matching (PSM)**, and **Bayesian Structural Time Series (BSTS)** methods.

---

## 🚀 Project Overview

This repository automates every step of a causal analysis workflow:

1. **Data ingestion & feature engineering** (lags, pre-policy means)  
2. **Causal estimations** via DiD, PSM, and BSTS  
3. **Result verification** and validation tests  
4. **Figure & HTML report generation** for clean outputs  

The pipeline is fully automated from raw data → verified results → publication-ready report.

---

## 🧩 Directory Structure
ontario-health-causal-analysis/
├── config.yaml # Global configuration file
├── data/ # Input datasets
│ └── ontario_cases.csv
├── src/ # Core analysis code
│ ├── run_analysis.py
│ └── ...
├── scripts/ # Utility scripts
│ ├── clean_all.py
│ ├── regenerate_figures.py
│ └── build_html_report.py
├── tests/ # Automated regression & smoke tests
│ ├── test_smoke.py
│ └── conftest.py
├── results/ # Output metrics (results.json)
├── figures/ # Generated plots (att_summary.png)
├── reports/ # Final HTML report
├── pyproject.toml # Packaging metadata
└── README.md
