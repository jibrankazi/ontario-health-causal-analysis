Ontario Health Causal Analysis

Overview

This repository contains my independent research project on causal inference for policy evaluation, focused on estimating the effect of a public-health intervention across Ontario’s regions.
The project integrates econometric and machine-learning techniques—Difference-in-Differences (DiD), Propensity Score Matching (PSM), and optional Bayesian Structural Time Series (BSTS)—to triangulate causal effects from real-world, observational data.

The goal is to build a reproducible, open, and modular causal-inference pipeline that can be adapted for similar public-policy or epidemiological studies.
All analyses are deterministic and version-controlled, with CI/CD pipelines for automated testing and documentation.

This project reflects my broader research direction in causal reasoning, interpretability, and reproducible AI, which I aim to pursue in the PhD program in Computer Science at the University of Toronto (Fall 2026).

Methodology

Causal estimators implemented:

Difference-in-Differences (DiD) with region-clustered standard errors.

Propensity Score Matching (PSM) using logistic regression for treatment assignment.

Bayesian Structural Time Series (BSTS) (toggleable) for counterfactual time-series modeling.

Core features:

YAML configuration for reproducible parameter management.

Modular workflow with deterministic execution order:

scripts/clean_all.py

src/run_analysis.py

scripts/regenerate_figures.py

scripts/build_html_report.py

Auto-generated figures and HTML report published via GitHub Pages.

Lightweight pytest smoke tests for schema validation and CI reproducibility checks.

Libraries used: pandas, numpy, statsmodels, scikit-learn, matplotlib, and yaml.

Results at a Glance
Method	ATT	Uncertainty	Notes
DiD	−245.82	SE ≈ 166.37	Not statistically significant
PSM	−105.20	—	Computed successfully
BSTS	—	—	Disabled in final configuration (bsts.enabled: false)

Source: results/results.json

Figure: figures/att_summary.png

Interpretation:
Across both estimators, the estimated treatment effects were negative but statistically inconclusive, suggesting that the observed intervention may have reduced incidence rates modestly but not with high confidence.
The focus of this work was reproducibility and methodological transparency, rather than policy advocacy.

Project Structure
ontario-health-causal-analysis/
├─ src/                   # Core analysis modules (DiD/PSM/BSTS)
│  └─ run_analysis.py
├─ scripts/               # Workflow orchestration scripts
│  ├─ clean_all.py
│  ├─ regenerate_figures.py
│  └─ build_html_report.py
├─ results/               # Auto-generated JSON output
├─ figures/               # Generated plots
├─ reports/               # HTML report (GitHub Pages)
├─ tests/                 # Smoke tests and schema validation
├─ config.yaml            # Parameter toggles and metadata
├─ LICENSE
├─ CITATION.cff
└─ README.md

Live Report

📄 Interactive HTML report:
https://jibrankazi.github.io/ontario-health-causal-analysis/
 → reports/report.html

Each new CI/CD run regenerates the analysis and automatically updates this report.

Research Context

This project embodies my interest in robust causal identification, open-source reproducibility, and interpretable AI.
It directly informs my PhD research agenda in causal and generative reasoning.

The methodological rigor and open-science practices here are conceptually aligned with:

Prof. Rahul G. Krishnan’s work on structured inference and deep generative models.

Prof. Sheila McIlraith’s work on explainable AI and reasoning-based reinforcement learning.

Both research directions inform my goal to design transparent, empirically grounded AI systems capable of causal reasoning and robust decision support.

How to Reproduce
# Clone repository
git clone https://github.com/jibrankazi/ontario-health-causal-analysis.git
cd ontario-health-causal-analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python scripts/clean_all.py
python src/run_analysis.py
python scripts/regenerate_figures.py
python scripts/build_html_report.py


All outputs are stored in /results, /figures, and /reports.

License

This project is released under the MIT License — see LICENSE
.

Citation

If you use this work or adapt the methodology, please cite:

@software{Kazi_OHCA_2025,
  author = {Kazi, Jibran Rafat Samie},
  title = {Ontario Health Causal Analysis: Reproducible Policy Evaluation using DiD, PSM, and BSTS},
  year = {2025},
  url = {https://github.com/jibrankazi/ontario-health-causal-analysis},
  license = {MIT}
}

Contact

Kazi Jibran Rafat Samie
📍 Toronto, Canada
📧 jibrankazi@gmail.com

🔗 github.com/jibrankazi

🔗 linkedin.com/in/jibrankazi

© 2025 Kazi Jibran Rafat Samie
Independent research project on causal inference and public-health policy evaluation.
Part of my doctoral research direction.
