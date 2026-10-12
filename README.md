# Ontario Health Causal Analysis — verified checked-in panel, but **causal effect not established**

The repository contains a historic `data/ontario_cases.csv` panel with **4,760 weekly PHU incidence observations across 34 coded public health units and 140 weeks from 2020-03-30 through 2022-11-28**. New reproducible identification diagnostics test original-file completeness, apparent differences between labeled PHUs, pre-intervention divergence, and a pre-policy placebo cutoff, then produce actual input-row CSVs, an observed trend chart and a detailed audit.

**[Automated checked-in PHU panel and pre-policy placebo audit](https://github.com/jibrankazi/ontario-health-causal-analysis/actions/workflows/real-phu-identification.yml)**.

**Important:** The original CSV's row-level provenance **has not been independently reconciled to a Government of Ontario dataset**. The original file's `treated` flags identify **5 PHUs as treated and 29 as comparison**, but the repository does **not** document an externally verified basis for treating those five PHUs differently on February 1, 2021. A province-wide pandemic policy cannot automatically be studied by calling some Ontario health units "untreated." No measured government-verified causal policy impact is established.

## What the actual existing panel checks show

| Executed diagnostic | Original checked-in data result |
| --- | ---: |
| Historical PHU-week rows | **4,760** |
| PHU codes / weeks | **34 / 140** |
| PHU codes labeled treated / comparison | **5 / 29** |
| Weekly observations before / on or after the assumed February 1, 2021 cutoff | **44 / 96** |
| Missing PHU-week cells or within-region changing labels | **None detected** |
| Pre-policy treated vs comparison **weekly trend difference** | **−12.11** source incidence units per week |
| Pre-policy weekly trend approximate 95% CI | **[−25.28, +1.07]** |
| Pre-policy *placebo* October 5, 2020 group DiD association | **−317.02** source incidence units |
| Pre-policy placebo approximate 95% CI | **[−680.49, +46.45]** |
| Original full-period post-minus-pre difference in group means | **−108.61** source incidence units |
| Publisher-record and treatment-assignment source documentation independently verified | **No** |

These are diagnostics on a checked-in dataset. The source's `incidence` column is used as recorded; its measurement units and derivation from original government case counts still require primary-source verification. The placebo difference is **entirely before the claimed policy date**, illustrating why negative group differences cannot simply be interpreted as effects of the February 2021 policy. A 95% interval crossing zero is *not* proof that parallel trends hold. The small treated-PHU cluster count is another reason to avoid strong significance claims.

The previously reported difference-in-differences point estimate was about **−108.61**, with an approximate 95% CI of **[−315.97, +98.75]**, so zero is included even before addressing identification. The earlier matching figures were not consistently reproduced and do not establish a causal effect. Existing R BSTS claims remain unverified in current CI.

## Run the new independent, actual-file diagnosis

```bash
python -m pip install "numpy<3" pandas statsmodels matplotlib pytest
python -m pytest -q tests/test_real_phu_identification_audit.py tests/test_data_contract.py
python scripts/real_phu_identification_audit.py
```

The new workflow reruns these tests, analyzes **all the checked-in records**, independently recomputes saved group means and confidence limits, validates the original file byte SHA-256, and publishes evidence as a GitHub Actions artifact.

Outputs in `results/verified_phu_identification/`:

- `verified_identification_diagnostics.json` — actual dataset counts, date cutoff, treatment-label distribution, pretrend and placebo estimates with clustered uncertainty, file SHA-256, and source/causal caveats.
- `original_phu_weekly_incidence_and_treatment_labels.csv` — all original checked-in rows and labels, not generated cases.
- `checked_in_original_phu_group_weekly_means.csv` — observed PHU group means for all 140 weeks, including the directly calculated difference.
- `observed_original_phu_incidence_by_group.png` — original-file group time series and a clearly marked assumed policy cutoff.

Existing scripts and the prior GitHub Pages report are preserved for reproducibility, but **their historical treatment-effect headings, matched effect comparisons, and language suggesting policy success are not validated causal findings**. Code execution is not a substitute for identifying a valid control group.

## What's needed to complete genuine policy-impact research

First obtain and reconcile the official public health unit records with archived Ontario source files, including original incidence definitions, population denominators, and changes to geography/reporting practices. Next identify and cite a specific dated policy with **documented variation in treatment exposure**, a credible comparable control group, a time-aware design, and policy anticipation/overlap checks. Then run pre-period placebo tests and modern inference with few treated clusters, multiple comparisons, other concurrent interventions and sensitivity analyses.

**Verified today:** an actual checked-in panel → complete PHU-week validation → independently calculated observed-group statistics and exploratory pre-policy falsification checks. **Not verified:** that the checked-in panel reproduces the official government source, the causal treatment assignment, or that any Ontario policy reduced COVID incidence.

## Public report generation integrity (October 2026)

The previous `scripts/regenerate_figures.py` file only declared `main()` and never executed it when called from CI. It also lacked required imports and incorrectly substituted zero for unavailable BSTS output. It now runs as a real CLI script, refuses non-finite estimates, omits absent methods rather than fabricating zero-valued bars, explicitly marks historical associations as noncausal, and produces `figures/att_summary.png`. CI tests the actual PNG and requires that the generated public HTML report embeds it. The old committed `docs/index.html`, `docs/analysis.html`, and `reports/report.html` contained stale values or misleading causal-language; these are replaced with source-limited previews.

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python src/run_analysis.py
python scripts/regenerate_figures.py
python scripts/build_html_report.py
python -m pytest -q tests/test_report_integrity.py tests/test_data_contract.py
```

This does **not** resolve underlying Government of Ontario source provenance, treatment identification, or real causal validity.
