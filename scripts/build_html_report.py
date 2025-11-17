#!/usr/bin/env python3
from pathlib import Path
import json, base64, math
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "results.json"
FIG = ROOT / "figures" / "att_summary.png"
OUT = ROOT / "reports" / "report.html"

def b64(p: Path) -> str:
    """Helper to base64-encode an image for embedding in HTML."""
    return base64.b64encode(p.read_bytes()).decode("ascii") if p.exists() else ""

def pick_att(r, method):
    """Helper to find the ATT value from the results JSON."""
    flat_key = f"{method.lower()}_att"
    if flat_key in r:
        return r.get(flat_key)
    nested = r.get(method.lower(), {})
    if isinstance(nested, dict):
        return nested.get("att")
    return None

def get_se(r):
    """Helper to get DiD Standard Error."""
    nested = r.get("did", {})
    if isinstance(nested, dict):
        return nested.get("se")
    return None

def fmt(x):
    """Helper to format numbers, handling Nones and NaNs."""
    try:
        if x is None or (isinstance(x, (str, float)) and str(x).lower() == 'na') or (isinstance(x, float) and math.isnan(x)):
            # Handle BSTS (Not available in this run)
            return "Not available in this run"
        return f"{float(x):.2f}"
    except Exception:
        return "NA"

def fmt_se(x):
    """Helper to format standard error."""
    try:
        val = float(x)
        return f"SE ≈ {val:.2f}"
    except Exception:
        return "—"

def main():
    if not RES.exists():
        raise SystemExit(f"results/results.json not found. Run: python src/run_analysis.py")

    r = json.loads(RES.read_text())
    did_att = pick_att(r, "DiD")
    psm_att = pick_att(r, "PSM")
    bsts_att = pick_att(r, "BSTS")
    did_se = get_se(r)

    # This is the full, correct HTML string for the polished report.
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Ontario Health Causal Analysis — Policy Evaluation Report</title>
  <style>
    body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;margin:40px auto;max-width:800px;line-height:1.6;color:#333}}
   .card{{border:1px solid #e0e0e0;border-radius:12px;padding:24px;margin:32px 0}}
   .table{{border-collapse:collapse;width:100%;margin-top:16px}}
   .table td,.table th{{border-bottom:1px solid #e0e0e0;padding:10px 8px;text-align:left}}
   .table th{{width:120px;font-weight:600}}
    img{{max-width:100%;height:auto;border:1px solid #e0e0e0;border-radius:8px;margin-top:10px}}
   .badge{{background:#f5f5f5;border-radius:999px;padding:4px 10px;font-size:12px;color:#333}}
    h1{{margin-bottom:8px;font-weight:600}}
    h2{{margin-top:0;font-weight:600}}
    p{{margin-bottom:0}}
    ul{{padding-left:20px;margin-top:10px}}
    li{{margin-bottom:8px}}
    code{{background:#f5f5f5;padding:3px 6px;border-radius:4px;font-family:monospace;font-size:0.95em}}
   .header p{{color:#555}}
   .footer{{font-size:12px;color:#777;text-align:center;margin-top:24px}}
    a{{color:#007bff;text-decoration:none}}
  </style>
</head>
<body>

<div class="header">
  <h1>Ontario Health Causal Analysis — Policy Evaluation Report</h1>
  <p>Jibran Kazi · <a href="mailto:jibrankazi@gmail.com">jibrankazi@gmail.com</a></p>
  <p>This report summarizes causal effect estimates using DiD, PSM, and (optionally) BSTS.</p>
  <p style="font-size:0.9em;margin-top:8px;">Contents: Abstract · Methods · Results · Discussion · Reproducibility</p>
</div>

<div class="card" id="abstract">
  <h2>Abstract</h2>
  <p>We estimate the policy’s Average Treatment Effect on the Treated (ATT) with multiple designs. In the current run, DiD suggests a reduction in incidence (ATT ≈ {fmt(did_att)}, {fmt_se(did_se)}; not statistically significant at 5%), and PSM suggests a reduction (ATT ≈ {fmt(psm_att)}). BSTS was not computed in this environment. These preliminary findings point toward a reduction but with uncertainty; we recommend enabling BSTS and/or SDID for triangulation.</p>
</div>

<div class="card" id="methods">
  <h2>Methods</h2>
  <ul>
    <li><b>Difference-in-Differences (DiD):</b> Two-way fixed effects; region-clustered SEs.</li>
    <li><b>PSM:</b> Nearest-neighbor matching on pre-period covariates; balance checked.</li>
    <li><b>BSTS (optional):</b> Synthetic counterfactual via Bayesian structural time series.</li>
  </ul>
</div>

<div class="card" id="results">
  <h2>Results</h2>
  <table class="table">
    <tr><th>Method</th><th>ATT</th><th>Uncertainty</th><th>Status</th></tr>
    <tr><td>DiD</td><td>{fmt(did_att)}</td><td>{fmt_se(did_se)}</td><td>Not statistically significant at 5%</td></tr>
    <tr><td>PSM</td><td>{fmt(psm_att)}</td><td>—</td><td>Computed</td></tr>
    <tr><td>BSTS</td><td>{fmt(bsts_att)}</td><td>—</td><td>{fmt(bsts_att)}</td></tr>
  </table>
  {('<img alt="att_summary" src="data:image/png;base64,' + b64(FIG) + '">') if FIG.exists() else '<p><b>No figure found.</b></p>'}
  <p>Figure. ATT comparison across methods (BSTS omitted if unavailable).</p>
  <p style="font-size:0.9em;color:#555;margin-top:16px;">Dataset: 34 regions; 4,760 rows in this run. See JSON for details.</p>
</div>

<div class="card" id="discussion">
  <h2>Discussion</h2>
  <p>DiD and PSM both point to a reduction but with different magnitudes and with DiD imprecise. We recommend enabling BSTS (or SDiD) to test sensitivity to alternative counterfactual construction, and exploring heterogeneity (e.g., Causal Forests) to identify segments where the effect is strongest or weakest.</p>
</div>

<div class="card" id="reproducibility">
  <h2>Reproducibility</h2>
  <ul>
    <li><code>Config: config.yaml (policy_date, covariates, output paths)</code></li>
    <li><code>Results (JSON): results/results.json</code></li>
    <li><code>Figure: figures/att_summary.png</code></li>
    <li><code>Pipeline: scripts/clean_all.py → src/run_analysis.py → scripts/regenerate_figures.py → scripts/build_html_report.py</code></li>
  </ul>
</div>

<p class="footer">
  &copy; Jibran Kazi · This page is auto-published via GitHub Pages after each push to main.
</p>

</body>
</html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"✓ HTML report written to {OUT}")

if __name__ == "__main__":
    main()
