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

def fmt(x):
    """Helper to format numbers, handling Nones and NaNs."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            # Handle BSTS (Not available in this run)
            return "Not available in this run"
        return f"{float(x):.2f}"
    except Exception:
        return "NA"

def main():
    if not RES.exists():
        raise SystemExit(f"results/results.json not found. Run: python src/run_analysis.py")

    r = json.loads(RES.read_text())
    did = pick_att(r, "DiD")
    psm = pick_att(r, "PSM")
    bsts = pick_att(r, "BSTS")

    # This is the full HTML string, with all corrections applied.
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Causal Analysis Report</title>
  <style>
    body{{font-family:Segoe UI,Arial,sans-serif;margin:24px auto;max-width:800px;line-height:1.6}}
  .card{{border:1px solid #eee;border-radius:12px;padding:16px 24px;margin:24px 0}}
  .table{{border-collapse:collapse;width:100%}}
  .table td,.table th{{border-bottom:1px solid #eee;padding:8px 6px;text-align:left}}
  .table th{{width:80px}}
    img{{max-width:100%;height:auto;border:1px solid #eee;border-radius:8px;margin-top:10px}}
  .badge{{background:#f5f5f5;border-radius:999px;padding:4px 10px;font-size:12px;color:#333}}
    h1{{margin-bottom:8px}}
    h2{{margin-top:0}}
    p{{margin-bottom:0}}
    ul{{padding-left:20px}}
    li{{margin-bottom:8px}}
    code{{background:#f5f5f5;padding:3px 6px;border-radius:4px;font-family:monospace;font-size:0.95em}}
  </style>
</head>
<body>

<h1>Ontario Health — Causal Analysis</h1>
<div class="badge">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>

<div class="card">
  <h2>ATT Summary</h2>
  <p>Dataset: 34 regions; 4,760 rows in this run. See JSON for details.</p>
  <table class="table">
    <tr><th>Method</th><th>ATT</th></tr>
    <tr><td>DiD</td><td>{fmt(did)}</td></tr>
    <tr><td>PSM</td><td>{fmt(psm)}</td></tr>
    <tr><td>BSTS</td><td>{fmt(bsts)}</td></tr>
  </table>
</div>

<div class="card">
  <h2>Figure</h2>
  {/* */}
  {('<img alt="att_summary" src="data:image/png;base64,' + b64(FIG) + '">') if FIG.exists() else '<p><b>No figure found.</b></p>'}
  {/* */}
  <p>Figure. ATT comparison across methods (BSTS omitted if unavailable).</p>
</div>

<div class="card" id="discussion">
  <h2>Discussion</h2>
  <p>DiD and PSM both point to a reduction but with different magnitudes and with DiD imprecise. We recommend enabling BSTS (or SDiD) to test sensitivity to alternative counterfactual construction, and exploring heterogeneity (e.g., Causal Forests) to identify segments where the effect is strongest or weakest.</p>
</div>

<div class="card" id="reproducibility">
  <h2>Reproducibility</h2>
  <ul>
    <li><code>Config: config.yaml (policy_date, covariates, output paths)</code></li>
    {/* */}
    <li><code>Results (JSON):./results/results.json</code></li>
    {/* */}
    <li><code>Figure:./figures/att_summary.png</code></li>
    <li><code>Pipeline: scripts/clean_all.py -- src/run_analysis.py -- scripts/regenerate_figures.py -- scripts/build_html_report.py</code></li>
  </ul>
</div>

<hr style="border:0;border-top:1px solid #f0f0f0;margin:24px 0;">
<p style="font-size:12px;color:#777;text-align:center;">
  &copy; Jibran Kazi - This page is auto-published via GitHub Pages after each push to main.
</p>

</body>
</html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"✓ HTML report written to {OUT}")

if __name__ == "__main__":
    main()
