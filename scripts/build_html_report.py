#!/usr/bin/env python3
from pathlib import Path
import json, base64
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "results.json"
FIG = ROOT / "figures" / "att_summary.png"
OUT = ROOT / "reports" / "report.html"

def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii") if p.exists() else ""

def main():
    if not RES.exists():
        raise SystemExit("results/results.json not found. Run: python src/run_analysis.py")

    data = json.loads(RES.read_text())
    did  = data.get("did", {}).get("att")
    psm  = data.get("psm", {}).get("att")
    bsts = data.get("bsts", {}).get("att")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Causal Analysis Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111}}
h1,h2{{margin:0 0 12px}} .card{{border:1px solid #eee;border-radius:12px;padding:16px;margin:16px 0}}
.table{{border-collapse:collapse;width:100%}} .table td,.table th{{border-bottom:1px solid #eee;padding:8px 6px;text-align:left}}
img{{max-width:100%;height:auto;border:1px solid #eee;border-radius:8px}}
.badge{{background:#f5f5f5;border-radius:999px;padding:4px 10px;font-size:12px}}
</style></head>
<body>
  <h1>Ontario Health — Causal Analysis</h1>
  <div class="badge">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>

  <div class="card">
    <h2>ATT Summary</h2>
    <table class="table">
      <tr><th>Method</th><th>ATT</th></tr>
      <tr><td>DiD</td><td>{'NA' if did is None else did}</td></tr>
      <tr><td>PSM</td><td>{'NA' if psm is None else psm}</td></tr>
      <tr><td>BSTS</td><td>{'NA' if bsts is None else bsts}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Figure</h2>
    {('<img alt="att_summary" src="data:image/png;base64,' + b64(FIG) + '">') if FIG.exists() else '<p>No figure found.</p>'}
  </div>
</body></html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"✓ HTML report written to {OUT}")

if __name__ == "__main__":
    main()
