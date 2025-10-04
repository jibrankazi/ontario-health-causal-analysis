#!/usr/bin/env python3
import json
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "results.json"
FIG = ROOT / "figures"

def to_num(x):
    try:
        f = float(x)
        return None if math.isnan(f) else f
    except Exception:
        return None

def main():
    if not RES.exists():
        raise SystemExit("results/results.json not found. Run: python src/run_analysis.py")

    r = json.loads(RES.read_text())
    did  = to_num(r.get("did_att"))
    psm  = to_num(r.get("psm_att"))
    bsts = to_num(r.get("bsts_att"))

    # Build a numeric series. If a value is None, plot 0 and label as NA.
    labels = ["DiD", "PSM", "BSTS"]
    values = [0 if v is None else v for v in [did, psm, bsts]]

    FIG.mkdir(parents=True, exist_ok=True)
    s = pd.Series(values, index=labels, dtype="float")
    ax = s.plot(kind="bar")
    ax.set_title("Estimated Treatment Effects (ATT)")
    ax.set_ylabel("ATT")

    shown = [did, psm, bsts]
    for i, v in enumerate(shown):
        ax.text(i, 0 if v is None else v, "NA" if v is None else f"{v:.2f}",
                ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG / "att_summary.png")
    plt.close()
    print("✓ Figure saved: figures/att_summary.png")

if __name__ == "__main__":
    main()
