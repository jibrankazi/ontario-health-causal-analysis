#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "results.json"
FIG = ROOT / "figures"

def main():
    if not RES.exists():
        raise SystemExit("results/results.json not found. Run: python src/run_analysis.py")

    res = json.loads(RES.read_text())
    did  = res.get("did", {}).get("att")
    psm  = res.get("psm", {}).get("att")
    bsts = res.get("bsts", {}).get("att")

    s = pd.Series({"DiD": did, "PSM": psm, "BSTS": bsts})

    FIG.mkdir(parents=True, exist_ok=True)
    ax = s.plot(kind="bar")
    ax.set_title("Estimated Treatment Effects (ATT)")
    ax.set_ylabel("ATT")
    for i, v in enumerate(s.values):
        label = "NA" if v is None else f"{v:.2f}"
        ax.text(i, (v if v is not None else 0), label, ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG / "att_summary.png")
    plt.close()
    print("✓ Figures saved to figures/att_summary.png")

if __name__ == "__main__":
    main()
