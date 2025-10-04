#!/usr/bin/env python3
import json
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "results.json"
FIG = ROOT / "figures"

def pick_att(r, method):
    """
    Supports both schemas:
      - flat:   did_att / psm_att / bsts_att
      - nested: {"did":{"att":...}}, {"psm":{"att":...}}, {"bsts":{"att":...}}
    """
    flat_key = f"{method.lower()}_att"
    if flat_key in r:
        return r.get(flat_key)
    nested = r.get(method.lower(), {})
    if isinstance(nested, dict):
        return nested.get("att")
    return None

def to_num(x):
    try:
        return float(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else None
    except Exception:
        return None

def main():
    if not RES.exists():
        raise SystemExit("results/results.json not found. Run: python src/run_analysis.py")

    r = json.loads(RES.read_text())

    did  = to_num(pick_att(r, "DiD"))
    psm  = to_num(pick_att(r, "PSM"))
    bsts = to_num(pick_att(r, "BSTS"))

    s = pd.Series({"DiD": did, "PSM": psm, "BSTS": bsts}, dtype="float")

    # If all are None, at least render NA bars at zero height
    plot_vals = [0 if v is None else v for v in s.values]

    FIG.mkdir(parents=True, exist_ok=True)
    ax = pd.Series(plot_vals, index=s.index).plot(kind="bar")
    ax.set_title("Estimated Treatment Effects (ATT)")
    ax.set_ylabel("ATT")
    for i, v in enumerate(s.values):
        label = "NA" if v is None else f"{v:.2f}"
        ax.text(i, 0 if v is None else v, label, ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG / "att_summary.png")
    plt.close()
    print("✓ Figure saved: figures/att_summary.png")

if __name__ == "__main__":
    main()
