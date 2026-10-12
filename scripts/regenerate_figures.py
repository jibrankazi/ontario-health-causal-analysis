"""Regenerate an observed-data figure, without inventing missing model results.

These are legacy PHU model associations, not established policy effects.
In particular, the dataset's publisher provenance, treatment assignment,
and incidence-unit definition have not been independently reconciled.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "results.json"
FIG = ROOT / "figures"


def available_estimates(results):
    """Return only finite, actually reported estimates (never replace null with zero)."""
    estimates = {}
    for key, label in (("did", "DiD"), ("psm", "PSM"), ("bsts", "BSTS")):
        entry = results.get(key, {})
        value = entry.get("att") if isinstance(entry, dict) else results.get(f"{key}_att")
        if value is None:
            continue
        try:
            value = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{label} is not a valid numeric estimate") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} is not a finite estimate")
        estimates[label] = value
    if not estimates:
        raise ValueError("No finite model estimates in actual results JSON")
    return estimates


def main():
    if not RES.is_file():
        raise SystemExit("results/results.json not found. Run: python src/run_analysis.py")
    results = json.loads(RES.read_text(encoding="utf-8"))
    estimates = available_estimates(results)

    fig, ax = plt.subplots(figsize=(8, 5.4))
    labels = list(estimates)
    values = list(estimates.values())
    bars = ax.bar(labels, values, color="#4472a4")
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_title("Exploratory historical PHU associations — NOT causal effects")
    ax.set_ylabel("Recorded incidence units (definition unverified)")
    ax.margins(y=0.25)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:+.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5 if value >= 0 else -5),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    fig.text(
        0.5, 0.03,
        "Checked-in PHU data and treated/comparison assignment not independently "
        "verified against Ontario publisher records. No policy effect established.",
        ha="center", va="center", fontsize=8, wrap=True,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    output = FIG / "att_summary.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Figure generation produced no image")
    print(f"Generated {output} using {len(estimates)} available estimates; no null imputation.")
    return output


if __name__ == "__main__":
    main()
