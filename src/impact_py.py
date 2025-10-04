# src/impact_py.py
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from causalimpact import CausalImpact
import matplotlib.pyplot as plt

@dataclass
class ImpactResult:
    att: Optional[float]
    ci: Tuple[Optional[float], Optional[float]]
    p: Optional[float]
    relative_effect: Optional[float]
    timeline: List[Dict[str, Any]] = None  # Optional, can be empty
    notes: Optional[str] = None
    plot_path: Optional[str] = None
    summary_path: Optional[str] = None

def estimate_causal_impact_python(panel_df: pd.DataFrame, policy_date: pd.Timestamp) -> ImpactResult:
    """
    Python implementation of CausalImpact, mirroring the R script logic.
    Performs matching, control selection, fitting, and extraction.
    """
    # Config paths (relative to project root)
    fig_dir = Path("figures")
    res_dir = Path("results")
    fig_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)
    plot_path = fig_dir / "fig_causalimpact.png"
    summary_txt = res_dir / "causalimpact_summary.txt"

    # --- Matching on pre-policy baseline ---
    pre_df = panel_df[panel_df['week'] < policy_date]
    if len(pre_df) == 0:
        raise ValueError(f"No pre-policy rows before {policy_date}")

    pre_baseline = pre_df.groupby('region').agg(
        mean_incidence=('incidence', 'mean'),
        treated=('treated', 'first')
    ).reset_index()

    treated = pre_baseline[pre_baseline['treated'] == 1]
    controls = pre_baseline[pre_baseline['treated'] == 0]
    if len(treated) == 0 or len(controls) == 0:
        raise ValueError("Need both treated and control regions pre-policy.")

    n_treat = len(treated)
    n_ctrl = len(controls)
    match_ratio = min(5, n_ctrl // max(1, n_treat))

    print(f"Running matching with ratio {match_ratio}:1")

    nn = NearestNeighbors(n_neighbors=match_ratio)
    nn.fit(controls[['mean_incidence']])
    dist, idx = nn.kneighbors(treated[['mean_incidence']])
    matched_control_regions = controls.iloc[idx.flatten()]['region'].tolist()
    treated_regions = treated['region'].tolist()
    control_regions = list(set(matched_control_regions))

    print(f"Matched treated regions: {', '.join(map(str, treated_regions))}")
    print(f"Matched control regions: {', '.join(map(str, control_regions))}")

    # --- Prepare wide data ---
    all_weeks = pd.date_range(start=panel_df['week'].min(), end=panel_df['week'].max(), freq='W-MON')  # Weekly Mondays

    treated_agg = panel_df[panel_df['region'].isin(treated_regions)].groupby('week')['incidence'].mean().rename('y')

    control_wide = panel_df[
        panel_df['region'].isin(control_regions)
    ].pivot(index='week', columns='region', values='incidence')
    control_wide.columns = [f"x_{col}" for col in control_wide.columns]

    df_ci = pd.DataFrame({'week': all_weeks})
    df_ci = df_ci.merge(treated_agg, on='week', how='left')
    df_ci = df_ci.merge(control_wide, on='week', how='left')
    df_ci = df_ci.set_index('week').fillna(method='ffill').fillna(method='bfill')

    if df_ci.index.min() >= policy_date or df_ci.index.max() <= policy_date:
        raise ValueError("Series does not straddle policy_date; fix policy_date or data.")

    # --- Diagnostics: sample sizes & pre-period correlation ---
    pre_mask = df_ci.index < policy_date
    n_pre = pre_mask.sum()
    n_post = (~pre_mask).sum()
    print(f"Pre points: {n_pre} | Post points: {n_post}")

    y_pre = df_ci.loc[pre_mask, 'y']
    x_cols = [col for col in df_ci.columns if col.startswith('x_')]
    if len(x_cols) > 0 and len(y_pre) > 1:
        X_pre = df_ci.loc[pre_mask, x_cols]
        cors = X_pre.corrwith(y_pre).sort_values(ascending=False)
        print("Top 10 pre-period correlations with Y:")
        print(cors.head(10))

        abs_cors = cors.abs()
        keep = abs_cors[abs_cors >= 0.25].index.tolist()
        if len(keep) < 2 and len(cors) >= 2:
            keep = abs_cors.nlargest(2).index.tolist()
        elif len(keep) == 0 and len(cors) > 0:
            keep = [cors.index[0]]

        if len(keep) > 0:
            df_ci = df_ci[['y'] + keep]
            print(f"Selected controls: {keep}")
        else:
            print("No controls selected.")
    else:
        print("No control data or insufficient pre-period data for correlation analysis.")

    # Final cleanup
    df_ci = df_ci.dropna()

    # Periods
    pre_period = [df_ci.index.min(), policy_date - pd.Timedelta(days=7)]
    post_period = [policy_date, df_ci.index.max()]

    # --- Fit CausalImpact ---
    try:
        ci = CausalImpact(df_ci, pre_period, post_period)
    except Exception as e:
        raise RuntimeError(f"CausalImpact fitting failed: {e}")

    # --- Save plot and summary ---
    fig = ci.plot()
    fig.figure.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig.figure)

    report = ci.summary('report')
    with open(summary_txt, 'w') as f:
        f.write(report)

    print(f"\nSaved:\n  {summary_txt}\n  {plot_path}\n")

    # --- Extract results by parsing summary ---
    summary_str = ci.summary()
    lines = [line.strip() for line in summary_str.split('\n') if line.strip()]

    # Find absolute effect line
    abs_idx = next((i for i, line in enumerate(lines) if 'Absolute effect (s.d.)' in line), None)
    if abs_idx is None:
        raise ValueError("Could not parse absolute effect from summary")

    # ATT from Average column
    abs_line = lines[abs_idx]
    att_match = re.search(r'Average\s+(-?\d+(?:\.\d+)?)', abs_line)
    att = float(att_match.group(1)) if att_match else np.nan

    # CI from next line
    ci_line = lines[abs_idx + 1]
    ci_match = re.search(r'\[(-?[\d.]+),\s*(-?[\d.]+)\]', ci_line)
    lo = float(ci_match.group(1)) if ci_match else np.nan
    hi = float(ci_match.group(2)) if ci_match else np.nan

    # Relative effect
    rel_idx = next((i for i, line in enumerate(lines) if 'Relative effect (s.d.)' in line), None)
    if rel_idx:
        rel_line = lines[rel_idx]
        rel_match = re.search(r'Average\s+(-?[\d.]+)%', rel_line)
        rel_pct = float(rel_match.group(1)) if rel_match else np.nan
        relative_effect = rel_pct / 100 if not np.isnan(rel_pct) else np.nan
    else:
        relative_effect = np.nan

    # P-value
    p_match = re.search(r'Posterior tail-area probability p:\s*([\d.]+)', summary_str)
    p = float(p_match.group(1)) if p_match else np.nan

    # Fallback for ATT if parsing failed
    actual_match = re.search(r'Actual\s+(-?[\d.]+)', summary_str)
    pred_match = re.search(r'Prediction \(s\.d\.\)\s+(-?[\d.]+)', summary_str)
    if np.isnan(att) and actual_match and pred_match:
        att = float(actual_match.group(1)) - float(pred_match.group(1))

    return ImpactResult(
        att=att if not np.isnan(att) else None,
        ci=(lo if not np.isnan(lo) else None, hi if not np.isnan(hi) else None),
        p=p if not np.isnan(p) else None,
        relative_effect=relative_effect if not np.isnan(relative_effect) else None,
        timeline=[],  # Not implemented
        notes=None,
        plot_path=str(plot_path),
        summary_path=str(summary_txt)
    )
