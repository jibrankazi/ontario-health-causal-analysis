#!/usr/bin/env python3
"""
Main analysis pipeline script with all causal inference methods.
Fixed version with resolved duplicates and logic errors.
"""
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Set up paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from figures import generate_analysis_figures  # noqa: E402
from shared import ROOT as PROJECT_ROOT, load_config, resolve_intervention_date  # noqa: E402

# ---------------------------
# Determinism
# ---------------------------
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)

# ---------------------------
# Config / Paths
# ---------------------------
cfg: dict[str, Any] = load_config()
data_path = PROJECT_ROOT / cfg.get("data_path", "data/ontario_cases.csv")
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv(data_path)

# Expected columns
required = {"week", "region", "incidence", "treated"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Parse dates
if pd.api.types.is_string_dtype(df["week"]):
    df["week"] = pd.to_datetime(df["week"], errors="coerce")

# Construct post if absent
policy_date = resolve_intervention_date(cfg)
if "post" not in df.columns:
    df["post"] = (df["week"] >= policy_date).astype(int)
else:
    inferred = pd.Timestamp(df.loc[df["post"] == 1, "week"].min())
    if not pd.isna(inferred):
        policy_date = inferred

# ---------------------------
# Difference-in-Differences
# ---------------------------
df["treat_post"] = df["treated"] * df["post"]
try:
    did = smf.ols("incidence ~ C(region) + C(week) + treat_post", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["region"]}
    )
    did_att = float(did.params.get("treat_post", float("nan")))
    did_se = float(did.bse.get("treat_post", float("nan")))
except Exception as exc:
    print(f"WARNING: DiD estimation failed: {exc}")
    did_att = float("nan")
    did_se = float("nan")

# ---------------------------
# Propensity Score Matching (robust)
# ---------------------------
psm_att: float | None = None
psm_reason: str | None = None
psm_diag: dict[str, Any] = {}
matched_treated_df: pd.DataFrame | None = None
matched_control_df: pd.DataFrame | None = None

pre = df[df["post"] == 0].copy()
drop_cols = {"week", "region", "incidence", "treated", "post", "treat_post"}
covars = [c for c in pre.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(pre[c])]
if not covars:
    covars = ["incidence"]

try:
    n_treat = int(pre["treated"].sum())
    n_ctrl = int((1 - pre["treated"]).sum())
    psm_diag.update(n_treat_pre=n_treat, n_ctrl_pre=n_ctrl, covars=covars)

    if n_treat == 0 or n_ctrl == 0:
        psm_reason = "No treated or control units in pre-period; skipping PSM."
        raise RuntimeError(psm_reason)

    X = pre[covars].fillna(pre[covars].median(numeric_only=True)).to_numpy()
    y = pre["treated"].to_numpy()

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X, y)
    pre["ps"] = lr.predict_proba(X)[:, 1]

    # Common support
    ps_t = pre.loc[pre["treated"] == 1, "ps"]
    ps_c = pre.loc[pre["treated"] == 0, "ps"]
    overlap_low = max(ps_t.min(), ps_c.min())
    overlap_high = min(ps_t.max(), ps_c.max())
    psm_diag.update(ps_overlap_low=float(overlap_low), ps_overlap_high=float(overlap_high))

    if overlap_low >= overlap_high:
        psm_reason = "No common support in propensity scores; skipping PSM."
        raise RuntimeError(psm_reason)

    # Restrict to support
    pre_cs = pre[pre["ps"].between(overlap_low, overlap_high, inclusive="both")].copy()
    n_treat_cs = int(pre_cs["treated"].sum())
    n_ctrl_cs = int((1 - pre_cs["treated"]).sum())
    psm_diag.update(n_treat_pre_cs=n_treat_cs, n_ctrl_pre_cs=n_ctrl_cs)

    if n_treat_cs == 0 or n_ctrl_cs == 0:
        psm_reason = "Common-support filter removed all treated or control units; skipping PSM."
        raise RuntimeError(psm_reason)

    # Caliper based on 0.2 * SD(logit(ps))
    eps = 1e-6
    ps_vals = pre_cs["ps"].clip(eps, 1 - eps)
    logit = np.log(ps_vals) - np.log1p(-ps_vals)
    caliper = 0.2 * float(np.nanstd(logit))
    caliper_limit_ps = caliper * 0.25
    psm_diag.update(caliper=caliper, caliper_ps_limit=float(caliper_limit_ps))

    # 1-NN with replacement on controls
    controls = pre_cs[pre_cs["treated"] == 0].copy()
    treats = pre_cs[pre_cs["treated"] == 1].copy()

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(controls[["ps"]].to_numpy())
    dist, idx = nn.kneighbors(treats[["ps"]].to_numpy())
    dist = dist.flatten()
    idx = idx.flatten()

    # Match pairs within caliper
    ps_pairs: list[tuple[pd.Series, pd.Series]] = []
    matched_treated_rows: list[pd.Series] = []
    matched_control_rows: list[pd.Series] = []
    
    for i, j in enumerate(idx):
        if dist[i] <= caliper_limit_ps:
            treated_row = treats.iloc[i]
            control_row = controls.iloc[j]
            ps_pairs.append((treated_row, control_row))
            matched_treated_rows.append(treated_row)
            matched_control_rows.append(control_row)

    psm_diag["n_matched"] = len(ps_pairs)
    if not ps_pairs:
        psm_reason = "No matches within caliper; skipping PSM."
        raise RuntimeError(psm_reason)

    # Post-period ATT using matched sets
    if matched_treated_rows and matched_control_rows:
        matched_treated_df = pd.DataFrame(matched_treated_rows)
        matched_control_df = pd.DataFrame(matched_control_rows)

    post = df[df["post"] == 1].copy()
    post_mean = post.groupby("region", as_index=True)["incidence"].mean()
    diffs: list[float] = []
    
    for treated_row, control_row in ps_pairs:
        t_reg = treated_row["region"]
        c_reg = control_row["region"]
        if t_reg in post_mean.index and c_reg in post_mean.index:
            diffs.append(float(post_mean.loc[t_reg] - post_mean.loc[c_reg]))

    if not diffs:
        psm_reason = "Matched regions missing from post-period; skipping PSM."
        raise RuntimeError(psm_reason)

    psm_att = float(np.mean(diffs))
    
except Exception as exc:
    if not psm_reason:
        psm_reason = f"PSM failed due to an unexpected error: {exc}"

# ---------------------------
# Python SARIMAX counterfactual
# ---------------------------
impact_att: float | None = None
impact_ci: tuple[float | None, float | None] = (None, None)
impact_reason: str | None = None
impact_series: list[Mapping[str, Any]] | None = None


def _estimate_impact_sarimax(panel: pd.DataFrame, policy_date: pd.Timestamp):
    """Estimate causal impact using SARIMAX counterfactual."""
    weekly = panel.copy()
    weekly["week"] = pd.to_datetime(weekly["week"], errors="coerce")
    weekly = weekly.dropna(subset=["week"])

    treated = (
        weekly[weekly["treated"] == 1]
        .groupby("week", as_index=False)["incidence"]
        .mean()
        .rename(columns={"incidence": "t"})
    )
    control = (
        weekly[weekly["treated"] == 0]
        .groupby("week", as_index=False)["incidence"]
        .mean()
        .rename(columns={"incidence": "c"})
    )

    series = pd.merge(treated, control, on="week", how="inner").dropna()
    series = series.sort_values("week").reset_index(drop=True)
    if series.empty:
        raise RuntimeError("No overlapping treated/control weeks to form a difference series.")

    series["y"] = series["t"] - series["c"]

    pre = series[series["week"] < policy_date].copy()
    post = series[series["week"] >= policy_date].copy()
    if pre.empty or post.empty:
        raise RuntimeError("Pre or post period empty; check intervention_date and data coverage.")

    y_pre = pre["y"].to_numpy(dtype=float)
    if len(y_pre) < 8:
        raise RuntimeError("Insufficient pre-period observations for SARIMAX fit (need >= 8 weeks).")

    best: tuple[float, tuple[int, int, int], Any] | None = None
    candidates = [(1, 0, 0), (0, 1, 1), (1, 1, 0), (0, 1, 0), (1, 1, 1)]
    for order in candidates:
        try:
            model = SARIMAX(
                y_pre,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False)
            aic = float(result.aic)
        except Exception:
            continue
        if best is None or aic < best[0]:
            best = (aic, order, result)

    if best is None:
        raise RuntimeError("SARIMAX failed to converge for all candidate orders.")

    _, order, fitted = best
    horizon = len(post)
    forecast = fitted.get_forecast(steps=horizon)
    predicted = np.asarray(forecast.predicted_mean, dtype=float)
    conf_int = forecast.conf_int(alpha=0.05)
    conf_arr = np.asarray(conf_int, dtype=float)

    lower = conf_arr[:, 0]
    upper = conf_arr[:, -1]
    actual_post = post["y"].to_numpy(dtype=float)

    effect = actual_post - predicted
    att = float(np.mean(effect))

    eff_lo = actual_post - upper
    eff_hi = actual_post - lower
    ci_lo = float(np.mean(eff_lo))
    ci_hi = float(np.mean(eff_hi))

    rows: list[Mapping[str, Any]] = []
    for i, week in enumerate(post["week"].to_list()):
        rows.append(
            {
                "date": pd.Timestamp(week).date().isoformat(),
                "actual": float(actual_post[i]),
                "predicted": float(predicted[i]),
                "lower": float(lower[i]),
                "upper": float(upper[i]),
                "effect": float(effect[i]),
            }
        )

    return att, (ci_lo, ci_hi), rows


try:
    impact_att, impact_ci, impact_series = _estimate_impact_sarimax(df, policy_date)
except Exception as exc:
    impact_reason = f"Python SARIMAX impact failed: {exc}"

# ---------------------------
# Generate Figures
# ---------------------------
artifacts = generate_analysis_figures(
    panel=df,
    policy_date=policy_date,
    pre_period=pre,
    covariates=covars,
    matched_treated=matched_treated_df,
    matched_control=matched_control_df,
    impact_series=impact_series,
    root=PROJECT_ROOT,
)

# ---------------------------
# Save & Print Results
# ---------------------------
impact_ci_payload: list[float | None]
if all(v is not None for v in impact_ci):
    impact_ci_payload = [float(impact_ci[0]), float(impact_ci[1])]
else:
    impact_ci_payload = [None, None]

out = {
    "did_att": did_att,
    "did_se": did_se,
    "psm_att": None
    if psm_att is None or (isinstance(psm_att, float) and np.isnan(psm_att))
    else float(psm_att),
    "impact_att": impact_att,
    "impact_ci": impact_ci_payload,
    "meta": {
        "psm_reason": psm_reason,
        "psm_diagnostics": psm_diag,
        "impact_reason": impact_reason,
        "impact_method": "SARIMAX(treated_minus_control)",
        "impact_series": impact_series,
        "figures": {
            "event_study": artifacts.event_study,
            "balance": artifacts.balance,
            "impact": artifacts.impact,
        },
        "n_rows": int(len(df)),
        "n_regions": int(df["region"].nunique()),
    },
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

(results_dir / "results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
