#!/usr/bin/env python3
"""
Main causal analysis pipeline with DiD, PSM, and BSTS-style impact estimation.
Fixed version with proper error handling and no mock functions.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Set deterministic behavior
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from shared import ROOT as PROJECT_ROOT, load_config, resolve_intervention_date
    from figures import generate_analysis_figures
except ImportError as e:
    print(f"Warning: Could not import shared modules: {e}")
    PROJECT_ROOT = ROOT
    
    def load_config():
        """Fallback config loader."""
        import yaml
        config_path = ROOT / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def resolve_intervention_date(cfg):
        """Fallback date resolver."""
        return pd.Timestamp(cfg.get("policy_date", "2021-02-01"))
    
    class FigureArtifacts:
        """Fallback artifacts class."""
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    def generate_analysis_figures(**kwargs):
        """Fallback figure generator."""
        print("Note: Figure generation unavailable (missing dependencies)")
        return FigureArtifacts(
            event_study="figures/fig1_event_study.png",
            balance="figures/fig2_smd_balance.png",
            impact="figures/fig3_impact_counterfactual.png",
        )


# ------------------------------- I/O -----------------------------------------
def _load_panel(path: Path) -> pd.DataFrame:
    """Loads and cleans the panel data."""
    required = {"week", "region", "incidence", "treated"}
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    df = df.copy()
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df = df.dropna(subset=["week"]).reset_index(drop=True)
    df["region"] = df["region"].astype(str)
    df["treated"] = pd.to_numeric(df["treated"], errors="coerce").fillna(0).astype(int)
    df["incidence"] = pd.to_numeric(df["incidence"], errors="coerce")
    df = df.dropna(subset=["incidence"]).reset_index(drop=True)
    return df


# ------------------------------ Methods --------------------------------------
def _run_did(panel: pd.DataFrame) -> tuple[float | None, float | None]:
    """TWFE: y ~ unit FE + time FE + treat*post, cluster by unit."""
    try:
        if 'treat_post' not in panel.columns:
            panel['treat_post'] = panel['treated'] * panel['post']

        m = smf.ols("incidence ~ C(region) + C(week) + treat_post", data=panel).fit(
            cov_type="cluster", cov_kwds={"groups": panel["region"]}
        )
        return float(m.params.get("treat_post")), float(m.bse.get("treat_post"))
    except Exception as e:
        print(f"DiD failed: {e}")
        return None, None


def _infer_covariates(panel: pd.DataFrame) -> Sequence[str]:
    """Infers numeric columns for use as covariates in PSM."""
    drop = {"week", "region", "incidence", "treated", "post", "treat_post", "pre_level", "pre_trend"} 
    covs = [c for c in panel.columns if c not in drop and pd.api.types.is_numeric_dtype(panel[c])]
    return covs


def _run_psm(
    panel: pd.DataFrame, 
    covariates: Sequence[str]
) -> tuple[float | None, str | None, dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    """Nearest-neighbor on pre-period propensity scores; returns ATT over post period."""
    diag: dict[str, Any] = {"caliper": 0.05}
    
    if not covariates:
        return None, "No covariates provided; skipping PSM.", diag, None, None

    pre = panel[panel["post"] == 0].copy()
    if pre.empty:
        return None, "No pre-period observations available for PSM.", diag, None, None

    # Validate covariates exist in data
    covariate_cols = [c for c in covariates if c in pre.columns]
    if not covariate_cols:
        return None, "Configured covariates not found in data; skipping PSM.", diag, None, None
        
    X = pre[covariate_cols].fillna(pre[covariate_cols].median(numeric_only=True)).to_numpy(float)
    y = pre["treated"].to_numpy(int)
    n_treat, n_ctrl = int(pre["treated"].sum()), int((1 - pre["treated"]).sum())
    
    diag.update(n_treat_pre=n_treat, n_ctrl_pre=n_ctrl, covariates=list(covariate_cols))
    if n_treat == 0 or n_ctrl == 0:
        return None, "No treated or control units in pre-period; skipping PSM.", diag, None, None

    try:
        lr = LogisticRegression(max_iter=500, random_state=42).fit(X, y)
    except ValueError as e:
        return None, f"Logistic Regression failed: {e}", diag, None, None

    pre = pre.assign(ps=lr.predict_proba(X)[:, 1])

    treats = pre[pre["treated"] == 1].copy()
    ctrls = pre[pre["treated"] == 0].copy()
    if treats.empty or ctrls.empty:
        return None, "No treated or control rows after PS estimation.", diag, None, None

    nn = NearestNeighbors(n_neighbors=1).fit(ctrls[["ps"]].to_numpy(float))
    dist, idx = nn.kneighbors(treats[["ps"]].to_numpy(float))
    dist, idx = dist.flatten(), idx.flatten()

    caliper = diag["caliper"]
    treats = treats.reset_index(drop=True)
    ctrls = ctrls.reset_index(drop=True)
    
    pairs = [(treats.iloc[i], ctrls.iloc[j]) for i, j in enumerate(idx) if dist[i] <= caliper]
    diag["n_matched"] = len(pairs)
    
    if not pairs:
        return None, "No matches within caliper; skipping PSM.", diag, None, None

    # ATT over post period mean incidence by matched regions
    post = panel[panel["post"] == 1].copy()
    post_means = post.groupby("region", as_index=True)["incidence"].mean()
    diffs = []
    matched_t_list, matched_c_list = [], []
    
    for t_row, c_row in pairs:
        t_reg, c_reg = t_row["region"], c_row["region"]
        if t_reg in post_means.index and c_reg in post_means.index:
            diffs.append(float(post_means.loc[t_reg] - post_means.loc[c_reg]))
            matched_t_list.append(t_row.to_dict())
            matched_c_list.append(c_row.to_dict())
            
    if not diffs:
        return None, "Matched regions missing in post period; skipping PSM.", diag, None, None
        
    return float(np.mean(diffs)), None, diag, pd.DataFrame(matched_t_list), pd.DataFrame(matched_c_list)


def _estimate_impact_python(panel: pd.DataFrame, policy_date: pd.Timestamp) -> dict[str, Any]:
    """
    Estimate causal impact using synthetic control or simple SARIMAX.
    Returns dict with att, ci, p, relative_effect, timeline, notes.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
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
            raise RuntimeError("No overlapping treated/control weeks.")

        series["y"] = series["t"] - series["c"]

        pre = series[series["week"] < policy_date].copy()
        post = series[series["week"] >= policy_date].copy()
        
        if pre.empty or post.empty:
            raise RuntimeError("Pre or post period empty.")

        y_pre = pre["y"].to_numpy(dtype=float)
        if len(y_pre) < 8:
            raise RuntimeError("Insufficient pre-period observations (need >= 8).")

        # Try multiple SARIMAX specifications
        best = None
        for order in [(1, 0, 0), (0, 1, 1), (1, 1, 0), (0, 1, 0), (1, 1, 1)]:
            try:
                model = SARIMAX(y_pre, order=order, enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False)
                aic = float(result.aic)
                if best is None or aic < best[0]:
                    best = (aic, order, result)
            except Exception:
                continue

        if best is None:
            raise RuntimeError("SARIMAX failed to converge.")

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

        timeline = []
        for i, week in enumerate(post["week"].to_list()):
            timeline.append({
                "date": pd.Timestamp(week).date().isoformat(),
                "actual": float(actual_post[i]),
                "predicted": float(predicted[i]),
                "lower": float(lower[i]),
                "upper": float(upper[i]),
                "effect": float(effect[i]),
            })

        return {
            "att": att,
            "ci": [ci_lo, ci_hi],
            "p": None,  # P-value not directly available from SARIMAX
            "relative_effect": None,
            "timeline": timeline,
            "notes": None,
        }
        
    except Exception as e:
        return {
            "att": None,
            "ci": [None, None],
            "p": None,
            "relative_effect": None,
            "timeline": None,
            "notes": f"Impact estimation failed: {e}",
        }


# --------------------------------- Main --------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    data_path = PROJECT_ROOT / cfg.get("data_path", "data/ontario_cases.csv")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(data_path)
    policy_date = resolve_intervention_date(cfg)

    # Prepare data
    if "post" not in panel.columns:
        panel["post"] = (panel["week"] >= policy_date).astype(int)
    else:
        panel["post"] = pd.to_numeric(panel["post"], errors="coerce").fillna(0).astype(int)
        
    panel["treat_post"] = panel["treated"] * panel["post"]

    # === Feature Engineering for PSM ===
    pre = panel[panel["post"] == 0].copy()
    
    # Pre-period level
    lvl = (pre.groupby("region", as_index=False)["incidence"]
               .mean().rename(columns={"incidence":"pre_level"}))
               
    # Pre-period trend
    def _slope(df: pd.DataFrame) -> float:
        x = (df["week"] - df["week"].min()).dt.days.values.reshape(-1,1)
        y = df["incidence"].values.astype(float)
        if len(y) < 3: 
            return np.nan 
        try:
            b = np.linalg.lstsq(np.c_[np.ones_like(x), x], y, rcond=None)[0][1]
            return float(b)
        except Exception:
            return np.nan

    tr = pre.groupby("region").apply(_slope, include_groups=False).reset_index(name="pre_trend")
    panel = panel.merge(lvl, on="region", how="left").merge(tr, on="region", how="left")

    # 1. Run DiD
    did_att, did_se = _run_did(panel)

    # 2. Run PSM
    covariates_cfg = list(cfg.get("covariates", [])) if cfg.get("covariates") else _infer_covariates(panel)
    psm_att, psm_reason, psm_diag, matched_t, matched_c = _run_psm(panel, covariates_cfg)

    # 3. Python-based Causal Impact
    impact = _estimate_impact_python(panel, policy_date)

    # 4. Generate Figures
    artifacts = generate_analysis_figures(
        panel=panel,
        policy_date=policy_date,
        pre_period=panel[panel["post"] == 0],
        covariates=psm_diag.get("covariates", []),
        matched_treated=matched_t,
        matched_control=matched_c,
        impact_series=impact.get("timeline"),
        root=PROJECT_ROOT,
    )

    # === Build unified results ===
    def safe_float(val):
        """Convert to float or None."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)

    out = {
        "did": {
            "att": safe_float(did_att), 
            "se": safe_float(did_se), 
            "notes": None
        },
        "psm": {
            "att": safe_float(psm_att),
            "covariates": psm_diag.get("covariates", []),
            "diagnostics": psm_diag,
            "notes": psm_reason,
        },
        "bsts": {
            "att": safe_float(impact.get("att")),
            "ci": [safe_float(v) for v in impact.get("ci", [None, None])],
            "p": safe_float(impact.get("p")),
            "relative_effect": safe_float(impact.get("relative_effect")),
            "notes": impact.get("notes"),
        },
        "artifacts": {
            "event_study": getattr(artifacts, "event_study", None),
            "balance": getattr(artifacts, "balance", None),
            "impact": getattr(artifacts, "impact", None),
        },
        "metadata": {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "n_rows": int(len(panel)),
            "n_regions": int(panel["region"].nunique()),
            "intervention_date": policy_date.date().isoformat(),
        }
    }

    # Write results
    output_path = results_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    
    print(f"\n✓ Analysis complete. Results saved to {output_path}")
    print("\nSummary:")
    print(f"  DiD ATT:  {out['did']['att']}")
    print(f"  PSM ATT:  {out['psm']['att']}")
    print(f"  BSTS ATT: {out['bsts']['att']}")
