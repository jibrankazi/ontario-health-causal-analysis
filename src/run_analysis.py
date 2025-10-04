from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
# SARIMAX is still imported but only used by impact_py.py for its fallback
# from statsmodels.tsa.statespace.sarimax import SARIMAX # Removed from top level
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from shared import ROOT, load_config, resolve_intervention_date
# NEW: Import the robust Python Causal Impact estimator
from impact_py import estimate_causal_impact_python


# Placeholder for generate_analysis_figures to allow the script to run standalone
class Artifacts:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def generate_analysis_figures(**kwargs) -> Artifacts:
    """Mock function for figure generation."""
    print("NOTE: Figure generation skipped in this environment.")
    return Artifacts(
        event_study="figures/fig1_event_study.png",
        balance="figures/fig2_smd_balance.png",
        impact="figures/fig3_impact_counterfactual.png",
    )
# End Placeholder


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
        # Check if the interaction term exists before running OLS
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
    # Updated: Exclude the new engineered features 'pre_level' and 'pre_trend'
    drop = {"week", "region", "incidence", "treated", "post", "treat_post", "pre_level", "pre_trend"} 
    covs = [c for c in panel.columns if c not in drop and pd.api.types.is_numeric_dtype(panel[c])]
    return covs


def _run_psm(panel: pd.DataFrame, covariates: Sequence[str]) -> tuple[float | None, str | None, dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    """Nearest-neighbor on pre-period propensity scores; returns ATT over post period."""
    diag: dict[str, Any] = {"caliper": 0.05} # Initialize caliper here for consistency
    if not covariates:
        return None, "No covariates provided; skipping PSM.", diag, None, None

    pre = panel[panel["post"] == 0].copy()
    if pre.empty:
        return None, "No pre-period observations available for PSM.", diag, None, None

    # Use specified covariates or the first few inferred ones
    covariate_cols = [c for c in covariates if c in pre.columns]
    
    # Simple check for 'pre_level' and 'pre_trend' which are often calculated elsewhere
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
        return None, f"Logistic Regression failed (check for perfect separation): {e}", diag, None, None

    pre = pre.assign(ps=lr.predict_proba(X)[:, 1])

    treats = pre[pre["treated"] == 1].copy()
    ctrls  = pre[pre["treated"] == 0].copy()
    if treats.empty or ctrls.empty:
        return None, "No treated or control rows after PS estimation.", diag, None, None

    nn = NearestNeighbors(n_neighbors=1).fit(ctrls[["ps"]].to_numpy(float))
    dist, idx = nn.kneighbors(treats[["ps"]].to_numpy(float))
    dist, idx = dist.flatten(), idx.flatten()

    caliper = diag["caliper"]
    # Ensure that `treats` and `ctrls` are correctly indexed before slicing
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
    
    # Collect unique regions to form matched treated/control dataframes
    for t_row, c_row in pairs:
        t_reg, c_reg = t_row["region"], c_row["region"]
        if t_reg in post_means.index and c_reg in post_means.index:
            diffs.append(float(post_means.loc[t_reg] - post_means.loc[c_reg]))
            matched_t_list.append(t_row.to_dict())
            matched_c_list.append(c_row.to_dict())
            
    if not diffs:
        return None, "Matched regions missing in post period; skipping PSM.", diag, None, None
        
    return float(np.mean(diffs)), None, diag, pd.DataFrame(matched_t_list), pd.DataFrame(matched_c_list)


# Note: The original _estimate_impact_sarimax function is removed 
# because estimate_causal_impact_python now handles the counterfactual logic, 
# including the SARIMAX fallback.


# --------------------------------- Main --------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    data_path = ROOT / cfg.get("data_path", "data/ontario_cases.csv")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(data_path)
    policy_date = resolve_intervention_date(cfg)

    # Prepare data for DiD and PSM
    if "post" not in panel.columns:
        panel["post"] = (panel["week"] >= policy_date).astype(int)
    else:
        panel["post"] = pd.to_numeric(panel["post"], errors="coerce").fillna(0).astype(int)
        
    panel["treat_post"] = panel["treated"] * panel["post"]

    # === Feature Engineering for PSM (pre_level and pre_trend) =================
    # --- pre-period covariates for PSM ---
    pre = panel[panel["post"] == 0].copy()
    
    # level
    lvl = (pre.groupby("region", as_index=False)["incidence"]
               .mean().rename(columns={"incidence":"pre_level"}))
               
    # trend (simple OLS slope per region)
    def _slope(df: pd.DataFrame) -> float:
        # Create continuous time index in days
        x = (df["week"] - df["week"].min()).dt.days.values.reshape(-1,1)
        y = df["incidence"].values.astype(float)
        
        # Need at least 3 points for a meaningful OLS trend calculation
        if len(y) < 3: 
            return np.nan 
            
        # Perform OLS using numpy.linalg.lstsq: [1, x] vs y
        # We extract the slope (index 1) from the coefficients
        try:
            # np.c_ adds a column of ones for the intercept
            b = np.linalg.lstsq(np.c_[np.ones_like(x), x], y, rcond=None)[0][1]
            return float(b)
        except Exception:
            return np.nan # Return NaN if calculation fails

    tr = pre.groupby("region").apply(_slope).reset_index(name="pre_trend")

    # Merge the new covariates back into the main panel dataframe
    panel = panel.merge(lvl, on="region", how="left").merge(tr, on="region", how="left")
    # ==========================================================================


    # 1. Run DiD
    did_att, did_se = _run_did(panel)

    # 2. Run PSM
    # Determine covariates: use config if present, otherwise infer
    covariates_cfg = list(cfg.get("covariates", [])) if cfg.get("covariates") else _infer_covariates(panel)
    psm_att, psm_reason, psm_diag, matched_t, matched_c = _run_psm(panel, covariates_cfg)

    # --- Python-only CausalImpact / BSTS-style impact --------------------------
    impact = estimate_causal_impact_python(panel, policy_date)
    
    # Use this in the unified writer below
    bsts_att = impact.att
    bsts_ci = [impact.ci[0], impact.ci[1]]
    bsts_p = impact.p
    bsts_rel = impact.relative_effect
    bsts_notes = impact.notes
    impact_timeline = impact.timeline
    impact_plot = impact.plot_path
    impact_txt = impact.summary_path
    # --------------------------------------------------------------------------

    # 3. Generate Figures
    artifacts = generate_analysis_figures(
        panel=panel,
        policy_date=policy_date,
        pre_period=panel[panel["post"] == 0],
        covariates=psm_diag.get("covariates", []), # Use the actual covariates used in PSM
        matched_treated=matched_t,
        matched_control=matched_c,
        impact_series=impact_timeline, # Use the timeline from the Python CI analysis
        root=ROOT,
    )
    
    print(f"[RUN] src/run_analysis.py :: __file__={__file__}")
    print("[RUN] unified-writer about to merge & write results/results.json")

    # === Unified results writer (drop-in) =======================================

    # Build 'out' from your existing variables
    out = {
        "did": {"att": did_att, "se": did_se, "notes": None},
        "psm": {
            "att": psm_att,
            "covariates": psm_diag.get("covariates", []) if isinstance(psm_diag, dict) else [],
            "diagnostics": psm_diag if isinstance(psm_diag, dict) else {},
            "notes": psm_reason,
        },
        # The dedicated SARIMAX section is deprecated, its functionality is now within "bsts"
        "sarimax": {  
            "att": None,
            "ci": [None, None],
            "timeline": [],
            "notes": "deprecated (Python CI implementation supersedes)"
        },
        # Populate BSTS section with results from the new Python CI tool
        "bsts": {
            "att": bsts_att,
            "ci": bsts_ci,
            "p": bsts_p,
            "relative_effect": bsts_rel,
            "notes": bsts_notes,
        },
        "artifacts": {
            "event_study": getattr(artifacts, "event_study", "figures/fig1_event_study.png"),
            "balance":     getattr(artifacts, "balance",     "figures/fig2_smd_balance.png"),
            "impact":      impact_plot or "figures/fig_causalimpact.png", # Use the CI plot as the main impact visualization
            "bsts_plot":   impact_plot or "figures/fig_causalimpact.png",
            "bsts_txt":    impact_txt or "results/causalimpact_summary.txt",
        }
    }

    # Metadata
    out["metadata"] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "n_rows": int(len(panel)),
        "n_regions": int(panel["region"].nunique()),
        "intervention_date": policy_date.date().isoformat(),
    }

    # Remove the bsts.json reading block as Python now generates the result directly
    # And there is no more need to worry about float('nan') which isn't json serializable
    def json_default_handler(obj):
        if pd.isna(obj) or obj is None:
            return None
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    # Write unified file
    (results_dir / "results.json").write_text(json.dumps(out, indent=2, default=json_default_handler))
    print("Wrote unified results to results/results.json")
    # ========================================================================
