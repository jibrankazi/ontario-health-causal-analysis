from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from shared import ROOT, load_config, resolve_intervention_date
# Assuming generate_analysis_figures is defined elsewhere, like in figures.py
# from figures import generate_analysis_figures 

# Placeholder for generate_analysis_figures to allow the script to run standalone
class Artifacts:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def generate_analysis_figures(**kwargs) -> Artifacts:
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
    drop = {"week", "region", "incidence", "treated", "post", "treat_post"}
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


def _estimate_impact_sarimax(panel: pd.DataFrame, policy_date: pd.Timestamp) -> tuple[float, tuple[float, float], list[Mapping[str, Any]]]:
    """Impact via SARIMAX on treated-minus-control weekly difference."""
    treated = (
        panel[panel["treated"] == 1].groupby("week", as_index=False)["incidence"].mean().rename(columns={"incidence": "treated"})
    )
    control = (
        panel[panel["treated"] == 0].groupby("week", as_index=False)["incidence"].mean().rename(columns={"incidence": "control"})
    )
    series = pd.merge(treated, control, on="week", how="inner").dropna().sort_values("week").reset_index(drop=True)
    if series.empty:
        raise RuntimeError("No overlapping treated/control weeks to construct series.")
    series["y"] = series["treated"] - series["control"]

    pre = series[series["week"] < policy_date].copy().reset_index(drop=True)
    post = series[series["week"] >= policy_date].copy().reset_index(drop=True)
    if pre.empty or post.empty:
        raise RuntimeError("Pre or post period empty; check intervention_date and data coverage.")

    mod = SARIMAX(pre["y"].to_numpy(float), order=(1,0,0), enforce_stationarity=False, enforce_invertibility=False)
    fit = mod.fit(disp=False)
    fc = fit.get_forecast(steps=len(post))
    pred = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05).to_numpy(float)

    att = float(np.mean(post["y"].to_numpy(float) - pred))
    
    # Re-calculate CI bounds based on the difference (y - prediction)
    impact_diff = post["y"].to_numpy(float) - pred.to_numpy(float)
    impact_lower = post["y"].to_numpy(float) - ci[:, 1] # Actual - Upper CI
    impact_upper = post["y"].to_numpy(float) - ci[:, 0] # Actual - Lower CI
    lo = float(np.mean(impact_lower))
    hi = float(np.mean(impact_upper))

    timeline: list[dict[str, Any]] = []
    for dt, actual_y, p, l, u in zip(post["week"], post["y"], pred, ci[:, 0], ci[:, 1]):
        timeline.append({
            "date": dt.date().isoformat(), 
            "actual": float(actual_y),
            "predicted": float(p), 
            "lower": float(l), 
            "upper": float(u)
        })
        
    return att, (lo, hi), timeline


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

    # 1. Run DiD
    did_att, did_se = _run_did(panel)

    # 2. Run PSM
    # Determine covariates: use config if present, otherwise infer
    covariates_cfg = list(cfg.get("covariates", [])) if cfg.get("covariates") else _infer_covariates(panel)
    psm_att, psm_reason, psm_diag, matched_t, matched_c = _run_psm(panel, covariates_cfg)

    # 3. Run SARIMAX impact
    try:
        impact_att, impact_ci, impact_series = _estimate_impact_sarimax(panel, policy_date)
    except RuntimeError as e:
        print(f"SARIMAX failed: {e}")
        impact_att, impact_ci, impact_series = None, (None, None), []


    # 4. Generate Figures
    artifacts = generate_analysis_figures(
        panel=panel,
        policy_date=policy_date,
        pre_period=panel[panel["post"] == 0],
        covariates=psm_diag.get("covariates", []), # Use the actual covariates used in PSM
        matched_treated=matched_t,
        matched_control=matched_c,
        impact_series=impact_series,
        root=ROOT,
    )

    # === Unified results writer =============================================
    from pathlib import Path
    import json, pandas as pd

    # Build 'out' from your existing variables
    out = {
        "did": {"att": did_att, "se": did_se, "notes": None},
        "psm": {
            "att": psm_att,
            "covariates": psm_diag.get("covariates", []) if isinstance(psm_diag, dict) else [],
            "diagnostics": psm_diag if isinstance(psm_diag, dict) else {},
            "notes": psm_reason if 'psm_reason' in locals() else None,
        },
        "sarimax": {
            "att": impact_att,
            # Use the refined CI handling logic from the patch
            "ci": ([float(impact_ci[0]), float(impact_ci[1])]
                   if impact_ci and all(v is not None for v in impact_ci) else [None, None]),
            "timeline": impact_series if isinstance(impact_series, list) else [],
            "notes": "treated-minus-control",
        },
        "artifacts": {
            # Use getattr for safer access, handling cases where 'artifacts' might be missing keys
            "event_study": getattr(artifacts, "event_study", "figures/fig1_event_study.png"),
            "balance":     getattr(artifacts, "balance",     "figures/fig2_smd_balance.png"),
            "impact":      getattr(artifacts, "impact",      "figures/fig3_impact_counterfactual.png"),
            "bsts_plot": "figures/fig_causalimpact.png",
            "bsts_txt":  "results/causalimpact_summary.txt",
        }
    }

    # Metadata
    out["metadata"] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "n_rows": int(len(panel)),
        "n_regions": int(panel["region"].nunique()),
        "intervention_date": policy_date.date().isoformat(),
    }

    # Merge BSTS JSON if present (written by R script)
    bsts_path = Path("results/bsts.json")
    if bsts_path.exists():
        try:
            bsts = json.loads(bsts_path.read_text())
            out["bsts"] = {
                "att": bsts.get("att"),
                "ci": bsts.get("ci") or [None, None],
                "p": bsts.get("p"),
                "relative_effect": bsts.get("relative_effect"),
                "notes": bsts.get("notes"),
            }
        except Exception as e:
            out["bsts"] = {"att": None, "ci": [None, None], "p": None,
                           "relative_effect": None, "notes": f"Failed to read bsts.json: {e}"}
    else:
        out["bsts"] = {"att": None, "ci": [None, None], "p": None,
                       "relative_effect": None, "notes": "BSTS not run or bsts.json missing"}

    # Write unified file
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/results.json").write_text(json.dumps(out, indent=2))
    print("Wrote unified results to results/results.json")
    # ========================================================================
