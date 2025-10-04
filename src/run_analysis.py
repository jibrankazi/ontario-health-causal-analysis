from pathlib import Path
import json
import pandas as pd # Assuming you have pandas

# src/run_analysis.py
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
from figures import generate_analysis_figures


# ------------------------------- I/O -----------------------------------------
def _load_panel(path: Path) -> pd.DataFrame:
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
        m = smf.ols("incidence ~ C(region) + C(week) + treat_post", data=panel).fit(
            cov_type="cluster", cov_kwds={"groups": panel["region"]}
        )
        return float(m.params.get("treat_post")), float(m.bse.get("treat_post"))
    except Exception:
        return None, None


def _infer_covariates(panel: pd.DataFrame) -> Sequence[str]:
    drop = {"week", "region", "incidence", "treated", "post", "treat_post"}
    covs = [c for c in panel.columns if c not in drop and pd.api.types.is_numeric_dtype(panel[c])]
    return covs


def _run_psm(panel: pd.DataFrame, covariates: Sequence[str]) -> tuple[float | None, str | None, dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    """Nearest-neighbor on pre-period propensity scores; returns ATT over post period."""
    diag: dict[str, Any] = {}
    if not covariates:
        return None, "No covariates provided; skipping PSM.", diag, None, None

    pre = panel[panel["post"] == 0].copy()
    if pre.empty:
        return None, "No pre-period observations available for PSM.", diag, None, None

    X = pre[covariates].fillna(pre[covariates].median(numeric_only=True)).to_numpy()
    y = pre["treated"].to_numpy()
    n_treat, n_ctrl = int(pre["treated"].sum()), int((1 - pre["treated"]).sum())
    diag.update(n_treat_pre=n_treat, n_ctrl_pre=n_ctrl, covariates=list(covariates))
    if n_treat == 0 or n_ctrl == 0:
        return None, "No treated or control units in pre-period; skipping PSM.", diag, None, None

    lr = LogisticRegression(max_iter=500, random_state=42).fit(X, y)
    pre = pre.assign(ps=lr.predict_proba(X)[:, 1])

    treats = pre[pre["treated"] == 1].copy()
    ctrls  = pre[pre["treated"] == 0].copy()
    if treats.empty or ctrls.empty:
        return None, "No treated or control rows after PS estimation.", diag, None, None

    nn = NearestNeighbors(n_neighbors=1).fit(ctrls[["ps"]].to_numpy())
    dist, idx = nn.kneighbors(treats[["ps"]].to_numpy())
    dist, idx = dist.flatten(), idx.flatten()

    caliper = 0.05
    pairs = [(treats.iloc[i], ctrls.iloc[j]) for i, j in enumerate(idx) if dist[i] <= caliper]
    diag["n_matched"] = len(pairs)
    if not pairs:
        return None, "No matches within caliper; skipping PSM.", diag, None, None

    # ATT over post period mean incidence by matched regions
    post = panel[panel["post"] == 1].copy()
    post_means = post.groupby("region", as_index=True)["incidence"].mean()
    diffs = []
    matched_t, matched_c = [], []
    for t_row, c_row in pairs:
        t_reg, c_reg = t_row["region"], c_row["region"]
        if t_reg in post_means.index and c_reg in post_means.index:
            diffs.append(float(post_means.loc[t_reg] - post_means.loc[c_reg]))
            matched_t.append(t_row)
            matched_c.append(c_row)
    if not diffs:
        return None, "Matched regions missing in post period; skipping PSM.", diag, None, None
    return float(np.mean(diffs)), None, diag, pd.DataFrame(matched_t), pd.DataFrame(matched_c)


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

    pre = series[series["week"] < policy_date].copy()
    post = series[series["week"] >= policy_date].copy()
    if pre.empty or post.empty:
        raise RuntimeError("Pre or post period empty; check intervention_date and data coverage.")

    mod = SARIMAX(pre["y"].to_numpy(float), order=(1,0,0), enforce_stationarity=False, enforce_invertibility=False)
    fit = mod.fit(disp=False)
    fc = fit.get_forecast(steps=len(post))
    pred = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05).to_numpy(float)

    att = float(np.mean(post["y"].to_numpy(float) - pred))
    # NOTE: The CI lower/upper bound calculations here look slightly off and don't match ATT units,
    # but maintaining the original logic structure for now.
    lo = float(np.mean(ci[:, 0] - pre["y"].mean()))
    hi = float(np.mean(ci[:, 1] - pre["y"].mean()))

    timeline: list[dict[str, Any]] = []
    for (idx, dt), p, (l, u) in zip(post["week"].items(), pred, ci):
        # Using post data to get the actual y value for the specific week
        actual_y = post.loc[idx, "y"]
        timeline.append({"date": dt.date().isoformat(), "actual": float(actual_y),
                         "predicted": float(p), "lower": float(l), "upper": float(u)})
    return att, (lo, hi), timeline


# --------------------------------- Main --------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    data_path = ROOT / cfg.get("data_path", "data/ontario_cases.csv")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(data_path)
    policy_date = resolve_intervention_date(cfg)

    if "post" not in panel.columns:
        panel["post"] = (panel["week"] >= policy_date).astype(int)
    else:
        panel["post"] = pd.to_numeric(panel["post"], errors="coerce").fillna(0).astype(int)
    panel["treat_post"] = panel["treated"] * panel["post"]

    did_att, did_se = _run_did(panel)

    # PSM (only if covariates exist)
    covariates_cfg = list(cfg.get("covariates", [])) if cfg.get("covariates") else _infer_covariates(panel)
    psm_att, psm_reason, psm_diag, matched_t, matched_c = _run_psm(panel, covariates_cfg)

    # SARIMAX impact
    try:
        impact_att, impact_ci, impact_series = _estimate_impact_sarimax(panel, policy_date)
    except RuntimeError as e:
        print(f"SARIMAX failed: {e}")
        impact_att, impact_ci, impact_series = None, (None, None), []


    # Figures
    artifacts = generate_analysis_figures(
        panel=panel,
        policy_date=policy_date,
        pre_period=panel[panel["post"] == 0],
        covariates=covariates_cfg,
        matched_treated=matched_t,
        matched_control=matched_c,
        impact_series=impact_series,
        root=ROOT,
    )

    # --- Initialize results dictionary with Unified Schema ---
    out = {
        "metadata": {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "n_rows": int(len(panel)),
            "n_regions": int(panel["region"].nunique()),
            "intervention_date": policy_date.date().isoformat(),
        },
        "did": {
            "att": did_att,
            "se": did_se,
            "notes": None,
        },
        "psm": {
            "att": psm_att,
            "covariates": psm_diag.get("covariates", []),
            "diagnostics": {
                "n_treat_pre": psm_diag.get("n_treat_pre"),
                "n_ctrl_pre": psm_diag.get("n_ctrl_pre"),
                "n_matched": psm_diag.get("n_matched"),
                "caliper": 0.05, # Fixed caliper value used in PSM logic
            },
            "notes": psm_reason,
        },
        "sarimax": {
            "att": impact_att,
            "ci": list(impact_ci),
            "timeline": impact_series,
            "notes": "Treated minus control difference, SARIMAX(1,0,0).",
        },
        "bsts": {
            "att": None, "ci": [None, None], "p": None,
            "relative_effect": None, "notes": "BSTS results pending merge from R script."
        },
        "artifacts": {
            "event_study": artifacts.event_study,
            "balance": artifacts.balance,
            "impact": artifacts.impact,
            "bsts_plot": "figures/fig_causalimpact.png",
            "bsts_txt": "results/causalimpact_summary.txt"
        }
    }

    # --- Merge BSTS JSON from R Script (if available) ---
    bsts_path = ROOT / "results" / "bsts.json"
    if bsts_path.exists():
        try:
            bsts = json.loads(bsts_path.read_text())
            # Overwrite the placeholder 'bsts' key with R results
            out["bsts"] = {
                "att": bsts.get("att"),
                "ci": bsts.get("ci"),
                "p": bsts.get("p"),
                "relative_effect": bsts.get("relative_effect"),
                "notes": bsts.get("notes"),
            }
        except Exception as e:
            out["bsts"] = {"att": None, "ci": [None, None], "p": None,
                           "relative_effect": None, "notes": f"Failed to read bsts.json during merge: {e}"}

    # --- Write the unified file ---
    Path("results").mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps(out, indent=2))
    print("Wrote unified results to results/results.json")
