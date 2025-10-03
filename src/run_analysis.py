import json, os, random, tempfile, subprocess
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import sys
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import yaml

# --- Determinism ---
from figures import generate_analysis_figures
from shared import ROOT, load_config, resolve_intervention_date

# --- Determinism -----------------------------------------------------------
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)

# --- Config ---
ROOT = Path(__file__).resolve().parents[1]
try:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
except FileNotFoundError:
    print("WARNING: config.yaml not found. Using defaults.")
    cfg = {}
cfg: dict[str, Any] = load_config()

data_path = ROOT / cfg.get("data_path", "data/ontario_cases.csv")
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# --- Load data ---
# --- Load data -------------------------------------------------------------
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"ERROR: Data file not found at {data_path}")
    sys.exit(1)
    raise SystemExit(1)

# expected columns
required = {"week", "region", "incidence", "treated"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# parse dates
if pd.api.types.is_string_dtype(df["week"]):
    df["week"] = pd.to_datetime(df["week"], errors="coerce")

# Determine policy date and 'post' period
policy_date = resolve_intervention_date(cfg)
if "post" not in df.columns:
    policy_date = pd.Timestamp(cfg.get("policy_date", "2021-02-01"))
    df["post"] = (df["week"] >= policy_date).astype(int)
else:
    # If 'post' is pre-defined, try to infer the policy date for R script
    policy_date = df.loc[df["post"] == 1, "week"].min()
    inferred = pd.Timestamp(df.loc[df["post"] == 1, "week"].min())
    if not pd.isna(inferred):
        policy_date = inferred

# --- DiD (Difference-in-Differences) ---
# --- Difference-in-Differences --------------------------------------------
df["treat_post"] = df["treated"] * df["post"]
try:
    did = smf.ols("incidence ~ C(region) + C(week) + treat_post", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["region"]}
    )
    did_att = float(did.params.get("treat_post", float("nan")))
    did_se  = float(did.bse.get("treat_post", float("nan")))
except Exception as e:
    did_att = float('nan')
    did_se = float('nan')
    print(f"WARNING: DiD estimation failed: {e}")
    did_se = float(did.bse.get("treat_post", float("nan")))
except Exception as exc:  # pragma: no cover - defensive
    print(f"WARNING: DiD estimation failed: {exc}")
    did_att = float("nan")
    did_se = float("nan")

# --- PSM (Propensity Score Matching) ---
psm_att = None
psm_reason = None
psm_diag = {}
# --- Propensity Score Matching --------------------------------------------
psm_att: float | None = None
psm_reason: str | None = None
psm_diag: dict[str, Any] = {}
matched_treated_df: pd.DataFrame | None = None
matched_control_df: pd.DataFrame | None = None

try:
    pre = df[df["post"] == 0].copy()
    drop_cols = {"week","region","incidence","treated","post","treat_post"}
    # Identify covariates for PS calculation (numeric columns not in drop_cols)
    covars = [c for c in pre.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(pre[c])]
    if not covars:
        covars = ["incidence"] # Fallback if no other numeric covariates are found
pre = df[df["post"] == 0].copy()
drop_cols = {"week", "region", "incidence", "treated", "post", "treat_post"}
covars = [c for c in pre.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(pre[c])]
if not covars:
    covars = ["incidence"]

try:
    n_treat = int(pre["treated"].sum())
    n_ctrl  = int((1 - pre["treated"]).sum())
    n_ctrl = int((1 - pre["treated"]).sum())
    psm_diag.update(n_treat_pre=n_treat, n_ctrl_pre=n_ctrl, covars=covars)

    if n_treat == 0 or n_ctrl == 0:
        psm_reason = "No treated or control units in pre-period; skipping PSM."
        raise RuntimeError(psm_reason)

    # Prepare features: fill NaNs with median
    X = pre[covars].fillna(pre[covars].median(numeric_only=True)).to_numpy()
    y = pre["treated"].to_numpy()

    # Calculate Propensity Score (PS)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X, y)
    pre["ps"] = lr.predict_proba(X)[:, 1]

    # Check Common Support
    ps_t = pre.loc[pre["treated"] == 1, "ps"]
    ps_c = pre.loc[pre["treated"] == 0, "ps"]
    overlap_low  = max(ps_t.min(), ps_c.min())
    overlap_low = max(ps_t.min(), ps_c.min())
    overlap_high = min(ps_t.max(), ps_c.max())
    psm_diag.update(ps_overlap_low=float(overlap_low), ps_overlap_high=float(overlap_high))

    if overlap_low >= overlap_high:
        psm_reason = "No common support in propensity scores; skipping PSM."
        raise RuntimeError(psm_reason)

    # Enforce Common Support
    pre_cs = pre[pre["ps"].between(overlap_low, overlap_high, inclusive="both")].copy()
    n_treat_cs = int(pre_cs["treated"].sum())
    n_ctrl_cs  = int((1 - pre_cs["treated"]).sum())
    n_ctrl_cs = int((1 - pre_cs["treated"]).sum())
    psm_diag.update(n_treat_pre_cs=n_treat_cs, n_ctrl_pre_cs=n_ctrl_cs)

    if n_treat_cs == 0 or n_ctrl_cs == 0:
        psm_reason = "Common-support filter removed all treated or control units; skipping PSM."
        raise RuntimeError(psm_reason)

    # Calculate Caliper (0.2 * SD of Logit PS is a common rule)
    eps = 1e-6
    logit = np.log(np.clip(pre_cs["ps"], eps, 1 - eps)) - np.log(1 - np.clip(pre_cs["ps"], eps, 1 - eps))
    ps_vals = pre_cs["ps"].clip(eps, 1 - eps)
    logit = np.log(ps_vals) - np.log1p(-ps_vals)
    caliper = 0.2 * float(np.nanstd(logit))
    psm_diag.update(caliper=caliper)

    # 1:1 Nearest Neighbor Matching with Caliper (on PS distance)
    caliper_limit_ps = caliper * 0.25 # Use a fraction of the logit SD caliper for PS distance
    psm_diag.update(caliper_ps_limit=float(caliper_limit_ps))
    caliper_limit_ps = caliper * 0.25
    psm_diag.update(caliper=caliper, caliper_ps_limit=float(caliper_limit_ps))

    controls = pre_cs[pre_cs["treated"] == 0].copy()
    treats   = pre_cs[pre_cs["treated"] == 1].copy()
    
    # Matching on the raw PS
    treats = pre_cs[pre_cs["treated"] == 1].copy()

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(controls[["ps"]].to_numpy())
    dist, idx = nn.kneighbors(treats[["ps"]].to_numpy())
    dist = dist.flatten(); idx = idx.flatten()
    dist = dist.flatten()
    idx = idx.flatten()

    ps_pairs = []
    ps_pairs: list[tuple[pd.Series, pd.Series]] = []
    matched_treated_rows: list[pd.Series] = []
    matched_control_rows: list[pd.Series] = []
    for i, j in enumerate(idx):
        # Check if the PS distance is within the caliper limit
        if dist[i] <= caliper_limit_ps:
            ps_pairs.append((treats.iloc[i], controls.iloc[j]))
            treated_row = treats.iloc[i]
            control_row = controls.iloc[j]
            ps_pairs.append((treated_row, control_row))
            matched_treated_rows.append(treated_row)
            matched_control_rows.append(control_row)

    psm_diag.update(n_matched=len(ps_pairs))
    if len(ps_pairs) == 0:
    psm_diag["n_matched"] = len(ps_pairs)
    if not ps_pairs:
        psm_reason = "No matches within caliper; skipping PSM."
        raise RuntimeError(psm_reason)

    # Calculate ATT using matched pairs in the post-period
    if matched_treated_rows and matched_control_rows:
        matched_treated_df = pd.DataFrame(matched_treated_rows)
        matched_control_df = pd.DataFrame(matched_control_rows)

    post = df[df["post"] == 1].copy()
    post_mean = post.groupby("region", as_index=True)["incidence"].mean()
    diffs = []
    
    for t_row, c_row in ps_pairs:
        t_reg = t_row["region"]; c_reg = c_row["region"]

    diffs: list[float] = []
    for treated_row, control_row in ps_pairs:
        t_reg = treated_row["region"]
        c_reg = control_row["region"]
        if t_reg in post_mean.index and c_reg in post_mean.index:
            # ATT = E[Y_t(1) - Y_t(0) | T=1]
            # Use post-period mean difference of matched regions
            diffs.append(float(post_mean.loc[t_reg] - post_mean.loc[c_reg]))
            
    if len(diffs) == 0:

    if not diffs:
        psm_reason = "Matched regions missing from post-period; skipping PSM."
        raise RuntimeError(psm_reason)

    psm_att = float(np.mean(diffs))
except Exception as e:
except Exception as exc:
    if not psm_reason:
        psm_reason = f"PSM failed due to an unexpected error: {e}"
    pass # psm_att remains None/NaN

# --- BSTS / CausalImpact (via Rscript) ---
bsts_att = None
bsts_reason = None

def try_rscript_path(agg: pd.DataFrame):
    """
    Runs CausalImpact via Rscript, dynamically finding Rscript using R_HOME if available.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_p = td / "series.csv"
        out_p = td / "out.json"
        
        # Save data to CSV for R
        agg.to_csv(csv_p, index=False)
        policy_date_str = pd.Timestamp(cfg.get("policy_date", "2021-02-01")).date().isoformat()
        
        # R script to run CausalImpact on the aggregated incidence time series
        r_code = f"""
            suppressMessages(library(CausalImpact))
            suppressMessages(library(jsonlite))
            suppressMessages(library(zoo))
            
            dat <- read.csv("{csv_p.as_posix()}")
            dat$week <- as.Date(dat$week)
            
            # Create a time series object (CausalImpact requires a zoo or xts object)
            time.series <- zoo(dat$incidence, order.by = dat$week)
            policy_date <- as.Date("{policy_date_str}")
            
            # Define periods (policy date is the first day of post-treatment)
            pre_period <- c(min(dat$week, na.rm=TRUE), policy_date - 1)
            post_period <- c(policy_date, max(dat$week, na.rm=TRUE))
            
            # Run CausalImpact (single time series analysis)
            ci <- CausalImpact(time.series, pre.period = pre_period, post.period = post_period)
            
            # Extract the cumulative average effect (ATT)
            res <- list(bsts_att=as.numeric(ci$summary$AbsEffect["Average"]))
            write(jsonlite::toJSON(res, auto_unbox=TRUE), "{out_p.as_posix()}")
        """
        r_script = td / "run_ci.R"
        r_script.write_text(r_code)
        
        # --- Determine Rscript executable path ---
        rscript_exec = "Rscript"
        r_home = os.environ.get("R_HOME")
        
        if r_home:
            # Construct the full path using R_HOME
            rscript_base = Path(r_home) / "bin" / "Rscript"
            if sys.platform == "win32":
                # On Windows, use .exe suffix and POSIX path for MINGW64/subprocess
                rscript_exec = str(rscript_base.with_suffix(".exe").as_posix())
            else:
                rscript_exec = str(rscript_base.as_posix())
                
        # The subprocess call uses the determined executable path
        subprocess.check_call([rscript_exec, r_script.as_posix()])
        
        out = json.loads(out_p.read_text())
        return float(out["bsts_att"])
        psm_reason = f"PSM failed due to an unexpected error: {exc}"

# --- Python SARIMAX counterfactual ---------------------------------------
impact_att: float | None = None
impact_ci: tuple[float | None, float | None] = (None, None)
impact_reason: str | None = None
impact_series: list[Mapping[str, Any]] | None = None


def _estimate_impact_sarimax(panel: pd.DataFrame, policy_date: pd.Timestamp):
    """Estimate post-policy impact via SARIMAX on the treated-control difference."""

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
    # Aggregate data for single time series analysis
    agg = df.sort_values("week").groupby("week", as_index=False)["incidence"].mean()
    bsts_att = try_rscript_path(agg)
    bsts_reason = None
except subprocess.CalledProcessError as e:
    bsts_reason = f"Rscript failed (exit code {e.returncode}). Did you install R packages? Error: {e.stderr.decode()}"
except FileNotFoundError as e:
    bsts_reason = f"Rscript command not found. Set R_HOME environment variable. Error: {e}"
except Exception as e:
    bsts_reason = "BSTS via Rscript failed: " + str(e)


# --- Save results ---
    impact_att, impact_ci, impact_series = _estimate_impact_sarimax(df, policy_date)
except Exception as exc:
    impact_reason = f"Python SARIMAX impact failed: {exc}"

artifacts = generate_analysis_figures(
    panel=df,
    policy_date=policy_date,
    pre_period=pre,
    covariates=covars,
    matched_treated=matched_treated_df,
    matched_control=matched_control_df,
    impact_series=impact_series,
    root=ROOT,
)

impact_ci_payload: list[float | None]
if all(v is not None for v in impact_ci):
    impact_ci_payload = [float(impact_ci[0]), float(impact_ci[1])]
else:
    impact_ci_payload = [None, None]

out = {
    "did_att": did_att,
    "did_se": did_se,
    "psm_att": (None if psm_att is None or (isinstance(psm_att, float) and np.isnan(psm_att)) else float(psm_att)),
    "bsts_att": bsts_att,
    "psm_att": None
    if psm_att is None or (isinstance(psm_att, float) and np.isnan(psm_att))
    else float(psm_att),
    "impact_att": impact_att,
    "impact_ci": impact_ci_payload,
    "meta": {
        "psm_reason": psm_reason,
        "psm_diagnostics": psm_diag,
        "bsts_reason": bsts_reason,
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
