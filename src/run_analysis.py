# --------------------------------------------
# Ontario Health Causal Analysis (DiD, PSM, optional BSTS via R/CausalImpact)
# - Deterministic seeds
# - Config-driven (config.yaml)
# - Robust PSM with common support + calipered 1-NN
# - Optional BSTS guarded by config: bsts.enabled: true|false
# - Results saved to results/results.json
# --------------------------------------------

from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
from datetime import datetime, timezone, date
from pathlib import Path
import math

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# ============================================================
# Determinism
# ============================================================
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)

# ============================================================
# Config / Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[1]  # repo root (folder containing src/)
cfg_path = ROOT / "config.yaml"
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
else:
    cfg = {}

data_path = ROOT / cfg.get("data_path", "data/ontario_cases.csv")

# ✅ define results_dir early and ensure it exists
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)
results_path = results_dir / "results.json"

# BSTS toggle (default: enabled)
cfg_bsts = (cfg.get("bsts") or {})
bsts_enabled = bool(cfg_bsts.get("enabled", True))

# ============================================================
# Load Data
# ============================================================
if not data_path.exists():
    raise FileNotFoundError(
        f"Data file not found at {data_path}. "
        "Update config.yaml (key: data_path) or place the CSV there."
    )

df = pd.read_csv(data_path)

# Expected columns
required = {"week", "region", "incidence", "treated"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Parse dates
if pd.api.types.is_string_dtype(df["week"]):
    df["week"] = pd.to_datetime(df["week"], errors="coerce")

# Construct 'post' variable if absent
if "post" not in df.columns:
    policy_date = pd.Timestamp(cfg.get("policy_date", "2021-02-01"))
    df["post"] = (df["week"] >= policy_date).astype(int)

# Basic hygiene
if df["week"].isna().any():
    # drop or raise; we will drop invalid weeks to avoid model errors
    df = df.dropna(subset=["week"]).copy()

# ============================================================
# Difference-in-Differences
# ============================================================
df["treat_post"] = df["treated"] * df["post"]
# region FE and week FE via categorical indicators
did_model = smf.ols("incidence ~ C(region) + C(week) + treat_post", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["region"]}
)
did_att = float(did_model.params.get("treat_post", float("nan")))
did_se = float(did_model.bse.get("treat_post", float("nan")))

# ============================================================
# Propensity Score Matching (robust)
# ============================================================
psm_att: float | None = None
psm_reason: str | None = None
psm_diag: dict = {}

try:
    pre = df[df["post"] == 0].copy()

    # Covariates: all numeric columns except keys/outcomes
    drop_cols = {"week", "region", "incidence", "treated", "post", "treat_post"}
    covars = [c for c in pre.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(pre[c])]
    if not covars:
        covars = ["incidence"]  # safe fallback if nothing else

    n_treat = int(pre["treated"].sum())
    n_ctrl = int((1 - pre["treated"]).sum())
    psm_diag.update(n_treat_pre=n_treat, n_ctrl_pre=n_ctrl, covars=covars)

    if n_treat == 0 or n_ctrl == 0:
        psm_reason = "No treated or control units in pre-period; skipping PSM."
        raise RuntimeError(psm_reason)

    X = pre[covars].copy()
    # fillna per column median to avoid data leaks with NaN
    X = X.fillna(X.median(numeric_only=True))
    X_np = X.to_numpy()
    y_np = pre["treated"].to_numpy()

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_np, y_np)
    pre["ps"] = lr.predict_proba(X_np)[:, 1]

    # Common support
    ps_t = pre.loc[pre["treated"] == 1, "ps"]
    ps_c = pre.loc[pre["treated"] == 0, "ps"]
    overlap_low = float(max(ps_t.min(), ps_c.min()))
    overlap_high = float(min(ps_t.max(), ps_c.max()))
    psm_diag.update(ps_overlap_low=overlap_low, ps_overlap_high=overlap_high)

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
    ps_clip = np.clip(pre_cs["ps"].to_numpy(), eps, 1 - eps)
    logit_ps = np.log(ps_clip / (1 - ps_clip))
    caliper = 0.2 * float(np.nanstd(logit_ps))
    psm_diag.update(caliper=caliper)

    # 1-NN with replacement on controls, Euclidean on ps
    controls = pre_cs[pre_cs["treated"] == 0].copy()
    treats = pre_cs[pre_cs["treated"] == 1].copy()
    nn = NearestNeighbors(n_neighbors=1, metric="minkowski", p=2)  # Euclidean on 1D 'ps'
    nn.fit(controls[["ps"]].to_numpy())
    dist, idx = nn.kneighbors(treats[["ps"]].to_numpy())

    # Filter matches by caliper
    matched_pairs = []
    d = dist.flatten()
    j = idx.flatten()
    for i in range(len(treats)):
        if d[i] <= caliper:
            matched_pairs.append((treats.iloc[i], controls.iloc[j[i]]))

    psm_diag.update(n_matched=len(matched_pairs))
    if not matched_pairs:
        psm_reason = "No matches found within the caliper; skipping PSM."
        raise RuntimeError(psm_reason)

    # Post-period ATT using matched sets (region-level post means)
    post = df[df["post"] == 1].copy()
    post_mean_incidence = post.groupby("region", as_index=True)["incidence"].mean()

    diffs = []
    for t_row, c_row in matched_pairs:
        t_reg, c_reg = t_row["region"], c_row["region"]
        if (t_reg in post_mean_incidence.index) and (c_reg in post_mean_incidence.index):
            diffs.append(float(post_mean_incidence.loc[t_reg] - post_mean_incidence.loc[c_reg]))

    if not diffs:
        psm_reason = "Matched regions missing from post-period; skipping PSM."
        raise RuntimeError(psm_reason)

    psm_att = float(np.mean(diffs))

except Exception as e:
    if psm_reason is None:  # unexpected error
        psm_reason = f"PSM failed with an unexpected error: {e}"
    # leave psm_att as None; reason recorded

# ============================================================
# BSTS / CausalImpact via Rscript (optional)
# ============================================================
def _bsts_via_rscript(agg_df: pd.DataFrame) -> float:
    """
    Run CausalImpact in an Rscript subprocess and return the average absolute effect.
    Requires Rscript on PATH and the 'CausalImpact' R package.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        csv_path = td_path / "series.csv"
        out_path = td_path / "out.json"
        agg_df.to_csv(csv_path, index=False)
        policy_date_str = pd.Timestamp(cfg.get("policy_date", "2021-02-01")).strftime("%Y-%m-%d")

        # Ensure dates are explicitly formatted for R
        pre_start = pd.to_datetime(agg_df["week"].min()).strftime("%Y-%m-%d")
        pre_end = (pd.to_datetime(policy_date_str) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        post_end = pd.to_datetime(agg_df["week"].max()).strftime("%Y-%m-%d")

        r_code = f"""
        suppressPackageStartupMessages(library(CausalImpact))
        suppressPackageStartupMessages(library(jsonlite))

        dat <- read.csv("{csv_path.as_posix()}")
        dat$week <- as.Date(dat$week)

        pre_period <- as.Date(c("{pre_start}", "{pre_end}"))
        post_period <- as.Date(c("{policy_date_str}", "{post_end}"))

        ci <- CausalImpact(dat$incidence, pre.period = pre_period, post.period = post_period)

        res <- list(bsts_att = as.numeric(ci$summary$AbsEffect["Average"]))
        write(jsonlite::toJSON(res, auto_unbox=TRUE), "{out_path.as_posix()}")
        """
        r_script_path = td_path / "run_ci.R"
        r_script_path.write_text(r_code, encoding="utf-8")

        # Run Rscript (surface stdout/stderr only on failure)
        subprocess.run(["Rscript", str(r_script_path)], check=True, capture_output=True, text=True)

        out = json.loads(out_path.read_text(encoding="utf-8"))
        return float(out["bsts_att"])

bsts_att: float | None = None
bsts_reason: str | None = None

if bsts_enabled:
    try:
        # Aggregate to weekly mean incidence for univariate CausalImpact
        agg = df.sort_values("week").groupby("week", as_index=False)["incidence"].mean()
        bsts_att = _bsts_via_rscript(agg)
    except Exception as e:
        bsts_reason = f"BSTS via Rscript failed: {e}"
else:
    bsts_reason = "BSTS disabled via config."

# ============================================================
# Save & Print Results
# ============================================================
def _jsonable(x):
    """Make objects JSON-serializable without extra deps."""
    # Primitives pass through
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # NumPy
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()

    # Pandas Timestamp / NA
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if x is pd.NaT:
        return None

    # Paths
    if isinstance(x, Path):
        return str(x)

    # Sets / tuples
    if isinstance(x, (set, tuple)):
        return [_jsonable(v) for v in x]

    # Dicts / lists
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]

    # Datetime / date
    if isinstance(x, (datetime, date)):
        return x.isoformat()

    # Fallback to string
    return str(x)

did_results = {
    "att": None if (did_att is None or math.isnan(did_att)) else float(did_att),
    "se": None if (did_se is None or math.isnan(did_se)) else float(did_se),
    "n_obs": int(len(df)),
    "n_regions": int(df["region"].nunique()),
}

psm_results = {
    "att": None if psm_att is None else float(psm_att),
    "reason": psm_reason,
    "diagnostics": {
        "n_treat_pre": int(psm_diag.get("n_treat_pre", 0)) if "n_treat_pre" in psm_diag else None,
        "n_ctrl_pre": int(psm_diag.get("n_ctrl_pre", 0)) if "n_ctrl_pre" in psm_diag else None,
        "n_treat_pre_cs": int(psm_diag.get("n_treat_pre_cs", 0)) if "n_treat_pre_cs" in psm_diag else None,
        "n_ctrl_pre_cs": int(psm_diag.get("n_ctrl_pre_cs", 0)) if "n_ctrl_pre_cs" in psm_diag else None,
        "ps_overlap_low": float(psm_diag.get("ps_overlap_low")) if "ps_overlap_low" in psm_diag else None,
        "ps_overlap_high": float(psm_diag.get("ps_overlap_high")) if "ps_overlap_high" in psm_diag else None,
        "caliper": float(psm_diag.get("caliper")) if "caliper" in psm_diag else None,
        "n_matched": int(psm_diag.get("n_matched", 0)) if "n_matched" in psm_diag else None,
        "covars": psm_diag.get("covars"),
    },
}

bsts_results = {
    "att": None if bsts_att is None else float(bsts_att),
    "reason": bsts_reason,
}

metadata = {
    "policy_date": str(cfg.get("policy_date", "2021-02-01")),
    "bsts_enabled": bool(bsts_enabled),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}

artifacts = {
    "results_path": str(results_path),
    "data_path": str(data_path),
}

output_data = {
    "did": did_results,
    "psm": psm_results,
    "bsts": bsts_results,
    "metadata": metadata,
    "artifacts": artifacts,
}

# Ensure dir exists and write JSON (guaranteed serializable)
results_dir.mkdir(parents=True, exist_ok=True)
results_path.write_text(json.dumps(_jsonable(output_data), indent=4), encoding="utf-8")

print("\n--- Analysis Complete ---")
print(json.dumps(_jsonable(output_data), indent=4))
