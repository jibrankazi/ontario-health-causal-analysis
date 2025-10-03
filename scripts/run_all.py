import json, os, random, tempfile, subprocess
import json
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shared import (  # type: ignore after dynamic path tweak
    ROOT as PROJECT_ROOT,
    load_config,
    resolve_intervention_date,
    resolve_rscript,
)

# ---------------------------
# Determinism
# ---------------------------
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)

# ---------------------------
# Config / Paths
# ---------------------------
import yaml
ROOT = Path(__file__).resolve().parents[1]  # project root (src/..)
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
cfg: dict[str, Any] = load_config()

data_path = ROOT / cfg.get("data_path", "data/ontario_cases.csv")
results_dir = ROOT / "results"
data_path = PROJECT_ROOT / cfg.get("data_path", "data/ontario_cases.csv")
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load Data
# ---------------------------
import pandas as pd
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
    policy_date = pd.Timestamp(cfg.get("policy_date", "2021-02-01"))
    df["post"] = (df["week"] >= policy_date).astype(int)

# ---------------------------
# Difference-in-Differences
# ---------------------------
import statsmodels.formula.api as smf
else:
    inferred = pd.Timestamp(df.loc[df["post"] == 1, "week"].min())
    if not pd.isna(inferred):
        policy_date = inferred

df["treat_post"] = df["treated"] * df["post"]
did = smf.ols("incidence ~ C(region) + C(week) + treat_post", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["region"]}
)
did_att = float(did.params.get("treat_post", float("nan")))
did_se  = float(did.bse.get("treat_post", float("nan")))

# ---------------------------
# Propensity Score Matching (robust)
# ---------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

psm_att = None
psm_reason = None
psm_diag = {}

try:
    pre = df[df["post"] == 0].copy()

    # Covariates: all numeric columns except keys/outcomes
    drop_cols = {"week", "region", "incidence", "treated", "post", "treat_post"}
    covars = [c for c in pre.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(pre[c])]
    if not covars:
        covars = ["incidence"]  # fallback
    did = smf.ols("incidence ~ C(region) + C(week) + treat_post", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["region"]}
    )
    did_att = float(did.params.get("treat_post", float("nan")))
    did_se = float(did.bse.get("treat_post", float("nan")))
except Exception as exc:  # pragma: no cover - defensive
    print(f"WARNING: DiD estimation failed: {exc}")
    did_att = float("nan")
    did_se = float("nan")

psm_att: float | None = None
psm_reason: str | None = None
psm_diag: dict[str, Any] = {}

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

    X = pre[covars].fillna(pre[covars].median(numeric_only=True)).to_numpy()
    y = pre["treated"].to_numpy()

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X, y)
    pre["ps"] = lr.predict_proba(X)[:, 1]

    # Common support
    ps_t = pre.loc[pre["treated"] == 1, "ps"]
    ps_c = pre.loc[pre["treated"] == 0, "ps"]
    overlap_low  = max(ps_t.min(), ps_c.min())
    overlap_low = max(ps_t.min(), ps_c.min())
    overlap_high = min(ps_t.max(), ps_c.max())
    psm_diag.update(ps_overlap_low=float(overlap_low), ps_overlap_high=float(overlap_high))

    if overlap_low >= overlap_high:
        psm_reason = "No common support in propensity scores; skipping PSM."
        raise RuntimeError(psm_reason)

    # Restrict to support
    pre_cs = pre[pre["ps"].between(overlap_low, overlap_high, inclusive="both")].copy()
    n_treat_cs = int(pre_cs["treated"].sum())
    n_ctrl_cs  = int((1 - pre_cs["treated"]).sum())
    n_ctrl_cs = int((1 - pre_cs["treated"]).sum())
    psm_diag.update(n_treat_pre_cs=n_treat_cs, n_ctrl_pre_cs=n_ctrl_cs)

    if n_treat_cs == 0 or n_ctrl_cs == 0:
        psm_reason = "Common-support filter removed all treated or control units; skipping PSM."
        raise RuntimeError(psm_reason)

    # Caliper based on 0.2 * SD(logit(ps))
    eps = 1e-6
    logit = np.log(np.clip(pre_cs["ps"], eps, 1 - eps)) - np.log(1 - np.clip(pre_cs["ps"], eps, 1 - eps))
    ps_vals = pre_cs["ps"].clip(eps, 1 - eps)
    logit = np.log(ps_vals) - np.log1p(-ps_vals)
    caliper = 0.2 * float(np.nanstd(logit))
    psm_diag.update(caliper=caliper)
    caliper_limit_ps = caliper * 0.25
    psm_diag.update(caliper=caliper, caliper_ps_limit=float(caliper_limit_ps))

    # 1-NN with replacement on controls
    controls = pre_cs[pre_cs["treated"] == 0].copy()
    treats   = pre_cs[pre_cs["treated"] == 1].copy()
    treats = pre_cs[pre_cs["treated"] == 1].copy()

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(controls[["ps"]].to_numpy())
    dist, idx = nn.kneighbors(treats[["ps"]].to_numpy())
    dist = dist.flatten(); idx = idx.flatten()
    dist = dist.flatten()
    idx = idx.flatten()

    # Simple ps-scale caliper gate (pragmatic bound)
    ps_pairs = []
    ps_pairs: list[tuple[pd.Series, pd.Series]] = []
    for i, j in enumerate(idx):
        ps_ti = float(treats["ps"].iloc[i]); ps_ci = float(controls["ps"].iloc[j])
        if abs(ps_ti - ps_ci) <= max(0.1, caliper * 0.25):
        if dist[i] <= caliper_limit_ps:
            ps_pairs.append((treats.iloc[i], controls.iloc[j]))

    psm_diag.update(n_matched=len(ps_pairs))
    if len(ps_pairs) == 0:
    psm_diag["n_matched"] = len(ps_pairs)
    if not ps_pairs:
        psm_reason = "No matches within caliper; skipping PSM."
        raise RuntimeError(psm_reason)

    # Post-period ATT using matched sets
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
            diffs.append(float(post_mean.loc[t_reg] - post_mean.loc[c_reg]))
    if len(diffs) == 0:

    if not diffs:
        psm_reason = "Matched regions missing from post-period; skipping PSM."
        raise RuntimeError(psm_reason)

    psm_att = float(np.mean(diffs))
except Exception:
    # leave psm_att as None; reason is recorded below
    pass

# ---------------------------
# BSTS / CausalImpact via Rscript (Windows-friendly)
# ---------------------------
bsts_att = None
bsts_reason = None

def _bsts_via_rscript(agg_df: pd.DataFrame) -> float:
    """
    Run CausalImpact in an Rscript subprocess and return the
    average absolute effect (bsts_att).
    Requires Rscript on PATH and packages: CausalImpact, bsts, Boom, BoomSpikeSlab, zoo.
    """
except Exception as exc:
    if not psm_reason:
        psm_reason = f"PSM failed due to an unexpected error: {exc}"

bsts_att: float | None = None
bsts_reason: str | None = None


def _bsts_via_rscript(agg: pd.DataFrame) -> float | None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_p = td / "series.csv"
        out_p = td / "out.json"
        # Save weekly mean series with columns: week, incidence
        agg_df.to_csv(csv_p, index=False)
        policy = pd.Timestamp(cfg.get("policy_date", "2021-02-01")).date()
        temp_dir = Path(td)
        csv_path = temp_dir / "series.csv"
        out_path = temp_dir / "out.json"

        series = agg.copy()
        if pd.api.types.is_datetime64_any_dtype(series["week"]):
            series["week"] = series["week"].dt.date
        series = series[["week", "incidence"]].dropna()
        series.to_csv(csv_path, index=False)

        policy_str = policy_date.date().isoformat()
        r_code = f"""
            suppressMessages(library(CausalImpact))
            suppressMessages(library(jsonlite))
            dat <- read.csv("{csv_p.as_posix()}")
            suppressMessages(library(zoo))

            dat <- read.csv("{csv_path.as_posix()}")
            dat$week <- as.Date(dat$week)
            pre_end  <- as.Date("{policy}") - 1
            post_end <- max(dat$week, na.rm=TRUE)
            ci <- CausalImpact(dat$incidence, c(min(dat$week, na.rm=TRUE), pre_end), c(pre_end+1, post_end))
            res <- list(bsts_att = as.numeric(ci$summary$AbsEffect["Average"]))
            write(jsonlite::toJSON(res, auto_unbox=TRUE), "{out_p.as_posix()}")
            dat$incidence <- as.numeric(dat$incidence)

            ts <- zoo(dat$incidence, order.by = dat$week)

            pre_start  <- min(index(ts))
            pre_end    <- as.Date("{policy_str}") - 1
            post_start <- pre_end + 1
            post_end   <- max(index(ts))

            ci <- CausalImpact(ts, c(pre_start, pre_end), c(post_start, post_end))

            att <- suppressWarnings(as.numeric(ci$summary$AbsEffect["Average"]))
            if (is.na(att)) att_json <- NULL else att_json <- att

            res <- list(bsts_att = att_json)
            write(jsonlite::toJSON(res, auto_unbox = TRUE, na = "null"), "{out_path.as_posix()}")
        """
        r_script = td / "run_ci.R"
        r_script = temp_dir / "run_ci.R"
        r_script.write_text(r_code)
        subprocess.check_call(["Rscript", r_script.as_posix()])
        out = json.loads(out_p.read_text())
        return float(out["bsts_att"])

        rscript_exec = resolve_rscript(cfg)
        subprocess.check_call([rscript_exec, r_script.as_posix()])

        payload = json.loads(out_path.read_text())
        val = payload.get("bsts_att")
        return None if val is None else float(val)


try:
    # Aggregate to weekly mean incidence for univariate CausalImpact
    agg = df.sort_values("week").groupby("week", as_index=False)["incidence"].mean()
    bsts_att = _bsts_via_rscript(agg)
except Exception as e:
    bsts_reason = f"BSTS via Rscript failed: {e}"
except subprocess.CalledProcessError as exc:
    stderr = getattr(exc, "stderr", b"")
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    bsts_reason = (
        f"Rscript failed (exit code {exc.returncode}). Did you install R packages? Error: {stderr.strip()}"
    )
except FileNotFoundError as exc:
    bsts_reason = f"Rscript command not found: {exc}"
except Exception as exc:
    bsts_reason = f"BSTS via Rscript failed: {exc}"

# ---------------------------
# Save & Print Results
# ---------------------------
out = {
    "did_att": did_att,
    "did_se": did_se,
    "psm_att": (None if psm_att is None or (isinstance(psm_att, float) and np.isnan(psm_att)) else float(psm_att)),
    "psm_att": None
    if psm_att is None or (isinstance(psm_att, float) and np.isnan(psm_att))
    else float(psm_att),
    "bsts_att": bsts_att,
    "meta": {
        "psm_reason": psm_reason,
        "psm_diagnostics": psm_diag,
        "bsts_reason": bsts_reason,
        "n_rows": int(len(df)),
        "n_regions": int(df["region"].nunique()),
    },
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

(results_dir / "results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
