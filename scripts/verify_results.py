import json, os, random, tempfile, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

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

data_path = ROOT / cfg.get("data_path", "data/ontario_cases.csv")
results_dir = ROOT / "results"
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
if "post" not in df.columns:
    policy_date = pd.Timestamp(cfg.get("policy_date", "2021-02-01"))
    df["post"] = (df["week"] >= policy_date).astype(int)

# ---------------------------
# Difference-in-Differences
# ---------------------------
import statsmodels.formula.api as smf

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
from __future__ import annotations

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

    n_treat = int(pre["treated"].sum())
    n_ctrl  = int((1 - pre["treated"]).sum())
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
    overlap_high = min(ps_t.max(), ps_c.max())
    psm_diag.update(ps_overlap_low=float(overlap_low), ps_overlap_high=float(overlap_high))

    if overlap_low >= overlap_high:
        psm_reason = "No common support in propensity scores; skipping PSM."
        raise RuntimeError(psm_reason)

    # Restrict to support
    pre_cs = pre[pre["ps"].between(overlap_low, overlap_high, inclusive="both")].copy()
    n_treat_cs = int(pre_cs["treated"].sum())
    n_ctrl_cs  = int((1 - pre_cs["treated"]).sum())
    psm_diag.update(n_treat_pre_cs=n_treat_cs, n_ctrl_pre_cs=n_ctrl_cs)
import json
import math
import sys
from pathlib import Path

    if n_treat_cs == 0 or n_ctrl_cs == 0:
        psm_reason = "Common-support filter removed all treated or control units; skipping PSM."
        raise RuntimeError(psm_reason)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

    # Caliper based on 0.2 * SD(logit(ps))
    eps = 1e-6
    logit = np.log(np.clip(pre_cs["ps"], eps, 1 - eps)) - np.log(1 - np.clip(pre_cs["ps"], eps, 1 - eps))
    caliper = 0.2 * float(np.nanstd(logit))
    psm_diag.update(caliper=caliper)
from shared import ROOT as PROJECT_ROOT

    # 1-NN with replacement on controls
    controls = pre_cs[pre_cs["treated"] == 0].copy()
    treats   = pre_cs[pre_cs["treated"] == 1].copy()
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(controls[["ps"]].to_numpy())
    dist, idx = nn.kneighbors(treats[["ps"]].to_numpy())
    dist = dist.flatten(); idx = idx.flatten()
CURRENT = PROJECT_ROOT / "results" / "results.json"
EXPECTED = PROJECT_ROOT / "results" / "expected_results.json"

    # Simple ps-scale caliper gate (pragmatic bound)
    ps_pairs = []
    for i, j in enumerate(idx):
        ps_ti = float(treats["ps"].iloc[i]); ps_ci = float(controls["ps"].iloc[j])
        if abs(ps_ti - ps_ci) <= max(0.1, caliper * 0.25):
            ps_pairs.append((treats.iloc[i], controls.iloc[j]))

    psm_diag.update(n_matched=len(ps_pairs))
    if len(ps_pairs) == 0:
        psm_reason = "No matches within caliper; skipping PSM."
        raise RuntimeError(psm_reason)
def _close(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a == b
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(af) and math.isnan(bf):
        return True
    return abs(af - bf) <= tol * max(1.0, abs(bf))

    # Post-period ATT using matched sets
    post = df[df["post"] == 1].copy()
    post_mean = post.groupby("region", as_index=True)["incidence"].mean()
    diffs = []
    for t_row, c_row in ps_pairs:
        t_reg = t_row["region"]; c_reg = c_row["region"]
        if t_reg in post_mean.index and c_reg in post_mean.index:
            diffs.append(float(post_mean.loc[t_reg] - post_mean.loc[c_reg]))
    if len(diffs) == 0:
        psm_reason = "Matched regions missing from post-period; skipping PSM."
        raise RuntimeError(psm_reason)

    psm_att = float(np.mean(diffs))
except Exception:
    # leave psm_att as None; reason is recorded below
    pass
def main() -> int:
    if not EXPECTED.exists():
        print("No expected_results.json yet. Create it after reviewing current results.")
        return 2

# ---------------------------
# BSTS / CausalImpact via Rscript (Windows-friendly)
# ---------------------------
bsts_att = None
bsts_reason = None
    cur = json.loads(CURRENT.read_text())
    exp = json.loads(EXPECTED.read_text())

def _bsts_via_rscript(agg_df: pd.DataFrame) -> float:
    """
    Run CausalImpact in an Rscript subprocess and return the
    average absolute effect (bsts_att).
    Requires Rscript on PATH and packages: CausalImpact, bsts, Boom, BoomSpikeSlab, zoo.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_p = td / "series.csv"
        out_p = td / "out.json"
        # Save weekly mean series with columns: week, incidence
        agg_df.to_csv(csv_p, index=False)
        policy = pd.Timestamp(cfg.get("policy_date", "2021-02-01")).date()
        r_code = f"""
            suppressMessages(library(CausalImpact))
            suppressMessages(library(jsonlite))
            dat <- read.csv("{csv_p.as_posix()}")
            dat$week <- as.Date(dat$week)
            pre_end  <- as.Date("{policy}") - 1
            post_end <- max(dat$week, na.rm=TRUE)
            ci <- CausalImpact(dat$incidence, c(min(dat$week, na.rm=TRUE), pre_end), c(pre_end+1, post_end))
            res <- list(bsts_att = as.numeric(ci$summary$AbsEffect["Average"]))
            write(jsonlite::toJSON(res, auto_unbox=TRUE), "{out_p.as_posix()}")
        """
        r_script = td / "run_ci.R"
        r_script.write_text(r_code)
        subprocess.check_call(["Rscript", r_script.as_posix()])
        out = json.loads(out_p.read_text())
        return float(out["bsts_att"])
    keys = ("did_att", "did_se", "psm_att", "bsts_att")
    ok = True
    for key in keys:
        if not _close(cur.get(key), exp.get(key)):
            print(f"Mismatch for {key}: got {cur.get(key)} expected {exp.get(key)}")
            ok = False

try:
    # Aggregate to weekly mean incidence for univariate CausalImpact
    agg = df.sort_values("week").groupby("week", as_index=False)["incidence"].mean()
    bsts_att = _bsts_via_rscript(agg)
except Exception as e:
    bsts_reason = f"BSTS via Rscript failed: {e}"
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

# ---------------------------
# Save & Print Results
# ---------------------------
out = {
    "did_att": did_att,
    "did_se": did_se,
    "psm_att": (None if psm_att is None or (isinstance(psm_att, float) and np.isnan(psm_att)) else float(psm_att)),
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
if __name__ == "__main__":
    raise SystemExit(main())
