cat > scripts/run_all.py <<'PY'
import json, os, random, tempfile, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# --- Determinism ---
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)

# --- Config ---
import yaml
ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

data_path = ROOT / cfg.get("data_path", "data/ontario_cases.csv")
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# --- Load data ---
import pandas as pd
df = pd.read_csv(data_path)

# expected columns
required = {"week", "region", "incidence", "treated"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# parse dates
if pd.api.types.is_string_dtype(df["week"]):
    df["week"] = pd.to_datetime(df["week"], errors="coerce")

if "post" not in df.columns:
    policy_date = pd.Timestamp(cfg.get("policy_date", "2021-02-01"))
    df["post"] = (df["week"] >= policy_date).astype(int)

# --- DiD ---
import statsmodels.formula.api as smf
df["treat_post"] = df["treated"] * df["post"]
did = smf.ols("incidence ~ C(region) + C(week) + treat_post", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["region"]}
)
did_att = float(did.params.get("treat_post", float("nan")))
did_se  = float(did.bse.get("treat_post", float("nan")))

# --- PSM (robust) ---
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

psm_att = None
psm_reason = None
psm_diag = {}

try:
    pre = df[df["post"] == 0].copy()
    drop_cols = {"week","region","incidence","treated","post","treat_post"}
    covars = [c for c in pre.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(pre[c])]
    if not covars:
        covars = ["incidence"]

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

    ps_t = pre.loc[pre["treated"] == 1, "ps"]
    ps_c = pre.loc[pre["treated"] == 0, "ps"]
    overlap_low  = max(ps_t.min(), ps_c.min())
    overlap_high = min(ps_t.max(), ps_c.max())
    psm_diag.update(ps_overlap_low=float(overlap_low), ps_overlap_high=float(overlap_high))

    if overlap_low >= overlap_high:
        psm_reason = "No common support in propensity scores; skipping PSM."
        raise RuntimeError(psm_reason)

    pre_cs = pre[pre["ps"].between(overlap_low, overlap_high, inclusive="both")].copy()
    n_treat_cs = int(pre_cs["treated"].sum())
    n_ctrl_cs  = int((1 - pre_cs["treated"]).sum())
    psm_diag.update(n_treat_pre_cs=n_treat_cs, n_ctrl_pre_cs=n_ctrl_cs)

    if n_treat_cs == 0 or n_ctrl_cs == 0:
        psm_reason = "Common-support filter removed all treated or control units; skipping PSM."
        raise RuntimeError(psm_reason)

    eps = 1e-6
    logit = np.log(np.clip(pre_cs["ps"], eps, 1 - eps)) - np.log(1 - np.clip(pre_cs["ps"], eps, 1 - eps))
    caliper = 0.2 * float(np.nanstd(logit))
    psm_diag.update(caliper=caliper)

    controls = pre_cs[pre_cs["treated"] == 0].copy()
    treats   = pre_cs[pre_cs["treated"] == 1].copy()
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(controls[["ps"]].to_numpy())
    dist, idx = nn.kneighbors(treats[["ps"]].to_numpy())
    dist = dist.flatten(); idx = idx.flatten()

    ps_pairs = []
    for i, j in enumerate(idx):
        ps_ti = float(treats["ps"].iloc[i]); ps_ci = float(controls["ps"].iloc[j])
        if abs(ps_ti - ps_ci) <= max(0.1, caliper * 0.25):
            ps_pairs.append((treats.iloc[i], controls.iloc[j]))

    psm_diag.update(n_matched=len(ps_pairs))
    if len(ps_pairs) == 0:
        psm_reason = "No matches within caliper; skipping PSM."
        raise RuntimeError(psm_reason)

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
    pass

# --- BSTS / CausalImpact ---
bsts_att = None
bsts_reason = None

def try_rpy2_path(agg: pd.DataFrame):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    ro.r('suppressMessages(library(CausalImpact))')
    pandas2ri.activate()
    r_df = pandas2ri.py2rpy(agg)
    ro.globalenv["dat"] = r_df
    policy = pd.Timestamp(cfg.get("policy_date", "2021-02-01"))
    pre_end = agg[agg["week"] < policy]["week"].max()
    post_end = agg["week"].max()
    ro.globalenv["pre_end"] = ro.r(f'as.Date("{pre_end.date()}")')
    ro.globalenv["post_end"] = ro.r(f'as.Date("{post_end.date()}")')
    ci = ro.r('CausalImpact(dat$incidence, c(min(dat$week), pre_end), c(pre_end+1, post_end))')
    return float(ro.r('as.numeric(ci$summary$AbsEffect["Average"])')[0])

def try_rscript_path(agg: pd.DataFrame):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_p = td / "series.csv"
        out_p = td / "out.json"
        agg.to_csv(csv_p, index=False)
        policy = pd.Timestamp(cfg.get("policy_date", "2021-02-01")).date()
        r_code = f"""
            suppressMessages(library(CausalImpact))
            suppressMessages(library(jsonlite))
            dat <- read.csv("{csv_p.as_posix()}")
            dat$week <- as.Date(dat$week)
            pre_end  <- as.Date("{policy}") - 1
            post_end <- max(dat$week, na.rm=TRUE)
            ci <- CausalImpact(dat$incidence, c(min(dat$week, na.rm=TRUE), pre_end), c(pre_end+1, post_end))
            res <- list(bsts_att=as.numeric(ci$summary$AbsEffect["Average"]))
            write(jsonlite::toJSON(res, auto_unbox=TRUE), "{out_p.as_posix()}")
        """
        r_script = td / "run_ci.R"
        r_script.write_text(r_code)
        subprocess.check_call(["Rscript", r_script.as_posix()])
        out = json.loads(out_p.read_text())
        return float(out["bsts_att"])

try:
    agg = df.sort_values("week").groupby("week", as_index=False)["incidence"].mean()
    try:
        bsts_att = try_rpy2_path(agg)
    except Exception:
        bsts_att = try_rscript_path(agg)
except Exception as e:
    bsts_reason = "BSTS via rpy2/Rscript failed: " + str(e)

# --- Save results ---
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
PY
