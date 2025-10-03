import json, os, random, tempfile, subprocess
import json, os, random, shutil, tempfile, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

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
# Config
ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

def _resolve_intervention_date(config: dict, default: str = "2021-02-01") -> str:
    for key in ("intervention_date", "policy_date"):
        if config.get(key):
            return str(config[key])
    return default

def _resolve_rscript(config: dict) -> str:
    def _ok(p):
        p = Path(p).expanduser() if p else None
        return p.as_posix() if p and p.exists() else None

    k = _ok(config.get("rscript_path"))
    if k: return k

    r_home = config.get("r_home") or os.environ.get("R_HOME")
    if r_home:
        bin_name = "Rscript.exe" if os.name == "nt" else "Rscript"
        for rel in (Path("bin")/bin_name, Path("bin")/"x64"/bin_name):
            k = _ok(Path(r_home)/rel)
            if k: return k

    which_name = "Rscript.exe" if os.name == "nt" else "Rscript"
    via = shutil.which(which_name)
    if via: return Path(via).as_posix()

    if os.name == "nt":
        cands = []
        for env_key in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(env_key)
            if not base: continue
            root = Path(base) / "R"
            if not root.exists(): continue
            cands += sorted(root.glob("R-*/bin/Rscript.exe"))
            cands += sorted(root.glob("R-*/bin/x64/Rscript.exe"))
        if cands:
            cands.sort()
            return cands[-1].as_posix()

    raise FileNotFoundError(
        "Could not find Rscript. Set rscript_path or r_home in config.yaml, set R_HOME, or add R to PATH."
    )

# Paths
data_path = ROOT / cfg.get("data_path", "data/ontario_cases.csv")
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load Data
# ---------------------------
import pandas as pd
# Load data
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
resolved_policy = pd.Timestamp(_resolve_intervention_date(cfg))
if "post" not in df.columns:
    policy_date = pd.Timestamp(cfg.get("policy_date", "2021-02-01"))
    policy_date = resolved_policy
    df["post"] = (df["week"] >= policy_date).astype(int)
else:
    policy_date = pd.Timestamp(df.loc[df["post"] == 1, "week"].min())
    if pd.isna(policy_date):
        policy_date = resolved_policy

# ---------------------------
# Difference-in-Differences
# ---------------------------
import statsmodels.formula.api as smf

# DiD
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

# PSM
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

    if n_treat_cs == 0 or n_ctrl_cs == 0:
        psm_reason = "Common-support filter removed all treated or control units; skipping PSM."
        raise RuntimeError(psm_reason)

    # Caliper based on 0.2 * SD(logit(ps))
    eps = 1e-6
    logit = np.log(np.clip(pre_cs["ps"], eps, 1 - eps)) - np.log(1 - np.clip(pre_cs["ps"], eps, 1 - eps))
    caliper = 0.2 * float(np.nanstd(logit))
    psm_diag.update(caliper=caliper)

    # 1-NN with replacement on controls
    controls = pre_cs[pre_cs["treated"] == 0].copy()
    treats   = pre_cs[pre_cs["treated"] == 1].copy()
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(controls[["ps"]].to_numpy())
    dist, idx = nn.kneighbors(treats[["ps"]].to_numpy())
    dist = dist.flatten(); idx = idx.flatten()

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
    pass  # reason captured above

# ---------------------------
# BSTS / CausalImpact via Rscript (Windows-friendly)
# ---------------------------
# BSTS via Rscript (zoo + Date periods)
bsts_att = None
bsts_reason = None

def _bsts_via_rscript(agg_df: pd.DataFrame) -> float:
    """
    Run CausalImpact in an Rscript subprocess and return the
    average absolute effect (bsts_att).
    Requires Rscript on PATH and packages: CausalImpact, bsts, Boom, BoomSpikeSlab, zoo.
    """
def _bsts_via_rscript(agg_df: pd.DataFrame) -> float | None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_p = td / "series.csv"
        out_p = td / "out.json"
        # Save weekly mean series with columns: week, incidence

        agg_df = agg_df.copy()
        if pd.api.types.is_datetime64_any_dtype(agg_df["week"]):
            agg_df["week"] = agg_df["week"].dt.date
        agg_df.to_csv(csv_p, index=False)
        policy = pd.Timestamp(cfg.get("policy_date", "2021-02-01")).date()
        policy = pd.Timestamp(_resolve_intervention_date(cfg)).date()

        r_code = f"""
            suppressMessages(library(CausalImpact))
            suppressMessages(library(jsonlite))
            suppressMessages(library(zoo))

            dat <- read.csv("{csv_p.as_posix()}")
            dat$week <- as.Date(dat$week)
            pre_end  <- as.Date("{policy}") - 1
            post_end <- max(dat$week, na.rm=TRUE)
            ci <- CausalImpact(dat$incidence, c(min(dat$week, na.rm=TRUE), pre_end), c(pre_end+1, post_end))
            res <- list(bsts_att = as.numeric(ci$summary$AbsEffect["Average"]))
            write(jsonlite::toJSON(res, auto_unbox=TRUE), "{out_p.as_posix()}")

            ts <- zoo(dat$incidence, order.by = dat$week)

            pre_start  <- min(dat$week, na.rm=TRUE)
            pre_end    <- as.Date("{policy}") - 1
            post_start <- pre_end + 1
            post_end   <- max(dat$week, na.rm=TRUE)

            ci <- CausalImpact(ts, c(pre_start, pre_end), c(post_start, post_end))

            att <- as.numeric(ci$summary$AbsEffect["Average"])
            res <- list(bsts_att = att)
            write(jsonlite::toJSON(res, auto_unbox=TRUE, na="null"), "{out_p.as_posix()}")
        """
        r_script = td / "run_ci.R"
        r_script.write_text(r_code)
        subprocess.check_call(["Rscript", r_script.as_posix()])
        rscript_exec = _resolve_rscript(cfg)
        subprocess.check_call([rscript_exec, r_script.as_posix()])
        out = json.loads(out_p.read_text())
        return float(out["bsts_att"])
        val = out.get("bsts_att", None)
        return None if val is None else float(val)

try:
    # Aggregate to weekly mean incidence for univariate CausalImpact
    agg = df.sort_values("week").groupby("week", as_index=False)["incidence"].mean()
    bsts_att = _bsts_via_rscript(agg)
except subprocess.CalledProcessError as e:
    err = getattr(e, "stderr", b"")
    if isinstance(err, bytes):
        err = err.decode(errors="replace")
    bsts_reason = f"Rscript failed (exit code {e.returncode}). Did you install R packages? Error: {err}"
except FileNotFoundError as e:
    bsts_reason = f"Rscript command not found: {e}"
except Exception as e:
    bsts_reason = f"BSTS via Rscript failed: {e}"

# ---------------------------
# Save & Print Results
# ---------------------------
# Save
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
