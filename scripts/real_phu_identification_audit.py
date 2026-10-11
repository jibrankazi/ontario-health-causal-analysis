"""Audit *observed checked-in* Ontario PHU incidence panel and identification.

NO independent Government of Ontario raw-source reconciliation is claimed.
Neither 'treated' assignment nor province-wide 2021-02-01 policy exposure
has a cited documented treatment/control mechanism in this repository.

This computes source-data integrity, pre-policy group divergence, a
pre-policy PLACEBO cutoff, and cluster-robust diagnostics. A placebo
association is NOT a causal policy effect. Results are descriptive only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = "data/ontario_cases.csv"
POLICY_DATE = pd.Timestamp("2021-02-01")
PLACEBO_DATE = pd.Timestamp("2020-10-05")
REQUIRED = {"week","region","incidence","treated"}


def load_checked_in_panel(path=DATA_PATH):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Original checked-in panel is missing: {path}")
    data = pd.read_csv(p, dtype={"region":str}, low_memory=False)
    if not REQUIRED.issubset(data.columns):
        raise ValueError(f"Checked-in Ontario panel missing {sorted(REQUIRED-set(data.columns))}")
    if len(data) < 1000:
        raise ValueError("Too few checked-in real-like case observations")
    data["week"] = pd.to_datetime(data.week, errors="raise")
    data["incidence"] = pd.to_numeric(data.incidence, errors="raise")
    data["treated"] = pd.to_numeric(data.treated, errors="raise")
    if data[list(REQUIRED)].isna().any().any():
        raise ValueError("Missing incidence, date, treatment labels, or PHU code")
    if not np.isfinite(data.incidence).all() or data.incidence.lt(0).any():
        raise ValueError("Missing/negative/nonfinite checked-in case incidence")
    if not data.treated.isin([0,1]).all():
        raise ValueError("Treatment indicator is not binary")
    if data[["week","region"]].duplicated().any():
        raise ValueError("Repeated public health unit within one observation week")
    codes = data.groupby("region").treated.nunique()
    if (codes!=1).any():
        raise ValueError("PHU treatment label changed during historical panel")
    if data.region.nunique()<10 or data.week.nunique()<60:
        raise ValueError("Too few original distinct PHU regions or weeks")
    periods = data.week.sort_values().drop_duplicates().reset_index(drop=True)
    if not periods.diff().iloc[1:].eq(pd.Timedelta(days=7)).all():
        raise ValueError("Original PHU panel dates are not a complete weekly grid")
    if not data.week.dt.dayofweek.eq(0).all():
        raise ValueError("Original case dates must be Mondays")
    # Explicitly require every PHU-week: otherwise levels/trends are
    # not aligned and coefficient interpretation differs by group.
    if len(data)!=data.region.nunique()*len(periods):
        raise ValueError("Original source panel is incomplete/unbalanced")
    if not (data.treated.eq(0).any() and data.treated.eq(1).any()):
        raise ValueError("Need observed treated and comparison region labels")
    return data.sort_values(["week","region"]).reset_index(drop=True), sha256(p.read_bytes()).hexdigest()


def regression_result(model, parameter):
    if parameter not in model.params:
        raise ValueError(f"Regression did not estimate coefficient {parameter}")
    att=float(model.params[parameter])
    se=float(model.bse[parameter])
    if not np.isfinite(att) or not np.isfinite(se) or se <= 0:
        raise ValueError("Cluster-robust regression did not yield valid uncertainty")
    return {
        "estimate":att,
        "phu_cluster_robust_standard_error":se,
        "approximate_normal_95pct_interval":[att-1.96*se,att+1.96*se],
        "interval_contains_zero":bool(att-1.96*se<=0<=att+1.96*se),
        "not_a_causal_effect":True,
    }


def analyze_panel(df, cutoff=POLICY_DATE, placebo=PLACEBO_DATE):
    cutoff=pd.Timestamp(cutoff)
    placebo=pd.Timestamp(placebo)
    pre=df.loc[df.week < cutoff].copy()
    if len(pre) < 500:
        raise ValueError("Cannot assess pre-policy identification with too few observations")
    if not pre.week.min()<placebo<pre.week.max():
        raise ValueError("Placebo must be strictly inside observed PRE-intervention weeks")
    pre["week_index"] = (pre.week-pre.week.min()).dt.days/7
    pre["treated_time_interaction"] = pre.treated*pre.week_index
    trend = smf.ols(
        "incidence ~ C(region) + C(week) + treated_time_interaction",
        data=pre,
    ).fit(cov_type="cluster",cov_kwds={"groups":pre.region})
    pre["placebo_post"] = pre.week.ge(placebo).astype(int)
    pre["placebo_treated_post"] = pre.treated*pre.placebo_post
    placebo_fit=smf.ols(
        "incidence ~ C(region) + C(week) + placebo_treated_post",
        data=pre,
    ).fit(cov_type="cluster",cov_kwds={"groups":pre.region})
    all_groups=df.groupby(["week","treated"],as_index=False).agg(
        mean_phu_incidence=("incidence","mean"),
        median_phu_incidence=("incidence","median"),
        n_phus=("region","nunique"),
    )
    observed_group_mean=all_groups.pivot(
        index="week",columns="treated",values="mean_phu_incidence"
    )
    observed_group_mean.columns=["comparison_phu_mean","labeled_treated_phu_mean"]
    observed_group_mean["treated_minus_comparison_mean"]=(
        observed_group_mean["labeled_treated_phu_mean"]
        -observed_group_mean["comparison_phu_mean"])
    before=observed_group_mean.loc[observed_group_mean.index<cutoff]
    after=observed_group_mean.loc[observed_group_mean.index>=cutoff]
    if before.empty or after.empty:
        raise ValueError("Observed comparison has no full pre/post data")
    regions=df[["region","treated"]].drop_duplicates().sort_values("region")
    metadata={
        "checked_in_original_panel_rows":int(len(df)),
        "distinct_original_phu_codes":int(df.region.nunique()),
        "weekly_observation_dates":int(df.week.nunique()),
        "first_week":str(df.week.min().date()),
        "last_week":str(df.week.max().date()),
        "weeks_before_selected_cutoff":int(pre.week.nunique()),
        "weeks_on_or_after_selected_cutoff":int(df.week.nunique()-pre.week.nunique()),
        "phus_labeled_treated":int(regions.treated.eq(1).sum()),
        "phus_labeled_comparison":int(regions.treated.eq(0).sum()),
        "complete_balanced_phu_week_grid":True,
        "original_label_mapping_available_as_checked_in_file":True,
        "original_documented_treatment_assignment_mechanism_available":False,
        "original_policy_applies_only_to_treated_group_independently_verified":False,
        "publisher_record_level_provenance_independently_verified":False,
    }
    return {
        "panel":metadata,
        "cutoff_under_investigation":str(cutoff.date()),
        "pre_period_placebo_cutoff":str(placebo.date()),
        "pre_policy_linear_treated_minus_comparison_trend_per_week":
            regression_result(trend,"treated_time_interaction"),
        "pre_policy_placebo_treated_group_difference_in_differences":
            regression_result(placebo_fit,"placebo_treated_post"),
        "pre_policy_observed_group_mean_difference":
            float(before.treated_minus_comparison_mean.mean()),
        "post_policy_observed_group_mean_difference":
            float(after.treated_minus_comparison_mean.mean()),
        "unadjusted_change_in_observed_group_difference":float(
            after.treated_minus_comparison_mean.mean()
            -before.treated_minus_comparison_mean.mean()
        ),
        "group_series":observed_group_mean.reset_index(),
    }


def run(data_path=DATA_PATH, output="results/verified_phu_identification"):
    data,source_hash=load_checked_in_panel(data_path)
    result=analyze_panel(data)
    by_week=result.pop("group_series")
    target=Path(output)
    target.mkdir(parents=True,exist_ok=True)
    by_week["week"]=by_week["week"].dt.strftime("%Y-%m-%d")
    by_week.to_csv(target/"checked_in_original_phu_group_weekly_means.csv",index=False)
    original=data[["week","region","treated","incidence"]].copy()
    original["week"]=original.week.dt.strftime("%Y-%m-%d")
    original.to_csv(target/"original_phu_weekly_incidence_and_treatment_labels.csv",index=False)
    result.update({
        "analyzed_at_utc":datetime.now(timezone.utc).isoformat(),
        "original_checked_in_file":str(data_path),
        "original_checked_in_file_bytes_sha256":source_hash,
        "independent_ontario_publisher_crosscheck":False,
        "valid_inference":(
            "Observed within-file incidence trends and exploratory placebo "
            "diagnostics only. These estimates are NOT treatment effects: "
            "the original checked-in dataset lacks individually audited "
            "government provenance, independent public-health-unit treatment "
            "assignment evidence and a credible contemporaneous untreated "
            "comparison for province-wide policies. Parallel trends cannot "
            "be assumed based on any single regression, and the placebo "
            "cutoff was chosen for diagnostic illustration, not prespecified "
            "as a controlled statistical testing protocol."
        ),
    })
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(by_week.week,by_week.labeled_treated_phu_mean,
            label="PHUs labeled treated (original checked-in file)",lw=1)
    ax.plot(by_week.week,by_week.comparison_phu_mean,
            label="PHUs labeled comparison (original checked-in file)",lw=1)
    ax.axvline(str(POLICY_DATE.date()),linestyle="--",label="Assumed 2021-02-01 cutoff")
    ax.set(title="Original checked-in Ontario PHU weekly mean incidence by label",
           ylabel="Mean across PHUs of source incidence column",
           xlabel="Original weekly observation date")
    ax.set_xticks(by_week.week.iloc[::20])
    ax.tick_params(axis="x",labelrotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(target/"observed_original_phu_incidence_by_group.png",dpi=150)
    plt.close(fig)
    (target/"verified_identification_diagnostics.json").write_text(
        json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    return result


if __name__=="__main__":
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("--data",default=DATA_PATH)
    args.add_argument("--output",default="results/verified_phu_identification")
    a=args.parse_args()
    run(a.data,a.output)
