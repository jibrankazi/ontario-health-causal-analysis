"""Reproducible, non-causal February 2021 PHU reopening descriptive audit.

Uses official Ontario PHU ACTIVE_CASES DAILY snapshot totals. Does NOT equate
daily active-case stock with incidence, infection risk, or policy impact.
"""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from scripts.official_phu_crosscheck import (
    OFFICIAL, PUBLISHER_PAGE, documented_february_2021_rollout, phu_code,
)

PRE = pd.date_range("2021-02-03", "2021-02-09", freq="D")
POST = pd.date_range("2021-02-10", "2021-02-15", freq="D")
DAYS = set(PRE) | set(POST)


def descriptive_comparison(official, early, later, shutdown):
    """Compare recorded PHU active-case stocks; never infer intervention effect."""
    codes = set(early) | set(later) | set(shutdown)
    if len(early) != 3 or len(later) != 27 or len(shutdown) != 4 or len(codes) != 34:
        raise ValueError("Expected government-documentary 3/27/4 reopening cohorts")
    df = official.copy()
    numer = pd.to_numeric(df.PHU_NUM, errors="coerce")
    df = df.loc[numer.notna() & (numer % 1).eq(0)].copy()
    df["code"] = phu_code(df.PHU_NUM)
    df["date"] = pd.to_datetime(df.FILE_DATE, errors="raise").dt.normalize()
    df = df.loc[df.code.isin(codes) & df.date.isin(DAYS)].copy()
    df["active"] = pd.to_numeric(df.ACTIVE_CASES, errors="raise")
    if not np.isfinite(df.active.to_numpy()).all() or df.active.lt(0).any():
        raise ValueError("Official active-case stock must be finite nonnegative")
    if df[["code", "date"]].duplicated().any():
        raise ValueError("Duplicate official PHU-day observations")
    # Official data can omit entire dates or report only some PHUs. Only
    # dates with ALL 34 recorded codes are retained. Never fill gaps or
    # silently compare different calendar days between cohorts.
    coverage = df.groupby("date").code.nunique()
    retained_dates = set(coverage.index[coverage.eq(34)])
    missing_days = sorted(DAYS - retained_dates)
    observed_pre = sorted(retained_dates & set(PRE))
    observed_post = sorted(retained_dates & set(POST))
    if len(observed_pre) < 3 or len(observed_post) < 3:
        raise ValueError("Insufficient complete government PHU daily dates for comparison")
    df = df.loc[df.date.isin(retained_dates)].copy()
    if len(df) != 34 * len(retained_dates):
        raise ValueError("Unexpected PHU/date coverage after complete-date filtering")
    df["cohort"] = np.select([
        df.code.isin(early), df.code.isin(later), df.code.isin(shutdown)
    ], ["Feb10 early", "Feb16 later", "Shutdown past Feb16"], default="ERROR")
    df["period"] = np.where(df.date.isin(PRE), "7 days before", "6 days Feb10-15")
    per_phu = df.groupby(["code", "cohort", "period"], as_index=False).active.mean()
    pivot = per_phu.pivot(index=["code", "cohort"], columns="period", values="active")
    pivot["observed_change_in_mean_active_stock"] = (
        pivot["6 days Feb10-15"] - pivot["7 days before"])
    pivot = pivot.reset_index()
    cohort = {}
    for label, count in [("Feb10 early", 3), ("Feb16 later", 27),
                         ("Shutdown past Feb16", 4)]:
        group = pivot.loc[pivot.cohort.eq(label)]
        if len(group) != count:
            raise ValueError("Official cohort count inconsistent")
        cohort[label] = {
            "n_phus": count,
            "mean_of_phu_preperiod_daily_active_case_stocks":
                float(group["7 days before"].mean()),
            "mean_of_phu_postperiod_daily_active_case_stocks":
                float(group["6 days Feb10-15"].mean()),
            "equal_phu_weighted_mean_within_unit_stock_change":
                float(group["observed_change_in_mean_active_stock"].mean()),
        }
    descriptive = (cohort["Feb10 early"]["equal_phu_weighted_mean_within_unit_stock_change"]
                   - cohort["Feb16 later"]["equal_phu_weighted_mean_within_unit_stock_change"])
    return {
        "measured_official_phu_day_rows": int(len(df)),
        "complete_source_observation_dates_used": len(retained_dates),
        "complete_pre_dates_used": [x.date().isoformat() for x in observed_pre],
        "complete_post_dates_used": [x.date().isoformat() for x in observed_post],
        "unavailable_or_partially_covered_dates_excluded": [x.date().isoformat() for x in missing_days],
        "unique_government_phu_ids": 34,
        "pre_days": [x.date().isoformat() for x in PRE],
        "post_days": [x.date().isoformat() for x in POST],
        "cohort_descriptive_active_stock_statistics": cohort,
        "early_minus_later_difference_in_within_phu_changes": float(descriptive),
        "outcome": "ACTIVE_CASES daily stock (not incident infections, rates or policy effects)",
        "aggregation": "Equal weight to each original PHU; within PHU mean of pre vs six day post stock",
        "not_a_difference_in_differences_causal_estimate": True,
        "no_unobserved_values_imputed": True,
        "no_effect_established": True,
        "confounding": "Feb 10 regions chosen due to already improving conditions; no randomized assignment, population adjustment, or credible parallel-trend evidence. Six-day exposure window is shorter than plausible infection-to-report delays.",
    }


def run(output="results/official_reopening_daily_descriptive.json", client=None):
    session = client or requests
    resp = session.get(OFFICIAL, timeout=90)
    resp.raise_for_status()
    raw = resp.content
    govt = pd.read_csv(BytesIO(raw), low_memory=False)
    checked = pd.read_csv("data/ontario_cases.csv")
    cohorts = documented_february_2021_rollout(govt, checked)
    result = descriptive_comparison(
        govt, set(cohorts["early_reopening_three_phu_codes"]),
        set(cohorts["feb_16_reopening_27_phu_codes"]),
        set(cohorts["remaining_four_shutdown_phu_codes"]))
    result.update({
        "publisher_url": PUBLISHER_PAGE, "raw_publisher_url": OFFICIAL,
        "raw_publisher_csv_sha256": hashlib.sha256(raw).hexdigest(),
        "documented_policy_dates": [
            cohorts["date_of_documented_early_reopening"],
            cohorts["date_of_27_other_regions_reopening"]],
        "policy_sources": cohorts["documented_rollout_sources"],
        "old_treated_flags_identify_early_regions":
            cohorts["original_treated_labels_identify_documented_early_cohort"],
        "checked_in_weekly_incidence_provenance_verified": False,
        "causal_reopening_effect_verified": False,
    })
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/official_reopening_daily_descriptive.json")
    run(p.parse_args().output)
