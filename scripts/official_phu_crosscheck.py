"""Independently inspect Ontario's official PHU COVID snapshots.

Official ACTIVE_CASES snapshots and a checked-in, undocumented 'incidence'
column are NOT interchangeable. The audit measures identity/date overlap
without making a false case-count or causal-policy provenance claim.
"""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import requests

OFFICIAL = ("https://data.ontario.ca/dataset/"
            "1115d5fe-dd84-4c69-b5ed-05bf0c0a0ff9/resource/"
            "d1bfe1ad-6575-4352-8302-09ca81f7ddfc/download/"
            "cases_by_status_and_phu.csv")
PUBLISHER_PAGE = ("https://data.ontario.ca/dataset/"
                  "status-of-covid-19-cases-in-ontario-by-public-health-unit-phu")


def phu_code(values):
    nums = pd.to_numeric(values, errors="raise")
    if nums.isna().any() or (nums % 1).ne(0).any():
        raise ValueError("PHU IDs must be nonmissing integers")
    return nums.astype("int64").astype(str)


def compare_official(checked_in, official, *, official_hash):
    local = checked_in.copy()
    govt = official.copy()
    if not {"week", "region", "incidence", "treated"}.issubset(local):
        raise ValueError("Checked-in panel schema missing")
    needed = {"FILE_DATE", "PHU_NUM", "ACTIVE_CASES", "RESOLVED_CASES", "DEATHS"}
    if not needed.issubset(govt):
        raise ValueError("Official PHU snapshot schema differs from published dictionary")
    local["code"] = phu_code(local.region)
    govt["code"] = phu_code(govt.PHU_NUM)
    local["week"] = pd.to_datetime(local.week, errors="raise").dt.normalize()
    govt["FILE_DATE"] = pd.to_datetime(govt.FILE_DATE, errors="raise").dt.normalize()
    if len(local) < 1000 or len(govt) < 100 or local.code.nunique() < 20 or govt.code.nunique() < 20:
        raise ValueError("Insufficient PHU original or publisher observations")
    if govt[["FILE_DATE", "code"]].duplicated().any():
        raise ValueError("Duplicate published date and PHU code, ambiguous join")
    codes_local = set(local.code)
    codes_official = set(govt.code)
    dates_local = set(local.week)
    dates_official = set(govt.FILE_DATE)
    shared_codes = codes_local & codes_official
    shared_dates = dates_local & dates_official
    return {
        "official_dataset_page": PUBLISHER_PAGE,
        "official_raw_csv_url": OFFICIAL,
        "official_csv_sha256": official_hash,
        "official_phu_snapshot_rows": int(len(govt)),
        "official_distinct_phu_codes": len(codes_official),
        "official_first_date": str(govt.FILE_DATE.min().date()),
        "official_last_date": str(govt.FILE_DATE.max().date()),
        "checked_in_panel_rows": int(len(local)),
        "checked_in_region_codes": len(codes_local),
        "same_phu_code_count": len(shared_codes),
        "checked_in_phu_codes_absent_from_official": sorted(codes_local - codes_official),
        "common_calendar_mondays": len(shared_dates),
        "checked_in_earliest_week_precedes_official_snapshots":
            bool(local.week.min() < govt.FILE_DATE.min()),
        "source_is_daily_status_snapshot_not_weekly_incidence": True,
        "official_source_is_independently_downloaded": True,
        "original_incidence_units_and_derivation_verified": False,
        "checked_in_weekly_incidence_values_row_reconciled": False,
        "feb_2021_policy_treated_vs_control_assignment_verified": False,
        "causal_effect_identified": False,
        "reason": "The government file contains daily active/resolved/death snapshot totals, whereas the repository labels values weekly 'incidence' without a documented outcome transform or PHU population denominator. Matching PHU codes/dates does not authenticate those incidence numbers, and a province-wide policy gives no verified untreated PHUs."
    }


def run(panel_path="data/ontario_cases.csv",
        output="results/official_phu_comparison.json", session=None):
    checked = pd.read_csv(panel_path)
    client = session or requests
    report = {
        "official_dataset_page": PUBLISHER_PAGE,
        "official_raw_csv_url": OFFICIAL,
        "official_source_is_independently_downloaded": False,
        "checked_in_weekly_incidence_values_row_reconciled": False,
        "causal_effect_identified": False,
    }
    try:
        resp = client.get(OFFICIAL, timeout=60)
        resp.raise_for_status()
        raw = resp.content
        official = pd.read_csv(BytesIO(raw), low_memory=False)
        report = compare_official(checked, official,
                                  official_hash=hashlib.sha256(raw).hexdigest())
        report["official_source_retrieval_error"] = None
    except (requests.RequestException, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        report["official_source_retrieval_error"] = type(exc).__name__ + ": " + str(exc)[:180]
        report["reason"] = "Official CSV retrieval or schema verification was unsuccessful; no independent source verification."
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/ontario_cases.csv")
    parser.add_argument("--output", default="results/official_phu_comparison.json")
    args = parser.parse_args()
    run(args.panel, args.output)


if __name__ == "__main__":
    main()
