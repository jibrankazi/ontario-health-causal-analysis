"""Official-CSV parsing tests use fixtures, never fabricated publisher evidence."""
from hashlib import sha256

import pandas as pd
import pytest

from scripts.official_phu_crosscheck import compare_official, phu_code, documented_february_2021_rollout


def make_frames():
    phus = list(range(2201, 2231))
    weeks = pd.date_range("2020-03-30", periods=70, freq="W-MON")
    local = pd.DataFrame([
        {"week": day, "region": float(region), "incidence": 1.0,
         "treated": int(region<2206)} for day in weeks for region in phus
    ])
    govt = pd.DataFrame([
        {"FILE_DATE": day, "PHU_NUM": region, "PHU_NAME": "FIXTURE PHU " + str(region), "ACTIVE_CASES": 2,
         "RESOLVED_CASES": 3, "DEATHS": 0}
        for day in weeks[2:] for region in phus
    ])
    return local, govt


def test_phu_code_rejects_fractional_identifiers():
    assert phu_code(pd.Series([2226.0, "2227"])).tolist() == ["2226", "2227"]
    with pytest.raises(ValueError, match="integers"):
        phu_code(pd.Series([2226.5]))


def test_official_identity_overlap_does_not_validate_incidence():
    local, govt = make_frames()
    report = compare_official(local, govt, official_hash=sha256(b"test").hexdigest())
    assert report["same_phu_code_count"] == 30
    assert report["common_calendar_mondays"] == 68
    assert report["checked_in_earliest_week_precedes_official_snapshots"] is True
    assert report["official_source_is_independently_downloaded"] is True
    assert report["checked_in_weekly_incidence_values_row_reconciled"] is False
    assert report["causal_effect_identified"] is False


def test_refuse_wrong_publisher_schema_and_duplicates():
    local, govt = make_frames()
    with pytest.raises(ValueError, match="schema"):
        compare_official(local, govt.drop(columns=["PHU_NUM"]), official_hash="x")
    with pytest.raises(ValueError, match="Duplicate"):
        compare_official(local, pd.concat([govt, govt.iloc[:1]]),
                         official_hash="x")


def test_government_provincewide_rows_without_phu_id_are_excluded_not_fabricated():
    local, govt = make_frames()
    provincewide = govt.iloc[:1].copy()
    provincewide["PHU_NUM"] = float("nan")
    audited = compare_official(local, pd.concat([govt, provincewide]),
                               official_hash="testhash")
    assert audited["official_excluded_missing_or_invalid_phu_id_rows"] == 1
    assert audited["same_phu_code_count"] == 30
    assert audited["official_phu_snapshot_rows"] == len(govt)


def test_policy_reopening_dates_are_not_arbitrary_original_treated_flags():
    local, official = make_frames()
    extra_local = local.loc[local.region.eq(2201)].copy()
    extra_official = official.loc[official.PHU_NUM.eq(2201)].copy()
    for n in (2231, 2232, 2233, 2234):
        new_local = extra_local.copy()
        new_local["region"] = float(n)
        new_local["treated"] = 0
        local = pd.concat([local, new_local], ignore_index=True)
        new_official = extra_official.copy()
        new_official["PHU_NUM"] = n
        official = pd.concat([official, new_official], ignore_index=True)
    # Names are artificial schema fixtures, not observations about Ontario.
    label = {2201: "Hastings Prince Edward", 2202: "Kingston Frontenac",
             2203: "Renfrew County", 2204: "Toronto",
             2205: "Peel", 2206: "York", 2207: "North Bay Parry Sound"}
    official["PHU_NAME"] = official.PHU_NUM.map(
        lambda n: label.get(n, "FICTIONAL region " + str(n)))
    result = documented_february_2021_rollout(official, local)
    assert len(result["early_reopening_three_phu_codes"]) == 3
    assert len(result["feb_16_reopening_27_phu_codes"]) == 27
    assert result["date_of_documented_early_reopening"] == "2021-02-10"
    assert result["original_treated_labels_identify_documented_early_cohort"] is False
    assert result["not_an_identified_causal_design"] is True
