"""Software fixtures; do not represent generated fixture numbers as Ontario cases."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from scripts.real_phu_identification_audit import (
    analyze_panel, load_checked_in_panel, regression_result,
)


def make_software_fixture():
    dates=pd.date_range("2020-03-30",periods=50,freq="W-MON")
    records=[]
    for r in range(24):
        treated=int(r < 12)
        for n,week in enumerate(dates):
            # deterministic unit-test fixtures, NOT empirical Ontario data
            records.append({
                "week":week, "region":str(2200+r), "incidence":
                    float(20 + 2*r + .25*n + treated*.18*n +
                          (treated*4 if n >= 35 else 0)),
                "treated":treated,
            })
    return pd.DataFrame(records)


def test_genuine_checked_in_panel_has_full_weekly_survey_calendar():
    df,digest=load_checked_in_panel()
    assert len(df)==4760
    assert df.region.nunique()==34
    assert df.week.nunique()==140
    assert str(df.week.min().date())=="2020-03-30"
    assert str(df.week.max().date())=="2022-11-28"
    assert len(digest)==64
    assert df[["week","region"]].duplicated().sum()==0
    assert not df.incidence.isna().any()


def test_checked_in_source_fails_closed_on_treatment_and_calendar_corruption(tmp_path):
    frame=make_software_fixture()
    p=tmp_path/"fake_unit_test_cases.csv"
    frame.to_csv(p,index=False)
    good,_=load_checked_in_panel(p)
    assert len(good)==len(frame)
    broken=frame.copy()
    broken.loc[12,"week"]=broken.loc[11,"week"]
    broken.to_csv(p,index=False)
    with pytest.raises(ValueError,match="Repeated public health unit"):
        load_checked_in_panel(p)
    broken=frame.copy()
    broken.loc[0,"incidence"]=-1
    broken.to_csv(p,index=False)
    with pytest.raises(ValueError,match="negative"):
        load_checked_in_panel(p)
    broken=frame.copy()
    broken.loc[24,"treated"]=1-broken.loc[24,"treated"]
    broken.to_csv(p,index=False)
    with pytest.raises(ValueError,match="treatment label changed"):
        load_checked_in_panel(p)


def test_observed_diagnostics_do_not_call_pretrend_or_placebo_a_policy_effect():
    fixture=make_software_fixture()
    result=analyze_panel(
        fixture,
        cutoff="2020-12-14",
        placebo="2020-09-07",
    )
    assert result["panel"]["complete_balanced_phu_week_grid"]
    assert result["panel"]["phus_labeled_treated"]==12
    assert result["panel"]["phus_labeled_comparison"]==12
    for key in (
        "pre_policy_linear_treated_minus_comparison_trend_per_week",
        "pre_policy_placebo_treated_group_difference_in_differences",
    ):
        outcome=result[key]
        assert outcome["not_a_causal_effect"] is True
        assert math.isfinite(outcome["estimate"])
        assert outcome["phu_cluster_robust_standard_error"]>0
        assert outcome["approximate_normal_95pct_interval"][0] < outcome["approximate_normal_95pct_interval"][1]
    assert result["panel"]["original_policy_applies_only_to_treated_group_independently_verified"] is False
    assert result["panel"]["publisher_record_level_provenance_independently_verified"] is False


def test_requires_actual_pre_policy_cutoff_and_preobservations():
    frame=make_software_fixture()
    with pytest.raises(ValueError,match="strictly inside"):
        analyze_panel(frame,cutoff="2020-12-14",placebo="2021-03-01")
    with pytest.raises(ValueError,match="too few"):
        analyze_panel(frame,cutoff="2020-06-01",placebo="2020-04-13")
