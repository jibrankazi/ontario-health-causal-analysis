"""Synthetic values below are structural fixtures, NOT Ontario observations."""
import pandas as pd
import pytest

from scripts.reopening_descriptive_study import descriptive_comparison

def frame():
    rows = []
    days = pd.date_range("2021-02-03", "2021-02-15", freq="D")
    for code in range(2200, 2234):
        for day in days:
            rows.append({"PHU_NUM": code, "FILE_DATE": day,
                         "ACTIVE_CASES": (100 + (5 if code in range(2200,2203) else 1)
                                          * (day.day - 3))})
    return pd.DataFrame(rows)

EARLY = {str(x) for x in range(2200,2203)}
LATER = {str(x) for x in range(2203,2230)}
SHUTDOWN = {str(x) for x in range(2230,2234)}

def test_real_policy_cohort_window_math_without_causal_claim():
    result = descriptive_comparison(frame(), EARLY, LATER, SHUTDOWN)
    assert result["measured_official_phu_day_rows"] == 34*13
    assert result["unique_government_phu_ids"] == 34
    assert result["no_effect_established"]
    assert result["no_unobserved_values_imputed"]
    assert result["not_a_difference_in_differences_causal_estimate"]
    assert result["early_minus_later_difference_in_within_phu_changes"] == pytest.approx(26)

def test_source_missing_day_cannot_be_imputed():
    original = frame().iloc[1:]
    with pytest.raises(ValueError, match="incomplete"):
        descriptive_comparison(original, EARLY, LATER, SHUTDOWN)

def test_bad_cohort_and_duplicate_and_negative_snapshot_refused():
    with pytest.raises(ValueError, match="3/27/4"):
        descriptive_comparison(frame(), set(), LATER, SHUTDOWN)
    bad = pd.concat([frame(),frame().iloc[:1]])
    with pytest.raises(ValueError, match="Duplicate"):
        descriptive_comparison(bad, EARLY, LATER, SHUTDOWN)
    bad = frame()
    bad.loc[0,"ACTIVE_CASES"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        descriptive_comparison(bad, EARLY, LATER, SHUTDOWN)
