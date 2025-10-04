
from src.shared import resolve_intervention_date

def test_intervention_prefers_modern_key():
    cfg = {"intervention_date":"2021-02-15", "policy_date":"2021-02-01"}
    assert str(resolve_intervention_date(cfg).date()) == "2021-02-15"

def test_legacy_policy_date_still_supported():
    cfg = {"policy_date":"2021-02-01"}
    assert str(resolve_intervention_date(cfg).date()) == "2021-02-01"
