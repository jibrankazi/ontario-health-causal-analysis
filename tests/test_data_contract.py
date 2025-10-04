import pandas as pd
from pathlib import Path

REQUIRED = {"week", "region", "incidence", "treated"}

def test_official_data_contract():
    p = Path("data/ontario_cases.csv")
    assert p.exists(), "Missing data/ontario_cases.csv"
    df = pd.read_csv(p)
    assert REQUIRED.issubset(df.columns), f"Missing columns: {REQUIRED - set(df.columns)}"
    pd.to_datetime(df["week"])  # raises if malformed
