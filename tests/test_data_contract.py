import pandas as pd
from pathlib import Path

REQUIRED = {"week", "region", "incidence", "treated"}

def test_official_data_contract():
    p = Path("data/ontario_cases.csv")
    assert p.exists(), "Missing data/ontario_cases.csv"
    df = pd.read_csv(p)
    assert REQUIRED.issubset(df.columns), f"Missing columns: {REQUIRED - set(df.columns)}"
    pd.to_datetime(df["week"])  # raises if malformed
import pandas as pd
from pathlib import Path

REQUIRED = {"week", "region", "incidence", "treated"}

def test_official_data_contract():
    """
    Tests for the presence of the expected data file and required columns.
    Also tests if the 'week' column can be successfully converted to datetime objects.
    """
    p = Path("data/ontario_cases.csv")
    assert p.exists(), "Missing data/ontario_cases.csv. Please ensure the data file is in the expected location."
    
    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise AssertionError(f"Failed to read CSV file: {e}")

    # Check required columns
    missing_cols = REQUIRED - set(df.columns)
    assert not missing_cols, f"Data file is missing required columns: {missing_cols}"
    
    # Check if 'week' column can be parsed as date
    try:
        pd.to_datetime(df["week"], errors='raise')
    except Exception as e:
        raise AssertionError(f"Failed to convert 'week' column to datetime. Data format issue: {e}")
