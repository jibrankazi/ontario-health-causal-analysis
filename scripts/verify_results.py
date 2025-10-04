# scripts/verify_results.py

import json
import math
import sys
from pathlib import Path

# Add the project root to the Python path to allow imports from other folders like 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------
# Paths
# ---------------------------
CURRENT_RESULTS_PATH = PROJECT_ROOT / "results" / "results.json"
EXPECTED_RESULTS_PATH = PROJECT_ROOT / "results" / "expected_results.json"

def are_close(a: float | None, b: float | None, rel_tol: float = 1e-6) -> bool:
    """Compares two floats for near-equality, handling None and NaN."""
    if a is None or b is None:
        return a == b
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rel_tol)

def main() -> int:
    """
    Compares current analysis results with expected results.
    Exits with code 0 on success, 1 on failure, 2 if expected file is missing.
    """
    if not CURRENT_RESULTS_PATH.exists():
        print(f"❌ Error: Current results file not found at '{CURRENT_RESULTS_PATH}'")
        return 1
        
    if not EXPECTED_RESULTS_PATH.exists():
        print(f"⚠️ Warning: No expected results file found at '{EXPECTED_RESULTS_PATH}'.")
        print("Please review the current results and save them as the expected results if they are correct.")
        return 2

    print(f"Comparing '{CURRENT_RESULTS_PATH.name}' with '{EXPECTED_RESULTS_PATH.name}'...")

    current_results = json.loads(CURRENT_RESULTS_PATH.read_text())
    expected_results = json.loads(EXPECTED_RESULTS_PATH.read_text())

    keys_to_check = ("did_att", "did_se", "psm_att", "bsts_att")
    mismatches = []

    for key in keys_to_check:
        current_val = current_results.get(key)
        expected_val = expected_results.get(key)
        
        if not are_close(current_val, expected_val):
            mismatches.append(f"  - Mismatch for '{key}':\n    Got:      {current_val}\n    Expected: {expected_val}")

    if not mismatches:
        print("✅ OK: Results match the expected values.")
        return 0
    else:
        print("❌ FAIL: Found mismatches in results:")
        for mismatch in mismatches:
            print(mismatch)
        return 1

if __name__ == "__main__":
    # The `raise SystemExit(...)` pattern is a robust way to exit
    # with a specific status code from a script.
    raise SystemExit(main())
