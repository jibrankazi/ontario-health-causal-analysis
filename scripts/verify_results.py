import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cur_p = ROOT / "results" / "results.json"
exp_p = ROOT / "results" / "expected_results.json"

def close(a, b, tol=1e-4):
    if a is None or b is None:
        return a == b
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return abs(a - b) <= tol * max(1.0, abs(b))

cur = json.loads(cur_p.read_text())
if not exp_p.exists():
    print("No expected_results.json yet. Create it after reviewing current results.")
    raise SystemExit(2)
exp = json.loads(exp_p.read_text())

keys = ["did_att", "psm_att"]
ok = True
for k in keys:
    if k in exp:
        if not close(cur.get(k), exp.get(k)):
            print(f"Mismatch for {k}: got {cur.get(k)} expected {exp.get(k)}")
            ok = False
print("OK" if ok else "NOT OK")
raise SystemExit(0 if ok else 1)
