"""Reassess inference and confidence using actual checked-in Ontario case rows.

Reads recomputed pipeline results; flags uncertainty instead of converting
a non-significant DiD estimate into a causal success claim. The dataset's
connection to its stated original public government source is NOT verified.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

def assess(data_path="data/ontario_cases.csv",results_path="results/results.json"):
    cases=pd.read_csv(data_path)
    must={"week","region","incidence","treated"}
    if not must.issubset(cases):
        raise ValueError("Original PHU case dataset columns missing")
    cases["week"]=pd.to_datetime(cases.week,errors="raise")
    cases["incidence"]=pd.to_numeric(cases.incidence,errors="raise")
    if len(cases)<1000 or cases["region"].nunique()<10:
        raise ValueError("Missing observed-like region incidence rows")
    metrics=json.loads(Path(results_path).read_text())
    att=float(metrics["did"]["att"])
    se=float(metrics["did"]["se"])
    if se<=0:
        raise ValueError("Invalid standard error")
    ci=[att-1.96*se,att+1.96*se]
    return {
        "dataset_path":data_path,
        "dataset_observation_rows":int(len(cases)),
        "unique_health_unit_codes":int(cases.region.nunique()),
        "first_week":str(cases.week.min().date()),
        "last_week":str(cases.week.max().date()),
        "did_att":att,
        "did_standard_error":se,
        "approximate_normal_95pct_interval":ci,
        "excludes_zero_at_95pct":bool(ci[0]>0 or ci[1]<0),
        "psm_point_estimate":metrics["psm"].get("att"),
        "psm_uncertainty_available":False,
        "bsts_estimate_available":metrics["bsts"].get("att") is not None,
        "checked_in_source_provenance_verified_with_ontario":False,
        "valid_conclusion":"Point estimates suggest lower incidence, but wide DiD interval includes zero; no statistically supported causal reduction established. Parallel-trends and confounding need separate diagnostics."
    }

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="results/independent_inference_review.json")
    a=p.parse_args()
    result=assess()
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
