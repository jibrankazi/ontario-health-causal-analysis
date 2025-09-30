#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
echo "Done. See figures/ and results/results.json"
