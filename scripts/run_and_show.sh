#!/usr/bin/env bash
# Main analysis runner script
set -euo pipefail

# Detect project root
if command -v git &> /dev/null && git rev-parse --show-toplevel &> /dev/null; then
    ROOT_DIR="$(git rev-parse --show-toplevel)"
else
    # Fallback: assume script is in scripts/ subdirectory
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$ROOT_DIR"

echo "=========================================="
echo "Running Causal Analysis Pipeline"
echo "=========================================="
echo "Project root: $ROOT_DIR"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Check for required data file
if [ ! -f "data/ontario_cases.csv" ]; then
    echo "Error: data/ontario_cases.csv not found!"
    echo "Please run: Rscript scripts/fetch_data.R"
    exit 1
fi

# Run main analysis
echo "Running main analysis (src/run_analysis.py)..."
python -u src/run_analysis.py

# Check if analysis succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Analysis failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Analysis Complete"
echo "=========================================="
echo ""

# Display results if they exist
if [ -f "results/results.json" ]; then
    echo "Results (results/results.json):"
    echo "---"
    cat results/results.json
    echo ""
else
    echo "Warning: results/results.json not found"
fi

# Display summary statistics
if command -v jq &> /dev/null && [ -f "results/results.json" ]; then
    echo ""
    echo "=========================================="
    echo "Summary Statistics"
    echo "=========================================="
    jq -r '
        "DiD ATT:        \(.did.att // "N/A")",
        "DiD SE:         \(.did.se // "N/A")",
        "PSM ATT:        \(.psm.att // "N/A")",
        "BSTS ATT:       \(.bsts.att // "N/A")",
        "Intervention:   \(.metadata.intervention_date // "N/A")",
        "N Regions:      \(.metadata.n_regions // "N/A")",
        "N Observations: \(.metadata.n_rows // "N/A")"
    ' results/results.json
fi

echo ""
echo "✓ Pipeline completed successfully!"
