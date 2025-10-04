# scripts/build_html_report.py
import sys
from pathlib import Path

# Add project root to sys.path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    # Assuming your report building logic lives here
    # You may need to adjust this import based on your actual file structure
    from src.reporting import html_builder
except ImportError:
    print("Error: Could not import report building module.")
    print("Please ensure your report logic is importable (e.g., in src/reporting/html_builder.py).")
    sys.exit(1)


def build_report():
    """Runs the HTML report generation logic."""
    print("--- Building HTML Report ---")
    try:
        # Assuming the function generates the report at reports/report.html
        html_builder.build_report() 
        print("--- HTML Report Built Successfully (Saved to reports/report.html) ---")
    except Exception as e:
        print(f"Error during report generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # This is a placeholder for the actual function call
    # You must ensure the actual function is implemented in your src/
    print("WARNING: This script requires a 'build_report()' function to be implemented in your source code.")
    
    # In a real environment, you would uncomment the line below:
    # build_report()
