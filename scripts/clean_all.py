# scripts/clean_all.py
import shutil
from pathlib import Path

def clean_and_recreate_directories(dirs):
    """Deletes and recreates a list of directories."""
    print("--- Cleaning and Recreating Output Directories ---")
    for d in dirs:
        p = Path(d)
        
        # 1. Delete the directory if it exists
        if p.exists() and p.is_dir():
            print(f"  - Deleting existing directory: {d}")
            shutil.rmtree(p)
        
        # 2. Recreate the directory
        print(f"  - Creating fresh directory: {d}")
        p.mkdir(parents=True, exist_ok=False)

if __name__ == "__main__":
    directories_to_clean = [
        "results", 
        "figures", 
        "reports"
    ]
    clean_and_recreate_directories(directories_to_clean)
    print("--- Cleanup Complete ---")
