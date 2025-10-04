# tests/conftest.py
import sys
from pathlib import Path

# Put the repo root (parent of /tests) on sys.path for imports like `from src...`
# Path(__file__).resolve() is the path to conftest.py
# .parents[1] is the path to the directory two levels up (the project root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
