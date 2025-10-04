# tests/conftest.py
import sys
from pathlib import Path

# Put the project root (parent of /tests) at the front of sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
