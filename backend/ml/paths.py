"""Paths work both from the repository and the /app Docker layout."""
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR.parent.parent / "data"
ARTIFACT_DIR = ML_DIR / "artifacts"
