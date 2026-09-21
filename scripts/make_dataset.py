"""Feature engineering + chronological dataset splitting.

Usage (from project root):

    python scripts/make_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.features.make_dataset import main  # noqa: E402

if __name__ == "__main__":
    main(load_config())
