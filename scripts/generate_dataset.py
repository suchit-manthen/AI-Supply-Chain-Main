"""Generate the synthetic supermarket sales dataset.

Usage (from project root):

    python scripts/generate_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data.generate_data import main  # noqa: E402

if __name__ == "__main__":
    main(load_config())
