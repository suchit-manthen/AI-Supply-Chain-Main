"""Central configuration loader for the AI Supply Chain project.

Loads ``config/config.yaml`` and exposes it as a nested ``dict``-like object.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin wrapper around a YAML config file with attribute-style access."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access with a default for optional keys."""
        return self.__dict__.get(key, default)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return repr(self.__dict__)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            result[key] = value.to_dict() if isinstance(value, Config) else value
        return result


def load_config(path: str | Path = "config/config.yaml") -> Config:
    """Load and parse the YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


# Convenience accessor used across the project.
def get_config(path: str | Path = "config/config.yaml") -> Config:
    return load_config(path)
