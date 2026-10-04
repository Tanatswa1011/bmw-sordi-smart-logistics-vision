"""Paths and config loading shared by the pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"
METRICS = OUTPUTS / "metrics"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
LOGS = OUTPUTS / "logs"
SPLITS = ROOT / "data" / "splits"
KAGGLE_TOKEN = Path.home() / ".kaggle" / "kaggle.json"
ACCESS_TOKEN = Path.home() / ".kaggle" / "access_token"
KAGGLE_TOKEN_MESSAGE = (
    "Kaggle token missing. Go to Kaggle → Settings → Create New Token, "
    "then place kaggle.json in C:\\Users\\tanam\\.kaggle\\kaggle.json."
)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def classes_config() -> dict:
    return load_yaml(CONFIGS / "classes.yaml")


def train_config() -> dict:
    return load_yaml(CONFIGS / "train.yaml")


def class_names() -> list[str]:
    names = classes_config()["classes"]
    expected = ["pallet", "stillage", "forklift", "dolly"]
    if list(names) != expected:
        raise ValueError(f"configs/classes.yaml must list {expected} in that order")
    return list(names)


def class_to_id() -> dict[str, int]:
    return {name: index for index, name in enumerate(class_names())}


def require_kaggle_token() -> Path:
    if os.environ.get("KAGGLE_API_TOKEN", "").strip():
        return ACCESS_TOKEN
    if ACCESS_TOKEN.is_file() and ACCESS_TOKEN.read_text(encoding="utf-8").strip():
        return ACCESS_TOKEN
    if KAGGLE_TOKEN.is_file():
        return KAGGLE_TOKEN
    raise SystemExit(KAGGLE_TOKEN_MESSAGE)


def kaggle_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("KAGGLE_API_TOKEN", "").strip() and ACCESS_TOKEN.is_file():
        env["KAGGLE_API_TOKEN"] = ACCESS_TOKEN.read_text(encoding="utf-8").strip()
    return env


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
