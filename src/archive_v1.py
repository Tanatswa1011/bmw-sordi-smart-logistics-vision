"""Copy the finished V1 artifacts into outputs/v1 without removing the originals."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.settings import FIGURES, METRICS, OUTPUTS, TABLES, read_json, write_json

V1_ROOT = OUTPUTS / "v1"


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def archive_v1() -> Path:
    metrics_dest = V1_ROOT / "metrics"
    tables_dest = V1_ROOT / "tables"
    metrics_dest.mkdir(parents=True, exist_ok=True)
    tables_dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in METRICS.iterdir():
        if path.is_file():
            _copy_file(path, metrics_dest / path.name)
            copied.append(str(path.relative_to(OUTPUTS)))
    for name in (
        "detections.csv",
        "detections.parquet",
        "false_negatives.csv",
        "false_negatives.parquet",
        "export_manifest.json",
    ):
        source = TABLES / name
        if source.is_file():
            _copy_file(source, tables_dest / name)
            copied.append(str(source.relative_to(OUTPUTS)))
    _copy_tree(FIGURES / "errors", V1_ROOT / "figures" / "errors")
    _copy_tree(FIGURES / "evaluation", V1_ROOT / "figures" / "evaluation")
    _copy_tree(FIGURES / "samples", V1_ROOT / "figures" / "samples")
    selection_path = metrics_dest / "model_selection.json"
    if not selection_path.is_file():
        raise SystemExit("V1 model_selection.json is missing, so the checkpoint reference was not invented.")
    selection = read_json(selection_path)
    reference = {
        "chosen_run": selection.get("chosen_run"),
        "chosen_weights": selection.get("chosen_weights"),
        "reason": selection.get("reason"),
        "selection_split": selection.get("selection_split"),
        "test_split_used": selection.get("test_split_used"),
        "note": "Path reference only. The weight file stays outside the repository.",
    }
    write_json(V1_ROOT / "checkpoint_reference.json", reference)
    write_json(
        V1_ROOT / "archive_manifest.json",
        {
            "copied_from": "outputs/metrics, outputs/tables, outputs/figures",
            "originals_removed": False,
            "files": sorted(copied),
            "checkpoint_reference": reference,
        },
    )
    print(f"Archived V1 references under {V1_ROOT}")
    return V1_ROOT


if __name__ == "__main__":
    archive_v1()
