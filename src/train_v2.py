"""Train the three V2 YOLO11n experiments. The frozen V2 test split is not read."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.prepare_v2 import v2_config, yolo_v2_root
from src.settings import CONFIGS, OUTPUTS, materialize_dataset_yaml, read_json, write_json
from src.train import _base_weights, _cuda_or_stop, _git_commit, _is_oom

V2_METRICS = OUTPUTS / "v2" / "metrics"
AUGMENTATION_PLAN = OUTPUTS / "v2" / "analysis" / "augmentation_plan.json"


def _require_frozen_split() -> None:
    frozen = OUTPUTS / "v2" / "splits" / "frozen_test_ids.json"
    dataset = CONFIGS / "dataset_v2.yaml"
    if not frozen.is_file() or not dataset.is_file():
        raise SystemExit("V2 split is missing. Run src.prepare_v2 before training.")
    if not read_json(frozen).get("frozen"):
        raise SystemExit("V2 test ids are not marked frozen.")


def _split_counts() -> dict[str, int]:
    root = yolo_v2_root()
    counts = {}
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        counts[split] = len([path for path in image_dir.iterdir() if path.is_file()]) if image_dir.is_dir() else 0
    if counts["train"] == 0 or counts["val"] == 0:
        raise SystemExit("V2 train or validation images are missing.")
    return counts


def _epochs_completed(results_path: Path) -> int:
    if not results_path.is_file():
        return 0
    frame = pd.read_csv(results_path)
    if frame.empty or "epoch" not in frame.columns:
        return 0
    return int(frame["epoch"].max())


def _train_one(letter: str) -> Path:
    from ultralytics import YOLO

    _cuda_or_stop()
    _require_frozen_split()
    config = v2_config()
    experiment = config["experiments"][letter]
    project = Path(config["project_dir"])
    project.mkdir(parents=True, exist_ok=True)
    run_name = experiment["run_name"]
    best = project / run_name / "weights" / "best.pt"
    if best.is_file():
        print(f"{letter} already has {best}. Leaving it in place.")
        return best
    batch = int(config["batch"])
    fallback = int(config["batch_fallback"])
    kwargs = dict(
        data=str(materialize_dataset_yaml(CONFIGS / "dataset_v2.yaml")),
        epochs=int(experiment["epochs"]),
        imgsz=int(config["imgsz"]),
        batch=batch,
        seed=int(config["seed"]),
        device=0,
        workers=0,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        cos_lr=bool(experiment["cos_lr"]),
        patience=int(experiment["patience"]),
        pretrained=True,
        project=str(project),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
        cache=False,
        amp=True,
        resume=False,
    )
    if letter == "C":
        if not AUGMENTATION_PLAN.is_file():
            raise SystemExit("Experiment C needs outputs/v2/analysis/augmentation_plan.json from the V1 failure analysis.")
        plan = read_json(AUGMENTATION_PLAN)
        for key in (
            "degrees",
            "shear",
            "perspective",
            "flipud",
            "fliplr",
            "translate",
            "scale",
            "mosaic",
            "mixup",
            "copy_paste",
            "erasing",
            "hsv_h",
            "hsv_s",
            "hsv_v",
        ):
            kwargs[key] = plan[key]
    try:
        YOLO(str(_base_weights())).train(**kwargs)
    except RuntimeError as exc:
        if batch != fallback and _is_oom(exc):
            print(f"CUDA OOM at batch={batch}. Retrying {run_name} at batch={fallback}.")
            torch.cuda.empty_cache()
            kwargs["batch"] = fallback
            batch = fallback
            YOLO(str(_base_weights())).train(**kwargs)
        else:
            raise
    if not best.is_file():
        raise RuntimeError(f"Training finished but {best} was not written.")
    run_dir = project / run_name
    for filename in ("results.csv", "args.yaml"):
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, V2_METRICS / f"{run_name}_{filename}")
    counts = _split_counts()
    args_path = V2_METRICS / f"{run_name}_args.yaml"
    recorded = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.is_file() else {}
    import ultralytics

    write_json(
        V2_METRICS / f"run_meta_{run_name}.json",
        {
            "experiment": letter,
            "run_name": run_name,
            "description": experiment["description"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": config["model"],
            "epochs_requested": int(experiment["epochs"]),
            "epochs_completed": _epochs_completed(V2_METRICS / f"{run_name}_results.csv"),
            "imgsz": int(config["imgsz"]),
            "batch": batch,
            "seed": int(config["seed"]),
            "cos_lr": bool(experiment["cos_lr"]),
            "patience": int(experiment["patience"]),
            "augmentation": {key: recorded.get(key) for key in ("mosaic", "scale", "fliplr", "flipud", "degrees", "translate", "hsv_v", "mixup", "erasing")}
            if letter == "C"
            else "ultralytics_defaults",
            "train_images": counts["train"],
            "val_images": counts["val"],
            "test_images": counts["test"],
            "test_split_used_for_selection": False,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "ultralytics_version": ultralytics.__version__,
            "git_commit_if_available": _git_commit(),
            "weights": str(best),
        },
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2 experiments A, B, and C.")
    parser.add_argument("experiment", choices=["A", "B", "C", "all"])
    args = parser.parse_args()
    letters = ["A", "B", "C"] if args.experiment == "all" else [args.experiment]
    for letter in letters:
        print(f"Starting V2 experiment {letter}")
        _train_one(letter)


if __name__ == "__main__":
    main()
