"""Fine-tune YOLO11n, then run one cosine-LR tuning cycle. Test data is not used."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from src.settings import CONFIGS, METRICS, ROOT, class_names, train_config, write_json


def _cuda_or_stop() -> str:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Training was not started and will not fall back to CPU. "
            f"torch={torch.__version__}"
        )
    name = torch.cuda.get_device_name(0)
    print(f"CUDA available on {name}; torch={torch.__version__}")
    return name


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _split_counts() -> dict[str, int]:
    from src.data_prepare import yolo_root

    root = yolo_root()
    counts = {}
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        counts[split] = len([path for path in image_dir.iterdir() if path.is_file()]) if image_dir.is_dir() else 0
    if counts["train"] == 0 or counts["val"] == 0:
        raise SystemExit("YOLO split folders are missing. Run data preparation first.")
    return counts


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _train_run(weights: str | Path, run_name: str, epochs: int, cos_lr: bool, batch: int) -> Path:
    from ultralytics import YOLO

    config = train_config()
    project = Path(config["project_dir"])
    project.mkdir(parents=True, exist_ok=True)
    fallback = int(config["batch_fallback"])
    kwargs = dict(
        data=str(CONFIGS / "dataset.yaml"),
        epochs=epochs,
        imgsz=int(config["imgsz"]),
        batch=batch,
        seed=int(config["seed"]),
        device=int(config["device"]),
        workers=int(config["workers"]),
        optimizer=config["optimizer"],
        lr0=float(config["lr0"]),
        lrf=float(config["lrf"]),
        cos_lr=cos_lr,
        pretrained=bool(config["pretrained"]),
        project=str(project),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
        cache=False,
        amp=True,
        resume=False,
    )
    try:
        YOLO(str(weights)).train(**kwargs)
    except RuntimeError as exc:
        if batch != fallback and _is_oom(exc):
            print(f"CUDA OOM at batch={batch}. Retrying {run_name} at batch={fallback}.")
            torch.cuda.empty_cache()
            kwargs["batch"] = fallback
            YOLO(str(weights)).train(**kwargs)
            batch = fallback
        else:
            raise
    best = project / run_name / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished but {best} was not written.")
    run_dir = project / run_name
    for filename in ("results.csv", "args.yaml"):
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, METRICS / f"{run_name}_{filename}")
    _write_run_meta(run_name, epochs, cos_lr, batch, best)
    return best


def _write_run_meta(run_name: str, epochs: int, cos_lr: bool, batch: int, weights: Path) -> None:
    import ultralytics

    config = train_config()
    counts = _split_counts()
    args_path = METRICS / f"{run_name}_args.yaml"
    recorded_args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.is_file() else {}
    payload = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "epochs": epochs,
        "imgsz": int(config["imgsz"]),
        "batch": batch,
        "seed": int(config["seed"]),
        "classes": class_names(),
        "train_images": counts["train"],
        "val_images": counts["val"],
        "test_images": counts["test"],
        "optimizer": recorded_args.get("optimizer", config["optimizer"]),
        "learning_rate_settings": {
            "lr0": recorded_args.get("lr0", float(config["lr0"])),
            "lrf": recorded_args.get("lrf", float(config["lrf"])),
            "cos_lr": cos_lr,
        },
        "device": str(config["device"]),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "git_commit_if_available": _git_commit(),
        "weights": str(weights),
        "test_split_used_for_selection": False,
    }
    write_json(METRICS / f"run_meta_{run_name}.json", payload)


def _base_weights() -> Path:
    config = train_config()
    destination = Path(config["weights_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / config["model"]
    if target.is_file():
        return target
    from ultralytics.utils.downloads import attempt_download_asset

    downloaded = Path(attempt_download_asset(config["model"]))
    if downloaded.resolve() != target.resolve():
        shutil.copy2(downloaded, target)
    return target


def _metric_block(metrics) -> dict:
    results = dict(metrics.results_dict)
    names = class_names()
    box = metrics.box
    indexes = [int(index) for index in list(getattr(box, "ap_class_index", range(len(names))))]
    precision = [float(value) for value in list(box.p)]
    recall = [float(value) for value in list(box.r)]
    ap50 = [float(value) for value in list(box.ap50)]
    ap = [float(value) for value in list(box.ap)]
    per_class = []
    for slot, class_index in enumerate(indexes):
        per_class.append(
            {
                "class_name": names[class_index],
                "precision": precision[slot],
                "recall": recall[slot],
                "ap50": ap50[slot] if slot < len(ap50) else None,
                "ap50_95": ap[slot] if slot < len(ap) else None,
            }
        )
    found = {row["class_name"] for row in per_class}
    for name in names:
        if name not in found:
            per_class.append(
                {"class_name": name, "precision": 0.0, "recall": 0.0, "ap50": 0.0, "ap50_95": 0.0}
            )
    per_class.sort(key=lambda row: names.index(row["class_name"]))
    return {
        "precision": float(results.get("metrics/precision(B)", box.mp)),
        "recall": float(results.get("metrics/recall(B)", box.mr)),
        "map50": float(results.get("metrics/mAP50(B)", box.map50)),
        "map50_95": float(results.get("metrics/mAP50-95(B)", box.map)),
        "per_class": per_class,
        "results_dict": {key: float(value) for key, value in results.items()},
    }


def validate_checkpoint(weights: Path, split: str) -> dict:
    from ultralytics import YOLO

    config = train_config()
    torch.manual_seed(int(config["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    batch = int(config["batch"])
    try:
        metrics = YOLO(str(weights)).val(
            data=str(CONFIGS / "dataset.yaml"),
            split=split,
            imgsz=int(config["imgsz"]),
            batch=batch,
            device=int(config["device"]),
            workers=0,
            half=False,
            plots=False,
            verbose=False,
            seed=int(config["seed"]),
        )
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        print(f"CUDA OOM during {split} validation at batch={batch}. Retrying at batch={config['batch_fallback']}.")
        torch.cuda.empty_cache()
        metrics = YOLO(str(weights)).val(
            data=str(CONFIGS / "dataset.yaml"),
            split=split,
            imgsz=int(config["imgsz"]),
            batch=int(config["batch_fallback"]),
            device=int(config["device"]),
            workers=0,
            half=False,
            plots=False,
            verbose=False,
            seed=int(config["seed"]),
        )
    payload = _metric_block(metrics)
    payload["split"] = split
    payload["weights"] = str(weights)
    return payload


def run_baseline() -> Path:
    _cuda_or_stop()
    config = train_config()
    return _train_run(
        _base_weights(),
        "baseline",
        int(config["baseline_epochs"]),
        bool(config["baseline_cos_lr"]),
        int(config["batch"]),
    )


def run_tune() -> Path:
    _cuda_or_stop()
    meta = METRICS / "run_meta_baseline.json"
    if not meta.is_file():
        raise SystemExit("Baseline metadata is missing. Run the baseline first.")
    import json

    baseline_weights = Path(json.loads(meta.read_text(encoding="utf-8"))["weights"])
    config = train_config()
    return _train_run(
        baseline_weights,
        "tuned",
        int(config["tune_epochs"]),
        bool(config["tune_cos_lr"]),
        int(config["batch"]),
    )


def select_model() -> dict:
    _cuda_or_stop()
    import json

    baseline_weights = Path(json.loads((METRICS / "run_meta_baseline.json").read_text(encoding="utf-8"))["weights"])
    tuned_weights = Path(json.loads((METRICS / "run_meta_tuned.json").read_text(encoding="utf-8"))["weights"])
    baseline = validate_checkpoint(baseline_weights, "val")
    tuned = validate_checkpoint(tuned_weights, "val")
    write_json(METRICS / "baseline_val.json", baseline)
    write_json(METRICS / "tuned_val.json", tuned)
    reason = "tuned has the higher validation mAP50-95"
    chosen_name = "tuned"
    chosen_weights = tuned_weights
    if tuned["map50_95"] < baseline["map50_95"]:
        chosen_name = "baseline"
        chosen_weights = baseline_weights
        reason = "baseline has the higher validation mAP50-95; the tuned run is kept and reported"
    elif tuned["map50_95"] == baseline["map50_95"] and tuned["map50"] < baseline["map50"]:
        chosen_name = "baseline"
        chosen_weights = baseline_weights
        reason = "validation mAP50-95 tied; baseline has the higher mAP50"
    selection = {
        "chosen_run": chosen_name,
        "chosen_weights": str(chosen_weights),
        "reason": reason,
        "selection_split": "val",
        "test_split_used": False,
        "baseline_val": baseline,
        "tuned_val": tuned,
    }
    write_json(METRICS / "model_selection.json", selection)
    print(json.dumps({"chosen_run": chosen_name, "reason": reason, "baseline_map50_95": baseline["map50_95"], "tuned_map50_95": tuned["map50_95"]}, indent=2))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and select YOLO11n using validation only.")
    parser.add_argument("stage", choices=["baseline", "tune", "select", "all"])
    args = parser.parse_args()
    if args.stage in {"baseline", "all"}:
        run_baseline()
    if args.stage in {"tune", "all"}:
        run_tune()
    if args.stage in {"select", "all"}:
        select_model()


if __name__ == "__main__":
    main()
