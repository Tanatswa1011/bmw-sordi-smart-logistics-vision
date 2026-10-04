"""Validation comparison, threshold choice, and the single V2 held-out test."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import torch

from src.boxes import box_area, clip_box, match_detections, yolo_to_pixels
from src.evaluate import _draw_error, _select_errors
from src.export_detections import REQUIRED_COLUMNS, validate_detections
from src.prepare_v2 import v2_config, yolo_v2_root
from src.settings import CONFIGS, OUTPUTS, class_names, materialize_dataset_yaml, read_json, write_json
from src.train import _cuda_or_stop, _metric_block

V2_METRICS = OUTPUTS / "v2" / "metrics"
V2_TABLES = OUTPUTS / "v2" / "tables"
V2_FIGURES = OUTPUTS / "v2" / "figures"
NAMES = ["pallet", "stillage", "forklift", "dolly"]
LETTERS = ("A", "B", "C")


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _weights(letter: str) -> Path:
    config = v2_config()
    run_name = config["experiments"][letter]["run_name"]
    path = Path(config["project_dir"]) / run_name / "weights" / "best.pt"
    if not path.is_file():
        raise SystemExit(f"Experiment {letter} weights are missing at {path}.")
    return path


def _epochs_completed(letter: str) -> int:
    config = v2_config()
    path = V2_METRICS / f"{config['experiments'][letter]['run_name']}_results.csv"
    if not path.is_file():
        return 0
    frame = pd.read_csv(path)
    if frame.empty or "epoch" not in frame.columns:
        return 0
    return int(frame["epoch"].max())


def _train_count() -> int:
    summary = read_json(V2_METRICS / "dataset_summary.json")
    return int(summary["by_split"]["train"]["images"])


def validate_v2(weights: Path, split: str) -> dict:
    from ultralytics import YOLO

    if split not in {"val", "test"}:
        raise SystemExit(f"Unexpected split {split}.")
    config = v2_config()
    torch.manual_seed(int(config["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    batch = int(config["batch"])
    data_yaml = str(materialize_dataset_yaml(CONFIGS / "dataset_v2.yaml"))
    try:
        metrics = YOLO(str(weights)).val(
            data=data_yaml,
            split=split,
            imgsz=int(config["imgsz"]),
            batch=batch,
            device=0,
            workers=0,
            half=False,
            plots=False,
            verbose=False,
            seed=int(config["seed"]),
        )
    except RuntimeError as exc:
        text = str(exc).lower()
        if "out of memory" not in text and "cuda oom" not in text:
            raise
        torch.cuda.empty_cache()
        metrics = YOLO(str(weights)).val(
            data=data_yaml,
            split=split,
            imgsz=int(config["imgsz"]),
            batch=int(config["batch_fallback"]),
            device=0,
            workers=0,
            half=False,
            plots=False,
            verbose=False,
            seed=int(config["seed"]),
        )
    payload = _metric_block(metrics)
    payload["split"] = split
    payload["weights"] = str(weights)
    payload["f1"] = _f1(payload["precision"], payload["recall"])
    return payload


def _read_ground_truth(split: str) -> dict[str, list[dict]]:
    label_dir = yolo_v2_root() / "labels" / split
    image_dir = yolo_v2_root() / "images" / split
    grouped: dict[str, list[dict]] = {}
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        grouped[image_path.stem] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        rows = grouped.setdefault(label_path.stem, [])
        # Width and height are filled when the image is predicted. Relative labels are converted then.
        for line_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            class_id, x_center, y_center, box_w, box_h = line.split()[:5]
            rows.append(
                {
                    "class_name": NAMES[int(class_id)],
                    "gt_id": f"{label_path.stem}:{line_index}",
                    "yolo": (float(x_center), float(y_center), float(box_w), float(box_h)),
                }
            )
    return grouped


def _predict(weights: Path, split: str, confidence: float) -> list[dict]:
    from ultralytics import YOLO

    config = v2_config()
    torch.manual_seed(int(config["seed"]))
    truth = _read_ground_truth(split)
    model = YOLO(str(weights))
    image_dir = yolo_v2_root() / "images" / split
    predicted = model.predict(
        source=str(image_dir),
        conf=confidence,
        imgsz=int(config["imgsz"]),
        device=0,
        batch=int(config["batch_fallback"]),
        stream=True,
        verbose=False,
        save=False,
    )
    images = []
    for result in predicted:
        image_path = Path(result.path)
        image_id = image_path.stem
        height, width = result.orig_shape
        ground_truth = []
        for item in truth.get(image_id, []):
            pixels = yolo_to_pixels(*item["yolo"], int(width), int(height))
            if pixels is None:
                continue
            ground_truth.append({"class_name": item["class_name"], "box": pixels, "gt_id": item["gt_id"]})
        predictions = []
        if result.boxes is not None and len(result.boxes):
            for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                clipped = clip_box(float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(width), int(height))
                if clipped is None:
                    continue
                predictions.append(
                    {
                        "class_name": result.names[int(class_id)],
                        "confidence": float(score),
                        "box": clipped,
                    }
                )
        images.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "width": int(width),
                "height": int(height),
                "split": split,
                "predictions": predictions,
                "ground_truth": ground_truth,
            }
        )
    return images


def _match_at(images: list[dict], confidence: float) -> dict:
    rows = []
    false_negatives = []
    reports = []
    for image in images:
        predictions = [prediction for prediction in image["predictions"] if prediction["confidence"] >= confidence]
        matches, unmatched_predictions, unmatched_truths = match_detections(predictions, image["ground_truth"], 0.50)
        matched = {pred_index: (gt_index, score) for pred_index, gt_index, score in matches}
        items = []
        match_ious = []
        for pred_index, prediction in enumerate(predictions):
            left, top, right, bottom = prediction["box"]
            pair = matched.get(pred_index)
            is_tp = pair is not None
            iou_score = pair[1] if pair else None
            if iou_score is not None:
                match_ious.append(iou_score)
            rows.append(
                {
                    "image_id": image["image_id"],
                    "class_name": prediction["class_name"],
                    "confidence": prediction["confidence"],
                    "x_min": left,
                    "y_min": top,
                    "x_max": right,
                    "y_max": bottom,
                    "bbox_area": box_area(left, top, right, bottom),
                    "model_version": "v2",
                    "split": image["split"],
                    "is_tp": is_tp,
                    "is_fp": not is_tp,
                    "matched_gt_id": image["ground_truth"][pair[0]]["gt_id"] if pair else None,
                    "iou": iou_score,
                    "image_width": image["width"],
                    "image_height": image["height"],
                }
            )
            items.append(
                {
                    "kind": "pred",
                    "class_name": prediction["class_name"],
                    "box": [left, top, right, bottom],
                    "confidence": prediction["confidence"],
                    "status": "tp" if is_tp else "fp",
                    "iou": iou_score,
                }
            )
        for gt_index, truth in enumerate(image["ground_truth"]):
            status = "fn" if gt_index in unmatched_truths else "tp"
            left, top, right, bottom = truth["box"]
            items.append({"kind": "gt", "class_name": truth["class_name"], "box": [left, top, right, bottom], "status": status})
            if status == "fn":
                false_negatives.append(
                    {
                        "image_id": image["image_id"],
                        "class_name": truth["class_name"],
                        "x_min": left,
                        "y_min": top,
                        "x_max": right,
                        "y_max": bottom,
                        "split": image["split"],
                        "gt_id": truth["gt_id"],
                    }
                )
        confused = False
        for pred_index in unmatched_predictions:
            prediction = predictions[pred_index]
            for truth in image["ground_truth"]:
                if truth["class_name"] == prediction["class_name"]:
                    continue
                from src.boxes import iou

                if iou(prediction["box"], truth["box"]) >= 0.50:
                    confused = True
        confidences = [prediction["confidence"] for prediction in predictions]
        reports.append(
            {
                "image_id": image["image_id"],
                "image_path": image["image_path"],
                "split": image["split"],
                "n_fp": len(unmatched_predictions),
                "n_fn": len(unmatched_truths),
                "n_tp": len(matches),
                "min_match_iou": min(match_ious) if match_ious else None,
                "min_conf": min(confidences) if confidences else None,
                "confused": confused,
                "items": items,
            }
        )
    tp = sum(1 for row in rows if row["is_tp"])
    fp = sum(1 for row in rows if row["is_fp"])
    fn = len(false_negatives)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    confidences = [row["confidence"] for row in rows]
    return {
        "rows": rows,
        "false_negatives": false_negatives,
        "reports": reports,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "mean_confidence": float(sum(confidences) / len(confidences)) if confidences else None,
        "low_confidence_rate": float(sum(value < 0.40 for value in confidences) / len(confidences)) if confidences else None,
    }


def _signature(images: list[dict]) -> list[tuple]:
    rows = []
    for image in images:
        for prediction in image["predictions"]:
            left, top, right, bottom = prediction["box"]
            rows.append(
                (
                    image["image_id"],
                    prediction["class_name"],
                    round(prediction["confidence"], 6),
                    round(left, 2),
                    round(top, 2),
                    round(right, 2),
                    round(bottom, 2),
                )
            )
    return sorted(rows)


def compare_validation() -> list[dict]:
    _cuda_or_stop()
    train_images = _train_count()
    rows = []
    for letter in LETTERS:
        print(f"Validating experiment {letter} on the V2 validation split.")
        metrics = validate_v2(_weights(letter), "val")
        write_json(V2_METRICS / f"{letter.lower()}_val.json", metrics)
        per_class = {item["class_name"]: item for item in metrics["per_class"]}
        row = {
            "experiment": letter,
            "train_image_count": train_images,
            "epochs_completed": _epochs_completed(letter),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
        }
        for name in NAMES:
            row[f"{name}_precision"] = per_class[name]["precision"]
            row[f"{name}_recall"] = per_class[name]["recall"]
        rows.append(row)
    _write_csv(V2_METRICS / "experiment_comparison.csv", rows)
    return rows


def _choose_experiment(rows: list[dict]) -> dict:
    epsilon = float(v2_config()["map_tie_epsilon"])
    ranked = sorted(rows, key=lambda row: (row["map50_95"], row["f1"] or -1), reverse=True)
    best = ranked[0]
    near = [row for row in ranked if best["map50_95"] - row["map50_95"] <= epsilon]
    chosen = max(near, key=lambda row: (row["f1"] or -1, row["map50_95"]))
    reason = (
        f"{chosen['experiment']} has the highest V2 validation mAP50-95 "
        f"({chosen['map50_95']:.6f})."
    )
    if chosen["experiment"] != best["experiment"]:
        reason = (
            f"{chosen['experiment']} is within {epsilon} validation mAP50-95 of {best['experiment']} "
            f"and has the higher validation F1."
        )
    return {"chosen": chosen, "reason": reason, "ranked": ranked}


def sweep_threshold(letter: str) -> tuple[float, list[dict]]:
    _cuda_or_stop()
    thresholds = [float(value) for value in v2_config()["confidence_thresholds"]]
    images = _predict(_weights(letter), "val", min(thresholds))
    rows = []
    for threshold in thresholds:
        matched = _match_at(images, threshold)
        rows.append(
            {
                "experiment": letter,
                "threshold": threshold,
                "precision": matched["precision"],
                "recall": matched["recall"],
                "f1": matched["f1"],
                "fp": matched["fp"],
                "fn": matched["fn"],
            }
        )
    _write_csv(V2_METRICS / "conf_threshold_validation.csv", rows)
    eligible = [row for row in rows if row["f1"] is not None]
    chosen = max(eligible, key=lambda row: (row["f1"], -abs((row["precision"] or 0) - (row["recall"] or 0)), row["recall"] or 0))
    return float(chosen["threshold"]), rows


def select_model() -> dict:
    comparison_path = V2_METRICS / "experiment_comparison.csv"
    if not comparison_path.is_file():
        rows = compare_validation()
    else:
        rows = list(csv.DictReader(comparison_path.open(encoding="utf-8")))
        for row in rows:
            for key, value in list(row.items()):
                if key != "experiment":
                    row[key] = float(value) if value not in {"", "None"} else None
            row["epochs_completed"] = int(row["epochs_completed"])
            row["train_image_count"] = int(row["train_image_count"])
    decision = _choose_experiment(rows)
    letter = decision["chosen"]["experiment"]
    threshold, sweep = sweep_threshold(letter)
    selection = {
        "chosen_experiment": letter,
        "chosen_weights": str(_weights(letter)),
        "reason": decision["reason"],
        "selection_metric": "validation mAP50-95",
        "selection_split": "val",
        "test_split_used": False,
        "confidence_threshold": threshold,
        "confidence_threshold_rule": "highest validation matcher F1; ties prefer precision closer to recall, then higher recall",
        "threshold_frozen": True,
        "validation_comparison": rows,
        "threshold_sweep": sweep,
    }
    write_json(V2_METRICS / "model_selection.json", selection)
    print(json.dumps({"chosen_experiment": letter, "confidence_threshold": threshold, "reason": decision["reason"]}, indent=2))
    return selection


def _core(payload: dict) -> dict:
    return {
        "precision": payload["precision"],
        "recall": payload["recall"],
        "map50": payload["map50"],
        "map50_95": payload["map50_95"],
        "per_class": payload["per_class"],
    }


def _write_tables(rows: list[dict], false_negatives: list[dict]) -> None:
    V2_TABLES.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    extra = ["matched_gt_id", "iou", "image_width", "image_height"]
    if frame.empty:
        frame = pd.DataFrame(columns=REQUIRED_COLUMNS + extra)
    image_sizes = {row["image_id"]: (int(row["image_width"]), int(row["image_height"])) for row in rows}
    validate_detections(frame, image_sizes=image_sizes, expected_rows=len(rows))
    frame = frame[REQUIRED_COLUMNS + extra]
    frame.to_csv(V2_TABLES / "detections.csv", index=False)
    frame.to_parquet(V2_TABLES / "detections.parquet", index=False)
    negatives = pd.DataFrame(false_negatives)
    if negatives.empty:
        negatives = pd.DataFrame(columns=["image_id", "class_name", "x_min", "y_min", "x_max", "y_max", "split", "gt_id"])
    negatives.to_csv(V2_TABLES / "false_negatives.csv", index=False)
    negatives.to_parquet(V2_TABLES / "false_negatives.parquet", index=False)


def _per_class_counts(rows: list[dict], false_negatives: list[dict], summary: dict) -> list[dict]:
    counts = []
    test_objects = summary["by_split"]["test"]["objects"]
    for name in NAMES:
        subset = [row for row in rows if row["class_name"] == name]
        tp = sum(1 for row in subset if row["is_tp"])
        fp = sum(1 for row in subset if row["is_fp"])
        fn = sum(1 for row in false_negatives if row["class_name"] == name)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        counts.append(
            {
                "class_name": name,
                "object_count": int(test_objects[name]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "matcher_precision": precision,
                "matcher_recall": recall,
            }
        )
    return counts


def run_test() -> dict:
    _cuda_or_stop()
    selection_path = V2_METRICS / "model_selection.json"
    if not selection_path.is_file():
        raise SystemExit("V2 model selection is missing. The test split was not evaluated.")
    selection = read_json(selection_path)
    if selection.get("test_split_used"):
        raise SystemExit("Selection file says the V2 test split was already used for selection.")
    if not selection.get("threshold_frozen"):
        raise SystemExit("The confidence threshold is not frozen.")
    weights = Path(selection["chosen_weights"])
    threshold = float(selection["confidence_threshold"])
    first = validate_v2(weights, "test")
    second = validate_v2(weights, "test")
    official_same = json.dumps(_core(first), sort_keys=True) == json.dumps(_core(second), sort_keys=True)
    images_first = _predict(weights, "test", threshold)
    images_second = _predict(weights, "test", threshold)
    predictions_same = _signature(images_first) == _signature(images_second)
    matched = _match_at(images_first, threshold)
    matched_again = _match_at(images_second, threshold)
    counts_same = (matched["tp"], matched["fp"], matched["fn"]) == (matched_again["tp"], matched_again["fp"], matched_again["fn"])
    identical = official_same and predictions_same and counts_same
    summary = read_json(V2_METRICS / "dataset_summary.json")
    per_class = {item["class_name"]: item for item in first["per_class"]}
    class_rows = _per_class_counts(matched["rows"], matched["false_negatives"], summary)
    for row in class_rows:
        official = per_class[row["class_name"]]
        row["precision"] = official["precision"]
        row["recall"] = official["recall"]
        row["ap50"] = official["ap50"]
        row["ap50_95"] = official["ap50_95"]
    payload = {
        "identical_repeat": identical,
        "official_repeat_identical": official_same,
        "prediction_repeat_identical": predictions_same,
        "matcher_count_repeat_identical": counts_same,
        "model_version": f"yolo11n-v2-{selection['chosen_experiment']}",
        "weights": str(weights),
        "confidence_threshold": threshold,
        "iou_threshold": 0.50,
        "run_1": first,
        "run_2": second,
        "matcher": {
            "tp": matched["tp"],
            "fp": matched["fp"],
            "fn": matched["fn"],
            "precision": matched["precision"],
            "recall": matched["recall"],
            "f1": matched["f1"],
            "mean_confidence": matched["mean_confidence"],
            "low_confidence_rate": matched["low_confidence_rate"],
        },
        "per_class": class_rows,
    }
    write_json(V2_METRICS / "test_metrics.json", payload)
    if not identical:
        raise SystemExit("Repeated V2 test evaluation did not match. Metrics were saved; nothing was rerun or retuned.")
    for row in matched["rows"]:
        row["model_version"] = payload["model_version"]
    _write_tables(matched["rows"], matched["false_negatives"])
    error_dir = V2_FIGURES / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    for old in error_dir.glob("*.jpg"):
        old.unlink()
    selected = _select_errors(matched["reports"])
    catalog = []
    for index, (report, reason) in enumerate(selected, start=1):
        filename = f"error_{index:02d}_{reason}_{report['image_id']}.jpg"
        _draw_error(report, reason, error_dir / filename)
        catalog.append({"file": filename, "image_id": report["image_id"], "reason": reason, "n_fp": report["n_fp"], "n_fn": report["n_fn"]})
    write_json(V2_METRICS / "error_analysis.json", {"count": len(catalog), "examples": catalog})
    _write_comparison(payload)
    print(json.dumps({"identical_repeat": True, "map50_95": first["map50_95"], "matcher_f1": matched["f1"], "threshold": threshold}, indent=2))
    return payload


def _metric_row(name: str, v1: float | None, v2: float | None, note: str) -> dict:
    absolute = None if v1 is None or v2 is None else v2 - v1
    relative = None if v1 in (None, 0) or v2 is None else (v2 - v1) / v1
    return {"metric": name, "v1": v1, "v2": v2, "absolute_change": absolute, "relative_change": relative, "note": note}


def _write_comparison(test_payload: dict) -> None:
    v1_official = read_json(OUTPUTS / "v1" / "metrics" / "test_metrics.json")["run_1"]
    v1_match = read_json(OUTPUTS / "v1" / "metrics" / "matcher_summary.json")
    v1_detections = pd.read_csv(OUTPUTS / "v1" / "tables" / "detections.csv")
    v1_detections = v1_detections[v1_detections["split"] == "test"]
    v1_mean = float(v1_detections["confidence"].mean())
    v1_low = float((v1_detections["confidence"] < 0.40).mean())
    v1_f1 = _f1(v1_official["precision"], v1_official["recall"])
    v1_matcher_f1 = _f1(v1_match["precision"], v1_match["recall"])
    official = test_payload["run_1"]
    matcher = test_payload["matcher"]
    threshold = test_payload["confidence_threshold"]
    note_official = "Ultralytics validation. Independent of the operating confidence threshold."
    note_matcher = f"One-to-one matcher at IoU 0.50. V1 confidence threshold 0.25. V2 confidence threshold {threshold}."
    rows = [
        _metric_row("precision", v1_official["precision"], official["precision"], note_official),
        _metric_row("recall", v1_official["recall"], official["recall"], note_official),
        _metric_row("f1", v1_f1, official["f1"], note_official),
        _metric_row("map50", v1_official["map50"], official["map50"], note_official),
        _metric_row("map50_95", v1_official["map50_95"], official["map50_95"], note_official),
        _metric_row("matcher_precision", v1_match["precision"], matcher["precision"], note_matcher),
        _metric_row("matcher_recall", v1_match["recall"], matcher["recall"], note_matcher),
        _metric_row("matcher_f1", v1_matcher_f1, matcher["f1"], note_matcher),
        _metric_row("fp", v1_match["fp"], matcher["fp"], note_matcher),
        _metric_row("fn", v1_match["fn"], matcher["fn"], note_matcher),
        _metric_row("mean_confidence", v1_mean, matcher["mean_confidence"], note_matcher),
        _metric_row("low_confidence_rate", v1_low, matcher["low_confidence_rate"], note_matcher),
    ]
    _write_csv(V2_METRICS / "v1_vs_v2.csv", rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare V2 runs on validation, then test the frozen choice once.")
    parser.add_argument("stage", choices=["val", "select", "test", "all"])
    args = parser.parse_args()
    if args.stage in {"val", "all"}:
        compare_validation()
    if args.stage in {"select", "all"}:
        select_model()
    if args.stage in {"test", "all"}:
        run_test()


if __name__ == "__main__":
    main()
