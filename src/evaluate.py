"""Held-out test evaluation, matcher counts, and the error gallery."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from src.boxes import box_area, clip_box, iou, match_detections, yolo_to_pixels
from src.export_detections import write_detection_tables
from src.settings import FIGURES, METRICS, TABLES, class_names, classes_config, read_json, train_config, write_json
from src.train import _cuda_or_stop, validate_checkpoint

COLORS = {
    "gt": (46, 204, 113),
    "tp": (52, 152, 219),
    "fp": (231, 76, 60),
    "fn": (243, 156, 18),
}


def _read_labels(path: Path, width: int, height: int) -> list[dict]:
    names = class_names()
    rows = []
    if not path.is_file():
        return rows
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        class_id, x_center, y_center, box_w, box_h = line.split()[:5]
        pixels = yolo_to_pixels(float(x_center), float(y_center), float(box_w), float(box_h), width, height)
        if pixels is None:
            continue
        rows.append(
            {
                "class_name": names[int(class_id)],
                "box": pixels,
                "gt_id": f"{path.stem}:{line_index}",
            }
        )
    return rows


def _predict_split(model, split: str, model_version: str) -> tuple[list[dict], list[dict], list[dict]]:
    from src.data_prepare import yolo_root

    config = train_config()
    iou_threshold = float(classes_config()["iou_match_threshold"])
    image_dir = yolo_root() / "images" / split
    label_dir = yolo_root() / "labels" / split
    rows: list[dict] = []
    false_negatives: list[dict] = []
    reports: list[dict] = []
    results = model.predict(
        source=str(image_dir),
        conf=float(config["confidence_threshold"]),
        imgsz=int(config["imgsz"]),
        device=int(config["device"]),
        batch=int(config["batch_fallback"]),
        stream=True,
        verbose=False,
        save=False,
    )
    for result in results:
        image_path = Path(result.path)
        image_id = image_path.stem
        height, width = result.orig_shape
        ground_truth = _read_labels(label_dir / f"{image_id}.txt", int(width), int(height))
        predictions = []
        if result.boxes is not None and len(result.boxes):
            for box, score, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
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
        matches, unmatched_predictions, unmatched_truths = match_detections(
            predictions,
            ground_truth,
            float(iou_threshold),
        )
        matched_predictions = {pred_index: (gt_index, score) for pred_index, gt_index, score in matches}
        items = []
        match_ious = []
        for pred_index, prediction in enumerate(predictions):
            left, top, right, bottom = prediction["box"]
            matched = matched_predictions.get(pred_index)
            is_tp = matched is not None
            iou_score = matched[1] if matched else None
            if iou_score is not None:
                match_ious.append(iou_score)
            rows.append(
                {
                    "image_id": image_id,
                    "class_name": prediction["class_name"],
                    "confidence": prediction["confidence"],
                    "x_min": left,
                    "y_min": top,
                    "x_max": right,
                    "y_max": bottom,
                    "bbox_area": box_area(left, top, right, bottom),
                    "model_version": model_version,
                    "split": split,
                    "is_tp": is_tp,
                    "is_fp": not is_tp,
                    "matched_gt_id": ground_truth[matched[0]]["gt_id"] if matched else None,
                    "iou": iou_score,
                    "image_width": int(width),
                    "image_height": int(height),
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
        for gt_index, truth in enumerate(ground_truth):
            status = "fn" if gt_index in unmatched_truths else "tp"
            left, top, right, bottom = truth["box"]
            items.append(
                {
                    "kind": "gt",
                    "class_name": truth["class_name"],
                    "box": [left, top, right, bottom],
                    "status": status,
                }
            )
            if status == "fn":
                false_negatives.append(
                    {
                        "image_id": image_id,
                        "class_name": truth["class_name"],
                        "x_min": left,
                        "y_min": top,
                        "x_max": right,
                        "y_max": bottom,
                        "split": split,
                        "gt_id": truth["gt_id"],
                    }
                )
        confused = False
        for pred_index in unmatched_predictions:
            prediction = predictions[pred_index]
            for truth in ground_truth:
                if truth["class_name"] == prediction["class_name"]:
                    continue
                if iou(prediction["box"], truth["box"]) >= 0.50:
                    confused = True
        confidences = [prediction["confidence"] for prediction in predictions]
        reports.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "split": split,
                "n_fp": len(unmatched_predictions),
                "n_fn": len(unmatched_truths),
                "n_tp": len(matches),
                "min_match_iou": min(match_ious) if match_ious else None,
                "min_conf": min(confidences) if confidences else None,
                "confused": confused,
                "items": items,
            }
        )
    return rows, false_negatives, reports


def _select_errors(reports: list[dict]) -> list[tuple[dict, str]]:
    chosen: list[tuple[dict, str]] = []
    used: set[str] = set()

    def take(candidates: list[dict], limit: int, reason: str) -> None:
        for report in candidates:
            if report["image_id"] in used:
                continue
            chosen.append((report, reason))
            used.add(report["image_id"])
            if sum(1 for _report, selected_reason in chosen if selected_reason == reason) >= limit:
                return

    take([report for report in reports if report["n_fp"] > 0], 3, "false_positive")
    take([report for report in reports if report["n_fn"] > 0], 3, "false_negative")
    take(
        sorted(
            [report for report in reports if report["min_match_iou"] is not None and report["min_match_iou"] < 0.75],
            key=lambda report: report["min_match_iou"],
        ),
        2,
        "low_iou",
    )
    take(
        sorted(
            [report for report in reports if report["min_conf"] is not None and report["min_conf"] < 0.40],
            key=lambda report: report["min_conf"],
        ),
        2,
        "low_confidence",
    )
    take([report for report in reports if report["confused"]], 2, "confused_class")
    leftovers = sorted(reports, key=lambda report: (report["n_fp"] + report["n_fn"], report["min_match_iou"] or 1.0), reverse=True)
    for report in leftovers:
        if len(chosen) >= 10:
            break
        if report["image_id"] in used:
            continue
        reason = "additional_error" if report["n_fp"] or report["n_fn"] else "lowest_overlap_true_positive"
        chosen.append((report, reason))
        used.add(report["image_id"])
    if len(chosen) < 10:
        raise RuntimeError(f"Only found {len(chosen)} images for the error gallery; expected at least 10.")
    return chosen


def _draw_error(report: dict, reason: str, destination: Path) -> None:
    image = Image.open(report["image_path"]).convert("RGB")
    drawing = ImageDraw.Draw(image)
    for item in report["items"]:
        if item["kind"] != "gt":
            continue
        color = COLORS["fn"] if item["status"] == "fn" else COLORS["gt"]
        left, top, right, bottom = item["box"]
        drawing.rectangle((left, top, right, bottom), outline=color, width=2)
        drawing.text((left + 2, top + 2), f"GT {item['class_name']} {item['status'].upper()}", fill=color)
    for item in report["items"]:
        if item["kind"] != "pred":
            continue
        color = COLORS[item["status"]]
        left, top, right, bottom = item["box"]
        drawing.rectangle((left, top, right, bottom), outline=color, width=3)
        iou_text = f" iou={item['iou']:.2f}" if item["iou"] is not None else ""
        drawing.text(
            (left + 2, max(0, bottom - 14)),
            f"{item['status'].upper()} {item['class_name']} {item['confidence']:.2f}{iou_text}",
            fill=color,
        )
    drawing.rectangle((0, 0, image.width, 18), fill=(20, 20, 20))
    drawing.text((4, 2), f"{reason}  {report['image_id']}", fill=(255, 255, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=90)


def _save_charts(rows: list[dict]) -> None:
    destination = FIGURES / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    confidences = [row["confidence"] for row in rows]
    ious = [row["iou"] for row in rows if row["is_tp"] and row["iou"] is not None]
    if confidences:
        plt.figure(figsize=(7, 4))
        plt.hist(confidences, bins=20, color="#1C69D4")
        plt.xlabel("Confidence")
        plt.ylabel("Detections")
        plt.title("Test-set confidence")
        plt.tight_layout()
        plt.savefig(destination / "confidence_hist.png", dpi=120)
        plt.close()
    if ious:
        plt.figure(figsize=(7, 4))
        plt.hist(ious, bins=20, color="#1A1A1A")
        plt.xlabel("IoU of true positives")
        plt.ylabel("Matches")
        plt.title("Test-set IoU")
        plt.tight_layout()
        plt.savefig(destination / "iou_hist.png", dpi=120)
        plt.close()


def _core_metrics(payload: dict) -> dict:
    return {
        "precision": payload["precision"],
        "recall": payload["recall"],
        "map50": payload["map50"],
        "map50_95": payload["map50_95"],
        "per_class": payload["per_class"],
    }


def _matcher_summary(rows: list[dict], false_negatives: list[dict]) -> dict:
    per_class = {}
    for name in class_names():
        tp = sum(1 for row in rows if row["class_name"] == name and row["is_tp"])
        fp = sum(1 for row in rows if row["class_name"] == name and row["is_fp"])
        fn = sum(1 for row in false_negatives if row["class_name"] == name)
        per_class[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
        }
    tp = sum(item["tp"] for item in per_class.values())
    fp = sum(item["fp"] for item in per_class.values())
    fn = sum(item["fn"] for item in per_class.values())
    return {
        "split": "test",
        "confidence_threshold": float(train_config()["confidence_threshold"]),
        "iou_threshold": 0.50,
        "matching": "one-to-one, same class, IoU >= 0.50",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "per_class": per_class,
    }


def run_evaluation() -> None:
    _cuda_or_stop()
    selection_path = METRICS / "model_selection.json"
    if not selection_path.is_file():
        raise SystemExit("Model selection is missing. Validation selection must finish before the test split is used.")
    selection = read_json(selection_path)
    if selection.get("test_split_used"):
        raise SystemExit("Selection file says the test split was already used for selection.")
    weights = Path(selection["chosen_weights"])
    model_version = f"yolo11n-{selection['chosen_run']}"
    first = validate_checkpoint(weights, "test")
    second = validate_checkpoint(weights, "test")
    same = json.dumps(_core_metrics(first), sort_keys=True) == json.dumps(_core_metrics(second), sort_keys=True)
    write_json(
        METRICS / "test_metrics.json",
        {
            "identical_repeat": same,
            "model_version": model_version,
            "weights": str(weights),
            "run_1": first,
            "run_2": second,
        },
    )
    if not same:
        raise SystemExit("Repeated test evaluation did not match. Metrics were saved; training was not rerun.")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    all_rows: list[dict] = []
    all_fn: list[dict] = []
    test_reports: list[dict] = []
    for split in ("train", "val", "test"):
        rows, false_negatives, reports = _predict_split(model, split, model_version)
        all_rows.extend(rows)
        all_fn.extend(false_negatives)
        if split == "test":
            test_reports = reports
    test_rows = [row for row in all_rows if row["split"] == "test"]
    test_fn = [row for row in all_fn if row["split"] == "test"]
    write_json(METRICS / "prediction_count.json", {"n_predictions": len(all_rows), "n_test_predictions": len(test_rows)})
    write_json(METRICS / "matcher_summary.json", _matcher_summary(test_rows, test_fn))
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "prediction_rows.json").write_text(json.dumps(all_rows), encoding="utf-8")
    (TABLES / "false_negative_rows.json").write_text(json.dumps(all_fn), encoding="utf-8")
    write_detection_tables(all_rows, all_fn, len(all_rows))
    error_dir = FIGURES / "errors"
    if error_dir.exists():
        for old in error_dir.glob("*.jpg"):
            old.unlink()
    selected = _select_errors(test_reports)
    catalog = []
    for index, (report, reason) in enumerate(selected, start=1):
        filename = f"error_{index:02d}_{reason}_{report['image_id']}.jpg"
        _draw_error(report, reason, error_dir / filename)
        catalog.append(
            {
                "file": filename,
                "image_id": report["image_id"],
                "reason": reason,
                "n_fp": report["n_fp"],
                "n_fn": report["n_fn"],
                "n_tp": report["n_tp"],
                "min_match_iou": report["min_match_iou"],
                "min_conf": report["min_conf"],
                "confused": report["confused"],
            }
        )
    write_json(METRICS / "error_analysis.json", {"count": len(catalog), "examples": catalog})
    _save_charts(test_rows)
    print(json.dumps({"identical_repeat": True, "test_predictions": len(test_rows), "error_images": len(catalog), "map50_95": first["map50_95"]}, indent=2))


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
