"""Measure V1 test-set failure patterns. Does not retrain or touch the V2 test split."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics import _as_bool
from src.settings import OUTPUTS, TABLES, read_json, write_json

NAMES = ["pallet", "stillage", "forklift", "dolly"]
ANALYSIS = OUTPUTS / "v2" / "analysis"
LABEL_ROOT = Path(r"C:\Users\tanam\sordi-cache\yolo\labels\test")
LOW_CONFIDENCE = 0.40


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet" and path.is_file():
        frame = pd.read_parquet(path)
    else:
        csv_path = path.with_suffix(".csv") if path.suffix == ".parquet" else path
        frame = pd.read_csv(csv_path)
    for column in ("is_tp", "is_fp"):
        if column in frame.columns:
            frame[column] = _as_bool(frame[column])
    if "split" in frame.columns:
        frame = frame[frame["split"] == "test"].copy()
    return frame


def _ground_truth() -> pd.DataFrame:
    if not LABEL_ROOT.is_dir():
        raise SystemExit(f"V1 test labels are missing at {LABEL_ROOT}.")
    rows = []
    for path in sorted(LABEL_ROOT.glob("*.txt")):
        for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            class_id, _x, _y, width, height = line.split()[:5]
            name = NAMES[int(class_id)]
            rows.append(
                {
                    "image_id": path.stem,
                    "gt_id": f"{path.stem}:{line_index}",
                    "class_name": name,
                    "relative_area": float(width) * float(height),
                }
            )
    if not rows:
        raise SystemExit("V1 test labels did not contain any boxes.")
    return pd.DataFrame(rows)


def _class_table(detections: pd.DataFrame, false_negatives: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in NAMES:
        predicted = detections[detections["class_name"] == name]
        missed = false_negatives[false_negatives["class_name"] == name]
        matched = predicted[predicted["is_tp"] & predicted["iou"].notna()]
        tp = int(predicted["is_tp"].sum()) if len(predicted) else 0
        fp = int(predicted["is_fp"].sum()) if len(predicted) else 0
        fn = int(len(missed))
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        rows.append(
            {
                "class_name": name,
                "ground_truth_count": int((truth["class_name"] == name).sum()),
                "prediction_count": int(len(predicted)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "average_confidence": float(predicted["confidence"].mean()) if len(predicted) else None,
                "median_confidence": float(predicted["confidence"].median()) if len(predicted) else None,
                "average_iou_matched": float(matched["iou"].mean()) if len(matched) else None,
            }
        )
    return pd.DataFrame(rows)


def _group_recall(truth: pd.DataFrame, tp_ids: set[str], mask: pd.Series) -> dict:
    subset = truth[mask]
    gt_count = int(len(subset))
    tp = int(subset["gt_id"].isin(tp_ids).sum())
    return {
        "ground_truth": gt_count,
        "tp": tp,
        "fn": gt_count - tp,
        "recall": (tp / gt_count) if gt_count else None,
    }


def _findings(detections: pd.DataFrame, false_negatives: pd.DataFrame, truth: pd.DataFrame, table: pd.DataFrame) -> dict:
    summary = read_json(OUTPUTS / "metrics" / "dataset_summary.json")
    train_objects = summary["by_split"]["train"]["objects"]
    train_total = sum(train_objects.values())
    tp_ids = set(detections.loc[detections["is_tp"], "matched_gt_id"].dropna().astype(str))
    area_cut = float(truth["relative_area"].quantile(0.25))
    median_area = float(truth["relative_area"].median())
    counts = truth.groupby("image_id").size()
    crowd_cut = float(counts.median())
    crowded_ids = set(counts[counts >= crowd_cut].index)
    small = _group_recall(truth, tp_ids, truth["relative_area"] <= area_cut)
    large = _group_recall(truth, tp_ids, truth["relative_area"] > area_cut)
    below_median = _group_recall(truth, tp_ids, truth["relative_area"] <= median_area)
    above_median = _group_recall(truth, tp_ids, truth["relative_area"] > median_area)
    crowded = _group_recall(truth, tp_ids, truth["image_id"].isin(crowded_ids))
    uncrowded = _group_recall(truth, tp_ids, ~truth["image_id"].isin(crowded_ids))
    low = detections[detections["confidence"] < LOW_CONFIDENCE]
    low_by_class = {}
    for name in NAMES:
        predicted = detections[detections["class_name"] == name]
        low_class = low[low["class_name"] == name]
        low_by_class[name] = {
            "low_confidence_predictions": int(len(low_class)),
            "low_confidence_rate": float(len(low_class) / len(predicted)) if len(predicted) else None,
            "low_confidence_fp": int(low_class["is_fp"].sum()) if len(low_class) else 0,
            "low_confidence_fp_share": float(low_class["is_fp"].mean()) if len(low_class) else None,
        }
    most_fp = table.sort_values(["fp", "class_name"], ascending=[False, True]).iloc[0]
    most_fn = table.sort_values(["fn", "class_name"], ascending=[False, True]).iloc[0]
    fn_ids = set(false_negatives["gt_id"].astype(str)) if "gt_id" in false_negatives.columns else set()
    covered = tp_ids | fn_ids
    return {
        "test_images": int(truth["image_id"].nunique()),
        "test_ground_truth": int(len(truth)),
        "matcher_tp": int(detections["is_tp"].sum()),
        "matcher_fp": int(detections["is_fp"].sum()),
        "matcher_fn": int(len(false_negatives)),
        "ground_truth_ids_not_in_tp_or_fn": int((~truth["gt_id"].isin(covered)).sum()),
        "most_fp_class": str(most_fp["class_name"]),
        "most_fp_count": int(most_fp["fp"]),
        "most_fn_class": str(most_fn["class_name"]),
        "most_fn_count": int(most_fn["fn"]),
        "train_objects": {name: int(train_objects[name]) for name in NAMES},
        "train_object_share": {name: train_objects[name] / train_total for name in NAMES},
        "train_images_with_class": summary["by_split"]["train"]["images_with_class"],
        "small_box_relative_area_p25": area_cut,
        "small_boxes": small,
        "larger_boxes": large,
        "below_median_area": below_median,
        "above_median_area": above_median,
        "median_relative_area": median_area,
        "crowded_image_min_objects": crowd_cut,
        "crowded_images": int((counts >= crowd_cut).sum()),
        "uncrowded_images": int((counts < crowd_cut).sum()),
        "crowded": crowded,
        "uncrowded": uncrowded,
        "low_confidence_threshold": LOW_CONFIDENCE,
        "low_confidence_predictions": int(len(low)),
        "low_confidence_rate": float(len(low) / len(detections)) if len(detections) else None,
        "low_confidence_by_class": low_by_class,
    }


def _augmentation_plan(findings: dict) -> dict:
    small_recall = findings["small_boxes"]["recall"]
    large_recall = findings["larger_boxes"]["recall"]
    crowded_recall = findings["crowded"]["recall"]
    uncrowded_recall = findings["uncrowded"]["recall"]
    overall_low = findings["low_confidence_rate"] or 0.0
    small_gap = large_recall is not None and small_recall is not None and large_recall - small_recall >= 0.05
    crowd_gap = uncrowded_recall is not None and crowded_recall is not None and uncrowded_recall - crowded_recall >= 0.05
    high_low_conf = [
        name
        for name, row in findings["low_confidence_by_class"].items()
        if row["low_confidence_rate"] is not None and row["low_confidence_rate"] >= overall_low * 1.5 and row["low_confidence_predictions"] >= 30
    ]
    plan = {
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "translate": 0.1,
        "scale": 0.45 if small_gap else 0.3,
        "mosaic": 0.3 if crowd_gap else (0.8 if small_gap else 0.5),
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.4 if high_low_conf else 0.3,
        "triggers": {
            "small_boxes_recall_gap_at_least_0.05": bool(small_gap),
            "crowded_images_recall_gap_at_least_0.05": bool(crowd_gap),
            "classes_with_elevated_low_confidence": high_low_conf,
        },
        "rejected": [
            "vertical flip, because an upside-down warehouse floor is not a useful industrial view",
            "rotation and shear, because they bend upright loads and aisles",
            "mixup and copy-paste, because they composite objects into scenes they were not rendered in",
            "random erasing, because it deletes parts of loads without a physical cause",
        ],
    }
    if small_gap and crowd_gap:
        plan["mosaic_note"] = (
            "Small boxes and crowded images both recall worse. Mosaic stays at 0.3 so training does not add more crowding. "
            "Scale is 0.45 so object size still varies."
        )
    return plan


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def _write_report(table: pd.DataFrame, findings: dict, plan: dict) -> None:
    lines = [
        "# V1 failure analysis",
        "",
        "This report uses the archived V1 test export and the V1 training counts. It does not change the V1 checkpoint and it is not a V2 test result.",
        "",
        "Matcher rule: one prediction to one ground-truth box, same class, IoU at least 0.50, confidence at least 0.25. These precision and recall figures are not mAP.",
        "",
        "## Per class",
        "",
        "| Class | GT | Predictions | TP | FP | FN | Precision | Recall | Mean confidence | Median confidence | Mean matched IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| {class_name} | {gt} | {pred} | {tp} | {fp} | {fn} | {precision} | {recall} | {mean_conf} | {median_conf} | {iou} |".format(
                class_name=row.class_name,
                gt=f"{row.ground_truth_count:,}",
                pred=f"{row.prediction_count:,}",
                tp=f"{row.tp:,}",
                fp=f"{row.fp:,}",
                fn=f"{row.fn:,}",
                precision=_fmt(row.precision),
                recall=_fmt(row.recall),
                mean_conf=_fmt(row.average_confidence),
                median_conf=_fmt(row.median_confidence),
                iou=_fmt(row.average_iou_matched),
            )
        )
    share_bits = ", ".join(f"{name} {_fmt(findings['train_object_share'][name], 3)}" for name in NAMES)
    train_bits = ", ".join(f"{name} {findings['train_objects'][name]:,}" for name in NAMES)
    scarce = min(NAMES, key=lambda name: findings["train_objects"][name])
    lines.extend(
        [
            "",
            "## What the counts show",
            "",
            f"The largest false-positive count is {findings['most_fp_class']} ({findings['most_fp_count']:,}). The largest false-negative count is {findings['most_fn_class']} ({findings['most_fn_count']:,}). Those are raw counts, so the most common class can lead both lists even when its rate is not the worst.",
            "",
            f"V1 training boxes: {train_bits}. Shares of training boxes: {share_bits}. The fewest training boxes belong to {scarce}.",
            "",
            f"Small boxes are the ground-truth boxes at or below the 25th percentile of relative area ({_fmt(findings['small_box_relative_area_p25'])}). Their recall is {_fmt(findings['small_boxes']['recall'])} ({findings['small_boxes']['tp']:,} / {findings['small_boxes']['ground_truth']:,}). Larger boxes recall {_fmt(findings['larger_boxes']['recall'])} ({findings['larger_boxes']['tp']:,} / {findings['larger_boxes']['ground_truth']:,}).",
            "",
            f"The same comparison at the median relative area ({_fmt(findings['median_relative_area'])}) is {_fmt(findings['below_median_area']['recall'])} below the median and {_fmt(findings['above_median_area']['recall'])} above it.",
            "",
            f"A crowded test image has at least { _fmt(findings['crowded_image_min_objects'], 1) } target boxes, the median count. Crowded images: recall {_fmt(findings['crowded']['recall'])} on {findings['crowded']['ground_truth']:,} boxes across {findings['crowded_images']:,} images. Other images: recall {_fmt(findings['uncrowded']['recall'])} on {findings['uncrowded']['ground_truth']:,} boxes across {findings['uncrowded_images']:,} images.",
            "",
            f"Predictions below confidence {LOW_CONFIDENCE:.2f}: {findings['low_confidence_predictions']:,} ({_fmt(findings['low_confidence_rate'], 3)} of test predictions).",
            "",
        ]
    )
    for name in NAMES:
        row = findings["low_confidence_by_class"][name]
        lines.append(
            f"- {name}: low-confidence rate {_fmt(row['low_confidence_rate'], 3)} ({row['low_confidence_predictions']:,} predictions), of which {_fmt(row['low_confidence_fp_share'], 3)} are false positives."
        )
    lines.extend(
        [
            "",
            f"Ground-truth ids absent from both the true-positive and false-negative tables: {findings['ground_truth_ids_not_in_tp_or_fn']}.",
            "",
            "## Augmentation chosen for experiment C",
            "",
            "Experiment C starts from the same pretrained YOLO11n weights as A and B. Only the augmentation settings below change relative to B. They were fixed from this V1 measurement before V2 training, not from a V2 test score.",
            "",
            f"- scale = {plan['scale']}",
            f"- mosaic = {plan['mosaic']}",
            f"- translate = {plan['translate']}",
            f"- horizontal flip = {plan['fliplr']}",
            f"- HSV value gain = {plan['hsv_v']}",
            f"- rotation, shear, perspective, vertical flip, mixup, copy-paste, and random erasing stay off",
            "",
            "Triggers:",
            "",
        ]
    )
    for key, value in plan["triggers"].items():
        lines.append(f"- {key}: {value}")
    if plan.get("mosaic_note"):
        lines.extend(["", plan["mosaic_note"]])
    lines.extend(["", "Settings that were rejected:"])
    for item in plan["rejected"]:
        lines.append(f"- {item}")
    lines.append("")
    path = Path("docs/v1_failure_analysis.md")
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_v1() -> None:
    detections_path = TABLES / "detections.parquet"
    if not detections_path.is_file():
        detections_path = TABLES / "detections.csv"
    fn_path = TABLES / "false_negatives.parquet"
    if not fn_path.is_file():
        fn_path = TABLES / "false_negatives.csv"
    detections = _load_frame(detections_path)
    false_negatives = _load_frame(fn_path)
    truth = _ground_truth()
    table = _class_table(detections, false_negatives, truth)
    findings = _findings(detections, false_negatives, truth, table)
    plan = _augmentation_plan(findings)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    table.to_csv(ANALYSIS / "v1_failure_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_json(ANALYSIS / "v1_failure_findings.json", findings)
    write_json(ANALYSIS / "augmentation_plan.json", plan)
    _write_report(table, findings, plan)
    print(table.to_string(index=False))
    print(f"Wrote {ANALYSIS / 'v1_failure_summary.csv'}")


if __name__ == "__main__":
    analyze_v1()
