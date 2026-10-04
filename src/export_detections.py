"""Write the detection table and check that every exported row is a real prediction."""

from __future__ import annotations

import argparse

import pandas as pd

from src.settings import METRICS, TABLES, class_names, read_json, write_json

REQUIRED_COLUMNS = [
    "image_id",
    "class_name",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "bbox_area",
    "model_version",
    "split",
    "is_tp",
    "is_fp",
]


def validate_detections(
    frame: pd.DataFrame,
    image_sizes: dict[str, tuple[int, int]] | None = None,
    expected_rows: int | None = None,
) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Detection table is missing columns: {missing}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Detection table has {len(frame)} rows; expected {expected_rows} predictions.")
    allowed = set(class_names())
    for index, row in frame.iterrows():
        if row["class_name"] not in allowed:
            raise ValueError(f"Row {index} has unknown class {row['class_name']}")
        if not (float(row["x_min"]) < float(row["x_max"]) and float(row["y_min"]) < float(row["y_max"])):
            raise ValueError(f"Row {index} has inverted box coordinates")
        width = float(row["x_max"]) - float(row["x_min"])
        height = float(row["y_max"]) - float(row["y_min"])
        area = float(row["bbox_area"])
        if area <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Row {index} has non-positive box area")
        if abs(area - (width * height)) > 1.0:
            raise ValueError(f"Row {index} bbox_area does not match its coordinates")
        confidence = float(row["confidence"])
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"Row {index} confidence is outside 0-1")
        is_tp = bool(row["is_tp"])
        is_fp = bool(row["is_fp"])
        if is_tp == is_fp:
            raise ValueError(f"Row {index} must be either a true positive or a false positive")
        if image_sizes and row["image_id"] in image_sizes:
            image_width, image_height = image_sizes[row["image_id"]]
            if (
                float(row["x_min"]) < -0.5
                or float(row["y_min"]) < -0.5
                or float(row["x_max"]) > image_width + 0.5
                or float(row["y_max"]) > image_height + 0.5
            ):
                raise ValueError(f"Row {index} box falls outside the image")


def write_detection_tables(rows: list[dict], false_negatives: list[dict], expected_rows: int) -> pd.DataFrame:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=REQUIRED_COLUMNS + ["matched_gt_id", "iou", "image_width", "image_height"])
    image_sizes = {
        row["image_id"]: (int(row["image_width"]), int(row["image_height"]))
        for row in rows
    }
    validate_detections(frame, image_sizes=image_sizes, expected_rows=expected_rows)
    column_order = REQUIRED_COLUMNS + ["matched_gt_id", "iou", "image_width", "image_height"]
    frame = frame[column_order]
    frame.to_csv(TABLES / "detections.csv", index=False)
    frame.to_parquet(TABLES / "detections.parquet", index=False)
    negatives = pd.DataFrame(false_negatives)
    if negatives.empty:
        negatives = pd.DataFrame(columns=["image_id", "class_name", "x_min", "y_min", "x_max", "y_max", "split", "gt_id"])
    negatives.to_csv(TABLES / "false_negatives.csv", index=False)
    negatives.to_parquet(TABLES / "false_negatives.parquet", index=False)
    write_json(
        TABLES / "export_manifest.json",
        {"n_predictions": expected_rows, "n_rows": int(len(frame)), "n_false_negatives": int(len(negatives))},
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite detection tables from the saved prediction rows.")
    parser.parse_args()
    count_path = METRICS / "prediction_count.json"
    rows_path = TABLES / "prediction_rows.json"
    fn_path = TABLES / "false_negative_rows.json"
    if not rows_path.is_file() or not count_path.is_file():
        raise SystemExit("Prediction rows are missing. Run src.evaluate first.")
    import json

    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    false_negatives = json.loads(fn_path.read_text(encoding="utf-8")) if fn_path.is_file() else []
    expected = int(read_json(count_path)["n_predictions"])
    write_detection_tables(rows, false_negatives, expected)
    print(f"Wrote {expected} detection rows.")


if __name__ == "__main__":
    main()
