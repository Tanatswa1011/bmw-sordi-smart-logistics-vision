"""Export schema checks."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.export_detections import REQUIRED_COLUMNS, validate_detections
from src.settings import METRICS, TABLES


def _row(**overrides) -> dict:
    row = {
        "image_id": "frame",
        "class_name": "pallet",
        "confidence": 0.8,
        "x_min": 10.0,
        "y_min": 12.0,
        "x_max": 40.0,
        "y_max": 36.0,
        "bbox_area": 720.0,
        "model_version": "yolo11n-test",
        "split": "test",
        "is_tp": True,
        "is_fp": False,
    }
    row.update(overrides)
    return row


class ExportTests(unittest.TestCase):
    def test_required_columns_and_geometry(self) -> None:
        frame = pd.DataFrame([_row(), _row(image_id="other", class_name="dolly", is_tp=False, is_fp=True)])
        validate_detections(frame, {"frame": (100, 80), "other": (100, 80)}, expected_rows=2)

    def test_rejects_bad_area_and_inverted_box(self) -> None:
        with self.assertRaises(ValueError):
            validate_detections(pd.DataFrame([_row(bbox_area=5.0)]))
        with self.assertRaises(ValueError):
            validate_detections(pd.DataFrame([_row(x_min=50.0, x_max=10.0, bbox_area=1.0)]))

    def test_saved_export_matches_prediction_count(self) -> None:
        table_path = TABLES / "detections.csv"
        count_path = METRICS / "prediction_count.json"
        if not table_path.is_file() or not count_path.is_file():
            self.skipTest("evaluation export has not been written")
        frame = pd.read_csv(table_path)
        expected = int(json.loads(count_path.read_text(encoding="utf-8"))["n_predictions"])
        self.assertEqual(len(frame), expected)
        for column in REQUIRED_COLUMNS:
            self.assertIn(column, frame.columns)
        sizes = {
            row.image_id: (int(row.image_width), int(row.image_height))
            for row in frame.itertuples()
        }
        frame["is_tp"] = frame["is_tp"].map(lambda value: str(value).lower() in {"1", "true"})
        frame["is_fp"] = frame["is_fp"].map(lambda value: str(value).lower() in {"1", "true"})
        validate_detections(frame, sizes, expected_rows=expected)


if __name__ == "__main__":
    unittest.main()
