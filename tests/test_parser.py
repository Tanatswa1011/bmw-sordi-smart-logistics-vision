"""Parser checks that do not need the SORDI download."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.boxes import boxes_look_normalized, iter_annotation_objects, pixel_box, to_yolo
from src.data_prepare import parse_target_boxes


class ParserTests(unittest.TestCase):
    def test_nested_object_is_found(self) -> None:
        payload = {
            "frames": [
                {
                    "objects": [
                        {"ObjectClassName": "dolly", "Left": 0, "Top": 0, "Right": 10, "Bottom": 12, "Id": "a"}
                    ]
                }
            ]
        }
        objects = list(iter_annotation_objects(payload))
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["ObjectClassName"], "dolly")

    def test_invalid_and_non_numeric_boxes_are_rejected(self) -> None:
        self.assertIsNone(to_yolo(80, 10, 40, 50, 200, 100))
        self.assertIsNone(pixel_box({"Left": "nope", "Top": 1, "Right": 2, "Bottom": 3}, 20, 20, False))

    def test_yolo_values_stay_inside_unit_interval(self) -> None:
        yolo = to_yolo(10, 20, 110, 80, 200, 100)
        self.assertIsNotNone(yolo)
        assert yolo is not None
        self.assertTrue(all(0.0 <= value <= 1.0 for value in yolo))
        self.assertGreater(yolo[2], 0.0)
        self.assertGreater(yolo[3], 0.0)

    def test_normalized_boxes_scale_to_pixels(self) -> None:
        raw = [{"ObjectClassName": "pallet", "Left": 0.1, "Top": 0.2, "Right": 0.5, "Bottom": 0.4}]
        self.assertTrue(boxes_look_normalized(raw, 1280, 720))
        self.assertEqual(pixel_box(raw[0], 100, 50, True), (10.0, 10.0, 50.0, 20.0))

    def test_parse_keeps_only_target_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "frame.jpg"
            Image.new("RGB", (200, 100), "white").save(image_path)
            label_path = root / "frame.json"
            label_path.write_text(
                json.dumps(
                    [
                        {"ObjectClassName": "pallet", "Left": 10, "Top": 10, "Right": 80, "Bottom": 60},
                        {"ObjectClassName": "logo", "Left": 1, "Top": 1, "Right": 20, "Bottom": 20},
                        {"ObjectClassName": "forklift", "Left": 90, "Top": 10, "Right": 40, "Bottom": 20},
                    ]
                ),
                encoding="utf-8",
            )
            record = parse_target_boxes(image_path, label_path, ["pallet", "stillage", "forklift", "dolly"])
            self.assertEqual([box["class_name"] for box in record["boxes"]], ["pallet"])
            self.assertEqual(record["rejected"], 1)
            self.assertTrue(all(0.0 <= value <= 1.0 for box in record["boxes"] for value in box["yolo"]))


if __name__ == "__main__":
    unittest.main()
