"""Split checks for the capped, stratified subset."""

from __future__ import annotations

import random
import unittest
from pathlib import Path

import pandas as pd

from src.data_prepare import assign_splits, stratified_cap
from src.settings import ROOT, class_names

CLASSES = class_names()
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _records(count: int) -> list[dict]:
    records = []
    for index in range(count):
        name = CLASSES[index % len(CLASSES)]
        present = {name}
        if index % 5 == 0:
            present.add(CLASSES[(index + 1) % len(CLASSES)])
        records.append({"image_id": f"img_{index:04d}", "present": present})
    return records


class SplitTests(unittest.TestCase):
    def test_cap_and_split_constraints(self) -> None:
        rng = random.Random(42)
        chosen = stratified_cap(_records(1500), 1200, CLASSES, rng)
        self.assertLessEqual(len(chosen), 1200)
        self.assertEqual(len({record["image_id"] for record in chosen}), len(chosen))
        for name in CLASSES:
            self.assertTrue(any(name in record["present"] for record in chosen))
        assignment = assign_splits(chosen, FRACTIONS, CLASSES, random.Random(42))
        groups = {split: set() for split in FRACTIONS}
        for image_id, split in assignment.items():
            groups[split].add(image_id)
        self.assertEqual(set().union(*groups.values()), set(assignment))
        self.assertEqual(len(groups["train"] & groups["val"]), 0)
        self.assertEqual(len(groups["train"] & groups["test"]), 0)
        self.assertEqual(len(groups["val"] & groups["test"]), 0)
        total = len(assignment)
        for split, fraction in FRACTIONS.items():
            self.assertLess(abs(len(groups[split]) / total - fraction), 0.03)
        by_id = {record["image_id"]: record for record in chosen}
        for split, image_ids in groups.items():
            present = set().union(*(by_id[image_id]["present"] for image_id in image_ids))
            self.assertTrue(set(CLASSES).issubset(present), split)

    def test_manifest_when_present(self) -> None:
        path = ROOT / "data" / "splits" / "manifest.csv"
        if not path.is_file():
            self.skipTest("manifest is created during data preparation")
        frame = pd.read_csv(path)
        self.assertLessEqual(len(frame), 1200)
        self.assertEqual(frame["image_id"].nunique(), len(frame))
        splits = {name: set(frame.loc[frame["split"] == name, "image_id"]) for name in ("train", "val", "test")}
        self.assertEqual(len(splits["train"] & splits["val"]), 0)
        self.assertEqual(len(splits["train"] & splits["test"]), 0)
        self.assertEqual(len(splits["val"] & splits["test"]), 0)
        for name in CLASSES:
            for split in ("train", "val", "test"):
                column = f"contains_{name}"
                self.assertGreater(int(frame.loc[frame["split"] == split, column].sum()), 0)


if __name__ == "__main__":
    unittest.main()
