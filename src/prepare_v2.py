"""Build a fresh DR-1 subset for V2. Does not rewrite the V1 YOLO dataset or manifest."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from src.data_prepare import (
    assign_splits,
    cache_dir,
    download_dataset,
    locate_dr1,
    pair_frames,
    parse_target_boxes,
    stratified_cap,
)
from src.settings import CONFIGS, OUTPUTS, SPLITS, class_names, classes_config, load_yaml, write_json

V2_METRICS = OUTPUTS / "v2" / "metrics"
CACHE_PATH = cache_dir() / "v2_qualifying_records.jsonl"


def v2_config() -> dict:
    return load_yaml(CONFIGS / "train_v2.yaml")


def yolo_v2_root() -> Path:
    return cache_dir() / v2_config()["yolo_dirname"]


def _record_to_json(record: dict) -> dict:
    return {
        "image_id": record["image_id"],
        "image_path": str(record["image_path"]),
        "label_path": str(record["label_path"]),
        "width": record["width"],
        "height": record["height"],
        "present": sorted(record["present"]),
        "counts": {name: int(record["counts"][name]) for name in class_names()},
        "boxes": [
            {
                "class_name": box["class_name"],
                "class_id": box["class_id"],
                "yolo": list(box["yolo"]),
            }
            for box in record["boxes"]
        ],
    }


def _record_from_json(payload: dict) -> dict:
    payload["image_path"] = Path(payload["image_path"])
    payload["label_path"] = Path(payload["label_path"])
    payload["present"] = set(payload["present"])
    payload["counts"] = Counter(payload["counts"])
    for box in payload["boxes"]:
        box["yolo"] = tuple(box["yolo"])
    return payload


def load_qualifying_records() -> list[dict]:
    names = class_names()
    if CACHE_PATH.is_file() and CACHE_PATH.stat().st_size > 0:
        records = [_record_from_json(json.loads(line)) for line in CACHE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"Reusing {len(records)} cached qualifying DR-1 frames from {CACHE_PATH}")
        return records
    download_dataset()
    pairs = pair_frames(locate_dr1())
    records = []
    seen_ids: set[str] = set()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as handle:
        for index, (image_path, label_path) in enumerate(pairs, start=1):
            record = parse_target_boxes(image_path, label_path, names)
            if not record["boxes"]:
                continue
            if record["image_id"] in seen_ids:
                record["image_id"] = f"{record['image_id']}_{len(seen_ids)}"
            seen_ids.add(record["image_id"])
            records.append(record)
            handle.write(json.dumps(_record_to_json(record)) + "\n")
            if index % 1000 == 0:
                print(f"Parsed {index}/{len(pairs)} DR-1 frames; {len(records)} contain a target class.")
    print(f"Qualifying frames: {len(records)} of {len(pairs)} paired DR-1 frames.")
    return records


def _representation(records: list[dict], names: list[str]) -> dict:
    objects = {name: int(sum(record["counts"][name] for record in records)) for name in names}
    total = sum(objects.values())
    return {
        "images": len(records),
        "images_with_class": {name: int(sum(1 for record in records if name in record["present"])) for name in names},
        "objects": objects,
        "box_share": {name: (objects[name] / total if total else None) for name in names},
    }


def _assert_splits(records: list[dict], assignment: dict[str, str], names: list[str]) -> None:
    grouped = {split: set() for split in ("train", "val", "test")}
    for record in records:
        grouped[assignment[record["image_id"]]].add(record["image_id"])
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        overlap = grouped[left] & grouped[right]
        if overlap:
            raise RuntimeError(f"{left} and {right} share {len(overlap)} image ids.")
    for split, image_ids in grouped.items():
        subset = [record for record in records if record["image_id"] in image_ids]
        missing = [name for name in names if not any(name in record["present"] for record in subset)]
        if missing:
            raise RuntimeError(f"{split} is missing classes: {missing}")


def _v1_overlap(assignment: dict[str, str]) -> dict:
    manifest = SPLITS / "manifest.csv"
    if not manifest.is_file():
        return {"v1_manifest_present": False}
    v1_test = set()
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                v1_test.add(row["image_id"])
    return {
        "v1_test_ids": len(v1_test),
        "v1_test_ids_in_v2_train": sum(1 for image_id in v1_test if assignment.get(image_id) == "train"),
        "v1_test_ids_in_v2_val": sum(1 for image_id in v1_test if assignment.get(image_id) == "val"),
        "v1_test_ids_in_v2_test": sum(1 for image_id in v1_test if assignment.get(image_id) == "test"),
        "note": "V2 is a new split. Overlap with V1 image ids is reported and is not used as a score.",
    }


def _write_yolo(records: list[dict], assignment: dict[str, str]) -> None:
    import shutil

    root = yolo_v2_root()
    if root.exists():
        shutil.rmtree(root)
    for record in records:
        split = assignment[record["image_id"]]
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        destination = image_dir / f"{record['image_id']}{record['image_path'].suffix.lower()}"
        shutil.copy2(record["image_path"], destination)
        lines = [
            f"{box['class_id']} {box['yolo'][0]:.6f} {box['yolo'][1]:.6f} {box['yolo'][2]:.6f} {box['yolo'][3]:.6f}"
            for box in record["boxes"]
        ]
        (label_dir / f"{record['image_id']}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(records: list[dict], assignment: dict[str, str]) -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "source_path",
        "label_path",
        "split",
        "width",
        "height",
        "contains_pallet",
        "contains_stillage",
        "contains_forklift",
        "contains_dolly",
        "n_pallet",
        "n_stillage",
        "n_forklift",
        "n_dolly",
    ]
    names = class_names()
    with (SPLITS / "manifest_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (assignment[item["image_id"]], item["image_id"])):
            present = record["present"]
            counts = record["counts"]
            writer.writerow(
                {
                    "image_id": record["image_id"],
                    "source_path": str(record["image_path"]),
                    "label_path": str(record["label_path"]),
                    "split": assignment[record["image_id"]],
                    "width": record["width"],
                    "height": record["height"],
                    **{f"contains_{name}": int(name in present) for name in names},
                    **{f"n_{name}": int(counts[name]) for name in names},
                }
            )


def _write_dataset_yaml() -> None:
    root = yolo_v2_root().resolve()
    lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for index, name in enumerate(class_names()):
        lines.append(f"  {index}: {name}")
    (CONFIGS / "dataset_v2.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(records: list[dict], assignment: dict[str, str], names: list[str], full: dict) -> dict:
    summary = {
        "seed": int(v2_config()["seed"]),
        "target_images": int(v2_config()["max_images"]),
        "qualifying_images_before_cap": full["images"],
        "images": len(records),
        "by_split": {},
        "objects": {},
        "images_with_class": {},
    }
    for name in names:
        summary["objects"][name] = int(sum(record["counts"][name] for record in records))
        summary["images_with_class"][name] = int(sum(1 for record in records if name in record["present"]))
    for split in ("train", "val", "test"):
        subset = [record for record in records if assignment[record["image_id"]] == split]
        summary["by_split"][split] = {
            "images": len(subset),
            "objects": {name: int(sum(record["counts"][name] for record in subset)) for name in names},
            "images_with_class": {name: int(sum(1 for record in subset if name in record["present"])) for name in names},
        }
    return summary


def prepare_v2() -> None:
    import random

    config = v2_config()
    names = class_names()
    records = load_qualifying_records()
    full = _representation(records, names)
    target = int(config["max_images"])
    if len(records) < target:
        print(f"Only {len(records)} qualifying DR-1 images exist. Using all of them instead of duplicating frames.")
        chosen = list(records)
    else:
        chosen = stratified_cap(records, target, names, random.Random(int(config["seed"])))
    if len(chosen) > target:
        raise RuntimeError("V2 subset exceeded the image cap.")
    sampled = _representation(chosen, names)
    assignment = assign_splits(chosen, classes_config()["split_fractions"], names, random.Random(int(config["seed"]) + 1))
    _assert_splits(chosen, assignment, names)
    print("Writing the V2 YOLO dataset outside OneDrive.")
    _write_yolo(chosen, assignment)
    _write_manifest(chosen, assignment)
    _write_dataset_yaml()
    test_ids = sorted(image_id for image_id, split in assignment.items() if split == "test")
    frozen = {
        "seed": int(config["seed"]),
        "count": len(test_ids),
        "image_ids": test_ids,
        "frozen": True,
        "allowed_uses_before_final_eval": [],
    }
    write_json(OUTPUTS / "v2" / "splits" / "frozen_test_ids.json", frozen)
    summary = _summary(chosen, assignment, names, full)
    summary["v1_image_overlap"] = _v1_overlap(assignment)
    summary["representation_before_sampling"] = full
    summary["representation_after_sampling"] = sampled
    write_json(V2_METRICS / "dataset_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("images", "by_split", "objects", "images_with_class")}, indent=2))


if __name__ == "__main__":
    prepare_v2()
