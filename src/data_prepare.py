"""Load DR-1, convert SORDI JSON boxes to YOLO labels, and write a capped split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from src.boxes import boxes_look_normalized, clip_box, iter_annotation_objects, pixel_box, to_yolo
from src.settings import (
    CONFIGS,
    FIGURES,
    LOGS,
    METRICS,
    SPLITS,
    class_names,
    class_to_id,
    classes_config,
    kaggle_subprocess_env,
    require_kaggle_token,
    write_json,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _path_has_part(name: str, part: str) -> bool:
    normalized = name.replace("\\", "/").strip("/")
    return part.upper() in [piece.upper() for piece in normalized.split("/")]
COLORS = {
    "pallet": (230, 126, 34),
    "stillage": (52, 152, 219),
    "forklift": (192, 57, 43),
    "dolly": (39, 174, 96),
}


def cache_dir() -> Path:
    path = Path(classes_config()["cache_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def yolo_root() -> Path:
    return cache_dir() / "yolo"


def find_named_dirs(root: Path, name: str) -> list[Path]:
    matches: list[Path] = []
    if not root.exists():
        return matches
    target = name.upper()
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [dirname for dirname in dirnames if dirname.upper() != "DR-2"]
        current = Path(dirpath)
        if current.name.upper() == target:
            matches.append(current)
            dirnames.clear()
    return matches


def tree_summary(root: Path, max_depth: int = 3) -> dict:
    summary: dict = {"root": str(root), "entries": []}
    if not root.exists():
        return summary
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)
        if depth > max_depth:
            dirnames.clear()
            continue
        summary["entries"].append(
            {
                "path": str(current),
                "depth": depth,
                "subdirs": sorted(dirnames)[:40],
                "n_files": len(filenames),
                "file_suffixes": dict(Counter(Path(name).suffix.lower() for name in filenames)),
            }
        )
    return summary


def _kaggle_executable() -> str:
    scripts = Path(sys.executable).parent
    for name in ("kaggle.exe", "kaggle"):
        candidate = scripts / name
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("The kaggle command is not installed in the project virtual environment.")


def download_dataset() -> None:
    require_kaggle_token()
    destination = cache_dir()
    if find_named_dirs(destination, "DR-1"):
        print(f"Reusing cached DR-1 under {destination}")
        return
    zip_files = list(destination.glob("*.zip"))
    if not zip_files:
        print(f"Downloading sordi-ai/industrial-scene-01-dataset into {destination}")
        subprocess.run(
            [
                _kaggle_executable(),
                "datasets",
                "download",
                "-d",
                classes_config()["kaggle_dataset"],
                "-p",
                str(destination),
            ],
            check=True,
            env=kaggle_subprocess_env(),
        )
        zip_files = list(destination.glob("*.zip"))
    if find_named_dirs(destination, "DR-1"):
        return
    if not zip_files:
        raise RuntimeError(f"Kaggle download finished but no zip or DR-1 folder was found in {destination}")
    for archive in zip_files:
        print(f"Extracting DR-1 members from {archive.name}")
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            dr1_members = [name for name in names if _path_has_part(name, "DR-1")]
            skipped = [name for name in names if _path_has_part(name, "DR-2")]
            write_json(
                LOGS / "zip_members.json",
                {"archive": archive.name, "members": len(names), "dr1_members": len(dr1_members), "dr2_members": len(skipped)},
            )
            if not dr1_members:
                handle.extractall(destination)
                continue
            for name in names:
                if _path_has_part(name, "DR-2"):
                    continue
                handle.extract(name, destination)
    if not find_named_dirs(destination, "DR-1"):
        write_json(LOGS / "dataset_structure.json", tree_summary(destination))
        raise RuntimeError(
            "DR-1 folder was not found after extract. "
            f"The discovered tree is in {LOGS / 'dataset_structure.json'}."
        )


def locate_dr1() -> Path:
    matches = find_named_dirs(cache_dir(), "DR-1")
    if not matches:
        write_json(LOGS / "dataset_structure.json", tree_summary(cache_dir()))
        raise RuntimeError("DR-1 was not found. Inspect outputs/logs/dataset_structure.json.")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda path: len(str(path)))
    chosen = matches[0]
    write_json(
        LOGS / "dataset_structure.json",
        {
            "dr1_root": str(chosen),
            "candidates": [str(path) for path in matches],
            "tree": tree_summary(chosen, max_depth=2),
        },
    )
    print(f"DR-1 root: {chosen}")
    return chosen


def pair_frames(dr1_root: Path) -> list[tuple[Path, Path]]:
    images = [path for path in dr1_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    json_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in dr1_root.rglob("*.json"):
        if path.name.lower() in {"objectclasses.json", "objectclassname.json"}:
            continue
        name = path.name.lower()
        json_by_stem[path.stem.lower()].append(path)
        for extension in IMAGE_EXTENSIONS:
            suffix = f"{extension}.json"
            if name.endswith(suffix):
                json_by_stem[name[: -len(suffix)]].append(path)
    pairs: list[tuple[Path, Path]] = []
    unresolved = 0
    for image_path in images:
        candidates = json_by_stem.get(image_path.stem.lower(), [])
        label = _choose_label(image_path, candidates)
        if label is None:
            sibling = image_path.with_suffix(".json")
            label = sibling if sibling.is_file() else None
        if label is None:
            unresolved += 1
            continue
        pairs.append((image_path, label))
    pairs.sort(key=lambda item: item[0].name.lower())
    write_json(
        LOGS / "pairing_summary.json",
        {
            "dr1_root": str(dr1_root),
            "images": len(images),
            "paired": len(pairs),
            "unresolved_images": unresolved,
            "sample_pair": [str(pairs[0][0]), str(pairs[0][1])] if pairs else None,
        },
    )
    if not pairs:
        raise RuntimeError("No image/JSON pairs were found under DR-1.")
    print(f"Paired {len(pairs)} DR-1 frames ({unresolved} images had no JSON).")
    return pairs


def _choose_label(image_path: Path, candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    image_parts = [part.lower() for part in image_path.parts]

    def score(label: Path) -> tuple[int, int]:
        same_parent = int(label.parent == image_path.parent)
        label_parts = [part.lower() for part in label.parts]
        shared = len(set(image_parts) & set(label_parts))
        return (same_parent, shared)

    return sorted(candidates, key=score, reverse=True)[0]


def read_objects(label_path: Path) -> list[dict]:
    payload = json.loads(label_path.read_text(encoding="utf-8-sig"))
    return list(iter_annotation_objects(payload))


def parse_target_boxes(image_path: Path, label_path: Path, names: list[str]) -> dict:
    with Image.open(image_path) as image:
        width, height = image.size
    raw_objects = read_objects(label_path)
    normalized = boxes_look_normalized(raw_objects, width, height)
    kept = []
    rejected = 0
    present = set()
    counts = Counter()
    for obj in raw_objects:
        class_name = str(obj.get("ObjectClassName", "")).strip()
        if class_name not in names:
            continue
        pixels = pixel_box(obj, width, height, normalized)
        if pixels is None:
            rejected += 1
            continue
        clipped = clip_box(*pixels, width, height)
        yolo = to_yolo(*pixels, width, height) if clipped else None
        if clipped is None or yolo is None:
            rejected += 1
            continue
        kept.append(
            {
                "class_name": class_name,
                "class_id": class_to_id()[class_name],
                "box": clipped,
                "yolo": yolo,
                "object_id": str(obj.get("Id", obj.get("ObjectClassId", ""))),
            }
        )
        present.add(class_name)
        counts[class_name] += 1
    return {
        "image_id": image_path.stem,
        "image_path": image_path,
        "label_path": label_path,
        "width": width,
        "height": height,
        "normalized": normalized,
        "boxes": kept,
        "present": present,
        "counts": counts,
        "rejected": rejected,
    }


def draw_overlay(record: dict, destination: Path) -> None:
    image = Image.open(record["image_path"]).convert("RGB")
    drawing = ImageDraw.Draw(image)
    for box in record["boxes"]:
        color = COLORS[box["class_name"]]
        left, top, right, bottom = box["box"]
        drawing.rectangle((left, top, right, bottom), outline=color, width=3)
        drawing.text((left + 2, max(0, top - 12)), box["class_name"], fill=color)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=90)


def run_smoke() -> None:
    download_dataset()
    pairs = pair_frames(locate_dr1())
    names = class_names()
    sample_label = read_objects(pairs[0][1])
    write_json(
        LOGS / "sample_annotation.json",
        {"label_path": str(pairs[0][1]), "objects_found": len(sample_label), "first_object": sample_label[0] if sample_label else None},
    )
    selected = []
    rejected_total = 0
    for image_path, label_path in pairs:
        record = parse_target_boxes(image_path, label_path, names)
        rejected_total += record["rejected"]
        if not record["boxes"]:
            continue
        selected.append(record)
        if len(selected) == int(classes_config()["smoke_image_count"]):
            break
    if len(selected) != int(classes_config()["smoke_image_count"]):
        raise RuntimeError(f"Smoke test needed 20 target-class images and found {len(selected)}.")
    sample_dir = FIGURES / "samples"
    if sample_dir.exists():
        for old in sample_dir.glob("smoke_*.jpg"):
            old.unlink()
    for index, record in enumerate(selected, start=1):
        draw_overlay(record, sample_dir / f"smoke_{index:02d}_{record['image_id']}.jpg")
    report = {
        "images": len(selected),
        "objects": sum(len(record["boxes"]) for record in selected),
        "rejected_boxes": rejected_total,
        "normalized_files": sum(1 for record in selected if record["normalized"]),
        "class_objects": dict(sum((record["counts"] for record in selected), Counter())),
        "overlays": [str(path) for path in sorted(sample_dir.glob("smoke_*.jpg"))],
        "visual_check_passed": False,
    }
    write_json(LOGS / "smoke_report.json", report)
    print(f"Wrote {len(selected)} overlays to {sample_dir}")
    print("Review the overlays before full preparation.")


def stratified_cap(records: list[dict], max_images: int, names: list[str], rng: random.Random) -> list[dict]:
    if len(records) <= max_images:
        return list(records)
    by_class = {name: [record for record in records if name in record["present"]] for name in names}
    for bucket in by_class.values():
        rng.shuffle(bucket)
    selected: list[dict] = []
    selected_ids: set[str] = set()
    counts = Counter()
    while len(selected) < max_images:
        available = [
            name
            for name in names
            if any(record["image_id"] not in selected_ids for record in by_class[name])
        ]
        if not available:
            break

        def coverage(name: str) -> tuple[float, int, str]:
            available_count = len(by_class[name])
            ratio = counts[name] / available_count if available_count else 1.0
            return (ratio, counts[name], name)

        name = min(available, key=coverage)
        chosen = next(record for record in by_class[name] if record["image_id"] not in selected_ids)
        selected.append(chosen)
        selected_ids.add(chosen["image_id"])
        counts.update(chosen["present"])
    return selected


def assign_splits(records: list[dict], fractions: dict, names: list[str], rng: random.Random) -> dict[str, str]:
    ordered = list(records)
    rng.shuffle(ordered)
    total = len(ordered)
    train_count = int(round(fractions["train"] * total))
    val_count = int(round(fractions["val"] * total))
    test_count = total - train_count - val_count
    quotas = {"train": train_count, "val": val_count, "test": test_count}
    assignment: dict[str, str] = {}
    split_counts = Counter()
    class_counts = {split: Counter() for split in quotas}

    def place(record: dict, split: str) -> None:
        assignment[record["image_id"]] = split
        split_counts[split] += 1
        class_counts[split].update(record["present"])

    for split in ("test", "val", "train"):
        for name in names:
            if class_counts[split][name] > 0 or split_counts[split] >= quotas[split]:
                continue
            donor = next(
                (
                    record
                    for record in ordered
                    if record["image_id"] not in assignment and name in record["present"]
                ),
                None,
            )
            if donor is not None:
                place(donor, split)

    for record in ordered:
        if record["image_id"] in assignment:
            continue
        options = [split for split in quotas if split_counts[split] < quotas[split]]
        if not options:
            raise RuntimeError("Split quotas were exhausted before every image was assigned.")

        def need(split: str) -> tuple[int, float, str]:
            class_need = min(class_counts[split][name] for name in record["present"])
            fill = split_counts[split] / quotas[split]
            return (class_need, fill, split)

        place(record, min(options, key=need))

    _repair_missing_classes(ordered, assignment, class_counts, names)
    for split, quota in quotas.items():
        actual = sum(1 for value in assignment.values() if value == split)
        if actual != quota:
            raise RuntimeError(f"{split} has {actual} images; expected {quota}.")
    return assignment


def _repair_missing_classes(records, assignment, class_counts, names) -> None:
    by_id = {record["image_id"]: record for record in records}
    for _ in range(12):
        missing = [
            (split, name)
            for split in ("train", "val", "test")
            for name in names
            if class_counts[split][name] == 0
        ]
        if not missing:
            return
        split, name = missing[0]
        donor_id = next(
            (
                image_id
                for image_id, owner in assignment.items()
                if owner != split and name in by_id[image_id]["present"] and _can_leave(image_id, by_id, assignment, class_counts, names)
            ),
            None,
        )
        receiver_id = next(
            (
                image_id
                for image_id, owner in assignment.items()
                if owner == split and _can_leave(image_id, by_id, assignment, class_counts, names)
            ),
            None,
        )
        if donor_id is None or receiver_id is None:
            raise RuntimeError(f"Could not place class {name} into the {split} split.")
        _swap(donor_id, receiver_id, by_id, assignment, class_counts)
    raise RuntimeError("Class coverage repair did not converge.")


def _can_leave(image_id, by_id, assignment, class_counts, names) -> bool:
    split = assignment[image_id]
    for name in by_id[image_id]["present"]:
        if class_counts[split][name] <= 1:
            return False
    return True


def _swap(first_id, second_id, by_id, assignment, class_counts) -> None:
    first_split = assignment[first_id]
    second_split = assignment[second_id]
    for name in by_id[first_id]["present"]:
        class_counts[first_split][name] -= 1
        class_counts[second_split][name] += 1
    for name in by_id[second_id]["present"]:
        class_counts[second_split][name] -= 1
        class_counts[first_split][name] += 1
    assignment[first_id] = second_split
    assignment[second_id] = first_split


def prepare_dataset(i_verified_smoke: bool) -> None:
    smoke_path = LOGS / "smoke_report.json"
    if not smoke_path.is_file():
        run_smoke()
        raise SystemExit("Smoke overlays are in outputs/figures/samples. Re-run prepare only after they look correct.")
    report = json.loads(smoke_path.read_text(encoding="utf-8"))
    if not i_verified_smoke and not report.get("visual_check_passed"):
        raise SystemExit(
            "Smoke overlays have not been marked correct. "
            "Review outputs/figures/samples, then run prepare with --i-verified-smoke."
        )
    download_dataset()
    names = class_names()
    pairs = pair_frames(locate_dr1())
    records = []
    seen_ids: set[str] = set()
    for image_path, label_path in pairs:
        record = parse_target_boxes(image_path, label_path, names)
        if not record["boxes"]:
            continue
        if record["image_id"] in seen_ids:
            record["image_id"] = f"{record['image_id']}_{len(seen_ids)}"
        seen_ids.add(record["image_id"])
        records.append(record)
    config = classes_config()
    rng = random.Random(int(config["seed"]))
    chosen = stratified_cap(records, int(config["max_images"]), names, rng)
    if len(chosen) > int(config["max_images"]):
        raise RuntimeError("Subset cap was exceeded.")
    missing = [name for name in names if not any(name in record["present"] for record in chosen)]
    if missing:
        raise RuntimeError(f"Subset is missing classes: {missing}")
    split_rng = random.Random(int(config["seed"]))
    assignment = assign_splits(chosen, config["split_fractions"], names, split_rng)
    _write_yolo(chosen, assignment)
    _write_manifest(chosen, assignment)
    _write_dataset_yaml()
    summary = _summary(chosen, assignment, names)
    write_json(METRICS / "dataset_summary.json", summary)
    print(json.dumps(summary, indent=2))


def _write_yolo(records: list[dict], assignment: dict[str, str]) -> None:
    root = yolo_root()
    if root.exists():
        shutil.rmtree(root)
    for record in records:
        split = assignment[record["image_id"]]
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        suffix = record["image_path"].suffix.lower()
        destination = image_dir / f"{record['image_id']}{suffix}"
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
    with (SPLITS / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (assignment[item["image_id"]], item["image_id"])):
            counts = record["counts"]
            present = record["present"]
            writer.writerow(
                {
                    "image_id": record["image_id"],
                    "source_path": str(record["image_path"]),
                    "label_path": str(record["label_path"]),
                    "split": assignment[record["image_id"]],
                    "width": record["width"],
                    "height": record["height"],
                    "contains_pallet": int("pallet" in present),
                    "contains_stillage": int("stillage" in present),
                    "contains_forklift": int("forklift" in present),
                    "contains_dolly": int("dolly" in present),
                    "n_pallet": int(counts["pallet"]),
                    "n_stillage": int(counts["stillage"]),
                    "n_forklift": int(counts["forklift"]),
                    "n_dolly": int(counts["dolly"]),
                }
            )


def _write_dataset_yaml() -> None:
    root = yolo_root().resolve()
    lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for index, name in enumerate(class_names()):
        lines.append(f"  {index}: {name}")
    (CONFIGS / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(records: list[dict], assignment: dict[str, str], names: list[str]) -> dict:
    summary = {"images": len(records), "by_split": {}, "objects": {}, "images_with_class": {}}
    for name in names:
        summary["objects"][name] = sum(record["counts"][name] for record in records)
        summary["images_with_class"][name] = sum(1 for record in records if name in record["present"])
    for split in ("train", "val", "test"):
        subset = [record for record in records if assignment[record["image_id"]] == split]
        summary["by_split"][split] = {
            "images": len(subset),
            "objects": {name: sum(record["counts"][name] for record in subset) for name in names},
            "images_with_class": {name: sum(1 for record in subset if name in record["present"]) for name in names},
        }
    return summary


def mark_smoke_verified() -> None:
    path = LOGS / "smoke_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["visual_check_passed"] = True
    write_json(path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the SORDI DR-1 logistics subset.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke")
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--i-verified-smoke", action="store_true")
    subparsers.add_parser("mark-smoke")
    args = parser.parse_args()
    if args.command == "smoke":
        run_smoke()
    elif args.command == "mark-smoke":
        mark_smoke_verified()
    else:
        prepare_dataset(args.i_verified_smoke)


if __name__ == "__main__":
    main()
