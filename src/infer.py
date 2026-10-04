"""Batch or single-image inference without the dashboard."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from src.boxes import box_area, clip_box
from src.settings import train_config
from src.train import _cuda_or_stop

COLORS = {
    "pallet": (230, 126, 34),
    "stillage": (52, 152, 219),
    "forklift": (192, 57, 43),
    "dolly": (39, 174, 96),
}


def run_inference(source: Path, weights: Path, output: Path, confidence: float | None = None) -> Path:
    _cuda_or_stop()
    from ultralytics import YOLO

    config = train_config()
    threshold = float(config["confidence_threshold"] if confidence is None else confidence)
    output.mkdir(parents=True, exist_ok=True)
    annotated = output / "annotated"
    annotated.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        conf=threshold,
        imgsz=int(config["imgsz"]),
        device=int(config["device"]),
        batch=int(config["batch_fallback"]),
        stream=True,
        verbose=False,
        save=False,
    )
    rows = []
    model_version = weights.stem
    for result in results:
        image_path = Path(result.path)
        image = Image.open(image_path).convert("RGB")
        drawing = ImageDraw.Draw(image)
        height, width = result.orig_shape
        if result.boxes is not None:
            for box, score, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                clipped = clip_box(float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(width), int(height))
                if clipped is None:
                    continue
                class_name = result.names[int(class_id)]
                left, top, right, bottom = clipped
                drawing.rectangle((left, top, right, bottom), outline=COLORS.get(class_name, (255, 255, 255)), width=3)
                drawing.text((left + 2, max(0, top - 12)), f"{class_name} {float(score):.2f}", fill=COLORS.get(class_name, (255, 255, 255)))
                rows.append(
                    {
                        "image_id": image_path.stem,
                        "class_name": class_name,
                        "confidence": float(score),
                        "x_min": left,
                        "y_min": top,
                        "x_max": right,
                        "y_max": bottom,
                        "bbox_area": box_area(left, top, right, bottom),
                        "model_version": model_version,
                    }
                )
        image.save(annotated / image_path.name, quality=90)
    csv_path = output / "detections.csv"
    fieldnames = ["image_id", "class_name", "confidence", "x_min", "y_min", "x_max", "y_max", "bbox_area", "model_version"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} detections to {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO11n on an image or a folder.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--conf", type=float, default=None)
    args = parser.parse_args()
    run_inference(args.source, args.weights, args.output, args.conf)


if __name__ == "__main__":
    main()
