"""Bounding-box geometry shared by preparation, evaluation, and tests."""

from __future__ import annotations

from typing import Any, Iterable

BOX_KEYS = ("Left", "Top", "Right", "Bottom")


def iter_annotation_objects(node: Any) -> Iterable[dict]:
    """Yield dicts that carry ObjectClassName and pixel (or normalized) box edges."""
    if isinstance(node, dict):
        if "ObjectClassName" in node and all(key in node for key in BOX_KEYS):
            yield node
            return
        for value in node.values():
            yield from iter_annotation_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_annotation_objects(item)


def boxes_look_normalized(objects: list[dict], width: int, height: int) -> bool:
    if not objects or width <= 2 or height <= 2:
        return False
    max_x = max(float(obj["Right"]) for obj in objects)
    max_y = max(float(obj["Bottom"]) for obj in objects)
    return max_x <= 1.5 and max_y <= 1.5


def pixel_box(obj: dict, width: int, height: int, normalized: bool) -> tuple[float, float, float, float] | None:
    try:
        left = float(obj["Left"])
        top = float(obj["Top"])
        right = float(obj["Right"])
        bottom = float(obj["Bottom"])
    except (TypeError, ValueError):
        return None
    if normalized:
        left *= width
        right *= width
        top *= height
        bottom *= height
    return left, top, right, bottom


def clip_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    width: int,
    height: int,
    min_pixels: float = 1.0,
) -> tuple[float, float, float, float] | None:
    left = min(max(left, 0.0), float(width))
    right = min(max(right, 0.0), float(width))
    top = min(max(top, 0.0), float(height))
    bottom = min(max(bottom, 0.0), float(height))
    if right - left < min_pixels or bottom - top < min_pixels:
        return None
    if not (left < right and top < bottom):
        return None
    return left, top, right, bottom


def to_yolo(
    left: float,
    top: float,
    right: float,
    bottom: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    clipped = clip_box(left, top, right, bottom, width, height)
    if clipped is None:
        return None
    left, top, right, bottom = clipped
    x_center = ((left + right) / 2.0) / width
    y_center = ((top + bottom) / 2.0) / height
    box_w = (right - left) / width
    box_h = (bottom - top) / height
    values = (
        _clamp01(x_center),
        _clamp01(y_center),
        _clamp01(box_w),
        _clamp01(box_h),
    )
    if values[2] <= 0.0 or values[3] <= 0.0:
        return None
    if any(value < 0.0 or value > 1.0 for value in values):
        return None
    return values


def yolo_to_pixels(
    x_center: float,
    y_center: float,
    box_w: float,
    box_h: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    left = (x_center - box_w / 2.0) * width
    top = (y_center - box_h / 2.0) * height
    right = (x_center + box_w / 2.0) * width
    bottom = (y_center + box_h / 2.0) * height
    return clip_box(left, top, right, bottom, width, height)


def box_area(left: float, top: float, right: float, bottom: float) -> float:
    return max(0.0, right - left) * max(0.0, bottom - top)


def iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = box_area(left, top, right, bottom)
    if intersection <= 0.0:
        return 0.0
    union = box_area(*first) + box_area(*second) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def match_detections(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.50,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Greedy one-to-one matching. A pair must share a class and IoU >= threshold.

    Returns matches as (prediction_index, ground_truth_index, iou), plus unmatched
    prediction indexes and unmatched ground-truth indexes.
    """
    candidates: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, truth in enumerate(ground_truth):
            if prediction["class_name"] != truth["class_name"]:
                continue
            score = iou(prediction["box"], truth["box"])
            if score >= iou_threshold:
                candidates.append((score, pred_index, gt_index))
    candidates.sort(key=lambda item: item[0], reverse=True)
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, pred_index, gt_index in candidates:
        if pred_index in used_predictions or gt_index in used_truths:
            continue
        used_predictions.add(pred_index)
        used_truths.add(gt_index)
        matches.append((pred_index, gt_index, score))
    unmatched_predictions = {index for index in range(len(predictions)) if index not in used_predictions}
    unmatched_truths = {index for index in range(len(ground_truth)) if index not in used_truths}
    return matches, unmatched_predictions, unmatched_truths


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
