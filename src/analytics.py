"""KPI calculations shared by the notebook and the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CLASS_ORDER = ["pallet", "stillage", "forklift", "dolly"]
LOW_CONFIDENCE_THRESHOLD = 0.40

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


def low_confidence_threshold() -> float:
    return LOW_CONFIDENCE_THRESHOLD


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def load_table(path: Path) -> pd.DataFrame:
    # CSV-first for the hosted dashboard. Parquet remains optional for local notebooks
    # when pyarrow is installed via requirements-training.txt.
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    for column in ("is_tp", "is_fp"):
        if column in frame.columns:
            frame[column] = _as_bool(frame[column])
    return frame


def default_detections_path() -> Path:
    csv_path = TABLES / "detections.csv"
    if csv_path.is_file():
        return csv_path
    return TABLES / "detections.parquet"


def default_fn_path() -> Path:
    csv_path = TABLES / "false_negatives.csv"
    if csv_path.is_file():
        return csv_path
    return TABLES / "false_negatives.parquet"


def filter_detections(
    detections: pd.DataFrame,
    classes: list[str] | None = None,
    splits: list[str] | None = None,
    status: str = "all",
    min_confidence: float | None = None,
) -> pd.DataFrame:
    view = detections
    if classes:
        view = view[view["class_name"].isin(classes)]
    if splits:
        view = view[view["split"].isin(splits)]
    if status == "tp":
        view = view[view["is_tp"]]
    elif status == "fp":
        view = view[view["is_fp"]]
    if min_confidence is not None:
        view = view[view["confidence"] >= float(min_confidence)]
    return view.copy()


def filter_false_negatives(
    false_negatives: pd.DataFrame,
    classes: list[str] | None = None,
    splits: list[str] | None = None,
) -> pd.DataFrame:
    view = false_negatives
    if classes:
        view = view[view["class_name"].isin(classes)]
    if splits:
        view = view[view["split"].isin(splits)]
    return view.copy()


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def summarize(
    detections: pd.DataFrame,
    false_negatives: pd.DataFrame | None = None,
    threshold: float | None = None,
) -> dict:
    """Operational KPIs. Precision/recall here use the one-to-one IoU matcher, not mAP."""
    if false_negatives is None:
        false_negatives = pd.DataFrame(columns=["class_name", "split"])
    threshold = low_confidence_threshold() if threshold is None else threshold
    total = int(len(detections))
    classes = [
        name
        for name in CLASS_ORDER
        if name in set(detections.get("class_name", [])) or name in set(false_negatives.get("class_name", []))
    ]
    by_class = {name: int((detections["class_name"] == name).sum()) if total else 0 for name in classes}
    mix = {name: (count / total if total else 0.0) for name, count in by_class.items()}
    avg_conf = {}
    precision = {}
    recall = {}
    fp_by_class = {}
    fn_by_class = {}
    for name in classes:
        subset = detections[detections["class_name"] == name] if total else detections
        avg_conf[name] = float(subset["confidence"].mean()) if len(subset) else None
        tp = int(subset["is_tp"].sum()) if len(subset) else 0
        fp = int(subset["is_fp"].sum()) if len(subset) else 0
        fn = int((false_negatives["class_name"] == name).sum()) if len(false_negatives) else 0
        precision[name] = _ratio(tp, tp + fp)
        recall[name] = _ratio(tp, tp + fn)
        fp_by_class[name] = fp
        fn_by_class[name] = fn
    ious = []
    if total and "iou" in detections.columns:
        matched = detections.loc[detections["is_tp"], "iou"].dropna()
        ious = [float(value) for value in matched.tolist()]
    return {
        "total_detections": total,
        "mean_confidence": float(detections["confidence"].mean()) if total else None,
        "fp_count": int(detections["is_fp"].sum()) if total else 0,
        "fn_count": int(len(false_negatives)),
        "low_confidence_rate": float((detections["confidence"] < threshold).mean()) if total else None,
        "low_confidence_threshold": threshold,
        "detections_by_class": by_class,
        "class_mix": mix,
        "average_confidence_by_class": avg_conf,
        "precision_by_class": precision,
        "recall_by_class": recall,
        "fp_by_class": fp_by_class,
        "fn_by_class": fn_by_class,
        "iou_values": ious,
    }
