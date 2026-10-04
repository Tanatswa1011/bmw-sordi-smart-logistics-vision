"""Prototype analytics view for the V1 and V2 synthetic logistics detectors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import (  # noqa: E402
    CLASS_ORDER,
    filter_detections,
    filter_false_negatives,
    load_table,
    summarize,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(page_title="BMW SORDI Smart Logistics Vision", layout="wide")
st.title("BMW SORDI Smart Logistics Vision")
st.caption(
    "Portfolio prototype using public synthetic SORDI data. "
    "Independent implementation. No BMW endorsement. "
    "Results do not represent real-plant performance."
)
st.info(
    "Prototype visual resource monitoring on synthetic industrial logistics scenes. "
    "Counts come from detectors trained on capped public DR-1 subsets. "
    "They are not a live warehouse inventory."
)

VERSIONS = {
    "V1": {
        "tables": ROOT / "outputs" / "v1" / "tables",
        "metrics": ROOT / "outputs" / "v1" / "metrics" / "test_metrics.json",
        "errors": ROOT / "outputs" / "v1" / "figures" / "errors",
        "threshold": 0.25,
    },
    "V2": {
        "tables": ROOT / "outputs" / "v2" / "tables",
        "metrics": ROOT / "outputs" / "v2" / "metrics" / "test_metrics.json",
        "errors": ROOT / "outputs" / "v2" / "figures" / "errors",
        "threshold": None,
    },
}


def _table(folder: Path, name: str) -> Path | None:
    for suffix in (".parquet", ".csv"):
        path = folder / f"{name}{suffix}"
        if path.is_file():
            return path
    return None


def _official_cards(payload: dict) -> None:
    run = payload["run_1"]
    precision = float(run["precision"])
    recall = float(run["recall"])
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else None
    cards = st.columns(5)
    cards[0].metric("Precision", f"{precision:.4f}")
    cards[1].metric("Recall", f"{recall:.4f}")
    cards[2].metric("F1", "n/a" if f1 is None else f"{f1:.4f}")
    cards[3].metric("mAP50", f"{float(run['map50']):.4f}")
    cards[4].metric("mAP50-95", f"{float(run['map50_95']):.4f}")
    per_class = run.get("per_class") or payload.get("per_class") or []
    if per_class and "ap50" in per_class[0]:
        frame = pd.DataFrame(per_class)
        keep = [
            column
            for column in ("class_name", "precision", "recall", "ap50", "ap50_95", "object_count")
            if column in frame.columns
        ]
        st.dataframe(frame[keep], hide_index=True, width="stretch")


def _direction(value: float) -> str:
    if value > 0:
        return "higher"
    if value < 0:
        return "lower"
    return "unchanged"


with st.sidebar:
    st.header("Model")
    available = [name for name, paths in VERSIONS.items() if _table(paths["tables"], "detections")]
    if not available:
        st.warning("No detection table is available yet.")
        st.stop()
    version = st.radio("Model version", available, index=0)
    paths = VERSIONS[version]

detections = load_table(_table(paths["tables"], "detections"))
fn_path = _table(paths["tables"], "false_negatives")
false_negatives = load_table(fn_path) if fn_path else pd.DataFrame(columns=["class_name", "split"])
official = read_json(paths["metrics"]) if paths["metrics"].is_file() else None
if version == "V2" and official is not None:
    paths["threshold"] = float(official.get("confidence_threshold", 0.25))

available_classes = [
    name
    for name in CLASS_ORDER
    if name in set(detections["class_name"]) or name in set(false_negatives.get("class_name", []))
]
available_splits = sorted(detections["split"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filters")
    selected_classes = st.multiselect("Class", available_classes, default=available_classes)
    default_split = ["test"] if "test" in available_splits else available_splits
    selected_splits = st.multiselect("Split", available_splits, default=default_split)
    status = st.selectbox("Prediction status", ["all", "tp", "fp"])
    operating = paths["threshold"] if paths["threshold"] is not None else 0.25
    min_confidence = st.slider("Minimum confidence", min_value=0.0, max_value=1.0, value=float(operating), step=0.05)
    st.caption("False negatives do not have a confidence score, so the confidence slider does not change the FN count.")

filtered = filter_detections(
    detections,
    classes=selected_classes,
    splits=selected_splits,
    status=status,
    min_confidence=min_confidence,
)
filtered_fn = filter_false_negatives(false_negatives, classes=selected_classes, splits=selected_splits)
kpis = summarize(filtered, filtered_fn)

st.subheader(f"{version} held-out detection metrics")
if official is None:
    st.caption("Held-out metrics for this version have not been written yet.")
else:
    st.caption("Ultralytics precision, recall, F1, and mAP on the full held-out test split. These cards do not follow the filters.")
    _official_cards(official)
    if version == "V2":
        matcher = official.get("matcher", {})
        if matcher:
            st.caption(
                f"Matcher at confidence {official['confidence_threshold']} and IoU 0.50: "
                f"FP {matcher['fp']}, FN {matcher['fn']}. This is not mAP."
            )

st.subheader("Filtered operating view")
cards = st.columns(4)
cards[0].metric("Total detections", kpis["total_detections"])
mean_confidence = kpis["mean_confidence"]
cards[1].metric("Mean confidence", "n/a" if mean_confidence is None else f"{mean_confidence:.3f}")
cards[2].metric("FP count", kpis["fp_count"])
cards[3].metric("FN count", kpis["fn_count"])
low_rate = kpis["low_confidence_rate"]
st.caption(
    "Low-confidence rate (confidence < 0.40): "
    + ("n/a" if low_rate is None else f"{low_rate:.1%}")
    + ". Precision and recall in the charts use the one-to-one matcher at IoU 0.50."
)

left, right = st.columns(2)
by_class = pd.Series(kpis["detections_by_class"], dtype="float64")
mix = pd.Series(kpis["class_mix"], dtype="float64")
avg_conf = pd.Series(
    {key: value for key, value in kpis["average_confidence_by_class"].items() if value is not None},
    dtype="float64",
)
precision = pd.Series(
    {key: value for key, value in kpis["precision_by_class"].items() if value is not None},
    dtype="float64",
)
recall = pd.Series(
    {key: value for key, value in kpis["recall_by_class"].items() if value is not None},
    dtype="float64",
)

with left:
    st.subheader("Detections by class")
    st.bar_chart(by_class)
    st.subheader("Precision by class")
    st.bar_chart(precision)
with right:
    st.subheader("Class mix")
    st.bar_chart(mix)
    st.subheader("Recall by class")
    st.bar_chart(recall)

st.subheader("Average confidence by class")
st.bar_chart(avg_conf)

confidence_left, iou_right = st.columns(2)
with confidence_left:
    st.subheader("Confidence distribution")
    if len(filtered):
        figure, axis = plt.subplots(figsize=(7, 3))
        axis.hist(filtered["confidence"], bins=20, color="#1C69D4")
        axis.set_xlabel("Confidence")
        axis.set_ylabel("Detections")
        st.pyplot(figure)
        plt.close(figure)
    else:
        st.caption("No detections in the current filter.")
with iou_right:
    st.subheader("IoU distribution")
    if kpis["iou_values"]:
        figure, axis = plt.subplots(figsize=(7, 3))
        axis.hist(kpis["iou_values"], bins=20, color="#1A1A1A")
        axis.set_xlabel("IoU of matched true positives")
        axis.set_ylabel("Matches")
        st.pyplot(figure)
        plt.close(figure)
    else:
        st.caption("No matched true positives in the current filter.")

st.subheader("Error gallery")
st.caption("Green is ground truth, blue is a true positive, red is a false positive, orange is a false negative.")
error_images = sorted(paths["errors"].glob("*.jpg")) if paths["errors"].is_dir() else []
if not error_images:
    st.caption("No error images have been saved for this version yet.")
else:
    chosen = st.selectbox("Example", [path.name for path in error_images])
    st.image(str(paths["errors"] / chosen), width="stretch")

st.subheader("V1 → V2")
comparison_path = ROOT / "outputs" / "v2" / "metrics" / "v1_vs_v2.csv"
if not comparison_path.is_file():
    st.caption("The V1 to V2 comparison is written after the V2 held-out test. It is not estimated before that measurement.")
else:
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison, hide_index=True, width="stretch")
    sentences = []
    for row in comparison.itertuples(index=False):
        change = row.absolute_change
        if pd.isna(change):
            continue
        sentences.append(f"{row.metric} is {_direction(float(change))} by {float(change):+.4f}")
    if sentences:
        st.caption(
            "Signed change is V2 minus V1. The test sets differ in size, so raw false-positive and false-negative counts are not a like-for-like rate. "
            "mAP50-95 is the detection metric that does not depend on the operating threshold. "
            + ". ".join(sentences)
            + "."
        )
