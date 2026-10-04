# Model card

BMW SORDI Smart Logistics Vision, YOLO11n

Independent portfolio model. Not a BMW Group model and not cleared for production use.

## Model name

`yolo11n` fine-tune for four SORDI logistics classes. The selected checkpoint name is recorded in `outputs/metrics/model_selection.json` after validation.

## Base model

Ultralytics YOLO11 Nano, pretrained weights `yolo11n.pt`.

## Intended use

Prototype visual resource monitoring on synthetic industrial logistics scenes. The intended outputs are bounding boxes, confidence scores, matcher labels, and class counts for:

- pallet
- stillage (pallet cage)
- forklift
- dolly

`stillage` is the class name in the model, the labels, and the exports.

## Out-of-scope use

- Live warehouse control or inventory
- Safety, collision, or pedestrian decisions
- Any claim about a real BMW plant
- Classes outside the four names above
- Deployment on factory hardware

## Dataset

Public Kaggle dataset `sordi-ai/industrial-scene-01-dataset`, DR-1 only. Images are kept only when a target class is present, then capped at 1,200 with seed 42. The split is 70/15/15 by image id. Test images are not used to train or to choose the checkpoint.

## Preprocessing

JSON boxes (`ObjectClassName`, `Left`, `Top`, `Right`, `Bottom`) are clipped to the frame and stored as normalized YOLO coordinates. Training uses Ultralytics' default letterbox resize to 640.

## Training configuration

- Baseline: 30 epochs, batch 8 (batch 4 if CUDA OOM), seed 42, cosine schedule off
- One tuning cycle: 15 epochs from the best baseline weights, cosine learning-rate schedule on
- Optimizer and learning-rate values are copied from the Ultralytics run into `outputs/metrics/run_meta_*.json`
- Hardware: NVIDIA RTX 3050 Laptop GPU, 4 GB
- Framework: Python 3.11, PyTorch CUDA 12.4 wheels, Ultralytics

## Evaluation methodology

Official precision, recall, mAP@0.50, and mAP@0.50:0.95 come from Ultralytics validation on the test split. That validation is run twice on the same checkpoint. The runs are required to match.

A separate one-to-one matcher labels each exported prediction. A prediction is a true positive only when it has the same class as a ground-truth box and IoU is at least 0.50. Each ground-truth box can match at most one prediction. Unmatched predictions are false positives. Unmatched ground truth is a false negative. V1 exported at confidence 0.25. V2 exported at confidence 0.40, chosen on validation matcher F1. Matcher precision and recall are not mAP.

## Metrics

<!-- METRICS:START -->
### V1 (1,200 images, seed 42)

Validation mAP@0.50:0.95 selected the tuned checkpoint (0.5647) over the baseline (0.5599). The test split was not used for selection.

Held-out test, YOLO11n tuned, both Ultralytics repeats identical: precision 0.8208, recall 0.6610, mAP@0.50 0.7473, mAP@0.50:0.95 0.5623.

Matcher at confidence 0.25 and IoU 0.50: precision 0.7380, recall 0.7172, 2,014 true positives, 715 false positives, 794 false negatives.

### V2 (4,000 images, seed 43)

Validation mAP@0.50:0.95 selected experiment B (0.6448) over A (0.6321) and C (0.6386). The V2 test split was not used for selection. Operating confidence 0.40 was chosen on validation matcher F1 and then frozen.

Held-out V2 test, YOLO11n experiment B, both Ultralytics repeats identical: precision 0.8863, recall 0.7043, F1 0.7849, mAP@0.50 0.7899, mAP@0.50:0.95 0.6213.

Matcher at confidence 0.40 and IoU 0.50: precision 0.8967, recall 0.7002, F1 0.7864, 6,690 true positives, 771 false positives, 2,864 false negatives.

V1 and V2 use different held-out splits. Exact rows are in `outputs/v1/metrics/` and `outputs/v2/metrics/`.
<!-- METRICS:END -->

## Class-level performance

<!-- CLASS_METRICS:START -->
### V1 held-out Ultralytics

| Class | Precision | Recall | AP@0.50 | AP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: |
| pallet | 0.8326 | 0.6111 | 0.6918 | 0.4621 |
| stillage | 0.8719 | 0.7135 | 0.8125 | 0.6547 |
| forklift | 0.8040 | 0.7595 | 0.8181 | 0.6752 |
| dolly | 0.7748 | 0.5601 | 0.6669 | 0.4573 |

### V2 held-out Ultralytics

| Class | Objects | Precision | Recall | AP@0.50 | AP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pallet | 2,470 | 0.8805 | 0.6445 | 0.7310 | 0.5287 |
| stillage | 4,987 | 0.9209 | 0.7359 | 0.8486 | 0.7003 |
| forklift | 347 | 0.8851 | 0.8184 | 0.8644 | 0.7481 |
| dolly | 1,787 | 0.8586 | 0.6184 | 0.7155 | 0.5081 |
<!-- CLASS_METRICS:END -->

## Known limitations

Synthetic images, capped subsets (1,200 then 4,000), four classes, and YOLO11n only. Domain shift to real factories is unmeasured. Small boxes still underperform on the V2 test set. Class confusion between similarly shaped load carriers is possible and is reported only when an error image shows it.

## Ethical and operational considerations

The pictures are synthetic, so this repository does not contain personal data. The counts must not be used to make safety decisions or to represent stock in a real building. Publishing the scores next to their limits is part of the project, not an afterthought.

## Reproducibility

Seed 42. Configs live in `configs/`. Run metadata includes the torch version, Ultralytics version, device name, and git commit when one exists. Weights stay in `C:\Users\tanam\sordi-cache\runs` and are not committed.
