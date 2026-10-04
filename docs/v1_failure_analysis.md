# V1 failure analysis

This report uses the archived V1 test export and the V1 training counts. It does not change the V1 checkpoint and it is not a V2 test result.

Matcher rule: one prediction to one ground-truth box, same class, IoU at least 0.50, confidence at least 0.25. These precision and recall figures are not mAP.

## Per class

| Class | GT | Predictions | TP | FP | FN | Precision | Recall | Mean confidence | Median confidence | Mean matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pallet | 716 | 701 | 478 | 223 | 235 | 0.6819 | 0.6704 | 0.5936 | 0.5805 | 0.8316 |
| stillage | 1,452 | 1,398 | 1,110 | 288 | 338 | 0.7940 | 0.7666 | 0.7300 | 0.8528 | 0.8919 |
| forklift | 108 | 112 | 83 | 29 | 25 | 0.7411 | 0.7685 | 0.7711 | 0.8909 | 0.9125 |
| dolly | 541 | 518 | 343 | 175 | 196 | 0.6622 | 0.6364 | 0.6240 | 0.6434 | 0.8379 |

## What the counts show

The largest false-positive count is stillage (288). The largest false-negative count is stillage (338). Those are raw counts, so the most common class can lead both lists even when its rate is not the worst.

V1 training boxes: pallet 3,686, stillage 7,071, forklift 475, dolly 2,748. Shares of training boxes: pallet 0.264, stillage 0.506, forklift 0.034, dolly 0.197. The fewest training boxes belong to forklift.

Small boxes are the ground-truth boxes at or below the 25th percentile of relative area (0.0081). Their recall is 0.3489 (246 / 705). Larger boxes recall 0.8371 (1,768 / 2,112).

The same comparison at the median relative area (0.0229) is 0.5458 below the median and 0.8842 above it.

A crowded test image has at least 13.5 target boxes, the median count. Crowded images: recall 0.6970 on 2,079 boxes across 90 images. Other images: recall 0.7656 on 738 boxes across 90 images.

Predictions below confidence 0.40: 579 (0.212 of test predictions).

- pallet: low-confidence rate 0.275 (193 predictions), of which 0.658 are false positives.
- stillage: low-confidence rate 0.167 (233 predictions), of which 0.609 are false positives.
- forklift: low-confidence rate 0.116 (13 predictions), of which 0.692 are false positives.
- dolly: low-confidence rate 0.270 (140 predictions), of which 0.650 are false positives.

Ground-truth ids absent from both the true-positive and false-negative tables: 9.

## Augmentation chosen for experiment C

Experiment C starts from the same pretrained YOLO11n weights as A and B. Only the augmentation settings below change relative to B. They were fixed from this V1 measurement before V2 training, not from a V2 test score.

- scale = 0.45
- mosaic = 0.3
- translate = 0.1
- horizontal flip = 0.5
- HSV value gain = 0.3
- rotation, shear, perspective, vertical flip, mixup, copy-paste, and random erasing stay off

Triggers:

- small_boxes_recall_gap_at_least_0.05: True
- crowded_images_recall_gap_at_least_0.05: True
- classes_with_elevated_low_confidence: []

Small boxes and crowded images both recall worse. Mosaic stays at 0.3 so training does not add more crowding. Scale is 0.45 so object size still varies.

Settings that were rejected:
- vertical flip, because an upside-down warehouse floor is not a useful industrial view
- rotation and shear, because they bend upright loads and aisles
- mixup and copy-paste, because they composite objects into scenes they were not rendered in
- random erasing, because it deletes parts of loads without a physical cause
