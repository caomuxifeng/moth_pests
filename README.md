# Moth Pest Recognition

This directory contains the code prepared for the public repository. It provides training and evaluation scripts for moth pest image classification and object detection, with experiments organized by data domain: `specimen`, `ecological`, and `mixed`.

## Dataset

The dataset is hosted on Figshare:

- DOI: [10.6084/m9.figshare.32576220](https://doi.org/10.6084/m9.figshare.32576220)
- URL: <https://figshare.com/s/e79722e7fcc2fc4d5980>

The scripts assume the downloaded data is arranged under `data/classification` and `data/detection` by default. You can use another location with `--data_root`.

## Files

```text
.
|-- train_classification.py      # Train a classification model
|-- evaluate_classification.py   # Evaluate a saved classification checkpoint
|-- train_detect.py              # Train an object detection model
|-- evaluate_detect.py           # Evaluate a saved detection checkpoint
|-- detect_trainers.py           # Custom Ultralytics trainer classes used by train_detect.py
|-- requirements.txt             # Python dependencies
`-- README.md
```

## Installation

```bash
pip install -r requirements.txt
```

The code uses PyTorch/torchvision for classification and Ultralytics for detection.

## Data Layout

### Classification Data

Classification data is loaded with `torchvision.datasets.ImageFolder`. Each domain must contain `train`, `val`, and `test` folders. The class-folder names must match across splits and domains used in evaluation.

```text
data/classification/
|-- specimen/
|   |-- train/
|   |   |-- class_001/
|   |   `-- class_002/
|   |-- val/
|   `-- test/
|-- ecological/
|   |-- train/
|   |-- val/
|   `-- test/
`-- mixed/
    |-- train/
    |-- val/
    `-- test/
```

### Detection Data

Detection data is loaded through an Ultralytics `dataset.yaml` file in each domain directory. `classes.txt` is optional, but if it exists it must match the `names` field in `dataset.yaml`.

```text
data/detection/
|-- specimen/
|   |-- dataset.yaml
|   |-- classes.txt
|   |-- images/
|   |   |-- train/
|   |   |-- val/
|   |   `-- test/
|   `-- labels/
|       |-- train/
|       |-- val/
|       `-- test/
|-- ecological/
`-- mixed/
```

Example `dataset.yaml`:

```yaml
path: /path/to/data/detection/mixed
train: images/train
val: images/val
test: images/test
nc: 2
names:
  - class_001
  - class_002
```

YOLO label files use this normalized format:

```text
class_id x_center y_center width height
```

## Classification

`train_classification.py` trains one of two torchvision models:

- `--model_type resnet`: ResNet-50 with ImageNet pretrained weights
- `--model_type efficientnet`: EfficientNet-B0 with ImageNet pretrained weights

Training example:

```bash
python train_classification.py \
  --data_root data/classification \
  --source_domain mixed \
  --model_type resnet \
  --batch_size 128 \
  --num_epochs 25 \
  --lr 0.0005 \
  --weight_decay 0.01 \
  --num_workers 4 \
  --image_size 224 \
  --seed 42 \
  --gpu_ids 0,1 \
  --output_dir outputs/classification/mixed_resnet_seed42
```

Training outputs:

```text
output_dir/
|-- config.json
|-- classes.json
|-- epoch_metrics.csv
|-- training_summary.json
`-- checkpoints/
    |-- best.pth
    `-- last.pth
```

`evaluate_classification.py` evaluates a saved `.pth` checkpoint on a target domain test split.

Evaluation example:

```bash
python evaluate_classification.py \
  --data_root data/classification \
  --source_domain mixed \
  --target_domain ecological \
  --model_type resnet \
  --checkpoint_path outputs/classification/mixed_resnet_seed42/checkpoints/best.pth \
  --batch_size 128 \
  --num_workers 4 \
  --image_size 224 \
  --seed 42 \
  --output_dir outputs/classification_eval/mixed_to_ecological_resnet_seed42
```

Evaluation outputs:

```text
output_dir/
|-- eval_config.json
|-- test_summary.csv
|-- test_summary.json
|-- per_class_metrics.csv
`-- predictions.csv
```

Classification metrics include loss, accuracy, macro/weighted precision, macro/weighted recall, and macro/weighted F1. `predictions.csv` stores the image path, true label, predicted label, and top-1 confidence.

## Object Detection

`train_detect.py` trains one of two Ultralytics detection models:

- `--model_type yolov11`: loads `yolo11n.pt`
- `--model_type rtdetr`: loads `rtdetr-l.pt`

Training example:

```bash
python train_detect.py \
  --data_root data/detection \
  --source_domain mixed \
  --model_type yolov11 \
  --batch_size 64 \
  --num_epochs 25 \
  --lr 0.001 \
  --weight_decay 0.0005 \
  --optimizer AdamW \
  --momentum 0.9 \
  --num_workers 8 \
  --image_size 640 \
  --seed 42 \
  --gpu_ids 0,1 \
  --cache_mode none \
  --output_dir outputs/detection/mixed_yolov11_seed42
```

`train_detect.py` uses the trainer classes in `detect_trainers.py` to keep the requested optimizer hyperparameters, disable dataloader `pin_memory`, and copy `checkpoints/best.pt` from the epoch with the best validation mAP50-95.

Training outputs produced by this code:

```text
output_dir/
|-- config.json
|-- classes.json
|-- epoch_metrics.csv
|-- training_summary.json
|-- checkpoints/
|   |-- best.pt
|   `-- last.pt
`-- train/
    |-- results.csv
    `-- weights/
```

`evaluate_detect.py` evaluates a saved `.pt` checkpoint on a target domain `val` or `test` split.

Evaluation example:

```bash
python evaluate_detect.py \
  --data_root data/detection \
  --source_domain mixed \
  --target_domain ecological \
  --model_type yolov11 \
  --checkpoint_path outputs/detection/mixed_yolov11_seed42/checkpoints/best.pt \
  --eval_split test \
  --batch_size 64 \
  --num_workers 8 \
  --image_size 640 \
  --seed 42 \
  --match_iou_threshold 0.5 \
  --eval_conf_threshold 0.001 \
  --pred_conf_threshold 0.25 \
  --output_dir outputs/detection_eval/mixed_to_ecological_yolov11_seed42
```

Evaluation outputs produced by this code:

```text
output_dir/
|-- eval_config.json
|-- test_summary.csv
|-- test_summary.json
|-- per_class_metrics.csv
|-- predictions.csv
|-- confusion_matrix_data.json
`-- val/
```

Detection metrics include precision, recall, F1, mAP50, and mAP50-95. `predictions.csv` stores matched detections, missed ground-truth boxes, and false positives with class labels, scores, IoU values, and box coordinates.

## Common Arguments

Valid domains are `specimen`, `ecological`, and `mixed`.

| Argument | Used by | Description |
| --- | --- | --- |
| `--data_root` | all scripts | Root directory for classification or detection data |
| `--source_domain` | all scripts | Domain used to train the checkpoint |
| `--target_domain` | evaluation scripts | Domain used for evaluation |
| `--model_type` | all scripts | Classification: `resnet`, `efficientnet`; detection: `yolov11`, `rtdetr` |
| `--checkpoint_path` | evaluation scripts | Path to a saved `.pth` or `.pt` checkpoint |
| `--output_dir` | all scripts | Directory where the script writes CSV, JSON, and checkpoint files |
| `--gpu_ids` | all scripts | Comma-separated CUDA device IDs; detection scripts set this before importing Ultralytics |
| `--seed` | all scripts | Random seed |

## Reproducibility Notes

The scripts set Python, NumPy, and PyTorch seeds and request deterministic PyTorch behavior. Results can still vary across hardware, CUDA/cuDNN versions, PyTorch versions, and Ultralytics versions.

## Citation

If you use this code or dataset, please cite the Figshare dataset:

```text
Moth pest recognition dataset. Figshare.
DOI: 10.6084/m9.figshare.32576220
URL: https://figshare.com/s/e79722e7fcc2fc4d5980
```
