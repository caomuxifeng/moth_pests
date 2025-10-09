# Moth Pest Recognition: Deep Learning for Image Classification and Object Detection

A comprehensive deep learning framework for moth pest identification using state-of-the-art image classification and object detection models. This repository provides ready-to-use training scripts for both classification (EfficientNet, ResNet) and detection (YOLOv11, RT-DETR) tasks.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **Multiple Model Architectures**: Support for EfficientNet-B0, ResNet-50, YOLOv11, and RT-DETR
- **Image Classification**: Fine-tune pretrained models for moth species identification
- **Object Detection**: Train state-of-the-art detectors for moth localization
- **Comprehensive Metrics**: Detailed evaluation with accuracy, precision, recall, F1-score, and mAP
- **Visualization**: Automatic generation of training curves, confusion matrices, and PR curves
- **Production Ready**: Clean, well-documented code suitable for academic research and deployment

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Image Classification](#image-classification)
  - [Object Detection](#object-detection)
- [Dataset Structure](#dataset-structure)
- [Model Architectures](#model-architectures)
- [Training Parameters](#training-parameters)
- [Output Files](#output-files)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory for detection models

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/moth-classification.git
cd moth-classification

# Install required packages
pip install -r requirements.txt
```

### Quick Install (Alternative)

```bash
# PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other dependencies
pip install ultralytics matplotlib seaborn scikit-learn pillow tqdm pyyaml pandas
```

## Quick Start

### Image Classification

Train a classification model to identify moth species:

    ```bash
# EfficientNet-B0 (lightweight, fast)
python classification.py --model_type efficientnet --data_dir your_dataset --num_epochs 25 --batch_size 128
    
# ResNet-50 (robust, production-ready)
python classification.py --model_type resnet --data_dir your_dataset --num_epochs 25 --batch_size 128
    ```

### Object Detection

Train a detection model to localize and classify moths:

    ```bash
# YOLOv11 (real-time detection)
python detect.py --model_type yolov11 --data_dir yolo_dataset --batch_size 64 --num_epochs 50

# RT-DETR (high accuracy)
python detect.py --model_type rtdetr --data_dir yolo_dataset --batch_size 32 --num_epochs 100
    ```

## Dataset Structure

### Classification Dataset Format

```
your_dataset/
├── train/
│   ├── species_A/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── species_B/
│       ├── image3.jpg
│       └── image4.jpg
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

### Detection Dataset Format (YOLO)

```
yolo_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   └── image2.txt
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── classes.txt
```

**YOLO Label Format** (one file per image):
```
class_id x_center y_center width height
```
All values normalized to [0, 1]. Example:
```
0 0.500 0.500 0.200 0.200
1 0.300 0.300 0.100 0.100
```

## Model Architectures

### Classification Models

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| **EfficientNet-B0** | 5.3M | Fast | High | Rapid prototyping, mobile deployment |
| **ResNet-50** | 25.6M | Medium | Very High | Production systems, high accuracy requirements |

### Detection Models

| Model | Architecture | Speed | mAP | Use Case |
|-------|-------------|-------|-----|----------|
| **YOLOv11** | Single-stage | Very Fast | High | Real-time applications, edge devices |
| **RT-DETR** | Transformer | Fast | Very High | Research, high-precision requirements |

## Training Parameters

### Classification (`classification.py`)

```bash
python classification.py \
  --model_type efficientnet \      # Model: efficientnet or resnet
  --data_dir dataset_path \         # Path to dataset
  --batch_size 128 \                # Batch size (adjust for GPU memory)
  --num_epochs 25 \                 # Training epochs
  --lr 0.0005 \                     # Learning rate
  --device cuda:0 \                 # GPU device
  --output_dir outputs              # Output directory
```

### Object Detection (`detect.py`)

```bash
python detect.py \
  --model_type yolov11 \            # Model: yolov11 or rtdetr
  --data_dir yolo_dataset \         # Path to YOLO format dataset
  --batch_size 64 \                 # Batch size
  --num_epochs 50 \                 # Training epochs
  --lr 0.001 \                      # Learning rate
  --img_size 640 \                  # Input image size
  --device cuda:0 \                 # GPU device
  --eval_split test \               # Evaluation split: val or test
  --output_dir outputs_det          # Output directory
```


## Output Files

### Classification Outputs

After training, the following files are generated in `outputs/[model_type]_[timestamp]/`:

```
outputs/efficientnet_20251009_120000/
├── best_model.pth                    # Best model weights
├── classes.json                      # Class names
├── history.json                      # Training history
├── training_history.png/svg          # Loss and F1-score curves
├── confusion_matrix.png/svg          # Confusion matrix
├── test_results.json                 # Detailed metrics
└── classification_report.txt         # Human-readable report
```

### Detection Outputs

```
outputs_det/yolov11_20251009_120000/
├── classes.json                      # Class information
├── data.yaml                         # Dataset configuration
├── best_model.pth                    # Path to best weights
├── test_results_det.json             # Evaluation metrics
├── confusion_matrix_det.png/svg      # Custom confusion matrix
└── train/                            # Ultralytics training outputs
    ├── weights/
    │   ├── best.pt                   # Best model
    │   └── last.pt                   # Last epoch
    ├── results.png                   # Training curves
    ├── results.csv                   # Metrics CSV
    ├── confusion_matrix.png          # Confusion matrix
    └── BoxPR_curve.png               # Precision-Recall curve
```

## Evaluation Metrics

### Classification Metrics

The framework provides comprehensive per-class and overall metrics:

- **Accuracy**: Overall and per-class accuracy
- **Precision**: Weighted and per-class precision
- **Recall**: Weighted and per-class recall
- **F1-Score**: Weighted and per-class F1-score
- **Confusion Matrix**: Visual representation of predictions

**Example Output:**
```
Class                   Accuracy   Precision  Recall     F1 Score   Samples   
--------------------------------------------------------------------------------
Agrotis ipsilon        0.9870     0.9828     0.9870     0.9849     231       
Chilo suppressalis     0.9856     0.9762     0.9856     0.9809     208       
...
```

### Detection Metrics

- **mAP50**: Mean Average Precision at IoU=0.50
- **mAP50-95**: Mean Average Precision at IoU=0.50:0.95
- **Precision-Recall Curves**: Visual performance analysis
- **Confusion Matrix**: Detection performance per class (includes background FN/FP)

**Detection Confusion Matrix Note:**
The confusion matrix has one extra row and column for background:
- **Last column (background-FN)**: False Negatives (missed detections)
- **Last row (background-FP)**: False Positives (false alarms)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
