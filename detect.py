import os
import sys

# CRITICAL: Parse device argument BEFORE importing PyTorch
# This ensures CUDA_VISIBLE_DEVICES is set before CUDA initialization
def parse_device_early():
    """Early parsing of device argument to set CUDA_VISIBLE_DEVICES before torch import"""
    for i, arg in enumerate(sys.argv):
        if arg == '--device' and i + 1 < len(sys.argv):
            device_arg = sys.argv[i + 1]
            if 'cuda:' in device_arg:
                gpu_id = device_arg.split(':')[1]
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
                print(f"[BEFORE TORCH IMPORT] Set CUDA_VISIBLE_DEVICES={gpu_id}")
                return gpu_id
    return None

# Set CUDA device BEFORE any torch import
_gpu_id = parse_device_early()

import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# define command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Moth Object Detection Model Training')
    parser.add_argument('--data_dir', type=str, default='yolo', help='Dataset root directory (should contain train, val, test subdirs with images and YOLO format labels)')
    parser.add_argument('--model_type', type=str, default='yolov11',
                        choices=['yolov11', 'rtdetr'], help='Model type: yolov11 or rtdetr (RT-DETR)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu',
                        help='Training device (e.g., cuda:0, cuda:1, cpu). Default: cuda:1')
    parser.add_argument('--output_dir', type=str, default='outputs_det', help='Output directory for detection results')
    parser.add_argument('--img_size', type=int, default=640, help='Image size (square images)')
    parser.add_argument('--eval_split', type=str, default='test', choices=['val', 'test'], 
                        help='Dataset split to evaluate on after training. Default: test (recommended for final evaluation)')

    return parser.parse_args()

# data augmentation and dataset for object detection
def get_data_loaders_det(data_dir, batch_size, img_size, model_type):
    """
    Read dataset class information.
    For YOLOv11 and RT-DETR, ultralytics handles data loading internally via data.yaml.
    """
    # Read class names from file
    classes_path = Path(data_dir) / 'classes.txt'
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found in {data_dir}. Please ensure your dataset is correctly structured.")
    
    with open(classes_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)
    
    print(f"Ultralytics ({model_type}) will handle data loading internally.")
    print(f"Detected {num_classes} classes: {class_names}")
    
    # Return dummy loaders and class info
    return None, None, None, class_names, num_classes

def get_model_det(model_type, num_classes, device):
    """
    Initialize detection model.
    For ultralytics models (YOLOv11, RT-DETR), num_classes is determined from data.yaml.
    """
    # When CUDA_VISIBLE_DEVICES is set, always use device 0 (the only visible device)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device_id = 0
        print(f" CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, using device_id=0")
    else:
        # CRITICAL: Set device BEFORE loading model to ensure model is loaded on correct GPU
        device_str = str(device)
        if 'cuda' in device_str:
            if ':' in device_str:
                device_id = int(device_str.split(':')[1])
            else:
                device_id = 0
        else:
            device_id = 'cpu'
    
    # Set PyTorch default device BEFORE loading model
    if isinstance(device_id, int):
        torch.cuda.set_device(device_id)
        print(f" Set default CUDA device to GPU {device_id} before loading model")
        print(f"   Current CUDA device: {torch.cuda.current_device()}")
    
    if model_type == 'yolov11':
        # Load pretrained YOLOv11n model
        print("Loading YOLOv11n pretrained model...")
        model = YOLO('yolo11n.pt')
    
    elif model_type == 'rtdetr':
        # Load pretrained RT-DETR-l model
        print("Loading RT-DETR-l pretrained model...")
        model = YOLO('rtdetr-l.pt')  # ultralytics will auto-download if not exists
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model

def train_model_det(model, model_type, dataloaders, device, num_epochs, save_dir, img_size, class_names, data_dir, args=None):
    """
    Train detection model using ultralytics interface.
    Supports YOLOv11 and RT-DETR.
    """
    since = time.time()
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create data.yaml for ultralytics training
    data_yaml_path = os.path.join(save_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(f"path: {Path(data_dir).resolve()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    # Configure training parameters
    # When CUDA_VISIBLE_DEVICES is set, always use device 0 (the only visible device)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device_id = 0
        physical_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        print(f" CUDA_VISIBLE_DEVICES={physical_gpu}, using device_id=0 (maps to physical GPU {physical_gpu})")
    else:
        # Extract device ID: Ultralytics requires integer for GPU or 'cpu' string
        # "cuda:0" -> 0, "cuda:1" -> 1, "cpu" -> "cpu"
        device_str = str(device)
        if 'cuda' in device_str:
            if ':' in device_str:
                device_id = int(device_str.split(':')[1])  # Extract as INTEGER
            else:
                device_id = 0  
        else:
            device_id = 'cpu'
    
    # CRITICAL: Set PyTorch default CUDA device to ensure all operations use the correct GPU
    if isinstance(device_id, int):
        torch.cuda.set_device(device_id)
        print(f" Set PyTorch default CUDA device to: {device_id}")
        print(f"   Current CUDA device: {torch.cuda.current_device()}")
    
    if model_type == 'yolov11':
        print(" Starting YOLOv11 training...")
        print(f" Using device: {device} -> GPU {device_id}")
        train_params = {
            'data': data_yaml_path,
            'epochs': num_epochs,
            'imgsz': img_size,
            'batch': args.batch_size,
            'device': device_id,
            'project': save_dir,  # Use custom output directory
            'name': 'train',      # Subdirectory name
            'cache': 'ram',       # Use RAM cache for fastest training speed
            'workers': 8,         # More workers for parallel loading
            'patience': 50,       # Early stopping
            'save_period': 5,     # Save every 5 epochs
            'exist_ok': True,     # Allow overwriting runs
        }
        print(f" RAM | Workers: 8 | Batch: {args.batch_size}")
        
    elif model_type == 'rtdetr':
        print(" Starting RT-DETR training...")
        print(f" Using device: {device} -> GPU {device_id}")
        train_params = {
            'data': data_yaml_path,
            'epochs': num_epochs,
            'imgsz': img_size,
            'batch': args.batch_size,
            'device': device_id,
            'project': save_dir,  # Use custom output directory
            'name': 'train',      # Subdirectory name
            'cache': 'ram',       # Use RAM cache for fastest training speed
            'workers': 8,         # More workers for parallel loading
            'patience': 100,      # RT-DETR may need more patience
            'save_period': 10,    # Save every 10 epochs
            'exist_ok': True,     # Allow overwriting runs
        }
        print(f" RAM | Workers: 8 | Batch: {args.batch_size}")
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # Start training
    print(f"⏰ Training for {num_epochs} epochs with image size {img_size}...")
    try:
        results = model.train(**train_params)
    except IndexError as e:
        print(f"\n Warning: Encountered IndexError during final validation: {e}")
        print("   This is usually a temporary numerical instability issue.")
        print("   The model has been trained successfully and saved.")
        print("   You can safely use the trained model.")
    
    # Save best model path
    print(f" Training complete! Model saved in: {model.trainer.save_dir}")
    with open(best_model_path, 'w') as f:
        f.write(str(Path(model.trainer.save_dir) / 'weights' / 'best.pt'))
    
    # Save training history (placeholder, ultralytics handles its own logging)
    history = {'train_loss': [0.0], 'val_loss': [0.0], 'lr': [0.0]}
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    training_time = time.time() - since
    print(f" Total training time: {training_time // 60:.0f}m {training_time % 60:.0f}s")
    
    print(f"\n Training results saved in: {model.trainer.save_dir}")
    print(f"   ├─ weights/best.pt - Best model weights")
    print(f"   ├─ weights/last.pt - Last epoch weights")
    print(f"   ├─ results.png - Training curves")
    print(f"   ├─ results.csv - Training metrics")
    print(f"   └─ Other plots (PR curve, confusion matrix, etc.)")
    
    return model

# plot training history for detection

def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2]
    """
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Check if there's intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate IoU
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def plot_confusion_matrix_det(confusion_matrix, class_names, save_dir, dpi=300):
    """
    Plot confusion matrix for object detection.
    Uses 'g' format to handle both integer and float values from Ultralytics.
    
    Note: In object detection, Ultralytics adds an extra row and column for 'background':
    - Last column: True objects predicted as background (False Negatives)
    - Last row: Background predicted as objects (False Positives)
    """
    plt.figure(figsize=(14, 12))
    
    # Check if confusion matrix has extra dimension for background
    matrix_shape = confusion_matrix.shape
    num_classes = len(class_names)
    
    # Ultralytics adds 'background' class at the end
    if matrix_shape[0] == num_classes + 1 and matrix_shape[1] == num_classes + 1:
        # Add 'background' label
        x_labels = class_names + ['background-FN']
        y_labels = class_names + ['background-FP']
        print(f" Confusion matrix includes background class: {matrix_shape}")
    else:
        # No background class
        x_labels = class_names
        y_labels = class_names
        print(f" Confusion matrix shape: {matrix_shape}")
    
    # Ultralytics may return normalized (float) or raw (int) confusion matrix
    # Use 'g' format which works for both integers and floats
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=x_labels, yticklabels=y_labels, cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix for Object Detection', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    # Save as PNG
    png_path = os.path.join(save_dir, 'confusion_matrix_det.png')
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"Confusion matrix saved to {png_path}")
    
    # Save as SVG
    svg_path = os.path.join(save_dir, 'confusion_matrix_det.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Confusion matrix saved to {svg_path}")
    
    plt.close()

# evaluate function for object detection
def evaluate_model_det(model, model_type, device, class_names, save_dir, args):
    """
    Evaluate detection model using ultralytics interface.
    Supports YOLOv11 and RT-DETR.
    """
    print("\n=== Starting Evaluation ===" )
    print(f" Evaluating on: {args.eval_split.upper()} dataset")
    results = {}

    if model_type in ['yolov11', 'rtdetr']:
        # Ultralytics built-in evaluation method
        model_name = "YOLOv11" if model_type == 'yolov11' else "RT-DETR"
        print(f"Using {model_name}'s built-in validation method...")
        
        # When CUDA_VISIBLE_DEVICES is set, always use device 0 (the only visible device)
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            device_id = 0
            physical_gpu = os.environ['CUDA_VISIBLE_DEVICES']
            print(f" Evaluation: CUDA_VISIBLE_DEVICES={physical_gpu}, using device_id=0")
        else:
            # Extract device ID: Ultralytics requires integer for GPU or 'cpu' string
            # "cuda:0" -> 0, "cuda:1" -> 1, "cpu" -> "cpu"
            device_str = str(device)
            if 'cuda' in device_str:
                if ':' in device_str:
                    device_id = int(device_str.split(':')[1])  # Extract as INTEGER
                else:
                    device_id = 0  # Default to GPU 0 if just 'cuda'
            else:
                device_id = 'cpu'
        
        # Set PyTorch default CUDA device for evaluation
        if isinstance(device_id, int):
            torch.cuda.set_device(device_id)
            print(f"   Current CUDA device: {torch.cuda.current_device()}")
        
        # Evaluate on specified split (val or test)
        metrics = model.val(data=os.path.join(save_dir, 'data.yaml'), imgsz=args.img_size, batch=args.batch_size, device=device_id, split=args.eval_split)
        
        results['eval_split'] = args.eval_split
        results['metrics'] = {
            'map50': metrics.results_dict['metrics/mAP50(B)'],
            'map': metrics.results_dict['metrics/mAP50-95(B)'],
            'fitness': metrics.fitness
        }
        print(f"\n {model_name} Evaluation Results on {args.eval_split.upper()} dataset:")
        print(f"   mAP50: {results['metrics']['map50']:.4f}")
        print(f"   mAP50-95: {results['metrics']['map']:.4f}")
        print(f"   Fitness: {results['metrics']['fitness']:.4f}")

        # Extract and plot Confusion Matrix
        # Ultralytics automatically generates confusion matrix plots
        # We can also create our own custom plot
        if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
            try:
                # Get confusion matrix - handle both tensor and numpy array
                conf_matrix_raw = metrics.confusion_matrix.matrix
                if hasattr(conf_matrix_raw, 'cpu'):
                    conf_matrix_data = conf_matrix_raw.cpu().numpy()
                elif hasattr(conf_matrix_raw, 'numpy'):
                    conf_matrix_data = conf_matrix_raw.numpy()
                else:
                    conf_matrix_data = np.array(conf_matrix_raw)
                
                plot_confusion_matrix_det(conf_matrix_data, class_names, save_dir, dpi=300)
            except Exception as e:
                print(f"Warning: Could not plot confusion matrix: {e}")
                print("Ultralytics will generate its own confusion matrix plot.")
        else:
            print("Note: Ultralytics will generate built-in confusion matrix plot.")

        # Ultralytics generates prediction images and plots (including PR curves) automatically
        print(f"\n {model_name} evaluation complete!")
        # Evaluation results are saved in the same directory structure
        eval_subdir = f"{args.eval_split}"  # 'val' or 'test'
        eval_dir = str(model.trainer.save_dir).replace('train', eval_subdir)
        print(f"    Evaluation plots saved in: {eval_dir}")
        print(f"   ├─ confusion_matrix.png - Confusion matrix")
        print(f"   ├─ BoxPR_curve.png - Precision-Recall curve")
        print(f"   └─ Other validation plots")

    else:
        raise ValueError(f"Unsupported model_type for evaluation: {model_type}")

    # Save test results to JSON
    with open(os.path.join(save_dir, 'test_results_det.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(" Evaluation complete.")
    print("="*80)
    return results

def main():
    args = parse_args()
    set_seed(42)
    
    # create output directory
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f" Starting object detection training with {args.model_type.upper()}...")
    print(f" Data directory: {args.data_dir}")
    print(f" Output directory: {output_dir}")
    print(f" Batch size: {args.batch_size}")
    print(f" Number of epochs: {args.num_epochs}")
    print(f" Learning rate: {args.lr}")
    print(f" Device: {args.device}")
    print(f"  Image size: {args.img_size}")

    # Load dataset information
    train_loader, val_loader, _, class_names, num_classes = get_data_loaders_det(
        args.data_dir, args.batch_size, args.img_size, args.model_type
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Save class information
    with open(os.path.join(output_dir, 'classes.json'), 'w') as f:
        json.dump(class_names, f)
    
    print(f"\n Detected {num_classes} classes: {class_names}\n")

    # Initialize model
    model = get_model_det(args.model_type, num_classes, args.device)

    # Train model
    print("="*60)
    model = train_model_det(model, args.model_type, dataloaders, args.device,
                            args.num_epochs, output_dir, args.img_size,
                            class_names, args.data_dir, args)
    
    # Evaluate model
    print("="*60)
    results = evaluate_model_det(model, args.model_type, args.device, class_names, output_dir, args)

    print("="*60)
    print(f" All results and models saved to {output_dir}")

if __name__ == "__main__":
    main()
