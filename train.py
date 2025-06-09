import os
import json
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import seaborn as sns

# set seed to make results reproducible
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# define command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Moth Image Classification Model Fine-tuning')
    parser.add_argument('--data_dir', type=str, default='moth', help='Dataset root directory')
    parser.add_argument('--model_type', type=str, default='efficientnet', 
                        choices=['efficientnet', 'resnet', 'swin'], help='Model type')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        help='Training device')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    return parser.parse_args()

# data augmentation
def get_data_loaders(data_dir, batch_size):
    # train data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # val and test data transformation
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # load dataset
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)
    
    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # get class names
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    return train_loader, val_loader, test_loader, class_names, num_classes

# get pretrained model and modify the last classification layer
def get_model(model_type, num_classes, device):
    if model_type == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_type == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_type == 'swin':
        model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model.to(device)

# train function
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, save_dir):
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    # create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # record training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # each epoch has train and val phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to train mode
            else:
                model.eval()   # set model to eval mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero gradients
                optimizer.zero_grad()
                
                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # backward pass + optimization (only in train phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                # update learning rate scheduler
                if scheduler is not None:
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    history['lr'].append(current_lr)
                    print(f'Current learning rate: {current_lr:.6f}')
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
            # if best val accuracy, save model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # save best model and delete previous best model
                torch.save(model.state_dict(), best_model_path)
                print(f'New best model saved with accuracy: {best_acc:.4f}')
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val accuracy: {best_acc:4f}')
    
    # save training history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    # plot training process
    plot_training_history(history, save_dir)
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# plot training history
def plot_training_history(history, save_dir):
    # set plot style
    plt.style.use('seaborn-v0_8-dark')
    
    # set Times New Roman font
    import matplotlib.font_manager as fm
    
    # try to use Times New Roman font, if not exist, use default font
    try:
        times_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
        if not os.path.exists(times_font_path):
            times_font_path = fm.findfont('Times New Roman')
        font_prop = fm.FontProperties(fname=times_font_path)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
    except:
        # if font setting failed, use default font
        font_prop = fm.FontProperties()
    
    # create square subplot - modify figsize to square ratio
    plt.figure(figsize=(12, 6))
    
    # plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontproperties=font_prop, fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontproperties=font_prop, fontsize=12)
    plt.xlabel('Epoch', fontproperties=font_prop, fontsize=12)
    plt.legend(prop=font_prop)
    plt.grid(True, alpha=0.3)
    
    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontproperties=font_prop, fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontproperties=font_prop, fontsize=12)
    plt.xlabel('Epoch', fontproperties=font_prop, fontsize=12)
    plt.legend(prop=font_prop)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # save PNG format (high DPI)
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    # save SVG format
    plt.savefig(os.path.join(save_dir, 'training_history.svg'), format='svg', bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader, criterion, device, class_names, save_dir):
    model.eval()
    
    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Test Accuracy: {test_acc:.4f}')
    
    # analyze test set class distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    
    print(f"\nTest set class distribution:")
    for i, (label_idx, count) in enumerate(zip(unique_labels, counts)):
        print(f"{class_names[label_idx]}: {count} samples")
    
    # calculate detailed classification metrics - use weighted average
    print("\n=== Detailed Classification Metrics ===")
    
    # overall metrics - mainly focus on weighted average
    overall_accuracy = accuracy_score(all_labels, all_preds)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    print(f"\nMain Evaluation Metrics (Weighted Average):")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Precision (Weighted): {precision_weighted:.4f}")
    print(f"Recall (Weighted): {recall_weighted:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    # detailed metrics for each class
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    print(f"\nDetailed Classification Metrics:")
    print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Samples':<10}")
    print("-" * 75)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
              f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}")
    
    # calculate precision for each class
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    
    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1
    
    # fix class distribution dictionary creation
    class_distribution = {}
    for label_idx, count in zip(unique_labels, counts):
        class_distribution[class_names[label_idx]] = int(count)
    
    # save test results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc.item(),
        'dataset_analysis': {
            'class_distribution': class_distribution
        },
        'overall_metrics': {
            'accuracy': overall_accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        },
        'class_metrics': {},
        'class_accuracy': {}
    }
    
    # save detailed metrics for each class
    for i, class_name in enumerate(class_names):
        results['class_metrics'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support_per_class[i])
        }
        
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            results['class_accuracy'][class_name] = float(accuracy)
    
    # generate confusion matrix and visualize
    plot_confusion_matrix(all_labels, all_preds, class_names, save_dir)
    
    # save detailed classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, 
                                 output_dict=True, zero_division=0)
    results['classification_report'] = report
    
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # save a readable text report
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Moth Image Classification Model Evaluation Report ===\n\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        
        f.write("Main Evaluation Metrics (Weighted Average):\n")
        f.write(f"Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Precision (Weighted): {precision_weighted:.4f}\n")
        f.write(f"Recall (Weighted): {recall_weighted:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n\n")
        
        f.write("Detailed Classification Metrics:\n")
        f.write(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Samples':<10}\n")
        f.write("-" * 75 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<25} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
                   f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}\n")
        
        f.write(f"\nDetailed Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    return results

# plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    # set Times New Roman font
    import matplotlib.font_manager as fm
    
    try:
        times_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
        times_italic_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf'
        
        if not os.path.exists(times_font_path):
            times_font_path = fm.findfont('Times New Roman')
            times_italic_font_path = times_font_path
        
        font_prop = fm.FontProperties(fname=times_font_path)
        font_prop_italic = fm.FontProperties(fname=times_italic_font_path)
        
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
    except:
        # 如果字体设置失败，使用默认字体
        font_prop = fm.FontProperties()
        font_prop_italic = fm.FontProperties()
    
    # calculate confusion matrix - original number version
    cm = confusion_matrix(y_true, y_pred)
    
    # create chart
    plt.figure(figsize=(10, 8))
    
    # plot heatmap - use original number (not normalized)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontproperties=font_prop)
    plt.ylabel('True Label', fontproperties=font_prop)
    plt.xlabel('Predicted Label', fontproperties=font_prop)
    
    # set tick labels to italic
    plt.xticks(fontproperties=font_prop_italic, rotation=45, ha='right')
    plt.yticks(fontproperties=font_prop_italic)
    
    plt.tight_layout()
    
    # save PNG format (high DPI)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    # save SVG format
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.svg'), format='svg')
    
    plt.close()

def main():
    args = parse_args()
    set_seed(42)
    
    # create output directory
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
        
    # load data
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names, num_classes = get_data_loaders(
        args.data_dir, args.batch_size)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # save class information
    with open(os.path.join(output_dir, 'classes.json'), 'w') as f:
        json.dump(class_names, f)
    
    print(f"Detected {num_classes} classes: {class_names}")
    
    # initialize model
    print(f"Initializing {args.model_type} model...")
    model = get_model(args.model_type, num_classes, args.device)
    
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # use CosineAnnealingLR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01
    )
    
    # train model
    print("Starting training...")
    model = train_model(model, dataloaders, criterion, optimizer, 
                        scheduler, args.device, args.num_epochs, output_dir)
    
    # test model
    print("Evaluating model on test set...")
    results = evaluate_model(model, test_loader, criterion, args.device, class_names, output_dir)
    
    print(f"All results and models saved to {output_dir}")

if __name__ == "__main__":
    main() 