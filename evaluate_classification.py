import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm


DOMAIN_CHOICES = ["specimen", "ecological", "mixed"]
MODEL_CHOICES = ["efficientnet", "resnet"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TEST_SUMMARY_FIELDS = [
    "source_domain",
    "target_domain",
    "seed",
    "model_type",
    "test_loss",
    "accuracy",
    "accuracy_macro",
    "accuracy_weighted",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "f1_macro",
    "f1_weighted",
    "best_epoch",
    "best_val_macro_f1",
]
PER_CLASS_FIELDS = ["row_type", "label", "support", "accuracy", "precision", "recall", "f1"]
PREDICTION_FIELDS = [
    "split",
    "source_domain",
    "target_domain",
    "seed",
    "image_path",
    "true_index",
    "true_label",
    "pred_index",
    "pred_label",
    "top1_confidence",
]


class SafeToTensor:
    def __call__(self, image):
        tensor = TF.pil_to_tensor(image)
        return TF.convert_image_dtype(tensor, torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained moth classification checkpoint.")
    parser.add_argument("--data_root", type=str, default="data/classification")
    parser.add_argument("--source_domain", type=str, required=True, choices=DOMAIN_CHOICES)
    parser.add_argument("--target_domain", type=str, required=True, choices=DOMAIN_CHOICES)
    parser.add_argument("--model_type", type=str, default="resnet", choices=MODEL_CHOICES)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg, gpu_ids_arg):
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    if gpu_ids_arg.strip():
        gpu_ids = [int(item) for item in gpu_ids_arg.split(",") if item.strip()]
        return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids
    return torch.device(device_arg), []


def save_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


class CsvLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

    def write_row(self, row):
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def build_test_loader(data_root, target_domain, class_names, batch_size, num_workers, image_size):
    test_dir = Path(data_root) / target_domain / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {test_dir}")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            SafeToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = ImageFolderWithPaths(test_dir, transform=transform)
    if dataset.classes != class_names:
        raise ValueError("Checkpoint classes do not match target-domain test classes")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    ), str(test_dir), len(dataset)


def get_model(model_type, num_classes):
    if model_type == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_type == "resnet":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model_type: {model_type}")


def compute_metric_bundle(labels, preds, num_classes):
    class_indices = list(range(num_classes))
    accuracy = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, labels=class_indices, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, labels=class_indices, average="weighted", zero_division=0
    )
    label_array = np.asarray(labels)
    pred_array = np.asarray(preds)
    per_class_accuracy = []
    for class_idx in class_indices:
        class_mask = label_array == class_idx
        support = int(class_mask.sum())
        if support == 0:
            per_class_accuracy.append(0.0)
        else:
            per_class_accuracy.append(float((pred_array[class_mask] == label_array[class_mask]).mean()))
    return {
        "accuracy": float(accuracy),
        "accuracy_macro": float(np.mean(per_class_accuracy)) if per_class_accuracy else 0.0,
        "accuracy_weighted": float(accuracy),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    best_epoch = checkpoint["epoch"]
    best_val_macro_f1 = checkpoint["metrics"].get("best_val_macro_f1", checkpoint["metrics"].get("val", {}).get("f1_macro"))

    device, gpu_ids = resolve_device(args.device, args.gpu_ids)
    model = get_model(args.model_type, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    if gpu_ids and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    model.eval()

    test_loader, test_dir, test_size = build_test_loader(
        args.data_root,
        args.target_domain,
        class_names,
        args.batch_size,
        args.num_workers,
        args.image_size,
    )
    criterion = nn.CrossEntropyLoss()
    prediction_logger = CsvLogger(output_dir / "predictions.csv", PREDICTION_FIELDS)

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_confidences = []
    all_paths = []

    with torch.no_grad():
        for inputs, labels, paths in tqdm(test_loader, desc="test", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, preds = probabilities.max(dim=1)
            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())
            all_paths.extend(paths)

    test_loss = running_loss / len(test_loader.dataset)
    metrics = compute_metric_bundle(all_labels, all_preds, len(class_names))
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )

    per_class_logger = CsvLogger(output_dir / "per_class_metrics.csv", PER_CLASS_FIELDS)
    label_array = np.asarray(all_labels)
    pred_array = np.asarray(all_preds)
    for class_idx, class_name in enumerate(class_names):
        class_mask = label_array == class_idx
        support = int(class_mask.sum())
        class_accuracy = float((pred_array[class_mask] == label_array[class_mask]).mean()) if support else 0.0
        per_class_logger.write_row(
            {
                "row_type": "class",
                "label": class_name,
                "support": support,
                "accuracy": class_accuracy,
                "precision": float(precision_per_class[class_idx]),
                "recall": float(recall_per_class[class_idx]),
                "f1": float(f1_per_class[class_idx]),
            }
        )
    per_class_logger.write_row({"row_type": "summary_macro", "label": "macro", "support": "", "accuracy": metrics["accuracy_macro"], "precision": metrics["precision_macro"], "recall": metrics["recall_macro"], "f1": metrics["f1_macro"]})
    per_class_logger.write_row({"row_type": "summary_weighted", "label": "weighted", "support": "", "accuracy": metrics["accuracy_weighted"], "precision": metrics["precision_weighted"], "recall": metrics["recall_weighted"], "f1": metrics["f1_weighted"]})

    for image_path, true_idx, pred_idx, confidence in zip(all_paths, all_labels, all_preds, all_confidences):
        prediction_logger.write_row(
            {
                "split": "test",
                "source_domain": args.source_domain,
                "target_domain": args.target_domain,
                "seed": args.seed,
                "image_path": image_path,
                "true_index": true_idx,
                "true_label": class_names[true_idx],
                "pred_index": pred_idx,
                "pred_label": class_names[pred_idx],
                "top1_confidence": float(confidence),
            }
        )

    summary_row = {
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "seed": args.seed,
        "model_type": args.model_type,
        "test_loss": float(test_loss),
        "accuracy": metrics["accuracy"],
        "accuracy_macro": metrics["accuracy_macro"],
        "accuracy_weighted": metrics["accuracy_weighted"],
        "precision_macro": metrics["precision_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_macro": metrics["recall_macro"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
    }
    summary_logger = CsvLogger(output_dir / "test_summary.csv", TEST_SUMMARY_FIELDS)
    summary_logger.write_row(summary_row)
    save_json(output_dir / "test_summary.json", {"summary": summary_row})
    save_json(output_dir / "eval_config.json", {**vars(args), "resolved_device": str(device), "resolved_gpu_ids": gpu_ids, "test_dir": test_dir, "test_size": test_size, "class_names": class_names})
    print(f"test loss={test_loss:.4f} acc={metrics['accuracy']:.4f} macro_f1={metrics['f1_macro']:.4f} weighted_f1={metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
