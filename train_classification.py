import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm


DOMAIN_CHOICES = ["specimen", "ecological", "mixed"]
MODEL_CHOICES = ["efficientnet", "resnet"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SafeToTensor:
    def __call__(self, image):
        tensor = TF.pil_to_tensor(image)
        return TF.convert_image_dtype(tensor, torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a moth classification model.")
    parser.add_argument("--data_root", type=str, default="data/classification")
    parser.add_argument("--source_domain", type=str, default="mixed", choices=DOMAIN_CHOICES)
    parser.add_argument("--model_type", type=str, default="resnet", choices=MODEL_CHOICES)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="0,1")
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


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def build_transforms(image_size):
    to_tensor = SafeToTensor()
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            to_tensor,
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            to_tensor,
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(data_root, source_domain, batch_size, num_workers, image_size, seed):
    data_root = Path(data_root)
    train_dir = data_root / source_domain / "train"
    val_dir = data_root / source_domain / "val"
    for directory in [train_dir, val_dir]:
        if not directory.exists():
            raise FileNotFoundError(f"Dataset split not found: {directory}")

    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = ImageFolderWithPaths(train_dir, transform=train_transform)
    val_dataset = ImageFolderWithPaths(val_dir, transform=eval_transform)

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train and val class mappings do not match")

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "class_names": train_dataset.classes,
        "num_classes": len(train_dataset.classes),
        "dataset_sizes": {"train": len(train_dataset), "val": len(val_dataset)},
        "split_dirs": {"train": str(train_dir), "val": str(val_dir)},
    }


def get_model(model_type, num_classes):
    if model_type == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_type == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model_type: {model_type}")


def resolve_device(device_arg, gpu_ids_arg):
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    if gpu_ids_arg.strip():
        gpu_ids = [int(item) for item in gpu_ids_arg.split(",") if item.strip()]
        return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids
    return torch.device(device_arg), []


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


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


EPOCH_METRIC_FIELDS = [
    "epoch",
    "phase",
    "loss",
    "accuracy",
    "accuracy_macro",
    "accuracy_weighted",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "f1_macro",
    "f1_weighted",
    "lr",
]


def save_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, class_names, args):
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "class_names": class_names,
            "config": vars(args),
        },
        path,
    )


def run_phase(model, dataloader, criterion, optimizer, device, phase, num_classes):
    is_train = phase == "train"
    model.train(mode=is_train)
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for inputs, labels, _paths in tqdm(dataloader, desc=phase, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    metrics = compute_metric_bundle(all_labels, all_preds, num_classes)
    metrics["loss"] = running_loss / len(dataloader.dataset)
    return metrics


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, class_names, args, output_dir):
    output_dir = Path(output_dir)
    logger = CsvLogger(output_dir / "epoch_metrics.csv", EPOCH_METRIC_FIELDS)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_epoch = -1
    best_val_macro_f1 = -1.0
    start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")
        print("-" * 40)

        train_metrics = run_phase(model, dataloaders["train"], criterion, optimizer, device, "train", len(class_names))
        if scheduler is not None:
            scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        logger.write_row({"epoch": epoch, "phase": "train", "lr": current_lr, **train_metrics})
        print(
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
            f"macro_f1={train_metrics['f1_macro']:.4f} weighted_f1={train_metrics['f1_weighted']:.4f} lr={current_lr:.6f}"
        )

        val_metrics = run_phase(model, dataloaders["val"], criterion, optimizer, device, "val", len(class_names))
        logger.write_row({"epoch": epoch, "phase": "val", "lr": current_lr, **val_metrics})
        print(
            f"val   loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['f1_macro']:.4f} weighted_f1={val_metrics['f1_weighted']:.4f}"
        )

        last_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        save_checkpoint(checkpoints_dir / "last.pth", model, optimizer, scheduler, epoch, last_metrics, class_names, args)

        if val_metrics["f1_macro"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            best_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "best_val_macro_f1": best_val_macro_f1}
            save_checkpoint(checkpoints_dir / "best.pth", model, optimizer, scheduler, epoch, best_metrics, class_names, args)
            print(f"Saved new best checkpoint at epoch {epoch} with val macro_f1={best_val_macro_f1:.4f}")
        print()

    save_json(
        output_dir / "training_summary.json",
        {
            "elapsed_seconds": time.time() - start_time,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
        },
    )
    return best_epoch, best_val_macro_f1


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, gpu_ids = resolve_device(args.device, args.gpu_ids)
    bundle = build_dataloaders(args.data_root, args.source_domain, args.batch_size, args.num_workers, args.image_size, args.seed)
    model = get_model(args.model_type, bundle["num_classes"])
    if gpu_ids and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss()

    save_json(
        output_dir / "config.json",
        {
            **vars(args),
            "resolved_device": str(device),
            "resolved_gpu_ids": gpu_ids,
            "dataset_sizes": bundle["dataset_sizes"],
            "split_dirs": bundle["split_dirs"],
            "num_classes": bundle["num_classes"],
        },
    )
    save_json(output_dir / "classes.json", bundle["class_names"])

    print(f"Output directory: {output_dir}")
    print(f"Training domain: {args.source_domain}")
    print(f"Dataset sizes: {bundle['dataset_sizes']}")
    print(f"Resolved device: {device}")
    if gpu_ids:
        print(f"Using GPU ids: {gpu_ids}")

    best_epoch, best_val_macro_f1 = train_model(
        model,
        {"train": bundle["train"], "val": bundle["val"]},
        criterion,
        optimizer,
        scheduler,
        device,
        bundle["class_names"],
        args,
        output_dir,
    )
    print(f"Finished training. best_epoch={best_epoch}, best_val_macro_f1={best_val_macro_f1:.4f}")


if __name__ == "__main__":
    main()
