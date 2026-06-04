import os
import sys
from pathlib import Path


def ensure_project_on_pythonpath():
    project_root = Path(__file__).resolve().parent
    existing = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if str(project_root) not in existing:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(project_root), *existing]) if existing else str(project_root)


def configure_cuda_visibility():
    """Set CUDA visibility before importing torch/ultralytics."""
    gpu_ids_arg = ""
    device_arg = ""
    for index, arg in enumerate(sys.argv):
        if arg == "--gpu_ids" and index + 1 < len(sys.argv):
            gpu_ids_arg = sys.argv[index + 1].strip()
        elif arg == "--device" and index + 1 < len(sys.argv):
            device_arg = sys.argv[index + 1].strip()

    if gpu_ids_arg:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_arg
        print(f"[EARLY CUDA SETUP] CUDA_VISIBLE_DEVICES={gpu_ids_arg}")
        return

    if device_arg.startswith("cuda:"):
        gpu_id = device_arg.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"[EARLY CUDA SETUP] CUDA_VISIBLE_DEVICES={gpu_id}")


configure_cuda_visibility()
ensure_project_on_pythonpath()

import atexit
import argparse
import csv
import gc
import json
import signal
import subprocess
import random
import shutil
import time
import warnings

import numpy as np
import torch
import yaml
from detect_trainers import StrictHyperparamDetectionTrainer, StrictHyperparamRTDETRTrainer
from ultralytics import YOLO


RUNTIME_CLEANED = False


def list_descendant_pids(root_pid):
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    children = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        pid, ppid = (int(parts[0]), int(parts[1]))
        children.setdefault(ppid, []).append(pid)

    ordered = []
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        ordered.append(pid)
        stack.extend(children.get(pid, []))
    return ordered


def terminate_descendants(root_pid):
    descendants = list_descendant_pids(root_pid)
    if not descendants:
        return

    for pid in reversed(descendants):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception:
            pass

    time.sleep(1.0)

    for pid in reversed(descendants):
        if not Path(f"/proc/{pid}").exists():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except Exception:
            pass


def cleanup_runtime():
    global RUNTIME_CLEANED
    if RUNTIME_CLEANED:
        return
    RUNTIME_CLEANED = True

    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass

    try:
        terminate_descendants(os.getpid())
    except Exception:
        pass

    try:
        gc.collect()
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def handle_termination(signum, _frame):
    cleanup_runtime()
    raise SystemExit(128 + signum)


atexit.register(cleanup_runtime)
signal.signal(signal.SIGINT, handle_termination)
signal.signal(signal.SIGTERM, handle_termination)

warnings.filterwarnings("ignore", category=UserWarning)

DOMAIN_CHOICES = ["specimen", "ecological", "mixed"]
MODEL_CHOICES = ["yolov11", "rtdetr"]
EPOCH_METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "precision",
    "recall",
    "f1_score",
    "map50",
    "map50_95",
    "lr",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a moth detection model.")
    parser.add_argument("--data_root", type=str, default="data/detection")
    parser.add_argument("--source_domain", type=str, default="mixed", choices=DOMAIN_CHOICES)
    parser.add_argument("--model_type", type=str, default="yolov11", choices=MODEL_CHOICES)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_mode", type=str, default="none", choices=["ram", "disk", "none"])
    parser.add_argument("--clear_stale_npy_cache", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def cleanup_stale_npy_cache(domain_dir, enabled):
    npy_files = sorted(domain_dir.rglob("*.npy"))
    if not npy_files:
        return 0

    if not enabled:
        print(
            f"Warning: found {len(npy_files)} stale .npy cache files under {domain_dir}. "
            "Use --clear_stale_npy_cache to remove them before training."
        )
        return len(npy_files)

    for npy_file in npy_files:
        npy_file.unlink(missing_ok=True)
    print(f"Removed {len(npy_files)} stale .npy cache files from {domain_dir}")
    return len(npy_files)


def load_dataset_bundle(data_root, source_domain):
    domain_dir = Path(data_root) / source_domain
    dataset_yaml = domain_dir / "dataset.yaml"
    classes_txt = domain_dir / "classes.txt"

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    with dataset_yaml.open("r", encoding="utf-8") as handle:
        dataset_config = yaml.safe_load(handle)

    class_names = dataset_config.get("names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Invalid or empty 'names' in {dataset_yaml}")

    num_classes = dataset_config.get("nc")
    if num_classes is None:
        num_classes = len(class_names)
    if int(num_classes) != len(class_names):
        raise ValueError(
            f"Mismatch between nc ({num_classes}) and names length ({len(class_names)}) in {dataset_yaml}"
        )

    if classes_txt.exists():
        with classes_txt.open("r", encoding="utf-8") as handle:
            txt_class_names = [line.strip() for line in handle if line.strip()]
        if txt_class_names and txt_class_names != class_names:
            raise ValueError(f"classes.txt does not match dataset.yaml names in {domain_dir}")

    for split in ["train", "val", "test"]:
        split_value = dataset_config.get(split)
        if not split_value:
            raise ValueError(f"Missing '{split}' entry in {dataset_yaml}")
        split_path = (domain_dir / split_value).resolve()
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split path not found for '{split}': {split_path}")

    return {
        "domain_dir": domain_dir,
        "dataset_yaml": dataset_yaml,
        "class_names": class_names,
        "num_classes": len(class_names),
    }


def resolve_training_device(device_arg, gpu_ids_arg):
    normalized_device = device_arg.strip().lower()

    if not torch.cuda.is_available():
        return {
            "resolved_device": "cpu",
            "requested_gpu_ids": [],
            "ultralytics_device": "cpu",
        }

    if gpu_ids_arg.strip():
        requested_gpu_ids = [int(item) for item in gpu_ids_arg.split(",") if item.strip()]
        visible_gpu_ids = list(range(len(requested_gpu_ids)))
        ultralytics_device = (
            ",".join(str(item) for item in visible_gpu_ids)
            if len(visible_gpu_ids) > 1
            else visible_gpu_ids[0]
        )
        return {
            "resolved_device": "cuda:0",
            "requested_gpu_ids": requested_gpu_ids,
            "ultralytics_device": ultralytics_device,
        }

    if normalized_device.startswith("cuda"):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            return {
                "resolved_device": "cuda:0",
                "requested_gpu_ids": [int(item) for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if item],
                "ultralytics_device": 0,
            }
        if ":" in normalized_device:
            gpu_index = int(normalized_device.split(":", 1)[1])
            return {
                "resolved_device": f"cuda:{gpu_index}",
                "requested_gpu_ids": [gpu_index],
                "ultralytics_device": gpu_index,
            }
        return {
            "resolved_device": "cuda:0",
            "requested_gpu_ids": [0],
            "ultralytics_device": 0,
        }

    return {
        "resolved_device": "cpu",
        "requested_gpu_ids": [],
        "ultralytics_device": "cpu",
    }


def get_model(model_type):
    if model_type == "yolov11":
        print("Loading YOLOv11n pretrained model...")
        return YOLO("yolo11n.pt")
    if model_type == "rtdetr":
        print("Loading RT-DETR-l pretrained model...")
        return YOLO("rtdetr-l.pt")
    raise ValueError(f"Unsupported model_type: {model_type}")


def compute_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def read_ultralytics_results(results_csv_path):
    if not results_csv_path.exists():
        raise FileNotFoundError(f"Ultralytics results CSV not found: {results_csv_path}")

    with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No epoch rows found in {results_csv_path}")

    epoch_rows = []
    best_row = None

    for row in rows:
        epoch = int(float(row["epoch"]))
        precision = float(row.get("metrics/precision(B)", 0.0) or 0.0)
        recall = float(row.get("metrics/recall(B)", 0.0) or 0.0)
        map50 = float(row.get("metrics/mAP50(B)", 0.0) or 0.0)
        map50_95 = float(row.get("metrics/mAP50-95(B)", 0.0) or 0.0)
        lr = float(row.get("lr/pg0", 0.0) or 0.0)

        train_loss = sum(
            float(value)
            for key, value in row.items()
            if key.startswith("train/") and key.endswith("loss") and value not in (None, "")
        )
        val_loss = sum(
            float(value)
            for key, value in row.items()
            if key.startswith("val/") and key.endswith("loss") and value not in (None, "")
        )

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": compute_f1_score(precision, recall),
            "map50": map50,
            "map50_95": map50_95,
            "lr": lr,
        }
        epoch_rows.append(epoch_row)

        if best_row is None or epoch_row["map50_95"] > best_row["map50_95"]:
            best_row = epoch_row

    return epoch_rows, best_row


def write_epoch_metrics(path, epoch_rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPOCH_METRIC_FIELDS)
        writer.writeheader()
        for row in epoch_rows:
            writer.writerow(row)


def copy_strict_best_checkpoint(train_dir, checkpoints_dir, best_epoch):
    source_epoch_checkpoint = train_dir / "weights" / f"epoch{best_epoch - 1}.pt"
    source_last_checkpoint = train_dir / "weights" / "last.pt"

    if not source_epoch_checkpoint.exists():
        raise FileNotFoundError(
            f"Expected epoch checkpoint for best_epoch={best_epoch} not found: {source_epoch_checkpoint}. "
            "save_period=1 is required for strict best checkpoint selection by mAP50-95."
        )
    if not source_last_checkpoint.exists():
        raise FileNotFoundError(f"Last checkpoint not found: {source_last_checkpoint}")

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    target_best = checkpoints_dir / "best.pt"
    target_last = checkpoints_dir / "last.pt"
    shutil.copy2(source_epoch_checkpoint, target_best)
    shutil.copy2(source_last_checkpoint, target_last)
    return target_best, target_last, source_epoch_checkpoint


def get_trainer_class(model_type):
    if model_type == "yolov11":
        return StrictHyperparamDetectionTrainer
    if model_type == "rtdetr":
        return StrictHyperparamRTDETRTrainer
    raise ValueError(f"Unsupported model_type: {model_type}")


def shutdown_loader_workers(loader):
    if loader is None:
        return
    iterator = getattr(loader, "iterator", None)
    if iterator is None:
        return
    try:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass


def train_model(model, dataset_yaml, device_bundle, args, output_dir):
    output_dir = Path(output_dir)
    train_subdir_name = "train"
    cache_value = False if args.cache_mode == "none" else args.cache_mode

    train_params = {
        "data": str(dataset_yaml),
        "epochs": args.num_epochs,
        "imgsz": args.image_size,
        "batch": args.batch_size,
        "device": device_bundle["ultralytics_device"],
        "project": str(output_dir),
        "name": train_subdir_name,
        "optimizer": args.optimizer,
        "lr0": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "workers": args.num_workers,
        "cache": cache_value,
        "patience": 0,
        "save_period": 1,
        "seed": args.seed,
        "deterministic": True,
        "exist_ok": True,
    }

    print(f"Ultralytics device argument: {device_bundle['ultralytics_device']}")
    print(f"Optimizer: {args.optimizer} | lr={args.lr} | momentum={args.momentum} | weight_decay={args.weight_decay}")
    print(f"Cache mode: {args.cache_mode}")
    print(f"Training for {args.num_epochs} epochs with image size {args.image_size}")

    trainer_class = get_trainer_class(args.model_type)
    start_time = time.time()
    try:
        model.train(trainer=trainer_class, **train_params)
    finally:
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            shutdown_loader_workers(getattr(trainer, "train_loader", None))
            shutdown_loader_workers(getattr(trainer, "test_loader", None))
    elapsed_seconds = time.time() - start_time

    train_dir = output_dir / train_subdir_name
    results_csv_path = train_dir / "results.csv"
    epoch_rows, best_row = read_ultralytics_results(results_csv_path)
    write_epoch_metrics(output_dir / "epoch_metrics.csv", epoch_rows)

    checkpoints_dir = output_dir / "checkpoints"
    best_checkpoint, last_checkpoint, strict_best_source = copy_strict_best_checkpoint(
        train_dir, checkpoints_dir, best_row["epoch"]
    )

    training_summary = {
        "elapsed_seconds": elapsed_seconds,
        "best_epoch": best_row["epoch"],
        "best_val_map50_95": best_row["map50_95"],
        "best_val_map50": best_row["map50"],
        "best_val_precision": best_row["precision"],
        "best_val_recall": best_row["recall"],
        "best_val_f1_score": best_row["f1_score"],
        "raw_train_dir": str(train_dir),
        "strict_best_checkpoint_source": str(strict_best_source),
        "strict_best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
    }
    save_json(output_dir / "training_summary.json", training_summary)
    return training_summary


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_bundle = load_dataset_bundle(args.data_root, args.source_domain)
    cleanup_stale_npy_cache(dataset_bundle["domain_dir"], args.clear_stale_npy_cache)
    device_bundle = resolve_training_device(args.device, args.gpu_ids)

    save_json(
        output_dir / "config.json",
        {
            **vars(args),
            "resolved_device": device_bundle["resolved_device"],
            "requested_gpu_ids": device_bundle["requested_gpu_ids"],
            "ultralytics_device": device_bundle["ultralytics_device"],
            "dataset_yaml": str(dataset_bundle["dataset_yaml"]),
            "domain_dir": str(dataset_bundle["domain_dir"]),
            "num_classes": dataset_bundle["num_classes"],
        },
    )
    save_json(output_dir / "classes.json", dataset_bundle["class_names"])

    print(f"Output directory: {output_dir}")
    print(f"Training domain: {args.source_domain}")
    print(f"Dataset YAML: {dataset_bundle['dataset_yaml']}")
    print(f"Resolved device: {device_bundle['resolved_device']}")
    if device_bundle["requested_gpu_ids"]:
        print(f"Requested GPU ids: {device_bundle['requested_gpu_ids']}")

    model = get_model(args.model_type)
    summary = train_model(model, dataset_bundle["dataset_yaml"], device_bundle, args, output_dir)
    print(
        "Finished training. "
        f"best_epoch={summary['best_epoch']}, "
        f"best_val_map50_95={summary['best_val_map50_95']:.4f}, "
        f"best_val_f1_score={summary['best_val_f1_score']:.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_runtime()
