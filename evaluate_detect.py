import os
import sys


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

import atexit
import argparse
import csv
import gc
import json
import signal
import subprocess
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops


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

DOMAIN_CHOICES = ["specimen", "ecological", "mixed"]
MODEL_CHOICES = ["yolov11", "rtdetr"]
TEST_SUMMARY_FIELDS = [
    "source_domain",
    "target_domain",
    "seed",
    "model_type",
    "precision",
    "recall",
    "f1_score",
    "map50",
    "map50_95",
    "best_epoch",
    "best_val_map50_95",
]
PER_CLASS_FIELDS = ["row_type", "label", "support", "precision", "recall", "f1", "map50", "map50_95"]
PREDICTION_FIELDS = [
    "split",
    "source_domain",
    "target_domain",
    "seed",
    "image_path",
    "row_type",
    "true_index",
    "true_label",
    "pred_index",
    "pred_label",
    "score",
    "iou",
    "true_x1",
    "true_y1",
    "true_x2",
    "true_y2",
    "pred_x1",
    "pred_y1",
    "pred_x2",
    "pred_y2",
]


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained moth detection checkpoint.")
    parser.add_argument("--data_root", type=str, default="data/detection")
    parser.add_argument("--source_domain", type=str, required=True, choices=DOMAIN_CHOICES)
    parser.add_argument("--target_domain", type=str, required=True, choices=DOMAIN_CHOICES)
    parser.add_argument("--model_type", type=str, default="yolov11", choices=MODEL_CHOICES)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--match_iou_threshold", type=float, default=0.5)
    parser.add_argument("--eval_conf_threshold", type=float, default=0.001)
    parser.add_argument("--pred_conf_threshold", type=float, default=0.25)
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


def load_dataset_bundle(data_root, domain):
    domain_dir = Path(data_root) / domain
    dataset_yaml = domain_dir / "dataset.yaml"
    classes_txt = domain_dir / "classes.txt"

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    with dataset_yaml.open("r", encoding="utf-8") as handle:
        dataset_config = yaml.safe_load(handle)

    class_names = dataset_config.get("names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Invalid or empty 'names' in {dataset_yaml}")

    if classes_txt.exists():
        with classes_txt.open("r", encoding="utf-8") as handle:
            txt_class_names = [line.strip() for line in handle if line.strip()]
        if txt_class_names and txt_class_names != class_names:
            raise ValueError(f"classes.txt does not match dataset.yaml names in {domain_dir}")

    split_dirs = {}
    for split in ["train", "val", "test"]:
        split_value = dataset_config.get(split)
        if not split_value:
            raise ValueError(f"Missing '{split}' entry in {dataset_yaml}")
        split_path = (domain_dir / split_value).resolve()
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split path not found for '{split}': {split_path}")
        split_dirs[split] = split_path

    return {
        "domain_dir": domain_dir,
        "dataset_yaml": dataset_yaml,
        "class_names": class_names,
        "num_classes": len(class_names),
        "split_dirs": split_dirs,
    }


def resolve_eval_device(device_arg, gpu_ids_arg):
    normalized_device = device_arg.strip().lower()
    if not torch.cuda.is_available():
        return {
            "resolved_device": "cpu",
            "requested_gpu_ids": [],
            "ultralytics_device": "cpu",
        }

    if gpu_ids_arg.strip():
        requested_gpu_ids = [int(item) for item in gpu_ids_arg.split(",") if item.strip()]
        return {
            "resolved_device": "cuda:0",
            "requested_gpu_ids": requested_gpu_ids,
            "ultralytics_device": 0,
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


def load_training_summary(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    training_summary_path = checkpoint_path.parent.parent / "training_summary.json"
    if not training_summary_path.exists():
        return {"best_epoch": "", "best_val_map50_95": ""}
    with training_summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0])) * max(0.0, (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes, iou_threshold):
    candidates = []
    for gt_index, gt in enumerate(gt_boxes):
        for pred_index, pred in enumerate(pred_boxes):
            iou = compute_box_iou(gt["box"], pred["box"])
            if iou >= iou_threshold:
                candidates.append((gt_index, pred_index, iou))

    candidates.sort(key=lambda item: item[2], reverse=True)
    used_gt = set()
    used_pred = set()
    matches = []
    for gt_index, pred_index, iou in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches, used_gt, used_pred


def scalar_or_blank(value):
    return "" if value is None else value


def normalize_ratio_pad(ratio_pad):
    if ratio_pad is None:
        return None
    if isinstance(ratio_pad, (tuple, list)) and len(ratio_pad) == 2:
        first = ratio_pad[0]
        second = ratio_pad[1]
        if isinstance(first, (tuple, list)):
            return ratio_pad
        if all(isinstance(item, (int, float)) for item in (first, second)):
            return ((float(first), float(second)), (0.0, 0.0))
    return ratio_pad


def scale_prediction_boxes_for_csv(validator, predn, pbatch):
    predn_scaled = {
        "bboxes": predn["bboxes"].clone(),
        "conf": predn["conf"],
        "cls": predn["cls"],
    }
    if isinstance(validator, RTDETRValidator):
        predn_scaled["bboxes"][..., [0, 2]] *= pbatch["ori_shape"][1] / validator.args.imgsz
        predn_scaled["bboxes"][..., [1, 3]] *= pbatch["ori_shape"][0] / validator.args.imgsz
        return predn_scaled
    return validator.scale_preds(predn_scaled, pbatch)


class SinglePassCsvValidatorMixin:
    csv_conf_threshold = 0.25
    csv_match_iou_threshold = 0.5

    def __init__(self, args=None, _callbacks=None):
        super().__init__(args=args, _callbacks=_callbacks)
        self.prediction_rows = []

    def _collect_prediction_rows(self, predn, pbatch):
        gt_boxes = []
        if pbatch["cls"].shape[0]:
            scaled_gt = ops.scale_boxes(
                pbatch["imgsz"],
                pbatch["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=normalize_ratio_pad(pbatch["ratio_pad"]),
            )
            gt_cls = pbatch["cls"].int().cpu().tolist()
            gt_xyxy = scaled_gt.cpu().tolist()
            gt_boxes = [{"cls": int(cls_idx), "box": box} for cls_idx, box in zip(gt_cls, gt_xyxy)]

        if predn["cls"].shape[0]:
            predn_scaled = scale_prediction_boxes_for_csv(self, predn, pbatch)
            conf_mask = predn_scaled["conf"] >= float(self.csv_conf_threshold)
            pred_boxes = [
                {"cls": int(cls_idx), "box": box, "score": float(score)}
                for cls_idx, box, score in zip(
                    predn_scaled["cls"][conf_mask].int().cpu().tolist(),
                    predn_scaled["bboxes"][conf_mask].cpu().tolist(),
                    predn_scaled["conf"][conf_mask].cpu().tolist(),
                )
            ]
        else:
            pred_boxes = []

        matches, used_gt, used_pred = match_boxes(gt_boxes, pred_boxes, float(self.csv_match_iou_threshold))
        class_names = self.names
        image_path = str(pbatch["im_file"])

        for gt_index, pred_index, iou in matches:
            gt = gt_boxes[gt_index]
            pred = pred_boxes[pred_index]
            self.prediction_rows.append(
                {
                    "image_path": image_path,
                    "row_type": "match",
                    "true_index": gt["cls"],
                    "true_label": class_names[gt["cls"]],
                    "pred_index": pred["cls"],
                    "pred_label": class_names[pred["cls"]],
                    "score": pred["score"],
                    "iou": float(iou),
                    "true_x1": gt["box"][0],
                    "true_y1": gt["box"][1],
                    "true_x2": gt["box"][2],
                    "true_y2": gt["box"][3],
                    "pred_x1": pred["box"][0],
                    "pred_y1": pred["box"][1],
                    "pred_x2": pred["box"][2],
                    "pred_y2": pred["box"][3],
                }
            )

        for gt_index, gt in enumerate(gt_boxes):
            if gt_index in used_gt:
                continue
            self.prediction_rows.append(
                {
                    "image_path": image_path,
                    "row_type": "missed_gt",
                    "true_index": gt["cls"],
                    "true_label": class_names[gt["cls"]],
                    "pred_index": -1,
                    "pred_label": "background",
                    "score": None,
                    "iou": None,
                    "true_x1": gt["box"][0],
                    "true_y1": gt["box"][1],
                    "true_x2": gt["box"][2],
                    "true_y2": gt["box"][3],
                    "pred_x1": None,
                    "pred_y1": None,
                    "pred_x2": None,
                    "pred_y2": None,
                }
            )

        for pred_index, pred in enumerate(pred_boxes):
            if pred_index in used_pred:
                continue
            self.prediction_rows.append(
                {
                    "image_path": image_path,
                    "row_type": "false_positive",
                    "true_index": -1,
                    "true_label": "background",
                    "pred_index": pred["cls"],
                    "pred_label": class_names[pred["cls"]],
                    "score": pred["score"],
                    "iou": None,
                    "true_x1": None,
                    "true_y1": None,
                    "true_x2": None,
                    "true_y2": None,
                    "pred_x1": pred["box"][0],
                    "pred_y1": pred["box"][1],
                    "pred_x2": pred["box"][2],
                    "pred_y2": pred["box"][3],
                }
            )

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    "im_name": Path(pbatch["im_file"]).name,
                }
            )
            self.confusion_matrix.process_batch(predn, pbatch, conf=float(self.csv_conf_threshold))
            self._collect_prediction_rows(predn, pbatch)

    def get_stats(self):
        stats = super().get_stats()
        self.metrics.prediction_rows = self.prediction_rows
        self.metrics.image_count = self.seen
        return stats


class SinglePassDetectionValidator(SinglePassCsvValidatorMixin, DetectionValidator):
    pass


class SinglePassRTDETRValidator(SinglePassCsvValidatorMixin, RTDETRValidator):
    pass


def evaluate_checkpoint(model, dataset_yaml, args, device_bundle, output_dir):
    validator_cls = SinglePassRTDETRValidator if args.model_type == "rtdetr" else SinglePassDetectionValidator
    validator_cls.csv_match_iou_threshold = float(args.match_iou_threshold)
    validator_cls.csv_conf_threshold = float(args.pred_conf_threshold)
    metrics = model.val(
        data=str(dataset_yaml),
        split=args.eval_split,
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=args.num_workers,
        device=device_bundle["ultralytics_device"],
        project=str(output_dir),
        name="val",
        exist_ok=True,
        plots=False,
        save_json=False,
        verbose=True,
        conf=float(args.eval_conf_threshold),
        validator=validator_cls,
    )
    return metrics


def scalar_or_blank(value):
    return "" if value is None else value


def write_predictions_csv(path, prediction_rows, args):
    logger = CsvLogger(path, PREDICTION_FIELDS)
    for row in prediction_rows:
        logger.write_row(
            {
                "split": args.eval_split,
                "source_domain": args.source_domain,
                "target_domain": args.target_domain,
                "seed": args.seed,
                "image_path": row["image_path"],
                "row_type": row["row_type"],
                "true_index": row["true_index"],
                "true_label": row["true_label"],
                "pred_index": row["pred_index"],
                "pred_label": row["pred_label"],
                "score": scalar_or_blank(row["score"]),
                "iou": scalar_or_blank(row["iou"]),
                "true_x1": scalar_or_blank(row["true_x1"]),
                "true_y1": scalar_or_blank(row["true_y1"]),
                "true_x2": scalar_or_blank(row["true_x2"]),
                "true_y2": scalar_or_blank(row["true_y2"]),
                "pred_x1": scalar_or_blank(row["pred_x1"]),
                "pred_y1": scalar_or_blank(row["pred_y1"]),
                "pred_x2": scalar_or_blank(row["pred_x2"]),
                "pred_y2": scalar_or_blank(row["pred_y2"]),
            }
        )


def build_per_class_rows(metrics, class_names, class_support):
    summary_rows = metrics.summary()
    summary_by_label = {row["Class"]: row for row in summary_rows}

    class_rows = []
    for class_name, support in zip(class_names, class_support):
        row = summary_by_label.get(class_name, {})
        class_rows.append(
            {
                "row_type": "class",
                "label": class_name,
                "support": int(support),
                "precision": float(row.get("Box-P", 0.0) or 0.0),
                "recall": float(row.get("Box-R", 0.0) or 0.0),
                "f1": float(row.get("Box-F1", 0.0) or 0.0),
                "map50": float(row.get("mAP50", 0.0) or 0.0),
                "map50_95": float(row.get("mAP50-95", 0.0) or 0.0),
            }
        )

    macro_row = {
        "row_type": "summary_macro",
        "label": "macro",
        "support": "",
        "precision": float(np.mean([row["precision"] for row in class_rows])) if class_rows else 0.0,
        "recall": float(np.mean([row["recall"] for row in class_rows])) if class_rows else 0.0,
        "f1": float(np.mean([row["f1"] for row in class_rows])) if class_rows else 0.0,
        "map50": float(np.mean([row["map50"] for row in class_rows])) if class_rows else 0.0,
        "map50_95": float(np.mean([row["map50_95"] for row in class_rows])) if class_rows else 0.0,
    }

    total_support = sum(class_support)
    if total_support > 0:
        weighted_row = {
            "row_type": "summary_weighted",
            "label": "weighted",
            "support": "",
            "precision": float(sum(row["precision"] * row["support"] for row in class_rows) / total_support),
            "recall": float(sum(row["recall"] * row["support"] for row in class_rows) / total_support),
            "f1": float(sum(row["f1"] * row["support"] for row in class_rows) / total_support),
            "map50": float(sum(row["map50"] * row["support"] for row in class_rows) / total_support),
            "map50_95": float(sum(row["map50_95"] * row["support"] for row in class_rows) / total_support),
        }
    else:
        weighted_row = {
            "row_type": "summary_weighted",
            "label": "weighted",
            "support": "",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
        }

    return class_rows, macro_row, weighted_row


def write_per_class_csv(path, class_rows, macro_row, weighted_row):
    logger = CsvLogger(path, PER_CLASS_FIELDS)
    for row in class_rows:
        logger.write_row(row)
    logger.write_row(macro_row)
    logger.write_row(weighted_row)


def export_confusion_matrix_artifacts(output_dir, confusion_matrix, class_names, conf_threshold):
    if confusion_matrix is None or getattr(confusion_matrix, "matrix", None) is None:
        return

    labels = list(class_names) + ["background"]
    matrix = np.asarray(confusion_matrix.matrix, dtype=float)
    normalized = matrix / (matrix.sum(0, keepdims=True) + 1e-9)

    payload = {
        "axis_convention": {
            "rows": "predicted",
            "cols": "true",
        },
        "confidence_threshold": float(conf_threshold),
        "labels": labels,
        "matrix": matrix.tolist(),
        "normalized_matrix": normalized.tolist(),
    }
    save_json(output_dir / "confusion_matrix_data.json", payload)


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_bundle = load_dataset_bundle(args.data_root, args.source_domain)
    target_bundle = load_dataset_bundle(args.data_root, args.target_domain)
    if source_bundle["class_names"] != target_bundle["class_names"]:
        raise ValueError("Source and target detection class names do not match")

    device_bundle = resolve_eval_device(args.device, args.gpu_ids)
    training_summary = load_training_summary(args.checkpoint_path)

    save_json(
        output_dir / "eval_config.json",
        {
            **vars(args),
            "resolved_device": device_bundle["resolved_device"],
            "requested_gpu_ids": device_bundle["requested_gpu_ids"],
            "ultralytics_device": device_bundle["ultralytics_device"],
            "source_dataset_yaml": str(source_bundle["dataset_yaml"]),
            "target_dataset_yaml": str(target_bundle["dataset_yaml"]),
            "target_split_dir": str(target_bundle["split_dirs"][args.eval_split]),
            "class_names": target_bundle["class_names"],
        },
    )

    print(f"Output directory: {output_dir}")
    print(f"Evaluating {args.source_domain} -> {args.target_domain} on {args.eval_split}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Target dataset YAML: {target_bundle['dataset_yaml']}")
    print(f"Resolved device: {device_bundle['resolved_device']}")
    if device_bundle["requested_gpu_ids"]:
        print(f"Requested GPU ids: {device_bundle['requested_gpu_ids']}")

    model = YOLO(args.checkpoint_path)
    metrics = evaluate_checkpoint(model, target_bundle["dataset_yaml"], args, device_bundle, output_dir)
    metrics_dict = metrics.results_dict

    precision = float(metrics_dict["metrics/precision(B)"])
    recall = float(metrics_dict["metrics/recall(B)"])
    map50 = float(metrics_dict["metrics/mAP50(B)"])
    map50_95 = float(metrics_dict["metrics/mAP50-95(B)"])
    f1_score = compute_f1_score(precision, recall)

    prediction_rows = getattr(metrics, "prediction_rows", [])
    image_count = int(getattr(metrics, "image_count", 0))
    class_support = [int(x) for x in np.asarray(metrics.nt_per_class).tolist()]
    write_predictions_csv(output_dir / "predictions.csv", prediction_rows, args)

    class_rows, macro_row, weighted_row = build_per_class_rows(metrics, target_bundle["class_names"], class_support)
    write_per_class_csv(output_dir / "per_class_metrics.csv", class_rows, macro_row, weighted_row)
    export_confusion_matrix_artifacts(
        output_dir, getattr(metrics, "confusion_matrix", None), target_bundle["class_names"], args.pred_conf_threshold
    )

    summary_row = {
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "seed": args.seed,
        "model_type": args.model_type,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "map50": map50,
        "map50_95": map50_95,
        "best_epoch": training_summary.get("best_epoch", ""),
        "best_val_map50_95": training_summary.get("best_val_map50_95", ""),
    }
    summary_logger = CsvLogger(output_dir / "test_summary.csv", TEST_SUMMARY_FIELDS)
    summary_logger.write_row(summary_row)
    save_json(output_dir / "test_summary.json", {"summary": summary_row})

    print(
        f"precision={precision:.4f} recall={recall:.4f} f1_score={f1_score:.4f} "
        f"map50={map50:.4f} map50_95={map50_95:.4f}"
    )
    print(f"Images evaluated: {image_count}")
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_runtime()
