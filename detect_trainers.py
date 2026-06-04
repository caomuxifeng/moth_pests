import math

import numpy as np
from ultralytics.data.build import build_dataloader
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOCAL_RANK, LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first


class StrictHyperparamDetectionTrainer(DetectionTrainer):
    """Detection trainer that preserves user-specified weight decay and disables pin_memory."""

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        worker_count = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=worker_count,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            pin_memory=False,
        )

    def _build_train_pipeline(self):
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task in {"obb", "semantic"} else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=self.args.weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()


class StrictHyperparamRTDETRTrainer(RTDETRTrainer, StrictHyperparamDetectionTrainer):
    """RT-DETR trainer with the same strict optimizer and dataloader behavior."""

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        dataset = super().build_dataset(img_path=img_path, mode=mode, batch=batch)
        if mode == "train":
            # Keep RT-DETR on true imgsz inputs instead of 2x mosaic canvases for fairer comparison with YOLO11.
            dataset.close_mosaic(self.args)
        return dataset
