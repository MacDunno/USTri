import os
import json
import argparse
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch.losses as smp_losses
import numpy as np
import cv2

from dataset import MultiTaskDataset
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import (
    multi_task_collate_fn,
    evaluate,
    DetectionLoss,
    HeatmapLoss,
    set_seed,
)

TASK_CONFIGS = TASK_CONFIGURATIONS

# Defaults mostly aligned with stage-2
LEARNING_RATE = 1e-4
BATCH_SIZE = 20
NUM_EPOCHS = 800 # 400 for seg/det, 200 for reg/cls
DATA_ROOT_PATH = "data/train"
ENCODER = "R50-ViT-B_16"
ENCODER_WEIGHTS = None
RANDOM_SEED = 42
VAL_BATCH_SIZE = 32
REGRESSION_HEATMAP_SIZE = 64
IMAGE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LOG_DIR_BASE = "xxx"
MODEL_SAVE_COPY_TEMPLATE = "best_phase2_single_{task_id}.pth"
MAX_BEST_CHECKPOINTS = 20


def _segmentation_scale_block(image_size: int) -> A.OneOf:
    return A.OneOf([
        A.Compose([
            A.RandomScale(scale_limit=(0.0, 0.3), p=1.0),
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
        ], p=1.0),
        A.NoOp(),
    ], p=0.5)


def get_segmentation_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomGamma(gamma_limit=(80, 200), p=0.5),
        _segmentation_scale_block(image_size),
        A.ColorJitter(contrast=(0.8, 2.0), p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_detection_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomGamma(gamma_limit=(80, 200), p=0.5),
        A.ColorJitter(contrast=(0.8, 2.0), p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], clip=True, min_visibility=0.1))


def get_regression_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomGamma(gamma_limit=(80, 200), p=0.5),
        A.ColorJitter(contrast=(0.8, 2.0), p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


def get_classification_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        # _segmentation_scale_block(image_size),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1), 
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_common_val_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_detection_val_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], clip=True, min_visibility=0.1))


def get_regression_val_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


def build_train_transform_map(image_size: int):
    return {
        "segmentation": get_segmentation_train_transforms(image_size),
        "classification": get_classification_train_transforms(image_size),
        "detection": get_detection_train_transforms(image_size),
        "Regression": get_regression_train_transforms(image_size),
    }


def build_val_transform_map(image_size: int):
    common = get_common_val_transform(image_size)
    detection = get_detection_val_transform(image_size)
    regression = get_regression_val_transform(image_size)
    return {
        "segmentation": common,
        "classification": common,
        "Regression": regression,
        "detection": detection,
    }


def parse_args():
    task_ids = [cfg["task_id"] for cfg in TASK_CONFIGURATIONS]
    parser = argparse.ArgumentParser(description="Stage-2 single-dataset fine-tuning")
    parser.add_argument("--task-id", type=str, required=True, choices=task_ids, help="Exact dataset/task identifier to fine-tune.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-checkpoint", type=str, default="best_model.pth", help="Stage-1 checkpoint that seeds fine-tuning.")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from this checkpoint (with optimizer state).")
    parser.add_argument("--run-dir", type=str, default=None, help="Existing run dir inside log_phase2_single to append logs.")
    parser.add_argument("--log-dir-base", type=str, default=LOG_DIR_BASE, help="Root directory for logs and weights.")
    parser.add_argument("--per-dataset-decoders", action="store_true", help="Instantiate brand-new decoder+head for the task.")
    parser.add_argument("--reinit", action="store_true", help="Reset trainable decoder/head parameters before fine-tuning.")
    parser.add_argument("--enable-task-adapters", action="store_true", help="Enable classification-only adapters on the ResNet stem.")
    parser.add_argument("--adapter-reduction", type=int, default=4, help="Bottleneck reduction factor for adapters when enabled.")
    return parser.parse_args()


def prepare_run_directories(task_id: str, base_run_dir: str = None, log_dir_base: str = LOG_DIR_BASE):
    task_log_root = os.path.join(log_dir_base, task_id)
    run_dir = base_run_dir or os.path.join(task_log_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    train_log_path = os.path.join(run_dir, "train.txt")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    weights_dir = os.path.join(run_dir, "weights")
    return run_dir, train_log_path, tensorboard_dir, weights_dir


def load_best_checkpoint_metadata(metadata_path: str):
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
            return data.get("checkpoints", []), data.get("best_val_score", -float("inf"))
        except (json.JSONDecodeError, OSError):
            pass
    return [], -float("inf")


def save_best_checkpoint_metadata(metadata_path: str, checkpoints: list, best_score: float):
    with open(metadata_path, "w") as f:
        json.dump({"checkpoints": checkpoints, "best_val_score": best_score}, f, indent=2)


def set_requires_grad(module: nn.Module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad


def reinitialize_modules(modules: List[nn.Module], names: List[str]):
    seen = set()
    for module, name in zip(modules, names):
        if module is None:
            continue
        identifier = id(module)
        if identifier in seen:
            continue
        seen.add(identifier)
        for submodule in module.modules():
            reset = getattr(submodule, "reset_parameters", None)
            if callable(reset):
                reset()
        print(f"Reinitialized parameters for {name}")


def configure_trainable_modules(model: MultiTaskModelFactory, task_id: str, task_name: str) -> Tuple[List[str], List[nn.Module]]:
    set_requires_grad(model, False)
    modules = []
    names = []

    if getattr(model, "backbone_type", None) == "transunet":
        adapter_bank = getattr(model.encoder, "cls_stem_adapters", None)
        if task_name == "classification" and adapter_bank is not None and task_id in adapter_bank:
            modules.append(adapter_bank[task_id])
            names.append(f"encoder.cls_stem_adapters[{task_id}]")
        if getattr(model, "per_dataset_decoders", False) and hasattr(model, "task_decoders"):
            if task_id not in model.task_decoders:
                raise ValueError(f"Decoder for task '{task_id}' not found")
            decoder = model.task_decoders[task_id]
            modules.append(decoder)
            names.append(f"decoder[{task_id}]")
        else:
            branch_map = {
                "segmentation": ("encoder.seg_decoder", getattr(model.encoder, "seg_decoder", None)),
                "detection": ("encoder.det_decoder", getattr(model.encoder, "det_decoder", None)),
                "Regression": ("encoder.reg_decoder", getattr(model.encoder, "reg_decoder", None)),
            }
            if task_name not in branch_map:
                if task_name != "classification":
                    raise ValueError(f"Unsupported task type: {task_name}")
            else:
                branch_name, branch_module = branch_map[task_name]
                if branch_module is None:
                    raise ValueError(f"Model branch '{branch_name}' is missing for task type '{task_name}'")
                modules.append(branch_module)
                names.append(branch_name)
    else:
        if task_name in ["segmentation", "detection", "Regression"]:
            modules.append(model.fpn_decoder)
            names.append("fpn_decoder")

    if task_id not in model.heads:
        raise ValueError(f"Head for task '{task_id}' not found")
    head = model.heads[task_id]
    modules.append(head)
    names.append(f"head[{task_id}]")

    for module in modules:
        if module is not None:
            set_requires_grad(module, True)

    return names, modules


def filter_dataset_by_task(dataset: MultiTaskDataset, task_id: str):
    df = dataset.dataframe
    filtered = df[df["task_id"] == task_id]
    if filtered.empty:
        raise ValueError(f"No samples found for task_id='{task_id}' after filtering {len(df)} rows.")
    dataset.dataframe = filtered.reset_index(drop=True)
    print(f"Filtered dataset to task '{task_id}'. Remaining samples: {len(dataset)}")


def build_single_task_dataset(task_transforms, split: str, task_id: str) -> MultiTaskDataset:
    dataset = MultiTaskDataset(
        data_root=DATA_ROOT_PATH,
        task_transforms=task_transforms,
        regression_heatmap_size=REGRESSION_HEATMAP_SIZE,
        split=split,
    )
    filter_dataset_by_task(dataset, task_id)
    return dataset


def compute_average_score(results_df: np.ndarray) -> float:
    if results_df is None or len(results_df) == 0:
        return 0.0
    numeric_cols = []
    for col in results_df.columns:
        series = results_df[col]
        non_empty = series.dropna()
        if not non_empty.empty and isinstance(non_empty.iloc[0], (int, float, np.floating)):
            numeric_cols.append(col)
    if not numeric_cols:
        return 0.0
    scores = []
    for col in numeric_cols:
        values = results_df[col].dropna().astype(float)
        if values.empty:
            continue
        mean_value = float(values.mean())
        lower_is_better = any(keyword in col.lower() for keyword in ["mae", "loss", "error"])
        scores.append(-mean_value if lower_is_better else mean_value)
    return float(np.mean(scores)) if scores else 0.0


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_config = next((cfg for cfg in TASK_CONFIGS if cfg["task_id"] == args.task_id), None)
    if target_config is None:
        raise ValueError(f"Task ID '{args.task_id}' not defined in TASK_CONFIGURATIONS")
    task_name = target_config["task_name"]

    run_dir, train_log_path, tensorboard_dir, weights_dir = prepare_run_directories(
        args.task_id, args.run_dir, args.log_dir_base
    )
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    log_file = open(train_log_path, "a")
    log_file.write(f"[{datetime.now().isoformat()}] ==== Stage-2 single-task session started (task_id={args.task_id}, task_name={task_name}, LR={args.learning_rate}, Batch={args.batch_size}, Epochs={args.epochs}) ====\n")
    log_file.flush()

    best_metadata_path = os.path.join(weights_dir, "best_checkpoints.json")
    best_checkpoint_records, best_val_score = load_best_checkpoint_metadata(best_metadata_path)

    train_transforms = build_train_transform_map(IMAGE_SIZE)
    val_transforms = build_val_transform_map(IMAGE_SIZE)

    train_dataset = build_single_task_dataset(train_transforms, split="train", task_id=args.task_id)
    val_dataset = build_single_task_dataset(val_transforms, split="val", task_id=args.task_id)
    val_dataset_all = MultiTaskDataset(
        data_root=DATA_ROOT_PATH,
        task_transforms=val_transforms,
        regression_heatmap_size=REGRESSION_HEATMAP_SIZE,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=multi_task_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=multi_task_collate_fn,
    )
    val_loader_all = DataLoader(
        val_dataset_all,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=multi_task_collate_fn,
    )

    model = MultiTaskModelFactory(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        task_configs=TASK_CONFIGURATIONS,
        regression_heatmap_size=REGRESSION_HEATMAP_SIZE,
        per_dataset_decoders=args.per_dataset_decoders,
        use_task_adapters=args.enable_task_adapters,
        adapter_reduction=args.adapter_reduction,
    ).to(device)

    if not os.path.isfile(args.base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {args.base_checkpoint}")
    base_state = torch.load(args.base_checkpoint, map_location=device)
    model.load_state_dict(base_state)
    print(f"Loaded base weights from {args.base_checkpoint}")

    trainable_names, trainable_modules = configure_trainable_modules(model, args.task_id, task_name)
    print("Trainable modules:")
    for name in trainable_names:
        print(f"  - {name}")
    if args.reinit:
        print("--reinit flag detected: resetting selected decoder/head parameters before training.")
        reinitialize_modules(trainable_modules, trainable_names)

    loss_functions = {
        "segmentation": smp_losses.DiceLoss(mode="multiclass"),
        "classification": nn.CrossEntropyLoss(),
        "Regression": HeatmapLoss(),
        "detection": DetectionLoss(),
    }
    criterion = loss_functions[task_name]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for the selected task.")
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {total_trainable:,}")

    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6) # cls: 1e-8 else: 1e-6

    start_epoch = 0
    latest_state_path = os.path.join(weights_dir, "latest_checkpoint.pth")

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        resume_state = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        start_epoch = resume_state.get("epoch", 0)
        best_val_score = resume_state.get("best_val_score", best_val_score)
        print(f"Resumed training from {args.resume} at epoch {start_epoch}")

    print("\n--- Pre-Training Validation (target task) ---")
    pre_val_target = evaluate(model, val_loader, device)
    pre_val_target_score = compute_average_score(pre_val_target)
    if not pre_val_target.empty:
        print(pre_val_target.to_string(index=False))
    else:
        print("No validation results.")
    print(f"--- Average Val Score (Pre-Train): {pre_val_target_score:.4f} ---")
    writer.add_scalar("Val/AverageScore", pre_val_target_score, start_epoch)

    print("\n--- Pre-Training Validation (all tasks) ---")
    pre_val_all = evaluate(model, val_loader_all, device)
    pre_val_all_score = compute_average_score(pre_val_all)
    if not pre_val_all.empty:
        print(pre_val_all.to_string(index=False))
    else:
        print("No validation results.")
    print(f"--- All-Task Average Val Score (Pre-Train): {pre_val_all_score:.4f} ---")
    writer.add_scalar("Val/AllTasksAverageScore", pre_val_all_score, start_epoch)

    pre_val_lines = [
        "--- Pre-Training Validation (target task) ---",
        pre_val_target.to_string(index=False) if not pre_val_target.empty else "No validation results.",
        f"--- Average Val Score (Pre-Train): {pre_val_target_score:.4f} ---",
        "",
        "--- Pre-Training Validation (all tasks) ---",
        pre_val_all.to_string(index=False) if not pre_val_all.empty else "No validation results.",
        f"--- All-Task Average Val Score (Pre-Train): {pre_val_all_score:.4f} ---",
        "-" * 40,
    ]
    log_file.write("\n".join(pre_val_lines) + "\n")
    log_file.flush()

    print("\n" + "=" * 50 + "\n--- Start Stage-2 Single-Task Training ---")
    target_task_id = args.task_id
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in loop:
            images = batch["image"].to(device)
            labels = torch.stack(batch["label"]).to(device)

            outputs = model(images, task_id=target_task_id)
            if task_name == "detection":
                outputs = torch.clamp(outputs, 0.0, 1.0)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            loop.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        writer.add_scalar("Train/Loss", avg_loss, epoch + 1)
        writer.add_scalar("Train/LearningRate", scheduler.get_last_lr()[0], epoch + 1)
        print(f"Train Loss (epoch {epoch + 1}): {avg_loss:.4f}")

        val_results = evaluate(model, val_loader, device)
        target_avg_score = compute_average_score(val_results)
        writer.add_scalar("Val/AverageScore", target_avg_score, epoch + 1)

        print(f"\n--- Epoch {epoch + 1} Validation Report (task_id={args.task_id}) ---")
        if not val_results.empty:
            print(val_results.to_string(index=False))
        else:
            print("No validation results.")
        print(f"--- Average Val Score: {target_avg_score:.4f} ---")

        log_block = [
            f"Epoch {epoch + 1} Summary",
            f"Train Loss: {avg_loss:.4f}",
            f"Val Score : {target_avg_score:.4f}",
        ]
        if not val_results.empty:
            log_block.append(val_results.to_string(index=False))
        log_block.append("-" * 40)
        log_file.write("\n".join(log_block) + "\n")
        log_file.flush()

        qualify = False
        if not val_results.empty:
            if len(best_checkpoint_records) < MAX_BEST_CHECKPOINTS:
                qualify = True
            elif target_avg_score > best_checkpoint_records[-1]["score"]:
                qualify = True

        if qualify:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ckpt_name = f"best_epoch{epoch + 1:04d}_score{target_avg_score:.4f}_{timestamp}.pth"
            ckpt_path = os.path.join(weights_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            best_checkpoint_records.append({"path": ckpt_path, "score": float(target_avg_score), "epoch": epoch + 1})
            best_checkpoint_records.sort(key=lambda x: x["score"], reverse=True)

            while len(best_checkpoint_records) > MAX_BEST_CHECKPOINTS:
                to_remove = best_checkpoint_records.pop(-1)
                try:
                    os.remove(to_remove["path"])
                except OSError:
                    pass

            if target_avg_score >= best_val_score:
                best_val_score = target_avg_score
                torch.save(model.state_dict(), MODEL_SAVE_COPY_TEMPLATE.format(task_id=args.task_id))
                print("Updated convenience copy with new best checkpoint.")

            save_best_checkpoint_metadata(best_metadata_path, best_checkpoint_records, best_val_score)
            print(f"Saved ranked checkpoint at {ckpt_path} (score={target_avg_score:.4f})")

        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_score": best_val_score,
            "run_dir": run_dir,
            "task_id": args.task_id,
        }, latest_state_path)

        scheduler.step()

    final_message = (
        f"\n--- Stage-2 Single-Task Finished ---\n"
        f"Run directory: {run_dir}\n"
        f"Best checkpoints stored in: {weights_dir}\n"
        f"Resume checkpoint: {latest_state_path if os.path.exists(latest_state_path) else 'N/A'}\n"
    )
    print(final_message)
    log_file.write(f"[{datetime.now().isoformat()}] {final_message}\n")
    log_file.flush()
    writer.close()
    log_file.close()


if __name__ == "__main__":
    main()
