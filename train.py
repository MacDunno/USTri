import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch.losses as smp_losses
import numpy as np
import random
import cv2

# Import local modules
from dataset import MultiTaskDataset, MultiTaskUniformSampler
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import (
    multi_task_collate_fn, 
    evaluate, 
    DetectionLoss, 
    HeatmapLoss,
    set_seed
)

# Training configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 20
NUM_EPOCHS = 400 
DATA_ROOT_PATH = 'data/train'
ENCODER = 'R50-ViT-B_16'
ENCODER_WEIGHTS = None
RANDOM_SEED = 42
MODEL_SAVE_PATH = 'best_model.pth' 
VAL_SPLIT = 0.2
REGRESSION_HEATMAP_SIZE = 64
IMAGE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LOG_DIR = 'log'
MAX_BEST_CHECKPOINTS = 400


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-task ultrasound training")
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to a training checkpoint that includes model/optimizer/scheduler states.',
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        default=None,
        help='Optional existing run directory under log/ to append logs and checkpoints.',
    )
    return parser.parse_args()


def prepare_run_directories(base_run_dir: str = None):
    run_dir = base_run_dir or os.path.join(LOG_DIR, datetime.now().strftime('%Y%m%d_%H%M%S'))
    train_log_path = os.path.join(run_dir, 'train.txt')
    tensorboard_dir = os.path.join(run_dir, 'tensorboard')
    weights_dir = os.path.join(run_dir, 'weights')
    return run_dir, train_log_path, tensorboard_dir, weights_dir


def load_best_checkpoint_metadata(metadata_path: str):
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            checkpoints = data.get('checkpoints', [])
            best_score = data.get('best_val_score', -float('inf'))
            return checkpoints, best_score
        except (json.JSONDecodeError, OSError):
            pass
    return [], -float('inf')


def save_best_checkpoint_metadata(metadata_path: str, checkpoints: list, best_score: float):
    metadata = {
        'checkpoints': checkpoints,
        'best_val_score': best_score,
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


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
        A.RandomGamma(gamma_limit=(100, 250), p=0.5),
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
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomGamma(gamma_limit=(100, 250), p=0.5),
        A.ColorJitter(contrast=(0.8, 2.0), p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))


def get_regression_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomGamma(gamma_limit=(100, 250), p=0.5),
        A.ColorJitter(contrast=(0.8, 2.0), p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def get_classification_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
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
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))


def get_regression_val_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def build_train_transform_map(image_size: int):
    return {
        'segmentation': get_segmentation_train_transforms(image_size),
        'classification': get_classification_train_transforms(image_size),
        'detection': get_detection_train_transforms(image_size),
        'Regression': get_regression_train_transforms(image_size),
    }


def build_val_transform_map(image_size: int):
    common = get_common_val_transform(image_size)
    detection = get_detection_val_transform(image_size)
    regression = get_regression_val_transform(image_size)
    return {
        'segmentation': common,
        'classification': common,
        'Regression': regression,
        'detection': detection,
    }

def main():
    args = parse_args()
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    os.makedirs(LOG_DIR, exist_ok=True)
    run_log_dir, train_log_path, tensorboard_dir, weights_dir = prepare_run_directories(args.run_dir)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    best_metadata_path = os.path.join(weights_dir, 'best_checkpoints.json')
    best_checkpoint_records, metadata_best_score = load_best_checkpoint_metadata(best_metadata_path)
    # Remove records that no longer have files on disk
    cleaned_records = []
    for record in best_checkpoint_records:
        path = record.get('path')
        score = record.get('score')
        epoch = record.get('epoch')
        if path and os.path.exists(path) and score is not None:
            cleaned_records.append({'path': path, 'score': float(score), 'epoch': epoch})
    if len(cleaned_records) != len(best_checkpoint_records):
        save_best_checkpoint_metadata(best_metadata_path, cleaned_records, metadata_best_score)
    best_checkpoint_records = sorted(cleaned_records, key=lambda x: x['score'], reverse=True)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    log_file = open(train_log_path, 'a')
    session_header = (
        f"[{datetime.now().isoformat()}] ==== Training session started "
        f"(LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={NUM_EPOCHS}, RunDir={run_log_dir}) ===="
    )
    log_file.write(session_header + "\n")
    log_file.flush()

    best_val_score = metadata_best_score if metadata_best_score != -float('inf') else -float('inf')
    if best_checkpoint_records:
        best_val_score = max(best_val_score, best_checkpoint_records[0]['score'])
    current_top_checkpoint_path = best_checkpoint_records[0]['path'] if best_checkpoint_records else None
    latest_state_path = os.path.join(weights_dir, 'latest_checkpoint.pth')

    # Data loading and splitting
    train_task_transforms = build_train_transform_map(IMAGE_SIZE)
    val_task_transforms = build_val_transform_map(IMAGE_SIZE)

    train_dataset = MultiTaskDataset(
        data_root=DATA_ROOT_PATH,
        task_transforms=train_task_transforms,
        regression_heatmap_size=REGRESSION_HEATMAP_SIZE,
        split='train',
    )
    val_dataset = MultiTaskDataset(
        data_root=DATA_ROOT_PATH,
        task_transforms=val_task_transforms,
        regression_heatmap_size=REGRESSION_HEATMAP_SIZE,
        split='val',
    )

    if len(train_dataset) == 0:
        raise ValueError("No training samples were found. Please verify the 'train' column in the CSV files.")
    if len(val_dataset) == 0:
        raise ValueError("No validation samples were found. Please verify the 'train' column in the CSV files.")

    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    train_sampler = MultiTaskUniformSampler(train_dataset, batch_size=BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    # Model and loss setup
    model = MultiTaskModelFactory(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        task_configs=TASK_CONFIGURATIONS,
        regression_heatmap_size=REGRESSION_HEATMAP_SIZE,
    ).to(device)
    
    loss_functions = {
        'segmentation': smp_losses.DiceLoss(mode='multiclass'), 
        'classification': nn.CrossEntropyLoss(),
        'Regression': HeatmapLoss(), 
        'detection': DetectionLoss()
    }
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    # Optimization setup
    print("\n--- Setting parameter groups ---")
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': LEARNING_RATE * 1},
    ]
    print(f"  - Shared Encoder                 -> LR: {LEARNING_RATE * 1}")
    
    for task_id, head in model.heads.items():
        lr_multiplier = 10.0
        current_lr = LEARNING_RATE * lr_multiplier
        param_groups.append({'params': head.parameters(), 'lr': current_lr})
        print(f"  - Task Head '{task_id:<25}' -> LR: {current_lr}")

    optimizer = optim.AdamW(param_groups)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    print("\n--- Cosine Annealing Scheduler configured ---")

    start_epoch = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        checkpoint_state = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint_state['model_state'])
        optimizer.load_state_dict(checkpoint_state['optimizer_state'])
        scheduler.load_state_dict(checkpoint_state['scheduler_state'])
        start_epoch = checkpoint_state.get('epoch', 0)
        best_val_score = checkpoint_state.get('best_val_score', best_val_score)
        resume_msg = f"[{datetime.now().isoformat()}] Resumed training from {args.resume} at epoch {start_epoch}"
        print(resume_msg)
        log_file.write(resume_msg + "\n")
        log_file.flush()

    print("\n" + "="*50 + "\n--- Start Training ---")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch in loop:
            images = batch['image'].to(device)
            task_ids = batch['task_id']
            # Manually stack labels list to tensor
            labels = torch.stack(batch['label']).to(device)

            # All samples in batch belong to the same task due to sampler
            current_task_id = task_ids[0]
            task_name = task_id_to_name[current_task_id]

            outputs = model(images, task_id=current_task_id)
            
            if task_name == 'detection':
                final_outputs = torch.clamp(outputs, 0.0, 1.0)
            else:
                final_outputs = outputs

            loss = loss_functions[task_name](final_outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses[current_task_id].append(loss.item())
            loop.set_postfix(loss=loss.item(), task=current_task_id, lr=scheduler.get_last_lr()[0])

        # Train reporting
        print("\n--- Epoch {} Average Train Loss Report ---".format(epoch + 1))
        sorted_task_ids = sorted(epoch_train_losses.keys())
        train_loss_summary = []
        train_log_lines = [f"--- Epoch {epoch + 1} Average Train Loss Report ---"]
        for task_id in sorted_task_ids:
            avg_loss = np.mean(epoch_train_losses[task_id])
            task_name = task_id_to_name.get(task_id, 'unknown')
            print(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
            train_loss_summary.append(f"{task_name}({task_id}):{avg_loss:.4f}")
            train_log_lines.append(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
            writer.add_scalar(f"Train/{task_name}_{task_id}_loss", avg_loss, epoch + 1)
        if not sorted_task_ids:
            writer.add_scalar("Train/NoTaskBatches", 0.0, epoch + 1)
        train_log_lines.append("-" * 40)
        writer.add_scalar("Train/LearningRate", scheduler.get_last_lr()[0], epoch + 1)
        print("-" * 40)

        # Validation
        val_results_df = evaluate(model, val_loader, device)
        
        score_cols = [col for col in val_results_df.columns if 'MAE' not in col and isinstance(val_results_df[col].iloc[0], (int, float))]
        avg_val_score = 0
        if not val_results_df.empty and score_cols:
            avg_val_score = val_results_df[score_cols].mean().mean()

        print("\n--- Epoch {} Validation Report ---".format(epoch + 1))
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")

        writer.add_scalar("Val/AverageScore", avg_val_score, epoch + 1)
        if not val_results_df.empty:
            for _, row in val_results_df.iterrows():
                task_id = row['Task ID']
                task_name = row['Task Name']
                for col, value in row.items():
                    if col in ('Task ID', 'Task Name'):
                        continue
                    if isinstance(value, (int, float, np.floating)):
                        writer.add_scalar(f"Val/{col}/{task_name}_{task_id}", float(value), epoch + 1)

        val_log_lines = [f"--- Epoch {epoch + 1} Validation Report ---"]
        if not val_results_df.empty:
            val_table_str = val_results_df.to_string(index=False)
            val_log_lines.append(val_table_str)
        else:
            val_log_lines.append("No validation results")
        val_log_lines.append(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")

        epoch_log_block = "\n".join(train_log_lines + [""] + val_log_lines)
        log_file.write(epoch_log_block + "\n")
        log_file.flush()

        qualifies_for_topk = False
        if not val_results_df.empty:
            if len(best_checkpoint_records) < MAX_BEST_CHECKPOINTS:
                qualifies_for_topk = True
            elif avg_val_score > best_checkpoint_records[-1]['score']:
                qualifies_for_topk = True

        if qualifies_for_topk:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_filename = f"best_epoch{epoch + 1:04d}_score{avg_val_score:.4f}_{timestamp_str}.pth"
            checkpoint_path = os.path.join(weights_dir, checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_path)
            best_checkpoint_records.append({
                'path': checkpoint_path,
                'score': float(avg_val_score),
                'epoch': epoch + 1,
            })
            best_checkpoint_records.sort(key=lambda x: x['score'], reverse=True)

            removed_checkpoint = None
            while len(best_checkpoint_records) > MAX_BEST_CHECKPOINTS:
                removed_checkpoint = best_checkpoint_records.pop(-1)
                try:
                    os.remove(removed_checkpoint['path'])
                except OSError:
                    pass

            if avg_val_score >= best_val_score:
                best_val_score = avg_val_score
                current_top_checkpoint_path = checkpoint_path
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                best_flag = " (new overall best)"
            else:
                current_top_checkpoint_path = best_checkpoint_records[0]['path']
                best_flag = ""

            save_best_checkpoint_metadata(best_metadata_path, best_checkpoint_records, best_val_score)

            best_message = f"-> Saved ranked checkpoint at: {checkpoint_path} (score={avg_val_score:.4f}){best_flag}"
            if removed_checkpoint:
                best_message += f" | Pruned: {removed_checkpoint['path']}"
            print(best_message + "\n")
            log_file.write(best_message + "\n")
            log_file.flush()

        training_state = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_score': best_val_score,
            'run_dir': run_log_dir,
        }
        torch.save(training_state, latest_state_path)
        
        # Update scheduler
        scheduler.step()

    final_message = (
        f"\n--- Training Finished ---\n"
        f"Run directory: {run_log_dir}\n"
        f"Best checkpoints stored in: {weights_dir}\n"
        f"Top checkpoint: {current_top_checkpoint_path or 'N/A'}\n"
        f"Resume checkpoint: {latest_state_path if os.path.exists(latest_state_path) else 'N/A'}\n"
        f"Convenience copy: {MODEL_SAVE_PATH}"
    )
    print(final_message)
    log_file.write(f"[{datetime.now().isoformat()}] {final_message}\n")
    log_file.flush()
    writer.close()
    log_file.close()

if __name__ == '__main__':
    main()