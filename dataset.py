import os
import cv2
import pandas as pd
import numpy as np
import torch
import glob
import random
import json
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from typing import Optional, Iterator, List, Dict
import albumentations as A

class MultiTaskDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        transforms: Optional[A.Compose] = None,
        task_transforms: Optional[Dict[str, A.Compose]] = None,
        regression_heatmap_size: int = 64,
        regression_heatmap_sigma: float = 4,
        split: Optional[str] = None,
        allowed_task_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.task_transforms = task_transforms or {}
        self.regression_heatmap_size = regression_heatmap_size
        self.regression_heatmap_sigma = regression_heatmap_sigma
        self.split = split.lower() if split else None
        self.allowed_task_names = [name.lower() for name in allowed_task_names] if allowed_task_names else None
        if self.split and self.split not in {'train', 'val', 'validation'}:
            raise ValueError("Split must be 'train', 'val', or 'validation'.")
        self.csv_path = os.path.join(self.data_root, 'csv_files')
        
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")
            
        all_csv_files = glob.glob(os.path.join(self.csv_path, '*.csv'))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")
            
        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        combined_df = pd.concat(df_list, ignore_index=True).reset_index(drop=True)

        if self.split:
            if 'train' not in combined_df.columns:
                raise ValueError(
                    "CSV files must contain a 'train' column (1=train, 0=val) when using split filtering."
                )

            normalized_split = 'train' if self.split == 'train' else 'val'
            train_flags = pd.to_numeric(combined_df['train'], errors='coerce')
            if normalized_split == 'train':
                mask = train_flags == 1
            else:
                mask = train_flags == 0

            mask = mask.fillna(False)
            filtered_df = combined_df[mask]
            if filtered_df.empty:
                raise ValueError(
                    f"No samples found for split '{self.split}' in {self.csv_path}."
                )

            self.dataframe = filtered_df.reset_index(drop=True)
            split_name = 'Training' if normalized_split == 'train' else 'Validation'
            print(f"Data loaded. {split_name} split samples: {len(self.dataframe)}")
        else:
            self.dataframe = combined_df
            print(f"Data loaded. Total samples: {len(self.dataframe)}")

        if self.allowed_task_names:
            filtered_df = self.dataframe[self.dataframe['task_name'].str.lower().isin(self.allowed_task_names)]
            if filtered_df.empty:
                raise ValueError(
                    f"No samples found for allowed task names: {allowed_task_names}."
                )
            self.dataframe = filtered_df.reset_index(drop=True)
            print(
                f"Task filter applied ({allowed_task_names}). Remaining samples: {len(self.dataframe)}"
            )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record['task_id']
        task_name = record['task_name']
        
        # Load image
        image_abs_path = os.path.normpath(os.path.join(self.csv_path, record['image_path']))
        image = cv2.imread(image_abs_path)
        
        # Robustness check: retry next index if image load fails
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save original image size BEFORE any transforms (for Regression coordinate normalization)
        original_height, original_width = image.shape[:2]

        # Load raw labels based on task
        label = None
        mask = None
        bboxes = []
        class_labels = []
        reg_coords_tensor = None
        regression_keypoints = None

        if task_name == 'segmentation':
            if pd.notna(record.get('mask_path')):
                mask_path = os.path.normpath(os.path.join(self.csv_path, record['mask_path']))
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        elif task_name == 'classification':
            label = int(record['mask'])

        elif task_name == 'Regression':
            num_points = int(record['num_classes'])
            coords = []
            regression_keypoints = []
            for i in range(1, num_points + 1):
                col = f'point_{i}_xy'
                if col in record and pd.notna(record[col]):
                    pt = json.loads(record[col])
                else:
                    pt = [0, 0]
                coords.extend(pt)
                regression_keypoints.append((float(pt[0]), float(pt[1])))
            label = np.array(coords, dtype=np.float32)

        elif task_name == 'detection':
            cols = ['x_min', 'y_min', 'x_max', 'y_max']
            if all(c in record and pd.notna(record[c]) for c in cols):
                box_coords = [float(record[c]) for c in cols]
                bboxes = [box_coords + [0]]
                class_labels = [0]

        # Apply augmentations
        transform = self.task_transforms.get(task_name, self.transforms)
        if transform:
            if task_name == 'segmentation':
                mask_to_use = mask if mask is not None else np.zeros_like(image[:, :, 0])
                augmented = transform(image=image, mask=mask_to_use)
                image = augmented['image']
                label = augmented.get('mask')
            elif task_name == 'detection':
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                image = augmented['image']
                if augmented['bboxes']:
                    label = np.array(augmented['bboxes'][0][:4], dtype=np.float32)
                else:
                    label = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            elif task_name == 'Regression':
                regression_keypoints = regression_keypoints or []
                augmented = transform(image=image, keypoints=regression_keypoints)
                image = augmented['image']
                transformed_keypoints = augmented.get('keypoints', regression_keypoints)
                if not transformed_keypoints:
                    transformed_keypoints = regression_keypoints
                flat_coords = []
                for kp in transformed_keypoints:
                    flat_coords.extend(kp)
                label = np.array(flat_coords, dtype=np.float32)
            else:
                augmented = transform(image=image)
                image = augmented['image']

        # Format conversion & normalization
        final_label = None
        h, w = image.shape[1], image.shape[2]

        # Ensure label is numpy for processing
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        if task_name == 'segmentation':
            if label is None:
                label = np.zeros((h, w), dtype=np.int64)
            final_label = torch.from_numpy(label).long()

        elif task_name == 'classification':
            final_label = torch.tensor(label).long()

        elif task_name in ['Regression', 'detection']:
            if not isinstance(label, np.ndarray):
                label = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

            # Normalize coordinates to [0, 1]
            if task_name == 'detection' and np.all(label >= 0):
                label[[0, 2]] /= w
                label[[1, 3]] /= h
            elif task_name == 'Regression':
                label[0::2] /= max(w, 1e-6)
                label[1::2] /= max(h, 1e-6)
                heatmaps = self._build_regression_heatmaps(label)
                final_label = torch.from_numpy(heatmaps).float()
                reg_coords_tensor = torch.from_numpy(label).float()
            
            if final_label is None:
                final_label = torch.from_numpy(label).float()
        
        sample = {'image': image, 'label': final_label, 'task_id': task_id}
        if reg_coords_tensor is not None:
            sample['reg_coords'] = reg_coords_tensor
        return sample

    def _build_regression_heatmaps(self, normalized_coords: np.ndarray) -> np.ndarray:
        num_points = normalized_coords.shape[0] // 2
        heatmaps = np.zeros((num_points, self.regression_heatmap_size, self.regression_heatmap_size), dtype=np.float32)
        for idx in range(num_points):
            x_norm = np.clip(normalized_coords[2 * idx], 0.0, 1.0)
            y_norm = np.clip(normalized_coords[2 * idx + 1], 0.0, 1.0)
            center_x = x_norm * (self.regression_heatmap_size - 1)
            center_y = y_norm * (self.regression_heatmap_size - 1)
            heatmaps[idx] = self._generate_gaussian_heatmap(center_x, center_y)
        return heatmaps

    def _generate_gaussian_heatmap(self, center_x: float, center_y: float) -> np.ndarray:
        size = self.regression_heatmap_size
        xs = np.arange(0, size, 1, dtype=np.float32)
        ys = np.arange(0, size, 1, dtype=np.float32)[:, None]
        heatmap = np.exp(-((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2 * self.regression_heatmap_sigma ** 2))
        return heatmap


class MultiTaskUniformSampler(Sampler[List[int]]):
    def __init__(self, dataset: MultiTaskDataset, batch_size: int, steps_per_epoch: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_by_task = {}

        # Group indices by task_id
        print("\n--- Initializing Sampler ---")
        for idx, task_id in enumerate(tqdm(dataset.dataframe['task_id'], desc="Grouping indices")):
            if task_id not in self.indices_by_task:
                self.indices_by_task[task_id] = []
            self.indices_by_task[task_id].append(idx)
            
        self.task_ids = list(self.indices_by_task.keys())
        
        # Initial shuffle
        for task_id in self.task_ids:
            random.shuffle(self.indices_by_task[task_id])

        # Determine epoch length
        if steps_per_epoch is None:
            self.steps_per_epoch = len(self.dataset) // self.batch_size
        else:
            self.steps_per_epoch = steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        task_cursors = {task_id: 0 for task_id in self.task_ids}

        for _ in range(self.steps_per_epoch):
            # Randomly select a task
            task_id = random.choice(self.task_ids)
            indices = self.indices_by_task[task_id]
            cursor = task_cursors[task_id]
            
            start_idx = cursor
            end_idx = start_idx + self.batch_size
            
            if end_idx > len(indices):
                # Wrap around
                batch_indices = indices[start_idx:]
                random.shuffle(indices)
                remaining = self.batch_size - len(batch_indices)
                batch_indices.extend(indices[:remaining])
                task_cursors[task_id] = remaining
            else:
                batch_indices = indices[start_idx:end_idx]
                task_cursors[task_id] = end_idx
            
            yield batch_indices
            
    def __len__(self) -> int:
        return self.steps_per_epoch