# Sample Code Submission for Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import json
import math
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from typing import Optional, List, Tuple, Set

# Import local modules
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import extract_coordinates

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TTAMode = Tuple[str, Optional[str]]


class InferenceDataset(Dataset):
    """Inference dataset class"""
    
    def __init__(self, data_root: str, transforms: Optional[A.Compose] = None):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.csv_path = os.path.join(self.data_root, 'csv_files')
        
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")
            
        all_csv_files = glob.glob(os.path.join(self.csv_path, '*.csv'))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")
            
        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        self.dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
        print(f"Data loading complete. Total samples: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record['task_id']
        task_name = record['task_name']
        
        # Load image
        image_rel_path = record['image_path']
        image_abs_path = os.path.normpath(os.path.join(self.csv_path, image_rel_path))
        image = cv2.imread(image_abs_path)
        
        if image is None:
            print(f"Warning: Unable to load image {image_abs_path}")
            # Return next sample
            return self.__getitem__((idx + 1) % len(self))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Get mask_path (if segmentation task)
        mask_path = None
        if task_name == 'segmentation' and 'mask_path' in record and pd.notna(record['mask_path']):
            mask_path = record['mask_path']
        
        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        # Return data including metadata
        return {
            'image': image,
            'task_id': task_id,
            'task_name': task_name,
            'image_path': image_rel_path,
            'mask_path': mask_path,
            'original_size': (original_height, original_width),
            'index': idx
        }


def inference_collate_fn(batch):
    """Inference collate function that preserves metadata"""
    images = torch.stack([item['image'] for item in batch], 0)
    task_ids = [item['task_id'] for item in batch]
    task_names = [item['task_name'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    mask_paths = [item['mask_path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    indices = [item['index'] for item in batch]
    
    return {
        'image': images,
        'task_id': task_ids,
        'task_name': task_names,
        'image_path': image_paths,
        'mask_path': mask_paths,
        'original_size': original_sizes,
        'index': indices
    }


class Model:
    """
    Foundation Model for Ultrasound Image Analysis
    Supports four task types: segmentation, classification, Regression, detection
    """
    
    def __init__(self):
        """Initialize model and load pretrained weights"""
        print("Initializing model...")
        
        # Set compute device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model variables, will be initialized in predict()
        self.model = None
        self.task_configs = None
        self.task_id_to_name = None
        self.regression_heatmap_size = 64
        self.input_image_size = 256
        self.reg_tta_outlier_px = 16.0
        self.det_tta_outlier_thresh = 0.05
        self.enable_tta = True
        self.disable_tta_task_ids: Set[str] = set()
        # Match the Stage-2 fine-tuning setup: each dataset/task owns its decoder
        self.per_dataset_decoders = True
        # Classification adapters on the ResNet stem (must match checkpoint)
        self.use_task_adapters = True
        self.adapter_reduction = 4
        self.mean_tensor = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.seg_tta_modes = self._build_segmentation_tta_modes()
        self.seg_tta_modes_noflip = self._remove_flip_modes(self.seg_tta_modes)
        self._set_classification_tta_modes(self._build_classification_tta_modes())
        self.reg_tta_modes = self._build_regression_tta_modes()
        self.reg_tta_modes_noflip = self._remove_flip_modes(self.reg_tta_modes)
        self.det_tta_modes = self._build_detection_tta_modes()
        self.det_tta_modes_noflip = self._remove_flip_modes(self.det_tta_modes)
        self.task_tta_cfg = {cfg['task_id']: cfg.get('tta_cfg', 'Flip') for cfg in TASK_CONFIGURATIONS}
        self.default_model_path = r'xxx.pth'
        
        # Define data preprocessing transforms (no augmentation for inference)
        self.transforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        
        print("Model initialization complete!\n")

    def _build_segmentation_tta_modes(self) -> List[TTAMode]:
        geometry_modes = ["identity", "hflip", "vflip"]
        color_modes: List[Optional[str]] = [None, "gamma_low", "gamma_high", "contrast_low", "contrast_high"]
        return [(geom, color) for geom in geometry_modes for color in color_modes]

    def _build_classification_tta_modes(self) -> List[TTAMode]:
        geometry_modes = ["identity", "hflip", "vflip", "rot+30", "rot-30"]
        color_modes: List[Optional[str]] = [None, "rand_brightness_contrast", "gaussian_noise"]
        return [(geom, color) for geom in geometry_modes for color in color_modes]

    def _build_regression_tta_modes(self) -> List[TTAMode]:
        geometry_modes = ["identity", "hflip", "vflip"]
        color_modes: List[Optional[str]] = [None, "gamma_high", "contrast_low", "contrast_high"]
        return [(geom, color) for geom in geometry_modes for color in color_modes]

    def _build_detection_tta_modes(self) -> List[TTAMode]:
        geometry_modes = ["identity", "hflip", "vflip"]
        color_modes: List[Optional[str]] = [None, "gamma_high", "contrast_low", "contrast_high"]
        return [(geom, color) for geom in geometry_modes for color in color_modes]

    def _remove_flip_modes(self, modes: List[TTAMode]) -> List[TTAMode]:
        filtered: List[TTAMode] = []
        for mode in modes:
            geom_mode, _ = self._split_mode(mode)
            if geom_mode not in ("hflip", "vflip"):
                filtered.append(mode)
        return filtered

    def _set_classification_tta_modes(self, modes: List[TTAMode]) -> None:
        self.cls_tta_modes = modes
        filtered = self._remove_flip_modes(modes)
        self.cls_tta_modes_noflip = filtered if filtered else [("identity", None)]

    def _get_task_tta_policy(self, task_id: str) -> str:
        return self.task_tta_cfg.get(task_id, "Flip")

    def _select_tta_modes(self, task_name: str, policy: str) -> List[TTAMode]:
        include_flips = policy != "NoFlip"
        if task_name == "segmentation":
            return self.seg_tta_modes if include_flips else self.seg_tta_modes_noflip
        if task_name == "classification":
            return self.cls_tta_modes if include_flips else self.cls_tta_modes_noflip
        if task_name == "Regression":
            return self.reg_tta_modes if include_flips else self.reg_tta_modes_noflip
        if task_name == "detection":
            return self.det_tta_modes if include_flips else self.det_tta_modes_noflip
        return self.cls_tta_modes if include_flips else self.cls_tta_modes_noflip

    def _split_mode(self, mode) -> TTAMode:
        if isinstance(mode, tuple):
            return mode
        return (str(mode), None)

    def _extract_rotation_angle(self, mode: str) -> float:
        if not mode.startswith("rot"):
            raise ValueError(f"Mode '{mode}' does not encode rotation")
        return float(mode[3:])

    def _rotate_tensor(self, tensor: torch.Tensor, angle: float, interpolation: InterpolationMode = InterpolationMode.BILINEAR) -> torch.Tensor:
        squeeze = False
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            squeeze = True
        rotated = torch.stack([
            TF.rotate(sample, angle, interpolation=interpolation, expand=False)
            for sample in tensor
        ], 0)
        return rotated.squeeze(0) if squeeze else rotated

    def _denormalize_tensor(self, images: torch.Tensor) -> torch.Tensor:
        mean = self.mean_tensor.to(images.device)
        std = self.std_tensor.to(images.device)
        return images * std + mean

    def _normalize_tensor(self, images: torch.Tensor) -> torch.Tensor:
        mean = self.mean_tensor.to(images.device)
        std = self.std_tensor.to(images.device)
        return (images - mean) / std

    def _apply_color_transform(self, images: torch.Tensor, mode: str) -> torch.Tensor:
        denorm = torch.clamp(self._denormalize_tensor(images), 0.0, 1.0)
        adjusted = []
        if mode == "gamma_low":
            gamma = 0.9
            for sample in denorm:
                adjusted.append(TF.adjust_gamma(sample, gamma))
        elif mode == "gamma_high":
            gamma = 1.3
            for sample in denorm:
                adjusted.append(TF.adjust_gamma(sample, gamma))
        elif mode == "contrast_low":
            factor = 0.9
            for sample in denorm:
                adjusted.append(TF.adjust_contrast(sample, factor))
        elif mode == "contrast_high":
            factor = 1.3
            for sample in denorm:
                adjusted.append(TF.adjust_contrast(sample, factor))
        elif mode == "rand_brightness_contrast":
            brightness = 1.1
            contrast = 1.1
            for sample in denorm:
                bc_sample = TF.adjust_brightness(sample, brightness)
                adjusted.append(TF.adjust_contrast(bc_sample, contrast))
        elif mode == "gaussian_noise":
            generator = torch.Generator(device=images.device)
            generator.manual_seed(2026)
            noise = torch.randn(denorm.shape, device=denorm.device, dtype=denorm.dtype, generator=generator)
            noisy = torch.clamp(denorm + noise * 0.05, 0.0, 1.0)
            return self._normalize_tensor(noisy)
        else:
            raise ValueError(f"Unsupported color mode: {mode}")
        stacked = torch.stack(adjusted, 0)
        return self._normalize_tensor(stacked)
    
    def predict(self, data_root: str, output_dir: str, batch_size: int = 8, use_tta: bool = True, checkpoint_path: Optional[str] = None, disable_cls_tta: bool = False, use_task_adapters: Optional[bool] = None, adapter_reduction: Optional[int] = None, per_dataset_decoders: Optional[bool] = None, disable_task_tta: Optional[List[str]] = None):
        """
        Perform prediction on input data
        
        Args:
            data_root: Data root directory containing csv_files subdirectory
            output_dir: Output results root directory
            batch_size: Batch size, default is 8
            use_tta: Whether to enable test-time augmentation
            checkpoint_path: Optional explicit checkpoint path; defaults to self.default_model_path
            disable_cls_tta: If True, classification uses identity-only (no TTA)
            use_task_adapters: Override whether to enable classification stem adapters (must match checkpoint)
            adapter_reduction: Override adapter reduction ratio (must match checkpoint)
            per_dataset_decoders: Override per-dataset decoder flag (must match checkpoint)
            disable_task_tta: Optional list of task_ids that should bypass TTA entirely
        
        Output:
            - Segmentation tasks: Save predicted masks as image files
            - Classification tasks: Save to classification_predictions.json
            - Detection tasks: Save to detection_predictions.json
            - Regression tasks: Save to regression_predictions.json
        """
        print(f"{'='*60}")
        print(f"Starting prediction...")
        print(f"Data directory: {data_root}")
        print(f"Output directory: {output_dir}")
        print(f"Batch size: {batch_size}")
        print(f"Checkpoint: {checkpoint_path or self.default_model_path}")
        print(f"Use task adapters: {use_task_adapters if use_task_adapters is not None else self.use_task_adapters}")
        print(f"Adapter reduction: {adapter_reduction if adapter_reduction is not None else self.adapter_reduction}")
        print(f"Per-dataset decoders: {per_dataset_decoders if per_dataset_decoders is not None else self.per_dataset_decoders}")
        print(f"{'='*60}\n")
        
        self.enable_tta = use_tta
        self.disable_tta_task_ids = set(disable_task_tta or [])
        if disable_cls_tta:
            self._set_classification_tta_modes([("identity", None)])
        if use_task_adapters is not None:
            self.use_task_adapters = use_task_adapters
        if adapter_reduction is not None:
            self.adapter_reduction = adapter_reduction
        if per_dataset_decoders is not None:
            self.per_dataset_decoders = per_dataset_decoders
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        print(f"Loading dataset...")
        dataset = InferenceDataset(data_root=data_root, transforms=self.transforms)
        
        # Build task_configs from dataset (dynamic construction)
        print(f"\nBuilding task configurations from dataset...")
        self.task_configs = []
        task_config_map = {}
        
        for _, row in dataset.dataframe.iterrows():
            task_id = row['task_id']
            if task_id not in task_config_map:
                task_config = {
                    'task_id': task_id,
                    'task_name': row['task_name'],
                        'num_classes': int(row['num_classes']),
                        'tta_cfg': self.task_tta_cfg.get(task_id, 'Flip')
                }
                task_config_map[task_id] = task_config
                self.task_configs.append(task_config)
        
        self.task_tta_cfg = {cfg['task_id']: cfg.get('tta_cfg', 'Flip') for cfg in self.task_configs}
        print(f"Detected {len(self.task_configs)} task configurations")
        for cfg in sorted(self.task_configs, key=lambda x: x['task_id']):
            print(f"  - {cfg['task_id']}: {cfg['task_name']}, num_classes={cfg['num_classes']}")
        
        # Build task_id to task_name mapping
        self.task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in self.task_configs}
        
        # Create and load model
        print(f"\nLoading model...")
        self.model = MultiTaskModelFactory(
            encoder_name='R50-ViT-B_16',
            encoder_weights=None,
            task_configs=self.task_configs,
            regression_heatmap_size=self.regression_heatmap_size,
            per_dataset_decoders=self.per_dataset_decoders,
            use_task_adapters=self.use_task_adapters,
            adapter_reduction=self.adapter_reduction,
        ).to(self.device)
        
        # Load trained model weights
        model_path = checkpoint_path or self.default_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("Model weights loaded successfully!")
        
        # Create data loader (batch processing for faster inference)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=inference_collate_fn
        )
        
        # Batch inference
        print(f"\nStarting inference...")
        classification_results = []
        detection_results = []
        regression_results = []
        task_counts = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Prediction progress"):
                images = batch['image'].to(self.device)
                task_ids = batch['task_id']
                task_names = batch['task_name']
                image_paths = batch['image_path']
                mask_paths = batch['mask_path']
                original_sizes = batch['original_size']
                
                # Process each task in batch
                unique_tasks = list(set(task_ids))
                
                for task_id in unique_tasks:
                    # Get indices for all samples of current task
                    task_indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
                    task_images = images[task_indices]
                    task_name = task_names[task_indices[0]]

                    # Model inference
                    outputs = self._run_model_with_tta(task_images, task_id=task_id, task_name=task_name)
                    
                    # Save prediction results for each sample
                    for i, batch_idx in enumerate(task_indices):
                        pred = outputs[i]
                        image_path = image_paths[batch_idx]
                        mask_path = mask_paths[batch_idx]
                        original_size = original_sizes[batch_idx]
                        
                        # Statistics
                        task_counts[task_id] = task_counts.get(task_id, 0) + 1
                        
                        # Process results by task type
                        if task_name == 'segmentation':
                            self._save_segmentation(pred, image_path, mask_path, output_dir, original_size)
                        
                        elif task_name == 'classification':
                            result = self._process_classification(pred, task_id, image_path)
                            classification_results.append(result)
                        
                        elif task_name == 'Regression':
                            result = self._process_regression(pred, task_id, image_path, original_size)
                            regression_results.append(result)
                        
                        elif task_name == 'detection':
                            result = self._process_detection(pred, task_id, image_path, original_size)
                            detection_results.append(result)
        
        # Save aggregated JSON results
        print("\nSaving prediction results...")
        
        if classification_results:
            json_path = os.path.join(output_dir, 'classification_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(classification_results, f, indent=2, ensure_ascii=False)
            print(f"  - Classification results: {json_path} ({len(classification_results)} samples)")
        
        if detection_results:
            json_path = os.path.join(output_dir, 'detection_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detection_results, f, indent=2, ensure_ascii=False)
            print(f"  - Detection results: {json_path} ({len(detection_results)} samples)")
        
        if regression_results:
            json_path = os.path.join(output_dir, 'regression_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(regression_results, f, indent=2, ensure_ascii=False)
            print(f"  - Regression results: {json_path} ({len(regression_results)} samples)")
        
        # Print statistics
        print(f"\n{'='*60}")
        print("Prediction complete!")
        print(f"{'='*60}")
        print("\nPrediction count by task:")
        for task_id in sorted(task_counts.keys()):
            task_name_str = self.task_id_to_name.get(task_id, 'unknown')
            count = task_counts[task_id]
            print(f"  - {task_id:<25} ({task_name_str:<15}): {count:>5} samples")
        print(f"\nTotal: {sum(task_counts.values())} samples")
        print()

    def _apply_geometric_transform(self, images: torch.Tensor, geom_mode: str) -> torch.Tensor:
        if geom_mode == "identity":
            return images
        if geom_mode == "hflip":
            return torch.flip(images, dims=[3])
        if geom_mode == "vflip":
            return torch.flip(images, dims=[2])
        if geom_mode.startswith("rot"):
            angle = self._extract_rotation_angle(geom_mode)
            return self._rotate_tensor(images, angle, interpolation=InterpolationMode.BILINEAR)
        raise ValueError(f"Unsupported geometric TTA mode: {geom_mode}")

    def _apply_tta_transform(self, images: torch.Tensor, mode: TTAMode) -> torch.Tensor:
        geom_mode, color_mode = self._split_mode(mode)
        transformed = self._apply_geometric_transform(images, geom_mode)
        if color_mode:
            transformed = self._apply_color_transform(transformed, color_mode)
        return transformed

    def _invert_spatial_tensor(self, tensor: torch.Tensor, mode) -> torch.Tensor:
        geom_mode, _ = self._split_mode(mode)
        dims = []
        if geom_mode == "hflip":
            dims.append(-1)
        if geom_mode == "vflip":
            dims.append(-2)
        if dims:
            return torch.flip(tensor, dims=dims)
        if geom_mode.startswith("rot"):
            angle = -self._extract_rotation_angle(geom_mode)
            interpolation = InterpolationMode.BILINEAR
            return self._rotate_tensor(tensor, angle, interpolation=interpolation)
        return tensor

    def _invert_detection_boxes(self, boxes: torch.Tensor, mode) -> torch.Tensor:
        geom_mode, _ = self._split_mode(mode)
        if geom_mode == "identity":
            return boxes

        squeeze = False
        if boxes.dim() == 1:
            boxes = boxes.unsqueeze(0)
            squeeze = True

        boxes = boxes.clone()
        if geom_mode == "hflip":
            x1 = 1.0 - boxes[:, 2]
            x2 = 1.0 - boxes[:, 0]
            boxes[:, 0] = x1
            boxes[:, 2] = x2
        elif geom_mode == "vflip":
            y1 = 1.0 - boxes[:, 3]
            y2 = 1.0 - boxes[:, 1]
            boxes[:, 1] = y1
            boxes[:, 3] = y2
        return boxes.squeeze(0) if squeeze else boxes

    def _invert_tta_predictions(self, predictions: torch.Tensor, task_name: str, mode) -> torch.Tensor:
        geom_mode, _ = self._split_mode(mode)
        if geom_mode == "identity":
            return predictions
        if task_name in ("segmentation", "Regression"):
            return self._invert_spatial_tensor(predictions, (geom_mode, None))
        if task_name == "detection":
            if predictions.dim() >= 3:
                return self._invert_spatial_tensor(predictions, (geom_mode, None))
            return self._invert_detection_boxes(predictions, (geom_mode, None))
        return predictions

    def _run_model_with_tta(self, images: torch.Tensor, task_id: str, task_name: str) -> torch.Tensor:
        tta_policy = self._get_task_tta_policy(task_id)
        policy_allows_tta = tta_policy != "Notta"
        use_tta = self.enable_tta and task_id not in self.disable_tta_task_ids and policy_allows_tta

        if task_name == "segmentation":
            if not use_tta:
                return self.model(images, task_id=task_id)
            modes = self._select_tta_modes(task_name, tta_policy)
            aggregate = None
            for mode in modes:
                augmented = self._apply_tta_transform(images, mode)
                preds = self.model(augmented, task_id=task_id)
                preds = self._invert_tta_predictions(preds, task_name, mode)
                preds = preds.float()
                aggregate = preds if aggregate is None else aggregate + preds
            return aggregate / len(modes)

        if task_name == "classification":
            if not use_tta:
                return self.model(images, task_id=task_id)
            modes = self._select_tta_modes(task_name, tta_policy)
            aggregate = None
            for mode in modes:
                augmented = self._apply_tta_transform(images, mode)
                preds = self.model(augmented, task_id=task_id)
                preds = preds.float()
                aggregate = preds if aggregate is None else aggregate + preds
            return aggregate / len(modes)

        if task_name == "Regression":
            if not use_tta:
                preds = self.model(images, task_id=task_id)
                preds = preds.float()
                return extract_coordinates(preds)
            coord_predictions = []
            modes = self._select_tta_modes(task_name, tta_policy)
            for mode in modes:
                augmented = self._apply_tta_transform(images, mode)
                preds = self.model(augmented, task_id=task_id)
                preds = self._invert_tta_predictions(preds, task_name, mode)
                coords = extract_coordinates(preds.float())
                coord_predictions.append(coords)
            stacked = torch.stack(coord_predictions, dim=0)
            median = stacked.median(dim=0).values
            num_points = median.shape[1] // 2
            pairs = stacked.view(stacked.size(0), stacked.size(1), num_points, 2)
            median_pairs = median.view(1, median.size(0), num_points, 2)
            diff = pairs - median_pairs
            dist = torch.sqrt((diff ** 2).sum(dim=-1))
            threshold = (self.reg_tta_outlier_px / float(self.input_image_size))
            valid_pairs = dist <= threshold
            valid_mask = valid_pairs.unsqueeze(-1).expand(-1, -1, -1, 2).reshape_as(stacked)
            valid_mask_float = valid_mask.float()
            valid_counts = valid_mask_float.sum(dim=0)
            weighted_sum = (stacked * valid_mask_float).sum(dim=0)
            result = torch.empty_like(median)
            nonzero = valid_counts > 0
            if nonzero.any():
                result[nonzero] = weighted_sum[nonzero] / valid_counts[nonzero]
            if (~nonzero).any():
                result[~nonzero] = median[~nonzero]
            return result

        if task_name == "detection":
            if not use_tta:
                preds = self.model(images, task_id=task_id)
                return preds.float()
            modes = self._select_tta_modes(task_name, tta_policy)
            stacked_preds: List[torch.Tensor] = []
            for mode in modes:
                augmented = self._apply_tta_transform(images, mode)
                preds = self.model(augmented, task_id=task_id)
                preds = self._invert_tta_predictions(preds, task_name, mode)
                stacked_preds.append(preds.float())

            stacked = torch.stack(stacked_preds, dim=0)
            median = stacked.median(dim=0).values
            flat_diff = torch.abs(stacked - median.unsqueeze(0)).view(stacked.size(0), stacked.size(1), -1)
            deviations = flat_diff.sum(dim=-1)
            mask = deviations <= self.det_tta_outlier_thresh
            if mask.any():
                expand_shape = mask.shape + (1,) * (stacked.dim() - 2)
                mask_float = mask.float().view(*expand_shape)
                weighted_sum = (stacked * mask_float).sum(dim=0)
                counts = mask_float.sum(dim=0)
                avg = weighted_sum / counts.clamp_min(1.0)
                return torch.where(counts > 0, avg, median)
            return median

        return self.model(images, task_id=task_id)
    
    def _save_segmentation(self, pred, image_path, mask_path, output_dir, original_size):
        """
        Save segmentation prediction results as image file
        
        Args:
            pred: Prediction result (C, H, W) or (H, W)
            image_path: Image path
            mask_path: Mask path (from CSV)
            output_dir: Output root directory
            original_size: Original image size (height, width)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # For multi-class segmentation (C, H, W), take argmax
        if pred.ndim == 3:
            mask = np.argmax(pred, axis=0).astype(np.uint8)
        else:
            mask = pred.astype(np.uint8)
        
        # Resize back to original size
        h, w = original_size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Determine output path
        if mask_path:
            # Use mask path specified in CSV, remove leading '../'
            mask_path_clean = mask_path.replace('../', '')
            output_path = os.path.join(output_dir, mask_path_clean)
        else:
            # Default: replace keywords in image_path
            default_mask_path = image_path.replace('img', 'mask').replace('IMG', 'MASK')
            output_path = os.path.join(output_dir, default_mask_path)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save mask image
        cv2.imwrite(output_path, mask)
    
    def _process_classification(self, pred, task_id, image_path):
        """
        Process classification task prediction results
        
        Args:
            pred: Prediction logits (num_classes,)
            task_id: Task ID
            image_path: Image path
            
        Returns:
            Dictionary containing prediction results
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # Get predicted class
        pred_class = int(np.argmax(pred))
        
        # Calculate probability distribution (using softmax)
        # Stable softmax to avoid numerical overflow
        pred_exp = np.exp(pred - np.max(pred))
        pred_probs = pred_exp / np.sum(pred_exp)
        
        return {
            'image_path': image_path,
            'task_id': task_id,
            'predicted_class': pred_class,
            'predicted_probs': pred_probs.tolist()
        }
    
    def _process_regression(self, pred, task_id, image_path, original_size):
        """
        Process regression task prediction results (keypoint localization)
        
        Args:
            pred: Predicted coordinates (num_points * 2,) - normalized coordinates
            task_id: Task ID
            image_path: Image path
            original_size: Original image size (height, width)
            
        Returns:
            Dictionary containing prediction results
        """
        if isinstance(pred, torch.Tensor):
            if pred.dim() <= 2:
                coords_tensor = pred.unsqueeze(0) if pred.dim() == 1 else pred
            else:
                pred_tensor = pred.unsqueeze(0) if pred.dim() == 3 else pred
                coords_tensor = extract_coordinates(pred_tensor)
            coords = coords_tensor.squeeze(0).cpu().numpy().tolist()
        else:
            coords = pred.flatten().tolist()
        
        # Convert to pixel coordinates
        h, w = original_size
        pixel_coords = []
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i+1]
            x_pixel = x_norm * w
            y_pixel = y_norm * h
            pixel_coords.extend([x_pixel, y_pixel])
        
        return {
            'image_path': image_path,
            'task_id': task_id,
            'predicted_points_normalized': coords,
            'predicted_points_pixels': pixel_coords
        }
    
    def _process_detection(self, pred, task_id, image_path, original_size):
        """Process detection predictions from either head format."""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()

        bbox_norm = None
        confidence = None

        if pred.ndim == 1 and pred.shape[0] == 4:
            # Current single-box head outputs 4 normalized coords
            bbox_norm = pred
        elif pred.ndim == 2 and 4 in pred.shape:
            # Gracefully handle potential (4, N) or (N, 4) tensors
            if pred.shape[0] == 4:
                bbox_norm = pred[:, 0]
            else:
                bbox_norm = pred[0]
        elif pred.ndim == 3 and pred.shape[0] == 5:
            # Legacy dense grid head: pick the highest-confidence cell
            scores = pred[4].reshape(-1)
            best_idx = int(np.argmax(scores))
            h = pred.shape[1]
            w = pred.shape[2]
            row = best_idx // w
            col = best_idx % w
            bbox_norm = pred[:4, row, col]
            confidence = float(scores[best_idx])
        else:
            raise ValueError(f"Unsupported detection prediction shape: {pred.shape}")

        bbox_norm = np.asarray(bbox_norm).flatten()
        img_h, img_w = original_size
        bbox_pixel = [
            bbox_norm[0] * img_w,
            bbox_norm[1] * img_h,
            bbox_norm[2] * img_w,
            bbox_norm[3] * img_h,
        ]

        result = {
            'image_path': image_path,
            'task_id': task_id,
            'bbox_normalized': bbox_norm.tolist(),
            'bbox_pixels': bbox_pixel,
        }
        if confidence is not None:
            result['confidence'] = confidence
        return result


if __name__ == '__main__':
    """
    Usage example
    """
    # Set paths
    data_root = r'xxx'
    output_dir = r'xxx'
    batch_size = 64
    
    # Create model and perform prediction
    model = Model()
    model.predict(
        data_root, 
        output_dir, 
        use_tta=True, 
        batch_size=batch_size, 
        disable_cls_tta=False,
        use_task_adapters=True, 
        per_dataset_decoders=True,
    )
    
    print("Inference complete!")

