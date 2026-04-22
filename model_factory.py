import os
import copy
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import List, Dict, Tuple, Optional

from transunet.vit_seg_modeling import (
    VisionTransformer,
    SegmentationHead as TransUNetSegmentationHead,
    DecoderCup,
    CONFIGS as VIT_CONFIGS,
)


def _resolve_transunet_name(name: str) -> str:
    normalized = name.replace('/', '_').replace('-', '_').replace(' ', '').lower()
    for key in VIT_CONFIGS.keys():
        key_normalized = key.replace('/', '_').replace('-', '_').replace(' ', '').lower()
        if normalized == key_normalized:
            return key
    raise ValueError(f"Unsupported TransUNet backbone identifier: {name}")

# Task configuration list
TASK_CONFIGURATIONS = [
    {'task_name': 'Regression', 'num_classes': 2, 'task_id': 'FUGC', 'tta_cfg': 'Flip'},
    {'task_name': 'Regression', 'num_classes': 3, 'task_id': 'IUGC', 'tta_cfg': 'Flip'},
    {'task_name': 'Regression', 'num_classes': 2, 'task_id': 'fetal_femur', 'tta_cfg': 'NoFlip'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'breast_2cls', 'tta_cfg': 'NoFlip'},
    {'task_name': 'classification', 'num_classes': 3, 'task_id': 'breast_3cls', 'tta_cfg': 'Flip'},
    {'task_name': 'classification', 'num_classes': 8, 'task_id': 'fetal_head_pos_cls', 'tta_cfg': 'Notta'},
    {'task_name': 'classification', 'num_classes': 6, 'task_id': 'fetal_plane_cls', 'tta_cfg': 'NoFlip'},
    {'task_name': 'classification', 'num_classes': 8, 'task_id': 'fetal_sacral_pos_cls', 'tta_cfg': 'Notta'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'liver_lesion_2cls', 'tta_cfg': 'Flip'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'lung_2cls', 'tta_cfg': 'Notta'},
    {'task_name': 'classification', 'num_classes': 3, 'task_id': 'lung_disease_3cls', 'tta_cfg': 'Flip'},
    {'task_name': 'classification', 'num_classes': 6, 'task_id': 'organ_cls', 'tta_cfg': 'Flip'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'spinal_cord_injury_loc', 'tta_cfg': 'Flip'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'thyroid_nodule_det', 'tta_cfg': 'Flip'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'uterine_fibroid_det', 'tta_cfg': 'Flip'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'breast_lesion', 'tta_cfg': 'Flip'},
    {'task_name': 'segmentation', 'num_classes': 4, 'task_id': 'cardiac_multi', 'tta_cfg': 'NoFlip'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'carotid_artery', 'tta_cfg': 'Flip'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'cervix', 'tta_cfg': 'Flip'},
    {'task_name': 'segmentation', 'num_classes': 3, 'task_id': 'cervix_multi', 'tta_cfg': 'NoFlip'},
    {'task_name': 'segmentation', 'num_classes': 5, 'task_id': 'fetal_abdomen_multi', 'tta_cfg': 'Flip'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'fetal_head', 'tta_cfg': 'Flip'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'fetal_heart', 'tta_cfg': 'Flip'},
    {'task_name': 'segmentation', 'num_classes': 3, 'task_id': 'head_symphysis_multi', 'tta_cfg': 'NoFlip'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'lung', 'tta_cfg': 'Flip'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'ovary_tumor', 'tta_cfg': 'Flip'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'thyroid_nodule', 'tta_cfg': 'Flip'},
]

# Task specific heads

class SmpClassificationHead(nn.Module):
    """Wrapper for SMP Classification Head."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = smp.base.ClassificationHead(
            in_channels=in_channels,
            classes=num_classes,
            pooling="avg",
            dropout=0.2,
            activation=None,
        )

    def forward(self, features: list):
        # Use the last feature map from encoder
        return self.head(features[-1])

class HeatmapRegressionHead(nn.Module):
    """Generates heatmaps for keypoint regression tasks."""

    def __init__(self, in_channels: int, num_points: int, heatmap_size: int):
        super().__init__()
        hidden = max(64, in_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, num_points, kernel_size=1)
        )
        self.upsample = nn.UpsamplingBilinear2d(size=(heatmap_size, heatmap_size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.conv(features)
        x = self.upsample(x)
        return x

class SingleBoxDetectionHead(nn.Module):
    """Regresses a single normalized bounding box from dense features."""

    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(128, in_channels // 2)
        self.feature_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
            nn.Sigmoid(),  # keep coords normalized to [0, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.feature_block(features)
        x = self.pool(x)
        return self.regressor(x)


class TransformerClassificationHead(nn.Module):
    """Classification head for transformer-based pooled features."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        hidden = max(256, in_features // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_features)


class TransUNetClassificationHead(nn.Module):
    """Classification head that consumes decoder feature maps."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        mid_channels = max(256, in_channels)
        out_channels = max(128, in_channels // 2)
        self.feature_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, num_classes)
        )

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        x = self.feature_block(decoder_features)
        x = self.pool(x)
        x = self.dropout(x)
        return self.classifier(x)


class ResNetStemClassificationHead(nn.Module):
    """Classification head that operates on the ResNet hybrid stem output."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        hidden = max(256, in_channels // 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, stem_features: torch.Tensor) -> torch.Tensor:
        x = self.pool(stem_features)
        return self.classifier(x)


class BottleneckAdapter(nn.Module):
    """Task-specific low-rank adapter injected after each transformer block."""

    def __init__(self, hidden_size: int, reduction: int = 4):
        super().__init__()
        bottleneck = max(1, hidden_size // reduction)
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return x


class ConvAdapter2d(nn.Module):
    """Residual bottleneck adapter applied to convolutional feature maps."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.down(x)
        delta = self.act(delta)
        delta = self.up(delta)
        delta = self.norm(delta)
        return x + delta


class MultiTaskTransUNet(VisionTransformer):
    """TransUNet backbone wrapper exposing shared features."""

    def __init__(
        self,
        backbone_name: str = 'R50-ViT-B_16',
        img_size: int = 256,
        pretrained: bool = True,
        pretrained_path: str = None,
        use_shared_decoders: bool = True,
        task_configs: Optional[List[Dict]] = None,
        use_task_adapters: bool = False,
        adapter_reduction: int = 4,
    ):
        resolved_name = _resolve_transunet_name(backbone_name)
        config = copy.deepcopy(VIT_CONFIGS[resolved_name])
        config.n_classes = 1
        super().__init__(config=config, img_size=img_size, num_classes=config.n_classes, vis=False)
        self.decoded_channels = config.decoder_channels[-1]
        self.hidden_size = config.hidden_size
        self.use_shared_decoders = use_shared_decoders
        self.use_task_adapters = use_task_adapters
        self.adapter_reduction = adapter_reduction
        self.cls_adapter_depth = 2
        self.cls_stem_adapters: Optional[nn.ModuleDict] = None
        hybrid_model = getattr(self.transformer.embeddings, 'hybrid_model', None)
        self.stem_channels = hybrid_model.width * 16 if hybrid_model is not None else None
        if use_shared_decoders:
            self.seg_decoder = self.decoder
            self.det_decoder = DecoderCup(config)
            self.reg_decoder = DecoderCup(config)
            self.cls_decoder = None
        else:
            self.seg_decoder = None
            self.det_decoder = None
            self.reg_decoder = None
            self.cls_decoder = None
            self.decoder = None
            self.segmentation_head = None
        if self.use_task_adapters:
            if not task_configs:
                raise ValueError("Task configs are required when task adapters are enabled.")
            self._build_classification_adapters(task_configs)
        ckpt_path = pretrained_path or getattr(config, 'pretrained_path', None)
        if pretrained and ckpt_path and os.path.exists(ckpt_path):
            weights = np.load(ckpt_path)
            self.load_from(weights)
            print(f"Loaded TransUNet pretrained weights from {ckpt_path}")
        elif pretrained:
            print(f"Warning: Pretrained weights not found at {ckpt_path}. Using random init.")

    def _build_classification_adapters(self, task_configs: List[Dict]):
        if self.stem_channels is None:
            raise ValueError("ResNet stem adapters require a hybrid (ResNet+ViT) backbone configuration.")
        adapters = nn.ModuleDict()
        for cfg in task_configs:
            if cfg['task_name'] != 'classification':
                continue
            adapter_layers = nn.ModuleList([
                ConvAdapter2d(self.stem_channels, reduction=self.adapter_reduction)
                for _ in range(self.cls_adapter_depth)
            ])
            adapters[cfg['task_id']] = adapter_layers
        if not adapters:
            raise ValueError("No classification tasks available to attach stem adapters.")
        self.cls_stem_adapters = adapters

    def encode(self, x: torch.Tensor, task_id: Optional[str] = None) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        stem_adapter = None
        if self.cls_stem_adapters is not None and task_id in self.cls_stem_adapters:
            stem_adapter = self.cls_stem_adapters[task_id]
        tokens, _, skip_features, stem_features = self.transformer(
            x,
            adapter_stack=None,
            stem_adapter_stack=stem_adapter,
        )
        return tokens, skip_features, stem_features

    def forward_features(self, x: torch.Tensor, task_id: Optional[str] = None) -> Dict[str, torch.Tensor]:
        if not self.use_shared_decoders:
            raise RuntimeError("Shared decoder features are disabled for this encoder.")
        tokens, skip_features, stem_features = self.encode(x, task_id=task_id)
        seg_decoded = self.seg_decoder(tokens, skip_features)
        det_decoded = self.det_decoder(tokens, skip_features)
        reg_decoded = self.reg_decoder(tokens, skip_features)
        pooled = tokens.mean(dim=1)
        return {
            'segmentation': seg_decoded,
            'detection': det_decoded,
            'regression': reg_decoded,
            'classification': stem_features,
            'pooled': pooled,
            'tokens': tokens,
            'skip_features': skip_features,
        }

# ====================================================================
# --- 2. Multi-Task Model Factory ---
# ====================================================================

class MultiTaskModelFactory(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str,
        task_configs: List[Dict],
        image_size: int = 256,
        regression_heatmap_size: int = 64,
        per_dataset_decoders: bool = False,
        use_task_adapters: bool = False,
        adapter_reduction: int = 4,
    ):
        super().__init__()

        self.backbone_type = 'transunet' if 'vit' in encoder_name.lower() else 'smp'
        self.regression_heatmap_size = regression_heatmap_size
        self.per_dataset_decoders = per_dataset_decoders and self.backbone_type == 'transunet'
        self.use_task_adapters = use_task_adapters and self.backbone_type == 'transunet'

        if self.backbone_type == 'transunet':
            print(f"Initializing TransUNet backbone: {encoder_name}")
            self.encoder = MultiTaskTransUNet(
                backbone_name=encoder_name,
                img_size=image_size,
                pretrained=True,
                pretrained_path=encoder_weights,
                use_shared_decoders=not self.per_dataset_decoders,
                task_configs=task_configs,
                use_task_adapters=self.use_task_adapters,
                adapter_reduction=adapter_reduction,
            )
            if self.per_dataset_decoders:
                self.task_decoders = nn.ModuleDict()
                for config in task_configs:
                    decoder = DecoderCup(copy.deepcopy(self.encoder.config))
                    self.task_decoders[config['task_id']] = decoder
        else:
            print(f"Initializing shared encoder: {encoder_name}")
            self.encoder = smp.encoders.get_encoder(
                name=encoder_name,
                in_channels=3,
                depth=5,
                weights=encoder_weights,
            )

            # Initialize shared FPN decoder
            temp_fpn_model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            )
            self.fpn_decoder = temp_fpn_model.decoder
        
        # Initialize task heads
        self.heads = nn.ModuleDict()
        
        print(f"Creating heads for {len(task_configs)} tasks...")
        for config in task_configs:
            task_id = config['task_id']
            task_name = config['task_name']
            num_classes = config['num_classes']
            
            head_module = None
            if task_name == 'segmentation':
                if self.backbone_type == 'transunet':
                    head_module = TransUNetSegmentationHead(
                        in_channels=self.encoder.decoded_channels,
                        out_channels=num_classes,
                        kernel_size=3,
                        upsampling=1,
                    )
                else:
                    head_module = smp.base.SegmentationHead(
                        in_channels=self.fpn_decoder.out_channels,
                        out_channels=num_classes,
                        kernel_size=1,
                        upsampling=4,
                    )

            elif task_name == 'classification':
                if self.backbone_type == 'transunet':
                    if getattr(self.encoder, 'stem_channels', None) is None:
                        raise ValueError("Hybrid ResNet stem is required for classification heads.")
                    head_module = ResNetStemClassificationHead(
                        in_channels=self.encoder.stem_channels,
                        num_classes=num_classes,
                    )
                else:
                    head_module = SmpClassificationHead(
                        in_channels=self.encoder.out_channels[-1],
                        num_classes=num_classes,
                    )

            elif task_name == 'Regression':
                num_points = config['num_classes']
                if self.backbone_type == 'transunet':
                    head_module = HeatmapRegressionHead(
                        in_channels=self.encoder.decoded_channels,
                        num_points=num_points,
                        heatmap_size=self.regression_heatmap_size,
                    )
                else:
                    head_module = HeatmapRegressionHead(
                        in_channels=self.fpn_decoder.out_channels,
                        num_points=num_points,
                        heatmap_size=self.regression_heatmap_size,
                    )

            elif task_name == 'detection':
                det_in_channels = self.encoder.decoded_channels if self.backbone_type == 'transunet' else self.fpn_decoder.out_channels
                head_module = SingleBoxDetectionHead(in_channels=det_in_channels)

            if head_module:
                self.heads[task_id] = head_module
            else:
                print(f"Warning: Unknown task type '{task_name}' for {task_id}")

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found.")

        task_config = next((item for item in TASK_CONFIGURATIONS if item["task_id"] == task_id), None)
        task_name = task_config['task_name'] if task_config else None

        if self.backbone_type == 'transunet':
            tokens, skip_features, stem_features = self.encoder.encode(x, task_id=task_id)
            pooled = tokens.mean(dim=1)
            if self.per_dataset_decoders:
                if task_name == 'classification':
                    if stem_features is None:
                        raise ValueError("Classification stem features are missing from encoder output.")
                    output = self.heads[task_id](stem_features)
                else:
                    if task_id not in self.task_decoders:
                        raise ValueError(f"Decoder for task '{task_id}' not found.")
                    decoded_features = self.task_decoders[task_id](tokens, skip_features)
                    output = self.heads[task_id](decoded_features)
            else:
                if task_name == 'segmentation' and self.encoder.seg_decoder is not None:
                    decoded = self.encoder.seg_decoder(tokens, skip_features)
                    output = self.heads[task_id](decoded)
                elif task_name == 'detection' and self.encoder.det_decoder is not None:
                    decoded = self.encoder.det_decoder(tokens, skip_features)
                    output = self.heads[task_id](decoded)
                elif task_name == 'Regression' and self.encoder.reg_decoder is not None:
                    decoded = self.encoder.reg_decoder(tokens, skip_features)
                    output = self.heads[task_id](decoded)
                elif task_name == 'classification':
                    if stem_features is None:
                        raise ValueError("Classification stem features are missing from encoder output.")
                    output = self.heads[task_id](stem_features)
                else:
                    output = self.heads[task_id](pooled)
        else:
            features = self.encoder(x)
            if task_name in ['segmentation', 'detection', 'Regression']:
                fpn_features = self.fpn_decoder(features)
                output = self.heads[task_id](fpn_features)
            else:
                output = self.heads[task_id](features)
        
        return output

    def load_state_dict(self, state_dict, strict: bool = True):
        adapter_bank = getattr(self.encoder, "cls_stem_adapters", None)
        if adapter_bank is not None:
            adapter_prefix = "encoder.cls_stem_adapters."
            current_state = self.state_dict()
            missing_keys = [
                key for key in current_state.keys()
                if key.startswith(adapter_prefix) and key not in state_dict
            ]
            if missing_keys:
                missing_preview = ", ".join(missing_keys[:3])
                if len(missing_keys) > 3:
                    missing_preview += ", ..."
                raise ValueError(
                    "Checkpoint is missing classification stem adapter parameters. "
                    "Please run 'upgrade_checkpoint_with_adapters.py' on legacy weights before loading. "
                    f"Missing keys (partial): {missing_preview}"
                )
        return super().load_state_dict(state_dict, strict=strict)

# Example usage

if __name__ == '__main__':
    model = MultiTaskModelFactory(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        task_configs=TASK_CONFIGURATIONS
    )

    print("\n--- Forward Pass Test ---")
    dummy_image_batch = torch.randn(2, 3, 256, 256) # Reduced batch size for test

    # Test specific tasks
    test_tasks = ['cardiac_multi', 'fetal_plane_cls', 'FUGC', 'thyroid_nodule_det']
    
    for t_id in test_tasks:
        try:
            out = model(dummy_image_batch, task_id=t_id)
            print(f"Task: {t_id:<25} | Output Shape: {out.shape}")
        except Exception as e:
            print(f"Task: {t_id:<25} | Error: {e}")
