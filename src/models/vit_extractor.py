import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights, MobileNet_V3_Small_Weights
import timm
from src.config import SPATIAL_FEATURE_DIM, DEVICE

if DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

class SpatialFeatureExtractor(nn.Module):
    """
    Multi-Mode Spatial Feature Extractor:
    1. 'vit' / 'torchvision_vit': Official Pretrained Vision Transformer (ViT-B/16 from torchvision).
    2. 'timm_vit': Official Pretrained Vision Transformer (vit_base_patch16_224 from timm repository).
    3. 'fast': Pretrained MobileNetV3-Small for ultra-fast CPU inference (< 0.5s).
    """
    def __init__(
        self,
        mode: str = "vit",
        pretrained: bool = True,
        feature_dim: int = SPATIAL_FEATURE_DIM
    ):
        super(SpatialFeatureExtractor, self).__init__()
        self.mode = mode.lower()
        self.feature_dim = feature_dim
        self.has_vit = False

        if "vit" in self.mode:
            # Try timm pretrained ViT if requested
            if self.mode == "timm_vit":
                try:
                    self.timm_model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
                    self.has_vit = True
                    self.vit_type = "timm"
                    print("[Info] Loaded Pretrained timm Vision Transformer (vit_base_patch16_224)")
                except Exception as e:
                    print(f"[Warning] Failed loading timm ViT ({e}). Falling back to torchvision ViT.")
                    self.mode = "vit"

            # Official torchvision Pretrained ViT-B/16
            if not self.has_vit:
                try:
                    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
                    vit = models.vit_b_16(weights=weights)
                    vit.heads = nn.Identity()
                    self.torchvision_vit = vit
                    self.has_vit = True
                    self.vit_type = "torchvision"
                    print("[Info] Loaded Pretrained Torchvision Vision Transformer (ViT-B/16)")
                except Exception as e:
                    print(f"[Warning] Could not load torchvision ViT weights ({e}). Falling back to fast MobileNet extractor.")
                    self.mode = "fast"

        if self.mode == "fast" or not self.has_vit:
            try:
                mnet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
                self.backbone = mnet.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc_proj = nn.Linear(576, feature_dim)
                self.is_mobilenet = True
            except Exception:
                self.is_mobilenet = False
                self.fallback_net = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, feature_dim)
                )

    def _extract_vit_single_chunk(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'vit_type') and self.vit_type == "timm":
            return self.timm_model(x)
        else:
            return self.torchvision_vit(x)


    def _extract_fast_single_chunk(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'is_mobilenet') and self.is_mobilenet:
            feats = self.backbone(x)
            pooled = self.pool(feats).flatten(1)
            return self.fc_proj(pooled)
        else:
            return self.fallback_net(x)

    def forward(self, x: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
        """
        Input: (B, 3, H, W) or (B, T, 3, H, W)
        Output: (B, feature_dim) or (B, T, feature_dim)
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            total_samples = B * T
            features_list = []

            for i in range(0, total_samples, chunk_size):
                sub_x = x_flat[i : i + chunk_size]
                if self.has_vit:
                    sub_feat = self._extract_vit_single_chunk(sub_x)
                else:
                    sub_feat = self._extract_fast_single_chunk(sub_x)
                features_list.append(sub_feat)

            feat_flat = torch.cat(features_list, dim=0)
            return feat_flat.view(B, T, self.feature_dim)
        else:
            total_samples = x.shape[0]
            features_list = []
            for i in range(0, total_samples, chunk_size):
                sub_x = x[i : i + chunk_size]
                if self.has_vit:
                    sub_feat = self._extract_vit_single_chunk(sub_x)
                else:
                    sub_feat = self._extract_fast_single_chunk(sub_x)
                features_list.append(sub_feat)
            return torch.cat(features_list, dim=0)

# Backward compatibility alias
ViTSpatialFeatureExtractor = SpatialFeatureExtractor
