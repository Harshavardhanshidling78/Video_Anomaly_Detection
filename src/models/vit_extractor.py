import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
from src.config import SPATIAL_FEATURE_DIM, DEVICE

if DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

class SpatialFeatureExtractor(nn.Module):
    """
    Vision Transformer (ViT) Spatial Feature Extractor:
    Official Pretrained Vision Transformer (ViT-B/16 from Torchvision).
    """
    def __init__(
        self,
        mode: str = "vit",
        pretrained: bool = True,
        feature_dim: int = SPATIAL_FEATURE_DIM
    ):
        super(SpatialFeatureExtractor, self).__init__()
        self.mode = "vit"
        self.feature_dim = feature_dim
        self.vit_type = "torchvision"

        # Official torchvision Pretrained ViT-B/16
        cache_file = os.path.expanduser("~/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth")
        if pretrained and os.path.exists(cache_file):
            try:
                with torch.device("meta"):
                    vit = models.vit_b_16(weights=None)
                sd = torch.load(cache_file, map_location="cpu", mmap=True, weights_only=True)
                vit.load_state_dict(sd, assign=True, strict=False)
                vit.heads = nn.Identity()
                del sd
                import gc
                gc.collect()
                self.torchvision_vit = vit
                print("[Info] Loaded Pretrained Torchvision ViT-B/16 (zero-copy mmap)")
            except Exception as e:
                print(f"[Warning] Fast mmap loading failed ({e}). Falling back to standard load.")
                vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
                vit.heads = nn.Identity()
                self.torchvision_vit = vit
        else:
            try:
                weights = ViT_B_16_Weights.DEFAULT if pretrained else None
                vit = models.vit_b_16(weights=weights)
                vit.heads = nn.Identity()
                self.torchvision_vit = vit
                print("[Info] Loaded Pretrained Torchvision Vision Transformer (ViT-B/16)")
            except Exception as e:
                print(f"[Warning] Could not load torchvision ViT weights ({e}). Initializing unweighted ViT-B/16.")
                import gc
                gc.collect()
                try:
                    vit = models.vit_b_16(weights=None)
                    vit.heads = nn.Identity()
                    self.torchvision_vit = vit
                except Exception as inner_e:
                    print(f"[Error] Failed to initialize ViT-B/16 due to memory constraints: {inner_e}")
                    raise RuntimeError(
                        f"System memory limit reached while initializing ViT-B/16: {inner_e}. "
                        "Please ensure sufficient RAM is available or close memory-heavy applications."
                    ) from inner_e

    def _extract_vit_single_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.torchvision_vit(x)

    def forward(self, x: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
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
                sub_feat = self._extract_vit_single_chunk(sub_x)
                features_list.append(sub_feat)

            feat_flat = torch.cat(features_list, dim=0)
            return feat_flat.view(B, T, self.feature_dim)
        else:
            total_samples = x.shape[0]
            features_list = []
            for i in range(0, total_samples, chunk_size):
                sub_x = x[i : i + chunk_size]
                sub_feat = self._extract_vit_single_chunk(sub_x)
                features_list.append(sub_feat)
            return torch.cat(features_list, dim=0)

# Backward compatibility alias
ViTSpatialFeatureExtractor = SpatialFeatureExtractor
