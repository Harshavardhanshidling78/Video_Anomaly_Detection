import torch
import torch.nn as nn
from typing import Dict, Tuple
from src.models.vit_extractor import SpatialFeatureExtractor
from src.models.temporal_model import RecurrentTemporalModel
from src.config import (
    SPATIAL_FEATURE_DIM,
    TEMPORAL_HIDDEN_DIM,
    TEMPORAL_NUM_LAYERS,
    RECURRENT_TYPE,
    ANOMALY_THRESHOLD,
    DEVICE
)

class VideoAnomalyDetectorModel(nn.Module):
    """
    End-to-End Deep Learning Architecture for Video Anomaly Detection:
    1. Spatial Extractor: ViT-B/16 or Fast MobileNetV3 (B, T, 768)
    2. Feature Norm: LayerNorm(768) for batch-independent scale stabilization
    3. LSTM/GRU: Analyzes temporal relationships across consecutive frames (B, T, 768 -> B, 256)
    4. Classification Head: LayerNorm FC + Sigmoid producing Anomaly Score [0.0 - 1.0]
    """
    def __init__(
        self,
        spatial_mode: str = "fast",
        pretrained_vit: bool = True,
        cell_type: str = RECURRENT_TYPE,
        hidden_dim: int = TEMPORAL_HIDDEN_DIM,
        threshold: float = ANOMALY_THRESHOLD
    ):
        super(VideoAnomalyDetectorModel, self).__init__()
        self.spatial_mode = spatial_mode
        self.cell_type = cell_type
        self.threshold = threshold
        self.spatial_extractor = SpatialFeatureExtractor(mode=spatial_mode, pretrained=pretrained_vit)
        self.feature_norm = nn.LayerNorm(SPATIAL_FEATURE_DIM)
        self.temporal_encoder = RecurrentTemporalModel(
            input_dim=SPATIAL_FEATURE_DIM,
            hidden_dim=hidden_dim,
            cell_type=cell_type
        )
        
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Set final classification layer bias negative to prevent false positive bias
        final_layer = self.classifier_head[-2]
        if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, -1.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 5:
            spatial_features = self.spatial_extractor(x)
        else:
            spatial_features = x

        spatial_features = self.feature_norm(spatial_features)
        temporal_representation = self.temporal_encoder(spatial_features)
        anomaly_score = self.classifier_head(temporal_representation)
        binary_prediction = (anomaly_score >= self.threshold).float()

        return anomaly_score, binary_prediction


    @torch.inference_mode()
    def get_anomaly_score(self, clip_tensor: torch.Tensor) -> float:
        self.eval()
        if clip_tensor.dim() == 4:
            clip_tensor = clip_tensor.unsqueeze(0)
        score, _ = self.forward(clip_tensor.to(DEVICE))
        return float(score.item())

    @torch.inference_mode()
    def predict_batch(self, clips_batch: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
        self.eval()
        N = clips_batch.shape[0]
        if N == 0:
            return torch.empty(0, device=DEVICE)

        scores_list = []
        for i in range(0, N, batch_size):
            chunk = clips_batch[i : i + batch_size].to(DEVICE)
            score_chunk, _ = self.forward(chunk)
            scores_list.append(score_chunk.squeeze(-1))

        return torch.cat(scores_list, dim=0)
