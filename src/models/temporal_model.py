import torch
import torch.nn as nn
from src.config import SPATIAL_FEATURE_DIM, TEMPORAL_HIDDEN_DIM, TEMPORAL_NUM_LAYERS, RECURRENT_TYPE

class RecurrentTemporalModel(nn.Module):
    """
    LSTM or GRU network for temporal sequence analysis of video frame embeddings.
    Captures temporal progression and temporal relationships across frames.
    """
    def __init__(
        self,
        input_dim: int = SPATIAL_FEATURE_DIM,
        hidden_dim: int = TEMPORAL_HIDDEN_DIM,
        num_layers: int = TEMPORAL_NUM_LAYERS,
        cell_type: str = RECURRENT_TYPE,
        dropout: float = 0.2
    ):
        super(RecurrentTemporalModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.upper()

        if self.cell_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False
            )
        else: # Default LSTM
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False
            )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Sequence of spatial features (B, T, D)
        Output: Aggregated temporal context vector (B, hidden_dim)
        """
        # rnn_out shape: (B, T, hidden_dim)
        if self.cell_type == "GRU":
            rnn_out, _ = self.rnn(x)
        else:
            rnn_out, (h_n, c_n) = self.rnn(x)

        # Temporal self-attention pooling across time dimension T
        attn_weights = self.attention(rnn_out) # (B, T, 1)
        context_vector = torch.sum(attn_weights * rnn_out, dim=1) # (B, hidden_dim)

        return context_vector
