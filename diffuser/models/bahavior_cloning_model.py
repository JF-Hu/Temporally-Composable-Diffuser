
import torch
import torch.nn as nn




class BC(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        dim=128,
        dropout=0.2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim
        self.dropout = dropout

        self.mlp = nn.Sequential(
                        nn.Linear(self.input_dim, 256),
                        nn.Mish(),
                        nn.Dropout(dropout),
                        nn.Linear(256, 256),
                        nn.Mish(),
                        nn.Dropout(dropout),
                        nn.Linear(256, self.action_dim),
                    )

    def forward(self, x):
        assert x.shape[-1] == self.input_dim
        out = self.mlp(x)
        return out