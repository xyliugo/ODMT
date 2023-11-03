import torch
import torch.nn as nn
from .modules import TransformerBlock


class TextEncoder(torch.nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.trm_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=args.embedding_dim,
                    d_inner=args.embedding_dim * 4,
                    dropout=args.drop_rate,
                    n_heads=args.num_attention_heads,
                )
                for _ in range(args.t_trm_layers)
            ]
        )

    def forward(self, hidden_states, text_mask, agg=True):
        text_mask = text_mask.unsqueeze(1).unsqueeze(2)
        text_mask = (1.0 - text_mask) * -10000.0
        for trm_layer in self.trm_layers:
            hidden_states = trm_layer.forward(hidden_states, text_mask)
        if agg:
            return hidden_states[:, 0]
        else:
            return hidden_states
