import torch
import torch.nn as nn
from .modules import TransformerEncoder
from torch.nn.init import xavier_normal_, constant_


class UserEncoder(torch.nn.Module):
    def __init__(self, item_num, args):
        super(UserEncoder, self).__init__()
        max_seq_len = args.max_seq_len
        item_dim = args.embedding_dim
        num_attention_heads = args.num_attention_heads
        dropout = args.drop_rate
        n_layers = args.transformer_block
        self.transformer_encoder = TransformerEncoder(
            n_vocab=item_num,
            n_position=max_seq_len,
            d_model=item_dim,
            n_heads=num_attention_heads,
            dropout=dropout,
            n_layers=n_layers,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank):
        att_mask = log_mask != 0
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(
            local_rank
        )
        att_mask = torch.where(att_mask, 0.0, -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)
