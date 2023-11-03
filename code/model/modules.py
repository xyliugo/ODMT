import torch
import torch.nn as nn
import torch.nn.functional as F


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer"""

    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor"""

    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList(
            [PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)]
        )
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits).to(x.device) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x, train):
        gates = self.noisy_top_k_gating(x, train)  # (B, n_E)
        expert_outputs = [
            self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)
        ]  # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.dropout(self.w_2(self.activate(self.w_1(x))))
        return self.layer_norm(residual + x)


class SelfAttention(nn.Module):
    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask):
        attn = torch.matmul(query, key.transpose(-2, -1)) / self.temperature
        attn = attn + mask
        p_attn = self.dropout(self.softmax(attn))
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_v = self.d_k

        self.w_Q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_K = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_V = nn.Linear(d_model, n_heads * self.d_v, bias=False)
        self.fc = nn.Linear(n_heads * self.d_v, d_model, bias=False)

        self.self_attention = SelfAttention(
            temperature=self.d_k**0.5, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, mask):
        sz_b, len_q, len_k, len_v = (
            query.size(0),
            query.size(1),
            key.size(1),
            value.size(1),
        )
        residual = query

        q = self.w_Q(query).view(sz_b, len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_K(key).view(sz_b, len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_V(value).view(sz_b, len_v, self.n_heads, self.d_v).transpose(1, 2)

        x, attn = self.self_attention(q, k, v, mask=mask)
        x = x.transpose(1, 2).contiguous().view(sz_b, len_q, self.d_model)
        x = self.dropout(self.fc(x))
        return self.layer_norm(residual + x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_inner, dropout):
        super().__init__()
        self.multi_head_attention = MultiHeadedAttention(
            n_heads=n_heads, d_model=d_model, dropout=dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_inner=d_inner, dropout=dropout
        )

    def forward(self, block_input, mask):
        output = self.multi_head_attention(block_input, block_input, block_input, mask)
        return self.feed_forward(output)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, n_vocab, n_position, d_model, n_heads, dropout, n_layers):
        super(TransformerEncoder, self).__init__()
        self.position_embedding = nn.Embedding(n_position, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_inner=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, input_embs, log_mask, att_mask):
        position_ids = torch.arange(
            log_mask.size(1), dtype=torch.long, device=log_mask.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(log_mask)
        output = self.layer_norm(input_embs + self.position_embedding(position_ids))
        output = self.dropout(output)
        for transformer in self.transformer_blocks:
            output = transformer.forward(output, att_mask)
        return output


class MergedAttentionFusion(torch.nn.Module):
    def __init__(self, args):
        super(MergedAttentionFusion, self).__init__()
        d_model = args.embedding_dim
        dropout = args.drop_rate
        n_heads = args.merged_att_layers
        n_layers = args.merged_att_layers
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.token_type_embeddings = nn.Embedding(2, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_inner=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        id_feats,
        lang_feats,
        lang_attention_mask,
        visn_feats,
        visn_attention_mask,
        cut,
    ):
        text_length = lang_feats.size(1)

        id_att_output = id_feats.unsqueeze(1)
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_attention_mask = lang_attention_mask.unsqueeze(1).unsqueeze(2)
        lang_attention_mask = lang_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        lang_attention_mask = (1.0 - lang_attention_mask) * -10000.0

        visn_attention_mask = visn_attention_mask.unsqueeze(1).unsqueeze(2)
        visn_attention_mask = visn_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        visn_attention_mask = (1.0 - visn_attention_mask) * -10000.0

        id_attention_mask = torch.ones((lang_attention_mask.size(0), 1, 1, 1))
        id_attention_mask = id_attention_mask.to(lang_attention_mask.device)
        id_attention_mask = (1.0 - id_attention_mask) * -10000.0

        att_output = torch.cat([lang_att_output, visn_att_output, id_att_output], dim=1)
        att_output = self.dropout(att_output)
        mask_output = torch.cat(
            [lang_attention_mask, visn_attention_mask, id_attention_mask], dim=-1
        )
        mask_output = mask_output.expand((-1, -1, mask_output.size(-1), -1)).clone()
        if cut == "v1":
            mask_output[:, :, :-1, -1] = -10000.0

        for transformer in self.transformer_blocks:
            att_output = transformer.forward(att_output, mask_output)

        # return att_outputs
        x = att_output[:, :text_length, :][:, 0]
        y = att_output[:, text_length:-1, :][:, 0]
        z = att_output[:, -1]

        return z, x, y
