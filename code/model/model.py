import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_normal_

from .text_encoders import TextEncoder
from .image_encoders import ImageEncoder
from .user_encoders import UserEncoder
from .fusion_module import ConcatFusion
from .modules import MoEAdaptorLayer, MergedAttentionFusion


def kd_ce_loss(logits_S, logits_T, temperature=1):
    """
    Calculate the cross entropy between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    """
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss * (temperature * temperature)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class Model(torch.nn.Module):
    def __init__(self, args, pop_prob_list, item_num, t_feat, v_feat):
        super(Model, self).__init__()
        self.args = args
        self.embs_dim = args.embedding_dim
        self.max_seq_len = args.max_seq_len
        self.item_tower = args.item_tower
        self.item_num = item_num
        self.pop_prob_list = torch.FloatTensor(pop_prob_list)
        self.ft_layer = args.ft_layer
        self.t_feat = t_feat
        self.v_feat = v_feat

        ### get user encoder
        self.id_user_encoder = UserEncoder(item_num, args)
        self.t_user_encoder = UserEncoder(item_num, args)
        self.v_user_encoder = UserEncoder(item_num, args)

        ### get item encoder
        self.id_encoder = nn.Embedding(
            item_num + 1, args.id_embedding_dim, padding_idx=0
        )

        self.id_proj = nn.Linear(args.id_embedding_dim, self.embs_dim)
        self.text_proj = nn.Linear(args.text_embedding_dim, self.embs_dim)
        self.image_proj = nn.Linear(args.image_embedding_dim, self.embs_dim)

        self.id_dnns = nn.Sequential(
            nn.Linear(self.embs_dim, self.embs_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embs_dim, self.embs_dim),
        )
        self.t_dnns = nn.Sequential(
            nn.Linear(self.embs_dim, self.embs_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embs_dim, self.embs_dim),
        )
        self.v_dnns = nn.Sequential(
            nn.Linear(self.embs_dim, self.embs_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embs_dim, self.embs_dim),
        )

        self.IMT = MergedAttentionFusion(args)
        ### ce loss
        self.criterion = nn.CrossEntropyLoss()

        ### initialization
        if self.args.init == 0:
            xavier_normal_(self.id_encoder.weight.data)
        elif self.args.init == 1:
            self.id_encoder.weight.data = (v_feat[:, 0] + t_feat[:, 0]) / 2
        elif self.args.init == 2:
            self.id_encoder.weight.data = v_feat[:, 0]
        elif self.args.init == 3:
            self.id_encoder.weight.data = t_feat[:, 0]

    def calculate_logits(
        self,
        debias_logits,
        item_embedding,
        sequence_encoder,
        log_mask,
        unused_item_mask,
        local_rank,
    ):
        pre_user_embedding = item_embedding.view(
            -1, self.max_seq_len + 1, self.embs_dim
        )
        post_user_embedding = sequence_encoder(
            pre_user_embedding[:, :-1, :], log_mask, local_rank
        )
        user_embedding = post_user_embedding.view(-1, self.embs_dim)

        logits = torch.matmul(
            user_embedding, item_embedding.t()
        )  # (B * S, B * (S + 1))
        logits = logits - debias_logits
        expanded_log_mask = torch.cat(
            (log_mask, torch.ones(log_mask.size(0)).unsqueeze(-1).to(local_rank)), dim=1
        )

        logits[:, expanded_log_mask.view(-1) == 0] = -1e4
        logits[unused_item_mask] = -1e4
        indices = torch.where(log_mask.view(-1) != 0)

        return logits[indices]

    def forward(self, epoch, sample_items_id, sample_items_text, log_mask, local_rank):
        ## get debias logits
        pop_prob_list = self.pop_prob_list.to(local_rank)
        debias_logits = torch.log(pop_prob_list[sample_items_id.view(-1)])

        ## mask
        text_length = sample_items_text.size(1) // 2
        text_mask = torch.narrow(sample_items_text, 1, text_length, text_length)
        image_mask = torch.ones((sample_items_text.size(0), 50)).to(local_rank)

        ## get id text image features after encoding
        id_embs = self.id_proj(self.id_encoder(sample_items_id))
        t_embs = self.text_proj(self.t_feat[sample_items_id])
        v_embs = self.image_proj(self.v_feat[sample_items_id])

        id_embs, t_embs, v_embs = self.IMT(
            id_embs, t_embs, text_mask, v_embs, image_mask, self.args.version
        )
        id_embs, t_embs, v_embs = (
            self.id_dnns(id_embs),
            self.t_dnns(t_embs),
            self.v_dnns(v_embs),
        )

        ## get unsed item mask
        bs, seq_len = log_mask.size(0), log_mask.size(1)
        label = torch.arange(bs * (seq_len + 1)).reshape(bs, seq_len + 1)
        label = label[:, 1:].to(local_rank).view(-1)

        flatten_item_seq = sample_items_id
        user_history = torch.zeros(bs, seq_len + 2).type_as(sample_items_id)
        user_history[:, :-1] = sample_items_id.view(bs, -1)
        user_history = user_history.unsqueeze(-1).expand(-1, -1, len(flatten_item_seq))
        history_item_mask = (user_history == flatten_item_seq).any(dim=1)
        history_item_mask = history_item_mask.repeat_interleave(seq_len, dim=0)
        unused_item_mask = torch.scatter(history_item_mask, 1, label.view(-1, 1), False)

        ## calculate logits
        indices = torch.where(log_mask.view(-1) != 0)
        label = label[indices]
        id_logits = self.calculate_logits(
            debias_logits,
            id_embs,
            self.id_user_encoder,
            log_mask,
            unused_item_mask,
            local_rank,
        )
        t_logits = self.calculate_logits(
            debias_logits,
            t_embs,
            self.t_user_encoder,
            log_mask,
            unused_item_mask,
            local_rank,
        )
        v_logits = self.calculate_logits(
            debias_logits,
            v_embs,
            self.v_user_encoder,
            log_mask,
            unused_item_mask,
            local_rank,
        )

        alpha, tau = self.args.alpha, self.args.tau
        weight = sigmoid_rampup(epoch, alpha) if alpha >= 1 else alpha
        ensemble_logits = (t_logits + v_logits + id_logits) / 3
        ce_loss = (
            self.criterion(id_logits, label)
            + self.criterion(t_logits, label)
            + self.criterion(v_logits, label)
        )
        kl_loss = weight * (
            kd_ce_loss(t_logits, id_logits, tau)
            + kd_ce_loss(v_logits, id_logits, tau)
            + kd_ce_loss(id_logits, ensemble_logits, tau)
        )

        return ce_loss, kl_loss
