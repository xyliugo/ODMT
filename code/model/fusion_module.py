import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


class SumFusion(nn.Module):
    def __init__(self, args):
        super(SumFusion, self).__init__()
        self.fc_xyz = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc = nn.Linear(args.embedding_dim, args.embedding_dim)
        xavier_normal_(self.fc_xyz.weight.data)
        xavier_normal_(self.fc.weight.data)

    def forward(self, x, y, z):
        output = F.leaky_relu(self.fc_xyz(x + y + z))
        return self.fc(output)


class ConcatFusion(nn.Module):
    def __init__(self, args):
        super(ConcatFusion, self).__init__()
        self.fc_1 = nn.Linear(args.embedding_dim * 3, args.embedding_dim)
        self.fc_2 = nn.Linear(args.embedding_dim, args.embedding_dim)
        xavier_normal_(self.fc_1.weight.data)
        xavier_normal_(self.fc_2.weight.data)

    def forward(self, x, y, z):
        output = torch.cat((x, y, z), dim=-1)
        output = self.fc_2(F.leaky_relu(self.fc_1(output)))
        return output


class NonInvasiveFusion(nn.Module):
    def __init__(self, merged_attn, args):
        super(NonInvasiveFusion, self).__init__()
        self.fc_1 = nn.Linear(args.embedding_dim, 4 * args.embedding_dim)
        self.fc_2 = nn.Linear(4 * args.embedding_dim, args.embedding_dim)
        self.fc_3 = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.merged_attn = merged_attn
        xavier_normal_(self.fc_1.weight.data)
        xavier_normal_(self.fc_2.weight.data)

    def forward(
        self, id_embs, text_embs, text_mask, image_embs, image_mask, local_rank
    ):
        id_embs = F.leaky_relu(self.fc_1(id_embs))
        mm_embs, text_branch_embs, image_branch_embs = self.merged_attn(
            text_embs, text_mask, image_embs, image_mask, local_rank
        )
        item_embs = F.leaky_relu(self.fc_2(id_embs) + mm_embs)
        return item_embs, text_branch_embs, image_branch_embs
