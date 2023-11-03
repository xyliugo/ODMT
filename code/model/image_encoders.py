import torch
import torch.nn as nn

from .modules import TransformerEncoder, TransformerBlock
from torch.nn.init import xavier_normal_, constant_


class ImageEncoder(torch.nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.trm_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=args.embedding_dim,
                    d_inner=args.embedding_dim * 4,
                    dropout=args.drop_rate,
                    n_heads=args.num_attention_heads,
                )
                for _ in range(args.v_trm_layers)
            ]
        )

    def forward(self, hidden_states, image_mask, agg=True):
        image_mask = image_mask.unsqueeze(1).unsqueeze(2)
        image_mask = (1.0 - image_mask) * -10000.0
        for trm_layer in self.trm_layers:
            hidden_states = trm_layer.forward(hidden_states, image_mask)
        if agg:
            return hidden_states[:, 0]
        else:
            return hidden_states


class SwinEncoder(torch.nn.Module):
    def __init__(self, image_net, args):
        super(SwinEncoder, self).__init__()

        self.image_net = image_net
        num_fc_ftr = self.image_net.classifier.in_features
        self.image_net.classifier = nn.Linear(num_fc_ftr, args.embedding_dim)

        xavier_normal_(self.image_net.classifier.weight.data)
        if self.image_net.classifier.bias is not None:
            constant_(self.image_net.classifier.bias.data, 0)

    def forward(self, image):
        return self.image_net(image)[0]


class ResnetEncoder(torch.nn.Module):
    def __init__(self, image_net, args):
        super(ResnetEncoder, self).__init__()
        self.resnet = image_net

        num_fc_ftr = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_fc_ftr, args.embedding_dim)

        xavier_normal_(self.resnet.fc.weight.data)
        if self.resnet.fc.bias is not None:
            constant_(self.resnet.fc.bias.data, 0)

    def forward(self, image):
        return self.resnet(image)
