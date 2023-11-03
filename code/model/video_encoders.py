import torch
import torch.nn as nn


class VideoMaeEncoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(VideoMaeEncoder, self).__init__()
        self.video_net = video_net
        # self.activate = nn.Tanh()
        self.activate = nn.GELU()
        self.args = args
        self.avg_pool = nn.AdaptiveAvgPool2d((1, args.word_embedding_dim))
        self.padding_label = torch.Tensor([-1]).to(args.local_rank)
        # self.add_nor = nn.LayerNorm(args.word_embedding_dim, eps=1e-6) ###
        self.linear = nn.Linear(args.word_embedding_dim, args.embedding_dim)

    def forward(self, item_content):
        # torch.Size([112, 4, 3, 224, 224])
        item_scoring = self.video_net(item_content).last_hidden_state
        # torch.Size([112, 392, 768])

        item_scoring = self.avg_pool(item_scoring)
        # torch.Size([112, 1, 768])

        item_scoring = self.linear(item_scoring.squeeze(1))  # torch.Size([112, 512])
        # item_scoring = self.linear(self.add_nor(item_scoring.view(item_scoring.shape[0], -1)))

        return self.activate(item_scoring)
