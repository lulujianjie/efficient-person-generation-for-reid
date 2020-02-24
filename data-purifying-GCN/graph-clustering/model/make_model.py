import torch
import torch.nn as nn
from .backbones.basic_blocks import MeanAggregator, GraphConv


class GCN(nn.Module):
    def __init__(self, input_dim=2048):
        super(GCN, self).__init__()
        self.bn0 = nn.BatchNorm1d(input_dim, affine=False)
        self.conv1 = GraphConv(2048, 1024, MeanAggregator)
        self.conv2 = GraphConv(1024, 512, MeanAggregator)
        self.conv3 = GraphConv(512, 256, MeanAggregator)
        self.conv4 = GraphConv(256, 256, MeanAggregator)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(256),
            nn.Linear(256, 2))

    def forward(self, x, A, one_hop_idcs, train=True):
        # data normalization l2 -> bn
        B, N, D = x.shape
        # xnorm = x.norm(2,2,keepdim=True) + 1e-8
        # xnorm = xnorm.expand_as(x)
        # x = x.div(xnorm)

        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)

        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)
        k1 = one_hop_idcs.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B, k1, dout).cuda()
        for b in range(B):
            edge_feat[b, :, :] = x[b, one_hop_idcs[b]]
        edge_feat = edge_feat.view(-1, dout)
        pred = self.classifier(edge_feat)

        # shape: (B*k1)x2
        return pred

def make_model(Cfg):
    model = GCN(input_dim=Cfg.INPUT_DIM)
    return model