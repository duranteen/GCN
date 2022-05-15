from model.GraphConvolution import GraphConvolution
import torch
from torch import nn
from torch.nn import functional as F


class GCN(nn.Module):
    def __init__(self, input_dim=1433, hidden_dim=16):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, 7)

    def forward(self, adjacency, features):
        h = F.relu(self.gcn1(adjacency, features))
        output = self.gcn2(adjacency, h)
        return output
