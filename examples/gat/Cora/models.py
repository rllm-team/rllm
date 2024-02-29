import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, in_heads=8, out_heads=1):
        super(GAT, self).__init__()
        self.in_heads = in_heads
        self.out_heads = out_heads
        self.dropout = dropout
        self.conv1 = GATConv(
            nfeat, nhid, heads=self.in_heads, dropout=self.dropout)
        self.conv2 = GATConv(nhid*self.in_heads, nclass, concat=False,
                             heads=self.out_heads, dropout=self.dropout)

    def forward(self,  x, adj):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, adj)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, adj)

        return F.log_softmax(x, dim=1)
