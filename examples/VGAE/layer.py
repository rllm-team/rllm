import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.bn(output)
        return self.act(output)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class Decoder(nn.Module):
    def __init__(self, nhid):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(nhid, nhid)
        self.lin2 = nn.Linear(nhid, 1)

    def forward(self, z, adj):
        row, col = adj.coalesce().indices()[0], adj.coalesce().indices()[1]
        z = torch.cat([z[row], z[col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

