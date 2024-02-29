import torch
import torch.nn as nn
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


class Decoder(nn.Module):
    def __init__(self, nhid):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(2 * nhid, nhid)
        self.lin2 = nn.Linear(nhid, 1)

    def forward(self, z, adj):
        row, col = adj.indices()[0], adj.indices()[1]

        z = torch.cat([z[row], z[col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(nn.Module):
    def __init__(self, nfeat, nhid, v_num):
        super(Model, self).__init__()
        self.encoder = GAT(nfeat, nhid, nhid)
        self.decoder = Decoder(nhid)
        self.v_num = v_num

    def forward(self, x_all, adj, adj_drop):
        z_all = self.encoder(x_all, adj_drop)
        return self.decoder(z_all, adj)
