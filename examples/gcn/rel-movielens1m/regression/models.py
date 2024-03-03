import sys
import os.path as osp
current_path = osp.dirname(__file__)
sys.path.append(current_path + '/../../')
from GCNconv import GraphConvolution

import torch
import torch.nn as nn
import torch.nn.functional as F
from GCNconv import GraphConvolution
# from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj).relu()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class Decoder(nn.Module):
    def __init__(self, nhid):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(2 * nhid, nhid)
        self.lin2 = nn.Linear(nhid, 1)
    
    def forward(self, z, adj):
        row, col = adj.indices()[0], adj.indices()[1]
        # print(z[col[:2]])
        # print(z[4000:4002])
        # z = (z[row] * z[col]).sum(dim=1)

        z = torch.cat([z[row], z[col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(nn.Module):
    def __init__(self, nfeat, nhid):
        super(Model, self).__init__()
        self.encoder = GCN(nfeat, nhid, nhid)
        self.decoder = Decoder(nhid)
    
    def forward(self, x_all, adj, adj_drop):
        z_all = self.encoder(x_all, adj_drop)
        return self.decoder(z_all, adj)