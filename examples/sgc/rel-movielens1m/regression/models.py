import sys
import os.path as osp
current_path = osp.dirname(__file__)
sys.path.append(current_path + '/../../')


import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x, adj):
        return self.W(x)

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
    def __init__(self, nfeat, nhid):
        super(Model, self).__init__()
        self.encoder = SGC(nfeat, nhid)
        self.decoder = Decoder(nhid)
    
    def forward(self, x_all, adj, adj_drop):
        z_all = self.encoder(x_all, adj_drop)
        return self.decoder(z_all, adj)