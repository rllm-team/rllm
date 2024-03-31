import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from layer import GraphConvolution, InnerProductDecoder


class GAE_CLASSIFICATION(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2,
                 num_classes, dropout):
        super(GAE_CLASSIFICATION, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1,
                                    dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2,
                                    dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2,
                                    dropout, act=lambda x: x)
        self.fc_out = nn.Linear(hidden_dim2, num_classes)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        out = self.fc_out(z)
        return out, self.dc(z), mu, logvar