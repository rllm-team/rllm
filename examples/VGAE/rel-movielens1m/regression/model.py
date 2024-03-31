import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from layer import GraphConvolution, InnerProductDecoder, Decoder


class GAE_REGRESSION(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GAE_REGRESSION, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1,
                                    dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2,
                                    dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2,
                                    dropout, act=lambda x: x)
        self.decoder = Decoder(hidden_dim1)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            # std = torch.exp(logvar)
            std = logvar
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        # return self.regression_decoder(z)
        return self.decoder(z, adj), self.dc(z), mu, logvar

