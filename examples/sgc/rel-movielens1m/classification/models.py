import torch
import torch.nn as nn


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def sgc_precompute(self, features, adj, degree=1):
        for i in range(degree):
            features = torch.spmm(adj, features)
        return features

    def forward(self, x, adj):
        x = self.sgc_precompute(x, adj)
        return self.W(x)
