import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, use_pred):
        super(MLP, self).__init__()
        self.use_pred = use_pred
        if self.use_pred:
            self.encoder = torch.nn.Embedding(out_dim + 1, hidden_dim)
        self.convs = torch.nn.ModuleList()
        self.convs.append(nn.Linear(in_dim, hidden_dim))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.convs.append(nn.Linear(hidden_dim, out_dim))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t=None):
        if self.use_pred:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)

        return x
