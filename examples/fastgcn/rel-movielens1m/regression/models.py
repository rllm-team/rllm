import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sampler):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.sampler = sampler
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj_list):
        # print(f"adj[0]:{adj[0]}")
        # x = torch.tensor(x, requires_grad=True)
        # adj_0 = torch.tensor(adj_list[0], requires_grad=True)
        # adj_1 = torch.tensor(adj_list[1], requires_grad=True)
        outputs1 = F.relu(self.gc1(x, adj_list[0]))
        outputs1 = F.dropout(outputs1, self.dropout, training=self.training)
        outputs2 = self.gc2(outputs1, adj_list[1])
        return F.logsigmoid(outputs2)
        # return F.log_softmax(outputs2, dim=1)
        # return F.log_softmax(outputs2, dim=1)
        # return self.out_softmax(outputs2)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, nhid):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(nhid, nhid)
        self.lin2 = nn.Linear(nhid, 1)

    def forward(self, z, adj):
        # print(f"z: {z.shape}")
        # row, col = adj._indices()[0], adj._indices()[1]
        # # print(z[col[:2]])
        # # print(z[4000:4002])
        # z = (z[row] * z[col]).sum(dim=1)
        # print(f"z[row].shape: {z[row].shape}")
        # print(f"z[col].shape: {z[col].shape}")

        # z = torch.cat([z[row], z[col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(nn.Module):
    def __init__(self, nfeat, nhid, dropout, sampler):
        super(Model, self).__init__()
        self.encoder = GCN(nfeat, nhid, nhid, dropout, sampler)
        self.decoder = Decoder(nhid)
        # self.v_num = v_num

    def forward(self, x_all, adj, adj_drop):
        z_all = self.encoder(x_all, adj_drop)
        # print(z_all[0])
        # print(z_all[1])
        return self.decoder(z_all, adj)

    def sampling(self, *args, **kwargs):
        return self.encoder.sampling(*args, **kwargs)
