from layers import GraphConvolution
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../src")


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
