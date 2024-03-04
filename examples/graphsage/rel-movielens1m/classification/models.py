import torch.nn as nn
import os.path as osp
import sys
current_path = osp.dirname(__file__)
sys.path.append(current_path + '/../../')
from SageConv import SageGCN


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1]))
        self.gcn.append(
            SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None)
        )

    def forward(self, node_features_list):
        hidden = node_features_list
        for layer in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[layer]
            for hop in range(self.num_layers - layer):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        # return F.log_softmax(hidden[0], dim=1)
        # return F.logsigmoid(hidden[0])
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )
