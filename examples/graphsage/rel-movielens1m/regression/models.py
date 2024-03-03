import torch
import torch.nn as nn
import sys
import os.path as osp
current_path = osp.dirname(__file__)
sys.path.append(current_path + '/../../')
from SageConv import SageGCN

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    # "swish": Swish(),
    'silu': nn.SiLU()
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True,
                 act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(num_neighbors_list)
        self.num_neighbors_list = num_neighbors_list
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1]))
        self.gcn.append(SageGCN(hidden_dim[0], hidden_dim[0], activation=None))
        self.mlp = MLP(hidden_dim[0]+6424, 1, hidden_dim[0])

    def forward(self, node_features_list):
        hidden = node_features_list
        for i in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[i]
            for hop in range(self.num_layers - i):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        repeat_self = hidden[0].unsqueeze(1).repeat(
            1,
            self.num_neighbors_list[0], 1).view(-1, self.hidden_dim[0])
        repeat_neighbor = node_features_list[1]
        edge_features = torch.cat([repeat_self, repeat_neighbor], dim=-1)
        edge_logits = self.mlp(edge_features)
        # return F.softmax(edge_logits)
        return edge_logits

    def test(self, node_features_list, node_index_test_movie,
             test_adj_dict, features):
        hidden = node_features_list
        for i in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[i]
            for hop in range(self.num_layers - i):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        movie_features = hidden[0]
        pred = {}
        for i, node in enumerate(node_index_test_movie):
            node = node.item()
            feature_one_movie = movie_features[i]
            feature_users = features[test_adj_dict[node]]
            num_users = feature_users.shape[0]
            final_feature = torch.cat(
                [feature_one_movie.unsqueeze(0).repeat(num_users, 1),
                 feature_users],
                dim=-1)
            logits = self.mlp(final_feature)
            pred[node] = logits
        return pred

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )
