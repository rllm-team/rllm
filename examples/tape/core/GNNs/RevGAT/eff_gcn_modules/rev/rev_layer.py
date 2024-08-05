import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from gcn_lib.sparse.torch_nn import norm_layer
except:
    print("An import exception occurred")


class SharedDropout(nn.Module):
    def __init__(self):
        super(SharedDropout, self).__init__()
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        if self.training:
            assert self.mask is not None
            out = x * self.mask
            return out
        else:
            return x


class BasicBlock(nn.Module):
    def __init__(self, norm, in_channels):
        super(BasicBlock, self).__init__()
        self.norm = norm_layer(norm, in_channels)
        self.dropout = SharedDropout()

    def forward(self, x, edge_index, dropout_mask=None, edge_emb=None):
        # dropout_mask = kwargs.get('dropout_mask', None)
        # edge_emb = kwargs.get('edge_emb', None)
        out = self.norm(x)
        out = F.relu(out)

        if isinstance(self.dropout, SharedDropout):
            if dropout_mask is not None:
                self.dropout.set_mask(dropout_mask)
        out = self.dropout(out)

        if edge_emb is not None:
            out = self.gcn(out, edge_index, edge_emb)
        else:
            out = self.gcn(out, edge_index)

        return out


class GCNBlock(BasicBlock):
    def __init__(self, in_channels, out_channels,
                 norm='layer'):
        super(GCNBlock, self).__init__(norm, in_channels)

        self.gcn = GCNConv(in_channels, out_channels)


class SAGEBlock(BasicBlock):
    def __init__(self, in_channels, out_channels,
                 norm='layer',
                 dropout=0.0):
        super(SAGEBlock, self).__init__(norm, in_channels)

        self.gcn = SAGEConv(in_channels, out_channels)


class GATBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 heads=1,
                 norm='layer',
                 att_dropout=0.0,
                 dropout=0.0):
        super(GATBlock, self).__init__(norm, in_channels)

        self.gcn = GATConv(in_channels, out_channels,
                           heads=heads,
                           concat=False,
                           dropout=att_dropout,
                           add_self_loops=False)
