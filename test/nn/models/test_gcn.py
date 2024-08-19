from sympy import li, prime
import rllm.transforms as T
from rllm.datasets.planetoid import PlanetoidDataset
import torch
import torch.nn.functional as F
import os.path as osp
import time
import os
import subprocess
EXAMPLE_ROOT = os.path.join(
    os.path.dirname(os.path.relpath(__file__)),
    "..",
    "..",
    "..",
    "examples",
)


# class GCN(torch.nn.Module):
#     r"""The Graph Neural Network from the `"Semi-supervised
#     Classification with Graph Convolutional Networks"
#     <https://arxiv.org/abs/1609.02907>`_ paper, using the
#     :class:`~rllm.nn.conv.gcn_conv.GCNConv` operator for message passing.

#     Args:
#         in_channels (int): Size of each input sample, or :obj:`-1` to derive
#             the size from the first input(s) to the forward method.
#         hidden_channels (int): Size of each hidden sample.
#         out_channels (int, optional): If not set to :obj:`None`, will apply a
#             final linear transformation to convert hidden node embeddings to
#             output size :obj:`out_channels`. (default: :obj:`None`)
#         num_layers (int): Number of message passing layers.
#         dropout (float, optional): Dropout probability. (default: :obj:`0.`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.GCNConv`.
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: int,
#         out_channels: int,
#         num_layers: int = 2,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels
#         self.num_layers = num_layers
#         self.dropout = dropout

#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GCNConv(in_channels, hidden_channels))
#         for _ in range(num_layers-2):
#             self.convs.append(GCNConv(hidden_channels, hidden_channels))
#         self.convs.append(GCNConv(hidden_channels, out_channels))

#     def forward(self, x, adj):
#         for i in range(self.num_layers-1):
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             x = F.relu(self.convs[i](x, adj))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, adj)
#         return x

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.hidden_channels}, '
#                 f'{self.out_channels}, num_layers={self.num_layers})')


# def test_gcn(in_channels, hidden_channels, out_channels):
#     x = torch.randn(3, in_channels)
#     adj = torch.tensor([
#         [0., 1., 0.],
#         [1., 0., 1.],
#         [0., 1., 0.]
#         ])

#     model = GCN(in_channels, hidden_channels, out_channels)

#     assert (
#         str(model)
#         == f"GCN({in_channels}, {hidden_channels}, {out_channels}, num_layers=2)"
#     )
#     assert model(x, adj).size() == (3, out_channels)


# def test_gcn_cora():

#     transform = T.Compose([T.NormalizeFeatures("l1"), T.GCNNorm()])
#     path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
#     dataset = PlanetoidDataset(path, "cora", transform=transform)
#     data = dataset[0]

#     model = GCN(
#         in_channels=data.x.shape[1],
#         hidden_channels=16,
#         out_channels=data.num_classes,
#         num_layers=2,
#         dropout=0.5,
#     )
#     print(list(model.parameters()))
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=0.01,
#         weight_decay=5e-4,
#     )
#     loss_fn = torch.nn.CrossEntropyLoss()

#     def train():
#         model.train()
#         optimizer.zero_grad()
#         out = model(data.x, data.adj)
#         loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()
#         return loss.item()

#     @torch.no_grad()
#     def test():
#         model.eval()
#         out = model(data.x, data.adj)
#         pred = out.argmax(dim=1)

#         accs = []
#         for mask in [data.train_mask, data.val_mask, data.test_mask]:
#             correct = float(pred[mask].eq(data.y[mask]).sum().item())
#             accs.append(correct / int(mask.sum()))
#         return accs

#     best_val_acc = best_test_acc = 0
#     times = []
#     st = time.time()
#     for epoch in range(1, args.epochs + 1):
#         start = time.time()
#         train_loss = train()
#         train_acc, val_acc, test_acc = test()
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_test_acc = test_acc
#         times.append(time.time() - start)
#     et = time.time()
#     print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
#     print(f"Total time: {et-st}s")
#     print(f"Best test acc: {best_test_acc:.4f}")

def test_gcn():
    print(EXAMPLE_ROOT)
    script = os.path.join(EXAMPLE_ROOT, "gcn.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    print(out)
    print(stdout[-9:])
    assert float(stdout[-5:]) > 0.75

test_gcn()
