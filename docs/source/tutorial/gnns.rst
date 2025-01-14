Design of GNNs
===============

What is a GNN?
----------------
In machine learning, **Graph Neural Networks (GNNs)** are a class of neural networks specifically designed to process graph-structured data. In a GNN, the input is represented as a graph, where nodes (vertices) correspond to entities and edges represent the relationships or interactions between these entities. A typical GNN architecture consists of an initial Transform followed by multiple Convolution layers, as detailed in *Understanding Transform* and *Understanding Convolution*.


Construct a GCN 
----------------
In this tutorial, we will discuss how to train a simple Graph Convolutional Network (GCN). Since their introduction, Graph Neural Networks (GNNs) have significantly impacted various fields such as social network analysis, recommendation systems, and link prediction. The GCN model, proposed in the paper `[Semi-supervised Classification with Graph Convolutional Networks] <https://arxiv.org/abs/1609.02907>`__, is one of the most classic models in GNN research. Next, we will build a simple GCN and use it to perform node classification on the Cora citation network dataset.

First, we need to load the Cora dataset, add self-loops to the adjacency matrix, and normalize it:

.. code-block:: python

    import os.path as osp

    from rllm.datasets import PlanetoidDataset
    from rllm.transforms.graph_transforms import GCNTransform

    # Set random seed and device
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
    data = PlanetoidDataset(path, args.dataset)[0]

    # Transform data
    transform = GCNTransform()
    data = transform(data).to(device)

Next, we can construct a two-layer GCN and use ReLU as the activation function:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from rllm.nn.conv.graph_conv import GCNConv

    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, x, adj):
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.conv1(x, adj))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, adj)
            return x

We can initialize the optimizer and loss function.

.. code-block:: python

    # Set up model, optimizer and loss function
    model = GCN(
        in_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=data.num_classes,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

Finally, we need to implement a :obj:`train()` function and a :obj:`test()` function, the latter of which does not require gradient tracking. The model can then be trained on the training and validation sets, and the classification results can be obtained from the test set.

.. code-block:: python

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.adj)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()


    @torch.no_grad()
    def test():
        model.eval()
        out = model(data.x, data.adj)
        pred = out.argmax(dim=1)

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = float(pred[mask].eq(data.y[mask]).sum().item())
            accs.append(correct / int(mask.sum()))
        return accs


    metric = "Acc"
    best_val_acc = best_test_acc = 0
    times = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = train()
        train_acc, val_acc, test_acc = test()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        times.append(time.time() - start)
        print(
            f"Epoch: [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
            f"Val {metric}: {val_acc:.4f}, Test {metric}: {test_acc:.4f} "
        )

    print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"Total time: {sum(times):.4f}s")
    print(f"Best test acc: {best_test_acc:.4f}")