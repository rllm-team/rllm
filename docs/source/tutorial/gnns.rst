Design of GNNs
===============

What is GNN?
----------------
In machine learning, **Graph Neural Networks (GNNs)** are a class of neural networks specifically designed to process graph-structured data. In a GNN, the input is represented as a graph, where nodes (vertices) correspond to entities and edges represent the relationships or interactions between these entities. A typical GNN architecture consists of an initial Transform followed by multiple Convolution layers, as detailed in :doc:`Understanding Transforms <transforms>` and :doc:`Understanding Convolutions <convolutions>`.


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

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.adj)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        out = model(data.x, data.adj)
        pred = out.argmax(dim=1)

        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / int(data.test_mask.sum())

    print(f"Accuracy: {acc:.4f}")
    >>> 0.8150
