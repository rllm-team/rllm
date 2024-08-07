Design of GNNs
===============

Construct a GCN 
----------------
In this tutorial, we will discuss how to train a simple Graph Convolutional Network (GCN). Since their introduction, Graph Neural Networks (GNNs) have significantly impacted various fields such as social network analysis, recommendation systems, and link prediction. The GCN model, proposed in the paper `[Semi-supervised Classification with Graph Convolutional Networks] <https://arxiv.org/abs/1609.02907>`__ , is one of the most classic models in GNN research. Next, we will build a simple GCN and use it to perform node classification on the Cora citation network dataset.

First, we need to load the Cora dataset, add self-loops to the adjacency matrix, and normalize it:

.. code-block:: python

    import os.path as osp

    from rllm.datasets import PlanetoidDataset
    import rllm.transforms as T

    transform = T.Compose([
        T.GCNNorm(), # Add self-loops and normalize
        T.NormalizeFeatures('l2') # Normalize node features
    ])

    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/Cora')
    dataset = PlanetoidDataset(path, name='cora', transform=transform)
    data = dataset[0]

Next, we can construct a two-layer GCN and use ReLU as the activation function:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from rllm.nn.conv import GCNConv

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

We can initialize the optimizer and loss function, and train the constructed GCN model on the Cora dataset for 200 epochs:

.. code-block:: python

    model = GCN(
        in_channels=data.x.shape[1],
        hidden_channels=32,
        out_channels=data.num_classes,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.adj)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

Finally, we can validate the training results on the predefined test set and print the accuracy:

.. code-block:: python

    with torch.no_grad():
        model.eval()
        out = model(data.x, data.adj)
        pred = out.argmax(dim=1)

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = float(pred[mask].eq(data.y[mask]).sum().item())
            accs.append(correct / int(mask.sum()))

    print(f"Train acc: {accs[0]:.4f}, Val acc: {accs[1]:.4f}, Test acc: {accs[2]:.4f}")