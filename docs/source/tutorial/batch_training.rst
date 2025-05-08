Mini-Batch Training
===============
In machine learning, due to limited computational resources, it is often impractical to train a model using the entire dataset at once.
To address this, a common solution is mini-batch training,
where only a portion of the training data is used in each iteration to update the model parameters, thereby reducing memory usage.

rLLM provides such dataloaders for mini-batch training in :obj:`rllm.dataloader`.
In the following, we will present two examples to illustrate how mini-batch training
is applied in GNNs (Graph Neural Networks) and RTL (Relationl Table Learning), respectively.

Training GNNs with mini-batch
----------------
The GNNs in rLLM are built upon a message-passing architecture, with process in the figure below:

.. image:: ../_static/message_passing.svg
   :width: 600px
   :align: center

It can be observed that, in each layer, the update of the target node depends only on the nodes from the previous layer.
Based on this, we can enable mini-batch training for large graphs with node features that can be loaded into memory but cannot fit entirely into GPU memory at once.

We take neighbor sampling as an example of mini-batch training.
As shown in the figure below, to train a 2-layer GNN in mini-batch mode,
we sample a subset of the 1-hop neighbors of the target node 0 (i.e., nodes 1 and 3),
along with a subset of its 2-hop neighbors (i.e., nodes 2, 0, and 5).

.. image:: ../_static/neighbor.svg
   :width: 600px
   :align: center

Then, the message passing process for mini-batch training is illustrated in the figure below.
Messages are propagated from the 2-hop neighbors to the 1-hop neighbors,
and finally to the target node 0, completing the forward pass.

.. image:: ../_static/mp_batch.svg
   :width: 600px
   :align: center

In rLLM, we provide the :obj:`NeighborLoader` class for neighbor-sampling-based training.
First, load the :obj:`GraphData` and define the :obj:`NeighborLoader`. The :obj:`NeighborLoader` is initialized with:

- A :obj:`GraphData` object to be sampled and batched.

- :obj:`num_neighbors`: Specifies the number of neighbors to sample per layer. For example, :obj:`[10, 5]` means:

  + Layer 1: Sample 10 1-hop neighbors per target node.

  + Layer 2: Sample 5 neighbors per sampled node in layer 1 (i.e. 2-hop neighbors for target nodes).

- :obj:`seeds`: The nodes to sample (here, :obj:`train_mask`).

- :obj:`batch_size`: The number of target nodes per batch.

.. code:: python

    from rllm.datasets import PlanetoidDataset
    from rllm.data import GraphData
    from rllm.dataloader import NeighborLoader

    data: GraphData = PlanetoidDataset(path, args.dataset)[0]

    trainloader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        seeds=data.train_mask,
        batch_size=args.batch_size,
        shuffle=False,
    )

After defining the dataloader, we can proceed with training.
We use the :obj:`GCN` model for training, and the model definition and training process are as follows.
Each iteration of :obj:`trainloader` returns three values:

- :obj:`batch`: The size of the current batch.

- :obj:`n_id`: The node IDs of the sampled subgraph, used to fetch node features from the original graph.

- :obj:`adjs`: A list of sparse matrices representing the edge connections in the neighbor-sampled subgraph for the current batch. These determine the message-passing direction during computation.

The :obj:`NeighborLoader` always places the target nodes at the beginning of the sampled nodes.
Thus, we can obtain the IDs of the current mini-batch's target nodes using :obj:`n_id[:batch]`.

.. code:: python

    class GCN(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, dropout):
            super().__init__()
            self.dropout = dropout
            self.conv1 = GCNConv(in_dim, hidden_dim, normalize=True)
            self.conv2 = GCNConv(hidden_dim, out_dim, normalize=True)

        def forward(self, x, adjs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv1(x, adjs[1]))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, adjs[0])
            return x

        def fulltest(self, x, adj):
            x = F.relu(self.conv1(x, adj))
            x = self.conv2(x, adj)
            return x

    def train():
        model.train()
        all_loss = 0
        for batch, n_id, adjs in trainloader:
            x = data.x[n_id]
            y = data.y[n_id[:batch]]

            optimizer.zero_grad()
            out = model(x, adjs)
            loss = loss_fn(out[:batch], y)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        return all_loss / len(trainloader)


For a complete example, please refer to `[Example code of gcn_batch] <https://github.com/rllm-team/rllm/blob/main/examples/gcn_batch.py>`__.


Training BRIDGE with mini-batch
----------------
Next, we will demonstrate mini-batch RTL model training using :obj:`BRIDGELoader` and the :obj:`BRIDGE` model.
For detailed specifications of the :obj:`BRIDGE` model, please refer to :doc:`Design of RTLs <rtls>`.

:obj:`BRIDGELoader` (a subclass of :obj:`NeighborLoader`) requires three input data for initialization:

- table: :obj:`TableData` object, which is the target table to be sampled and batched.

- non_table: :obj:`Tensor` object, which is the non-table data to be sampled and batched. If there is no non-table data, set it to :obj:`None`.

- graph: :obj:`GraphData` object, which is the graph to be sampled and batched.

Other parameters maintain identical definitions to :obj:`NeighborLoader`, where :obj:`train_mask` is parameter :obj:`seeds`.

.. code:: python

    from rllm.dataloader import BRIDGELoader

    train_loader = BRIDGELoader(
        table=target_table,
        non_table=None,
        graph=graph,
        num_samples=[10, 5],
        train_mask=train_mask,
        batch_size=args.batch_size,
        shuffle=False,
    )


Similarly, we now utilize this :obj:`train_loader` to facilitate training with the `BRIDGE` model.
The batch training process yields five outputs per iteration from :obj:`BRIDGELoader`:

- :obj:`batch`: Size of the current batch.

- :obj:`n_id`: Node IDs of the sampled subgraph.

- :obj:`adjs`: List of sparse matrices representing edge connections in the neighbor-sampled subgraph.

- :obj:`table_data`: Table data for the current batch.

- :obj:`non_table_data`: Non-table data for the current batch.

:obj:`BRIDGELoader` always positions target nodes at the beginning of sampled nodes.
Thus, target node IDs for the current mini-batch can be retrieved via :obj:`n_id[:batch]`.

.. code:: python

    def train() -> float:
        model.train()
        loss_all = 0
        for batch, n_id, adjs, table_data, non_table in train_loader:
            optimizer.zero_grad()
            logits = model(
                table=table_data,
                non_table=non_table,
                adj=adjs,
            )
            loss = F.cross_entropy(
                logits[:batch], table_data.y[:batch].to(torch.long)
            )
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
        return loss_all / len(train_loader)


For a complete example, please refer to `[Example code of bridge_tacm12k_batch] <https://github.com/rllm-team/rllm/blob/main/examples/bridge/bridge_tacm12k_batch.py>`__.
