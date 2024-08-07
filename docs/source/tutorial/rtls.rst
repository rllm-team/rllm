Design of RTLs
==============

We can jointly use TNNs and GNNs to address multi-table relational learning problems.

First, let's create datasets with Table-MovieLens1M dataset as an example.

.. code-block:: python

    import os.path as osp
    import torch
    from rllm.datasets import TML1MDataset

    # Prepare datasets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = osp.join(osp.dirname(osp.realpath(__file__)), "data")
    dataset = TML1MDataset(cached_dir=path, force_reload=True)
    # movie_embeddings are obtained from the movie table using LM.
    user_table, movie_table, rating_table, movie_embeddings = dataset.data_list

Then, since user and movie are entities, and rating is an interaction relationship, we need to construct a graph to represent the interaction relationships.

For convenience, we will construct a basic homogeneous graph here, even though movie and user are heterogeneous at the node level.

.. code-block:: python

    import rllm.transforms as T
    from rllm.utils import build_homo_graph

    # Original movie id in datasets is unordered, so we reorder them. 
    movie2id = {user_id: idx for idx, user_id in enumerate(movie_table.df.index.to_numpy())}
    ordered_rating = rating_table.df.assign(MovieID=rating_table.df['MovieID'].map(movie2id))

    # Making simple homogeneous graph
    user_embeddings = torch.ones(len(user_table), movie_embeddings.size(1))
    x = torch.cat([user_embeddings, movie_embeddings], dim=0)
    # simple undirected and unweighted graph
    graph = build_homo_graph(
        df=ordered_rating, 
        A_nodes=len(user_table),	# user amount
        B_nodes=len(movie_table),	# movie amount
        x=x,						# feature tensors
        y=user_table.y.long(),		# label
        transform=T.GCNNorm(),
    )

    graph.user_table = user_table
    graph.movie_table = movie_table
    graph = graph.to(device)
    train_mask, val_mask, test_mask = graph.user_table.train_mask, graph.user_table.val_mask, graph.user_table.test_mask
    output_dim = graph.user_table.num_classes

After initializing the data, we define the model. Since the task of the TML1M dataset is user age classification, we perform TNN (Table Neural Network) only on the user table and extract embeddings for all users through GNN (Graph Neural Network).

.. code-block:: python

    class Bridge(torch.nn.Module):
        def __init__(
            self, table_hidden_dim, table_output_dim, graph_hidden_dim, graph_output_dim, stats_dict
        ):
            super().__init__()
            self.table_encoder = TabTransformer(
                hidden_dim = table_hidden_dim,
                output_dim = table_output_dim,
                col_stats_dict = stats_dict,
            )
            self.graph_encoder = torch.nn.ModuleList(
                [
                    GCNConv(table_output_dim, graph_hidden_dim),
                    GCNConv(graph_hidden_dim, graph_output_dim),
                ]
            )

        def forward(self, graph):
            feat_dict = graph.user_table.get_feat_dict() # A dict contains feature tensor.
            x_user = self.table_encoder(feat_dict)
            x = torch.cat([x_user, graph.x[len(graph.user_table):len(graph.user_table)+len(graph.movie_table), :]], dim=0)
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.graph_encoder[0](x, graph.adj))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.graph_encoder[1](x, graph.adj)
            return x[: len(graph.user_table), :]

    model = Bridge(
        table_hidden_dim=128,
        table_output_dim=movie_embeddings.size(1),
        graph_hidden_dim=128,
        graph_output_dim=output_dim,
        stats_dict=graph.user_table.stats_dict,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())

Finally, we jointly train the model and evaluate the results on the test set.

.. code-block:: python

    for epoch in range(50):
        pred = model(graph)
        loss = F.cross_entropy(
        pred[train_mask].squeeze(), graph.y[train_mask]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    logits = model(graph)
    preds = logits.argmax(dim=1)
    acc = (preds[test_mask] == y[test_mask]).sum(dim=0) / len(test_mask)
    print(f'Accuracy: {acc:.4f}')
    >>> 0.3860