Design of RTLs
==============

What is RTL?
----------------
In machine learning, **Relational Table Learning (RTL)** refers to the process of learning from data organized in multiple interconnected tables, as seen in a relational database. These tables are linked through primary and foreign key relationships, enabling the extraction of meaningful patterns and insights. A typical RTL architecture consists of one or more Transforms followed by multiple Convolution layers, as detailed in :doc:`Understanding Transforms <transforms>` and :doc:`Understanding Convolutions <convolutions>`.


Construct a BRIDGE
----------------
In this tutorial, we will describe how to train a BRIDGE on multiple interconnected tables. The BRIDGE, proposed in the paper `[rLLM: Relational Table Learning with LLMs] <https://arxiv.org/abs/2407.20157>`__, integrates TNNs and GNNs to learn from both tabular features and non-tabular features in relational table data. Next, we will construct a BRIDGE and evaluate its node classification performance on the :obj:`Table-MovieLens1M` dataset(hereafter referred to as :obj:`TML1M`).

First, we load the :obj:`TML1M` dataset and extract the necessary tables.

.. code-block:: python

    import os.path as osp
    import torch
    from rllm.datasets import TML1MDataset

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    path = osp.join(osp.dirname(osp.realpath(__file__)), "data")
    dataset = TML1MDataset(cached_dir=path, force_reload=True)
    
    # Get the required data
    (
        user_table,
        _,
        rating_table,
        movie_embeddings,
    ) = dataset.data_list
    emb_size = movie_embeddings.size(1)
    user_size = len(user_table)

Then, since users and movies are entities, and ratings capture their interactions, we need to construct a graph to represent these relationships.

For convenience, we will construct a basic homogeneous graph here, although a heterogeneous graph would be more appropriate given the inherent heterogeneity of user and movie nodes.

.. code-block:: python

    from examples.bridge.utils import build_homo_graph, reorder_ids

    # Original movie id in datasets is unordered, so we reorder them. 
    ordered_rating = reorder_ids(
        relation_df=rating_table.df,
        src_col_name="UserID",
        tgt_col_name="MovieID",
        n_src=user_size,
    )
    target_table = user_table.to(device)
    y = user_table.y.long().to(device)
    movie_embeddings = movie_embeddings.to(device)
    # Build graph
    graph = build_homo_graph(
        relation_df=ordered_rating,
        n_all=user_size + movie_embeddings.size(0),
    ).to(device)

Additionally, the :obj:`BRIDGE` method necessitates distinct transformations for the table and graph data. After data processing, the dataset is split into training, validation, and test sets.

.. code-block:: python

    from rllm.transforms.graph_transforms import GCNTransform
    from rllm.transforms.table_transforms import TabTransformerTransform

    # Transform data
    table_transform = TabTransformerTransform(
        out_dim=emb_size, metadata=target_table.metadata
    )
    target_table = table_transform(target_table)
    graph_transform = GCNTransform()
    adj = graph_transform(graph).adj

    # Split data
    train_mask, val_mask, test_mask = (
        user_table.train_mask,
        user_table.val_mask,
        user_table.test_mask,
    )

After processing the data, we instantiate the BRIDGE model. Since the task of the TML1M dataset is user age classification, we perform :obj:`TableEncoder` only on the user table, while using the precomputed embeddings for movies. We then employ the :obj:`GraphEncoder` to learn representations for both users and movies.

.. code-block:: python
    
    from rllm.nn.conv.graph_conv import GCNConv
    from rllm.nn.conv.table_conv import TabTransformerConv
    from rllm.nn.models import BRIDGE, TableEncoder, GraphEncoder

    # Set up model and optimizer
    t_encoder = TableEncoder(
        in_dim=emb_size,
        out_dim=emb_size,
        table_conv=TabTransformerConv,
        metadata=target_table.metadata,
    )
    g_encoder = GraphEncoder(
        in_dim=emb_size,
        out_dim=target_table.num_classes,
        graph_conv=GCNConv,
    )
    model = BRIDGE(
        table_encoder=t_encoder,
        graph_encoder=g_encoder,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())

Finally, we train the model on the training and validation sets and evaluate the results on the test set.

.. code-block:: python

    for epoch in range(50):
        optimizer.zero_grad()
        logits = model(
            table=user_table,
            non_table=movie_embeddings,
            adj=adj,
        )
        loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        logits = model(
            table=user_table,
            non_table=movie_embeddings,
            adj=adj,
        )
        preds = logits.argmax(dim=1)
        acc = (preds[test_mask] == y[test_mask]).sum(dim=0) / test_mask.sum()
        
    print(f'Accuracy: {acc:.4f}')
    >>> 0.3860
