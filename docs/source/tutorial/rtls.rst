Design of RTLs
==============

What is RTL?
----------------
In machine learning, **Relational Table Learning (RTL)** refers to the process of learning from data organized in multiple interconnected tables, as seen in a relational database. These tables are linked through primary and foreign key relationships, enabling the extraction of meaningful patterns and insights. A typical RTL architecture consists of one or more Transforms followed by multiple Convolution layers, as detailed in :doc:`Understanding Transforms <transforms>` and :doc:`Understanding Convolutions <convolutions>`.


Construct a BRIDGE
----------------
In this tutorial, we will describe how to train a BRIDGE on multiple interconnected tables. The BRIDGE, proposed in the paper `[rLLM: Relational Table Learning with LLMs] <https://arxiv.org/abs/2407.20157>`__, integrates TNNs and GNNs to learn from both tabular features and non-tabular features in relational table data. Next, we will construct a BRIDGE and evaluate its node classification performance on the :obj:`Table-MovieLens1M` dataset(hereafter referred to as :obj:`TML1M`).

First, we load the :obj:`TML1M` dataset and extract the necessary tables. Then, we use the ``data_prepare`` utility to extract the essential components for our model, including the target table, non-tabular embeddings, the adjacency matrix, and the embedding size.

.. code-block:: python

    import os.path as osp
    import torch
    import torch.nn.functional as F
    from rllm.datasets import TML1MDataset
    from examples.bridge.utils import data_prepare

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data")
    dataset = TML1MDataset(cached_dir=path, force_reload=True)
    
    # Prepare target table, non-table embeddings, graph adjacency, and embedding size
    target_table, non_table_embeddings, adj, emb_size = data_prepare(
        dataset, "tml1m", device
    )

Then, we split the target table into training, validation, and test masks for supervision and evaluation.

.. code-block:: python

    # Split data
    train_mask, val_mask, test_mask = (
        target_table.train_mask,
        target_table.val_mask,
        target_table.test_mask,
    )

After processing the data, we instantiate the BRIDGE model. Since the task in the TML1M dataset is user age classification, we apply :obj:`TableEncoder` only to the target table (which is the user table), while using precomputed embeddings for movies. We then use :obj:`GraphEncoder` to learn representations for both users and movies.

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

Finally, we train the model on the training and validation sets and evaluate the results on the test set.

.. code-block:: python

    y = target_table.y
    best_val_acc = 0
    test_acc_at_best_val = 0
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(
            table=target_table,
            non_table=non_table_embeddings,
            adj=adj,
        )
        loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
        loss.backward()
        optimizer.step()

        val_acc = (logits[val_mask].argmax(dim=1) == y[val_mask]).sum(dim=0) / val_mask.sum()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc_at_best_val = (logits[test_mask].argmax(dim=1) == y[test_mask]).sum(dim=0) / test_mask.sum()
        
    print(f"Accuracy:{test_acc_at_best_val:.4f}")
    >>> 0.3860
