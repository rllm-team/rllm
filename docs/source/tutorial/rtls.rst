Design of RTLs
==============

What is a RTL?
----------------
In machine learning, **Relational Table Learnings (RTLs)** typically refers to the learning of relational table data, which consists of multiple interconnected tables with significant heterogeneity. In an RTL, the input comprises multiple table signals that are interrelated.  A typical RTL architecture consists of one or more Transforms followed by multiple Convolution layers, as detailed in *Understanding Transform* and *Understanding Convolution*.


Construct a BRIDGE
----------------
We can jointly construct a `[BRIDGE] <https://arxiv.org/abs/2407.20157>`__ using TNNs and GNNs to address multi-table relational learning problems.

First, let's create datasets with Table-MovieLens1M dataset as an example.

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

Then, since user and movie are entities, and rating is an interaction relationship, we need to construct a graph to represent the interaction relationships.

For convenience, we will construct a basic homogeneous graph here, even though movie and user are heterogeneous at the node level.

.. code-block:: python

    from utils import build_homo_graph, reorder_ids

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

Additionally, the :obj:`BRIDGE` method requires separate transformations for the table data and the graph data. Similarly, after data processing, the dataset is split into training, validation, and test sets.

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


After initializing the data, we instantiate the model. Since the task of the TML1M dataset is user age classification, we perform :obj:`TableEncoder` only on the user table and extract embeddings for all users through :obj:`GraphEncoder`.

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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

Finally, we need to implement a :obj:`train()` function and a :obj:`test()` function, the latter of which does not require gradient tracking. The model can then be trained on the training and validation sets, and the classification results can be obtained from the test set.

.. code-block:: python

    def train() -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(
        table=user_table,
        non_table=movie_embeddings,
        adj=adj,
    )
    loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

    @torch.no_grad()
    def test():
        model.eval()
        logits = model(
            table=user_table,
            non_table=movie_embeddings,
            adj=adj,
        )
        preds = logits.argmax(dim=1)

        accs = []
        for mask in [train_mask, val_mask, test_mask]:
            correct = float(preds[mask].eq(y[mask]).sum().item())
            accs.append(correct / int(mask.sum()))
        return accs

    start_time = time.time()
    best_val_acc = best_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train()
        train_acc, val_acc, test_acc = test()
        print(
            f"Epoch: [{epoch}/{args.epochs}]"
            f"Loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"val_acc: {val_acc:.4f} test_acc: {test_acc:.4f} "
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    print(f"Total Time: {time.time() - start_time:.4f}s")
    print(
        "BRIDGE result: "
        f"Best Val acc: {best_val_acc:.4f}, "
        f"Best Test acc: {best_test_acc:.4f}"
    )