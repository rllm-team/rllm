Design of LLM Methods
=====================

Large language models (LLMs) excel at zero-shot tasks involving text.
In this tutorial, we demonstrate how to use LLMs as predictors to label a subset of entities in relational tables, followed by applying BRIDGE model for relational machine learning.

We begin by loading the original dataset, building a graph and selecting nodes by degree for annotation.

.. code-block:: python

    import os.path as osp
    import random

    import torch
    import networkx as nx
    import dashscope
    from langchain_community.llms import Tongyi

    from examples.bridge.bridge import build_bridge_model, train_bridge_model
    from examples.bridge.utils import data_prepare
    from rllm.datasets import TLF2KDataset
    from rllm.llm import Predictor
    from rllm.llm.llm_module.langchain_llm import LangChainLLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")

    dataset = TLF2KDataset(cached_dir=path, force_reload=True)

    target_table, non_table_embeddings, adj, emb_size = data_prepare(dataset, "tlf2k", device)
    _, val_mask, test_mask = (
    target_table.train_mask.cpu(),
    target_table.val_mask.cpu(),
    target_table.test_mask.cpu(),
    )
    label_names = sorted(set(map(str, target_table.df[target_table.target_col].tolist())))
    select_mask = ~(test_mask | val_mask)
    train_num = min(100, select_mask.sum())
    val_num = min(100, val_mask.sum())

    target_nodes = list(range(len(target_table.df)))
    edges = adj.coalesce().indices().t().tolist()
    G = nx.Graph(edges)
    target_degrees = [(node, G.degree(node)) for node in target_nodes if select_mask[node]]
    target_degrees.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in target_degrees][:train_num]
    train_indices = torch.tensor(top_nodes, dtype=torch.long)
    train_mask = torch.zeros(target_table.df.shape[0], dtype=torch.bool)
    train_mask[train_indices] = True
    target_table.train_mask = train_mask

    val_indices = torch.nonzero(val_mask).squeeze()
    val_indices = val_indices[torch.randperm(len(val_indices))[:val_num]]
    val_mask[:] = False
    val_mask[val_indices] = True
    target_table.val_mask = val_mask

    mask = train_mask | val_mask

Next, we query LLM for label predictions.

.. code-block:: python

    pseudo_labels = -1 * torch.ones(target_table.df.shape[0], dtype=torch.long)

    df = target_table.df.loc[mask.cpu().numpy()].drop(columns=[target_table.target_col])

    scenario = "Classify the artists into one of the given labels."
    labels = ", ".join(label_names)

    DASHSCOPE_API_KEY = "your-api-key"
    llm = Tongyi(dashscope_api_key=DASHSCOPE_API_KEY, model_kwargs={"api_key": DASHSCOPE_API_KEY, "model": "qwen-max-2025-01-25"}, client=dashscope.Generation)

    predictor = Predictor(llm=LangChainLLM(llm), type="classification")
    outputs = predictor(df, scenario=scenario, labels=labels)

    select_pred = []
    for output in outputs:
        output = output.lower()
        matches = []
        for label in label_names:
            if label.lower() in output.lower():
                matches.append(label)
        if matches:
            matched = max(matches, key=len)
        else:
            matched = random.choice(label_names)
        select_pred.append(label_names.index(matched))

    select_pred = torch.tensor(select_pred)
    pseudo_labels[mask] = select_pred

Finally, we use the obtained pseudo-labels to train a BRIDGE model.

.. code-block:: python

    real_labels = torch.tensor([label_names.index(_) if _ in label_names else random.choice(label_names) for _ in target_table.df[target_table.target_col].astype(str).tolist()])
    y = real_labels.long().to(device)
    y[train_mask | val_mask] = pseudo_labels.long().to(device)[train_mask | val_mask]
    target_table.y = y

    model = build_bridge_model(target_table.num_classes, target_table.metadata, emb_size).to(device)
    train_bridge_model(model, target_table, non_table_embeddings, adj, 100, 0.001, 1e-4)
