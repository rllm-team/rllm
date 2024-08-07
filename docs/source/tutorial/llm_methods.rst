Design of LLM Methods
===============


Large language models excel in zero-shot tasks on text.
This tutorial will show how to adopt LLMs as predictors to label some nodes, and then adopt GNNs for node classification.

First, we load the original data and select nodes for annotation.

.. code-block:: python

    import os.path as osp
    import rllm.transforms as T
    from rllm.datasets.tagdataset import TAGDataset
    from node_selection.node_selection import active_generate_mask

    transform = T.Compose([
        T.NormalizeFeatures('l2'),
        T.GCNNorm()
    ])

    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'cached')
    dataset = TAGDataset(path, 'citeseer', use_cache=False, transform=transform, force_reload=True)
    data = dataset[0]
    train_mask, val_mask, test_mask = active_generate_mask(data, method='Random')
Next, we  query LLM for node label predictions and confidence scores.

.. code-block:: python

    import os
    import torch
    from langchain_community.llms import LlamaCpp
    from annotation.annotation import annotate
    from rllm.llm.llm_module.langchain_llm import LangChainLLM

    model_path = "/path/to/llm"
    llm = LangChainLLM(LlamaCpp(model_path=model_path, n_gpu_layers=33))
    pl_indices = torch.nonzero(train_mask | val_mask, as_tuple=False).squeeze()
    data = annotate(data, pl_indices, llm)
Here we define a simple GCN.

.. code-block:: python

    import torch.nn.functional as F
    from rllm.nn.conv.gcn_conv import GCNConv

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
Finally, we use the obtained pseudo-labels for GCN training.

.. code-block:: python

    from tqdm import tqdm

    class Trainer:
        def __init__(self, data, model, optimizer, masks, weighted_loss):
            self.data = data
            self.model = model
            self.optimizer = optimizer
            self.train_mask = masks['train_mask']
            self.val_mask = masks['val_mask']
            self.test_mask = masks['test_mask']
            self.weighted_loss = weighted_loss

        def train(self):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data.x, data.adj)
            loss_fn = torch.nn.CrossEntropyLoss()
            if self.weighted_loss:
                loss = loss_fn(out[train_mask], data.pl[train_mask]) * data.conf[train_mask].mean()
            else:
                loss = loss_fn(out[train_mask], data.pl[train_mask])
            loss.backward()
            self.optimizer.step()
            return loss.item()

        @torch.no_grad()
        def test(self):
            self.model.eval()
            out = self.model(data.x, data.adj)
            pred = out.argmax(dim=1)

            accs = []
            correct = float(pred[train_mask].eq(data.pl[train_mask]).sum().item())
            accs.append(correct / int(train_mask.sum()))

            correct = float(pred[val_mask].eq(data.pl[val_mask]).sum().item())
            accs.append(correct / int(val_mask.sum()))

            correct = float(pred[test_mask].eq(data.y[test_mask]).sum().item())
            accs.append(correct / int(test_mask.sum()))

            return accs

    model = GCN(
        in_channels=data.x.shape[1],
        hidden_channels=64,
        out_channels=data.num_classes,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    masks = {'train_mask': train_mask, 'val_mask':val_mask, 'test_mask': test_mask}

    trainer = Trainer(data, model, optimizer, masks, weighted_loss=True)
    best_val_acc = 0
    best_test_acc = 0
    train_accs = []
    val_accs = []
    test_accs = []
    for epoch in tqdm(range(30)):
        train_loss = trainer.train()

        train_acc, val_acc, test_acc = trainer.test()
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        val_accs.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
    print(f'best test acc: {best_test_acc:.4f}')
