Design of TNNs
===============
What is TNN?
----------------
In machine learning, **Table/Tabular Neural Networks (TNNs)** are recently emerging neural networks specifically designed to process tabular data. In a TNN, the input is structured tabular data, usually organized in rows and columns. A typical TNN architecture consists of an initial Transform followed by multiple Convolution layers, as detailed in :doc:`Understanding Transforms <transforms>` and :doc:`Understanding Convolutions <convolutions>`.


Construct a TabTransformer
----------------
In this tutorial, we will learn the basic workflow of using `[TabTransformer] <https://arxiv.org/abs/2012.06678>`__ for tabular classification, i.e., predicting the category of a row in a table. Next, we will build a TabTransformer and use it to perform node classification on the :obj:`Titanic` dataset.

First, we load the :obj:`Titanic` dataset. Also, we instantiate a :obj:`TabTransformerTransform`, corresponding to the :obj:`TabTransformer` method. After applying the transformation and shuffling the data, we proceed to split the dataset into training, testing, and validation sets, following standard practices in deep learning.

.. code-block:: python

    import os.path as osp

    import torch
    import torch.nn.functional as F

    from rllm.types import ColType
    from rllm.datasets import Titanic
    from rllm.transforms.table_transforms import TabTransformerTransform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = osp.join(osp.dirname(osp.realpath(__file__)), "data")
    data = Titanic(cached_dir=path, forced_reload=True)[0]

    # Transform data
    emb_dim = 32
    transform = TabTransformerTransform(out_dim=emb_dim)
    data = transform(data).to(device)
    data.shuffle()

    # Split dataset, here the ratio of train-val-test is 80%-10%-10%
    train_loader, val_loader, test_loader = data.get_dataloader(
        train_split=0.8, val_split=0.1, test_split=0.1, batch_size=128
    )

Next, we construct a simple :obj:`TabTransformer` model. Note that we first use a :obj:`TabTransformerPreEncoder`, which requires the ``metadata`` for initialization to process the inputs. This is followed by multiple :obj:`TabTransformerConv` layers. Finally, the processed categorical and numerical features are flattened and concatenated before being passed through an MLP.

.. code-block:: python
    
    from typing import Any, Dict, List
    from rllm.nn.encoder import TabTransformerPreEncoder
    from rllm.nn.conv.table_conv import TabTransformerConv
    
    # Define model
    class TabTransformer(torch.nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            num_heads: int,
            metadata: Dict[ColType, List[Dict[str, Any]]],
        ):
            super().__init__()

            self.pre_encoder = TabTransformerPreEncoder(
                out_dim=hidden_dim,
                metadata=metadata,
            )
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    TabTransformerConv(conv_dim=hidden_dim, num_heads=num_heads)
                )

            self.mlp = torch.nn.Linear(
                len(metadata[ColType.CATEGORICAL]) * hidden_dim
                + len(metadata[ColType.NUMERICAL]),
                out_dim,
            )

        def forward(self, x):
            x = self.pre_encoder(x, return_dict=True)
            for conv in self.convs:
                x = conv(x)
            x[ColType.CATEGORICAL] = x[ColType.CATEGORICAL].flatten(1)
            x[ColType.NUMERICAL] = x[ColType.NUMERICAL].flatten(1)
            x = torch.cat(list(x.values()), dim=1)
            out = self.mlp(x)
            return out

We can initialize the model and optimizer.

.. code-block:: python
          
    # Set up model and optimizer
    model = TabTransformer(
        hidden_dim=emb_dim,
        out_dim=data.num_classes,
        num_layers=2,
        num_heads=8,
        metadata=data.metadata,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=5e-4,
    )


Finally, we train our model and get the classification results on the test set.

.. code-block:: python
    
    for epoch in range(50):
        for batch in train_loader:
            x, y = batch
            pred = model(x)
            loss = F.cross_entropy(pred, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for tf in test_loader:
            x, y = batch
            pred = model(x)
            pred_class = pred.argmax(dim=-1)
            correct += (y == pred_class).sum()
            total += len(y)
        acc = int(correct) / total
        
    print(f'Accuracy: {acc:.4f}')
    >>> 0.8356
