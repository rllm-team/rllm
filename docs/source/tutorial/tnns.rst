Design of TNNs
===============
What is TNN?
----------------
In machine learning, **Table/Tabular Neural Networks (TNNs)** are recently emerging neural networks specifically designed to process tabular data. In a TNN, the input is structured tabular data, usually organized in rows and columns. A typical TNN architecture consists of an initial Transform followed by multiple Convolution layers, as detailed in *Understanding Transforms* and *Understanding Convolutions*.


Construct a TabTransformer
----------------
In this tutorial, we will learn the basic workflow of using `[TabTransformer] <https://arxiv.org/abs/2012.06678>`__ for tabular classification, i.e., predicting the category of a row in a table.

First, we use the :obj:`Titanic` dataset as an example, which can be loaded using the built-in dataloaders. Also, we instantiate a :obj:`TabTransformerTransform`, corresponding to the :obj:`TabTransformer` method. After applying the transformation and shuffling the data, we proceed to split the dataset into training, testing, and validation sets, following standard practices in deep learning.
.. code-block:: python

    import argparse
    import os.path as osp

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from rllm.types import ColType
    from rllm.datasets import Titanic
    from rllm.transforms.table_transforms import TabTransformerTransform

    # Set random seed and device
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
    data = Titanic(cached_dir=path)[0]

    # Transform data
    transform = TabTransformerTransform(out_dim=args.emb_dim)
    data = transform(data).to(device)
    data.shuffle()

    # Split dataset, here the ratio of train-val-test is 80%-10%-10%
    train_loader, val_loader, test_loader = data.get_dataloader(
        train_split=0.8, val_split=0.1, test_split=0.1, batch_size=args.batch_size
    )

Next, we construct a simple :obj:`TabTransformer` model using the :obj:`TabTransformerConv` layer. Note that the first layer needs to pass in the metadata for initialization of the pre-encoder.

.. code-block:: python
    
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

            self.convs = torch.nn.ModuleList()
            self.convs.append(
                TabTransformerConv(
                    conv_dim=hidden_dim,
                    num_heads=num_heads,
                    use_pre_encoder=True,
                    metadata=metadata,
                )
            )
            for _ in range(num_layers - 1):
                self.convs.append(
                    TabTransformerConv(conv_dim=hidden_dim, num_heads=num_heads)
                )

            self.fc = torch.nn.Linear(hidden_dim, out_dim)

        def forward(self, x):
            for conv in self.convs:
                x = conv(x)
            x = torch.cat(list(x.values()), dim=1)
            out = self.fc(x.mean(dim=1))
            return out
            
    # Set up model and optimizer
    model = TabTransformer(
        hidden_dim=args.emb_dim,
        out_dim=data.num_classes,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        metadata=data.metadata,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )


Finally, we need to implement a :obj:`train()` function and a :obj:`test()` function, the latter of which does not require gradient tracking. The model can then be trained on the training and validation sets, and the classification results can be obtained from the test set.

.. code-block:: python
    
    import time

    def train(epoch: int) -> float:
        model.train()
        loss_accum = total_count = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch: {epoch}"):
            x, y = batch
            pred = model.forward(x)
            loss = F.cross_entropy(pred, y.long())
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * y.size(0)
            total_count += y.size(0)
            optimizer.step()
        return loss_accum / total_count


    @torch.no_grad()
    def test(loader: DataLoader) -> float:
        model.eval()
        correct = total = 0
        for batch in loader:
            feat_dict, y = batch
            pred = model.forward(feat_dict)
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        accuracy = correct / total
        return accuracy

    metric = "Acc"
    best_val_metric = best_test_metric = 0
    times = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = train(epoch)
        train_metric = test(train_loader)
        val_metric = test(val_loader)
        test_metric = test(test_loader)

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric

        times.append(time.time() - start)
        print(
            f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
            f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}"
        )

    print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"Total time: {sum(times):.4f}s")
    print(
        f"Best Val {metric}: {best_val_metric:.4f}, "
        f"Best Test {metric}: {best_test_metric:.4f}"
    )
