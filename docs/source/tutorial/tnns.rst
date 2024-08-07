Design of TNNs
===============

Tabular/Table Neural Networks (TNNs) are deep learning architectures designed for table data learning. In this tutorial, you will learn the basic workflow of using TNNs for tabular classification, i.e., predicting the category of a row in a table.


First, we use the Titanic dataset as an example. We can load this built-in dataset with dataloaders. We also split the dataset into train/test/validation parts, following a standard deep learning approach.

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from rllm.datasets import Titanic
    from rllm.types import ColType

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Titanic('data', forced_reload=True)[0]
    dataset = dataset.to(device)
    # Create train-validation-test split 
    train_dataset, val_dataset, test_dataset = dataset.get_dataset(0.8, 0.1, 0.1)
    train_loader = DataLoader(train_dataset, batch_size=128,
                            shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

Next, we create a very simple TNN model using `[TabTransformer] <https://arxiv.org/abs/2012.06678>`__ :

.. code-block:: python
    
    from rllm.nn.models import TabTransformer

    model = TabTransformer(
        hidden_dim = 32,            
        output_dim = dataset.num_classes,      
        layers = 2,     
        heads = 8,      
        col_stats_dict = dataset.stats_dict
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())

Finally, we train our model and get the classification results on the test set.

.. code-block:: python

    for epoch in range(50):
        for batch in train_loader:
            feat_dict, y = batch
            pred = model(feat_dict)
            loss = F.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    for tf in test_loader:
        feat_dict, y = batch
        pred = model(feat_dict)
        pred_class = pred.argmax(dim=-1)
        correct += (y == pred_class).sum()
    acc = int(correct) / len(test_dataset)
    print(f'Accuracy: {acc:.4f}')
    >>> 0.7279
