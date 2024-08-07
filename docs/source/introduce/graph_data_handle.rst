Graph Data Handle
===================

Data Handling of Graphs
-----------------------
Graph data typically includes node connectivity and features. In rLLM, a simple graph data instance is defined by  :obj:`rllm.data.GraphData` .
It generally contains the following information:

- :obj:`data.x`: Node feature matrix, shape: :obj:`[num_nodes, feature_dims]`
- :obj:`data.adj`: Adjacency matrix representing graph structure
- :obj:`data.y`: Node labels used for supervised training
We can create an instance to store graph data as follows:

.. code-block:: python

    import torch
    from rllm.data import GraphData

    x = torch.tensor([[0],[1],[2]]).float()
    y = torch.tensor([0, 1, 2]).long()
    adj = torch.tensor([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
    data = GraphData(x=x, y=y, adj=adj)

:class:`~rllm.data.GraphData` also provides various convenient functions to infer information from the graph data and perform operations on it, such as:

.. code-block:: python

    data.train_mask = torch.tensor([True, True, False])
    print(data.train_mask == data['train_mask'])
    >>> True

    data['test_mask'] = torch.tensor([False, False, True])
    print(data.test_mask == data['test_mask'])
    >>> True

    print(data.num_nodes)
    >>> 3

    print(data.num_classes)
    >>> 3

    # transfer data to device.
    data.to('cuda')
    data.to('cpu')

Graph Transforms
-----------------------
The :obj:`Transform` module offers various methods to modify and preprocess graph data features stored in subclasses of  :class:`~rllm.data.BaseGraph` , such as :class:`~rllm.data.GraphData` and :class:`~rllm.data.HeteroGraphData` . These methods can be explicitly called after initialization or implicitly invoked by passing them as :obj:`transform` parameters to dataset loaders. Additionally, the module supports using the :class:`~rllm.transforms.Compose` class to chain multiple methods for simplified usage:

.. code-block:: python

    import os.path as osp
    import rllm.transforms as T
    from rllm.datasets.planetoid import PlanetoidDataset
        
    transform = T.Compose([
    T.NormalizeFeatures('l2'), # Normalize node features
    T.GCNNorm() # add self-loops and row-normalize adjacency
    ])

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    dataset = PlanetoidDataset(path, args.dataset, transform=transform)
