Graph Data Handle
===================

Data Handling of Graphs
-----------------------
Graph data typically includes node connectivity and features. In rLLM, a simple graph data instance is defined by  :obj:`rllm.data.GraphData` .
It generally contains the following information:

- :obj:`data.x`: Node feature matrix, shape: :obj:`[num_nodes, feature_dims]`
- :obj:`data.adj`: Adjacency matrix representing graph structure, shape: :obj:`[num_nodes, num_nodes]`
- :obj:`data.y`: Node labels used for supervised training

An instance for storing graph data can be created as follows:

.. code-block:: python

    import torch
    from rllm.data import GraphData

    x = torch.tensor([[0],[1],[2]]).float()
    y = torch.tensor([0, 1, 2]).long()
    adj = torch.tensor([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
    data = GraphData(x=x, y=y, adj=adj)

The :class:`~rllm.data.GraphData` also provides various convenient functions fro inferring information from graph data and performing operations on it, such as:

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
The :obj:`Transform` module provides a range of methods for modifying and preprocessing graph data features contained in subclasses of :class:`~rllm.data.BaseGraph`, such as :class:`~rllm.data.GraphData` and :class:`~rllm.data.HeteroGraphData`.
These methods can be applied explicitly after initialization or implicitly by specifying them as the :obj:transform parameter when loading datasets. 
Furthermore, the module supports the :class:`~rllm.transforms.GraphTransform` class, which allows users to chain multiple transformation methods for streamlined usage.

.. code-block:: python

    import os.path as osp
    import rllm.transforms.graph_transform as GT
    from rllm.datasets.planetoid import PlanetoidDataset
        
    transform = GT.GraphTransform([
        GT.NormalizeFeatures('l2'), # Normalize node features
        GT.GCNNorm() # add self-loops and row-normalize adjacency
    ])

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    dataset = PlanetoidDataset(path, args.dataset, transform=transform)
