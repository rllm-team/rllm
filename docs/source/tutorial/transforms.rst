Understanding Transform
===============

What is a Transform?
----------------
In machine learning, transform generally refers to the essential preprocessing steps applied to data before using various methods. For instance, in Graph Neural Networks (GNNs), the graph adjacency matrix is often normalized. Similarly, in Tabular Neural Networks (TNNs), numerical features are scaled, and categorical features are embedded using one-hot encoding. These operations do not involve any trainable parameters and only need to be performed once before training.

When creating a :obj:`Transform` module in rLLM, we recommend implementing each atomic operation first and then assembling them, much like building with blocks. The advantage of this approach is that it makes the module easier to maintain and extend.


Construct a GCNTransform
----------------

:obj:`GCNTransform` is introduced in the classic GNN method, GCN. It processes node features by applying :obj:`L1 normalization` along the rows and performs :obj:`symmetric normalization` on the adjacency matrix after :obj:`adding self-loops`.

First, we implement the row normalization function :obj:`normalize_features`. This function supports three types of normalization: 'L1', 'L2', and 'sum' normalization.

.. code-block:: python

    def normalize_features(X: Tensor, norm: str = "l2", return_norm: bool = False):
        if X.is_sparse:
            X = X.to_dense()

        if norm == "l1":
            norms = LA.norm(X, ord=1, dim=1, keepdim=True)
        elif norm == "l2":
            norms = LA.norm(X, dim=1, keepdim=True)
        elif norm == "sum":
            X -= X.min()
            norms = X.sum(dim=-1, keepdim=True)

        X = X.div_(norms.clamp_(min=1.0))

        if return_norm:
            norms = norms.squeeze(1)
            return X, norms
        else:
            return X

Next, we extend this function into a class :obj:`NormalizeFeatures`. The class needs to inherit from a base class: the transform that operates on the nodes inherits from :obj:`NodeTransform`, while the one that operates on edges inherits from :obj:`EdgeTransform`.

.. code-block:: python

    class NormalizeFeatures(NodeTransform):
        def __init__(self, norm: str = "l2"):
            self.norm = norm

        def forward(self, x: Tensor) -> Tensor:
            return normalize_features(x, self.norm)

Similarly, we can implement additional operations, such as :obj:`adding self-loops` and :obj:`symmetric normalization`, and organize them into a unified :obj:`GCNNorm` module for convenience.

.. code-block:: python

    class GCNNorm(EdgeTransform):
        def __init__(self):
            self.data = None

        @lru_cache()
        def forward(self, adj: Tensor) -> Tensor:
            adj = add_remaining_self_loops(adj)
            return symmetric_norm(adj)

Finally, :obj:`GCNTransform` inherits from the :obj:`GraphTransform` class. Therefore, we simply pass a list of transformations to the parent class.

.. code-block:: python

    class GCNTransform(GT.GraphTransform):

    def __init__(self, normalize_features: str = "l1"):
        super().__init__(
            transforms=[
                GT.NormalizeFeatures(normalize_features),
                GT.GCNNorm(),
            ]
        )

Construct a TabTransformerTransform
----------------
:obj:`TabTransformer` is a typical Transformer-based deep learning method for tabular data. In addition to the default handling of missing values, :obj:`TabTransformerTransform` also performs dimensionality expansion (also called pre-encoding in our project) on numerical features. Currently, the submodules of :obj:`TableTransform` are relatively simple, so they are not abstracted into separate functions.

First, we implement the :obj:`StackNumerical` transform, which inherits from the :obj:`ColTransform` base class. This transform initially applies standard normalization to the input columns, followed by dimensionality expansion.

.. code-block:: python

    class StackNumerical(ColTransform):
        def __init__(
            self,
            out_dim: int,
        ) -> None:
            self.out_dim = out_dim

        def forward(
            self,
            data: TableData,
        ) -> TableData:
            if ColType.NUMERICAL in data.feat_dict.keys():

                metadata = data.metadata[ColType.NUMERICAL]
                self.mean = torch.tensor([stats[StatType.MEAN] for stats in metadata])
                self.std = torch.tensor([stats[StatType.STD] for stats in metadata]) + 1e-6

                feat = data.feat_dict[ColType.NUMERICAL]
                feat = (feat - self.mean) / self.std

                data.feat_dict[ColType.NUMERICAL] = feat.unsqueeze(2).repeat(
                    1, 1, self.out_dim
                )
            return data

Next, the :obj:`TabTransformerTransform` class inherits from the :obj:`TableTransform` base class. The :obj:`TableTransform` class provides a foundation for table transformations, with its default behavior being the imputation of missing values. Additionally, :obj:`TableTransform` requires a member variable, metadata, which must be explicitly defined within :obj:`TabTransformerTransform`.

.. code-block:: python

    class TabTransformerTransform(TableTransform):
        def __init__(
            self,
            out_dim: int,
            metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        ) -> None:
            super().__init__(
                out_dim=out_dim,
                transforms=[StackNumerical(out_dim)],
            )
            self.metadata = metadata

The :obj:`TableTransform` class can also support custom methods; for instance, :obj:`TabTransformerTransform` defines its own reset_parameters method.

.. code-block:: python

        def reset_parameters(self) -> None:
            super().reset_parameters()
            for transform in self.transforms:
                transform.reset_parameters()
