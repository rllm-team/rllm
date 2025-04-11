Understanding Convolution
===============

What is a Convolution?
----------------
In machine learning, convolution generally involves combining an input signal with a filter to produce an output signal.
Specifically, for image signals, convolution refers to aggregating nearby pixels around the central pixel.
For graph signals, it involves aggregating information from connected nodes around the central node.
For table signals, it refers to aggregating entries from different columns within each row.
Therefore, designing an effective convolution operation is a key challenge in deep learning methods.


Construct a GCN Convolution Layer
----------------
Graph Convolutional Networks (GCNs) are a classic type of Graph Neural Network,
where the convolution operation is applied to node features based on the input adjacency matrix.
The formula of :obj:`GCNConv` layer is defined as :math:`\tilde A X W`,
where :math:`\tilde A` is the normalized adjacency matrix with added self-loops, :math:`X` represents the node features, and :math:`W` is the parameter matrix..

Before delving into the details of the :obj:`GCNConv` class, it is important to first understand the structure of the :obj:`MessagePassing` class.
:obj:`MessagePassing` is the base class for all graph convolution layers implemented in `rllm.nn.conv.graph_conv`, including :obj:`GCNConv`.
It consists of three main steps: message computation ( :math:`\text{Message}` ), aggregation ( :math:`\text{Aggregate}` ), and update ( :math:`\text{Update}` ) as shown below:

.. math::
    \mathbf{x}_i^{(k+1)} = \text{Update}^{(k)}
    \left( \mathbf{x}_i^{(k)},
    \text{Aggregate}^{(k)} \left( \left\{ \text{Message}^{(k)} \left(
    \mathbf{x}_i^{(k)}, \mathbf{x}_j^{(k)}, \mathbf{e}_{j,i}^{(k)}
    \right) \right\}_{j \in \mathcal{N}(i)} \right) \right)

As described above, the formulation of :obj:`GCNConv` based on the :obj:`MessagePassing` is given by the following equation:

.. math::
    \mathbf{x}_i^{(k+1)} = \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{\deg(i) \deg(j)}} \mathbf{x}_j^{(k)}

Where the :math:`\sum`` operation corresponds to the aggregation step, and the term :math:`\frac{1}{\sqrt{\deg(i) \deg(j)}}` serves as the normalization factor.
Here, :math:`\deg(i)` and :math:`\deg(j)` denote the degrees of nodes :math:`i` and :math:`j`, respectively.
The message computation function simply retrieves the neighboring nodes of :math:`\mathbf{x}_i^{(k)}` at the current layer and returns the message vectors from nodes :math:`\mathbf{x}_j^{(k)}` to :math:`\mathbf{x}_i^{(k)}`.
The aggregation step combines the retrieved neighbor information according to a specified rule, producing the aggregated message received by node :math:`\mathbf{x}_i^{(k)}` at the current layer.
And the update step then assigns the aggregated message to the next-layer representation of node :math:`i`, denoted as :math:`\mathbf{x}_i^{(k+1)}`.

Next, We examine the implementation of the :obj:`GCNConv` class, which inherits from the :obj:`MessagePassing` base class and consists of two methods:  :obj:`__init__()` and :obj:`forward()`.

.. code-block:: python

    class GCNConv(MessagePassing):
        def __init__(self, in_dim, out_dim, bias):
            ...

        def forward(self, x, edge_index, edge_weight, dim_size) -> Tensor:
            ...

The :obj:`__init__()` method is responsible for initializing the parameters of the :obj:`GCNConv` layer.
It takes two main parameters: :obj:`in_dim` (the input dimension) and :obj:`out_dim` (the output dimension).
These parameters are used to initialize the weight matrix :math:`W`.
Additionally, a bias parameter :obj:`bias` can be included, which determines whether or not to use bias in the convolution operation.
Importantly, the :obj:`GCNConv` layer uses the 'gcn' aggregation method to initialize :obj:`MessagePassing`, which can be modified to use other aggregation strategies (e.g., 'mean').

.. code-block:: python

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__(aggr='gcn')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = Linear(in_dim, out_dim, bias=False)
        if bias:
            self.bias = Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

The :obj:`forward()` method defines the forward pass of the :obj:`GCNConv` layer. Its parameters include the node feature :obj:`inputs` (denoted as :math:`X` in the formula) and the adjacency matrix or edge list :obj:`edge_index` (denoted as :math:`\tilde{A}` in the formula).
First, the input node features are passed through a linear transformation via :obj:`self.linear` to produce the transformed features :obj:`x`.
Next, the :obj:`propagate()` method is called to perform the three message passing steps: message computation, aggregation, and update.
Finally, if the :obj:`bias` parameter is not None, the bias term is added to the output features.

.. code-block:: python

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        edge_weight: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        x = self.linear(x)
        out = self.propagate(x, edge_index, edge_weight=edge_weight, dim_size=dim_size)
        if self.bias is not None:
            out += self.bias
        return out

If we go deeper into the :obj:`propagate()` method, we can see that it calls the :obj:`message()`, :obj:`aggregate()`, and :obj:`update()` methods in sequence.

.. code-block:: python

    def propagate(self, x, edge_index, **kwargs) -> Tensor:
        ... # omitted for brevity
        out = self.message(**msg_kwargs)  # 1. Compute messages
        ...
        out = self.aggregate(out, **aggr_kwargs)  # 2. Aggregate
        ...
        out = self.update(out, **update_kwargs)  # 3. Update
        return out

    def message(self, x, edge_index, edge_weight) -> Tensor:
        # In default, retrieve and return the node feature of the neighbor node
        ...

    def aggregate(self, msgs, edge_index, ...) -> Tensor:
        # Call `self.aggr_module` to aggregate the messages, for GCNConv, it is the 'gcn' aggregator (i.e., sum)
        ...

    def update(self, aggr_out: Tensor) -> Tensor:
        # In default, just return the aggregated message
        ...

To construct a different type of convolutional layer,
you can subclass the :obj:`MessagePassing` class,
define the :obj:`__init__` and :obj:`forward` methods,
and override the :obj:`message`, :obj:`aggregate`, and :obj:`update` functions as needed.
This approach provides flexibility in customizing the message passing mechanism to suit specific graph neural network architectures.

In addition to the :obj:`__init__()` and :obj:`forward()` methods, we can define custom methods as needed.
For example, the :obj:`GCNConv` class can include a :obj:`reset_parameters()` method, which reinitializes the layer's parameters (i.e., the weight matrix :math:`W`) to their original values.

.. code-block:: python

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


Construct a TabTransformer Convolution Layer
----------------
TabTransformer is a classic Tabular/Table Neural Network that relies on the attention mechanism from Transformers to perform column-wise convolution.
It focuses exclusively on convolving categorical columns in tabular data. In this section, we will construct the convolution layer of TabTransformer — :obj:`TabTransformerConv`.
Different to GraphConv, :obj:`TabTransformerConv` is a class that inherits from torch.nn.Module, and its two core methods are :obj:`__init__()` and :obj:`forward()`.

.. code-block:: python

    class TabTransformerConv(torch.nn.Module):
        def __init__(
            self,
            conv_dim,
            num_heads,
            dropout,
            activation,
            use_pre_encoder,
            metadata,
        ):
            super().__init__()
            ...

        def forward(
            self,
            x,
        ):
            ...


The :obj:`__init__()` method is responsible for initializing the parameters of the :obj:`TabTransformerConv` layer.
It requires a :obj:`dim` parameter to specify the input and output dimensions, as well as other relevant Transformer parameters, such as the number of attention heads (:obj:`num_heads`), dropout rate (:obj:`dropout`), and activation function type (:obj:`activation`).
Unlike Graph Neural Networks, the :obj:`TabTransformerConv` also requires a :obj:`metadata` parameter due to the strong heterogeneity of tabular data.
The :obj:`metadata` contains information about the table structure and is used to initialize the pre-encoder.

.. code-block:: python

    def __init__(
        self,
        dim,
        num_heads: int = 8,
        dropout: float = 0.3,
        activation: str = "relu",
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        encoder_norm = LayerNorm(dim)
        self.transformer = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=1,
            norm=encoder_norm,
        )

        self.pre_encoder = None
        if metadata:
            self.pre_encoder = TabTransformerPreEncoder(
                out_dim=dim,
                metadata=metadata,
            )

The :obj:`forward()` method defines the forward pass of the :obj:`TabTransformerConv` layer.
Its primary input is the tabular data :obj:`x`, provided as a dictionary.
If a :obj:`pre_encoder` is defined within this layer, the input data undergoes additional encoding before the convolution operation.
The :obj:`TabTransformerConv` performs convolution exclusively on the categorical features in the table.

.. code-block:: python

    def forward(self, x):
        if self.pre_encoder is not None:
            x = self.pre_encoder(x, return_dict=True)
        x[ColType.CATEGORICAL] = self.transformer(x[ColType.CATEGORICAL])
        return x

Similar to convolution in Graph Neural Networks, we can define custom methods as needed in :obj:`TabTransformerConv`.
For instance, we also define a :obj:`reset_parameters()` method to handle the initialization of the parameters, ensuring that the weight matrices and other learnable parameters are properly reset.

.. code-block:: python

    def reset_parameters(self) -> None:
        if self.pre_encoder is not None:
            self.pre_encoder.reset_parameters()

