Understanding Convolution
===============

What is a Convolution?
----------------
In machine learning, convolution generally involves combining an input signal with a filter to produce an output signal. Specifically, for image signals, convolution refers to aggregating nearby pixels around the central pixel. For graph signals, it involves aggregating information from connected nodes around the central node. For table signals, it refers to aggregating entries from different columns within each row. Therefore, designing an effective convolution operation is a key challenge in deep learning methods.


Construct a GCN Convolution Layer
----------------
Graph Convolutional Networks (GCNs) are a classic type of Graph Neural Network, where the convolution operation is applied to node features based on the input adjacency matrix. The convolution formula is defined as :math:`\tilde A X W`, where :math:`\tilde A` is the normalized adjacency matrix, :math:`X` represents the node features, and :math:`W` is the parameter matrix.

Before we dive into the details of the :obj:`GCNConv` class, let's first understand the structure of the :obj:`MessagePassing` class.
:obj:`MessagePassing`` is the base class for all graph convolution layers implemented in `rllm.nn.conv.graph_conv` including :obj:`GCNConv`.
It consists of three main steps: message computation ( :math:`\text{Message}` ), aggregation ( :math:`\text{Aggregate}` ), and update ( :math:`\text{Update}` ) as shown below:

.. math::
    \mathbf{x}_i^{(k+1)} = \text{Update}^{(k)}
    \left( \mathbf{x}_i^{(k)},
    \text{Aggregate}^{(k)} \left( \left\{ \text{Message}^{(k)} \left(
    \mathbf{x}_i^{(k)}, \mathbf{x}_j^{(k)}, \mathbf{e}_{j,i}^{(k)}
    \right) \right\}_{j \in \mathcal{N}(i)} \right) \right)

As above, the formula of :obj:`GCNConv` with message passing is as follows:

.. math::
    \mathbf{x}_i^{(k+1)} = \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{\deg(i) \deg(j)}} \mathbf{x}_j^{(k)}

Where the :math:`\sum` operation is the aggregation step,
and the :math:`\frac{1}{\sqrt{\deg(i) \deg(j)}}` term is the normalization factor.
The :math:`\deg(i)` and :math:`\deg(j)` terms represent the degrees of nodes :math:`i` and :math:`j`, respectively.
The message computation function simply retrieves the current layer neighbor node of :math:`\mathbf{x}_i^{(k)}` and returns the node feature of node :math:`\mathbf{x}_j^{(k)}`.
And the update step assigns the aggregated message to the next layer node representation :math:`\mathbf{x}_i^{(k+1)}`.

Now let's take a look at the implementation of the :obj:`GCNConv` class, which inherits from the :obj:`MessagePassing` class and consists of two main methods: :obj:`__init__()` and :obj:`forward()`.

The :obj:`__init__()` method is responsible for initializing the parameters of the :obj:`GCNConv` layer. This method takes two main parameters: :obj:`in_dim` (the input dimension) and :obj:`out_dim` (the output dimension).
These parameters are used to initialize the weight matrix :math:`W`. Additionally, a bias parameter :obj:`bias` can be included, which determines whether or not to use bias in the convolution operation.
Notably, the :obj:`GCNConv` layer uses the 'gcn' aggregation method to initialize :obj:`MessagePassing` (which can be changed to other aggregators, such as 'mean' etc.).

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

The :obj:`forward()` method defines the forward pass of the :obj:`GCNConv` layer. Its parameters include the node feature :obj:`inputs` (:math:`X` in the formula) and the adjacency matrix or edge list :obj:`edge_index` (:math:`\tilde{A}` in the formula) .
First, the input node features are passed through a linear layer, :obj:`self.linear`, to obtain the output features :obj:`x`.
Then, the :obj:`propagate()` method is called to perform the three message passing steps: message computation, aggregation, and update.
Finally, the bias term is added to the output features if the :obj:`bias` parameter is not None.

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

To construct another type of convolution layer, you can follow a similar process: 
1. Inherit from the :obj:`MessagePassing` class.
2. Define the :obj:`__init__` and :obj:`forward` methods.
3. Override the implementation of the :obj:`message`, :obj:`aggregate`, and :obj:`update` methods as needed.

In addition to the :obj:`__init__()` and :obj:`forward()` methods, you can define custom methods as needed. For example, the :obj:`GCNConv` class can include a :obj:`reset_parameters()` method, which reinitializes the parameters (i.e., the weight matrix :math:`W`) to their original values.

.. code-block:: python

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

Construct a TabTransformer Convolution Layer
----------------
TabTransformer is a classic Tabular/Table Neural Network that relies on the attention mechanism from Transformers to perform column-wise convolution. It focuses exclusively on convolving categorical features in tabular data. In this section, we will construct the convolution layer of TabTransformer — :obj:`TabTransformerConv`. Similar to GraphConv, :obj:`TabTransformerConv` is a class that inherits from torch.nn.Module, and its two core methods are :obj:`__init__()` and :obj:`forward()`.

The :obj:`__init__()` method is responsible for initializing the parameters of the :obj:`TabTransformerConv` layer. This method requires a dim parameter to specify the input and output dimensions. Additionally, it requires other relevant parameters for the Transformer, such as the number of attention heads (:obj:`num_heads`), dropout rate (:obj:`dropout`), and activation function type (:obj:`activation`). Due to the strong heterogeneity of tabular data, unlike Graph Neural Networks, the :obj:`TabTransformerConv` also requires a :obj:`metadata` parameter. The :obj:`metadata` contains information about the table structure and is used to initialize the pre-encoder.

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
The :obj:`forward()` method defines the forward pass of the :obj:`TabTransformerConv` layer. Its primary input is the tabular data x, passed as a dictionary. If a :obj:`pre_encoder` is defined within this layer, the data will undergo further encoding before the convolution operation. :obj:`TabTransformerConv` performs convolution only on the categorical features in the table.

.. code-block:: python

    def forward(self, x):
        if self.pre_encoder is not None:
            x = self.pre_encoder(x, return_dict=True)
        x[ColType.CATEGORICAL] = self.transformer(x[ColType.CATEGORICAL])
        return x

Similar to convolution in Graph Neural Networks, you can define custom methods as needed in :obj:`TabTransformerConv`. For instance, we also define a :obj:`reset_parameters()` method to handle the initialization of the parameters, ensuring that the weight matrices and other learnable parameters are properly reset.

.. code-block:: python

    def reset_parameters(self) -> None:
        if self.pre_encoder is not None:
            self.pre_encoder.reset_parameters()

