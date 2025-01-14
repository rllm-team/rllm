Understanding Convolution
===============

What is a Convolution?
----------------
In machine learning, convolution generally involves combining an input signal with a filter to produce an output signal. Specifically, for image signals, convolution refers to aggregating nearby pixels around the central pixel. For graph signals, it involves aggregating information from connected nodes around the central node. For table signals, it refers to aggregating entries from different columns within each row. Therefore, designing an effective convolution operation is a key challenge in deep learning methods.


Construct a GCN Convolution Layer
----------------
Graph Convolutional Networks (GCNs) are a classic type of Graph Neural Network, where the convolution operation is applied to node features based on the input adjacency matrix. The convolution formula is defined as :math:`\tilde A X W`, where :math:`\tilde A` is the normalized adjacency matrix, :math:`X` represents the node features, and :math:`W` is the parameter matrix. In this context, we will construct the graph convolution layer, :obj:`GCNConv`. :obj:`GCNConv` is a class that inherits from torch.nn.Module and consists of two primary methods: :obj:`__init__()` and :obj:`forward()`.

The :obj:`__init__()` method is responsible for initializing the parameters of the :obj:`GCNConv` layer. This method takes two main parameters: :obj:`in_dim` (the input dimension) and :obj:`out_dim` (the output dimension). These parameters are used to initialize the weight matrix :math:`W`. Additionally, a bias parameter :obj:`bias` can be included, which determines whether or not to use bias in the convolution operation.

.. code-block:: python

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.empty(in_dim, out_dim))

        if bias:
            self.bias = Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

The :obj:`forward()` method defines the forward pass of the :obj:`GCNConv` layer. Its parameters include the node :obj:`inputs` (:math:`X` in formula) and the adjacency matrix :obj:`adj` (:math:`\tilde A` in formula) . This method applies the graph convolution to the nodes based on the formula outlined earlier.

.. code-block:: python

    def forward(self, inputs: Tensor, adj: Tensor):
        support = torch.mm(inputs, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


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

