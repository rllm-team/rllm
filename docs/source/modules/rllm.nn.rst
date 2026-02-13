rllm.nn
=======

Conv
----------------------

Graph Conv
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: rllm.nn.conv.graph_conv

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   MessagePassing
   GCNConv
   LGCConv
   GATConv
   HANConv
   HGTConv
   SAGEConv

Table Conv
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: rllm.nn.conv.table_conv

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   ExcelFormerConv
   FTTransformerConv
   SAINTConv
   TabTransformerConv
   TromptConv
   TransTabConv
   ResNetConv

Pre-Encoder
----------------

.. currentmodule:: rllm.nn.pre_encoder

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   FTTransformerPreEncoder
   TabTransformerPreEncoder
   TransTabPreEncoder
   ResNetPreEncoder
   HeteroTemporalEncoder


Models
------

.. currentmodule:: rllm.nn.models

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   RECT_L
   TabNet
   BRIDGE
   TransTab
   TableResNet
   HeteroSAGE
   RDL
   RelGNN
   RelGNNModel


Loss
------

.. currentmodule:: rllm.nn.loss

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   ContrastiveLoss
   SelfSupervisedVPCL
   SupervisedVPCL