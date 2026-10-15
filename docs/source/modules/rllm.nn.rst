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
   GTransformerConv
   RelGNNConv

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

Encoder
----------------

.. currentmodule:: rllm.nn.encoder

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   ColATE
   TablePreEncoder
   FTTransformerPreEncoder
   TabTransformerPreEncoder
   TransTabPreEncoder
   ResNetPreEncoder
   TromptPreEncoder
   HeteroTemporalEncoder
   MetaRTLEncoder


Models
------

.. currentmodule:: rllm.nn.models

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   RECT_L
   BRIDGE
   TableEncoder
   GraphEncoder
   LinearAttentionLayer
   IntraTableInteraction
   InterTableInteraction
   InRTL
   TransTab
   TransTabClassifier
   TransTabForCL
   TableResNet
   HeteroSAGE
   RDL
   RelGNN
   RelGNNModel
   MetaPathFusion
   MetaPathProp


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
