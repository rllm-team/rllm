rllm.transforms
====================


Graph Transforms
-----------------------

Basic Level
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: rllm.transforms.graph_transforms

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

    NETransform
    Compose
    AddRemainingSelfLoops
    RemoveSelfLoops
    KNNGraph
    GCNNorm
    GDC

Model Level
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: rllm.transforms.graph_transforms

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

    GraphTransform
    GCNTransform
    RECTTransform

Table Transforms
-----------------------

Basic Level
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: rllm.transforms.table_transforms

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

    ColTransform
    ColumnNormalize
    OneHotTransform
    StackNumerical

Model Level
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: rllm.transforms.table_transforms

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

    TableTransform
    DefaultTableTransform
    TabTransformerTransform

Utils
-----------------------

.. currentmodule:: rllm.transforms.utils

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

    BaseTransform
    NormalizeFeatures
    SVDFeatureReduction
    RemoveTrainingClasses






