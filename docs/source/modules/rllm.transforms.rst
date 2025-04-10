rllm.transforms
====================


Graph Transforms
-----------------------

Basic Level
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: rllm.transforms.graph_transforms

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

    NodeTransform
    EdgeTransform
    NormalizeFeatures
    SVDFeatureReduction
    KNNGraph
    AddRemainingSelfLoops
    RemoveSelfLoops
    GCNNorm
    GDC

Model Level
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: rllm.transforms.graph_transforms

.. autosummary::
   :nosignatures:
   :toctree: ../generated
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
   :toctree: ../generated
   :template: autosummary/class.rst

    ColTransform
    ColNormalize
    OneHotTransform
    StackNumerical

Model Level
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: rllm.transforms.table_transforms

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

    TableTransform
    DefaultTableTransform
    TabTransformerTransform

Utils
-----------------------

.. currentmodule:: rllm.transforms.utils

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

    RemoveTrainingClasses
