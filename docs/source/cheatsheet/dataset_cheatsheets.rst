Dataset Cheatsheet
==================

Graph Cheatsheet
----------------

Heterogeneous Graph Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
    :header-rows: 0
    :class: custom-table

    * - DBLP
      - DBLP is a heterogeneous graph containing four types of entities, as collected in the `MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding <https://arxiv.org/abs/2002.01680>`__ paper.
    * - IMDB
      - IMDB is a heterogeneous graph containing three types of entities, as collected in the `MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding <https://arxiv.org/abs/2002.01680>`__ paper.


Homogeneous Graph Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
    :header-rows: 0
    :class: custom-table
   
    * - PlanetoidDataset
      - The citation network datasets from the `Revisiting Semi-Supervised Learning with Graph Embeddings <https://arxiv.org/abs/1603.08861>`__ paper, which include :obj:`"Cora"`, :obj:`"CiteSeer"` and :obj:`"PubMed"`. Nodes represent documents and edges represent citation links.
    * - TAPEDataset
      - The citation network datasets, include `cora` and `pubmed`, collected from paper `Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning <https://arxiv.org/abs/1603.08861>`__ paper.
    * - TAGDataset
      - Three text-attributed-graph datasets, including `cora` from `Automating the Construction of Internet Portals <https://link.springer.com/content/pdf/10.1023/A:1009953814988.pdf>`__ paper, `pubmed` from `Collective Classification in Network Data <https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157>`__ paper and `citeseer` from `CiteSeer: an automatic citation indexing system <https://dl.acm.org/doi/10.1145/276675.276685>`__ paper. This dataset also contains cached LLM predictions and confidences provided by the paper `Label-free Node Classification on Graphs with Large Language Models (LLMS) <https://arxiv.org/abs/2310.04668>`__ .



Table Cheatsheet
----------------

Single Table Datasets
^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
    :header-rows: 0
    :class: custom-table

    * - Titanic
      - The Titanic dataset is a widely-used dataset for machine learning and statistical analysis, as featured in the `Titanic: Machine Learning from Disaster <https://www.kaggle.com/c/titanic>`__ competition on Kaggle.


Multi-Table Datasets
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 0
    :class: custom-table

    * - TML1M
      - Table-MovieLens1M (TML1M) is a relational table dataset enhanced from the classical MovieLens1M dataset, comprising three tables: users, movies and ratings. This dataset is derived from the `"rLLM:Relational Table Learning with LLMs" <https://arxiv.org/abs/2407.20157>`_ paper.
    * - TLF2K
      - Table-LastFm2K (TLF2K) is a relational table dataset enhanced from the classical LastFm2k dataset, containing three tables: artists, user_artists and user_friends. This dataset is derived from the `"rLLM:Relational Table Learning with LLMs" <https://arxiv.org/abs/2407.20157>`_ paper.
    * - TACM12K
      - Table-ACM12K (TACM12K) is a relational table dataset enhanced from the ACM heterogeneous graph dataset. It includes four tables: papers, authors, citations and writings. This dataset is derived from the `"rLLM:Relational Table Learning with LLMs" <https://arxiv.org/abs/2407.20157>`_ paper.


