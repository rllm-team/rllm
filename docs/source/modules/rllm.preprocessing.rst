rllm.preprocessing
==================

DataFrame to Tensor
-------------------
.. currentmodule:: rllm.preprocessing
.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/function.rst

   df_to_tensor

Text Tokenize
-------------
.. currentmodule:: rllm.preprocessing
.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   TokenizerConfig

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/function.rst

   process_tokenized_column
   tokenize_strings
   standardize_tokenizer_output
   tokenize_merged_cols
   save_column_name_tokens

Word Embedding
--------------
.. currentmodule:: rllm.preprocessing
.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   TextEmbedderConfig

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/function.rst

   embed_text_column

Timestamp
---------
.. currentmodule:: rllm.preprocessing
.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   TimestampPreprocessor

Fillna
------
.. currentmodule:: rllm.preprocessing
.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   FillNAConfig

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/function.rst

   fillna_by_coltype
