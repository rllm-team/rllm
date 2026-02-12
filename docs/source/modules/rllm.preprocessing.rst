rllm.preprocessing
==========

DataFrame to Tensor
------------------
.. currentmodule:: rllm.preprocessing
.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   df_to_tensor


Fill Missing Values
--------------
.. currentmodule:: rllm.preprocessing
.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fillna_numerical
   fillna_categorical
   fillna_binary
   fillna_by_coltype

Type Convert
------------------
.. currentmodule:: rllm.preprocessing
.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   encode_categorical
   convert_binary

Text Tokenize
--------------
.. currentmodule:: rllm.preprocessing
.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   TokenizerConfig
   process_tokenized_column
   tokenize_strings
   standardize_tokenizer_output
   tokenize_merged_cols
   save_column_name_tokens
   TransTabDataExtractor

Word Embedding
------------------
.. currentmodule:: rllm.preprocessing
.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   embed_text_column

Timestamp
------------------
.. currentmodule:: rllm.preprocessing
.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   TimestampPreprocessor
