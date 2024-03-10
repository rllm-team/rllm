# rLLM

**rLLM** (relation LLM) focuses on LLM-based relational data mining, prioritizing: Accuracy, Efficiency, and Economy.

- Accuracy: MAE for regression; Micro-F1 and Macro-F1 for classification.
- Efficiency: Runtime, measured in seconds.
- Economy: Money, measured in dollars.

## Dependencies

- pytorch	2.1.2
- scikit-learn	1.4.0
- llama_cpp_python	0.2.52
- langchain	0.1.8
- langchain-community	0.0.21
- langchain-experimental	0.0.52
- tiktoken	0.6.0
- sentence-transformers	2.3.1
- numpy	1.26.4
- pandas	2.1.4

## LLM models

- We recommmend 4-bit quantized Gemma 2b model, which can be Downloaded from the SJTU cloud storage or [HuggingFace](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/blob/main/gemma-2b-it-q4_k_m.gguf).

## LM Model

- We recommend a light BERT-like model  all-MiniLM-L6-v2 to make sentence embedding, which can be obtained from the SJTU cloud storage, or directly from [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
