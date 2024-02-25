# rLLM

**rLLM** (relation LLM) focuses on LLM-based relational data mining, prioritizing: Accuracy, Efficiency, and Economy.

- Accuracy: MAE for regression; Micro-F1 and Macro-F1 for classification.
- Efficiency: Runtime, measured in seconds.
- Economy: Money, measured in dollars.

# Environment Setup

**Due to variations in the configurations of everyone's computer, achieving uniform setup is not feasible. Therefore, the following instructions address potential installation issues:**

It is recommended to use a Linux system for experimentation, which also facilitates submission.

For Windows systems, installing WSL is advised.

## PyTorch Installation

- PCs with Nvidia GPUs can use the `nvidia-smi` command to check their CUDA support version.
- PCs without dedicated Nvidia GPUs should install the CPU version.
- [PyTorch official website](https://pytorch.org/)

## llama-cpp-python and langchain

- Default installation method: **CPU only** (Windows/Linux/MacOS)

```bash
pip install llama-cpp-python
```

- If you want to **use GPU**, you need to first install CUDA and then install llama-cpp-python:

This allows specifying the **`n_gpu_layers`** parameter when instantiating the llama object, which determines how many layers of parameters are placed on the GPU to accelerate runtime.

```bash
# Instructions for installing GPU-enabled llama-cpp-python on Linux
# First, install the CUDA Toolkit. Tutorial: https://blog.csdn.net/qq_32033383/article/details/135015041. CUDNN installation is not necessary.

# Then use the following command
export LLAMA_CUBLAS=1
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

For detailed instructions, refer to [abetlen/llama-cpp-python: Python bindings for llama.cpp (github.com)](https://github.com/abetlen/llama-cpp-python)

- We have used `langchain`, `langchain-experimental` and `langchain-community` packages. For detailed instructions, refer to [Introduction | 🦜️🔗 Langchain](https://python.langchain.com/docs/get_started/introduction)

## Download 4-bit quantized llama models

- Download the 4-bit quantized llama models directly from the SJTU cloud storage. Currently, llama-2-7b-chat.Q4_K_M.gguf and gemma-2b-it-q4_k_m.gguf are provided.

- Download the llama-2-7b-chat.Q4_K_M.gguf file directly from [llama-2-7b-chat.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf). 
- Download gemma-2b-it-q4_k_m.gguf from [gemma-2b-it-q4_k_m.gguf](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/blob/main/gemma-2b-it-q4_k_m.gguf)

## Choosing Embedding Models

- If you need to use the BERT model for sentence embedding, it is recommended to use [sentence-transformers/all-MiniLM-L6-v2 · Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Downloads can be obtained from the SJTU cloud storage, or directly from Hugging Face.
- Use Sentence-Transformers or HuggingFace Transformers library to invoke the model.
- You can also use LLM to make sentence embedding.
