# rLLM

**rLLM** (relationLLM) focuses on LLM-powered relational data learning, prioritizing: Accuracy, Efficiency, and Economy.

- Accuracy: quality of being true, correct, or exact.
- Efficiency: running time, measured in seconds.
- Economy: money cost, measured in dollars.

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

- We recommmend 4-bit quantized Gemma 2b model, which can be Downloaded from [HuggingFace](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/blob/main/gemma-2b-it-q4_k_m.gguf).
- In practice, the above Gemma 2b model is too weak to generate accurate responses. We use Mistral-7B model from [ollama](https://github.com/ollama/ollama).

## LM Model

- We recommend a light BERT-like model all-MiniLM-L6-v2 to make sentence embedding, which can be obtained directly from [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- The embedding function during constructing the VectorStore database include GPT4AllEmbedding and HuggingFaceEmbedding.

## Retrieval Augmented Generation (RAG)
- RAG is utilized to refine the generation process for more accurate prediction and less hallucinated results.
- The implementation code referes the paper [Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflection](http://arxiv.org/abs/2310.11511)
- Self-RAG is a strategy for RAG that incorporates self-reflection(grading) on retrieved documents and generations. In the paper, a few decisions are made:
  - Should I retrieve documents
    - Input: `x (question)`,  `y (generation)`
    - Decides when to retrieve `D` chunks with `R`
    - Output: `{yes, no, continue}`
  - Are the retrieved passages `D` relevant to the question `x`
    - Input: `x (question)`, `d(chunk)` for `d` in `D`
    - `d` provides useful information to solve `x`
    - output: `{relevant, irrelevant}`
  - Are the LLM generation from each chunk in D is relevant to the chunk (hallucinations, etc) -
    - Input: `x (question)`, `d (chunk)`, `y (generation)` for `d` in `D`
    - All of the verification-worthy statements in y (generation) are supported by d
    - Output: {fully supported, partially supported, no support
  - The LLM generation from each chunk in D is a useful response to x (question) -
    - Input: `x (question)`, `y (generation)` for `d` in `D`
    - `y (generation)` is a useful response to `x (question)`.
    - Output: `{yes, no}`

