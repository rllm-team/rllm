from langchain_community.llms import LlamaCpp
from rllm.llm import LangChainLLM
from rllm.llm.prompt.base import PromptTemplate

model_path = "/path/to/llm"
template = "Please write five random names related to {topic}."
prompt = PromptTemplate(template=template)
llm = LangChainLLM(LlamaCpp(model_path=model_path, n_gpu_layers=33))
output = llm.predict(prompt, topic='dogs')
print(output)