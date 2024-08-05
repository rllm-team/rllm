import pandas as pd

from langchain_community.llms import LlamaCpp
from rllm.llm import LangChainLLM
from rllm.llm.enhancer import Enhancer
from rllm.llm.parser.base import BaseOutputParser

model_path = "/path/to/llm"
data = {
    "title": ["The Shawshank Redemption", "Farewell My Concubine"],
    "year": ["1994", "1993"],
    "director": ["Frank Darabont", "Kaige Chen"]
}
df = pd.DataFrame(data)

llm = LangChainLLM(LlamaCpp(model_path=model_path, n_gpu_layers=33),
                   output_parser=output_parser)
enhancer = Enhancer(llm=llm, type='explanation')
output = enhancer(df, scenario='movie explanation')
print(output)
