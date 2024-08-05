import pandas as pd

from langchain_community.llms import LlamaCpp
from rllm.llm import LangChainLLM
from rllm.llm.predictor import Predictor
from rllm.llm.parser.base import BaseOutputParser

model_path = "/path/to/llm"
data = {
    "EmployeeID": [123, 456],
    "FirstName": ["John", "Jane"],
    "LastName": ["Doe", "Smith"],
    "BirthDate": ["1980-01-01", "1990-02-02"],
    "Salary": [70000, 80000]
}
df = pd.DataFrame(data)
class my_parser(BaseOutputParser):
    def parse(self, output: str):
        start = output.find("ANSWER:") + len("ANSWER:")
        first_word_after_answer = output[start:].strip().split()[0]
        return first_word_after_answer
output_parser = my_parser()

llm = LangChainLLM(LlamaCpp(model_path=model_path, n_gpu_layers=33),
                   output_parser=output_parser)
predictor = Predictor(llm=llm, type='classification')
output = predictor(df, scenario='career classification', labels='doctor, engineer')
print(output)



