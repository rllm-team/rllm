from llama_cpp import Llama
model_path = "path/to/llm"
llm = Llama(model_path = model_path, embedding=True)
output = llm.create_embedding("Hello")
print(output['data'][0]['embedding'])
print(len(output['data'][0]['embedding']))