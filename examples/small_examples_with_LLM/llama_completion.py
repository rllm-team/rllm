from llama_cpp import Llama
model_path = "path/to/llm"
llm = Llama(model_path=model_path, verbose=True, n_gpu_layers=30)
output = llm(
      "Q: What's the result of 1+1? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
)
print(output)