from llama_cpp import Llama
model_path = "llama-2-7b-chat.Q4_K_M.gguf"
llm = Llama(model_path = model_path, n_gpu_layers=40)
respnse = \
    llm.create_chat_completion(
      messages = [
          {"role": "system", "content": "You are an assistant who perfectly makes story."},
          {
              "role": "user",
              "content": "Tell me a story about dinosaur."
          }
      ]
)
print(respnse)