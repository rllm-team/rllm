from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain.llms import LlamaCpp
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

model_path = "path/to/llm"

template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)



llm = LlamaCpp(
    model_path=model_path,
    streaming=False,
    n_gpu_layers=30
)
model = Llama2Chat(llm=llm)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)

print(
    chain.run(
        text="What can I see in Vienna? Propose a few locations. Names only, no details."
    )
)
print(chain.run(text="Tell me more about #2."))