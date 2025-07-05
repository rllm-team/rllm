import sys

from langchain_openai import ChatOpenAI

sys.path.append("./")
sys.path.append("../")
from rllm.llm import LLMWithRetriever

if __name__ == "__main__":
    """
    Example usage of the LLMWithRetriever class. Replace placeholders with actual values.
    """
    API_KEY = "<Your API KEY>"
    API_URL = "<Your API URL>"
    # Example usage
    llm = ChatOpenAI(
        model_name="Your Model Name", openai_api_base=API_URL, openai_api_key=API_KEY
    )
    Inference = LLMWithRetriever(
        file_path="Your File Path",
        metadata_path="Your Metadata Path",
        task_info_path="Your Task Info Path",
        llm=llm,
        # For example:
        # file_path="./adult.csv",
        # metadata_path="./adult-metadata.json",
        # task_info_path="./adult-task.txt",
    )
    Inference.invoke()
