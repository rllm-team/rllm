import sys
from langchain_openai import ChatOpenAI

sys.path.append("./")
sys.path.append("../")
from rllm.llm import FeatLLM

if __name__ == "__main__":
    """
    Example usage of the FeatLLM class with a specified LLM and file paths.
    """
    API_KEY = "<Your API KEY>"
    API_URL = "<Your API URL>"
    # Example usage
    llm = ChatOpenAI(
        model_name="Your Model Name", openai_api_base=API_URL, openai_api_key=API_KEY
    )
    featllm = FeatLLM(
        file_path="Your File Path",
        metadata_path="Your Metadata Path",
        task_info_path="Your Task Info Path",
        llm=llm,
        # For example:
        # file_path="./adult.csv",
        # metadata_path="./adult-metadata.json",
        # task_info_path="./adult-task.txt",
        query_num=1,
    )
    featllm.invoke()
