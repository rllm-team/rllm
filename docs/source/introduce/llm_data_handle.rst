LLM Data Handle
================

LLM Handler
----------------------
We have implemented a simple LLM mechanism composed of three parts: Prompt, LLM, and Output Parser. We recommend using Langchain `[Introduction | 🦜️🔗 LangChain] <https://python.langchain.com/v0.1/docs/get_started/introduction/>`__  to initialize the LLM to provide a unified interface. Here is a simple example:

.. code-block:: python

    from langchain_openai import OpenAI
    from rllm.llm import PromptTemplate, LangChainLLM, BaseOutputParser

    template = "Please write five random names related to {topic}."
    prompt = PromptTemplate(template=template)

    class my_parser(BaseOutputParser):
        def parse(self, output: str):
            return output
    output_parser = my_parser()

    llm = LangChainLLM(OpenAI(openai_api_key="YOUR_API_KEY"), output_parser=output_parser)


    output = llm.predict(prompt, topic='dogs')
    print(output)

We have also implemented two utility classes: Predictor(跳转) and Enhancer(跳转). The former can annotate data with pseudo-labels, while the latter can generate explanations for the data to obtain information, or embed the additional information. The data needs to be organized in the form of a pandas.DataFrame(跳转).

.. code-block:: python

    import pandas as pd

    from langchain_openai import OpenAI
    from rllm.llm import LangChainLLM, BaseOutputParser, Predictor

    # Generate some data
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

    llm = LangChainLLM(OpenAI(openai_api_key="YOUR_API_KEY"),
                    output_parser=output_parser)
    predictor = Predictor(llm=llm, type='classification')

    # Two parameters should be provided: `scenario` and `labels`
    output = predictor(df, scenario='career classification', labels='doctor, engineer')
    print(output)

.. code-block:: python

    import pandas as pd

    from langchain_openai import OpenAI
    from rllm.llm import LangChainLLM, BaseOutputParser, Predictor

    data = {
        "title": ["The Shawshank Redemption", "Farewell My Concubine"],
        "year": ["1994", "1993"],
        "director": ["Frank Darabont", "Kaige Chen"]
    }
    df = pd.DataFrame(data)

    llm = LangChainLLM(OpenAI(openai_api_key="YOUR_API_KEY"),
                    output_parser=output_parser)
    enhancer = Enhancer(llm=llm, type='explanation')

    # Parameter `scenario` should be provided! 
    # If you want to get embedding, you should provide a list-like
    # parameter `cols` to identify which columns should be embedded.
    output = enhancer(df, scenario='movie explanation')
    print(output)