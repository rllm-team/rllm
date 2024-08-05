from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from rllm.llm.prompt.default_prompt import DEFAULT_SCENARIO_EXPLANATION_TMPL
from rllm.llm.prompt.utils import (
    generate_sample_description,
    get_template_vars
)

from rllm.llm.llm_module.general_llm import LLM
from rllm.llm.prompt.base import BasePromptTemplate


class Enhancer:
    r"""Enhancer for relational data. Data should be
    organized into a  :class:`pandas.dataframe` format.
    If attribute `type` is 'explanation|embedding', enhancer will explain
    them firstly and embedding them into vectors.

    Args:
        prompt (Optional[:class:`rllm.llm.prompt.base.BasePromptTemplate`]):
            The prompt to instruct llm make enhancement.
        llm (:class:`rllm.llm.llm_module.general_llm.LLM`):
            The llm used for explanation, it is recommended
            to be initialized with LangChain. Only useful in explanation step.
        llm_embed (:class:`rllm.llm.llm_module.general_llm.LLM`):
            The llm used for embedding, it is recommended
            to be initialized with LangChain. Only useful in embedding step.
        type (Optional[
                Literal['explanation|embedding', 'explanation', 'embedding']
            ]):
            Task type, default type is 'explanation|embedding'.

    Explanation|Embedding:
    .. code-block:: python
        import pandas as pd
        from langchain_openai import OpenAI, OpenAIEmbeddings
        from rllm.llm import LangChainLLM, Enhancer

        data = pd.read_csv('data.csv')
        scenario = 'Your_task_description'
        llm = LangChainLLM(OpenAI(openai_api_key="YOUR_API_KEY"))
        llm_embed = LangChainLLM(
            OpenAIEmbeddings(openai_api_key="YOUR_API_KEY")
        )

        enhancer = Enhancer(
            llm=llm, llm_embed=llm_embed, type='explanation|embedding'
        )
        outputs = enhancer(data.head(10), scenario=scenario)

    Explanation:
    .. code-block:: python
        import pandas as pd
        from langchain_openai import OpenAI
        from rllm.llm import LangChainLLM, Enhancer

        data = pd.read_csv('data.csv')
        scenario = 'Your_task_description'
        llm = LangChainLLM(OpenAI(openai_api_key="YOUR_API_KEY"))

        enhancer = Enhancer(llm=llm, type='explanation')
        outputs = enhancer(data.head(10), scenario=scenario)

    Embedding:
    .. code-block:: python
        import pandas as pd
        from langchain.embeddings import OpenAIEmbeddings
        from rllm.llm import LangChainLLM, Enhancer

        data = pd.read_csv('data.csv')
        scenario = 'Your_task_description'
        llm = LangChainLLM(OpenAIEmbeddings(openai_api_key="YOUR_API_KEY"))

        enhancer = Enhancer(llm_embed=llm, type='embedding')
        # Embedding columns 'text' and 'explanation'
        outputs = enhancer(data, cols=['text', 'explanation'])
    """
    def __init__(
        self,
        prompt: Optional['BasePromptTemplate'] = None,
        llm: LLM = None,
        llm_embed: LLM = None,
        type: Optional[
            Literal['explanation|embedding', 'explanation', 'embedding']
        ] = 'explanation|embedding',
    ) -> None:
        # NOTE: Only support `PromptTemplate` so far!
        # NOTE: Only support `explanation` so far!
        if 'explanation' in type:
            self._llm = llm
        if 'embedding' in type:
            self._llm_embed = llm_embed

        assert type in ['explanation|embedding', 'explanation', 'embedding'], \
            "type error!"
        self.type = type

        if 'explanation' in self.type:
            from rllm.llm.prompt.base import PromptTemplate
            if prompt is None:
                function_mapping = {
                    'sample_description': generate_sample_description
                }
                self.prompt = PromptTemplate(
                    DEFAULT_SCENARIO_EXPLANATION_TMPL,
                    function_mappings=function_mapping
                )
            else:
                self.prompt = prompt

    def invoke(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> List[str]:

        if 'explanation' in self.type:
            # Check if all variables in the prompt are provided.
            input_variables = {
                **kwargs,
                **self.prompt.function_mappings
            }.keys()
            required_variables = get_template_vars(self.prompt.template)
            for var in required_variables:
                assert var in input_variables, \
                    f"Variable '{var}' not found in input variables."

            # Make explanation, remeber `row` is a default argument.
            outputs = []
            for index, row in tqdm(df.iterrows(), total=len(df)):
                outputs.append(
                    self._llm.predict(self.prompt, row=row, **kwargs)
                )

        if 'embedding' in self.type:
            if 'explanation' in self.type:
                inputs = [outputs]
            else:
                # default target column is 'text'.
                cols = kwargs['cols'] if 'cols' in kwargs else ['text']
                inputs = [
                    col.values.tolist() for col_name, col in df[cols].items()
                ]

            outputs = []
            for input in inputs:
                embed = self._llm_embed.embedding(input)
                outputs.append(np.array(embed))
            outputs = outputs[0] if len(outputs) == 1 else outputs

        return outputs

    def __call__(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> Any:
        return self.invoke(df, **kwargs)
