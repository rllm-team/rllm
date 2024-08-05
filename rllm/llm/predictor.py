from typing import Any, List, Literal, Optional

import pandas as pd
from tqdm import tqdm

from rllm.llm.prompt.default_prompt import (
    DEFAULT_SCENARIO_CLASSIFICATION_TMPL,
    DEFAULT_SCENARIO_REGRESSION_TMPL
)
from rllm.llm.prompt.utils import (
    generate_sample_description,
    get_template_vars
)


from rllm.llm.llm_module.general_llm import LLM
from rllm.llm.prompt.base import BasePromptTemplate
from rllm.llm.prompt.base import PromptTemplate


class Predictor:
    r"""Predictor for relational data. Data should be
    organized into a  :class:`pandas.dataframe` format,
    with any prediction labels removed if present.

    Args:
        prompt (Optional[:class:`rllm.llm.prompt.base.BasePromptTemplate`]):
            The prompt to instruct llm make prediction.
        llm (:class:`rllm.llm.llm_module.general_llm.LLM`):
            The llm used for prediction, it is recommended
            to be initialized with LangChain.
        type (Optional[Literal['classification', 'regression']] ):
            Task type.

    .. code-block:: python

        import pandas as pd
        from langchain_openai import OpenAI
        from rllm.llm import LangChainLLM, Predictor

        # labels in dataframe should be removed.
        data = pd.read_csv('data.csv')
        scenario = 'Your_task_description'
        labels = 'Your_task_labels'
        llm = LangChainLLM(OpenAI(openai_api_key="YOUR_API_KEY"))

        predictor = Predictor(llm=llm, type='classification')
        outputs = predictor(data.head(10), scenario=scenario, labels=labels)
    """
    def __init__(
        self,
        prompt: Optional[BasePromptTemplate] = None,
        llm: LLM = None,
        type: Optional[Literal['classification', 'regression']] = None,
    ) -> None:
        # NOTE: Only support `PromptTemplate` so far
        self._llm = llm

        if prompt is None:
            assert type in ['classification', 'regression'], \
                "type must be 'classification' or 'regresssion'!"
            function_mapping = {
                'sample_description': generate_sample_description
            }
            if type == 'classification':
                self.prompt = PromptTemplate(
                    DEFAULT_SCENARIO_CLASSIFICATION_TMPL,
                    function_mappings=function_mapping
                )
            else:
                self.prompt = PromptTemplate(
                    DEFAULT_SCENARIO_REGRESSION_TMPL,
                    function_mappings=function_mapping
                )
        else:
            self.prompt = prompt

    def invoke(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> List[str]:
        # Check if all variables in the prompt are provided.
        input_variables = {
            **kwargs,
            **self.prompt.function_mappings
        }.keys()
        required_variables = get_template_vars(self.prompt.template)
        for var in required_variables:
            assert var in input_variables, \
                f"Variable '{var}' not found in input variables."

        # Make prediction, remeber `row` is a default argument.
        outputs = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            outputs.append(self._llm.predict(self.prompt, row=row, **kwargs))

        return outputs

    def __call__(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> Any:
        return self.invoke(df, **kwargs)
