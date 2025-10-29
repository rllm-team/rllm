import os
import time
from typing import Union, List, Any
import json
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class LC:
    from langchain_core.language_models import BaseLLM
    from langchain.chat_models.base import BaseChatModel
    from langchain_core.messages import (
        HumanMessage,
        SystemMessage,
    )


from .retriever import SingleTableRetriever

system_prompt = (
    "You are a helpful data analyst. I'll give you a tabular dataset's task description,"
    "features, label classes, and some labeled instances in json format,"
    "from which you will make classification prediction for new instance."
    "No analyzing, directly give the prediction answer class, "
    "there can only be one category of prediction."
)
user_prompt = (
    "Task description: {task_description}\n"
    "Features: {features}\n"
    "Target label classes: {classes}\n"
    "Labeled instances: {serialized_labeled_instance}\n\n"
    "Now use the provided metadata and instances to infer by analogy about the label of this new instance: "
    "{serialized_unlabeled_instance}"
)


class LLMWithRetriever:
    """
    A class that integrates a language model with a retriever for classification tasks.

    Attributes:
        df (pd.DataFrame): The dataframe loaded from the CSV file.
        dir (str): The directory of the input file.
        data_name (str): The name of the dataset (without extension).
        metadata (dict): Metadata loaded from the JSON file.
        task_info (str): Task description loaded from the TXT file.
        label_info (str): Label information loaded from the TXT file.
        llm (Union[LC.BaseChatModel, LC.BaseLLM]): The language model instance.
        shots (int): Number of examples to retrieve for few-shot learning.
        seed (int): Random seed for reproducibility.
        train_file_path (str): Path to the training data CSV file.
        test_file_path (str): Path to the testing data CSV file.
        label_file_path (str): Path to the label data CSV file.
        retriever (SingleTableRetriever): Instance of the retriever for fetching similar examples.
    """

    def __init__(
        self,
        file_path: str,
        metadata_path: str,
        task_info_path: str,
        label_info_path: str,
        llm: Union[LC.BaseChatModel, LC.BaseLLM] = None,
        *,
        shots: int = 4,
        test_size: Union[float, int] = 0.2,
        target_column: str = None,
        seed: int = 0,
    ) -> None:
        """
        Initializes the LLMWithRetriever class.

        Args:
            file_path (str): Path to the CSV file containing the dataset.
            metadata_path (str): Path to the JSON file containing metadata.
            task_info_path (str): Path to the TXT file containing task description.
            label_info_path (str): Path to the TXT file containing label information.
            llm (Union[LC.BaseChatModel, LC.BaseLLM], optional): Language model instance. Defaults to None.
            shots (int, optional): Number of examples for few-shot learning. Defaults to 4.
            test_size (Union[float, int], optional): Proportion or count of test data. Defaults to 0.2.
            target_column (str, optional): Name of the target column. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
        """
        assert file_path.endswith(".csv"), "file_path must be a CSV file."
        assert metadata_path.endswith(".json"), "metadata_path must be a JSON file."
        assert task_info_path.endswith(".txt"), "task_info_path must be a TXT file."
        assert label_info_path.endswith(".txt"), "label_info_path must be a TXT file."

        # Read the CSV file
        self.df = pd.read_csv(file_path)
        self.dir = os.path.dirname(os.path.abspath(file_path))
        self.data_name = os.path.splitext(os.path.basename(file_path))[0]
        # If target_column is specified, move it to the last column
        if target_column and target_column in self.df.columns:
            self.df = self.df[
                [col for col in self.df.columns if col != target_column]
                + [target_column]
            ]

        # Check test_size validity
        if isinstance(test_size, float):
            assert 0 < test_size < 1, "test_size as float must be in the range (0, 1)."
        elif isinstance(test_size, int):
            assert (
                0 < test_size < len(self.df)
            ), "test_size as int must be in the range [1, len(df))."

        # Read the JSON file
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Read the TXT file
        with open(task_info_path, "r") as f:
            self.task_info = f.read()

        with open(label_info_path, "r") as f:
            self.label_info = f.read()

        if not isinstance(llm, (LC.BaseChatModel, LC.BaseLLM)):
            raise TypeError(
                f"llm must be an instance of BaseChatModel or BaseLLM, "
                f"but got {type(llm).__name__}"
            )
        self.llm = llm
        self.shots = shots
        self.seed = seed

        self._set_seed(seed)
        self.train_file_path, self.test_file_path, self.label_file_path = (
            self._get_split(test_size=test_size)
        )
        self.retriever = SingleTableRetriever(self.train_file_path)

    def invoke(self):
        """
        Executes the inference process by retrieving neighbors for test instances
        and querying the language model for predictions.

        Saves the results to a JSON file in the same directory as the input file.
        """
        test_df = pd.read_csv(self.test_file_path)
        results = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            row_json = row.to_json()
            neighbors = [
                self._to_json(_.page_content)
                for _ in self.retriever(row_json, top_k=self.shots)
            ]
            neighbors_str = "\n".join(neighbors)
            text = user_prompt.format(
                task_description=self.task_info,
                features=self.metadata,
                classes=self.label_info,
                serialized_labeled_instance=neighbors_str,
                serialized_unlabeled_instance=row_json,
            )
            results.append(self._query_llm(text, temperature=0.5))

        output_file_path = os.path.join(self.dir, "results.json")
        print(f"Saving results to {output_file_path}")
        with open(output_file_path, "w") as f:
            json.dump(results, f, indent=4)

    def __call__(self, *args: Any, **kwargs: Any):
        """
        Allows the class instance to be called like a function, invoking the `invoke` method.
        """
        self.invoke()

    def _set_seed(self, seed: int):
        """
        Sets the random seed for reproducibility.

        Args:
            seed (int): The random seed value.
        """
        random.seed(seed)
        np.random.seed(seed)

    def _get_split(
        self,
        test_size: Union[int, float],
    ):
        """
        Splits the dataset into training and testing sets and saves them to CSV files.

        Args:
            test_size (Union[int, float]): Proportion or count of test data.

        Returns:
            tuple: Paths to the training, testing, and label CSV files.
        """
        train_file_path = os.path.join(self.dir, f"{self.data_name}_train.csv")
        test_file_path = os.path.join(self.dir, f"{self.data_name}_test.csv")
        label_file_path = os.path.join(self.dir, f"{self.data_name}_y.csv")
        X_train, X_test = train_test_split(
            self.df,
            test_size=test_size,
            random_state=self.seed,
            stratify=self.df.iloc[:, -1],
        )

        X_test.iloc[:, -1].to_csv(label_file_path, index=False)

        X_test = X_test.iloc[:, :-1]

        X_train.to_csv(train_file_path, index=False)
        X_test.to_csv(test_file_path, index=False)

        return train_file_path, test_file_path, label_file_path

    def _to_json(self, input_str: str):
        """
        Converts a string with key-value pairs into a JSON-formatted string.

        Args:
            input_str (str): Input string containing key-value pairs.

        Returns:
            str: JSON-formatted string.
        """
        result = {}

        lines = input_str.split("\n")

        for line in lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                result[key] = value

        return json.dumps(result)

    def _is_chat_model(self) -> bool:
        """
        Checks if the language model is a chat-based model.

        Returns:
            bool: True if the model is a chat-based model, False otherwise.
        """
        return isinstance(self.llm, LC.BaseChatModel)

    def _query_llm(
        self,
        text: str,
        max_tokens: int = 30,
        temperature: float = 0.0,
        max_try: int = 10,
    ) -> List[str]:
        """
        Queries the language model with the given text and parameters.

        Args:
            text (str): Input text for the language model.
            max_tokens (int, optional): Maximum number of tokens in the response. Defaults to 30.
            temperature (float, optional): Sampling temperature for randomness. Defaults to 0.0.
            max_try (int, optional): Maximum number of retry attempts in case of failure. Defaults to 10.

        Returns:
            List[str]: The response from the language model.
        """
        for _ in range(max_try):
            try:
                response = (
                    self.llm.invoke(
                        [LC.SystemMessage(system_prompt), LC.HumanMessage(text)],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if self._is_chat_model()
                    else self.llm.invoke(
                        system_prompt + "\n" + text,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                )
                result = response.content if self._is_chat_model() else response
                break
            except Exception as e:
                print(e)
                time.sleep(10)
        else:
            result = -1
        return result
