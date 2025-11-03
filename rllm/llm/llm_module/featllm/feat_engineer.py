ASK_LLM_TMPL = (
    "You are an expert. Given the task description and the list of features and data examples, you are extracting conditions for each answer class to solve the task.\n\n"
    "Task: {task}\n\n"
    "Features:\n"
    "{features}\n\n"
    "Examples:\n"
    "{examples}\n\n"
    "Let's first understand the problem and solve the problem step by step.\n\n"
    "Step 1. Analyze the causal relationship or tendency between each feature and task description based on general knowledge and common sense within a short sentence. \n\n"
    "Step 2. Based on the above examples and Step 1's results, infer 10 different conditions per answer, following the format below. The condition should make sense, well match examples, and must match the format for [condition] according to value type.\n\n"
    "Format for Response:\n"
    "{format}\n\n\n"
    "Format for [Condition]:\n"
    "For the categorical variable only,\n"
    "- [Feature_name] is in [list of Categorical_values]\n"
    "For the numerical variable only,\n"
    "- [Feature_name] (> or >= or < or <=) [Numerical_value]\n"
    "- [Feature_name] is within range of [Numerical_range_start, Numerical_range_end]\n\n\n"
    "Please avoid using any formatting symbols like bold, italics, or other special characters (e.g., **, *, #). The output should be plain text without any embellishments or special formatting, as it may interfere with the readability and interpretation of the response.\n"
    "Answer: \n"
    "Step 1. The relationship between each feature and the task description: "
)
ASK_FOR_FUNCTION_TMPL = (
    "Provide me a python code for function, given description below.\n\n"
    "Function name: {name}\n\n"
    "Input: Dataframe df_input\n\n"
    "Input Features:\n"
    "{features}\n\n"
    "Output: Dataframe df_output. Create a new dataframe df_output. Each column in df_output refers whether the selected column in df_input follows the condition (1) or not (0). Be sure that the function code well matches with its feature type (i.e., numerical, categorical).\n\n"
    "Conditions: \n"
    "{conditions}\n\n\n"
    "Wrap only the function part with <start> and <end>, and do not add any comments, descriptions, and package importing lines in the code."
)


import os
import time
from typing import Union, List, Any, Tuple, Dict
import json
import random

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class LC:
    """
    A container class for importing and organizing various language model-related classes
    from the LangChain library. This class serves as a namespace for easier access to
    LangChain components.
    """

    from langchain_core.language_models import BaseLLM, BaseLanguageModel
    from langchain.chat_models.base import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
    )


class FeatLLMEngineer:
    """
    A feature engineering class that leverages a language model to generate feature extraction
    rules and functions based on input data, metadata, and task descriptions.

    Attributes:
        df (pd.DataFrame): The input dataset loaded from a CSV file.
        data_name (str): The name of the dataset (derived from the file name).
        metadata (dict): Metadata information loaded from a JSON file.
        task_info (str): Task description loaded from a TXT file.
        llm (Union[LC.BaseChatModel, LC.BaseLLM]): The language model used for querying.
        query_num (int): Number of queries to generate for the language model.
        shots (int): Number of examples to use for few-shot learning.
        test_size (Union[float, int]): Proportion or number of test samples.
        target_column (str): The target column in the dataset.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        file_path: str,
        metadata_path: str,
        task_info_path: str,
        llm: Union[LC.BaseChatModel, LC.BaseLLM] = None,
        *,
        query_num: int = 5,
        shots: int = 4,
        test_size: Union[float, int] = 0.2,
        target_column: str = None,
        seed: int = 0,
    ) -> None:
        """
        Initializes the FeatLLMEngineer class by loading the dataset, metadata, and task
        description, and setting up the language model and other configurations.

        Args:
            file_path (str): Path to the CSV file containing the dataset.
            metadata_path (str): Path to the JSON file containing metadata.
            task_info_path (str): Path to the TXT file containing task description.
            llm (Union[LC.BaseChatModel, LC.BaseLLM], optional): Language model instance.
            query_num (int, optional): Number of queries to generate. Defaults to 5.
            shots (int, optional): Number of examples for few-shot learning. Defaults to 4.
            test_size (Union[float, int], optional): Test set size. Defaults to 0.2.
            target_column (str, optional): Target column name. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
        """
        assert file_path.endswith(".csv"), "file_path must be a CSV file."
        assert metadata_path.endswith(".json"), "metadata_path must be a JSON file."
        assert task_info_path.endswith(".txt"), "task_info_path must be a TXT file."

        # Read the CSV file
        self.df = pd.read_csv(file_path)
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

        if not isinstance(llm, (LC.BaseChatModel, LC.BaseLLM)):
            raise TypeError(
                f"llm must be an instance of BaseChatModel or BaseLLM, "
                f"but got {type(llm).__name__}"
            )
        self.llm = llm
        self.query_num = query_num
        self.shots = shots
        self.seed = seed

        self._set_seed(seed)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.target_column,
            self.label_list,
            self.is_cat,
        ) = self._get_dataset(self.df, shots, seed, test_size)

    def _get_dataset(
        self,
        df: pd.DataFrame,
        shots: int,
        seed: int,
        test_size: Union[int, float],
    ):
        """
        Splits the dataset into training and testing sets, balances the training set,
        and identifies categorical features.

        Args:
            df (pd.DataFrame): The input dataset.
            shots (int): Number of examples for few-shot learning.
            seed (int): Random seed for reproducibility.
            test_size (Union[int, float]): Test set size.

        Returns:
            Tuple: Training and testing sets, target column, label list, and categorical indicators.
        """
        default_target_column = df.columns[-1]
        categorical_indicator = [
            True if (dt == np.dtype("O") or pd.api.types.is_string_dtype(dt)) else False
            for dt in df.dtypes.tolist()
        ][:-1]

        X = df.convert_dtypes()
        y = df[default_target_column].to_numpy()
        label_list = np.unique(y).tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X.drop(default_target_column, axis=1),
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        assert shots <= 128, "Shot must be less than 128 for efficiency!"
        # Resample to make each class balanced
        X_new_train = X_train.copy()
        X_new_train[default_target_column] = y_train
        sampled_list = []
        remainder = shots % len(np.unique(y_train))
        for _, grouped in X_new_train.groupby(default_target_column):
            sample_num = shots // len(np.unique(y_train))
            if remainder > 0:
                sample_num += 1
                remainder -= 1
            grouped = grouped.sample(sample_num, random_state=seed)
            sampled_list.append(grouped)
        X_balanced = pd.concat(sampled_list)
        X_train = X_balanced.drop([default_target_column], axis=1)
        y_train = X_balanced[default_target_column].to_numpy()
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            default_target_column,
            label_list,
            categorical_indicator,
        )

    def _query_llm(
        self,
        text_list: List[str],
        max_tokens: int = 30,
        temperature: float = 0.0,
        max_try: int = 10,
    ) -> List[str]:
        """
        Queries the language model with a list of prompts and returns the responses.

        Args:
            text_list (List[str]): List of prompts to query.
            max_tokens (int, optional): Maximum tokens for the response. Defaults to 30.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            max_try (int, optional): Maximum retry attempts for each query. Defaults to 10.

        Returns:
            List[str]: List of responses from the language model.
        """
        result_list = []
        for prompt in tqdm(text_list):
            for _ in range(max_try):
                try:
                    response = (
                        self.llm.invoke(
                            [LC.HumanMessage(prompt)],
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        if self._is_chat_model()
                        else self.llm.invoke(
                            prompt, max_tokens=max_tokens, temperature=temperature
                        )
                    )
                    result_list.append(
                        response.content if self._is_chat_model() else response
                    )
                    break
                except Exception as e:
                    print(e)
                    time.sleep(10)
            else:
                result_list.append(-1)
        return result_list

    def _generate_asking_prompt(
        self,
        df_all: pd.DataFrame,
        df_x: pd.DataFrame,
        df_y: pd.DataFrame,
        label_list: List[Any],
        default_target_column: str,
        is_cat: List[bool],
        query_num: int = 5,
    ) -> Tuple[List[str], str]:
        """
        Generates prompts for querying the language model to extract feature conditions.

        Args:
            df_all (pd.DataFrame): The entire dataset excluding the target column.
            df_x (pd.DataFrame): Training features.
            df_y (pd.DataFrame): Training labels.
            label_list (List[Any]): List of unique labels.
            default_target_column (str): Name of the target column.
            is_cat (List[bool]): List indicating whether each feature is categorical.
            query_num (int, optional): Number of queries to generate. Defaults to 5.

        Returns:
            Tuple[List[str], str]: List of prompts and feature descriptions.
        """
        task_desc = f"{self.task_info}\n"
        df_incontext = df_x.copy()
        df_incontext[default_target_column] = df_y

        format_list = [
            f'10 different conditions for class "{label}":\n- [Condition]\n...'
            for label in label_list
        ]
        format_desc = "\n\n".join(format_list)

        template_list = []
        current_query = 0
        while True:
            if current_query >= query_num:
                break

            # Feature bagging
            if len(df_incontext.columns) >= 20:
                total_column_list = []
                for i in range(len(df_incontext.columns) // 10):
                    column_list = df_incontext.columns.tolist()[:-1]
                    random.shuffle(column_list)
                    total_column_list.append(column_list[i * 10 : (i + 1) * 10])
            else:
                total_column_list = [df_incontext.columns.tolist()[:-1]]

            for selected_column in total_column_list:
                if current_query >= query_num:
                    break

                # Sample bagging
                threshold = 16
                if len(df_incontext) > threshold:
                    sample_num = int(
                        threshold / df_incontext[default_target_column].nunique()
                    )
                    df_incontext = df_incontext.groupby(
                        default_target_column, group_keys=False
                    ).apply(lambda x: x.sample(sample_num))
                feature_name_list = []
                sel_cat_idx = [
                    df_incontext.columns.tolist().index(col_name)
                    for col_name in selected_column
                ]
                is_cat_sel = np.array(is_cat)[sel_cat_idx]

                # TODO: Handle text column
                for cidx, cname in enumerate(selected_column):
                    if is_cat_sel[cidx] == True:
                        clist = df_all[cname].unique().tolist()
                        if len(clist) > 20:
                            clist_str = f"{clist[0]}, {clist[1]}, ..., {clist[-1]}"
                        else:
                            clist_str = ", ".join(clist)
                        desc = (
                            self.metadata[cname]
                            if cname in self.metadata.keys()
                            else ""
                        )
                        feature_name_list.append(
                            f"- {cname}: {desc} (categorical variable with categories [{clist_str}])"
                        )
                    else:
                        desc = (
                            self.metadata[cname]
                            if cname in self.metadata.keys()
                            else ""
                        )
                        feature_name_list.append(
                            f"- {cname}: {desc} (numerical variable)"
                        )

                feature_desc = "\n".join(feature_name_list)
                in_context_desc = ""
                df_current = df_incontext.copy()
                df_current = df_current.groupby(
                    default_target_column,
                    group_keys=False,
                )[df_current.columns.to_list()].apply(lambda x: x.sample(frac=1))

                for _, icl_row in df_current.iterrows():
                    answer = icl_row[default_target_column]
                    icl_row = icl_row.drop(labels=default_target_column)
                    icl_row = icl_row[selected_column]
                    in_context_desc += self._serialize(icl_row)
                    in_context_desc += f"\nAnswer: {answer}\n"

                fill_in_dict = {
                    "task": task_desc,
                    "examples": in_context_desc,
                    "features": feature_desc,  # TODO: Handle text column
                    "format": format_desc,
                }
                template = self._fill_in_templates(fill_in_dict, ASK_LLM_TMPL)
                template_list.append(template)
                current_query += 1

        return template_list, feature_desc

    def _generate_function_prompt(
        self,
        parsed_rule: Dict[str, List[str]],
        feature_desc: str,
    ) -> List[str]:
        """
        Generates prompts for querying the language model to create feature extraction functions.

        Args:
            parsed_rule (Dict[str, List[str]]): Parsed rules for each class.
            feature_desc (str): Description of the features.

        Returns:
            List[str]: List of prompts for function generation.
        """
        template_list = []
        for class_id, each_rule in parsed_rule.items():
            function_name = f"extracting_features_{class_id}"
            rule_str = "\n".join([f"- {k}" for k in each_rule])

            fill_in_dict = {
                "name": function_name,
                "conditions": rule_str,
                "features": feature_desc,
            }
            template = self._fill_in_templates(fill_in_dict, ASK_FOR_FUNCTION_TMPL)
            template_list.append(template)

        return template_list

    def _convert_to_binary_vectors(
        self,
        fct_strs_all: List[List[str]],
        fct_names: List[List[str]],
        label_list: List[str],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ):
        """
        Converts feature extraction functions into binary vectors for training and testing sets.

        Args:
            fct_strs_all (List[List[str]]): List of function strings.
            fct_names (List[List[str]]): List of function names.
            label_list (List[str]): List of unique labels.
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.

        Returns:
            Tuple: Executable function indices, training and testing binary vectors.
        """
        X_train_all_dict = {}
        X_test_all_dict = {}
        executable_list = (
            []
        )  # Save the parsed functions that are properly working for both train/test sets
        for i in range(
            len(fct_strs_all)
        ):  # len(fct_strs_all) == # of trials for ensemble
            X_train_dict, X_test_dict = {}, {}
            for label in label_list:
                X_train_dict[label] = {}
                X_test_dict[label] = {}

            # Match function names with each answer class
            fct_idx_dict = {}
            for idx, name in enumerate(fct_names[i]):
                for label in label_list:
                    label_name = "_".join(label.split(" "))
                    if label_name.lower() in name.lower():
                        fct_idx_dict[label] = idx

            # If the number of inferred rules are not the same as the number of answer classes, remove the current trial
            if len(fct_idx_dict) != len(label_list):
                continue

            try:
                for label in label_list:
                    fct_idx = fct_idx_dict[label]
                    exec(fct_strs_all[i][fct_idx].strip('` "'))
                    X_train_each = (
                        locals()[fct_names[i][fct_idx]](X_train)
                        .astype("int")
                        .to_numpy()
                    )
                    X_test_each = (
                        locals()[fct_names[i][fct_idx]](X_test).astype("int").to_numpy()
                    )
                    assert X_train_each.shape[1] == X_test_each.shape[1]
                    X_train_dict[label] = torch.tensor(X_train_each).float()
                    X_test_dict[label] = torch.tensor(X_test_each).float()

                X_train_all_dict[i] = X_train_dict
                X_test_all_dict[i] = X_test_dict
                executable_list.append(i)
            except (
                Exception
            ):  # If error occurred during the function call, remove the current trial
                continue

        return executable_list, X_train_all_dict, X_test_all_dict

    def _is_chat_model(self) -> bool:
        """
        Checks if the language model is a chat-based model.

        Returns:
            bool: True if the model is chat-based, False otherwise.
        """
        return isinstance(self.llm, LC.BaseChatModel)

    def _extract_rules(
        self,
        result_texts: List[str],
        label_list: List[str] = [],
    ) -> List[Dict[str, List[str]]]:
        """
        Extracts rules from the language model's responses.

        Args:
            result_texts (List[str]): List of responses from the language model.
            label_list (List[str], optional): List of unique labels. Defaults to [].

        Returns:
            List[Dict[str, List[str]]]: List of parsed rules for each class.
        """
        total_rules = []
        splitter = "onditions for class"
        for text in result_texts:
            splitted = text.split(splitter)
            if splitter not in text:
                continue
            if len(label_list) != 0 and len(splitted) != len(label_list) + 1:
                continue

            rule_raws = splitted[1:]
            rule_dict = {}
            for rule_raw in rule_raws:
                class_name = rule_raw.split(":")[0].strip(" .'").strip(' []"')
                rule_parsed = []
                for txt in rule_raw.strip().split("\n")[1:]:
                    if len(txt) < 2:
                        break
                    rule_parsed.append(" ".join(txt.strip().split(" ")[1:]))
                    rule_dict[class_name] = rule_parsed
            total_rules.append(rule_dict)
        return total_rules

    def _serialize(self, row):
        """
        Serializes a row of data into a descriptive string.

        Args:
            row (pd.Series): A row of data.

        Returns:
            str: Serialized string representation of the row.
        """
        target_str = f""
        for attr_idx, attr_name in enumerate(list(row.index)):
            if attr_idx < len(list(row.index)) - 1:
                target_str += " is ".join(
                    [attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()]
                )
                target_str += ". "
            else:
                if len(attr_name.strip()) < 2:
                    continue
                target_str += " is ".join(
                    [attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()]
                )
                target_str += "."
        return target_str

    def _fill_in_templates(self, fill_in_dict, template_str):
        """
        Fills in a template string with values from a dictionary.

        Args:
            fill_in_dict (dict): Dictionary of values to fill in.
            template_str (str): Template string.

        Returns:
            str: Filled-in template string.
        """
        return template_str.format(**fill_in_dict)

    def _set_seed(self, seed: int):
        """
        Sets the random seed for reproducibility.

        Args:
            seed (int): Random seed value.
        """
        random.seed(seed)
        np.random.seed(seed)

    def __call__(self, *args: Any, **kwargs: Any):
        """
        Executes the feature engineering process, including querying the language model,
        parsing rules, generating functions, and converting to binary vectors.

        Returns:
            Tuple: Executable function indices, label list, training and testing binary vectors,
                   and training and testing labels.
        """
        _DIVIDER = "\n\n---DIVIDER---\n\n"
        _VERSION = "\n\n---VERSION---\n\n"
        templates, feature_desc = self._generate_asking_prompt(
            df_all=self.df.drop(self.target_column, axis=1),
            df_x=self.X_train,
            df_y=self.y_train,
            label_list=self.label_list,
            default_target_column=self.target_column,
            is_cat=self.is_cat,
            query_num=self.query_num,
        )

        # File path to save
        current_folder = os.path.dirname(__file__)
        rules_folder = os.path.join(current_folder, "rules")
        if not os.path.exists(rules_folder):
            os.makedirs(rules_folder)
        rule_file_path = os.path.join(
            current_folder,
            "rules",
            f"rule-{self.data_name}-{self.shots}-{self.seed}.out",
        )
        function_file_path = os.path.join(
            current_folder,
            "rules",
            f"function-{self.data_name}-{self.shots}-{self.seed}.out",
        )

        # Generate rules and save it
        if os.path.isfile(rule_file_path) == False:
            results = self._query_llm(
                text_list=templates, max_tokens=1500, temperature=0.5
            )
            print(f"Writing rules to file: {rule_file_path}")
            with open(rule_file_path, "w") as f:
                total_rules = _DIVIDER.join(results)
                f.write(total_rules)
        else:
            print(f"loading rules from file: {rule_file_path}")
            with open(rule_file_path, "r") as f:
                total_rules_str = f.read().strip()
                results = total_rules_str.split(_DIVIDER)

        # Parse rules
        parsed_rules = self._extract_rules(
            result_texts=results, label_list=self.label_list
        )

        if os.path.isfile(function_file_path) == False:
            fct_strs_all = []
            for parsed_rule in tqdm(parsed_rules):
                fct_templates = self._generate_function_prompt(
                    parsed_rule, feature_desc
                )
            fct_results = self._query_llm(fct_templates, max_tokens=1500, temperature=0)
            fct_strs = [
                fct_txt.split("<start>")[1].split("<end>")[0].strip()
                for fct_txt in fct_results
            ]
            fct_strs_all.append(fct_strs)

            print(f"Writing functions to file: {function_file_path}")
            with open(function_file_path, "w") as f:
                total_str = _VERSION.join([_DIVIDER.join(x) for x in fct_strs_all])
                f.write(total_str)
        else:
            print(f"loading functions from file: {function_file_path}")
            with open(function_file_path, "r") as f:
                total_str = f.read().strip()
                fct_strs_all = [x.split(_DIVIDER) for x in total_str.split(_VERSION)]

        # Get function names and strings
        fct_names = []
        fct_strs_final = []
        for fct_str_pair in fct_strs_all:
            fct_pair_name = []
            if "def" not in fct_str_pair[0]:
                continue

            for fct_str in fct_str_pair:
                fct_pair_name.append(fct_str.split("def")[1].split("(")[0].strip())
            fct_names.append(fct_pair_name)
            fct_strs_final.append(fct_str_pair)

        # Convert to multi-hot vectors
        executable_list, X_train_all_dict, X_test_all_dict = (
            self._convert_to_binary_vectors(
                fct_strs_final, fct_names, self.label_list, self.X_train, self.X_test
            )
        )

        # Convert y to tensor
        y_train = torch.tensor([self.label_list.index(k) for k in self.y_train])
        y_test = torch.tensor([self.label_list.index(k) for k in self.y_test])
        return (
            executable_list,
            self.label_list,
            X_train_all_dict,
            X_test_all_dict,
            y_train,
            y_test,
        )


# example
if __name__ == "__main__":
    """
    Example usage of the FeatLLMEngineer class. Initializes the class with sample inputs
    and executes the feature engineering process.
    """
    API_KEY = "<Your API KEY>"
    API_URL = "<Your API URL>"
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model_name="Your Model Name", openai_api_base=API_URL, openai_api_key=API_KEY
    )
    feat_engineer = FeatLLMEngineer(
        file_path="Your File Path",
        metadata_path="Your Metadata Path",
        task_info_path="Your Task Info Path",
        llm=llm,
        query_num=1,
    )
    el, _, a, b, c, d = feat_engineer()
    print(d.dtype)
