import copy
from typing import Union, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from langchain_openai import ChatOpenAI
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
class LC:
    from langchain_core.language_models import BaseLLM
    from langchain.chat_models.base import BaseChatModel


from feat_engineer import FeatLLMEngineer


class simple_model(nn.Module):
    def __init__(self, X):
        super(simple_model, self).__init__()
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.ones(x_each.shape[1], 1) / x_each.shape[1])
                for x_each in X
            ]
        )

    def forward(self, x):
        x_total_score = []
        for idx, x_each in enumerate(x):
            x_score = x_each @ torch.clamp(self.weights[idx], min=0)
            x_total_score.append(x_score)
        x_total_score = torch.cat(x_total_score, dim=-1)
        return x_total_score


class FeatLLM:
    def __init__(
        self,
        file_path: str,
        metadata_path: str,
        task_info_path: str,
        llm: Union[LC.BaseChatModel, LC.BaseLLM] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.feat_engineer = FeatLLMEngineer(
            file_path=file_path,
            metadata_path=metadata_path,
            task_info_path=task_info_path,
            llm=llm,
            **kwargs,
        )

    def invoke(self):
        (
            executable_list,
            label_list,
            X_train_all_dict,
            X_test_all_dict,
            y_train,
            y_test,
        ) = self.feat_engineer()

        test_outputs_all = []
        multiclass = True if len(label_list) > 2 else False
        for i in executable_list:
            X_train_now = list(X_train_all_dict[i].values())
            X_test_now = list(X_test_all_dict[i].values())

            # Train
            trained_model = self._train(
                X_train_now, label_list, self.kwargs.get("shots", 4), y_train
            )

            # Evaluate
            test_outputs = trained_model(X_test_now).detach().cpu()
            test_outputs = F.softmax(test_outputs, dim=1).detach()
            result_auc = self._evaluate(
                test_outputs.numpy(), y_test.numpy(), multiclass=multiclass
            )
            print("AUC:", result_auc)
            test_outputs_all.append(test_outputs)
        test_outputs_all = np.stack(test_outputs_all, axis=0)
        ensembled_probs = test_outputs_all.mean(0)
        result_auc = self._evaluate(
            ensembled_probs, y_test.numpy(), multiclass=multiclass
        )
        print("Ensembled AUC:", result_auc)
        pass

    def _train(
        self,
        X_train_now: List[Tensor],
        label_list: List,
        shot: int,
        y_train: Tensor,
    ):
        criterion = nn.CrossEntropyLoss()
        if shot // len(label_list) == 1:
            model = simple_model(X_train_now)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(200):
                optimizer.zero_grad()
                outputs = model(X_train_now)
                preds = outputs.argmax(dim=1)
                acc = (y_train == preds).sum() / len(preds)
                if acc == 1:
                    break
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
        else:
            # K-fold cross-validation.
            if shot // len(label_list) <= 2:
                n_splits = 2
            else:
                n_splits = 4

            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
            model_list = []
            for _, (train_ids, valid_ids) in enumerate(
                kfold.split(X_train_now[0], y_train)
            ):
                model = simple_model(X_train_now)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
                X_train_now_fold = [
                    x_train_now[train_ids] for x_train_now in X_train_now
                ]
                X_valid_now_fold = [
                    x_train_now[valid_ids] for x_train_now in X_train_now
                ]
                y_train_fold = y_train[train_ids]
                y_valid_fold = y_train[valid_ids]

                max_acc = -1
                for _ in range(200):
                    optimizer.zero_grad()
                    outputs = model(X_train_now_fold)
                    loss = criterion(outputs, y_train_fold)
                    loss.backward()
                    optimizer.step()

                    valid_outputs = model(X_valid_now_fold)
                    preds = valid_outputs.argmax(dim=1)
                    acc = (y_valid_fold == preds).sum() / len(preds)
                    if max_acc < acc:
                        max_acc = acc
                        final_model = copy.deepcopy(model)
                        if max_acc >= 1:
                            break
                model_list.append(final_model)

            sdict = model_list[0].state_dict()
            for key in sdict:
                sdict[key] = torch.stack(
                    [model.state_dict()[key] for model in model_list], dim=0
                ).mean(dim=0)

            model = simple_model(X_train_now)
            model.load_state_dict(sdict)
        return model

    def _evaluate(self, pred_probs, answers, multiclass=False):
        if multiclass == False:
            result_auc = roc_auc_score(answers, pred_probs[:, 1])
        else:
            result_auc = roc_auc_score(
                answers, pred_probs, multi_class="ovr", average="macro"
            )
        return result_auc
    
    def __call__(self, *args: Any, **kwargs: Any):
        self.invoke()


if __name__ == "__main__":
    API_KEY = "<Your API KEY>"
    API_URL = "<Your API URL>"
    # Example usage
    llm = ChatOpenAI(
        model_name="Your Model Name",
        openai_api_base=API_URL,
        openai_api_key=API_KEY
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
