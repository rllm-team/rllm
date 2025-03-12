import os.path as osp

import torch
from time import time
import numpy as np


from core.GNNs.gnn_utils import EarlyStopping
from core.utils import time_logger

import sys

sys.path.append("../../")
from rllm.datasets.tape import TAPEDataset

LOG_FREQ = 10


class GNNTrainer:

    def __init__(self, cfg, feature_type):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = feature_type
        self.epochs = cfg.gnn.train.epochs

        # ! Load data
        # data, num_classes = load_data(
        #     self.dataset_name, use_dgl=False, use_text=False, seed=self.seed)
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../../../..", "data")
        dataset = TAPEDataset(
            path,
            self.dataset_name,
            use_text=False,
            use_gpt=False,
            use_preds=True,
            topk=3 if self.dataset_name == "pubmed" else 5,
        )
        data = dataset[0]

        self.num_nodes = data.y.shape[0]
        self.num_classes = data.num_classes
        data.y = data.y.squeeze()

        # ! Init gnn feature
        topk = 3 if self.dataset_name == "pubmed" else 5
        if self.feature_type == "TA":
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = (
                f"./examples/tape/prt_lm/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.emb"
            )
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(
                np.array(
                    np.memmap(
                        LM_emb_path,
                        mode="r",
                        dtype=np.float16,
                        shape=(self.num_nodes, 768),
                    )
                )
            ).to(torch.float32)
        elif self.feature_type == "E":
            print("Loading pretrained LM features (explanations) ...")
            LM_emb_path = (
                f"./examples/tape/prt_lm/{self.dataset_name}2/{self.lm_model_name}-seed{self.seed}.emb"
            )
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(
                np.array(
                    np.memmap(
                        LM_emb_path,
                        mode="r",
                        dtype=np.float16,
                        shape=(self.num_nodes, 768),
                    )
                )
            ).to(torch.float32)
        elif self.feature_type == "P":
            print("Loading top-k prediction features ...")
            features = data.gpt_preds

        self.features = features.to(self.device)
        self.data = data.to(self.device)

        # ! Trainer init
        use_pred = self.feature_type == "P"

        if self.gnn_model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif self.gnn_model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        elif self.gnn_model_name == "MLP":
            from core.GNNs.MLP.model import MLP as GNN
        else:
            print(f"Model {self.gnn_model_name} is not supported! Loading MLP ...")
            from core.GNNs.MLP.model import MLP as GNN

        self.model = GNN(
            in_dim=self.hidden_dim * topk if use_pred else self.features.shape[1],
            hidden_dim=self.hidden_dim,
            out_dim=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_pred=use_pred,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0
        )

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"./examples/tape/output/{self.dataset_name}/{self.gnn_model_name}.pt"
        self.stopper = (
            EarlyStopping(patience=cfg.gnn.train.early_stop, path=self.ckpt)
            if cfg.gnn.train.early_stop > 0
            else None
        )
        self.loss_func = torch.nn.CrossEntropyLoss()

        from core.GNNs.gnn_utils import Evaluator

        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels.view(-1, 1)}
        )["acc"]

    def _forward(self, x, adj):
        logits = self.model(x, adj)  # small-graph
        return logits

    def _train(self):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.features, self.data.adj)
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask]
        )
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask]
        )
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self._forward(self.features, self.data.adj)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask]
        )
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask]
        )
        return val_acc, test_acc, logits

    @time_logger
    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ""
            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f"Early stopped, loading model from epoch-{self.stopper.best_epoch}"
                    )
                    break
            if epoch % LOG_FREQ == 0:
                print(
                    f"Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}"
                )

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, logits = self._evaluate()
        print(
            f"[{self.gnn_model_name} + {self.feature_type}] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n"
        )
        res = {"val_acc": val_acc, "test_acc": test_acc}
        return logits, res
