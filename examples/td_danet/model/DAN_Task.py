import torch
import numpy as np
from scipy.special import softmax
from lib.utils import PredictDataset
from model.abstract_model import DANsModel
from lib.multiclass_utils import infer_output_dim, check_output_dim
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy, mse_loss


class DANetClassifier(DANsModel):
    def __post_init__(self):
        super(DANetClassifier, self).__post_init__()
        self._task = 'classification'
        self._default_loss = cross_entropy
        self._default_metric = 'accuracy'

    def weight_updater(self, weights):
        """
        Updates weights dictionary according to target_mapper.

        Parameters
        ----------
        weights : bool or dict
            Given weights for balancing training.

        Returns
        -------
        bool or dict
            Same bool if weights are bool, updated dict otherwise.

        """
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    def prepare_target(self, y):
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.long())

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set
    ):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = 'accuracy'
        self.classes_ = train_labels
        self.target_mapper = {class_label: index for index, class_label in enumerate(self.classes_)}
        self.preds_mapper = {str(index): class_label for index, class_label in enumerate(self.classes_)}

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        y_score = softmax(y_score, axis=1)
        return y_true, y_score

    def predict_func(self, outputs):
        outputs = np.argmax(outputs, axis=1)
        return outputs

    def predict_proba(self, X):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data

        Returns
        -------
        res : np.ndarray

        """
        self.network.eval()

        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=32,
            shuffle=False,
        )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            output = self.network(data)
            predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return res


class DANetRegressor(DANsModel):
    def __post_init__(self):
        super(DANetRegressor, self).__post_init__()
        self._task = 'regression'
        self._default_loss = mse_loss
        self._default_metric = 'mse'

    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set
    ):
        if len(y_train.shape) != 2:
            msg = "Targets should be 2D : (n_samples, n_regression) " + \
                  f"but y_train.shape={y_train.shape} given.\n" + \
                  "Use reshape(-1, 1) for single regression."
            raise ValueError(msg)
        self.output_dim = y_train.shape[1]
        self.preds_mapper = None

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score
