from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    log_loss,
    balanced_accuracy_score,
    mean_squared_log_error,
)
import torch


def UnsupervisedLoss(y_pred, embedded_x, obf_vars, eps=1e-9):
    errors = y_pred - embedded_x
    reconstruction_errors = torch.mul(errors, obf_vars) ** 2
    batch_means = torch.mean(embedded_x, dim=0)
    batch_means[batch_means == 0] = 1

    batch_stds = torch.std(embedded_x, dim=0) ** 2
    batch_stds[batch_stds == 0] = batch_means[batch_stds == 0]
    features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
    # take the mean of the reconstructed variable errors
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    # here we take the mean per batch, contrary to the paper
    loss = torch.mean(features_loss)
    return loss


def UnsupervisedLossNumpy(y_pred, embedded_x, obf_vars, eps=1e-9):
    errors = y_pred - embedded_x
    reconstruction_errors = np.multiply(errors, obf_vars) ** 2
    batch_means = np.mean(embedded_x, axis=0)
    batch_means = np.where(batch_means == 0, 1, batch_means)

    batch_stds = np.std(embedded_x, axis=0, ddof=1) ** 2
    batch_stds = np.where(batch_stds == 0, batch_means, batch_stds)
    features_loss = np.matmul(reconstruction_errors, 1 / batch_stds)
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables = np.sum(obf_vars, axis=1)
    # take the mean of the reconstructed variable errors
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    # here we take the mean per batch, contrary to the paper
    loss = np.mean(features_loss)
    return loss


@dataclass
class MetricContainer:

    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self):
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_true, y_pred):
        logs = {}
        for metric in self.metrics:
            if isinstance(y_pred, list):
                res = np.mean(
                    [metric(y_true[:, i], y_pred[i]) for i in range(len(y_pred))]
                )
            else:
                res = metric(y_true, y_pred)
            logs[self.prefix + metric._name] = res
        return logs


class Metric:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError("Custom Metrics must implement this function")

    @classmethod
    def get_metrics_by_names(cls, names):
        available_metrics = cls.__subclasses__()
        available_names = [metric()._name for metric in available_metrics]
        metrics = []
        for name in names:
            assert (
                name in available_names
            ), f"{name} is not available, choose in {available_names}"
            idx = available_names.index(name)
            metric = available_metrics[idx]()
            metrics.append(metric)
        return metrics


class AUC(Metric):
    """
    AUC.
    """

    def __init__(self):
        self._name = "auc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return roc_auc_score(y_true, y_score[:, 1])


class Accuracy(Metric):
    """
    Accuracy.
    """

    def __init__(self):
        self._name = "accuracy"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return accuracy_score(y_true, y_pred)


class BalancedAccuracy(Metric):
    """
    Balanced Accuracy.
    """

    def __init__(self):
        self._name = "balanced_accuracy"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return balanced_accuracy_score(y_true, y_pred)


class LogLoss(Metric):
    """
    LogLoss.
    """

    def __init__(self):
        self._name = "logloss"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return log_loss(y_true, y_score)


class MAE(Metric):
    """
    Mean Absolute Error.
    """

    def __init__(self):
        self._name = "mae"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return mean_absolute_error(y_true, y_score)


class MSE(Metric):
    """
    Mean Squared Error.
    """

    def __init__(self):
        self._name = "mse"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return mean_squared_error(y_true, y_score)


class RMSLE(Metric):

    def __init__(self):
        self._name = "rmsle"
        self._maximize = False

    def __call__(self, y_true, y_score):
        y_score = np.clip(y_score, a_min=0, a_max=None)
        return np.sqrt(mean_squared_log_error(y_true, y_score))


class UnsupervisedMetric(Metric):
    """
    Unsupervised metric
    """

    def __init__(self):
        self._name = "unsup_loss"
        self._maximize = False

    def __call__(self, y_pred, embedded_x, obf_vars):
        loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)
        return loss.item()


class UnsupervisedNumpyMetric(Metric):
    """
    Unsupervised metric
    """

    def __init__(self):
        self._name = "unsup_loss_numpy"
        self._maximize = False

    def __call__(self, y_pred, embedded_x, obf_vars):
        return UnsupervisedLossNumpy(
            y_pred,
            embedded_x,
            obf_vars
        )


class RMSE(Metric):
    """
    Root Mean Squared Error.
    """

    def __init__(self):
        self._name = "rmse"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return np.sqrt(mean_squared_error(y_true, y_score))


def check_metrics(metrics):
    val_metrics = []
    for metric in metrics:
        if isinstance(metric, str):
            val_metrics.append(metric)
        elif issubclass(metric, Metric):
            val_metrics.append(metric()._name)
        else:
            raise TypeError("You need to provide a valid metric format")
    return val_metrics
