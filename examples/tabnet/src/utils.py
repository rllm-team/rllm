from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import scipy
import json

from collections.abc import Sequence
from itertools import chain

from scipy.sparse import issparse
from scipy.sparse.base import spmatrix
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import scipy.sparse as sp

import pandas as pd


class TorchDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x, y


class SparseTorchDataset(Dataset):
    """
    Format for csr_matrix

    Parameters
    ----------
    X : CSR matrix
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index].toarray()[0]).float()
        y = self.y[index]
        return x, y


class PredictDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        return x


class SparsePredictDataset(Dataset):
    """
    Format for csr_matrix

    Parameters
    ----------
    X : CSR matrix
        The input matrix
    """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index].toarray()[0]).float()
        return x


def create_sampler(weights, y_train):
    """
    This creates a sampler from the given weights

    Parameters
    ----------
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    y_train : np.array
        Training targets
    """
    if isinstance(weights, int):
        if weights == 0:
            need_shuffle = True
            sampler = None
        elif weights == 1:
            need_shuffle = False
            class_sample_count = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
            )

            weights = 1.0 / class_sample_count

            samples_weight = np.array([weights[t] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            raise ValueError("Weights should be either 0, 1, dictionnary or list.")
    elif isinstance(weights, dict):
        # custom weights per class
        need_shuffle = False
        samples_weight = np.array([weights[t] for t in y_train])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        # custom weights
        if len(weights) != len(y_train):
            raise ValueError("Custom weights should match number of train samples.")
        need_shuffle = False
        samples_weight = np.array(weights)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return need_shuffle, sampler


def create_dataloaders(
    X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory
):
    """
    Create dataloaders with or without subsampling depending on weights and balanced.
    """
    need_shuffle, sampler = create_sampler(weights, y_train)

    if scipy.sparse.issparse(X_train):
        train_dataloader = DataLoader(
            SparseTorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
    else:
        train_dataloader = DataLoader(
            TorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    valid_dataloaders = []
    for X, y in eval_set:
        if scipy.sparse.issparse(X):
            valid_dataloaders.append(
                DataLoader(
                    SparseTorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )
        else:
            valid_dataloaders.append(
                DataLoader(
                    TorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )

    return train_dataloader, valid_dataloaders


def create_explain_matrix(input_dim, cat_emb_dim, cat_idxs, post_embed_dim):
    """
    This is a computational trick.
    In order to rapidly sum importances from same embeddings
    to the initial index.
    """

    if isinstance(cat_emb_dim, int):
        all_emb_impact = [cat_emb_dim - 1] * len(cat_idxs)
    else:
        all_emb_impact = [emb_dim - 1 for emb_dim in cat_emb_dim]

    acc_emb = 0
    nb_emb = 0
    indices_trick = []
    for i in range(input_dim):
        if i not in cat_idxs:
            indices_trick.append([i + acc_emb])
        else:
            indices_trick.append(
                range(i + acc_emb, i + acc_emb + all_emb_impact[nb_emb] + 1)
            )
            acc_emb += all_emb_impact[nb_emb]
            nb_emb += 1

    reducing_matrix = np.zeros((post_embed_dim, input_dim))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1

    return scipy.sparse.csc_matrix(reducing_matrix)


def create_group_matrix(list_groups, input_dim):
    """
    Create the group matrix corresponding to the given list_groups

    """
    check_list_groups(list_groups, input_dim)

    if len(list_groups) == 0:
        group_matrix = torch.eye(input_dim)
        return group_matrix
    else:
        n_groups = input_dim - int(np.sum([len(gp) - 1 for gp in list_groups]))
        group_matrix = torch.zeros((n_groups, input_dim))

        remaining_features = [feat_idx for feat_idx in range(input_dim)]

        current_group_idx = 0
        for group in list_groups:
            group_size = len(group)
            for elem_idx in group:
                # add importrance of element in group matrix and corresponding group
                group_matrix[current_group_idx, elem_idx] = 1 / group_size
                # remove features from list of features
                remaining_features.remove(elem_idx)
            # move to next group
            current_group_idx += 1
        # features not mentionned in list_groups get assigned their own group of singleton
        for remaining_feat_idx in remaining_features:
            group_matrix[current_group_idx, remaining_feat_idx] = 1
            current_group_idx += 1
        return group_matrix


def check_list_groups(list_groups, input_dim):
    """
    Check that list groups:
        - is a list of list
        - does not contain twice the same feature in different groups
        - does not contain unknown features (>= input_dim)
        - does not contain empty groups
    """
    assert isinstance(list_groups, list), "list_groups must be a list of list."

    if len(list_groups) == 0:
        return
    else:
        for group_pos, group in enumerate(list_groups):
            msg = f"Groups must be given as a list of list, but found {group} in position {group_pos}."  # noqa
            assert isinstance(group, list), msg
            assert len(group) > 0, "Empty groups are forbidding please remove empty groups []"

    n_elements_in_groups = np.sum([len(group) for group in list_groups])
    flat_list = []
    for group in list_groups:
        flat_list.extend(group)
    unique_elements = np.unique(flat_list)
    n_unique_elements_in_groups = len(unique_elements)
    msg = "One feature can only appear in one group, please check your grouped_features."
    assert n_unique_elements_in_groups == n_elements_in_groups, msg

    highest_feat = np.max(unique_elements)
    assert highest_feat < input_dim, f"Number of features is {input_dim} but one group contains {highest_feat}."  # noqa
    return


def filter_weights(weights):
    """
    This function makes sure that weights are in correct format for
    regression and multitask TabNet
    """
    err_msg = """Please provide a list or np.array of weights for """
    err_msg += """regression, multitask or pretraining: """
    if isinstance(weights, int):
        if weights == 1:
            raise ValueError(err_msg + "1 given.")
    if isinstance(weights, dict):
        raise ValueError(err_msg + "Dict given.")
    return


def validate_eval_set(eval_set, eval_name, X_train, y_train):
    """Check if the shapes of eval_set are compatible with (X_train, y_train).
    """
    eval_name = eval_name or [f"val_{i}" for i in range(len(eval_set))]

    assert len(eval_set) == len(
        eval_name
    ), "eval_set and eval_name have not the same length"
    if len(eval_set) > 0:
        assert all(
            len(elem) == 2 for elem in eval_set
        ), "Each tuple of eval_set need to have two elements"
    for name, (X, y) in zip(eval_name, eval_set):
        msg = (
            f"Dimension mismatch between X_{name} "
            + f"{X.shape} and X_train {X_train.shape}"
        )
        assert len(X.shape) == len(X_train.shape), msg

        msg = (
            f"Dimension mismatch between y_{name} "
            + f"{y.shape} and y_train {y_train.shape}"
        )
        assert len(y.shape) == len(y_train.shape), msg

        msg = (
            f"Number of columns is different between X_{name} "
            + f"({X.shape[1]}) and X_train ({X_train.shape[1]})"
        )
        assert X.shape[1] == X_train.shape[1], msg

        if len(y_train.shape) == 2:
            msg = (
                f"Number of columns is different between y_{name} "
                + f"({y.shape[1]}) and y_train ({y_train.shape[1]})"
            )
            assert y.shape[1] == y_train.shape[1], msg
        msg = (
            f"You need the same number of rows between X_{name} "
            + f"({X.shape[0]}) and y_{name} ({y.shape[0]})"
        )
        assert X.shape[0] == y.shape[0], msg

    return eval_name, eval_set


def define_device(device_name):
    """
    Define the device to use during training and inference.
    If auto it will detect automatically whether to use cuda or cpu
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    else:
        return device_name


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim):
    if (cat_dims == []) ^ (cat_idxs == []):
        if cat_dims == []:
            msg = "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."
        else:
            msg = "If cat_dims is non-empty, cat_idxs must be defined as a list of same length."
        raise ValueError(msg)
    elif len(cat_dims) != len(cat_idxs):
        msg = "The lists cat_dims and cat_idxs must have the same length."
        raise ValueError(msg)

    if isinstance(cat_emb_dim, int):
        cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
    else:
        cat_emb_dims = cat_emb_dim

    # check that all embeddings are provided
    if len(cat_emb_dims) != len(cat_dims):
        msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(cat_emb_dims)}
                    and {len(cat_dims)}"""
        raise ValueError(msg)

    # Rearrange to get reproducible seeds with different ordering
    if len(cat_idxs) > 0:
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        cat_emb_dims = [cat_emb_dims[i] for i in sorted_idxs]

    return cat_dims, cat_idxs, cat_emb_dims

def _assert_all_finite(X, allow_nan=False):
    """Like assert_all_finite, but only for ndarray."""

    X = np.asanyarray(X)
    is_float = X.dtype.kind in "fc"
    if is_float and (np.isfinite(np.sum(X))):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if (
            allow_nan
            and np.isinf(X).any()
            or not allow_nan
            and not np.isfinite(X).all()
        ):
            type_err = "infinity" if allow_nan else "NaN, infinity"
            raise ValueError(msg_err.format(type_err, X.dtype))
    elif X.dtype == np.dtype("object") and not allow_nan:
        if np.isnan(X).any():
            raise ValueError("Input contains NaN")


def assert_all_finite(X, allow_nan=False):
    _assert_all_finite(X.data if sp.issparse(X) else X, allow_nan)


def _unique_multiclass(y):
    if hasattr(y, "__array__"):
        return np.unique(np.asarray(y))
    else:
        return set(y)


def _unique_indicator(y):
    """
    Not implemented
    """
    raise IndexError(
        f"""Given labels are of size {y.shape} while they should be (n_samples,) \n"""
        + """If attempting multilabel classification, try using TabNetMultiTaskClassification """
        + """or TabNetRegressor"""
    )


_FN_UNIQUE_LABELS = {
    "binary": _unique_multiclass,
    "multiclass": _unique_multiclass,
    "multilabel-indicator": _unique_indicator,
}


def unique_labels(*ys):
    if not ys:
        raise ValueError("No argument has been passed.")
    # Check that we don't mix label format

    ys_types = set(type_of_target(x) for x in ys)
    if ys_types == {"binary", "multiclass"}:
        ys_types = {"multiclass"}

    if len(ys_types) > 1:
        raise ValueError("Mix type of y not allowed, got types %s" % ys_types)

    label_type = ys_types.pop()

    # Get the unique set of labels
    _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
    if not _unique_labels:
        raise ValueError("Unknown label type: %s" % repr(ys))

    ys_labels = set(chain.from_iterable(_unique_labels(y) for y in ys))

    # Check that we don't mix string type with number type
    if len(set(isinstance(label, str) for label in ys_labels)) > 1:
        raise ValueError("Mix of label input types (string and number)")

    return np.array(sorted(ys_labels))


def _is_integral_float(y):
    return y.dtype.kind == "f" and np.all(y.astype(int) == y)


def is_multilabel(y):
    """Check if ``y`` is in a multilabel format.
    """
    if hasattr(y, "__array__"):
        y = np.asarray(y)
    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False

    if issparse(y):
        if isinstance(y, (dok_matrix, lil_matrix)):
            y = y.tocsr()
        return (
            len(y.data) == 0
            or np.unique(y.data).size == 1
            and (
                y.dtype.kind in "biu"
                or _is_integral_float(np.unique(y.data))  # bool, int, uint
            )
        )
    else:
        labels = np.unique(y)

        return len(labels) < 3 and (
            y.dtype.kind in "biu" or _is_integral_float(labels)  # bool, int, uint
        )


def check_classification_targets(y):
    y_type = type_of_target(y)
    if y_type not in [
        "binary",
        "multiclass",
        "multiclass-multioutput",
        "multilabel-indicator",
        "multilabel-sequences",
    ]:
        raise ValueError("Unknown label type: %r" % y_type)


def type_of_target(y):
    valid = (
        isinstance(y, (Sequence, spmatrix)) or hasattr(y, "__array__")
    ) and not isinstance(y, str)

    if not valid:
        raise ValueError(
            "Expected array-like (array or non-string sequence), " "got %r" % y
        )

    sparseseries = y.__class__.__name__ == "SparseSeries"
    if sparseseries:
        raise ValueError("y cannot be class 'SparseSeries'.")

    if is_multilabel(y):
        return "multilabel-indicator"

    try:
        y = np.asarray(y)
    except ValueError:
        # Known to fail in numpy 1.3 for array of arrays
        return "unknown"

    # The old sequence of sequences format
    try:
        if (
            not hasattr(y[0], "__array__")
            and isinstance(y[0], Sequence)
            and not isinstance(y[0], str)
        ):
            raise ValueError(
                "You appear to be using a legacy multi-label data"
                " representation. Sequence of sequences are no"
                " longer supported; use a binary array or sparse"
                " matrix instead - the MultiLabelBinarizer"
                " transformer can convert to this format."
            )
    except IndexError:
        pass

    # Invalid inputs
    if y.ndim > 2 or (y.dtype == object and len(y) and not isinstance(y.flat[0], str)):
        return "unknown"  # [[[1, 2]]] or [obj_1] and not ["label_1"]

    if y.ndim == 2 and y.shape[1] == 0:
        return "unknown"

    if y.ndim == 2 and y.shape[1] > 1:
        suffix = "-multioutput" 
    else:
        suffix = ""

    # check float and contains non-integer float values
    if y.dtype.kind == "f" and np.any(y != y.astype(int)):
        _assert_all_finite(y)
        return "continuous" + suffix

    if (len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        return "multiclass" + suffix 
    else:
        return "binary" 


def check_unique_type(y):
    target_types = pd.Series(y).map(type).unique()
    if len(target_types) != 1:
        raise TypeError(
            f"Values on the target must have the same type. Target has types {target_types}"
        )


def infer_output_dim(y_train):
    """
    Infer output_dim from targets
    """
    check_unique_type(y_train)
    train_labels = unique_labels(y_train)
    output_dim = len(train_labels)

    return output_dim, train_labels


def check_output_dim(labels, y):
    if y is not None:
        check_unique_type(y)
        valid_labels = unique_labels(y)
        if not set(valid_labels).issubset(set(labels)):
            raise ValueError(
                f"""Valid set -- {set(valid_labels)} --
                             contains unkown targets from training --
                             {set(labels)}"""
            )
    return


def infer_multitask_output(y_train):

    if len(y_train.shape) < 2:
        raise ValueError(
            "y_train should be of shape (n_examples, n_tasks)"
            + f"but got {y_train.shape}"
        )
    nb_tasks = y_train.shape[1]
    tasks_dims = []
    tasks_labels = []
    for task_idx in range(nb_tasks):
        try:
            output_dim, train_labels = infer_output_dim(y_train[:, task_idx])
            tasks_dims.append(output_dim)
            tasks_labels.append(train_labels)
        except ValueError as err:
            raise ValueError(f"""Error for task {task_idx} : {err}""")
    return tasks_dims, tasks_labels
