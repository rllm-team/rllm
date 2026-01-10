import torch
import numpy as np
from typing import Union, Dict


# Feature-wise Mixup
def batch_feat_shuffle(Xs: Union[torch.Tensor, Dict[str, torch.Tensor]], beta=0.5):
    # Handle dictionary input
    if isinstance(Xs, dict):
        # Get batch size from first tensor in dict
        first_key = list(Xs.keys())[0]
        b = Xs[first_key].shape[0]
        device = Xs[first_key].device

        # Generate shared shuffle_rates and shuffled_sample_ids for all tensors
        shuffle_rates = np.random.beta(beta, beta, size=(b, 1))
        shuffled_sample_ids = np.random.permutation(b)

        # Apply mixup to each tensor in dictionary with its own feat_masks
        Xs_mixup = {}
        feat_masks_dict = {}
        for key, X in Xs.items():
            f = X.shape[1]  # Number of features for this specific tensor
            # Generate feat_masks specific to this tensor's feature size
            feat_masks = np.random.random(size=(b, f)) > shuffle_rates  # b f
            feat_masks = torch.from_numpy(feat_masks).to(device)

            Xs_shuffled = X[shuffled_sample_ids]
            feat_masks_key = feat_masks.unsqueeze(-1) if X.ndim == 3 else feat_masks
            Xs_mixup[key] = feat_masks_key * X + ~feat_masks_key * Xs_shuffled
            feat_masks_dict[key] = feat_masks

        # Return average feat_masks or feat_masks from first key as scalar representation
        return Xs_mixup, feat_masks_dict[first_key], shuffled_sample_ids

    # Handle tensor input (original logic)
    b, f = Xs.shape[0], Xs.shape[1]
    shuffle_rates = np.random.beta(beta, beta, size=(b, 1))
    feat_masks = np.random.random(size=(b, f)) > shuffle_rates  # b f
    feat_masks = torch.from_numpy(feat_masks).to(Xs.device)

    shuffled_sample_ids = np.random.permutation(b)

    Xs_shuffled = Xs[shuffled_sample_ids]
    feat_masks = feat_masks.unsqueeze(-1) if Xs.ndim == 3 else feat_masks
    Xs_mixup = feat_masks * Xs + ~feat_masks * Xs_shuffled

    return Xs_mixup, feat_masks.squeeze(-1), shuffled_sample_ids


# Dim-wise Mixup
def batch_dim_shuffle(Xs: Union[torch.Tensor, Dict[str, torch.Tensor]], beta=0.5):
    # Handle dictionary input
    if isinstance(Xs, dict):
        # Get batch size from first tensor in dict
        first_key = list(Xs.keys())[0]
        b = Xs[first_key].shape[0]
        device = Xs[first_key].device

        # Generate shared shuffle_rates and shuffled_sample_ids for all tensors
        shuffle_rates = np.random.beta(beta, beta, size=(b, 1))
        shuffled_sample_ids = np.random.permutation(b)

        # Apply mixup to each tensor in dictionary with its own dim_masks
        Xs_mixup = {}
        for key, X in Xs.items():
            d = X.shape[2]  # Dimension size for this specific tensor
            # Generate dim_masks specific to this tensor's dimension size
            dim_masks = np.random.random(size=(b, d)) < shuffle_rates  # b d
            dim_masks = torch.from_numpy(dim_masks).to(device)

            Xs_shuffled = X[shuffled_sample_ids]
            dim_masks_key = dim_masks.unsqueeze(1)  # b 1 d
            Xs_mixup[key] = dim_masks_key * X + ~dim_masks_key * Xs_shuffled

        return (
            Xs_mixup,
            torch.from_numpy(shuffle_rates[:, 0]).float().to(device),
            shuffled_sample_ids,
        )

    # Handle tensor input (original logic)
    b, f, d = Xs.shape
    shuffle_rates = np.random.beta(beta, beta, size=(b, 1))
    dim_masks = np.random.random(size=(b, d)) < shuffle_rates  # b d
    dim_masks = torch.from_numpy(dim_masks).to(Xs.device)

    shuffled_sample_ids = np.random.permutation(b)

    Xs_shuffled = Xs[shuffled_sample_ids]
    dim_masks = dim_masks.unsqueeze(1)  # b 1 d
    Xs_mixup = dim_masks * Xs + ~dim_masks * Xs_shuffled

    return (
        Xs_mixup,
        torch.from_numpy(shuffle_rates[:, 0]).float().to(Xs.device),
        shuffled_sample_ids,
    )


# Naive Mixup
def mixup_data(Xs: Union[torch.Tensor, Dict[str, torch.Tensor]], beta=0.5):
    # Handle dictionary input
    if isinstance(Xs, dict):
        # Get batch size from first tensor in dict
        first_key = list(Xs.keys())[0]
        b = Xs[first_key].shape[0]

        lam = np.random.beta(beta, beta)
        shuffle_sample_ids = np.random.permutation(b)

        # Apply mixup to each tensor in dictionary
        mixed_X = {}
        for key, X in Xs.items():
            mixed_X[key] = lam * X + (1 - lam) * X[shuffle_sample_ids]

        return mixed_X, lam, shuffle_sample_ids

    # Handle tensor input (original logic)
    b, f = Xs.shape
    lam = np.random.beta(beta, beta)
    shuffle_sample_ids = np.random.permutation(b)
    mixed_X = lam * Xs + (1 - lam) * Xs[shuffle_sample_ids]
    # shuffle_sample_ids = torch.randperm(b).to(Xs.device)
    # mixed_X = lam * Xs + (1 - lam) * Xs[shuffle_sample_ids, :]
    return mixed_X, lam, shuffle_sample_ids
