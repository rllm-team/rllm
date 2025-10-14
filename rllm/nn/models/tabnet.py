from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from rllm.types import ColType
from rllm.nn.pre_encoder import FTTransformerPreEncoder


def check_list_groups(list_groups, in_dim):
    """
    Check that list groups:
        - is a list of list
        - does not contain twice the same feature in different groups
        - does not contain unknown features (>= in_dim)
        - does not contain empty groups
    """
    assert isinstance(list_groups, list), "list_groups must be a list of list."

    if len(list_groups) == 0:
        return
    else:
        for group_pos, group in enumerate(list_groups):
            msg = f"Groups must be given as a list of list, but found {group} in position {group_pos}."  # noqa
            assert isinstance(group, list), msg
            assert len(group) > 0, "Empty groups are forbiddien, "
            "please remove empty groups []"

    n_elements_in_groups = np.sum([len(group) for group in list_groups])
    flat_list = []
    for group in list_groups:
        flat_list.extend(group)
    unique_elements = np.unique(flat_list)
    n_unique_elements_in_groups = len(unique_elements)
    msg = "One feature can only appear in one group, "
    "please check your grouped_features."
    assert n_unique_elements_in_groups == n_elements_in_groups, msg

    highest_feat = np.max(unique_elements)
    assert (
        highest_feat < in_dim
    ), f"Number of features is {in_dim} but one group contains {highest_feat}."  # noqa
    return


def create_group_matrix(list_groups, in_dim):
    """
    Create the group matrix corresponding to the given list_groups

    """
    check_list_groups(list_groups, in_dim)

    if len(list_groups) == 0:
        group_matrix = torch.eye(in_dim)
        return group_matrix
    else:
        n_groups = in_dim - int(np.sum([len(gp) - 1 for gp in list_groups]))
        group_matrix = torch.zeros((n_groups, in_dim))

        remaining_features = [feat_idx for feat_idx in range(in_dim)]

        current_group_idx = 0
        for group in list_groups:
            group_size = len(group)
            for elem_idx in group:
                # add importrance of element in group matrix and
                # corresponding group
                group_matrix[current_group_idx, elem_idx] = 1 / group_size
                # remove features from list of features
                remaining_features.remove(elem_idx)
            # move to next group
            current_group_idx += 1
        # features not mentionned in list_groups will
        # get assigned their own group of singleton
        for remaining_feat_idx in remaining_features:
            group_matrix[current_group_idx, remaining_feat_idx] = 1
            current_group_idx += 1
        return group_matrix


def create_emb_group_matrix(
    group_matrix, cat_idxs, in_dim, cat_emb_dim, post_embed_dim
):
    # record continuous indices
    continuous_idx = torch.ones(in_dim, dtype=torch.bool)
    continuous_idx[cat_idxs] = 0

    # update group matrix
    n_groups = group_matrix.shape[0]
    embedding_group_matrix = torch.empty(
        (n_groups, post_embed_dim), device=group_matrix.device
    )
    for group_idx in range(n_groups):
        post_emb_idx = 0
        for init_feat_idx in range(in_dim):
            if continuous_idx[init_feat_idx] == 1:
                # this means that no embedding is applied to this column
                embedding_group_matrix[group_idx, post_emb_idx] = group_matrix[
                    group_idx, init_feat_idx
                ]  # noqa
                post_emb_idx += 1
            else:
                # a categorical feature which creates multiple embeddings
                n_embeddings = cat_emb_dim
                embedding_group_matrix[
                    group_idx, post_emb_idx : post_emb_idx + n_embeddings
                ] = (
                    group_matrix[group_idx, init_feat_idx] / n_embeddings
                )  # noqa
                post_emb_idx += n_embeddings
    return embedding_group_matrix


def initialize_non_glu(module, in_dim, out_dim):
    gain_value = np.sqrt((in_dim + out_dim) / np.sqrt(4 * in_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


def initialize_glu(module, in_dim, out_dim):
    gain_value = np.sqrt((in_dim + out_dim) / np.sqrt(in_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class GBN(torch.nn.Module):
    r"""Ghost Batch Normalization,
    see `"Train longer, generalize better: closing the
    generalization gap in large batch training of neural networks"
    <https://arxiv.org/abs/1705.08741>`_ paper
    """

    def __init__(self, in_dim, virtual_batch_size=128, momentum=0.01):
        super().__init__()

        self.in_dim = in_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = torch.nn.BatchNorm1d(self.in_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


class AttentiveTransformer(torch.nn.Module):
    r"""Initialize an attention transformer.

    Args:
        in_dim (int): Input size.
        group_dim (int): Number of groups for features.
        virtual_batch_size (int): Batch size for Ghost Batch Normalization.
        momentum (float):
            Float value between 0 and 1,
            which will be used for momentum in batch norm.
        mask_type (str):
            Either "sparsemax" or "entmax", this is masking function to use.
    """

    def __init__(
        self,
        in_dim,
        group_dim,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, group_dim, bias=False)
        initialize_non_glu(self.fc, in_dim, group_dim)
        self.bn = GBN(
            group_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        # Use softmax instead of sparsemax
        x = F.softmax(x, dim=-1)
        return x


class FeatTransformer(torch.nn.Module):
    r"""Initialize a feature transformer.

    Args:
        in_dim (int): Input dimensonality.
        out_dim (int): Output dimensonality.
        shared_layers (:class:`torch.nn.ModuleList`):
            The shared block that should be common to every step
        n_glu_independent (int): Number of independent GLU layers
        virtual_batch_size (int): Batch size for Ghost Batch
            Normalization within GLU block(s)
        momentum (float): Float value between 0 and 1
            which will be used for momentum in batch norm
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        shared_layers,
        n_glu_independent,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super().__init__()

        params = {
            "n_glu": n_glu_independent,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum,
        }

        if shared_layers is None:
            # no shared layers
            self.shared = torch.nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(
                in_dim,
                out_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            is_first = False

        if n_glu_independent == 0:
            # no independent layers
            self.specifics = torch.nn.Identity()
        else:
            spec_in_dim = in_dim if is_first else out_dim
            self.specifics = GLU_Block(spec_in_dim, out_dim, first=is_first, **params)

    def forward(self, x):
        x = self.shared(x)
        x = self.specifics(x)
        return x


class GLU_Block(torch.nn.Module):
    """
    Independent GLU block, specific to each step
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        n_glu=2,
        first=False,
        shared_layers=None,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super().__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = torch.nn.ModuleList()

        params = {"virtual_batch_size": virtual_batch_size, "momentum": momentum}

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(in_dim, out_dim, fc=fc, **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(out_dim, out_dim, fc=fc, **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:  # first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x


class GLU_Layer(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        fc=None,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super().__init__()

        self.out_dim = out_dim
        if fc:
            self.fc = fc
        else:
            self.fc = torch.nn.Linear(in_dim, 2 * out_dim, bias=False)
        initialize_glu(self.fc, in_dim, 2 * out_dim)

        self.bn = GBN(
            2 * out_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.out_dim], torch.sigmoid(x[:, self.out_dim :]))
        return out


class TabNetEncoder(torch.nn.Module):
    r"""Defines main part of the TabNet network without embedding layers."""

    def __init__(
        self,
        in_dim,
        out_dim,
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        group_attention_matrix=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_multi_task = isinstance(out_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.initial_bn = torch.nn.BatchNorm1d(self.in_dim, momentum=0.01)
        self.group_attention_matrix = group_attention_matrix

        if self.group_attention_matrix is None:
            # no groups
            self.group_attention_matrix = torch.eye(self.in_dim)
            self.attention_dim = self.in_dim
        else:
            self.attention_dim = self.group_attention_matrix.shape[0]

        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(
                        torch.nn.Linear(self.in_dim, 2 * (n_d + n_a), bias=False)
                    )
                else:
                    shared_feat_transform.append(
                        torch.nn.Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)
                    )

        else:
            shared_feat_transform = None

        self.initial_splitter = FeatTransformer(
            self.in_dim,
            n_d + n_a,
            shared_feat_transform,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=momentum,
        )

        self.feat_transformers = torch.nn.ModuleList()
        self.att_transformers = torch.nn.ModuleList()

        for _ in range(n_steps):
            transformer = FeatTransformer(
                self.in_dim,
                n_d + n_a,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            attention = AttentiveTransformer(
                n_a,
                self.attention_dim,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

    def forward(self, x, prior=None):
        x = self.initial_bn(x)

        bs = x.shape[0]  # batch size
        if prior is None:
            prior = torch.ones((bs, self.attention_dim)).to(x.device)

        mask_loss = 0
        att = self.initial_splitter(x)[:, self.n_d :]
        steps_output = []
        for step in range(self.n_steps):
            mask = self.att_transformers[step](prior, att)
            mask_loss += torch.mean(
                torch.sum(torch.mul(mask, torch.log(mask + self.epsilon)), dim=1)
            )
            # update prior
            prior = torch.mul(self.gamma - mask, prior)
            # output
            M_feature_level = torch.matmul(
                mask, self.group_attention_matrix.to(mask.device)
            )
            masked_x = torch.mul(M_feature_level, x)
            out = self.feat_transformers[step](masked_x)
            d = torch.nn.ReLU()(out[:, : self.n_d])
            steps_output.append(d)
            # update attention
            att = out[:, self.n_d :]

        mask_loss /= self.n_steps
        return steps_output, mask_loss

    def forward_masks(self, x):
        x = self.initial_bn(x)
        bs = x.shape[0]  # batch size
        prior = torch.ones((bs, self.attention_dim)).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        att = self.initial_splitter(x)[:, self.n_d :]
        masks = {}

        for step in range(self.n_steps):
            mask = self.att_transformers[step](prior, att)
            M_feature_level = torch.matmul(mask, self.group_attention_matrix)
            masks[step] = M_feature_level
            # update prior
            prior = torch.mul(self.gamma - mask, prior)
            # output
            masked_x = torch.mul(M_feature_level, x)
            out = self.feat_transformers[step](masked_x)
            d = torch.nn.ReLU()(out[:, : self.n_d])
            # explain
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M_feature_level, step_importance.unsqueeze(dim=1))
            # update attention
            att = out[:, self.n_d :]

        return M_explain, masks


class TabNetNoEmbeddings(torch.nn.Module):
    r"""Defines main part of the TabNet network without embedding layers."""

    def __init__(
        self,
        in_dim,
        out_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        group_attention_matrix=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_multi_task = isinstance(out_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.initial_bn = torch.nn.BatchNorm1d(self.in_dim, momentum=0.01)

        self.encoder = TabNetEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            group_attention_matrix=group_attention_matrix,
        )

        if self.is_multi_task:
            self.multi_task_mappings = torch.nn.ModuleList()
            for task_dim in out_dim:
                task_mapping = torch.nn.Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = torch.nn.Linear(n_d, out_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, out_dim)

    def forward(self, x):
        res = 0
        steps_output, mask_loss = self.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        if self.is_multi_task:
            # Result will be in list format
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
        else:
            out = self.final_mapping(res)
        return out, mask_loss

    def forward_masks(self, x):
        return self.encoder.forward_masks(x)


class TabNet(torch.nn.Module):
    r"""The TabNet model introduced in the
    `"TabNet: Attentive Interpretable Tabular Learning"
    <https://arxiv.org/abs/1908.07442>`_ paper.

    Args:
        out_dim (int): Dimension of network output.
            1 for regression, 2 for binary classification etc...
        n_d (int):
            Dimension of the prediction  layer (usually between 4 and 64).
        n_a (int):
            intDimension of the attention  layer (usually between 4 and 64).
        n_steps (int):
            Number of successive steps in the network
            (usually between 3 and 10).
        gamma (int):
            Float above 1, scaling factor for attention updates
            (usually between 1.0 to 2.0).
        cat_emb_dim Union ([int, List[int]]):
            Size of the embedding of categorical features.
            if int, all categorical features will have same embedding size,
            if list of int, each corresponding feature will have specific size.
        n_independent (int):
            Number of independent GLU layer in each GLU block (default: :obj:`2`).
        n_shared (int):
            Number of independent GLU layer in each GLU block (default: :obj:`2`).
        epsilon (float):
            Avoid log(0), this should be kept very low.
        virtual_batch_size (int):
            Batch size for Ghost Batch Normalization.
        momentum float:
            Float value between 0 and 1,
            which will be used for momentum in all batch norm.
        mask_type (str):
            Either "sparsemax" or "entmax" : this is masking function to use.
        group_attention_matrix (Tensor):
            Matrix of size (n_groups, in_dim),
            m_ij = importance within group i of feature j
        metadata (Dict[ColType, List[Dict[str, Any]]]):
            Metadata for each column type, specifying the statistics and
            properties of the columns. (default: :obj:`None`).
    """

    def __init__(
        self,
        out_dim: int,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        cat_emb_dim: int = 1,
        num_emb_dim: int = 1,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        grouped_features: List[int] = [],
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.cat_emb_dim = cat_emb_dim
        self.num_emb_dim = cat_emb_dim
        self.out_dim = out_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.grouped_features = grouped_features
        self.metadata = metadata
        self.pre_encoder = None
        self.cat_idxs = [i for i in range(len(metadata[ColType.CATEGORICAL]))]
        self.in_dim = (
            len(self.metadata[ColType.CATEGORICAL])
            if ColType.NUMERICAL not in self.metadata.keys()
            else len(self.metadata[ColType.CATEGORICAL])
            + len(self.metadata[ColType.NUMERICAL])
        )
        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")
        self.virtual_batch_size = virtual_batch_size

        # Initialize group_matrix and emb_group_matrix
        group_attention_matrix = create_group_matrix(self.grouped_features, self.in_dim)
        self.post_embed_dim = (
            len(self.metadata[ColType.NUMERICAL]) * num_emb_dim
            + len(self.metadata[ColType.CATEGORICAL]) * cat_emb_dim
        )
        emb_group_matrix = create_emb_group_matrix(
            group_attention_matrix,
            self.cat_idxs,
            self.in_dim,
            self.cat_emb_dim,
            self.post_embed_dim,
        )

        if metadata:
            self.pre_encoder = FTTransformerPreEncoder(
                out_dim=self.cat_emb_dim,
                metadata=metadata,
            )

        # Initialize TabNet network
        self.tabnet = TabNetNoEmbeddings(
            self.post_embed_dim,
            out_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            n_independent,
            n_shared,
            epsilon,
            virtual_batch_size,
            momentum,
            emb_group_matrix,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.pre_encoder:
            self.pre_encoder.reset_parameters()

    def forward(self, x):
        if self.pre_encoder:
            x = self.pre_encoder(x).flatten(start_dim=1)
        return self.tabnet(x)

    def forward_masks(self, x):
        return self.tabnet.forward_masks(x)
