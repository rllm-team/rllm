from typing import Any, Dict, Tuple, Callable, List, Union
import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

from rllm.data.graph_data import GraphData, HeteroGraphData
from rllm.transforms.base_transform import BaseTransform
from rllm.utils.sparse import (
    is_torch_sparse_tensor,
    sparse_mx_to_torch_sparse_tensor
)


class GDC(BaseTransform):
    r"""Processes the graph via Graph Diffusion Convolution (GDC) from the
    `"Diffusion Improves Graph Learning" <https://arxiv.org/abs/1911.05485>`_
    paper (functional name: :obj:`gdc`).

    .. note::

        The paper offers additional advice on how to choose the
        hyperparameters.
        For an example of using GCN with GDC, see `examples/gcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        gcn.py>`_.

    Args:
        self_loop_weight (float, optional): Weight of the added self-loop.
            Set to :obj:`None` to add no self-loops. (default: :obj:`1`)
        normalization_in (str, optional): Normalization of the transition
            matrix on the original (input) graph. Possible values:
            :obj:`"sym"`, :obj:`"col"`, and :obj:`"row"`.
            See :func:`GDC.get_transition_matrix` for details.
            (default: :obj:`"sym"`)
        normalization_out (str, optional): Normalization of the transition
            matrix on the transformed GDC (output) graph. Possible values:
            :obj:`"sym"`, :obj:`"col"`, :obj:`"row"`, and :obj:`None`.
            See :func:`GDC.get_transition_matrix` for details.
            (default: :obj:`"col"`)
        diffusion_kwargs (dict, optional): Dictionary containing the parameters
            for diffusion.
            `method` specifies the diffusion method (:obj:`"ppr"`,
            :obj:`"heat"` or :obj:`"coeff"`).
            Each diffusion method requires different additional parameters.
            See :func:`GDC.diffusion_matrix_exact` or
            :func:`GDC.diffusion_matrix_approx` for details.
            (default: :obj:`dict(method='ppr', alpha=0.15)`)
        sparsification_kwargs (dict, optional): Dictionary containing the
            parameters for sparsification.
            `method` specifies the sparsification method (:obj:`"threshold"` or
            :obj:`"topk"`).
            Each sparsification method requires different additional
            parameters.
            See :func:`GDC.sparsify_dense` for details.
            (default: :obj:`dict(method='threshold', avg_degree=64)`)
        exact (bool, optional): Whether to exactly calculate the diffusion
            matrix.
            Note that the exact variants are not scalable.
            They densify the adjacency matrix and calculate either its inverse
            or its matrix exponential.
            However, the approximate variants do not support edge weights and
            currently only personalized PageRank and sparsification by
            threshold are implemented as fast, approximate versions.
            (default: :obj:`True`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    def __init__(
        self,
        self_loop_weight: float = 1.,
        normalize_in: str = 'sym',
        normalize_out: str = 'col',
        diffusion: Dict[str, Any] = dict(method='ppr', alpha=0.15),
        sparsification: Dict[str, Any] = dict(
            method='threshold',
            avg_degree=64,
        ),
    ) -> None:
        super().__init__()
        self.self_loop_weight = self_loop_weight
        self.normalize_in = normalize_in
        self.normalize_out = normalize_out
        self.diffusion = diffusion
        self.sparsification = sparsification

    @torch.no_grad()
    def forward(self, data: Union[GraphData, HeteroGraphData]):
        adj = None
        if isinstance(data, GraphData):
            assert data.adj is not None
            adj = data.adj
        elif isinstance(data, HeteroGraphData):
            if 'adj' in data:
                adj = data.adj
        if self.self_loop_weight:
            adj = self.add_weighted_self_loop(adj, self.self_loop_weight)

        # 1.get the transition matrix
        trans_matrix = self.get_transition_matrix(adj, self.normalize_in)

        # 2.Sum over T^k, generalized graph diffusion
        diff_matrix = self.diffusion_matrix(
            trans_matrix, **self.diffusion)

        # 3.sparsify the graph
        diff_matrix_sparsified = self.sparsify_matrix(
            diff_matrix, **self.sparsification)

        # 4.get transition matrix on ~s
        data.adj = self.get_transition_matrix(
            diff_matrix_sparsified.to_sparse_coo(), self.normalize_out)

        return data

    def add_weighted_self_loop(
        self,
        adj: Tensor,
        weight: float = 1.,
    ) -> Tensor:
        adj = torch.eye(adj.size(0)) * weight + adj
        indices = torch.nonzero(adj, as_tuple=False)
        values = adj[indices[:, 0], indices[:, 1]]
        sparse_tensor = torch.sparse_coo_tensor(
            indices.t(), values, adj.size())
        return sparse_tensor

    def get_transition_matrix(
        self,
        adj: Tensor,
        normalize: str,
    ) -> Tensor:
        r"""Calculate the approximate, sparse diffusion on a given sparse
        matrix.

        Args:
            adj (Tensor): The adjacency matrix.
            normalization (str): Normalization scheme:

                1. :obj:`"sym"`: Symmetric normalization
                   :math:`\mathbf{T} = \mathbf{D}^{-1/2} \mathbf{A}
                   \mathbf{D}^{-1/2}`.
                2. :obj:`"col"`: Column-wise normalization
                   :math:`\mathbf{T} = \mathbf{A} \mathbf{D}^{-1}`.
                3. :obj:`"row"`: Row-wise normalization
                   :math:`\mathbf{T} = \mathbf{D}^{-1} \mathbf{A}`.
                4. :obj:`None`: No normalization.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        adj = adj.coalesce()
        indices = adj.indices()

        # D
        adj_data = adj.values()
        adj_sp = sp.csr_matrix(
            (adj_data, (indices[0], indices[1])),
            shape=adj.shape
        )

        if normalize == 'sym':
            # D-1/2
            deg = np.array(adj_sp.sum(axis=1)).flatten()
            deg[deg == 0] = 1e-10
            deg_sqrt_inv = np.power(deg, -0.5)
            deg_sqrt_inv[deg_sqrt_inv == 100000.] = 0.0
            deg_sqrt_inv = sp.diags(deg_sqrt_inv)
            adj = sp.coo_matrix(deg_sqrt_inv * adj_sp * deg_sqrt_inv)

        elif normalize == 'col':
            deg = np.array(adj_sp.sum(axis=1)).flatten()
            deg[deg == 0] = 1e-10
            deg_sqrt_inv = np.power(deg, -1)
            deg_sqrt_inv[deg_sqrt_inv == float(1e10)] = 0.0
            deg_sqrt_inv = sp.diags(deg_sqrt_inv)
            adj = sp.coo_matrix(adj_sp * deg_sqrt_inv)

        elif normalize == 'row':
            deg = np.array(adj_sp.sum(axis=0)).flatten()
            deg[deg == 0] = 1e-10
            deg_sqrt_inv = np.power(deg, -1)
            deg_sqrt_inv[deg_sqrt_inv == float(1e10)] = 0.0
            deg_sqrt_inv = sp.diags(deg_sqrt_inv)
            adj = sp.coo_matrix(deg_sqrt_inv * adj_sp)

        elif normalize is None:
            pass
        else:
            pass
        return sparse_mx_to_torch_sparse_tensor(adj)

    def diffusion_matrix(  # noqa: D417
        self,
        adj: Tensor,
        method: str,
        **kwargs: Any,
    ) -> Tensor:
        r"""Calculate the (dense) diffusion on a given sparse graph.
        Note that these exact variants are not scalable. They densify the
        adjacency matrix and calculate either its inverse or its matrix
        exponential.

        Args:
            adj (Tensor): The adjacency matrix.
            num_nodes (int): Number of nodes.
            method (str): Diffusion method:

                1. :obj:`"ppr"`: Use personalized PageRank as diffusion.
                   Additionally expects the parameter:

                   - **alpha** (*float*) - Return probability in PPR.
                     Commonly lies in :obj:`[0.05, 0.2]`.

                2. :obj:`"heat"`: Use heat kernel diffusion.
                   Additionally expects the parameter:

                   - **t** (*float*) - Time of diffusion. Commonly lies in
                     :obj:`[2, 10]`.

                3. :obj:`"coeff"`: Freely choose diffusion coefficients.
                   Additionally expects the parameter:

                   - **coeffs** (*List[float]*) - List of coefficients
                     :obj:`theta_k` for each power of the transition matrix
                     (starting at :obj:`0`).

        :rtype: (:class:`Tensor`)
        """
        if method == 'ppr':
            # α (I_n + (α - 1) A)^-1
            diff_matrix = (kwargs['alpha'] - 1) * adj
            diff_matrix = self.add_weighted_self_loop(diff_matrix).to_dense()
            diff_matrix = kwargs['alpha'] * torch.inverse(diff_matrix)

        elif method == 'heat':
            # exp(t (A - I_n))
            diff_matrix = self.add_weighted_self_loop(adj, -1)
            diff_matrix = kwargs['t'] * diff_matrix
            diff_matrix = diff_matrix.exp()

        elif method == 'coeff':
            mat = torch.eye(adj.shape(0), device=adj.device)

            diff_matrix = kwargs['coeffs'][0] * mat
            for coeff in kwargs['coeffs'][1:]:
                mat = mat @ adj
                diff_matrix += coeff * mat
        else:
            raise ValueError(f"Exact GDC diffusion '{method}' unknown")

        return diff_matrix

    def sparsify_matrix(  # noqa: D417
        self,
        mx: Tensor,
        method: str,
        **kwargs: Any,
    ) -> Tensor:
        r"""Sparsifies a given sparse graph further.

        Args:
            adj (Tensor): The adjacency matrix.
            num_nodes (int): Number of nodes.
            method (str): Method of sparsification:

                1. :obj:`"threshold"`: Remove all edges with weights smaller
                   than :obj:`eps`.
                   Additionally expects one of these parameters:

                   - **eps** (*float*) - Threshold to bound edges at.

                   - **avg_degree** (*int*) - If :obj:`eps` is not given,
                     it can optionally be calculated by calculating the
                     :obj:`eps` required to achieve a given :obj:`avg_degree`.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        if method == 'threshold':
            if 'eps' not in kwargs.keys():
                kwargs['eps'] = self.__calculate_eps__(
                    mx,
                    kwargs['avg_degree'],
                )
            mx[mx < kwargs['eps']] = 0
        elif method == 'topk':
            k, dim = min(mx.size(0), kwargs['k']), kwargs['dim']
            assert dim in [0, 1]
            sort_idx = torch.argsort(mx, dim=dim, descending=True)
            if dim == 0:
                top_idx = sort_idx[:k]
                edge_weight = torch.gather(mx, dim=dim,
                                           index=top_idx).flatten()
                row_idx = torch.arange(
                    mx.size(0), device=mx.device).repeat(k)
                mx = torch.sparse_coo_tensor(
                    torch.stack(
                        [row_idx, top_idx.flatten()]),
                    edge_weight,
                    mx.size())
            else:
                top_idx = sort_idx[:, :k]
                edge_weight = torch.gather(mx, dim=dim,
                                           index=top_idx).flatten()
                col_idx = torch.arange(
                    mx.size(0), device=mx.device).repeat(k)
                mx = torch.sparse_coo_tensor(
                    torch.stack(
                        [top_idx, col_idx.flatten()]),
                    edge_weight,
                    mx.size())
        else:
            raise ValueError(f"GDC sparsification '{method}' unknown")

        return mx

    def __calculate_eps__(
        self,
        adj: Tensor,
        avg_degree: int,
    ) -> float:
        r"""Calculates threshold necessary to achieve a given average degree.

        Args:
            adj (Tensor): Adjacency matrix.
            avg_degree (int): Target average degree.

        :rtype: (:class:`float`)
        """
        edge_weights = adj.flatten()
        sorted_edges = torch.sort(
            edge_weights.flatten(), descending=True).values
        if avg_degree * adj.size(0) > len(sorted_edges):
            return -np.inf

        left = sorted_edges[avg_degree * adj.size(0) - 1]
        right = sorted_edges[avg_degree * adj.size(0)]
        return float(left + right) / 2.0
