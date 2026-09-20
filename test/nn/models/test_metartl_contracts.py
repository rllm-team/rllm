from __future__ import annotations

import torch

from rllm.data import HeteroGraphData
from rllm.nn.models.metartl import MetaPathFusion, MetaPathProp


def test_metapath_propagation_preserves_base_and_relation_features_in_stable_order():
    graph = HeteroGraphData()
    graph["source"].num_nodes = 2
    graph["target"].num_nodes = 2
    relation = ("source", "rates", "target")
    graph[relation].edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # RelBenchLoader attaches this batch-level mapping for relational models.
    graph.edge_index_dict = {relation: graph[relation].edge_index}
    features = {
        "source": torch.tensor([[10.0], [20.0]]),
        "target": torch.tensor([[1.0], [2.0]]),
    }
    propagate = MetaPathProp(
        target_node_type="target", min_hops=0, max_hops=1, edge_schema=[relation]
    )

    output = propagate(graph, features)

    assert output.shape == (2, 2, 1)
    assert torch.equal(output[:, 0], features["target"])
    assert torch.equal(output[:, 1], torch.tensor([[20.0], [10.0]]))


def test_metapath_fusion_has_declared_shape_and_finite_backward_pass():
    torch.manual_seed(13)
    model = MetaPathFusion(
        num_nodes=5,
        in_dim=4,
        hidden_dim=4,
        num_metapaths=3,
        out_dim=2,
        num_centroids=4,
        num_heads=2,
        att_drop=0.0,
        readout_drop=0.0,
        num_proj_layers=1,
    )
    model.eval()
    features = torch.randn(2, 3, 4, requires_grad=True)

    output = model(features, torch.tensor([0, 3]))

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
