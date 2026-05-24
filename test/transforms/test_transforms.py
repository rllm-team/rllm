import copy

import pytest
import torch

from rllm.data.graph_data import GraphData
from rllm.transforms.graph_transforms import (
    AddRemainingSelfLoops,
    GCNNorm,
    GCNTransform,
    GraphTransform,
    KNNGraph,
    NormalizeFeatures,
    RECTTransform,
    RemoveSelfLoops,
    SVDFeatureReduction,
)
from rllm.transforms.graph_transforms.functional import (
    add_remaining_self_loops,
    knn_graph,
    normalize_features,
    remove_self_loops,
    svd_feature_reduction,
    symmetric_norm,
)
from rllm.transforms.graph_transforms.gdc import GDC
from rllm.transforms.table_transforms import (
    ColNormalize,
    DefaultTableTransform,
    OneHotTransform,
    StackNumerical,
    TabTransformerTransform,
)
from rllm.transforms.table_transforms.col_transform import ColTransform
from rllm.transforms.table_transforms.table_transform import TableTransform, _get_na_mask
from rllm.transforms.utils import BaseTransform, RemoveTrainingClasses
from rllm.transforms.utils.functional import remove_training_classes
from rllm.types import ColType, NAMode, StatType


class DummyTableData:
    def __init__(self, feat_dict, metadata):
        self.feat_dict = feat_dict
        self.metadata = metadata

    def get_feat_dict(self):
        return self.feat_dict


class AddOneTransform(BaseTransform):
    def forward(self, data):
        data.value += 1
        return data


class DoubleColumn(ColTransform):
    def forward(self, data):
        data.value *= 2
        return data


class MinimalTableTransform(TableTransform):
    def reset_parameters(self):
        super().reset_parameters()


def sparse_adj(indices, values, size):
    return torch.sparse_coo_tensor(
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(values, dtype=torch.float32),
        size,
    ).coalesce()


def make_table_data():
    feat_dict = {
        ColType.NUMERICAL: torch.tensor([[1.0, float("nan")], [3.0, 4.0]]),
        ColType.CATEGORICAL: torch.tensor([[0, -1], [1, 2]], dtype=torch.long),
    }
    metadata = {
        ColType.NUMERICAL: [
            {StatType.MEAN: 2.0, StatType.STD: 1.0},
            {StatType.MEAN: 5.0, StatType.STD: 2.0},
        ],
        ColType.CATEGORICAL: [
            {StatType.COUNT: 3, StatType.MOST_FREQUENT: 1},
            {StatType.COUNT: 4, StatType.MOST_FREQUENT: 2},
        ],
    }
    return DummyTableData(feat_dict, metadata)


def test_add_and_remove_self_loops_dense_and_sparse():
    dense = torch.tensor([[9.0, 1.0], [2.0, 8.0]])
    sparse = sparse_adj([[0, 0, 1], [0, 1, 1]], [9.0, 1.0, 8.0], (2, 2))

    dense_added = add_remaining_self_loops(dense, fill_value=5.0)
    dense_removed = remove_self_loops(dense)
    sparse_added = add_remaining_self_loops(sparse, fill_value=5.0).coalesce()
    sparse_removed = remove_self_loops(sparse).coalesce()

    assert dense_added.tolist() == [[5.0, 1.0], [2.0, 5.0]]
    assert dense_removed.tolist() == [[0.0, 1.0], [2.0, 0.0]]
    assert sparse_added.to_dense().tolist() == [[5.0, 1.0], [0.0, 5.0]]
    assert sparse_removed.to_dense().tolist() == [[0.0, 1.0], [0.0, 0.0]]


def test_edge_transform_wrappers_apply_to_tensor_and_graphdata():
    adj = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    data = GraphData(adj=adj)

    no_loops = RemoveSelfLoops()(data)
    with_loops = AddRemainingSelfLoops(fill_value=7.0)(adj)

    assert no_loops is not data
    assert no_loops.adj.tolist() == [[0.0, 2.0], [3.0, 0.0]]
    assert with_loops.tolist() == [[7.0, 2.0], [3.0, 7.0]]
    assert repr(RemoveSelfLoops()) == "RemoveSelfLoops()"


def test_normalize_features_modes_and_wrapper():
    x = torch.tensor([[3.0, 4.0], [1.0, 1.0], [0.0, 0.0]])

    l2, l2_norm = normalize_features(x.clone(), "l2", return_norm=True)
    l1 = normalize_features(x.clone(), "l1")
    summed = normalize_features(torch.tensor([[1.0, 2.0], [3.0, 5.0]]), "sum")
    wrapped = NormalizeFeatures("l1")(GraphData(x=x.clone()))

    assert torch.allclose(l2[0], torch.tensor([0.6, 0.8]))
    assert l2_norm.tolist() == [5.0, pytest.approx(2 ** 0.5), 0.0]
    assert l1.tolist()[0] == [pytest.approx(3 / 7), pytest.approx(4 / 7)]
    assert summed.tolist() == [[0.0, 1.0], [pytest.approx(3 / 7), pytest.approx(5 / 7)]]
    assert wrapped.x.tolist()[0] == [pytest.approx(3 / 7), pytest.approx(4 / 7)]

    with pytest.raises(ValueError):
        normalize_features(x.clone(), "bad")


def test_svd_feature_reduction_function_and_wrapper():
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])

    reduced = svd_feature_reduction(x, out_dim=2)
    unchanged = svd_feature_reduction(x, out_dim=4)
    wrapped = SVDFeatureReduction(2)(GraphData(x=x))

    assert reduced.shape == (3, 2)
    assert unchanged.shape == (3, 3)
    assert wrapped.x.shape == (3, 2)


def test_symmetric_norm_gcn_norm_and_pipeline_transforms():
    adj = sparse_adj([[0, 1], [1, 0]], [1.0, 1.0], (2, 2))

    normed = symmetric_norm(adj).to_dense()
    gcn_normed = GCNNorm()(adj).to_dense()
    gcn_data = GraphData(x=torch.tensor([[1.0, 1.0], [2.0, 0.0]]), adj=adj)
    rect_data = GraphData(x=torch.tensor([[1.0, 1.0], [2.0, 0.0]]), adj=adj)
    transformed = GCNTransform(normalize_features="l1")(gcn_data)
    rected = RECTTransform(normalize_features="l1", svd_out_dim=1, use_gdc=False)(rect_data)

    assert torch.allclose(normed, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert torch.allclose(gcn_normed, torch.full((2, 2), 0.5))
    assert transformed.x.tolist() == [[0.5, 0.5], [1.0, 0.0]]
    assert transformed.adj.shape == (2, 2)
    assert rected.x.shape == (2, 1)


def test_knn_graph_function_and_wrapper():
    x = torch.tensor([[0.0], [1.0], [4.0]])

    adj = knn_graph(x, num_neighbors=1, include_self=False)
    wrapped = KNNGraph(num_neighbors=1, include_self=True).forward(x)

    assert adj.shape == (3, 3)
    assert adj._nnz() == 3
    assert wrapped.to_dense().diag().tolist() == [1.0, 1.0, 1.0]


def test_knn_graph_call_accepts_feature_matrix():
    x = torch.tensor([[0.0], [1.0], [4.0]])

    adj = KNNGraph(num_neighbors=1, include_self=False)(x)

    assert adj.shape == (3, 3)


def test_graph_transform_composes_single_and_list_inputs():
    transform = GraphTransform([NormalizeFeatures("l1")])
    data = GraphData(x=torch.tensor([[1.0, 1.0]]))
    data_list = [GraphData(x=torch.tensor([[2.0, 2.0]]))]

    assert transform(data).x.tolist() == [[0.5, 0.5]]
    assert transform(data_list)[0].x.tolist() == [[0.5, 0.5]]


def test_gdc_components_and_error_paths():
    adj = sparse_adj([[0, 1], [1, 0]], [1.0, 1.0], (2, 2))
    gdc = GDC(
        self_loop_weight=1.0,
        diffusion={"method": "ppr", "alpha": 0.2},
        sparsification={"method": "threshold", "eps": 0.0},
    )

    out = gdc(adj)
    heat = gdc.diffusion_matrix(adj.to_dense(), method="heat", t=1.0)
    topk = gdc.sparsify_matrix(torch.tensor([[0.1, 0.9], [0.8, 0.2]]), method="topk", k=1, dim=1)

    assert out.shape == (2, 2)
    assert heat.shape == (2, 2)
    assert topk.is_sparse

    with pytest.raises(ValueError):
        gdc.get_transition_matrix(adj, "bad")
    with pytest.raises(ValueError):
        gdc.diffusion_matrix(adj, method="bad")
    with pytest.raises(ValueError):
        gdc.sparsify_matrix(torch.ones(2, 2), method="topk", k=1, dim=2)
    with pytest.raises(ValueError):
        gdc.sparsify_matrix(torch.ones(2, 2), method="bad")


def test_gdc_forward_supports_heat_diffusion_configuration():
    adj = sparse_adj([[0, 1], [1, 0]], [1.0, 1.0], (2, 2))
    gdc = GDC(
        self_loop_weight=1.0,
        diffusion={"method": "heat", "t": 1.0},
        sparsification={"method": "threshold", "eps": 0.0},
    )

    out = gdc(adj)

    assert out.shape == (2, 2)


def test_na_mask_and_table_transform_missing_value_handling():
    table = make_table_data()
    transform = DefaultTableTransform()

    transformed = transform(table)

    assert _get_na_mask(torch.tensor([[1.0, float("nan")]])).tolist() == [[False, True]]
    assert _get_na_mask(torch.tensor([[1, -1]])).tolist() == [[False, True]]
    assert transformed.feat_dict[ColType.NUMERICAL].tolist() == [[1.0, 5.0], [3.0, 4.0]]
    assert transformed.feat_dict[ColType.CATEGORICAL].tolist() == [[0, 2], [1, 2]]

    with pytest.raises(ValueError):
        MinimalTableTransform(col_type=ColType.NUMERICAL, na_mode=NAMode.MOST_FREQUENT)


def test_col_transforms_normalize_one_hot_and_stack():
    table = make_table_data()
    table.feat_dict[ColType.NUMERICAL] = torch.tensor([[1.0, 7.0], [3.0, 9.0]])
    table.feat_dict[ColType.CATEGORICAL] = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)

    normalized = ColNormalize()(table)
    one_hot = OneHotTransform(out_dim=5)(table)
    stacked = StackNumerical(out_dim=3)(table)

    assert normalized is not table
    assert normalized.feat_dict[ColType.NUMERICAL].shape == (2, 2, 3)
    assert one_hot.feat_dict[ColType.CATEGORICAL].shape == (2, 2, 5)
    assert stacked.feat_dict[ColType.NUMERICAL].shape == (2, 2, 3)
    assert repr(StackNumerical(3)) == "StackNumerical()"


def test_tab_transformer_transform_current_reset_behavior_is_detected():
    transform = TabTransformerTransform(out_dim=2)

    with pytest.raises(AttributeError):
        transform.reset_parameters()


def test_base_and_col_transform_are_shallow_copy_wrappers():
    obj = type("Obj", (), {"value": 1})()

    base_out = AddOneTransform()(obj)
    col_out = DoubleColumn()(obj)

    assert base_out is not obj
    assert col_out is not obj
    assert base_out.value == 2
    assert col_out.value == 2
    assert obj.value == 1
    assert repr(AddOneTransform()) == "AddOneTransform()"


def test_remove_training_classes_function_and_wrapper():
    mask = torch.tensor([True, True, True, False])
    labels = torch.tensor([0, 1, 2, 1])
    data = GraphData(y=labels, train_mask=mask)

    filtered = remove_training_classes(mask, labels, classes=[1])
    transformed = RemoveTrainingClasses(classes=[1])(data)

    assert filtered.tolist() == [True, False, True, False]
    assert transformed.train_mask.tolist() == [True, False, True, False]
    assert data.train_mask.tolist() == [True, True, True, False]

    with pytest.raises(ValueError):
        RemoveTrainingClasses(classes=[1])(GraphData(y=labels))
