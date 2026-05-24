import pytest
import torch

from rllm.data import HeteroGraphData
from rllm.dataloader.sampler.data_type import (
    EdgeTypeStr,
    HeteroSamplerOutput,
    NodeSamplerInput,
    NumNeighbors,
)
from rllm.dataloader.sampler.hetero_sampler import HeteroSampler
from rllm.dataloader.sampler.utils import (
    _sample_uniform_without_replacement,
    convert_hdata_to_csc,
    hetero_neighbor_sample_cpu,
)


def make_temporal_hetero_graph():
    data = HeteroGraphData()
    data["user"].num_nodes = 3
    data["item"].num_nodes = 2
    data["user"].time = torch.tensor([10, 20, 30], dtype=torch.long)
    data["item"].time = torch.tensor([0, 0], dtype=torch.long)
    data["user", "click", "item"].edge_index = torch.tensor(
        [[0, 1, 2, 0], [0, 0, 0, 1]],
        dtype=torch.long,
    )
    return data


def test_node_sampler_input_cast_slice_and_cpu_conversion():
    input_data = NodeSamplerInput(
        input_id=torch.tensor([5, 6, 7]),
        node=torch.tensor([0, 1, 2]),
        time=torch.tensor([10, 20, 30]),
        input_type="user",
    )
    sliced = input_data[torch.tensor([0, 2])]
    from_tuple = NodeSamplerInput.castinit((None, torch.tensor([1]), None, "item"))

    assert sliced.input_id.tolist() == [5, 7]
    assert sliced.node.tolist() == [0, 2]
    assert sliced.time.tolist() == [10, 30]
    assert from_tuple.input_type == "item"


def test_edge_type_str_and_num_neighbors_validation():
    edge = EdgeTypeStr("user", "click", "item")
    default_edge = EdgeTypeStr("user", "item")
    from_tuple = EdgeTypeStr(("user", "click", "item"))

    assert str(edge) == "user__click__item"
    assert default_edge.to_tuple() == ("user", "to", "item")
    assert from_tuple.to_tuple() == ("user", "click", "item")

    neighbors = NumNeighbors({("user", "click", "item"): [2, 1]}, default=[3, 3])
    assert neighbors.get_values([("user", "click", "item"), ("item", "rev_click", "user")]) == {
        ("user", "click", "item"): [2, 1],
        ("item", "rev_click", "user"): [3, 3],
    }
    assert neighbors.get_mapped_values([("user", "click", "item")]) == {
        "user__click__item": [2, 1]
    }
    assert neighbors.num_hops == 2

    with pytest.raises(ValueError):
        EdgeTypeStr("bad__edge")
    with pytest.raises(ValueError):
        NumNeighbors([1], default=[1])
    with pytest.raises(ValueError):
        NumNeighbors({("missing", "rel", "type"): [1]}).get_values([("a", "b", "c")])
    with pytest.raises(ValueError):
        NumNeighbors({("a", "b", "c"): [1], ("c", "d", "e"): [1, 2]}).get_values()


def test_convert_hdata_to_csc_uses_destination_node_count():
    data = make_temporal_hetero_graph()

    colptr, row, perm = convert_hdata_to_csc(data)
    edge_type = ("user", "click", "item")

    assert colptr[edge_type].tolist() == [0, 3, 4]
    assert row[edge_type].tolist() == [0, 1, 2, 0]
    assert perm[edge_type].tolist() == [0, 1, 2, 3]


def test_sample_uniform_without_replacement_boundaries():
    torch.manual_seed(0)

    assert _sample_uniform_without_replacement(2, 5, -1, torch.device("cpu")) == [2, 3, 4]
    assert _sample_uniform_without_replacement(2, 5, 0, torch.device("cpu")) == []
    assert sorted(_sample_uniform_without_replacement(2, 6, 2, torch.device("cpu"))) in [
        [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]
    ]


def test_hetero_neighbor_sample_cpu_static_and_error_paths():
    edge_type = ("user", "click", "item")
    rowptr = {edge_type: torch.tensor([0, 2, 3])}
    col = {edge_type: torch.tensor([1, 2, 0])}
    seed = {"user": torch.tensor([0])}
    num_neighbors = {edge_type: [2]}

    row, col_out, node, batch, edge, sampled_nodes, sampled_edges = hetero_neighbor_sample_cpu(
        rowptr,
        col,
        seed,
        num_neighbors,
    )

    assert node["user"].tolist() == [0]
    assert set(node["item"].tolist()) == {1, 2}
    assert row[edge_type].tolist() == [0, 0]
    assert col_out[edge_type].tolist() == [0, 1]
    assert edge[edge_type].tolist() == [0, 1]
    assert sampled_nodes["item"] == [0, 2]
    assert sampled_edges[edge_type] == [2]

    with pytest.raises(ValueError):
        hetero_neighbor_sample_cpu(rowptr, col, seed, num_neighbors, temporal_strategy="last")
    with pytest.raises(ValueError):
        hetero_neighbor_sample_cpu(
            rowptr,
            col,
            seed,
            num_neighbors,
            node_time_dict={"item": torch.tensor([1, 2, 3])},
            edge_time_dict={edge_type: torch.tensor([1, 2, 3])},
        )
    with pytest.raises(KeyError):
        hetero_neighbor_sample_cpu(rowptr, {}, seed, num_neighbors)
    with pytest.raises(KeyError):
        hetero_neighbor_sample_cpu(rowptr, col, seed, {})


def test_hetero_sampler_reverse_temporal_sampling():
    data = make_temporal_hetero_graph()
    sampler = HeteroSampler(
        hdata=data,
        num_neighbors=[3],
        time_attr="time",
        temporal_strategy="uniform",
        csc=True,
        use_pyg_lib=False,
    )
    input_data = NodeSamplerInput(
        input_id=torch.tensor([99]),
        node=torch.tensor([0]),
        time=torch.tensor([25]),
        input_type="item",
    )

    out = sampler.sample_neighbors(input_data)

    assert out.metadata[0].tolist() == [99]
    assert out.metadata[1].tolist() == [25]
    assert set(out.node["user"].tolist()) == {0, 1}
    assert 2 not in out.node["user"].tolist()
    assert out.num_sampled_edges[("user", "click", "item")] == [2]


def test_hetero_sampler_requires_time_attr_for_uniform_strategy():
    with pytest.raises(ValueError):
        HeteroSampler(make_temporal_hetero_graph(), num_neighbors=[1], time_attr=None)

    with pytest.raises(AssertionError):
        HeteroSampler(make_temporal_hetero_graph(), num_neighbors=[1], time_attr="time", device=torch.device("cuda"))


def test_hetero_sampler_output_bidirectional_and_keep_original():
    output = HeteroSamplerOutput(
        node={"n": torch.tensor([0, 1])},
        row={("n", "rel", "n"): torch.tensor([0])},
        col={("n", "rel", "n"): torch.tensor([1])},
    )

    bidir = output.to_bidirectional(keep_org=True)

    assert sorted(zip(bidir.row[("n", "rel", "n")].tolist(), bidir.col[("n", "rel", "n")].tolist())) == [(0, 1), (1, 0)]
    assert bidir.original_row[("n", "rel", "n")].tolist() == [0]

