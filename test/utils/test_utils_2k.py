import io
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from rllm.types import ColType
from rllm.utils._mixin import CastMixin
from rllm.utils._remap import remap_keys
from rllm.utils._sort import lexsort
from rllm.utils.atomic_routes import get_atomic_routes
from rllm.utils.col_process import timecol_to_unix_time
from rllm.utils.csv_utils import clean_df_by_col_types, read_csv_with_fallback_encodings
from rllm.utils.download import download_google_url, download_url
from rllm.utils.extract import extract_zip
from rllm.utils.graph_utils import (
    _to_csc,
    adj_to_edge_index,
    gcn_norm,
    index_to_ptr,
    remove_self_loops,
    sort_edge_index,
    to_bidirectional,
)
from rllm.utils.seg_reduce import seg_softmax, seg_softmax_, seg_sum
from rllm.utils.sparse import (
    get_indices,
    is_torch_sparse_tensor,
    set_values,
    sparse_mx_to_torch_sparse_tensor,
)
from rllm.utils.type_infer import TypeInferencer
from rllm.utils.undirected import is_undirected, to_undirected


@dataclass
class ExampleCast(CastMixin):
    left: int
    right: str = "x"


class FakeUrlResponse:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def info(self):
        return {"Content-Length": str(sum(len(chunk) for chunk in self.chunks))}

    def read(self, _size):
        if not self.chunks:
            return b""
        return self.chunks.pop(0)


def test_sparse_matrix_conversion_indices_and_values():
    scipy_adj = sp.coo_matrix(([1.0, 2.0], ([0, 2], [1, 0])), shape=(3, 3))

    torch_adj = sparse_mx_to_torch_sparse_tensor(scipy_adj).coalesce()
    indices = get_indices(torch_adj)
    replaced = set_values(torch_adj, torch.tensor([10.0, 20.0]))

    assert is_torch_sparse_tensor(torch_adj)
    assert not is_torch_sparse_tensor(torch.ones(3, 3))
    assert indices.tolist() == [[0, 2], [1, 0]]
    assert replaced.coalesce().values().tolist() == [10.0, 20.0]


def test_sparse_dense_csr_csc_branches():
    dense = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    csr = dense.to_sparse_csr()
    csc = dense.to_sparse_csc()

    assert get_indices(dense).tolist() == [[0, 1], [1, 0]]
    assert set_values(csr, torch.tensor([5.0, 6.0])).to_dense().tolist() == [[0.0, 5.0], [6.0, 0.0]]
    assert set_values(csc, torch.tensor([7.0, 8.0])).to_dense().tolist() == [[0.0, 8.0], [7.0, 0.0]]

    with pytest.raises(ValueError):
        set_values(dense, torch.tensor([1.0, 2.0]))


def test_adj_to_edge_index_returns_weights_only_when_needed():
    weighted = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([1.0, 2.0]),
        (2, 2),
    ).coalesce()
    unweighted = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]),
        torch.ones(2),
        (2, 2),
    ).coalesce()

    weighted_index, weighted_attr = adj_to_edge_index(weighted)
    _, unweighted_attr = adj_to_edge_index(unweighted)

    assert weighted_index.tolist() == [[0, 1], [1, 0]]
    assert weighted_attr.tolist() == [1.0, 2.0]
    assert unweighted_attr is None


def test_graph_sort_index_to_ptr_and_csc_are_consistent():
    edge_index = torch.tensor([[2, 0, 1, 1], [1, 1, 0, 2]])

    sorted_by_row = sort_edge_index(edge_index, sort_by_row=True)
    sorted_by_col = sort_edge_index(edge_index, sort_by_row=False)
    ptr = index_to_ptr(torch.tensor([0, 1, 1, 2]), num_nodes=3)
    col_ptr, row, perm = _to_csc(edge_index, num_nodes=3)

    assert sorted_by_row.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert sorted_by_col.tolist() == [[1, 0, 2, 1], [0, 1, 1, 2]]
    assert ptr.tolist() == [0, 1, 3, 4]
    assert col_ptr.tolist() == [0, 1, 3, 4]
    assert row.tolist() == [1, 2, 0, 1]
    assert perm.tolist() == [2, 0, 1, 3]


def test_to_csc_accepts_sparse_input_and_empty_index():
    empty_ptr = index_to_ptr(torch.tensor([], dtype=torch.long), num_nodes=2)
    assert empty_ptr.tolist() == [0, 0, 0]

    adj = torch.sparse_coo_tensor(
        torch.tensor([[0, 2], [1, 0]]),
        torch.tensor([1.0, 2.0]),
        (3, 3),
    ).coalesce()
    col_ptr, row, perm = _to_csc(adj, num_nodes=3)

    assert col_ptr.tolist() == [0, 1, 2, 2]
    assert row.tolist() == [2, 0]
    assert perm is None


def test_remove_self_loops_accepts_sparse_adjacency():
    adj = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1, 2], [0, 1, 1, 0]]),
        torch.tensor([9.0, 1.0, 8.0, 2.0]),
        (3, 3),
    ).coalesce()

    cleaned = remove_self_loops(adj).coalesce()

    assert cleaned.indices().tolist() == [[0, 2], [1, 0]]
    assert cleaned.values().tolist() == [1.0, 2.0]


def test_remove_self_loops_accepts_dense_adjacency():
    dense = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    cleaned = remove_self_loops(dense)

    assert cleaned.tolist() == [[0.0, 2.0], [3.0, 0.0]]
    assert dense.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_gcn_norm_matches_two_node_reference_with_self_loops():
    adj = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]),
        torch.ones(2),
        (2, 2),
    ).coalesce()

    dense = gcn_norm(adj).to_dense()

    assert torch.allclose(dense, torch.full((2, 2), 0.5))


def test_to_bidirectional_deduplicates_forward_and_reverse_edges():
    row = torch.tensor([0, 1])
    col = torch.tensor([1, 0])
    rev_row = col
    rev_col = row

    bi_row, bi_col = to_bidirectional(row, col, rev_row, rev_col)
    edges = sorted(zip(bi_row.tolist(), bi_col.tolist()))

    assert edges == [(0, 1), (1, 0)]


def test_segment_reductions_match_manual_reference():
    data = torch.tensor([[1.0, 2.0], [3.0, 0.0], [2.0, -2.0], [4.0, 4.0]])
    segment_ids = torch.tensor([0, 0, 1, 1])

    summed = seg_sum(data, segment_ids, num_segments=3)
    softmax_fast = seg_softmax(data, segment_ids, num_segs=3)
    softmax_slow = seg_softmax_(data, segment_ids, num_segs=3)
    expected_softmax = torch.vstack(
        [
            torch.softmax(data[:2], dim=0),
            torch.softmax(data[2:], dim=0),
        ]
    )

    assert summed.tolist() == [[4.0, 2.0], [6.0, 2.0], [0.0, 0.0]]
    assert torch.allclose(softmax_fast, expected_softmax)
    assert torch.allclose(softmax_slow, expected_softmax)


def test_remap_keys_and_lexsort_have_deterministic_order():
    remapped = remap_keys({"a": 1, "b": 2, "c": 3}, {"a": "A", "b": "B"}, exclude=["b"])
    key_a = torch.tensor([1, 5, 1, 4, 3, 4, 4])
    key_b = torch.tensor([9, 4, 0, 4, 0, 2, 1])

    assert remapped == {"A": 1, "b": 2, "c": 3}
    assert lexsort([key_b, key_a]).tolist() == [2, 0, 4, 6, 5, 3, 1]


def test_cast_mixin_and_atomic_routes():
    from_tuple = ExampleCast.castinit((1, "a"))
    from_dict = ExampleCast.castinit({"left": 2, "right": "b"})
    existing = ExampleCast(3, "c")

    assert list(from_tuple) == [1, "a"]
    assert from_dict == ExampleCast(2, "b")
    assert ExampleCast.castinit(existing) is existing
    assert ExampleCast.castinit(None) is None

    routes = get_atomic_routes(
        [
            ("orders", "f2p_customer", "customer"),
            ("orders", "f2p_product", "product"),
            ("ignored", "p2f_other", "orders"),
        ]
    )
    assert ("dim-fact-dim", "orders", "f2p_customer", "customer", "product", "rev_f2p_product", "orders") in routes
    assert ("dim-fact-dim", "orders", "f2p_product", "product", "customer", "rev_f2p_customer", "orders") in routes


def test_csv_fallback_and_cleaning(tmp_path):
    path = tmp_path / "gbk.csv"
    pd.DataFrame({" name ": ["张三", "李四"], "cat": ["nan", "ok"]}).to_csv(
        path,
        index=False,
        encoding="gbk",
    )

    loaded = read_csv_with_fallback_encodings(path)
    cleaned = clean_df_by_col_types(
        loaded,
        {"cat": ColType.CATEGORICAL, "name": ColType.TEXT},
    )

    assert loaded.shape == (2, 2)
    assert cleaned.columns.tolist() == ["name", "cat"]
    assert pd.isna(cleaned.loc[0, "cat"])
    assert cleaned.loc[1, "cat"] == "ok"


def test_download_extract_and_time_column_helpers(tmp_path, monkeypatch):
    opened_urls = []

    def fake_urlopen(url, context):
        opened_urls.append((url, context is not None))
        return FakeUrlResponse([b"abc", b"def"])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    downloaded = download_url("https://example.com/data.csv?token=1", tmp_path)
    google_downloaded = download_google_url("file-id", tmp_path, "drive.bin")

    assert downloaded.endswith("data.csv")
    assert (tmp_path / "data.csv").read_bytes() == b"abcdef"
    assert (tmp_path / "drive.bin").read_bytes() == b"abcdef"
    assert opened_urls[1][0] == "https://drive.usercontent.google.com/download?id=file-id&confirm=t"

    with pytest.raises(AssertionError):
        download_url("ftp://example.com/file.txt", tmp_path)

    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("folder/file.txt", "content")
    extract_zip(zip_path, tmp_path / "unzipped")
    assert (tmp_path / "unzipped" / "folder" / "file.txt").read_text() == "content"

    unix_seconds = timecol_to_unix_time(pd.Series(pd.to_datetime(["1970-01-01", "1970-01-02"])))
    assert unix_seconds.tolist() == [0, 86400]

    with pytest.raises(AssertionError):
        timecol_to_unix_time(pd.Series(["not", "datetime"]))


def test_type_inferencer_identifies_common_series_types():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.5, None],
            "cat": ["a", "b", "a"],
            "time": ["2020-01-01", "2020-01-02", None],
        }
    )

    inferred = TypeInferencer.infer_df_coltype(df)

    assert inferred["num"] == ColType.NUMERICAL
    assert TypeInferencer.infer_series_coltype(df["num"]) == ColType.NUMERICAL
    assert set(inferred).issubset(df.columns)


def test_type_inferencer_covers_binary_text_timestamp_and_table_dict():
    bool_ser = pd.Series([True, False, True])
    text_ser = pd.Series(["long free text", "another sentence", "third value"])
    timestamp_ser = pd.Series(["2020/01/01", "2020/01/02"])
    empty_ser = pd.Series([None, None])

    inferred_tables = TypeInferencer.infer_table_df_dict_coltype(
        {"table": pd.DataFrame({"flag": bool_ser, "text": text_ser})}
    )

    assert TypeInferencer.infer_series_coltype(bool_ser) == ColType.BINARY
    assert TypeInferencer.infer_series_coltype(text_ser) == ColType.TEXT
    assert TypeInferencer.infer_series_coltype(timestamp_ser) == ColType.TIMESTAMP
    assert TypeInferencer.infer_series_coltype(empty_ser) is None
    assert inferred_tables["table"]["flag"] == ColType.BINARY


def test_undirected_conversion_adds_reverse_edges_with_matching_values():
    directed = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([3.0, 4.0]),
        (3, 3),
    ).coalesce()

    undirected = to_undirected(directed)

    assert not is_undirected(directed)
    assert is_undirected(undirected)
    assert undirected.to_dense().tolist() == [
        [0.0, 3.0, 0.0],
        [3.0, 0.0, 4.0],
        [0.0, 4.0, 0.0],
    ]


def test_undirected_rejects_non_square_adjacency():
    adj = torch.sparse_coo_tensor(
        torch.tensor([[0], [1]]),
        torch.tensor([1.0]),
        (2, 3),
    ).coalesce()

    assert is_undirected(adj) is False


@pytest.mark.parametrize("seed", range(10))
def test_seeded_random_sparse_roundtrip_smoke(seed):
    generator = torch.Generator().manual_seed(seed)
    n = 6
    edge_index = torch.randint(0, n, (2, 12), generator=generator)
    values = torch.arange(12, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_index, values, (n, n)).coalesce()

    roundtrip_edge_index, edge_attr = adj_to_edge_index(adj)

    assert roundtrip_edge_index.shape[0] == 2
    assert roundtrip_edge_index.shape[1] == adj._nnz()
    assert edge_attr is not None
