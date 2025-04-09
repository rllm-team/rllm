from typing import List, Optional

from torch import Tensor

from rllm.data import TableData, GraphData
from rllm.dataloader.neighbor_loader import NeighborLoader


class BRIDGELoader(NeighborLoader):
    r"""BRIDGELoader is a specialized data loader for the BRIDGE model.
    It is designed to handle the unique requirements of the BRIDGE model
    and provides additional functionality for processing graph and table
    data in a unified manner.

    BRIDGE always put table before non-table embeddings in the graph
    node index. After sampling `n_id`, we need to split it by the lengh
    of table: `n_id <= self.sep` is the table node index and
    `n_id > self.sep` is the non-table node index.

    Args:
        table (TableData): The table data to be sampled.
        non_table (Optional[Tensor]): The non-table data to be sampled.
        graph (GraphData): The graph data to be sampled.
        num_samples (List[int]): The number of samples to be taken
            from each layer of the graph.
        train_mask (Optional[Tensor]): The mask to be used for sampling.
        **kwargs: Additional keyword arguments to be passed to the
            `NeighborLoader` class.
    """

    def __init__(
        self,
        table: TableData,
        non_table: Optional[Tensor],
        graph: GraphData,
        num_samples: List[int],
        train_mask: Optional[Tensor] = None,
        **kwargs
    ):
        super().__init__(
            data=graph,
            num_neighbors=num_samples,
            seeds=train_mask,
            replace=False,
            transform=None,
            **kwargs
        )
        self.table = table
        self.non_table = non_table
        self.sep = len(table)

    def collate_fn(self, batch):
        batch, n_id, adjs = super().collate_fn(batch)
        if self.non_table is None:
            return batch, n_id, adjs, self.table[n_id], None
        table_id = n_id[n_id < self.sep]
        non_table_id = n_id[n_id >= self.sep] - self.sep
        table_data = self.table[table_id]
        non_table_data = self.non_table[non_table_id]
        return batch, n_id, adjs, table_data, non_table_data
