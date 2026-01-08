from copy import copy
from typing import Optional, List, Tuple, Callable, Union, Dict

import torch
from torch import Tensor

from rllm.data import HeteroGraphData, NodeStorage, TableData
from rllm.datasets import RelBenchDataset, RelBenchTask, RelBenchTaskType
from rllm.dataloader.sampler import NodeSamplerInput, HeteroSamplerOutput, HeteroSampler
from rllm.utils.col_process import timecol_to_unix_time


class AttachTargetTransform:
    r"""Attach the target label to the heterogeneous mini-batch.

    The batch consists of disjoins subgraphs loaded via temporal sampling. The same
    input node can occur multiple times with different timestamps, and thus different
    subgraphs and labels. Hence labels cannot be stored in the graph object directly,
    and must be attached to the batch after the batch is created.
    """

    def __init__(self, entity: str, target: Tensor):
        self.entity = entity
        self.target = target

    def __call__(self, batch: HeteroGraphData) -> HeteroGraphData:
        batch[self.entity].y = self.target[batch[self.entity].input_id]
        return batch


class RelbenchLoader(torch.utils.data.DataLoader):
    r"""DataLoader for RelBench dataset with heterogeneous neighbor sampling.
    Args:
        dataset (RelBenchDataset): The RelBench dataset.
        task (Union[RelBenchTask, str]): The task to load.
        split (str): The data split to load. (default: :obj:`'train'`)
        shuffle (bool): Whether to shuffle the data. (default: :obj:`False`)
        batch_size (int): The batch size. (default: :obj:`512`)
        num_neighbors (List[int]): Number of neighbors to sample at each hop.
            (default: :obj:`[15, 10]`)
        to_bidirectional (bool): Whether to convert the graph to bidirectional
            by adding reverse edges. Default is False.
    """

    def __init__(
        self,
        dataset: RelBenchDataset,
        task: Union[RelBenchTask, str],
        split: str = 'train',
        shuffle: bool = False,
        batch_size: int = 512,
        num_neighbors: List[int] = [15, 10],
        to_bidirectional: bool = False,
        use_pyg_lib: bool = True,
    ):
        dataset.load_all()  # make sure dataset is processed

        self.hdata: HeteroGraphData = dataset.hdata

        if isinstance(task, str):
            assert task in dataset.tasks, \
                f"Task {task} not found in dataset tasks {dataset.tasks}"
            self.task = dataset.task_dict[task]
        else:
            self.task = task

        self.split = split
        self.shuffle = shuffle
        self.batch_size = batch_size

        # build sampler
        self.hetero_sampler = HeteroSampler(
            hdata=self.hdata,
            num_neighbors=num_neighbors,
            replace=False,
            temporal_strategy='uniform',
            time_attr="time",
            to_bidirectional=to_bidirectional,
            csc=True,
            use_pyg_lib=use_pyg_lib,
        )

        # build sampler input
        self.transform: Optional[Callable] = None
        self.sampler_input: NodeSamplerInput = self._get_sampler_input_from_task()

        iterator = range(self.sampler_input.node.size(0))

        # super init
        super().__init__(
            iterator,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            batch_size=batch_size
        )

    def _get_sampler_input_from_task(self) -> NodeSamplerInput:
        """
        Get the sampler input from the task,
        and set the transform to attach the target label to the batch.
        """
        task = self.task
        task_df, _ = task.task_data_dict[self.split]
        nodes = torch.from_numpy(task_df[task.entity_col].astype(int).values)
        time = torch.from_numpy(
            timecol_to_unix_time(task_df[task.time_col])
        )
        target = None
        if task.task_type == RelBenchTaskType.BINARY_CLASSIFICATION:
            target = torch.from_numpy(
                task_df[task.target_col].values.astype(float)
            )
        elif task.task_type == RelBenchTaskType.REGRESSION:
            target = torch.from_numpy(
                task_df[task.target_col].values.astype(float)
            )
        else:
            raise ValueError(
                f"Unsupported task type: {task.task_type} "
                f"for task: {task.task_name}"
            )

        self.transform = AttachTargetTransform(task.entity_table, target)

        return NodeSamplerInput(
            input_id=None,  # will be set after sampling
            node=nodes,
            time=time,
            input_type=task.entity_table,
        )

    def collate_fn(self, index: Union[List[int], Tensor]) -> HeteroSamplerOutput:
        """Sample a mini-batch sub-heterogeneous graph from input nodes."""
        input: NodeSamplerInput = self.sampler_input[index]
        out: HeteroSamplerOutput = self.hetero_sampler.sample_neighbors(input)  # TODO: validate this
        return out

    def filter_fn(self, out: HeteroSamplerOutput) -> HeteroGraphData:
        """Join the sampled nodes with their features."""
        # 1. filter hetero data with node features
        batch_hdata: HeteroGraphData = self._filter_hetero_data(
            node_dict=out.node,
            row_dict=out.row,
            col_dict=out.col,
            perm_dict=self.hetero_sampler.edge_permutation,
        )

        # 2. add `n_id` as original node indices for each node type
        for key, node in out.node.items():
            if 'n_id' not in batch_hdata[key]:
                batch_hdata[key].n_id = node

        # 3. set metadata
        batch_hdata.set_value_dict('num_sampled_nodes', out.num_sampled_nodes)
        batch_hdata.set_value_dict('num_sampled_edges', out.num_sampled_edges)

        input_type = self.sampler_input.input_type
        batch_hdata[input_type].input_id = out.metadata[0]
        batch_hdata[input_type].seed_time = out.metadata[1]
        batch_hdata[input_type].batch_size = out.metadata[0].size(0)

        batch_hdata.time_dict = {
            node_type: node_storage.time
            for node_type, node_storage in batch_hdata.node_items()
            if hasattr(node_storage, "time")
        }

        batch_hdata.batch_dict = {
            node_type: batch
            for node_type, batch in (out.batch or {}).items()
        }

        batch_hdata.edge_index_dict = {
            edge_type: edge_storage.edge_index
            for edge_type, edge_storage in batch_hdata.edge_items()
        }

        # 4. attach target label
        return self.transform(batch_hdata)

    def __call__(self, index: Union[List[int], Tensor]) -> HeteroGraphData:
        out: HeteroSamplerOutput = self.collate_fn(index)
        out: HeteroGraphData = self.filter_fn(out)
        return out

    def __iter__(self):
        for batch in super().__iter__():
            batch = self.filter_fn(batch)
            yield batch

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @staticmethod
    def get_loaders(
        dataset: RelBenchDataset,
        task: Union[RelBenchTask, str],
        batch_size: int = 512,
        num_neighbors: List[int] = [15, 10],
        to_bidirectional: bool = False,
    ) -> List['RelbenchLoader']:
        """
        Get the loaders for each split.

        Args:
            dataset: The dataset to load.
            task: The task to load.
            shuffle: Whether to shuffle the data.
            batch_size: The batch size.
            num_neighbors: The number of neighbors to sample.
        Returns:
            A list of [train_loader, val_loader, test_loader].
        """
        return [
            RelbenchLoader(
                dataset=dataset,
                task=task,
                split=split,
                shuffle=True if split == "train" else False,
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                to_bidirectional=to_bidirectional,
            ) for split in ["train", "val", "test"]
        ]


    # TODO: For now, we only have this loader with hetero sampler.
    # So these functions are specific.
    # If adding other loader with sampler,
    # these functions should be moved out to provide a more generic interface.
    def _filter_hetero_data(
        self,
        node_dict: Dict[str, Tensor],
        row_dict: Dict[Tuple, Tensor],
        col_dict: Dict[Tuple, Tensor],
        perm_dict: Optional[Dict[Tuple, Tensor]] = None,
    ) -> HeteroGraphData:
        data = self.hdata
        out = copy(data)    # shallow copy, add new storages

        for node_type in out.node_types:
            # Handle the case of disconneted graph sampling:
            if node_type not in node_dict:
                node_dict[node_type] = torch.empty(0, dtype=torch.long)

            self.filter_node_store_(
                data[node_type],
                out[node_type],
                node_dict[node_type]
            )

        for edge_type in out.edge_types:
            # Handle the case of disconneted graph sampling:
            if edge_type not in row_dict:
                row_dict[edge_type] = torch.empty(0, dtype=torch.long)
            if edge_type not in col_dict:
                col_dict[edge_type] = torch.empty(0, dtype=torch.long)

            self.filter_edge_store_(
                data[edge_type],
                out[edge_type],
                row_dict[edge_type],
                col_dict[edge_type],
                perm_dict.get(edge_type, None) if perm_dict else None,
            )

        return out

    @staticmethod
    def filter_edge_store_(
        store: NodeStorage,
        out_store: NodeStorage,
        row: Tensor,
        col: Tensor,
        perm: Optional[Tensor] = None
    ):
        # Filters a edge storage object to only hold the edges in `index`,
        # which represents the new graph as denoted by `(row, col)`:
        for key, value in store.items():
            if key == 'edge_index':
                edge_index = torch.stack([row, col], dim=0).to(value.device)
                out_store.edge_index = edge_index
            else:
                raise NotImplementedError(
                    f"Edge attribute key: {key} type: {type(value)} not supported."
                    "For now, edge attributes other than edge_index are not supported."
                )

    @staticmethod
    def filter_node_store_(
        store: NodeStorage,
        out_store: NodeStorage,
        index: Tensor
    ):
        # Filters a node storage object to only hold the nodes in `index`:
        for key, value in store.items():
            if key == 'num_nodes':
                out_store.num_nodes = index.numel()

            elif store.is_node_attr(key):
                if isinstance(value, TableData):
                    out_store[key] = RelbenchLoader.index_select(value, index)
                elif isinstance(value, Tensor):
                    # For now, hardcode for `time` tensor in Pkey-fkey graph.
                    assert value.dim() == 1, f"Tensor should be 1-D, but {value.dim()} found."
                    out_store[key] = value[index]
                else:
                    raise NotImplementedError(
                        f"Node attribute type {type(value)} not supported."
                    )

    @staticmethod
    def index_select(
        value: TableData,
        index: Tensor,
        dim: int = 0,
    ) -> Tensor:
        r"""Indexes the :obj:`value` table along dimension :obj:`dim` using the
        entries in :obj:`index`.

        Args:
            value (TableData): The input table.
            index (torch.Tensor): The 1-D tensor containing the indices to index.
            dim (int, optional): The dimension in which to index.
                (default: :obj:`0`)
        """
        index = index.to(torch.int64)

        if isinstance(value, TableData):
            assert dim == 0
            # only slice feature_dict, other attributes
            # like df will be shallow copied.
            return value[index]

        raise ValueError(f"Encountered invalid feature tensor type "
                        f"(got '{type(value)}')")
