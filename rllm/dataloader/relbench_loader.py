from copy import copy
from typing import Optional, List, Tuple, Callable, Union, Any, Dict

import numpy as np
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

    def __init__(
        self,
        dataset: RelBenchDataset,
        task: Union[RelBenchTask, str],
        split: str = 'train',
        shuffle: bool = False,
        batch_size: int = 512,
    ):
        dataset.load_all()  # make sure dataset is processed

        self.hdata: HeteroGraphData = dataset.hdata

        if isinstance(task, str):
            assert (task in dataset.tasks,
                f"Task {task} not found in dataset tasks {dataset.tasks}")
            self.task = dataset.task_dict[task]
        else:
            self.task = task

        self.split = split
        self.shuffle = shuffle
        self.batch_size = batch_size

        # build sampler
        self.hetero_sampler = HeteroSampler()

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
        out: HeteroSamplerOutput = self.hetero_sampler.sample(input)    # TODO
        return out

    def filter_fn(self, batch: HeteroSamplerOutput) -> HeteroGraphData:
        """Join the sampled nodes with their features."""
        batch_hdata = self._filter_hetero_data(
            node_dict=batch.node,
            row_dict=batch.row,
            col_dict=batch.col,
            perm_dict=self.hetero_sampler.edge_permutation,
        )
        # TODO


    def __call__(self, index: Union[List[int], Tensor]) -> HeteroGraphData:
        out: HeteroSamplerOutput = self.collate_fn(index)
        out: HeteroGraphData = self.filter_fn(out)
        return out

    # helper func
    def _filter_hetero_data(
        self,
        node_dict: Dict[str, Tensor],
        row_dict: Dict[Tuple, Tensor],
        col_dict: Dict[Tuple, Tensor],
        perm_dict: Optional[Dict[Tuple, Tensor]] = None,
    ) -> HeteroGraphData:
        data = self.hdata
        out = copy(data)

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

            filter_edge_store_(
                data[edge_type],
                out[edge_type],
                row_dict[edge_type],
                col_dict[edge_type],
                perm_dict.get(edge_type, None) if perm_dict else None,
            )

        return out

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
                if isinstance(value, (Tensor, TableData)):  # TODO: add TableData support
                    index = index.to(value.device)
                elif isinstance(value, np.ndarray):
                    index = index.cpu()
                dim = store._parent().__cat_dim__(key, value, store)
                out_store[key] = RelbenchLoader.index_select(value, index, dim=dim)

    @staticmethod
    def index_select(
        value: Union[Tensor, np.ndarray],
        index: Tensor,
        dim: int = 0,
    ) -> Tensor:
        r"""Indexes the :obj:`value` tensor along dimension :obj:`dim` using the
        entries in :obj:`index`.

        Args:
            value (torch.Tensor or np.ndarray): The input tensor.
            index (torch.Tensor): The 1-D tensor containing the indices to index.
            dim (int, optional): The dimension in which to index.
                (default: :obj:`0`)

        .. warning::

            :obj:`index` is casted to a :obj:`torch.int64` tensor internally, as
            `PyTorch currently only supports indexing
            <https://github.com/pytorch/pytorch/issues/61819>`_ via
            :obj:`torch.int64`.
        """
        # PyTorch currently only supports indexing via `torch.int64`:
        # https://github.com/pytorch/pytorch/issues/61819
        index = index.to(torch.int64)

        if isinstance(value, Tensor):
            out: Optional[Tensor] = None
            if torch.utils.data.get_worker_info() is not None:
                # If we are in a background process, we write directly into a
                # shared memory tensor to avoid an extra copy:
                size = list(value.shape)
                size[dim] = index.numel()
                numel = math.prod(size)
                if torch_geometric.typing.WITH_PT20:
                    storage = value.untyped_storage()._new_shared(
                        numel * value.element_size())
                else:
                    storage = value.storage()._new_shared(numel)
                out = value.new(storage).view(size)

            return torch.index_select(value, dim, index, out=out)

        if isinstance(value, TensorFrame):
            assert dim == 0
            return value[index]

        elif isinstance(value, np.ndarray):
            return torch.from_numpy(np.take(value, index, axis=dim))

        raise ValueError(f"Encountered invalid feature tensor type "
                        f"(got '{type(value)}')")