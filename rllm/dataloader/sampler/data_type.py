from typing import Any, Optional, Union, Dict, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor

from rllm.utils._mixin import CastMixin


@dataclass(init=False)
class NodeSamplerInput(CastMixin):
    r"""The sampling input data class.

    Args:
        input_id (torch.Tensor, optional): The indices of the data loader input
            of the current mini-batch.
        node (torch.Tensor): The indices of seed nodes to start sampling from.
        time (torch.Tensor, optional): The timestamp for the seed nodes.
            (default: :obj:`None`)
        input_type (str, optional): The input node type (in case of sampling in
            a heterogeneous graph). (default: :obj:`None`)
    """
    input_id: Optional[Tensor]
    node: Tensor
    time: Optional[Tensor] = None
    input_type: Optional[str] = None

    def __init__(
        self,
        input_id: Optional[Tensor],
        node: Tensor,
        time: Optional[Tensor] = None,
        input_type: Optional[str] = None,
    ):
        if input_id is not None:
            input_id = input_id.cpu()
        node = node.cpu()
        if time is not None:
            time = time.cpu()

        self.input_id = input_id
        self.node = node
        self.time = time
        self.input_type = input_type

    def __getitem__(self, index: Union[Tensor, Any]) -> 'NodeSamplerInput':
        if not isinstance(index, Tensor):
            index = torch.tensor(index, dtype=torch.long)

        return NodeSamplerInput(
            self.input_id[index] if self.input_id is not None else index,
            self.node[index],
            self.time[index] if self.time is not None else None,
            self.input_type,
        )


@dataclass
class HeteroSamplerOutput(CastMixin):
    """
    Outout of the heterosampler, which only contains
    the indices of the sampled nodes and edges.
    
    Edge type: Tuple[str, str, str]
    Node type: str
    """
    node: Dict[str, Tensor]
    row: Dict[Tuple, Tensor]
    col: Dict[Tuple, Tensor]
    batch: Optional[Dict[str, Tensor]] = None
    num_sampled_nodes: Optional[Dict[str, int]] = None
    num_sampled_edges: Optional[Dict[Tuple, int]] = None
    original_row: Optional[Dict[Tuple, Tensor]] = None
    original_col: Optional[Dict[Tuple, Tensor]] = None
    metadata: Optional[Any] = None