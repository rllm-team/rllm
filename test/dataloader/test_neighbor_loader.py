import torch

from rllm.dataloader import NeighborLoader


def test_neighbor_loader():
    edge_index = torch.tensor([
        [0, 0, 1, 2, 2, 3, 4, 4, 5],
        [1, 2, 3, 3, 4, 4, 5, 6, 6]
    ])

    num_samples = [1]
    node_idx = torch.tensor([5, 4])
    num_nodes = 7

    loader = NeighborLoader(
        edge_index=edge_index,
        num_samples=num_samples,
        node_idx=node_idx,
        num_nodes=num_nodes,
        replace=False,
        return_oeid=True,
        batch_size=2,
    )

    batch, n_id, e_ids, o_eid = next(iter(loader))

    mask = n_id.unsqueeze(0) == node_idx.unsqueeze(1)

    res = torch.where(
        mask.any(dim=1),
        mask.long().argmax(dim=1),
        torch.tensor(-1, device=batch.device)
    )

    assert torch.all(res == batch)
    assert len(e_ids) == len(o_eid)
