import torch

from rllm.data import HeteroGraphData
from rllm.dataloader.sampler import HeteroSampler
from rllm.dataloader.sampler.data_type import NodeSamplerInput

def test_reverse_sampling():
    # 1. Construct Graph: User -> Item
    # We want to verify that if we sample from 'Item', we find 'User'.
    data = HeteroGraphData()

    # Nodes
    data['user'].num_nodes = 3
    data['item'].num_nodes = 1

    # Node Times
    # User 0: t=10
    # User 1: t=20
    # User 2: t=30
    data['user'].time = torch.tensor([10, 20, 30], dtype=torch.long)
    # Item time doesn't strictly matter for
    # this direction if we filter by neighbor time
    data['item'].time = torch.tensor([0], dtype=torch.long)

    # Edges: User -> Item
    # All users click Item 0
    src = torch.tensor([0, 1, 2], dtype=torch.long)
    dst = torch.tensor([0, 0, 0], dtype=torch.long)
    data['user', 'click', 'item'].edge_index = torch.stack([src, dst], dim=0)

    # 2. Initialize Sampler
    # We want to sample neighbors for 'item'.
    # Since edges are 'user' -> 'item',
    # the neighbors of 'item' are 'user's (incoming edges).
    # This requires the sampler to traverse edges in REVERSE (CSC format).

    sampler = HeteroSampler(
        hdata=data,
        num_neighbors=[3], # Sample up to 3 neighbors
        time_attr='time',
        temporal_strategy='uniform',
        csc=True  # Enable CSC for reverse sampling
    )

    # 3. Perform Sampling
    # Seed: Item 0 at time t=25
    # Expected result: User 0 (t=10) and User 1 (t=20).
    # User 2 (t=30) is in the future.
    seed_node = torch.tensor([0], dtype=torch.long)
    seed_time = torch.tensor([25], dtype=torch.long)

    input_data = NodeSamplerInput(
        input_id=None,
        node=seed_node,
        time=seed_time,
        input_type='item'
    )

    out = sampler.sample_neighbors(input_data)

    # 4. Verification
    user_nodes = out.node['user'].tolist()

    # Check correctness
    assert 0 in user_nodes, "User 0 (t=10) should be sampled (10 <= 25)"
    assert 1 in user_nodes, "User 1 (t=20) should be sampled (20 <= 25)"
    assert 2 not in user_nodes, "User 2 (t=30) should NOT be sampled (30 > 25)"
