import torch
# import sys
# import os

# # Add project root to path to ensure imports work
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rllm.data import HeteroGraphData
from rllm.dataloader.sampler import HeteroSampler
from rllm.dataloader.sampler.data_type import NodeSamplerInput

def test_hetero_sampler_node_time():
    # 1. Construct Graph
    # User (0, 1, 2)
    # Item (0, 1)
    # Edges (User -> Item):
    # User 0 -> Item 0
    # User 1 -> Item 0
    # User 2 -> Item 0
    data = HeteroGraphData()

    # Node times
    # User 0: t=5
    # User 1: t=15
    # User 2: t=25
    data['user'].time = torch.tensor([5, 15, 25], dtype=torch.long)

    # Item times (not strictly used for filtering neighbors, but good for consistency)
    data['item'].time = torch.tensor([0, 0], dtype=torch.long)

    # Edge indices
    src = torch.tensor([0, 1, 2], dtype=torch.long)
    dst = torch.tensor([0, 0, 0], dtype=torch.long)
    data['user', 'click', 'item'].edge_index = torch.stack([src, dst], dim=0)

    # Note: No edge times assigned.

    # 2. Initialize HeteroSampler
    # We want to sample neighbors for Item 0.
    # The neighbors are Users.
    # Filtering should be based on User's time.

    sampler = HeteroSampler(
        hdata=data,
        num_neighbors=[3], # Sample all valid neighbors
        time_attr='time',
        temporal_strategy='uniform'
    )

    # 3. Test Case 1: Sample Item 0 at t=20
    # Valid neighbors (User.time <= 20):
    # User 0 (t=5)  -> Valid
    # User 1 (t=15) -> Valid
    # User 2 (t=25) -> Invalid (Future)

    seed_node = torch.tensor([0], dtype=torch.long)
    seed_time = torch.tensor([20], dtype=torch.long)

    input_data = NodeSamplerInput(
        input_id=None,
        node=seed_node,
        time=seed_time,
        input_type='item'
    )

    out = sampler.sample_neighbors(input_data)

    item_nodes = out.node['item']
    user_nodes = out.node['user']

    # Verify seed
    assert 0 in item_nodes.tolist(), "Seed Item 0 should be present"

    # Verify neighbors
    user_list = user_nodes.tolist()

    assert 0 in user_list, "User 0 (t=5) should be sampled"
    assert 1 in user_list, "User 1 (t=15) should be sampled"
    assert 2 not in user_list, "User 2 (t=25) should NOT be sampled (Future)"

    # 4. Test Case 2: Sample Item 0 at t=10
    # Valid neighbors (User.time <= 10):
    # User 0 (t=5)  -> Valid
    # User 1 (t=15) -> Invalid
    # User 2 (t=25) -> Invalid

    seed_time_2 = torch.tensor([10], dtype=torch.long)
    input_data_2 = NodeSamplerInput(
        input_id=None,
        node=seed_node,
        time=seed_time_2,
        input_type='item'
    )

    out_2 = sampler.sample_neighbors(input_data_2)
    user_list_2 = out_2.node['user'].tolist()

    assert 0 in user_list_2, "User 0 (t=5) should be sampled"
    assert 1 not in user_list_2, "User 1 (t=15) should NOT be sampled"
    assert 2 not in user_list_2, "User 2 (t=25) should NOT be sampled"


# if __name__ == "__main__":
#     test_hetero_sampler_node_time()
