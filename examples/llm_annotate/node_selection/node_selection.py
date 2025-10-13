from .active import *
from .ns_helpers import *


def active_generate_mask(data, method, select_mask, train_num):
    """
    Actively select nodes for annotation. Supported active methods include VertexCover, FeatProp and Random.
    """

    select_mask = remove_isolated_nodes(select_mask, data.adj)

    total_budget = min(train_num, select_mask.sum())
    if method == 'Degree':
        selected_indices = degree_query(total_budget, data.adj.coalesce().indices(), select_mask)
    elif method == 'Random':
        selected_indices = random_query(total_budget, data.num_nodes, select_mask)
    else:
        print('invalid active method:', method)
        print('use Random instead')
        selected_indices = random_query(total_budget, data.num_nodes, select_mask)

    train_mask = index_to_mask(selected_indices, data.num_nodes)

    return train_mask
