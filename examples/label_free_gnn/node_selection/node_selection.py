from .active import *
from .ns_helpers import *
from tqdm import tqdm


def active_generate_mask(data, method, val=True, budget=20):
    """
    Actively select nodes for annotation. Supported active methods include VertexCover, FeatProp and Random.
    """
    test_mask = generate_test_mask(data.num_nodes)
    select_mask = ~test_mask
    if 'cache_mask' in data.keys():
        select_mask = select_mask & data['cache_mask']
    select_mask = remove_isolated_nodes(select_mask, data.adj)

    total_budget = min(budget * data.num_classes, select_mask.sum())
    if method == 'VertexCover':
        selected_indices = vertex_cover_query(total_budget, data.adj.coalesce().indices(), select_mask)
    elif method == 'FeatProp':
        selected_indices = featprop_query(total_budget, data.x, data.adj.coalesce().indices(), select_mask)
    elif method == 'Random':
        selected_indices = random_query(total_budget, data.num_nodes, select_mask)
    else:
        print('invalid active method:', method)
        print('use Random instead')
        selected_indices = random_query(total_budget, data.num_nodes, select_mask)

    if val:
        train_budget = total_budget * 3 // 4
    else:
        train_budget = total_budget
    train_indices = selected_indices[:train_budget]
    val_indices = selected_indices[train_budget:]
    train_mask = index_to_mask(train_indices, data.num_nodes)
    val_mask = index_to_mask(val_indices, data.num_nodes)

    return train_mask, val_mask, test_mask


def post_filter(data, mask, strategy, ratio=0.3):
    """
    Post filter nodes using combinations of confidence, density and change of entropy.
    """
    budget = mask.sum()
    b = int(budget * ratio)
    N = data.num_nodes
    labels = data.y
    conf = data.conf.clone()
    density = get_density(data)
    if strategy == 'conf+density':
        conf[~mask] = 0
        density[~mask] = 0
        percentile = (torch.arange(N, dtype=data.x.dtype) / N)
        id_sorted = conf.argsort(descending=True)
        conf[id_sorted] = percentile
        id_sorted = density.argsort(descending=True)
        density[id_sorted] = percentile
        score = conf + density
        score[~mask] = 0
        _, indices = torch.topk(score, k=b)
        mask[indices] = 0
    elif strategy == 'conf+density+entropy':
        echange = -get_entropy_change(labels)
        conf[~mask] = 0
        density[~mask] = 0
        echange[~mask] = 0
        percentile = (torch.arange(N, dtype=data.x.dtype) / N)
        id_sorted = conf.argsort(descending=True)
        conf[id_sorted] = percentile
        id_sorted = density.argsort(descending=True)
        density[id_sorted] = percentile
        id_sorted = echange.argsort(descending=True)
        echange[id_sorted] = percentile
        score = conf + density + echange
        score[~mask] = 0
        _, indices = torch.topk(score, k=b)
        mask[indices] = 0
    else:
        print(f'invalid filter strategy: {strategy}')
    return mask
