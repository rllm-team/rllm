from collections import defaultdict

def get_atomic_routes(edge_type_list):
    r"""Decompose relational edge types into atomic message passing routes
    for the RelGNN model.

    Each foreign-key edge type (prefixed with :obj:`'f2p'`) is mapped to
    one of two route patterns:

    - **dim-dim**: A single foreign-key edge and its reverse, used when a
      fact table connects to exactly one dimension table.
    - **dim-fact-dim**: A pair of foreign-key edges through a shared fact
      table, used when a fact table connects to multiple dimension tables.

    Args:
        edge_type_list (List[Tuple[str, str, str]]): The list of edge types
            in :obj:`(src, rel, dst)` format from the heterogeneous graph.

    Returns:
        List[Tuple]: A list of atomic routes. Each entry is a tuple whose
        first element is the route type (:obj:`'dim-dim'` or
        :obj:`'dim-fact-dim'`) followed by the edge type components.
    """
    src_to_tuples = defaultdict(list)
    for src, rel, dst in edge_type_list:
        if rel.startswith('f2p'):
            if src == dst:
                src = src + '--' + rel
            src_to_tuples[src].append((src, rel, dst))

    atomic_routes_list = []
    get_rev_edge = lambda edge: (edge[2], 'rev_' + edge[1], edge[0])
    for src, tuples in src_to_tuples.items():
        if '--' in src:
            src = src.split('--')[0]
        if len(tuples) == 1:
            _, rel, dst = tuples[0]
            edge = (src, rel, dst)
            atomic_routes_list.append(('dim-dim',) + edge)
            atomic_routes_list.append(('dim-dim',) + get_rev_edge(edge))
        else:
            for _, rel_q, dst_q in tuples:
                for _, rel_v, dst_v in tuples:
                    if rel_q != rel_v:
                        edge_q = (src, rel_q, dst_q)
                        edge_v = (src, rel_v, dst_v)
                        atomic_routes_list.append(('dim-fact-dim',) + edge_q + get_rev_edge(edge_v))

    return atomic_routes_list
