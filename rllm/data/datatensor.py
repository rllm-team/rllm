import torch

def _get_id_mapping(ids):
    r"""
    Create an index mapping from index List `ids`.
    """
    mapping, cnt = {}, 0
    for _ in ids:
        x = _.item()
        assert (x not in mapping)
        mapping[x] = cnt
        cnt += 1
    return mapping

def _I_id_mapping(n):
    r"""
    Create an identical index mapping. (`ids`=[0, 1, 2,...])
    """
    return {i: i for i in range(n)}

class feature:
    r"""
    Representing the feature / label of dataset, in the form of Tensor.

    Args:
        x (Dict[metadata, Tensor]): feature of each node class.
    """

    def __init__(self, x):
        self.x = x
        self.meta = list(x.keys())
        self.v_num = {_: x[_].shape[0] for _ in self.meta}

    def __getitem__(self, key):
        r"""
        Returns feature (Tensor) if key is not List.
        Returns a new :class:`feature` if key is List.

        .. code-block:: python

            feature = datatensor.x['movie']
            new_x = datatensor.x[['movie', 'user']]
        """
        if type(key) is not list:
            return self.x[key]
        else:
            return feature({_: self.x[_] for _ in key})
    
    def __setitem__(self, key, value):
        r"""
        Assigns value (Tensor) to feature if key is list.
        Successively assigns value (a List of Tensor) to feature if key is list.
        Note that if metadata `key` or `key[i]` does not exist yet, it will be automatically created.

        .. code-block:: python

            datatensor.x['movie'] = feature
            datatensor.x[['movie', 'user']] = [mfeat, ufeat]
        """
        if type(key) is not list:
            self.x[key] = value
            if (key not in self.meta):
                self.meta.append(key)
                self.v_num[key] = self.x[key].shape[0]
        else:
            assert (len(key) == len(value))
            for i in len(key):
                self.x[key[i]] = value[i]
                if (key[i] not in self.meta):
                    self.meta.append(key[i])
                    self.v_num[key[i]] = self.x[key[i]].shape[0]

    def to_homo(self):
        r"""
        Returns feature of the whole dataset taken as homogeneous graph.
        Features of different categories are connected into a block diagonal matrix.

        .. code-block:: python

            feature = datatensor.x.to_homo()
        """
        return torch.block_diag(*[self.x[_] for _ in self.meta])

class edgeset:
    r"""
    Representing the edgeset of dataset, in the form of torch.Tensor.
    
    Args:
        e (Dict[metadata, Tensor]): Edge index of each edge class.
        w (Dict[metadata, Tensor]): Edge weight of each edge class.
        meta (List[metadata]): List of metadata of edges.
        v_num (Dict[metadata, Int]): Node numbers of each node class.
    """
    def __init__(self, e, w, meta, _parent):
        v_num = _parent.x.v_num
        self.e = {}
        for i in range(len(meta)):
            _ = meta[i]
            self.e[_[0]] = torch.sparse_coo_tensor(e[_[0]], w[_[0]], (v_num[_[1]], v_num[_[2]])).coalesce()
        self.meta = meta
        self._parent = _parent

    def _getkey_single(self, key):
        if type(key) is not tuple:
            return key
        if type(key) is tuple and len(key) == 3:
            return key[0]
        if type(key) is tuple and len(key) == 2:
            for _ in self.meta:
                if _[1] == key[0] and _[2] == key[1]:
                    return _[0]
            return None

    def __getitem__(self, key):
        r"""
        Returns adjancency matrix (torch.sparse) if key is not list or a new :class:`edgeset` if key is list.
        `key` or `key[i]` can be: 1) edge metadata Edge_key, 2) 2-tuple (From_key, To_key) or 3) 3-tuple (Edge_key, From_key, To_key).
        Note that 2) is not recommended, especially when different edge classes have same (From_key, To_key).

        .. code-block:: python

            adj = datatensor.e['rating']
            adj = datatensor.e[('user', 'movie')]
            adj = datatensor.e[('rating', 'user', 'movie')]
            new_e = datatensor.e[['rating1', 'rating2']]
        """
        if type(key) is not list:
            return self.e[self._getkey_single(key)]
        if type(key) is list:
            keys = [self._getkey_single(_) for _ in key]
            return edgeset({_: self.e[_].indices() for _ in keys},
                           {_: self.e[_].values() for _ in keys},
                           keys,
                           self._parent)

    def __setitem__(self, key, value):
        r"""
        Assigns adjancency matrix (torch.sparse) or a list of matrices to key.
        `key` or `key[i]` can be: 1) edge metadata Edge_key, 2) 2-tuple (From_key, To_key) or 3) 3-tuple (Edge_key, From_key, To_key).
        Note that if `key` or `key[i]` does not exist yet and is a 3-tuple, it will be automatically created.

        .. code-block:: python

            datatensor.e['rating'] = adj
            datatensor.e[('user', 'movie')] = adj
            datatensor.e[('rating', 'user', 'movie')] = adj
            datatensor.e[['rating1', 'rating2']] = [adj1, adj2]
        """
        if type(key) is not list:
            _ = self._getkey_single(key)
            if _ not in self.e.keys():
                if type(key) is tuple and len(key) == 3:
                    self.meta.append(key)
                else:
                    raise KeyError('information insufficient to automatically create edge class:', key)
            self.e[_] = value
            return
        if type(key) is list:
            assert(len(key) == len(value))
            for i in range(len(key)):
                _ = self._getkey_single(i)
                if _ not in self.e.keys():
                    if type(key[i]) is tuple and len(key[i]) == 3:
                        self.meta.append(key[i])
                    else:
                        raise KeyError('information insufficient to automatically create edge class:', key[i])
                self.e[_] = value[i]

    def to_homo(self):
        r"""
        Returns adjancency matrix of the whole dataset taken as homogeneous graph.

        .. code-block:: python

            adj = datatensor.e.to_homo()
        """
        v_num = self._parent.x.v_num
        xmeta = list(v_num.keys())
        sum = 0
        sft = {}
        for _ in xmeta:
            sft[_] = sum
            sum += v_num[_]
        shape = (sum, sum)
        res = torch.zeros(shape).to_sparse().to(self.e[self.meta[0][0]].device)
        for _ in self.meta:
            e = self.e[_[0]].indices()
            w = self.e[_[0]].values()
            e[0] = e[0] + sft[_[1]]
            e[1] = e[1] + sft[_[2]]
            res += torch.sparse_coo_tensor(e, w, shape)
        res = res.coalesce()
        return res
    
    def to_homo_weighted(self, beta):
        r"""
        Returns adjancency matrix of the whole dataset taken as homogeneous graph with weight vector beta.
        Different types of edges are multiplied with weight beta (Tensor).
        
        .. code-block:: python

            adj = datatensor.e.to_homo_weighted(beta)
        """
        v_num = self._parent.x.v_num
        xmeta = list(v_num.keys())
        sum = 0
        sft = {}
        for _ in xmeta:
            sft[_] = sum
            sum += v_num[_]
        shape = (sum, sum)
        res = []
        for _ in self.meta:
            e = self.e[_[0]].indices()
            w = self.e[_[0]].values()
            e[0] = e[0] + sft[_[1]]
            e[1] = e[1] + sft[_[2]]
            res.append(torch.sparse_coo_tensor(e, w, shape))
        A_t = torch.stack(res, dim=2).to_dense()
        temp = torch.matmul(A_t, beta)
        temp = torch.squeeze(temp, 2)
        return temp + temp.transpose(0, 1)
    
    def hop_2(self, keys):
        res = self[self._getkey_single(keys[0])]
        for i in range(1, len(keys)):
            res = torch.spmm(res, self[self._getkey_single(keys[i])])
        return res

class GraphStore:
    r""""
    Describes a dataset. Containing feature, label and edgeset.
    Example: data.x, data.y, data.e, data.normalize(), print(data)
    """
    def __init__(self):
        self.x = feature({})
        self.y = feature({})
        self.e = edgeset({}, {}, [], self)

    def __getitem__(self, key):
        r"""
        Returns sub-dataset with only nodes of types in key and edges between them.
        """
        if (type(key) is not tuple and type(key) is not list):
            key = tuple([key])

        x = [self.x[_] for _ in key]
        xmeta = key
        y = [self.y[_] for _ in self.y.meta if _ in key]
        ymeta = [_ for _ in self.y.meta if _ in key]
        edge_index = [self.e.e[_[0]] for _ in self.e.meta if _[1] in key and _[2] in key]
        edge_weight = [self.e.w[_[0]] for _ in self.e.meta if _[1] in key and _[2] in key]
        emeta = [_ for _ in self.e.meta if _[1] in key and _[2] in key]

        return GraphStore(x, xmeta, y, ymeta, edge_index, emeta, edge_weight=edge_weight)
    
    def node_count(self, key):
        return self.x[key].shape[0]

    def normalize(self, type='GCNNorm'):
        r"""
        Normalize features and adjancency matrix.
        """
        # todo: normalization types
        for _ in self.x.meta:
            self.x[_] /= self.x[_].sum(dim=1).view(-1, 1)
        for _ in self.e.meta:
            i = _[0]
            deg = self.e[i].to_dense().sum(dim=1) ** (-0.5)
            deg = torch.where(torch.isinf(deg), torch.full_like(deg, 0), deg)
            deg = torch.diag(deg)
            self.e[i] = torch.spmm(deg, torch.spmm(self.e[i], deg)).to_sparse()

    def to(self, device):
        for _ in self.x.meta:
            self.x[_] = self.x[_].to(device)
        for _ in self.y.meta:
            self.y[_] = self.y[_].to(device)
        for _ in self.e.meta:
            self.e[_[0]] = self.e[_[0]].to(device)
    
    def __repr__(self):
        res1 = self.__class__.__name__ + '(\n'
        res2 = '\tnum_nodes sum: ' + str(sum([self.x[_].shape[0] for _ in self.x.meta])) + '; '
        for _ in self.x.meta:
            res2 += '{}: {}; '.format(_, self.x[_].shape[0])
        res2 += '\n'
        res3 = '\tfeature '
        for _ in self.x.meta:
            res3 += '{}: {} * {}; '.format(_, self.x[_].shape[0], self.x[_].shape[1])
        res3 += '\n'
        res4 = '\tlabel '
        for _ in self.y.meta:
            if len(self.y[_].shape) > 1:
                res4 += '{}: {} * {}; '.format(_, self.y[_].shape[0], self.y[_].shape[1])
            else:
                res4 += '{}: {} * {}; '.format(_, self.y[_].shape[0], 1)
        res4 += '\n'
        res5 = '\tedge '
        for _ in self.e.meta:
            res5 += '({}, {} -> {}): {}; '.format(_[0], _[1], _[2], self.e[_[0]]._nnz())
        res5 += '\n'

        return res1 + res2 + res3 + res4 + res5 + ')\n'
    
def from_datadf(ddf):
    res = GraphStore()
    res.x = feature({_: torch.FloatTensor(ddf.x[_].values) for _ in ddf.x.meta})
    res.y = feature({_: torch.FloatTensor(ddf.y[_].values) for _ in ddf.y.meta})
    res.e = edgeset({_[0]: torch.FloatTensor(ddf.e.e[_[0]].values) for _ in ddf.e.meta},
                    {_[0]: torch.FloatTensor(ddf.e.w[_[0]].values) for _ in ddf.e.meta},
                    ddf.e.meta,
                    res)
    return res

def legacy_init(x, xmeta, y, ymeta, e, emeta, node_index=None, edge_weight=None):
    r"""
    Create a GraphStore object with node features, metadata of nodes, labels, metadata of labels, edges, metadata of edges.
    node_index (List of index of nodes, used in ei) and edge_weight (Edge weights) are optional.
    Example: legacy_init([ufeat, mfeat], ['user', 'movie'], [label], ['movie'], [ei], [('rating', 'user', 'movie')], node_index=[uid, mid], edge_weight=[edge_weight])
    """
    res = GraphStore()

    assert (len(x) == len(xmeta))
    res.x = feature({xmeta[i]: x[i] for i in range(len(x))})
    vmap = {}
    for i in range(len(x)):
        vmap[xmeta[i]] = _get_id_mapping(node_index) if node_index is not None\
                            else _I_id_mapping(x[i].shape[0])

    assert (len(y) == len(ymeta))
    res.y = feature({ymeta[i]: y[i] for i in range(len(y))})

    assert (len(e) == len(emeta))
    e_tmp, w_tmp = {}, {}
    for i in range(len(e)):
        ei = e[i]
        ev = torch.ones(ei.shape[1]) if edge_weight is None else edge_weight[i]
        e_tmp[emeta[i][0]] = ei
        w_tmp[emeta[i][0]] = ev
    res.e = edgeset(e_tmp, w_tmp, emeta, res)

    return res