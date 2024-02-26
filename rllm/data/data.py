import torch

class feature:
    r"""
    The feature of dataset. Support Tensor and Dataframe.
    Example: data.x['movie'], data.x[('movie', 'user')].to_homo()
    """
    def __init__(self, vmap, x, meta):
        self.vmap = vmap
        self.x = x
        self.meta = meta

    def __getitem__(self, key):
        r"""
        Returns feature (Tensor or Dataframe) if key is not tuple or list
        Returns a new :obj:`feature` if key is tuple or list
        """
        if type(key) is not tuple and type(key) is not list:
            assert (key in self.meta)
            return self.x[key]
        else:
            for _ in key:
                assert (_ in self.meta)
            return DataLoader.feature({_: self.vmap[_] for _ in key},
                                {_: self.x[_] for _ in key},
                                key)

    def __setitem__(self, key, value):
        r"""
        Assigns value (Tensor or Dataframe) to feature if key is not tuple or list
        Successively assigns value (a List of Tensor or Dataframe) to feature if key is tuple or list
        """
        if type(key) is not tuple and type(key) is not list:
            assert (key in self.meta)
            self.x[key] = value
        else:
            assert (len(key) == len(value))
            for i in len(key):
                assert (key[i] in self.meta)
                assert (self.x[key[i]].shape == value[i].shape)
                self.x[key[i]] = value[i]

    def to_homo(self):
        r"""
        Returns feature of the whole dataset taken as homogeneous graph
        Features of different categories are connected into a block diagonal matrix
        """
        return torch.block_diag(*[self.x[_] for _ in self.meta])


class label:
    r"""
    The label of dataset. Support Tensor and Dataframe.
    Example: data.y['movie'], data.y[('movie', 'user')].to_homo()
    """
    def __init__(self, y, meta):
        self.y = y
        self.meta = meta

    # data.y['qaq']
    def __getitem__(self, key):
        r"""
        Returns label (Tensor or Dataframe) if key is not tuple or list
        Returns a new :obj:`label` if key is tuple or list
        """
        if type(key) is not tuple and type(key) is not list:
            assert (key in self.meta)
            return self.y[key]
        else:
            # label maybe incomplete
            return DataLoader.label({_: self.y[_]
                                for _ in key if _ in self.meta},
                                key)

    def __setitem__(self, key, value):
        r"""
        Assigns value (Tensor or Dataframe) to label if key is not tuple or list
        Successively assigns value (a List of Tensor or Dataframe) to label if key is tuple or list
        """
        if type(key) is not tuple and type(key) is not list:
            assert (key in self.meta)
            self.y[key] = value
        else:
            assert (len(key) == len(value))
            for i in len(key):
                assert (key[i] in self.meta)
                self.y[key[i]] = value[i]
    
    def to_homo(self):
        r"""
        Returns label of all labeled nodes taken as homogeneous graph
        Labels of different categories are connected into a block diagonal matrix
        """
        return torch.block_diag(*[self.y[_] for _ in self.meta])

class edgeset:
    r"""
    The edgeset of dataset. edge_index must be Tensor. edge_weight can be Dataframe or Tensor.
    Example: data.e['rating'], data.e[('user', 'movie')], data.e.to_homo(), data.e.to_homo_weighted(beta)
    """
    def __init__(self, e, w, meta, vmeta, v_num, vmap):
        self.e = e
        self.w = w
        self.meta = meta
        self.vmeta = vmeta
        self.v_num = v_num
        self.vmap = vmap

    def __getitem__(self, key):
        r"""
        Returns adjacency matrix (torch.sparse) or 2-tuple of indices and values (Tensor, Dataframe) if key is tuple or list
        Otherwise Returns a new :obj:`feature`
        (Not Recommended): if key is a 2-tuple then edge with source name = key[0] and desitination name = key[1] will be matched
        """
        if type(key) is not tuple and type(key) is not list:
            for _ in self.meta:
                if _[0] == key:
                    if type(self.w[_[0]]) is torch.Tensor:
                        return torch.sparse_coo_tensor(self.e[_[0]], self.w[_[0]], (self.v_num[_[1]], self.v_num[_[2]])).coalesce()
                    else:
                        return (self.e, self.w)
            return None

        for _ in self.meta:
            if _[1] == key[0] and _[2] == key[1]:
                if type(self.w[_[0]]) is torch.Tensor:
                    return torch.sparse_coo_tensor(self.e[_[0]], self.w[_[0]], (self.v_num[_[1]], self.v_num[_[2]])).coalesce()
                else:
                    return (self.e, self.w)

        return edgeset({_[0]: self.e[_[0]]
                            for _ in self.meta if _[0] in key},
                       {_[0]: self.w[_[0]]
                            for _ in self.meta if _[0] in key},
                       [_ for _ in self.meta if _[0] in key],
                       self.vmeta,
                       self.v_num,
                       self.vmap)

    def __setitem__(self, key, value):
        r"""
        Assigns adjancency matrix (torch.sparse) or 2-tuple of indices and values (Tensor, Dataframe) to label if key is not tuple or list
        Successively assigns value (a List) to label if key is tuple or list
        """
        if type(key) is not tuple and type(key) is not list:
            if type(value) is torch.Tensor:
                value = (value.indices(), value.values())
            self.e[key], self.w[key] = value
        else:
            assert(len(key) == len(value))
            for i in range(len(key)) and type(key) is not list:
                if type(value[i]) is torch.Tensor:
                    value[i] = (value[i].indices(), value[i].values())
                self.e[key], self.w[key] = value[i]

    def to_homo(self):
        r"""
        Returns adjancency matrix of the whole dataset taken as homogeneous graph
        """
        sum = 0
        sft = {}
        for _ in self.vmeta:
            sft[_] = sum
            sum += self.v_num[_]
        shape = (sum, sum)
        res = torch.zeros(shape).to_sparse()
        for _ in self.meta:
            e = self.e[_[0]]
            w = self.w[_[0]]
            e[0] = e[0] + sft[_[1]]
            e[1] = e[1] + sft[_[2]]
            res += torch.sparse_coo_tensor(e, w, shape)
        res = res.coalesce()
        return res
    
    def to_homo_weighted(self, beta):
        r"""
        Returns adjancency matrix of the whole dataset taken as homogeneous graph
        Different types of edges are multiplied with weight beta (Tensor)
        """
        # todo: support various edge types
        A_t = torch.stack([self[_[0]] for _ in self.meta], dim=2).to_dense()
        temp = torch.matmul(A_t, beta)
        temp = torch.squeeze(temp, 2)
        return temp + temp.transpose(0, 1)


class DataLoader:
    r""""
    Describes a dataset. Containing feature, label and edgeset.
    Example: data.x, data.y, data.e, data.normalize(), print(data)
    """
    def _I_id_mapping(n):
        return {i: i for i in range(n)}
    
    def _get_id_mapping(ids):
        mapping, cnt = {}, 0
        for _ in ids:
            x = _.item()
            assert (x not in mapping)
            mapping[x] = cnt
            cnt += 1
        return mapping
        
    # def hop_2(self, key):
    #     adj = self[key]
    #     hop = torch.sparse.mm(adj.T, adj)
    #     return torch.sparse_coo_tensor(hop.indices(), torch.ones_like(hop.values()), hop.shape)

    def __init__(self, v, vmeta, y, ymeta, e, emeta, node_index=None, edge_weight=None):
        r"""
        Create a DataLoader object with node features, metadata of nodes, labels, metadata of labels, edges, metadata of edges.
        node_index (List of index of nodes, used in ei) and edge_weight (Edge weights) are optional.
        Example: DataLoader([ufeat, mfeat], ['user', 'movie'], [label], ['movie'], [ei], [('rating', 'user', 'movie')], node_index=[uid, mid], edge_weight=[edge_weight])
        """
        assert (len(v) == len(vmeta))
        self.v_class = len(v)
        # suppose id is first row
        if node_index is None:
            self.x = feature({vmeta[i]: DataLoader._I_id_mapping(v[i].shape[0])
                             for i in range(self.v_class)},
                             {vmeta[i]: v[i] for i in range(self.v_class)},
                             vmeta)
        else:
            assert (len(v) == len(node_index))
            self.x = feature({vmeta[i]: DataLoader._get_id_mapping(node_index[i])
                             for i in range(self.v_class)},
                             {vmeta[i]: v[i] for i in range(self.v_class)},
                             vmeta)
        self.v_num = {vmeta[i]: v[i].shape[0] for i in range(self.v_class)}
        self.num_nodes = sum([v[i].shape[0] for i in range(self.v_class)])

        assert (len(y) == len(ymeta))
        self.y = label({ymeta[i]: y[i] for i in range(len(y))},
                       ymeta)

        assert (len(e) == len(emeta))
        self.e_class = len(e)
        e_tmp, w_tmp = {}, {}
        for i in range(self.e_class):
            ei = e[i]
            ev = torch.ones(ei.shape[1]) if edge_weight is None else edge_weight[i]
            # print(ei)
            _from, _to = emeta[i][1], emeta[i][2]
            # print('map', self.x.vmap[_from])
            e_tmp[emeta[i][0]] = \
                torch.LongTensor([[self.x.vmap[_from][ei[0, j].item()]
                                   for j in range(ei.shape[1])],
                                  [self.x.vmap[_to][ei[1, j].item()]
                                   for j in range(ei.shape[1])]])
            w_tmp[emeta[i][0]] = ev
        self.e = edgeset(e_tmp,
                         w_tmp,
                         emeta,
                         self.x.meta,
                         self.v_num,
                         self.x.vmap)

    def __getitem__(self, key):
        r"""
        Returns sub-dataset with only nodes of types in key and edges between them.
        """
        if (type(key) is not tuple and type(key) is not list):
            key = tuple([key])

        v = [self.x[_] for _ in key]
        vmeta = key
        y = [self.y[_] for _ in self.y.meta if _ in key]
        ymeta = [_ for _ in self.y.meta if _ in key]
        edge_index = [self.e.e[_[0]] for _ in self.e.meta if _[1] in key and _[2] in key]
        edge_weight = [self.e.w[_[0]] for _ in self.e.meta if _[1] in key and _[2] in key]
        emeta = [_ for _ in self.e.meta if _[1] in key and _[2] in key]

        return DataLoader(v, vmeta, y, ymeta, edge_index, emeta, edge_weight=edge_weight)

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
    
    def __repr__(self):
        res1 = self.__class__.__name__ + '(\n'
        res2 = '\tnum_nodes sum: ' + str(self.num_nodes) + '; '
        for _ in self.x.meta:
            res2 += '{}: {}; '.format(_, self.v_num[_])
        res2 += '\n'
        res3 = '\tfeature '
        for _ in self.x.meta:
            res3 += '{}: {} * {}; '.format(_, self.v_num[_], self.x[_].shape[1])
        res3 += '\n'
        res4 = '\tlabel '
        for _ in self.y.meta:
            if len(self.y[_].shape) > 1:
                res4 += '{}: {} * {}; '.format(_, self.v_num[_], self.y[_].shape[1])
            else:
                res4 += '{}: {} * {}; '.format(_, self.v_num[_], 1)
        res4 += '\n'
        res5 = '\tedge '
        for _ in self.e.meta:
            res5 += '({}, {} -> {}): {}; '.format(_[0], _[1], _[2], self.e[_[0]]._nnz())
        res5 += '\n'

        return res1 + res2 + res3 + res4 + res5 + ')\n'