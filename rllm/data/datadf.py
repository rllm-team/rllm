def _get_id_mapping(ids):
    r"""
    Create an index mapping from index List `ids`.
    """
    mapping, cnt = {}, 0
    for _ in ids:
        assert (_ not in mapping)
        mapping[_] = cnt
        cnt += 1
    return mapping

def _I_id_mapping(n):
    r"""
    Create an identical index mapping. (`ids`=[0, 1, 2,...])
    """
    return {i: i for i in range(n)}

class feature:
    r"""
    Representing the feature / label of dataset, in the form of Dataframe.

    Args:
        x (Dict[metadata, Dataframe]): feature of each node class.
    """

    def __init__(self, x):
        self.x = x
        self.meta = list(x.keys())
        self.v_num = {_: x[_].shape[0] for _ in self.meta}

    def __getitem__(self, key):
        r"""
        Returns feature (Dataframe) if key is not List.
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
        Assigns value (Dataframe) to feature if key is list.
        Successively assigns value (a List of Dataframe) to feature if key is list.
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

class edgeset:
    r"""
    Representing the edgeset of dataset, in the form of torch.Dataframe.
    
    Args:
        e (Dict[metadata, Dataframe]): Edge index of each edge class.
        w (Dict[metadata, Dataframe]): Edge weight of each edge class.
        meta (List[metadata]): List of metadata of edges.
        v_num (Dict[metadata, Int]): Node numbers of each node class.
    """
    def __init__(self, e, w, meta):
        self.e, self.w = {}, {}
        for i in range(len(meta)):
            _ = meta[i]
            self.e[_[0]] = e[_[0]]
            self.w[_[0]] = w[_[0]]
        self.meta = meta

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
        Returns 2-tuple (edge_index, edge_attr) if key is not list or a new :class:`edgeset` if key is list.
        `key` or `key[i]` can be: 1) edge metadata Edge_key, 2) 2-tuple (From_key, To_key) or 3) 3-tuple (Edge_key, From_key, To_key).
        Note that 2) is not recommended, especially when different edge classes have same (From_key, To_key).

        .. code-block:: python

            edge_index, edge_attr = datatensor.e['rating']
            edge_index, edge_attr = datatensor.e[('user', 'movie')]
            edge_index, edge_attr = datatensor.e[('rating', 'user', 'movie')]
            new_e = datatensor.e[['rating1', 'rating2']]
        """
        if type(key) is not list:
            return self.e[self._getkey_single(key)],\
                   self.w[self._getkey_single(key)]
        if type(key) is list:
            keys = [self._getkey_single(_) for _ in key]
            return edgeset({_: self.e[_].indices() for _ in keys},
                        {_: self.e[_].values() for _ in keys},
                        keys)

    def __setitem__(self, key, value):
        r"""
        Assigns 2-tuple (edge_index, edge_attr) or a list of 2-tuples to key.
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
            self.e[_] = value[0]
            self.w[_] = value[1]
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
                self.e[_] = value[i][0]
                self.w[_] = value[i][1]

class GraphStore:
    r""""
    Describes a dataset. Containing feature, label and edgeset.
    Example: datatensor.x, datatensor.y, datatensor.e, datatensor.normalize(), print(data)
    """
    def __init__(self):
        self.x = feature({})
        self.y = feature({})
        self.e = edgeset({}, {}, [])

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