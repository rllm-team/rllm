
def load_data(dataname):
    if (dataname == 'movielens-regression'):
        from movielens_regression import load
        return load()
    
    if (dataname == 'movielens-classification'):
        from movielens_classification import load
        return load()
    
    if (dataname == 'small_alibaba_1_10'):    
        from alibaba import load
        return load()
    
    if (dataname == 'cora'):
        from cora import load
        return load(dataname)
    
    if (dataname == 'titanic'):
        from titanic import load
        return load()
