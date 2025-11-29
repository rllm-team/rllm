from rllm.datasets.relbench.f1 import RelF1Dataset


ds = RelF1Dataset(cached_dir='./data/')
print(len(ds))
print(ds.table_dict.keys())
print(ds.task_dict.keys())