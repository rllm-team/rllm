import sys
import types
from enum import Enum


class RelBenchTaskType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"
    MULTI_CLASSIFICATION = "multi_classification"


class RelBenchDataset:
    pass


class RelBenchTask:
    pass


fake_datasets = types.ModuleType("rllm.datasets")
fake_datasets.RelBenchDataset = RelBenchDataset
fake_datasets.RelBenchTask = RelBenchTask
fake_datasets.RelBenchTaskType = RelBenchTaskType

sys.modules["rllm.datasets"] = fake_datasets

