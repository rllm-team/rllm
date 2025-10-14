Table Data Handle
==================

Data Handling of Tables
-----------------------

A table contains many different columns with many different types. Each column type in rLLM is described by a certain semantic type, i.e., ColType. rLLM supports two basic column types so far:

- :obj:`ColType.CATEGORICAL`: represent categorical or discrete data, such as grade levels in a student dataset and diabetes types in a diabetes dataset.
- :obj:`ColType.NUMERICAL`: represent numerical or continuous data, such as such as temperature in a weather dataset and income in a salary dataset.

A table in rLLM is described by an instance of :class:`~rllm.data.table_data.TableData` with many default attributes:

- :obj:`df`: A `pandas.DataFrame`_ stores raw tabular data.
- :obj:`col_types`: A dictionary indicating :class:`~rllm.types.ColType` of each column.
- :obj:`target_col` (optional): A string indicating target column for certain task.
- :obj:`feat_dict`: A dictionary stores the `Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`__ of different :class:`~rllm.types.ColType`. Each column in tensor represents a column in the DataFrame :obj:`df`, corresponding to the column order in :obj:`col_types`.
- :obj:`y` (optional): A tensor containing the target values for certain task.
- :obj:`metadata`: Once a :class:`~rllm.data.table_data.TableData` is instantiated, each column will compute a set of statistics based on its :class:`~rllm.types.ColType`.

- For categorical features:

  - :obj:`COUNT`: Indicating category count.

  - :obj:`MOST_FREQUENT`: Indicating the most frequent category.

- For numerical features:

  - :obj:`MEAN`: Indicating the mean value of the column.

  - :obj:`MAX`: Indicating the max value of the column.

  - :obj:`MIN`: Indicating the min value of the column.

  - :obj:`STD`: Indicating the standard deviation of the column.

  - :obj:`QUANTILES`: Indicating a list containing minimum value, first quartile (25th percentile), median (50th percentile), third quartile (75th percentile) and maximum value of the column.

.. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/dev/reference/api/pandas.DataFrame.html#pandas.DataFrame


.. code-block:: python

    from rllm.types import ColType
    from rllm.datasets.titanic import Titanic

    dataset = Titanic('data', forced_reload=True)[0]

    print(dataset.col_types)
    >>> {'Survived': <ColType.CATEGORICAL: 'categorical'>, 'Pclass': <ColType.CATEGORICAL: 'categorical'>, ..., 'Embarked': <ColType.CATEGORICAL: 'categorical'>}

    print(dataset.feat_dict.keys())
    >>> dict_keys([<ColType.CATEGORICAL: 'categorical'>, <ColType.NUMERICAL: 'numerical'>])

    print(dataset.feat_dict[ColType.NUMERICAL])
    >>> tensor([[22.0000,  1.0000,  0.0000,  7.2500],
                [38.0000,  1.0000,  0.0000, 71.2833],
                [26.0000,  0.0000,  0.0000,  7.9250],
                ...,
                [    nan,  1.0000,  2.0000, 23.4500],
                [26.0000,  0.0000,  0.0000, 30.0000],
                [32.0000,  0.0000,  0.0000,  7.7500]])

    print(dataset.feat_dict[ColType.CATEGORICAL])
    >>> tensor([[0, 0, 0],
                [1, 1, 1],
                [0, 1, 0],
                ...,
                [0, 1, 0],
                [1, 0, 1],
                [0, 0, 2]])

    print(dataset.y)
    >>> tensor([0, 1, 1,  ..., 0, 1, 0])

    print(dataset.metadata[ColType.CATEGORICAL][0])
    >>> {<StatType.COUNT: 'COUNT'>: 3, <StatType.MOST_FREQUENT: 'MOST_FREQUENT'>: 2, <StatType.COLNAME: 'COLNAME'>: 'Pclass'}

    print(dataset.metadata[ColType.NUMERICAL][0])
    >>> {<StatType.MEAN: 'MEAN'>: 29.69911766052246, <StatType.MAX: 'MAX'>: 80.0, <StatType.MIN: 'MIN'>: 0.41999998688697815, <StatType.STD: 'STD'>: 14.526496887207031, <StatType.QUANTILES: 'QUANTILES'>: [0.41999998688697815, 20.125, 28.0, 38.0, 80.0], <StatType.COLNAME: 'COLNAME'>: 'Age'}

An instance of :class:`~rllm.data.table_data.TableData` also includes several basic properties:

.. code-block:: python

    print(dataset.num_cols)
    >>> 7

    print(dataset.num_rows)
    >>> 891

    print(dataset.num_classes)
    >>> 2

We support transferring the data in a :class:`~rllm.data.table_data.TableData` instance to devices supported by PyTorch.

.. code-block:: python

    dataset.to("cpu")

    dataset.to("cuda")


Common Benchmark Datasets (Table Part)
---------------------------------------

rLLM contains a large number of common benchmark datasets.
A complete list of available datasets can be found in :mod:`~rllm.datasets`.
The collection consists of both graph and tabular datasets, with tabular data being used for demonstration purposes here.

Initializing tabular datasets is straightforward in rLLM.
Upon initialization, the dataset automatically downloads its raw files and processes the columns.

In the following example, we will use one of the pre-loaded datasets, which contains data on Titanic passengers.

.. code-block:: python

    from rllm.datasets import Titanic

    dataset = Titanic('data', forced_reload=True)[0]

    print(len(dataset))
    >>> 891

    print(dataset.feat_cols)
    >>> ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    print(dataset.df.head(5))
    >>> PassengerId   Survived   Pclass    ...      Cabin     Embarked
        1                    0        3    ...        NaN            S
        2                    1        1    ...        C85            C
        3                    1        3    ...        NaN            S
        4                    1        1    ...       C123            S
        5                    0        3    ...        NaN            S

        [5 rows x 11 columns]

rLLM also supports custom datasets, enabling the use of rLLM for specific problems. 
Suppose a `pandas.DataFrame`_ named :obj:`df` is prepared with five columns: :obj:`cat1`, :obj:`cat2`, :obj:`num1`, :obj:`num2`, and :obj:`y`.
Creating a :class:`~rllm.data.table_data.TableData` object is very easy.

.. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/dev/reference/api/pandas.DataFrame.html#pandas.DataFrame

.. code-block:: python

    from rllm.types import ColType
    from rllm.data.TableData import TableData

    # Specify the coltype of each column with a dictionary.
    col_types = {
        "cat1": ColType.CATEGORICAL,
        "cat2": ColType.CATEGORICAL,
        "num1": ColType.NUMERICAL,
        "num2": ColType.NUMERICAL,
        "y": ColType.CATEGORICAL,
    }

    # Set "y" as the target column.
    dataset = TableData(df, col_types=col_types, target_col="y")
