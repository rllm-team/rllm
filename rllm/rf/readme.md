# Relation Frame

- RDB -> `pandas`
- batch from raw `pandas` # 期望绕过dataset这一步
- construct batch from raw `pandas`, called `relation frame`

- transform `relation frame`

- `relation frame` -> model forward


# Hypothesis
1. 所有表都有单独的index列作为 pkey

# output
输出应该是sampled table data && sampled edge list (BRIGE do not need entire graph, but only adj, i.e. edge list here for Massge Passing)

# TODO:
- tml1m里面的UserID是从1开始的