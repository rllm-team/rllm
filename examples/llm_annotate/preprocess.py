import sys
import os.path as osp
import pandas as pd
import torch

sys.path.append("./")
sys.path.append("../")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")


def load_bi_graph(
    df: pd.DataFrame,
    src_col_name: str,
    tgt_col_name: str,
    n: int
):
    src = df[src_col_name].tolist()
    tgt = df[tgt_col_name].tolist()
    edges = [src + tgt, tgt + src]
    edges = torch.tensor(edges)
    values = torch.ones((edges.shape[1],), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edges, values, (n, n))
    return adj


def build_bi_graph(
    relation_df: pd.DataFrame,
    src_col_name: str,
    tgt_col_name: str,
    score_col_name: str,
    n: int,
    num: int
):
    topk_df = (
        relation_df
        .sort_values(score_col_name, ascending=False)
        .drop_duplicates(subset=[src_col_name, tgt_col_name])
        .groupby(src_col_name)
        .head(num)
    )

    src_to_tgts = topk_df.groupby(src_col_name)[tgt_col_name].apply(list).to_dict()

    edges = []
    for tgts in src_to_tgts.values():
        for i in range(len(tgts)):
            for j in range(i + 1, len(tgts)):
                edges.append([tgts[i], tgts[j]])
                edges.append([tgts[j], tgts[i]])

    if not edges:
        return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long),
                                       torch.empty((0,), dtype=torch.float32),
                                       (n, n))

    edges = torch.tensor(edges).T
    values = torch.ones((edges.shape[1],), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edges, values, (n, n))
    return adj


def combine_columns(df, columns, sep=', '):
    return df[columns].apply(
        lambda row: sep.join([f"{col}: {row[col]}" for col in columns]),
        axis=1
    )


class MyData():
    def __init__(self, y, adj, text, label_names):
        self.y = y
        self.num_nodes = len(self.y)
        self.adj = adj
        self.text = text
        self.label_names = label_names
        self.num_classes = len(self.label_names)


def preprocess(data_list, name):
    if name == 'tacm12k' or name == 'TACM12K':
        y = data_list[0].df['conference'].tolist()
        adj = load_bi_graph(
            df=data_list[2].df,
            src_col_name='paper_id',
            tgt_col_name='paper_id_cited',
            n=len(y),
        ).to(device)
        cols = ['title', 'abstract']
        text = combine_columns(data_list[0].df, cols)
    elif name == 'tlf2k' or name == 'TLF2K':
        y = data_list[0].df['label'].tolist()
        adj = build_bi_graph(
            relation_df=data_list[1].df,
            src_col_name='userID',
            tgt_col_name='artistID',
            score_col_name='weight',
            n=len(y),
            num=2
        ).to(device)
        cols = ['name', 'biography']
        text = combine_columns(data_list[0].df, cols)
    elif name == 'tml1m' or name == 'TML1M':
        y = data_list[0].df['Age'].tolist()
        for i in range(len(y)):
            y[i] = str(y[i])
        adj = build_bi_graph(
            relation_df=data_list[2].df,
            src_col_name='MovieID',
            tgt_col_name='UserID',
            score_col_name='Rating',
            n=len(y),
            num=10
        ).to(device)
        cols = ['FavMovies', 'Gender', 'Occupation']
        udf = add_user_favs(data_list[0].df, data_list[1].df, ratings_df=data_list[2].df, num=10)
        text = combine_columns(udf, cols)
    else:
        print('no such dataset')
        return None
    label_names = sorted(list(set(y)))
    y = torch.tensor([label_names.index(label) for label in y])

    data = MyData(y, adj, text, label_names)
    return data


def add_user_favs(users_df,
                  movies_df,
                  ratings_df,
                  num: int = 2):
    topk = (
        ratings_df
        .sort_values('Rating', ascending=False)
        .drop_duplicates(subset=['UserID', 'MovieID'])
        .groupby('UserID')
        .head(num)
    )

    topk = topk.copy()
    topk['Title'] = topk['MovieID'].map(lambda idx: movies_df.iloc[idx - 1])
    topk['Title'] = topk['Title'].apply(lambda row: row['Title'] if isinstance(row, pd.Series) else None)

    fav_titles = (
        topk
        .groupby('UserID')['Title']
        .apply(lambda titles: ' '.join(titles.dropna()))
        .to_dict()
    )

    users_df = users_df.copy()
    users_df['FavMovies'] = [fav_titles.get(user_id, "") for user_id in users_df.index]

    return users_df