import torch


def read_interactions(file_path, to_type=int):
    user_ratings = dict()
    with open(file_path, 'r') as file:
        for line in file:
            parts = list(map(to_type, line.split()))
            if to_type == int:
                user_ratings[parts[0]] = torch.tensor(parts[1:]).long()
            elif to_type == float:
                user_ratings[parts[0]] = torch.tensor(parts[1:])
            else:
                raise ValueError()
    return user_ratings
