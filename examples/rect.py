import argparse
import copy
import os.path as osp

import torch
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append('../')
import rllm.transforms as T
from rllm.datasets.planetoid import PlanetoidDataset
from rllm.nn.models.rect import RECT_L

# RECT focuses on the zero-shot setting,
# where only parts of classes have labeled samples.
# As such, "unseen" classes are first removed from the training set.
# Then we train a RECT (or more specifically its supervised part RECT-L) model.
# Lastly, we use the Logistic Regression model to evaluate the performance
# of the resulted embeddings based on the original balanced labels.

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--unseen-classes', type=int, nargs='*', default=[1, 2, 3])
args = parser.parse_args()

transform = T.Compose([
    T.NormalizeFeatures('l2'),
    T.SVDFeatureReduction(200),
    T.GCNNorm('loop')
])

path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
dataset = PlanetoidDataset(path, args.dataset, transform=transform, force_reload=True)
data = dataset[0]

zs_data = T.RemoveTrainingClasses(args.unseen_classes)(copy.deepcopy(data))

model = RECT_L(200, 200, dropout=0.0)
zs_data.y = model.get_semantic_labels(zs_data.x, zs_data.y, zs_data.train_mask)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model, zs_data = model.to(device), zs_data.to(device)

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()
for epoch in range(1, 201):
    optimizer.zero_grad()
    out = model(zs_data.x, zs_data.adj)
    loss = criterion(out[zs_data.train_mask], zs_data.y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch:03d}, Loss {loss:.4f}')

model.eval()
with torch.no_grad():
    h = model.embed(zs_data.x, zs_data.adj).cpu()

reg = LogisticRegression()
reg.fit(h[data.train_mask].numpy(), data.y[data.train_mask].numpy())
test_acc = reg.score(h[data.test_mask].numpy(), data.y[data.test_mask].numpy())
print(f'Test Acc: {test_acc:.4f}')
