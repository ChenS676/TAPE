import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

x = torch.randn(8, 32)  # Node features of shape [num_nodes, num_features]
y = torch.randint(0, 4, (8, ))  # Node labels of shape [num_nodes]
edge_index = torch.tensor([
    [2, 3, 3, 4, 5, 6, 7],
    [0, 0, 1, 1, 2, 3, 4]],
)

#   0  1
#  / \/ \
# 2  3  4
# |  |  |
# 5  6  7

data = Data(x=x, y=y, edge_index=edge_index, node_index=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))

loader = NeighborLoader(
    data,
    input_nodes=torch.tensor([0, 1]),
    num_neighbors=[2] * 2,
    batch_size=1,
    replace=False,
    shuffle=False,
)

batch = next(iter(loader))

print(batch.edge_index)
print(batch.n_id)
for batch in loader:
    print(batch)