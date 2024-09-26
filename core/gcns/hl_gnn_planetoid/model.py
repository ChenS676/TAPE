from torch_sparse import matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
    
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
import numpy as np

class HLGNN(torch.nn.Module):
    def __init__(self, data, args):
        super(HLGNN, self).__init__()
        self.K = args.K
        self.init = args.init
        self.alpha = args.alpha
        self.dropout = args.dropout
        self.norm_func = globals()[args.norm_func]
        self.lin1 = Linear(data.num_features, data.num_features)

        assert self.init in ['SGC', 'RWR', 'KI', 'Random']
        if self.init == 'SGC':
            alpha = int(self.alpha)
            TEMP = 0.0 * np.ones(self.K+1)
            TEMP[alpha] = 1.0
        elif self.init == 'RWR':
            TEMP = (1 - self.alpha) * self.alpha ** np.arange(self.K+1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif self.init == 'KI':
            TEMP = self.alpha ** np.arange(self.K+1)
        elif self.init == 'Random':
            bound = np.sqrt(3 / (self.K+1))
            TEMP = np.random.uniform(-bound, bound, self.K+1)
            TEMP = TEMP / np.sum(np.abs(TEMP))

        self.temp = Parameter(torch.tensor(TEMP, dtype=torch.float))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.init == 'SGC':
            self.alpha = int(self.alpha)
            self.temp.data[self.alpha] = 1.0
        elif self.init == 'RWR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha * (1-self.alpha) ** k
            self.temp.data[-1] = (1-self.alpha) ** self.K
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.init == 'KI':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha ** k
        elif self.init == 'Random':
            bound = np.sqrt(3 / (self.K+1))
            torch.nn.init.uniform_(self.temp, -bound, bound)
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))

    def forward(self, x, adj_t, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        adj_t = self.norm_func(adj_t, edge_weight, adj_t.size(0), dtype=torch.float)

        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(adj_t, x=x, edge_weight=edge_weight)
            gamma = self.temp[k+1]
            hidden = hidden + gamma * x
        return hidden

    def propagate(self, adj_t, x, edge_weight=None):
        return matmul(adj_t, x)


class GCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.lin = Linear(in_channels, out_channels)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = self.lin(out)
        return out

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNLayer(hidden_channels, hidden_channels))
        self.convs.append(GCNLayer(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        for conv in self.convs[:-1]:
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x


class SAGELayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGELayer, self).__init__()
        self.lin = Linear(in_channels, out_channels)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = self.lin(out)
        return out

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGELayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGELayer(hidden_channels, hidden_channels))
        self.convs.append(SAGELayer(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        for conv in self.convs[:-1]:
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)