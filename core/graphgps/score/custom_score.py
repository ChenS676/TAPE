import torch
import torch.nn.functional as F
    
    
class DotProduct(torch.nn.Module):
    """解码器，用向量内积表示重建的图结构"""
    
    def forward(self, x, y):
        """
        参数说明：
        z: 节点表示
        edge_index: 边索引，也就是节点对
        """
        return x * y
    
    

class InnerProduct(torch.nn.Module):
    """解码器，用向量内积表示重建的图结构"""
    
    def forward(self, x, y, sigmoid=True):
        """
        参数说明：
        z: 节点表示
        edge_index: 边索引，也就是节点对
        """
        # print(edge_index[0])
        # print(z)
        # raise(0)
        value = (x * y).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


product_dict = {'inner': InnerProduct(), 'dot': DotProduct()}

class mlp_score(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers,
                 dropout,
                 product):
        super(mlp_score, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.product = product_dict[product]
        
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, h1, h2):
        x = self.product(h1, h2) 

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    
import torch

import torch
import torch.nn as nn

class EuclideanDistance(nn.Module):
    def forward(self, x, y):
        return torch.sqrt(torch.sum((x - y) ** 2))

class ManhattanDistance(nn.Module):
    def forward(self, x, y):
        return torch.sum(torch.abs(x - y))

class ChebyshevDistance(nn.Module):
    def forward(self, x, y):
        return torch.max(torch.abs(x - y))

class MinkowskiDistance(nn.Module):
    def __init__(self, p):
        super(MinkowskiDistance, self).__init__()
        self.p = p

    def forward(self, x, y):
        return torch.sum(torch.abs(x - y) ** self.p) ** (1 / self.p)

class CosineSimilarity(nn.Module):
    def forward(self, x, y):
        return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

class HammingDistance(nn.Module):
    def forward(self, x, y):
        return torch.sum(x != y)

class MahalanobisDistance(nn.Module):
    def __init__(self, cov_matrix):
        super(MahalanobisDistance, self).__init__()
        self.cov_matrix = cov_matrix
        self.inv_cov_matrix = torch.inverse(cov_matrix)

    def forward(self, x, y):
        diff = x - y
        return torch.sqrt(torch.dot(torch.dot(diff, self.inv_cov_matrix), diff))

class JaccardSimilarity(nn.Module):
    def forward(self, x, y):
        intersection = torch.sum(torch.min(x, y))
        union = torch.sum(torch.max(x, y))
        return intersection / union