import os
import sys
# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Standard library imports
import torch
import torch_scatter
import torch_geometric

import torch.nn.functional as F 
from sklearn.metrics import *
import wandb
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import softmax
from graphgps.loss.custom_loss import InnerProductDecoder

class GraphSage(MessagePassing):
    
    def __init__(self, cfg, normalize = True,
                 bias = False, **kwargs):  
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = cfg.data.num_features
        self.out_channels = cfg.model.out_channels
        self.normalize = normalize

        self.lin_l = nn.Linear(in_features=self.in_channels, out_features=self.out_channels)
        self.lin_r = nn.Linear(in_features=self.in_channels, out_features=self.out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size = None):
        """"""
        out = self.propagate(edge_index=edge_index, x=(x, x), size=size)
        out = self.lin_l(x) + self.lin_r(out)
        if self.normalize:
            out = F.normalize(out, p=2)
        return out

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size = None):
        # The axis along which to index number of nodes.
        node_dim = self.node_dim
        return torch_scatter.scatter(
            inputs, index=index, dim=node_dim, dim_size=dim_size, reduce='mean'
        )


class GAT(MessagePassing):
                        
    def __init__(self, cfg, **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = cfg.data.num_features
        self.out_channels = cfg.model.out_channels
        self.heads = cfg.model.heads
        self.negative_slope = cfg.model.negative_slope
        self.dropout = cfg.model.dropout
        
        self.lin_l = nn.Linear(in_features=self.in_channels, out_features=self.out_channels)
        self.lin_r = self.lin_l
        self.att_l = nn.Parameter(data=torch.zeros(self.heads, self.out_channels))
        self.att_r = nn.Parameter(data=torch.zeros(self.heads, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):
        
        H, C = self.heads, self.out_channels
        # reshape to [num_nodes, num_heads, hidden_size]
        x_l = self.lin_l(x).reshape(-1, H, C)
        x_r = self.lin_r(x).reshape(-1, H, C)
        # alpha vectors
        alpha_l = self.att_l * x_l
        alpha_r = self.att_r * x_r
        return self.propagate(
            edge_index=edge_index,
            x=(x_l, x_r),
            alpha=(alpha_l, alpha_r),
            size=size,
        ).reshape(-1, H * C)


    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)
        att_weight = softmax(alpha, ptr) if ptr else softmax(alpha, index)
        att_weight = F.dropout(att_weight, p=self.dropout)

        return att_weight * x_j


    def aggregate(self, inputs, index, dim_size = None):
        return torch_scatter.scatter(
            inputs, index=index, dim=self.node_dim, dim_size=dim_size, reduce='sum'
        )
    

class LinkPredModel(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super(LinkPredModel, self).__init__()

        self.encoder = encoder
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.decoder = decoder
        

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """In this script we use the binary cross entropy loss function.

        params
        ----
        z: output of encoder
        pos_edge_index: positive edge index
        neg_edge_index: negative edge index
        """
        EPS = 1e-15
        
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean() # loss for positive samples

        if neg_edge_index is None:
            neg_edge_index = torch_geometric.utils.negative_sampling(pos_edge_index, z.size(0)) # negative sampling
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index) + EPS).mean() # loss for negative samples

        return pos_loss + neg_loss


class GCNEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        """GCN encoder with 2 layers
        in_channels: number of input features of node (dimension of node feature)
        hidden_channels: number of hidden features of node (dimension of hidden layer)
        out_channels: number of output features of node (dimension of node embedding)
        """
        in_channels = cfg.model.in_channels
        out_channels = cfg.model.out_channels
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


    
class GAE(torch.nn.Module):
    """graph auto encoder。
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder()

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """In this script we use the binary cross entropy loss function.

        params
        ----
        z: output of encoder
        pos_edge_index: positive edge index
        neg_edge_index: negative edge index
        """
        EPS = 1e-15 # prevent log(0)
        # positive loss log()
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean() # loss for positive samples

        if neg_edge_index is None:
            neg_edge_index = torch_geometric.utils.negative_sampling(pos_edge_index, z.size(0)) # negative sampling
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index) + EPS).mean() # loss for negative samples

        return pos_loss + neg_loss


MAX_LOGSTD = 10  # Sets an upper limit for the logarithmic standard deviation
        

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
            
        in_channels = cfg.data.num_features
        out_channels = cfg.model.out_channels
        
        self.conv1 = GCNConv(in_channels, 2 * out_channels)  # 2*out_channels because we want to output both mu and logstd
        self.conv_mu = GCNConv(2 * out_channels, out_channels)  # We use 2*out_channels for the input because we want to concatenate mu and logstd
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)  # We use 2*out_channels for the input because we want to concatenate mu and logstd

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
class VGAE(GAE):
    """inhert GAE class, since we need to use encode, decode and loss
    """

    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = InnerProductDecoder()

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """encoder"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs) # mu stad stands for distribution for mean and std
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD) # upper bound of logstd
        return self.reparametrize(self.__mu__, self.__logstd__)

    def kl_loss(self, mu=None, logstd=None):
        """We add a prior of (0, I) Gaussian variables to the distribution of the hidden variables,
        i.e., we want the distribution of the hidden variables to obey a (0, I) Gaussian distribution
        The difference between these two distributions is measured by the KL loss."""
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1)) # KL loss between gaussian distribution and hidden variables



def create_model(cfg):
    if cfg.model.type == 'GAT':
        model = LinkPredModel(encoder=GAT(cfg),
                              decoder=InnerProductDecoder())
    elif cfg.model.type == 'GraphSage':
        model = LinkPredModel(encoder=GraphSage(cfg),
                              decoder=InnerProductDecoder())
    elif cfg.model.type == 'GCNEncode':
        model = LinkPredModel(encoder=GCNEncoder(cfg),
                              decoder=InnerProductDecoder())
    
    if cfg.model.type == 'GAE':
        model = GAE(encoder = GCNEncoder(cfg) )
    elif cfg.model.type == 'VGAE':
        model = VGAE(encoder= VariationalGCNEncoder(cfg),
                     decoder=InnerProductDecoder())
    else:
        # Without this else I got: UnboundLocalError: local variable 'model' referenced before assignment
        raise ValueError('Current model does not exist')

    return model 