import os
import sys
# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Standard library imports
import torch
import torch_scatter
import torch.nn.functional as F 
import wandb
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import softmax
from graphgps.loss.custom_loss import InnerProductDecoder
from torch_geometric.utils import negative_sampling

class GraphSage(MessagePassing):
    
    def __init__(self, cfg, normalize = True,
                 bias = False, **kwargs):  
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = cfg.model.in_channels
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
                        
    def __init__(self, in_channels, out_channels, heads, negative_slope, dropout, 
                 **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.H, self.C = self.heads, self.out_channels
        assert self.C % self.H == 0, "Fatal error: hidden size must be divisible by number of heads."
        
        self.lin_l = nn.Linear(in_features=self.in_channels, out_features=self.out_channels)
        self.lin_r = self.lin_l
        self.att_l = nn.Parameter(data=torch.zeros(self.H, int(self.C/self.H)))
        self.att_r = nn.Parameter(data=torch.zeros(self.H, int(self.C/self.H)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):
        self.N = x.size(0)
        # reshape to [num_nodes, num_heads, hidden_size]
        x_l = self.lin_l(x).reshape(self.N, self.H, int(self.C/self.H))
        x_r = self.lin_r(x).reshape(self.N, self.H, int(self.C/self.H))
        # alpha vectors
        alpha_l = self.att_l * x_l
        alpha_r = self.att_r * x_r
        return self.propagate(
            edge_index=edge_index,
            x=(x_l, x_r),
            alpha=(alpha_l, alpha_r),
            size=size,
        ).reshape(-1, self.C)


    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)
        att_weight = softmax(alpha, ptr) if ptr else softmax(alpha, index)
        att_weight = F.dropout(att_weight, p=self.dropout)

        return att_weight * x_j


    def aggregate(self, inputs, index, dim_size = None):
        return torch_scatter.scatter(
            inputs, index=index, dim=self.node_dim, dim_size=dim_size, reduce='sum'
        )
    


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
        hidden_channels = cfg.model.hidden_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

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
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0)) # negative sampling
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index) + EPS).mean() # loss for negative samples

        return pos_loss + neg_loss


MAX_LOGSTD = 10  # Sets an upper limit for the logarithmic standard deviation
        

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
            
        in_channels = cfg.model.in_channels
        out_channels = cfg.model.out_channels
        hidden_channels =cfg.model.hidden_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)  # 2*out_channels because we want to output both mu and logstd
        self.conv_mu = GCNConv(hidden_channels, out_channels)  # We use 2*out_channels for the input because we want to concatenate mu and logstd
        self.conv_logstd = GCNConv(hidden_channels, out_channels)  # We use 2*out_channels for the input because we want to concatenate mu and logstd

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class VGAE(GAE): 
    """变分自编码器。继承自GAE这个类，可以使用GAE里面定义的函数。
    """
    
    def __init__(self, encoder):
        super().__init__(encoder)
        self.encoder = encoder

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, *args, **kwargs):
        """编码功能"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs) # 编码后的mu和std表示一个分布
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD) # 这里把std最大值限制一下
        return self.reparametrize(self.__mu__, self.__logstd__) # 进行reparametrization，这样才能够训练模型

    def kl_loss(self, mu=None, logstd=None):
        """我们给隐变量的分布加上（0，I）高斯变量的先验，即希望隐变量分布服从（0，I）的高斯分布
        这两个分布的差别用KL损失来衡量。"""
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1)) # 两个高斯分布之间的KL损失


def create_model(cfg):
    if cfg.model.type == 'GAT':
        model = GAE(encoder=GAT(cfg))
    elif cfg.model.type == 'GraphSage':
        model = GAE(encoder=GraphSage(cfg))
    elif cfg.model.type == 'GAE':
        model = GAE(encoder = GCNEncoder(cfg) )
    elif cfg.model.type == 'VGAE':
        model = VGAE(encoder= VariationalGCNEncoder(cfg))
    else:
        # Without this else I got: UnboundLocalError: local variable 'model' referenced before assignment
        raise ValueError('Current model does not exist')
    model.to(cfg.device)
    return model 