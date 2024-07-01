"""
The ELPH model
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from torch.nn import Linear, Parameter, BatchNorm1d as BN
from torch_sparse import SparseTensor
from time import time
import logging

from tqdm import tqdm
import torch
from torch import float
import numpy as np
from pandas.util import hash_array
from datasketch import HyperLogLogPlusPlus, hyperloglog_const
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import DataLoader
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LinkPredictor(torch.nn.Module):
    def __init__(self, args, use_embedding=False):
        super(LinkPredictor, self).__init__()
        self.use_embedding = use_embedding
        self.use_feature = args.use_feature
        self.feature_dropout = args.feature_dropout
        self.label_dropout = args.label_dropout
        self.dim = args.max_hash_hops * (args.max_hash_hops + 2)
        self.label_lin_layer = Linear(self.dim, self.dim)
        if args.use_feature:
            self.bn_feats = torch.nn.BatchNorm1d(args.hidden_channels)
        if self.use_embedding:
            self.bn_embs = torch.nn.BatchNorm1d(args.hidden_channels)
        self.bn_labels = torch.nn.BatchNorm1d(self.dim)
        if args.use_feature:
            self.lin_feat = Linear(args.hidden_channels,
                                   args.hidden_channels)
            self.lin_out = Linear(args.hidden_channels, args.hidden_channels)
        out_channels = self.dim + args.hidden_channels if self.use_feature else self.dim
        if self.use_embedding:
            self.lin_emb = Linear(args.hidden_channels,
                                  args.hidden_channels)
            self.lin_emb_out = Linear(args.hidden_channels, args.hidden_channels)
            out_channels += args.hidden_channels
        self.lin = Linear(out_channels, 1)

    def feature_forward(self, x):
        """
        small neural network applied edgewise to hadamard product of node features
        @param x: node features torch tensor [batch_size, 2, hidden_dim]
        @return: torch tensor [batch_size, hidden_dim]
        """
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_out(x)
        x = self.bn_feats(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        return x

    def embedding_forward(self, x):
        x = self.lin_emb(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_emb_out(x)
        x = self.bn_embs(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)

        return x

    def forward(self, sf, node_features, emb=None):
        sf = self.label_lin_layer(sf)
        sf = self.bn_labels(sf)
        sf = F.relu(sf)
        x = F.dropout(sf, p=self.label_dropout, training=self.training)
        # process node features
        if self.use_feature:
            node_features = self.feature_forward(node_features)
            x = torch.cat([x, node_features.to(torch.float)], 1)
        if emb is not None:
            node_embedding = self.embedding_forward(emb)
            x = torch.cat([x, node_embedding.to(torch.float)], 1)
        x = self.lin(x)
        return x

    def print_params(self):
        print(f'model bias: {self.lin.bias.item():.3f}')
        print('model weights')
        for idx, elem in enumerate(self.lin.weight.squeeze()):
            if idx < self.dim:
                print(f'{self.idx_to_dst[idx % self.emb_dim]}: {elem.item():.3f}')
            else:
                print(f'feature {idx - self.dim}: {elem.item():.3f}')


class ELPH(torch.nn.Module):
    """
    propagating hashes, features and degrees with message passing
    """

    def __init__(self, args, num_features, node_embedding=None):
        super(ELPH, self).__init__()
        # hashing things
        self.elph_hashes = ElphHashes(args)
        self.init_hashes = None
        self.init_hll = None
        self.num_perm = args.minhash_num_perm
        self.hll_size = 2 ^ args.hll_p
        # gnn things
        self.use_feature = args.use_feature
        self.feature_prop = args.feature_prop  # None, residual, cat
        self.node_embedding = node_embedding
        self.propagate_embeddings = args.propagate_embeddings
        self.sign_k = args.sign_k
        self.label_dropout = args.label_dropout
        self.feature_dropout = args.feature_dropout
        self.num_layers = args.max_hash_hops
        self.dim = args.max_hash_hops * (args.max_hash_hops + 2)
        # construct the nodewise NN components
        self._convolution_builder(num_features, args.hidden_channels,
                                  args)  # build the convolutions for features and embs
        # construct the edgewise NN components
        self.predictor = LinkPredictor(args, node_embedding is not None)
        if self.sign_k != 0 and self.propagate_embeddings:
            # only used for the ddi where nodes have no features and transductive node embeddings are needed
            self.sign_embedding = SIGNEmbedding(args.hidden_channels, args.hidden_channels, args.hidden_channels,
                                                args.sign_k, args.sign_dropout)

    def _convolution_builder(self, num_features, hidden_channels, args):
        self.convs = torch.nn.ModuleList()
        if args.feature_prop in {'residual', 'cat'}:  # use a linear encoder
            self.feature_encoder = Linear(num_features, hidden_channels)
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        else:
            self.convs.append(
                GCNConv(num_features, hidden_channels))
        for _ in range(self.num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        if self.node_embedding is not None:
            self.emb_convs = torch.nn.ModuleList()
            for _ in range(self.num_layers):  # assuming the embedding has hidden_channels dims
                self.emb_convs.append(GCNConv(hidden_channels, hidden_channels))

    def propagate_embeddings_func(self, edge_index):
        num_nodes = self.node_embedding.num_embeddings
        gcn_edge_index, _ = gcn_norm(edge_index, num_nodes=num_nodes)
        return self.sign_embedding(self.node_embedding.weight, gcn_edge_index, num_nodes)

    def feature_conv(self, x, edge_index, k):
        if not self.use_feature:
            return None
        out = self.convs[k - 1](x, edge_index)
        out = F.dropout(out, p=self.feature_dropout, training=self.training)
        if self.feature_prop == 'residual':
            out = x + out
        return out

    def embedding_conv(self, x, edge_index, k):
        if x is None:
            return x
        out = self.emb_convs[k - 1](x, edge_index)
        out = F.dropout(out, p=self.feature_dropout, training=self.training)
        if self.feature_prop == 'residual':
            out = x + out
        return out

    def _encode_features(self, x):
        if self.use_feature:
            x = self.feature_encoder(x)
            x = F.dropout(x, p=self.feature_dropout, training=self.training)
        else:
            x = None

        return x

    def forward(self, x, edge_index):
        """
        @param x: raw node features tensor [n_nodes, n_features]
        @param adj_t: edge index tensor [2, num_links]
        @return:
        """
        hash_edge_index, _ = add_self_loops(edge_index)  # unnormalised, but with self-loops
        # if this is the first call then initialise the minhashes and hlls - these need to be the same for every model call
        num_nodes, num_features = x.shape
        if self.init_hashes == None:
            self.init_hashes = self.elph_hashes.initialise_minhash(num_nodes).to(x.device)
        if self.init_hll == None:
            self.init_hll = self.elph_hashes.initialise_hll(num_nodes).to(x.device)
        # initialise data tensors for storing k-hop hashes
        cards = torch.zeros((num_nodes, self.num_layers))
        node_hashings_table = {}
        for k in range(self.num_layers + 1):
            '''logger.info(f"Calculating hop {k} hashes")'''
            node_hashings_table[k] = {
                'hll': torch.zeros((num_nodes, self.hll_size), dtype=torch.int8, device=edge_index.device),
                'minhash': torch.zeros((num_nodes, self.num_perm), dtype=torch.int64, device=edge_index.device)}
            start = time()
            if k == 0:
                node_hashings_table[k]['minhash'] = self.init_hashes
                node_hashings_table[k]['hll'] = self.init_hll
                if self.feature_prop in {'residual', 'cat'}:  # need to get features to the hidden dim
                    x = self._encode_features(x)

            else:
                node_hashings_table[k]['hll'] = self.elph_hashes.hll_prop(node_hashings_table[k - 1]['hll'],
                                                                          hash_edge_index)
                node_hashings_table[k]['minhash'] = self.elph_hashes.minhash_prop(node_hashings_table[k - 1]['minhash'],
                                                                                  hash_edge_index)
                cards[:, k - 1] = self.elph_hashes.hll_count(node_hashings_table[k]['hll'])
                x = self.feature_conv(x, edge_index, k)

            '''logger.info(f'{k} hop hash generation ran in {time() - start} s')'''

        return x, node_hashings_table, cards


class BUDDY(torch.nn.Module):
    """
    Scalable version of ElPH that uses precomputation of subgraph features and SIGN style propagation
    of node features
    """

    def __init__(self, args, num_features=None, node_embedding=None):
        super(BUDDY, self).__init__()

        self.use_feature = args.use_feature
        self.label_dropout = args.label_dropout
        self.feature_dropout = args.feature_dropout
        self.node_embedding = node_embedding
        self.propagate_embeddings = args.propagate_embeddings
        # using both unormalised and degree normalised counts as features, hence * 2
        self.append_normalised = args.add_normed_features
        ra_counter = 1 if args.use_RA else 0
        num_labelling_features = args.max_hash_hops * (args.max_hash_hops + 2)
        self.dim = num_labelling_features * 2 if self.append_normalised else num_labelling_features
        self.use_RA = args.use_RA
        self.sign_k = args.sign_k
        if self.sign_k != 0:
            if self.propagate_embeddings:
                # this is only used for the ddi dataset where nodes have no features and transductive node embeddings are needed
                self.sign_embedding = SIGNEmbedding(args.hidden_channels, args.hidden_channels, args.hidden_channels,
                                                    args.sign_k, args.sign_dropout)
            else:
                self.sign = SIGN(num_features, args.hidden_channels, args.hidden_channels, args.sign_k,
                                 args.sign_dropout)
        self.label_lin_layer = Linear(self.dim, self.dim)
        if args.use_feature:
            self.bn_feats = torch.nn.BatchNorm1d(args.hidden_channels)
        if self.node_embedding is not None:
            self.bn_embs = torch.nn.BatchNorm1d(args.hidden_channels)
        self.bn_labels = torch.nn.BatchNorm1d(self.dim)
        self.bn_RA = torch.nn.BatchNorm1d(1)

        if args.use_feature:
            self.lin_feat = Linear(num_features,
                                   args.hidden_channels)
            self.lin_out = Linear(args.hidden_channels, args.hidden_channels)
        hidden_channels = self.dim + args.hidden_channels if self.use_feature else self.dim
        if self.node_embedding is not None:
            self.lin_emb = Linear(args.hidden_channels,
                                  args.hidden_channels)
            self.lin_emb_out = Linear(args.hidden_channels, args.hidden_channels)
            hidden_channels += self.node_embedding.embedding_dim

        self.lin = Linear(hidden_channels + ra_counter, 1)

    def propagate_embeddings_func(self, edge_index):
        num_nodes = self.node_embedding.num_embeddings
        gcn_edge_index, _ = gcn_norm(edge_index, num_nodes=num_nodes)
        return self.sign_embedding(self.node_embedding.weight, gcn_edge_index, num_nodes)

    def _append_degree_normalised(self, x, src_degree, dst_degree):
        """
        Create a set of features that have the spirit of a cosine similarity x.y / ||x||.||y||. Some nodes (particularly negative samples)
        have zero degree
        because part of the graph is held back as train / val supervision edges and so divide by zero needs to be handled.
        Note that we always divide by the src and dst node's degrees for every node in ¬the subgraph
        @param x: unormalised features - equivalent to x.y
        @param src_degree: equivalent to sum_i x_i^2 as x_i in (0,1)
        @param dst_degree: equivalent to sum_i y_i^2 as y_i in (0,1)
        @return:
        """
        # this doesn't quite work with edge weights as instead of degree, the sum_i w_i^2 is needed,
        # but it probably doesn't matter
        normaliser = torch.sqrt(src_degree * dst_degree)
        normed_x = torch.divide(x, normaliser.unsqueeze(dim=1))
        normed_x[torch.isnan(normed_x)] = 0
        normed_x[torch.isinf(normed_x)] = 0
        return torch.cat([x, normed_x], dim=1)

    def feature_forward(self, x):
        """
        small neural network applied edgewise to hadamard product of node features
        @param x: node features torch tensor [batch_size, 2, hidden_dim]
        @return: torch tensor [batch_size, hidden_dim]
        """
        if self.sign_k != 0:
            x = self.sign(x)
        else:
            x = self.lin_feat(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_out(x)
        x = self.bn_feats(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        return x

    def embedding_forward(self, x):
        x = self.lin_emb(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_emb_out(x)
        x = self.bn_embs(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)

        return x

    def forward(self, sf, node_features, src_degree=None, dst_degree=None, RA=None, emb=None):
        """
        forward pass for one batch of edges
        @param sf: subgraph features [batch_size, num_hops*(num_hops+2)]
        @param node_features: raw node features [batch_size, 2, num_features]
        @param src_degree: degree of source nodes in batch
        @param dst_degree:
        @param RA:
        @param emb:
        @return:
        """
        if self.append_normalised:
            sf = self._append_degree_normalised(sf, src_degree, dst_degree)
        x = self.label_lin_layer(sf)
        x = self.bn_labels(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.label_dropout, training=self.training)
        if self.use_feature:
            node_features = self.feature_forward(node_features)
            x = torch.cat([x, node_features.to(torch.float)], 1)
        if self.node_embedding is not None:
            node_embedding = self.embedding_forward(emb)
            x = torch.cat([x, node_embedding.to(torch.float)], 1)
        if self.use_RA:
            RA = RA.unsqueeze(-1)
            RA = self.bn_RA(RA)
            x = torch.cat([x, RA], 1)
        x = self.lin(x)
        return x

    def print_params(self):
        print(f'model bias: {self.lin.bias.item():.3f}')
        print('model weights')
        for idx, elem in enumerate(self.lin.weight.squeeze()):
            if idx < self.dim:
                print(f'{self.idx_to_dst[idx % self.emb_dim]}: {elem.item():.3f}')
            else:
                print(f'feature {idx - self.dim}: {elem.item():.3f}')

class SIGNBaseClass(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout):
        super(SIGNBaseClass, self).__init__()

        self.K = K
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(self.K + 1):
            self.lins.append(Linear(in_channels, hidden_channels))
            self.bns.append(BN(hidden_channels))
        self.lin_out = Linear((K + 1) * hidden_channels, out_channels)
        self.dropout = dropout
        self.adj_t = None

    def reset_parameters(self):
        for lin, bn in zip(self.lins, self.bns):
            lin.reset_parameters()
            bn.reset_parameters()

    def cache_adj_t(self, edge_index, num_nodes):
        row, col = edge_index
        adj_t = SparseTensor(row=col, col=row,
                             sparse_sizes=(num_nodes, num_nodes))

        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    def forward(self, *args):
        raise NotImplementedError

class SIGNEmbedding(SIGNBaseClass):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout):
        super(SIGNEmbedding, self).__init__(in_channels, hidden_channels, out_channels, K, dropout)

    def forward(self, x, adj_t, num_nodes):
        if self.adj_t is None:
            self.adj_t = self.cache_adj_t(adj_t, num_nodes)
        hs = []
        for lin, bn in zip(self.lins, self.bns):
            h = lin(x)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
            x = self.adj_t @ x
        h = torch.cat(hs, dim=-1)
        x = self.lin_out(h)
        return x


class SIGN(SIGNBaseClass):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout):
        super(SIGN, self).__init__(in_channels, hidden_channels, out_channels, K, dropout)

    def forward(self, xs):
        """
        apply the sign feature transform where each component of the polynomial A^n x is treated independently
        @param xs: [batch_size, 2, n_features * (K + 1)]
        @return: [batch_size, 2, hidden_dim]
        """
        xs = torch.tensor_split(xs, self.K + 1, dim=-1)
        hs = []
        # split features into k+1 chunks and put each tensor in a list
        for lin, bn, x in zip(self.lins, self.bns, xs):
            h = lin(x)
            # the next line is a fuggly way to apply the same batch norm to both source and destination edges
            h = torch.cat((bn(h[:, 0, :]).unsqueeze(1), bn(h[:, 1, :]).unsqueeze(1)), dim=1)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        x = self.lin_out(h)
        return x


LABEL_LOOKUP = {1: {0: (1, 1), 1: (0, 1), 2: (1, 0)},
                2: {0: (1, 1), 1: (2, 1), 2: (1, 2), 3: (2, 2), 4: (0, 1), 5: (1, 0), 6: (0, 2), 7: (2, 0)},
                3: {0: (1, 1), 1: (2, 1), 2: (1, 2), 3: (2, 2), 4: (3, 1), 5: (1, 3), 6: (3, 2), 7: (2, 3), 8: (3, 3),
                    9: (0, 1), 10: (1, 0), 11: (0, 2), 12: (2, 0), 13: (0, 3), 14: (3, 0)}}


class MinhashPropagation(MessagePassing):
    def __init__(self):
        super().__init__(aggr='max')

    @torch.no_grad()
    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=-x)
        return -out


class HllPropagation(MessagePassing):
    def __init__(self):
        super().__init__(aggr='max')

    @torch.no_grad()
    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out


class ElphHashes(object):
    """
    class to store hashes and retrieve subgraph features
    """

    def __init__(self, args):
        assert args.max_hash_hops in {1, 2, 3}, f'hashing is not implemented for {args.max_hash_hops} hops'
        self.max_hops = args.max_hash_hops
        self.floor_sf = args.floor_sf  # if true set minimum sf to 0 (they're counts, so it should be)
        # minhash params
        self._mersenne_prime = np.uint64((1 << 61) - 1)
        self._max_minhash = np.uint64((1 << 32) - 1)
        self._minhash_range = (1 << 32)
        self.minhash_seed = 1
        self.num_perm = args.minhash_num_perm
        self.minhash_prop = MinhashPropagation()
        # hll params
        self.p = args.hll_p
        self.m = 1 << self.p  # the bitshift way of writing 2^p
        self.use_zero_one = args.use_zero_one
        self.label_lookup = LABEL_LOOKUP[self.max_hops]
        tmp = HyperLogLogPlusPlus(p=self.p)
        # store values that are shared and only depend on p
        self.hll_hashfunc = tmp.hashfunc
        self.alpha = tmp.alpha
        # the rank is the number of leading zeros. The max rank is the number of bits used in the hashes (64) minus p
        # as p bits are used to index the different counters
        self.max_rank = tmp.max_rank
        assert self.max_rank == 64 - self.p, 'not using 64 bits for hll++ hashing'
        self.hll_size = len(tmp.reg)
        self.hll_threshold = hyperloglog_const._thresholds[self.p - 4]
        self.bias_vector = torch.tensor(hyperloglog_const._bias[self.p - 4], dtype=torch.float)
        self.estimate_vector = torch.tensor(hyperloglog_const._raw_estimate[self.p - 4], dtype=torch.float)
        self.hll_prop = HllPropagation()

    def _np_bit_length(self, bits):
        """
        Get the number of bits required to represent each int in bits array
        @param bits: numpy [n_edges] array of ints
        @return:
        """
        return np.ceil(np.log2(bits + 1)).astype(int)

    def _get_hll_rank(self, bits):
        """
        get the number of leading zeros when each int in bits is represented as a self.max_rank-p bit array
        @param bits: a numpy array of ints
        @return:
        """
        # get the number of bits needed to represent each integer in bits
        bit_length = self._np_bit_length(bits)
        # the rank is the number of leading zeros, no idea about the +1 though
        rank = self.max_rank - bit_length + 1
        if min(rank) <= 0:
            raise ValueError("Hash value overflow, maximum size is %d\
                        bits" % self.max_rank)
        return rank

    def _init_permutations(self, num_perm):
        # Create parameters for a random bijective permutation function
        # that maps a 32-bit hash value to another 32-bit hash value.
        # http://en.wikipedia.org/wiki/Universal_hashing
        gen = np.random.RandomState(self.minhash_seed)
        return np.array([
            (gen.randint(1, self._mersenne_prime, dtype=np.uint64),
             gen.randint(0, self._mersenne_prime, dtype=np.uint64)) for
            _ in
            range(num_perm)
        ], dtype=np.uint64).T

    def initialise_minhash(self, n_nodes):
        init_hv = np.ones((n_nodes, self.num_perm), dtype=np.int64) * self._max_minhash
        a, b = self._init_permutations(self.num_perm)
        hv = hash_array(np.arange(1, n_nodes + 1))
        phv = np.bitwise_and((a * np.expand_dims(hv, 1) + b) % self._mersenne_prime, self._max_minhash)
        hv = np.minimum(phv, init_hv)
        return torch.tensor(hv, dtype=torch.int64)  # this conversion should be ok as self._max_minhash < max(int64)

    def initialise_hll(self, n_nodes):
        regs = np.zeros((n_nodes, self.m), dtype=np.int8)  # the registers to store binary values
        hv = hash_array(np.arange(1, n_nodes + 1))  # this function hashes 0 -> 0, so avoid
        # Get the index of the register using the first p bits of the hash
        # e.g. p=2, m=2^2=4, m-1=3=011 => only keep the right p bits
        reg_index = hv & (self.m - 1)
        # Get the rest of the hash. Python right shift drops the rightmost bits
        bits = hv >> self.p
        # Update the register
        ranks = self._get_hll_rank(bits)  # get the number of leading zeros in each element of bits
        regs[np.arange(n_nodes), reg_index] = np.maximum(regs[np.arange(n_nodes), reg_index], ranks)
        return torch.tensor(regs, dtype=torch.int8)  # int8 is fine as we're storing leading zeros in 64 bit numbers

    def build_hash_tables(self, num_nodes, edge_index):
        """
        Generate a hashing table that allows the size of the intersection of two nodes k-hop neighbours to be
        estimated in constant time
        @param num_nodes: The number of nodes in the graph
        @param adj: Int Tensor [2, edges] edges in the graph
        @return: hashes, cards. Hashes is a dictionary{dictionary}{tensor} with keys num_hops, 'hll' or 'minhash', cards
        is a tensor[n_nodes, max_hops-1]
        """
        hash_edge_index, _ = add_self_loops(edge_index)
        cards = torch.zeros((num_nodes, self.max_hops))
        node_hashings_table = {}
        for k in range(self.max_hops + 1):
            '''logger.info(f"Calculating hop {k} hashes")'''
            node_hashings_table[k] = {'hll': torch.zeros((num_nodes, self.hll_size), dtype=torch.int8),
                                      'minhash': torch.zeros((num_nodes, self.num_perm), dtype=torch.int64)}
            start = time()
            if k == 0:
                node_hashings_table[k]['minhash'] = self.initialise_minhash(num_nodes)
                node_hashings_table[k]['hll'] = self.initialise_hll(num_nodes)
            else:
                node_hashings_table[k]['hll'] = self.hll_prop(node_hashings_table[k - 1]['hll'], hash_edge_index)
                node_hashings_table[k]['minhash'] = self.minhash_prop(node_hashings_table[k - 1]['minhash'],
                                                                      hash_edge_index)
                cards[:, k - 1] = self.hll_count(node_hashings_table[k]['hll'])
            '''logger.info(f'{k} hop hash generation ran in {time() - start} s')'''
        return node_hashings_table, cards

    def _get_intersections(self, edge_list, hash_table):
        """
        extract set intersections as jaccard * union
        @param edge_list: [n_edges, 2] tensor to get intersections for
        @param hash_table:
        @param max_hops:
        @param p: hll precision parameter. hll uses 6 * 2^p bits
        @return:
        """
        intersections = {}
        # create features for each combination of hops.
        for k1 in range(1, self.max_hops + 1):
            for k2 in range(1, self.max_hops + 1):
                src_hll = hash_table[k1]['hll'][edge_list[:, 0]]
                src_minhash = hash_table[k1]['minhash'][edge_list[:, 0]]
                dst_hll = hash_table[k2]['hll'][edge_list[:, 1]]
                dst_minhash = hash_table[k2]['minhash'][edge_list[:, 1]]
                jaccard = self.jaccard(src_minhash, dst_minhash)
                unions = self._hll_merge(src_hll, dst_hll)
                union_size = self.hll_count(unions)
                intersection = jaccard * union_size
                intersections[(k1, k2)] = intersection
        return intersections

    def get_hashval(self, x):
        return x.hashvals

    def _linearcounting(self, num_zero):
        return self.m * torch.log(self.m / num_zero)

    def _estimate_bias(self, e):
        """
        Not exactly sure what this is doing or why exactly 6 nearest neighbours are used.
        @param e: torch tensor [n_links] of estimates
        @return:
        """
        nearest_neighbors = torch.argsort((e.unsqueeze(-1) - self.estimate_vector.to(e.device)) ** 2)[:, :6]
        return torch.mean(self.bias_vector.to(e.device)[nearest_neighbors], dim=1)

    def _refine_hll_count_estimate(self, estimate):
        idx = estimate <= 5 * self.m
        estimate_bias = self._estimate_bias(estimate)
        estimate[idx] = estimate[idx] - estimate_bias[idx]
        return estimate

    def hll_count(self, regs):
        """
        Estimate the size of set unions associated with regs
        @param regs: A tensor of registers [n_nodes, register_size]
        @return:
        """
        if regs.dim() == 1:
            regs = regs.unsqueeze(dim=0)
        retval = torch.ones(regs.shape[0], device=regs.device) * self.hll_threshold + 1
        num_zero = self.m - torch.count_nonzero(regs, dim=1)
        idx = num_zero > 0
        lc = self._linearcounting(num_zero[idx])
        retval[idx] = lc
        # we only keep lc values where lc <= self.hll_threshold, otherwise
        estimate_indices = retval > self.hll_threshold
        # Use HyperLogLog estimation function
        e = (self.alpha * self.m ** 2) / torch.sum(2.0 ** (-regs[estimate_indices]), dim=1)
        # for some reason there are two more cases
        e = self._refine_hll_count_estimate(e)
        retval[estimate_indices] = e
        return retval

    def _hll_merge(self, src, dst):
        if src.shape != dst.shape:
            raise ValueError('source and destination register shapes must be the same')
        return torch.maximum(src, dst)

    def hll_neighbour_merge(self, root, neighbours):
        all_regs = torch.cat([root.unsqueeze(dim=0), neighbours], dim=0)
        return torch.max(all_regs, dim=0)[0]

    def minhash_neighbour_merge(self, root, neighbours):
        all_regs = torch.cat([root.unsqueeze(dim=0), neighbours], dim=0)
        return torch.min(all_regs, dim=0)[0]

    def jaccard(self, src, dst):
        """
        get the minhash Jaccard estimate
        @param src: tensor [n_edges, num_perms] of hashvalues
        @param dst: tensor [n_edges, num_perms] of hashvalues
        @return: tensor [n_edges] jaccard estimates
        """
        if src.shape != dst.shape:
            raise ValueError('source and destination hash value shapes must be the same')
        return torch.count_nonzero(src == dst, dim=-1) / self.num_perm

    def get_subgraph_features(self, links, hash_table, cards, batch_size=11000000):
        """
        extracts the features that play a similar role to the labeling trick features. These can be thought of as approximations
        of path distances from the source and destination nodes. There are k+2+\sum_1^k 2k features
        @param links: tensor [n_edges, 2]
        @param hash_table: A Dict{Dict} of torch tensor [num_nodes, hash_size] keys are hop index and hash type (hyperlogloghash, minhash)
        @param cards: Tensor[n_nodes, max_hops] of hll neighbourhood cardinality estimates
        @param batch_size: batch size for computing intersections. 11m splits the large ogb datasets into 3.
        @return: Tensor[n_edges, max_hops(max_hops+2)]
        """
        if links.dim() == 1:
            links = links.unsqueeze(0)
        link_loader = DataLoader(range(links.size(0)), batch_size, shuffle=False, num_workers=0)
        all_features = []
        for batch in tqdm(link_loader):
            intersections = self._get_intersections(links[batch], hash_table)
            cards1, cards2 = cards.to(links.device)[links[batch, 0]], cards.to(links.device)[links[batch, 1]]
            features = torch.zeros((len(batch), self.max_hops * (self.max_hops + 2)), dtype=float, device=links.device)
            features[:, 0] = intersections[(1, 1)]
            if self.max_hops == 1:
                features[:, 1] = cards2[:, 0] - features[:, 0]
                features[:, 2] = cards1[:, 0] - features[:, 0]
            elif self.max_hops == 2:
                features[:, 1] = intersections[(2, 1)] - features[:, 0]  # (2,1)
                features[:, 2] = intersections[(1, 2)] - features[:, 0]  # (1,2)
                features[:, 3] = intersections[(2, 2)] - features[:, 0] - features[:, 1] - features[:, 2]  # (2,2)
                features[:, 4] = cards2[:, 0] - torch.sum(features[:, 0:2], dim=1)  # (0, 1)
                features[:, 5] = cards1[:, 0] - features[:, 0] - features[:, 2]  # (1, 0)
                features[:, 6] = cards2[:, 1] - torch.sum(features[:, 0:5], dim=1)  # (0, 2)
                features[:, 7] = cards1[:, 1] - features[:, 0] - torch.sum(features[:, 0:4], dim=1) - features[:,
                                                                                                      5]  # (2, 0)
            elif self.max_hops == 3:
                features[:, 1] = intersections[(2, 1)] - features[:, 0]  # (2,1)
                features[:, 2] = intersections[(1, 2)] - features[:, 0]  # (1,2)
                features[:, 3] = intersections[(2, 2)] - features[:, 0] - features[:, 1] - features[:, 2]  # (2,2)
                features[:, 4] = intersections[(3, 1)] - features[:, 0] - features[:, 1]  # (3,1)
                features[:, 5] = intersections[(1, 3)] - features[:, 0] - features[:, 2]  # (1, 3)
                features[:, 6] = intersections[(3, 2)] - torch.sum(features[:, 0:4], dim=1) - features[:, 4]  # (3,2)
                features[:, 7] = intersections[(2, 3)] - torch.sum(features[:, 0:4], dim=1) - features[:, 5]  # (2,3)
                features[:, 8] = intersections[(3, 3)] - torch.sum(features[:, 0:8], dim=1)  # (3,3)
                features[:, 9] = cards2[:, 0] - features[:, 0] - features[:, 1] - features[:, 4]  # (0, 1)
                features[:, 10] = cards1[:, 0] - features[:, 0] - features[:, 2] - features[:, 5]  # (1, 0)
                features[:, 11] = cards2[:, 1] - torch.sum(features[:, 0:5], dim=1) - features[:, 6] - features[:,
                                                                                                       9]  # (0, 2)
                features[:, 12] = cards1[:, 1] - torch.sum(features[:, 0:5], dim=1) - features[:, 7] - features[:,
                                                                                                       10]  # (2, 0)
                features[:, 13] = cards2[:, 2] - torch.sum(features[:, 0:9], dim=1) - features[:, 9] - features[:,
                                                                                                       11]  # (0, 3)
                features[:, 14] = cards1[:, 2] - torch.sum(features[:, 0:9], dim=1) - features[:, 10] - features[:,
                                                                                                        12]  # (3, 0)
            else:
                raise NotImplementedError("Only 1, 2 and 3 hop hashes are implemented")
            if not self.use_zero_one:
                if self.max_hops == 2:  # for two hops any positive edge that's dist 1 from u must be dist 2 from v etc.
                    features[:, 4] = 0
                    features[:, 5] = 0
                elif self.max_hops == 3:  # in addition for three hops 0,2 is impossible for positive edges
                    features[:, 4] = 0
                    features[:, 5] = 0
                    features[:, 11] = 0
                    features[:, 12] = 0
            if self.floor_sf:  # should be more accurate, but in practice makes no difference
                features[features < 0] = 0
            all_features.append(features)
        features = torch.cat(all_features, dim=0)
        return features
