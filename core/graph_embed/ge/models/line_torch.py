import torch
from torch.nn import Embedding, Module
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import random

import numpy as np

from ..alias import create_alias_table, alias_sample
from ..utils import preprocess_nxgraph
from tqdm import tqdm 

class LineLoss(Module):
    def forward(self, y_true, y_pred):
        # mean sigmoid x1*x2 dot product 
        order1 = -torch.mean(torch.log(torch.sigmoid(y_true[0] * y_pred[0])))
        order2 = -torch.mean(torch.log(torch.sigmoid(y_true[1] * y_pred[1])))
        return order1+order2

class LINEModel(Module):
    def __init__(self, num_nodes, embedding_size, order='second'):
        super(LINEModel, self).__init__()

        self.first_emb = Embedding(num_nodes, embedding_size)
        self.second_emb = Embedding(num_nodes, embedding_size)
        self.context_emb = Embedding(num_nodes, embedding_size)

        self.order = order

    def forward(self, v_i, v_j):
        v_i_emb = self.first_emb(v_i)
        v_j_emb = self.first_emb(v_j)
        v_i_emb_second = self.second_emb(v_i)
        v_j_context_emb = self.context_emb(v_j)

        first_order = torch.sum(v_i_emb * v_j_emb, dim=-1)
        second_order = torch.sum(v_i_emb_second * v_j_context_emb, dim=-1)
        # print(first_order)
        if self.order == 'first':
            return first_order
        elif self.order == 'second':
            return second_order
        else:
            return [first_order, second_order]


class LINE_torch:
    def __init__(self, graph, embedding_size=8, negative_ratio=5, order='second', device='cpu', lr=None):
        """

        :param graph:
        :param embedding_size:
        :param negative_ratio:
        :param order: 'first','second','all'
        """
        if order not in ['first', 'second', 'all']:
            raise ValueError('mode must be fisrt,second,or all')

        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.use_alias = False # True
        self.lr = lr

        self.rep_size = embedding_size
        self.order = order

        self._embeddings = {}
        self.negative_ratio = negative_ratio
        self.order = order

        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.samples_per_epoch = self.edge_size*(1+negative_ratio)

        # self._gen_sampling_table()
        
        self.model = LINEModel(self.node_size, self.rep_size, self.order).to(device)
        self.batch_it = self.batch_iter(self.node2idx)

        self.criterion = LineLoss()
        
        print(self.model.parameters)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.device = device 
        
    def reset_training_args(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = (
            (self.samples_per_epoch - 1) // self.batch_size + 1)*times

    def count_parameters(self):
        total_params = 0
        self.embedding_dict =  {'first': self.model.first_emb, 'second': self.model.second_emb}
        if self.order == 'first':
            total_params += np.prod(self.embedding_dict['first'].weight.shape)
        elif self.order == 'second':
            total_params += np.prod(self.embedding_dict['second'].weight.shape)
        else:
            total_params += np.prod(self.embedding_dict['first'].weight.shape)
            total_params += np.prod(self.embedding_dict['second'].weight.shape)
        
        return total_params
    
    def _gen_sampling_table(self):

        # create sampling table for vertex
        power = 0.75
        numNodes = self.node_size
        node_degree = np.zeros(numNodes)  # out degree
        node2idx = self.node2idx

        for edge in self.graph.edges():
            node_degree[node2idx[edge[0]]
                        ] += self.graph[edge[0]][edge[1]].get('weight', 1.0)

        total_sum = sum([math.pow(node_degree[i], power)
                         for i in range(numNodes)])
        norm_prob = [float(math.pow(node_degree[j], power)) /
                     total_sum for j in range(numNodes)]

        self.node_accept, self.node_alias = create_alias_table(norm_prob)

        # create sampling table for edge
        numEdges = self.graph.number_of_edges()
        total_sum = sum([self.graph[edge[0]][edge[1]].get('weight', 1.0)
                         for edge in self.graph.edges()])
        norm_prob = [self.graph[edge[0]][edge[1]].get('weight', 1.0) *
                     numEdges / total_sum for edge in self.graph.edges()]

        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)
        
    def batch_iter(self, node2idx):

        edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph.edges()]

        data_size = self.graph.number_of_edges()
        shuffle_indices = np.random.permutation(np.arange(data_size))
        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0
        count = 0
        start_index = 0
        end_index = min(start_index + self.batch_size, data_size)
        
        while True:
            if mod == 0:

                h = []
                t = []
                for i in range(start_index, end_index):
                    if random.random() >= self.edge_accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
                sign = np.ones(len(h))
            else:
                sign = np.ones(len(h))*-1
                t = []
                for i in range(len(h)):

                    t.append(alias_sample(
                        self.node_accept, self.node_alias))

            if self.order == 'all':
                yield ([torch.IntTensor(h), torch.IntTensor(t)], [torch.LongTensor(sign), torch.LongTensor(sign)])
            else:
                yield ([torch.IntTensor(h), torch.IntTensor(t)], [torch.LongTensor(sign)])
                
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)

            if start_index >= data_size:
                count += 1
                mod = 0
                h = []
                shuffle_indices = np.random.permutation(np.arange(data_size))
                start_index = 0
                end_index = min(start_index + self.batch_size, data_size)
                

    def get_embeddings(self,):
        
        self.embedding_dict =  {'first': self.model.first_emb, 'second': self.model.second_emb}
        self._embeddings = {}
        if self.order == 'first':
            embeddings = self.embedding_dict['first'].get_weights()[0]
        elif self.order == 'second':
            embeddings = self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = torch.hstack((self.embedding_dict['first'].weight, self.embedding_dict['second'].weight))
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding
            

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1, times=1):
        self.reset_training_args(batch_size, times)

        for epoch in range(initial_epoch, initial_epoch + epochs):
            running_loss = 0.0
            for i in range(self.steps_per_epoch):
                data = next(self.batch_it)
                inputs, labels = data
                if self.model.order == 'all':
                    inputs[0] = inputs[0].to(self.device)
                    inputs[1] = inputs[1].to(self.device)
                    labels[0] = labels[0].to(self.device)
                    labels[1] = labels[1].to(self.device)
                    
                self.optimizer.zero_grad()
                # print(inputs[0], inputs[1])
                outputs = self.model(inputs[0], inputs[1])
                
                if self.model.order == 'all':
                    outputs[0] = outputs[0].to(self.device)
                    
                loss = self.criterion(labels, outputs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                print(f"running loss", loss.item())

            if verbose and (epoch + 1) % verbose == 0:
                print(f'Epoch {epoch + 1}, Loss: {running_loss / self.steps_per_epoch}')

        print('Finished Training')

