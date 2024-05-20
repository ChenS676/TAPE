import copy
import os, sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from graphgps.train.opt_train import Trainer_SEAL
from graphgps.network.heart_gnn import DGCNN

from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from utils import set_cfg, parse_args, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
    create_logger
from gcns_subgraph.SEAL.utils import *
from torch_geometric.data import InMemoryDataset, Dataset
from data_utils.load_data_nc import load_graph_cora, load_graph_pubmed, load_tag_arxiv23, load_graph_ogbn_arxiv

class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, splits, num_hops, percent=100, split='train',node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False):
        self.data = data
        self.split_edge = splits
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data'.format(self.split)
        else:
            name = 'SEAL_{}_data_{}'.format(self.split, self.percent)
        name += '.pt'
        return [name]

    def process(self):
        pos_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent)
        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)

        edge_weight_cpu = edge_weight.cpu()
        edge_index_cpu = self.data.edge_index.cpu()

        A = ssp.csr_matrix(
            (edge_weight_cpu, (edge_index_cpu[0], edge_index_cpu[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None

        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            pos_edge, A, self.data.x, 1, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
        neg_list = extract_enclosing_subgraphs(
            neg_edge, A, self.data.x, 0, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, splits, num_hops, percent=100, split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False, **kwargs):
        self.data = data
        self.split_edge = splits
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDynamicDataset, self).__init__(root)

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)


        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        edge_weight_cpu = edge_weight.cpu()
        edge_index_cpu = self.data.edge_index.cpu()

        self.A = ssp.csr_matrix(
            (edge_weight_cpu, (edge_index_cpu[0], edge_index_cpu[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None

    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                             self.max_nodes_per_hop, node_features=self.data.x,
                             y=y, directed=self.directed, A_csc=self.A_csc)
        data = construct_pyg_graph(*tmp, self.node_label)

        return data


if __name__ == "__main__":
    FILE_PATH = get_git_repo_root_path() + '/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    torch.set_num_threads(cfg.num_threads)
    batch_sizes = [cfg.train.batch_size]  # [8, 16, 32, 64]


    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
    for batch_size in batch_sizes:
        for run_id, seed, split_index in zip(
                *run_loop_settings(cfg, args)):
            custom_set_run_dir(cfg, run_id)
            set_printing(cfg)
            cfg.seed = seed
            cfg.run_id = run_id
            seed_everything(cfg.seed)
            if cfg.data.name == 'pubmed':
                data = load_graph_pubmed(False)
            elif cfg.data.name == 'cora':
                data, _ = load_graph_cora(False)
            elif cfg.data.name == 'arxiv_2023':
                data, _ = load_tag_arxiv23()
            splits = do_edge_split(copy.deepcopy(data), cfg.data.val_pct, cfg.data.test_pct)
            path = os.path.dirname(__file__) + '/seal_{}'.format(cfg.data.name)
            dataset = {}

            if cfg.train.dynamic_train == True:
                dataset['train'] = SEALDynamicDataset(
                    path,
                    data,
                    splits,
                    num_hops=cfg.model.num_hops,
                    split='train',
                    node_label= cfg.model.node_label,
                    directed=not cfg.data.undirected,
                )
                dataset['valid'] = SEALDynamicDataset(
                    path,
                    data,
                    splits,
                    num_hops=cfg.model.num_hops,
                    split='valid',
                    node_label= cfg.model.node_label,
                    directed=not cfg.data.undirected,
                )
                dataset['test'] = SEALDynamicDataset(
                    path,
                    data,
                    splits,
                    num_hops=cfg.model.num_hops,
                    split='test',
                    node_label= cfg.model.node_label,
                    directed=not cfg.data.undirected,
                )
            else:
                dataset['train'] = SEALDataset(
                    path,
                    data,
                    splits,
                    num_hops=cfg.model.num_hops,
                    split='train',
                    node_label= cfg.model.node_label,
                    directed=not cfg.data.undirected,
                )
                dataset['valid'] = SEALDataset(
                    path,
                    data,
                    splits,
                    num_hops=cfg.model.num_hops,
                    split='valid',
                    node_label= cfg.model.node_label,
                    directed=not cfg.data.undirected,
                )
                dataset['test'] = SEALDataset(
                    path,
                    data,
                    splits,
                    num_hops=cfg.model.num_hops,
                    split='test',
                    node_label= cfg.model.node_label,
                    directed=not cfg.data.undirected,
                )
            model = DGCNN(cfg.model.hidden_channels, cfg.model.num_layers, cfg.model.max_z, cfg.model.k,
                          dataset['train'], False, use_feature=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.base_lr)

            # Execute experiment
            trainer = Trainer_SEAL(FILE_PATH,
                                   cfg,
                                   model,
                                   optimizer,
                                   dataset,
                                   run_id,
                                   args.repeat,
                                   loggers,
                                   print_logger = None,
                                   batch_size=batch_size)

            start = time.time()
            trainer.train()
            end = time.time()
            print('Training time: ', end - start)

        print('All runs:')

        result_dict = {}
        for key in loggers:
            print(key)
            _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats()
            result_dict.update({key: valid_test})

        trainer.save_result(result_dict)
