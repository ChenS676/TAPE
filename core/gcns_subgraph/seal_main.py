import os, sys
import warnings


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from graphgps.train.opt_train import Trainer_SEAL
from graphgps.network.heart_gnn import DGCNN

from data_utils.load import load_data_lp
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from utils import set_cfg, parse_args, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
    create_logger
from gcns_subgraph.SEAL.utils import *
from torch_geometric.data import InMemoryDataset


class SEALDataset(InMemoryDataset):
    def __init__(self, data, num_hops, percent=100, split='train',node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False):
        self.data = data[split]
        self.split_edge = do_edge_split(data)
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDataset, self).__init__(os.path.dirname(__file__))
        self._data, self.slices = torch.load(self.processed_paths[0])
        self.pos_edge_label = data[split].pos_edge_label
        self.neg_edge_label = data[split].neg_edge_label
        self.pos_edge_label_index = data[split].pos_edge_label_index
        self.neg_edge_label_index = data[split].neg_edge_label_index


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
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
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



if __name__ == "__main__":
    FILE_PATH = get_git_repo_root_path() + '/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    torch.set_num_threads(cfg.num_threads)
    batch_sizes = [64]  # [8, 16, 32, 64]

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
            auto_select_device()
            splits, text = load_data_lp[cfg.data.name](cfg.data)

            dataset = {}

            dataset['train'] = SEALDataset(
                splits,
                num_hops=cfg.model.num_hops,
                split='train',
                node_label= cfg.model.node_label,
                directed=cfg.data.undirected,
            )
            dataset['valid'] = SEALDataset(
                splits,
                num_hops=cfg.model.num_hops,
                split='valid',
                node_label= cfg.model.node_label,
                directed=cfg.data.undirected,
            )
            dataset['test'] = SEALDataset(
                splits,
                num_hops=cfg.model.num_hops,
                split='test',
                node_label= cfg.model.node_label,
                directed=cfg.data.undirected,
            )
            model = DGCNN(cfg.model.hidden_channels, cfg.model.num_layers, cfg.model.max_z, cfg.model.k,
                          dataset['train'], False, use_feature=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # Execute experiment
            trainer = Trainer_SEAL(FILE_PATH,
                                    cfg,
                                    model,
                                    optimizer,
                                    dataset,
                                    run_id,
                                    args.repeat,
                                    loggers,
                                    batch_size)

            start = time.time()
            trainer.train()
            end = time.time()

    print("Best Parameters Found:")
    print(best_params)