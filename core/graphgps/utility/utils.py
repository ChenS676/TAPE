import os
import logging
import sys
import numpy as np
import time
import datetime
import pytz
import torch
import git
import subprocess
import pandas as pd
import argparse
import torch.optim as optim
from torch_scatter import scatter
from yacs.config import CfgNode as CN
from torch_geometric.graphgym.config import (cfg,
                                             makedirs_rm_exist)
from torch_geometric.utils import remove_self_loops
from typing import Tuple, List, Dict
import logging
from yacs.config import CfgNode
import torch 

set_float = lambda result: float(result.split(' ± ')[0])

def get_git_repo_root_path():
    try:
        # Using git module
        git_repo = git.Repo('.', search_parent_directories=True)
        return git_repo.working_dir
    except git.InvalidGitRepositoryError:
        # Fallback to using subprocess if not a valid Git repository
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True)

        if result.returncode == 0:
            return result.stdout.strip()
        print("Error:", result.stderr)
        return None

        
        
def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path, exist_ok=True)
        if log:
            print(f'Created directory {path}')
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print(f'Directory {path} already exists.')
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper


def get_root_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "..")




# Define a function that uses the lambda function
def process_value(v):
    return (lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)(v)


def append_acc_to_excel(uuid_val, metrics_acc, root, name, method):
    # if not exists save the first row

    csv_columns = ['Metric'] + list(metrics_acc) 
    # load old csv
    try:
        Data = pd.read_csv(root)[:-1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)

    acc_lst = [process_value(v) for k, v in metrics_acc.items()]
    # merge with old lines, 
    v_lst = [f'{name}_{uuid_val}_{method}'] + acc_lst
    new_df = pd.DataFrame([v_lst], columns=csv_columns)
    new_Data = pd.concat([Data, new_df])

    # best value
    highest_values = new_Data.apply(lambda column: max(column, default=None))

    # concat and save
    Best_list = ['Best'] + highest_values[1:].tolist()
    Best_df = pd.DataFrame([Best_list], columns=Data.columns)
    upt_Data = pd.concat([new_Data, Best_df])
    upt_Data.to_csv(root,index=False)

    return upt_Data


def append_mrr_to_excel(uuid_val, metrics_mrr, root, name, method):
 
    csv_columns, csv_numbers = [], []
    for i, (k, v) in enumerate(metrics_mrr.items()): 
        if i == 0:
            csv_columns = ['Metric'] + list(v.keys())
        csv_numbers.append([f'{k}_{uuid_val}_{name}_{method}'] + list(v.values()))
    
    print(csv_numbers)

    try:
        Data = pd.read_csv(root)[:-1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)

    
    new_df = pd.DataFrame(csv_numbers, columns = csv_columns)
    new_Data = pd.concat([Data, new_df])
    
    highest_values = new_Data.apply(lambda column: max(column, default=None))
    Best_list = ['Best'] + highest_values[1:].tolist()
    Best_df = pd.DataFrame([Best_list], columns=csv_columns)
    upt_Data = pd.concat([new_Data, Best_df])
    
    upt_Data.to_csv(root, index=False)
    
    return upt_Data


def config_device(cfg):
    
    # detect gpu
    num_cuda_devices = 0
    if torch.cuda.is_available():
        # Get the number of available CUDA devices
        num_cuda_devices = torch.cuda.device_count()
        print(f'Number of available CUDA devices: {num_cuda_devices}')
    # enviorment setting
    if num_cuda_devices <= 0:
        cfg.device = 'cpu'
    elif hasattr(cfg, 'data') and hasattr(cfg.data, 'device'):
        cfg.device = cfg.data.device
    elif hasattr(cfg, 'train') and hasattr(cfg.data, 'device'):
        cfg.device = cfg.train.device
    else:
        cfg.device = 0
    
    return cfg
    

def set_cfg(file_path, args):
    with open(file_path + args.cfg_file, "r") as f:
        return CN.load_cfg(f)
    
    
def init_cfg_test():
    """
    Initialize a CfgNode instance to test dataloader for link prediction.

    Args:
        cfg_dict (dict): Dictionary containing configuration parameters.

    Returns:
        CN: Initialized CfgNode instance.
    """
    cfg_dict = {
        'data': {  
            'undirected': True,
            'include_negatives': True,
            'val_pct': 0.1,
            'test_pct': 0.1,
            'split_labels': True,
            'device': 'cpu',
            'split_index': [0.8, 0.15, 0.05]
            },
        'train':  {
                'device': 'cpu'
            }
    }
    return CN(cfg_dict)


def create_logger(repeat):
    return {
        'Hits@1': Logger(repeat),
        'Hits@3': Logger(repeat),
        'Hits@10': Logger(repeat),
        'Hits@20': Logger(repeat),
        'Hits@50': Logger(repeat),
        'Hits@100': Logger(repeat),
        'MRR': Logger(repeat),
        'mrr_hit1': Logger(repeat),
        'mrr_hit3': Logger(repeat), 
        'mrr_hit10': Logger(repeat),
        'mrr_hit20': Logger(repeat),
        'mrr_hit50': Logger(repeat),
        'mrr_hit100': Logger(repeat),
        'AUC': Logger(repeat),
        'AP': Logger(repeat),
        'ACC': Logger(repeat)
    }


class Logger(object):
    """
    Creates a Logger object for tracking and printing various metrics during the execution of an experiment.

    Args:
        runs: The number of runs to track metrics for.
        info: Additional information or context to include in the logger.

    Methods:
        - add_result(run, result): Add a result Dict for a specific run int to the logger.
        - print_statistics(run=None): Print statistics for a specific run or aggregated statistics across all runs.
            Calculating statistics across all runs:
            Iterate over all runs and calculate statistics for each run.
            Append these statistics (highest train, highest validation, final train, final test) to best_results.
            Convert best_results to a PyTorch tensor.
            Print the overall statistics across all runs:
            Mean and standard deviation of the highest train accuracy (r.mean():.2f ± r.std():.2f)
            Mean and standard deviation of the highest validation accuracy (r.mean():.2f ± r.std():.2f)
            Mean and standard deviation of the final train accuracy (r.mean():.2f ± r.std():.2f)
            Mean and standard deviation of the final test accuracy (r.mean():.2f ± r.std():.2f)
            Return the mean and variance of the highest validation accuracy for potential further use.
        - get_best_result(): Get the results stored in the logger.
   
    """
    def __init__(self, runs, info=None):
        self.info = info
        self.runs = runs
        self.results = [[] for _ in range(runs)]

    def reset(self):
        return [[] for _ in range(self.runs)]
    
    
    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)


    def calc_run_stats(self, 
                       run:int =None, 
                       print_mode:bool =True) -> Tuple[float, float, float, float]:
        result = 100 * torch.tensor(self.results[run])
        print(result.shape)
        best_valid_epoch = result[:, 1].argmax().item()
        best_train_valid, _, best_test_valid = result[best_valid_epoch]

        if print_mode:
            print(f'Highest Train: {result[:, 0].max().item():.2f} at Epoch {100*result[:, 0].argmax().item()}, Highest Valid: {result[:, 1].max().item():.2f} at Epoch {100*best_valid_epoch}, Final Train: {best_train_valid:.2f} at Epoch {100*best_valid_epoch} Final Test: {best_test_valid:.2f} at Epoch {100*best_valid_epoch}.')
        
        # best train, best valid, train with the best valid epoch, test with the best valid epoch
        return round(result[:, 0].max().item(), 2), round(result[:, 1].max().item(), 2), round(best_train_valid.item(), 2), round(best_test_valid.item(), 2)
    
    
    def calc_all_stats(self, print_mode: bool=True) -> Tuple[str, str, str, List[float], List[float]]:
        
        best_results = [self.calc_run_stats(run=i, print_mode=False) for i in range(self.runs)]
        
        best_result = torch.tensor(best_results)

        # best train
        r = best_result[:, 0].float()
        best_train = f'{r.mean():.2f} ± {r.std():.2f}'

        # best valid 
        r = best_result[:, 1].float()
        best_valid_mean = round(r.mean().item(), 2)
        best_valid_var = round(r.std().item(), 2)
        best_valid = f'{best_valid_mean:.2f} ± {best_valid_var:.2f}'


        # train with best valid
        r = best_result[:, 2].float()
        valid_train_mean = round(r.mean().item(), 2)
        valid_train_var = round(r.std().item(), 2) 
        valid_train = f'{valid_train_mean:.2f} ± {valid_train_var:.2f}'
        
        # test with best valid
        r = best_result[:, 3].float()
        valid_test_mean = round(r.mean().item(), 2)
        valid_test_var = round(r.std().item(), 2)
        valid_test = f'{valid_train_mean:.2f} ± {valid_train_var:.2f}'
        
        # neglect best train and best valid
        mean_list = [valid_train_mean, best_valid_mean, valid_test_mean]
        var_list = [valid_train_var, best_valid_var, valid_test_var]

        if print_mode:
            print(f'Highest Train: {best_train}')
            print(f'Highest Valid: {best_valid}')
            print(f'Train with the best valid: {valid_train}')
            print(f'Test with the best valid epoch: {valid_test}')
        
        return best_train, best_valid, valid_train, valid_test, mean_list, var_list

    def save2dict(self):
        "save the result into csv based on calc_all_stats"
        

def get_logger(name, log_dir, config_dir):

    """
    Set up printing options

    """
    logging.root.handlers = []
    logging_cfg = {'level': logging.INFO, 'format': '%(message)s'}
    os.makedirs(cfg.run_dir, exist_ok=True)
    h_file = logging.FileHandler(f'{cfg.run_dir}/logging.log')
    h_stdout = logging.StreamHandler(sys.stdout)
    if cfg.print == 'file':
        logging_cfg['handlers'] = [h_file]
    elif cfg.print == 'stdout':
        logging_cfg['handlers'] = [h_stdout]
    elif cfg.print == 'both':
        logging_cfg['handlers'] = [h_file, h_stdout]
    else:
        raise ValueError('Print option not supported')

    logging.basicConfig(**logging_cfg)
    return logging_cfg



def save_emb(score_emb, save_path):

    if len(score_emb) == 6:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x1, x2= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x1,
        'node_emb_with_valid_edges': x2

        }
        
    elif len(score_emb) == 5:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x
        }
        
    elif len(score_emb) == 4:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, = score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        }
   
    torch.save(state, save_path)
        

def negate_edge_index(edge_index, batch=None):
    """Negate batched sparse adjacency matrices given by edge indices.

    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.

    Implementation inspired by `torch_geometric.utils.to_dense_adj`

    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.

    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        # Remove existing edges from the full N x N adjacency matrix
        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(_idx1.numel(), dtype=torch.short,
                           device=edge_index.device)
        scatter(zero, idx, dim=0, out=adj, reduce='mul')

        # Convert to edge index format
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    return torch.cat(negative_index_list, dim=1).contiguous()


def flatten_dict(metrics):
    """Flatten a list of train/val/test metrics into one dict to send to wandb.

    Args:
        metrics: List of Dicts with metrics

    Returns:
        A flat dictionary with names prefixed with "train/" , "val/" , "test/"
    """
    prefixes = ['train', 'val', 'test']
    result = {}
    for i in range(len(metrics)):
        # Take the latest metrics.
        stats = metrics[i][-1]
        result |= {f"{prefixes[i]}/{k}": v for k, v in stats.items()}
    return result


def cfg_to_dict(cfg_node, key_list=None):
    """Convert a config node to dictionary.

    Yacs doesn't have a default function to convert the cfg object to plain
    python dict. The following function was taken from
    https://github.com/rbgirshick/yacs/issues/19
    """
    if key_list is None:
        key_list = []
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            logging.warning(f"Key {'.'.join(key_list)} with "
                            f"value {type(cfg_node)} is not "
                            f"a valid type; valid types: {_VALID_TYPES}")
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_to_dict(v, key_list + [k])
        return cfg_dict


def make_wandb_name(cfg):
    # Format dataset name.
    dataset_name = cfg.dataset.format
    if dataset_name.startswith('OGB'):
        dataset_name = dataset_name[3:]
    if dataset_name.startswith('PyG-'):
        dataset_name = dataset_name[4:]
    if dataset_name in ['GNNBenchmarkDataset', 'TUDataset']:
        # Shorten some verbose dataset naming schemes.
        dataset_name = ""
    if cfg.dataset.name != 'none':
        dataset_name += "-" if dataset_name != "" else ""
        if cfg.dataset.name == 'LocalDegreeProfile':
            dataset_name += 'LDP'
        else:
            dataset_name += cfg.dataset.name
    # Format model name.
    model_name = cfg.model.type
    if cfg.model.type in ['gnn', 'custom_gnn']:
        model_name += f".{cfg.gnn.layer_type}"
    elif cfg.model.type == 'GPSModel':
        model_name = f"GPS.{cfg.gt.layer_type}"
    model_name += f".{cfg.name_tag}" if cfg.name_tag else ""
    # Compose wandb run name.
    name = f"{dataset_name}.{model_name}.r{cfg.run_id}"
    return name


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=False,
                        default='cora',
                        help='data name')
        
    parser.add_argument('--repeat', type=int, default=2,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()


def set_cfg(file_path, cfg_file):
    with open(file_path + cfg_file, "r") as f:
        return CN.load_cfg(f)

def run_loop_settings(cfg: CN,
                      args: argparse.Namespace) -> Tuple[List[int], List[int], List[int]]:
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if cfg.run.multiple_splits == 'None':
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.data.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run.multiple_splits)
        seeds = [cfg.run.seed] * num_iterations
        split_indices = cfg.run.multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices



def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer



class LinearDecayLR:
    def __init__(self, optimizer, start_lr, end_lr, num_epochs):
        """
        Initialize the LinearDecayLR scheduler.
        
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            start_lr (float): Initial learning rate.
            end_lr (float): Final learning rate.
            num_epochs (int): Number of epochs over which to linearly decay the learning rate.
        """
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_epochs = num_epochs
        self.step_count = 0

    def step(self):
        """Update the learning rate for the current step."""
        self.step_count += 1
        lr = self.start_lr + (self.end_lr - self.start_lr) * (self.step_count / self.num_epochs)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """Get the current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]



def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)



def custom_set_run_dir(cfg, wandb_tag):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(wandb_tag))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)



def set_printing(cfg):
    """
    Set up printing options

    """
    import logging

    # Step 1: Create a logger
    logger = logging.getLogger(__name__)

    # Step 2: Set the overall log level for the logger
    logger.setLevel(logging.INFO)

    # Step 3: Create handlers
    file_handler = logging.FileHandler(f'{cfg.run_dir}/logging.log')
    # console_handler = logging.StreamHandler(sys.stdout) # if you dont want to see the log in the console
    console_handler = logging.StreamHandler() 
    
    # Step 4: Set log levels for handlers
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    # Step 5: Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Step 6: Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_optimizer(model, optimizer_config):
    # sourcery skip: list-comprehension
    r"""
    Create optimizer for the model

    Args:
        params: PyTorch model parameters

    Returns: PyTorch optimizer

    """
    params = []

    params.extend(
        param for _, param in model.named_parameters() if param.requires_grad
    )
    optimizer = optimizer_config.optimizer
    if optimizer.type == 'adam':
        optimizer = optim.Adam(params, lr=optimizer.base_lr)
    elif optimizer.type == 'sgd':
        optimizer = optim.SGD(params, lr=optimizer.base_lr)
    else:
        raise ValueError(f'Optimizer {optimizer_config.optimizer} not supported')
    optimizer.zero_grad()
    return optimizer


def create_scheduler(optimizer, scheduler_config):
    r"""
    Create learning rate scheduler for the optimizer

    Args:
        optimizer: PyTorch optimizer

    Returns: PyTorch scheduler

    """

    # Try to load customized scheduler
    if scheduler_config.scheduler == 'none':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_config.max_epoch + 1)
    elif scheduler_config.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=scheduler_config.steps,
            gamma=scheduler_config.lr_decay)
    elif scheduler_config.scheduler == 'cos':
        scheduler = \
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.max_epoch)
    else:
        raise ValueError(f'Scheduler {scheduler_config.scheduler} not supported')
    return scheduler

import os.path as osp
def init_model_from_pretrained(model, pretrained_dir, freeze_pretrained=False) -> torch.Tensor:
    # raise NotImplementedError #assign to constantin
    # TODO start with torch.Embedding as preliminary step
    """ Load the generated embedding from LLM to update data.x
    
    Step1: load themd
    Step2: postprocessing to torch.Tensor in shape (num_nodes, n_dim)
    num_nodes = data.x.shape[0]
    n_dim = free to choose
    Step3: Normalization 
    
    Return 
    """



def negate_edge_index(edge_index, batch=None):
    """Negate batched sparse adjacency matrices given by edge indices.

    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.

    Implementation inspired by `torch_geometric.utils.to_dense_adj`

    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.

    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        # Remove existing edges from the full N x N adjacency matrix
        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(_idx1.numel(), dtype=torch.short,
                           device=edge_index.device)
        scatter(zero, idx, dim=0, out=adj, reduce='mul')

        # Convert to edge index format
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_negative


def flatten_dict(metrics):
    """Flatten a list of train/val/test metrics into one dict to send to wandb.

    Args:
        metrics: List of Dicts with metrics

    Returns:
        A flat dictionary with names prefixed with "train/" , "val/" , "test/"
    """
    prefixes = ['train', 'val', 'test']
    result = {}
    for i in range(len(metrics)):
        # Take the latest metrics.
        stats = metrics[i][-1]
        result.update({f"{prefixes[i]}/{k}": v for k, v in stats.items()})
    return result


def cfg_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary.

    Yacs doesn't have a default function to convert the cfg object to plain
    python dict. The following function was taken from
    https://github.com/rbgirshick/yacs/issues/19
    """
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            logging.warning(f"Key {'.'.join(key_list)} with "
                            f"value {type(cfg_node)} is not "
                            f"a valid type; valid types: {_VALID_TYPES}")
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_to_dict(v, key_list + [k])
        return cfg_dict


def make_wandb_name(cfg):
    # Format dataset name.
    dataset_name = cfg.dataset.format
    if dataset_name.startswith('OGB'):
        dataset_name = dataset_name[3:]
    if dataset_name.startswith('PyG-'):
        dataset_name = dataset_name[4:]
    if dataset_name in ['GNNBenchmarkDataset', 'TUDataset']:
        # Shorten some verbose dataset naming schemes.
        dataset_name = ""
    if cfg.dataset.name != 'none':
        dataset_name += "-" if dataset_name != "" else ""
        if cfg.dataset.name == 'LocalDegreeProfile':
            dataset_name += 'LDP'
        else:
            dataset_name += cfg.dataset.name
    # Format model name.
    model_name = cfg.model.type
    if cfg.model.type in ['gnn', 'custom_gnn']:
        model_name += f".{cfg.gnn.layer_type}"
    elif cfg.model.type == 'GPSModel':
        model_name = f"GPS.{cfg.gt.layer_type}"
    model_name += f".{cfg.name_tag}" if cfg.name_tag else ""
    # Compose wandb run name.
    name = f"{dataset_name}.{model_name}.r{cfg.run_id}"
    return name
