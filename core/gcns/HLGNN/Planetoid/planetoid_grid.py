import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import time
from logger import Logger
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from ogb.linkproppred import Evaluator
from torch.utils.tensorboard import SummaryWriter

from utils import *
from model import *
from Synthetic.regular_tiling import *
from visualization import visualization_geom_fig, visualization_beta
from Synthetic.generate_graph import GraphGeneration

def train(model, predictor, data, split_edge, optimizer, batch_size, writer, epoch):
    predictor.train()
    model.train()
    
    pos_train_edge = split_edge['train']['edge'].to(data.edge_index.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = model(data.x, data.adj_t, data.edge_weight)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long, device=data.x.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    avg_loss = total_loss / total_examples
    writer.add_scalar('Loss/train', avg_loss, epoch)  # Log the training loss

    return avg_loss


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, writer, epoch):
    predictor.eval()
    model.eval()
    h = model(data.x, data.adj_t, data.edge_weight)

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device)
    pos_test_edge = split_edge['test']['edge'].to(data.x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

        # Log the hits@K values
        writer.add_scalar(f'Accuracy/Train_Hits@{K}', train_hits, epoch)
        writer.add_scalar(f'Accuracy/Valid_Hits@{K}', valid_hits, epoch)
        writer.add_scalar(f'Accuracy/Test_Hits@{K}', test_hits, epoch)

    return results

def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--mlp_num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--init', type=str, choices=['SGC', 'RWR', 'KI', 'Random', 'WS', 'Null'], default='KI')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--norm_func', type=str, choices=['gcn_norm', 'col_stochastic_matrix', 'row_stochastic_matrix'], required=True)
    
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    writer = SummaryWriter()
    M = 20
    N = 50
    results_all = dict()
    for m in range(M, M + 1):
        for n in range(N, N + 1):
            for graph_type in ['hexagonal']: #['grid', 'square_grid', 'triangle', 'hexagonal', 'kagome']:
                # for hetero in [False, True]:
                #     for homo in [True, False]:
                #         if homo == hetero:
                #             continue
                for alpha in np.arange(0, 1, 0.1):
                    args.alpha = alpha
                    args.dataset = graph_type
                    homo = False # delete after experiments
                    hetero=True
                    graph = GraphGeneration(m, n, emb_dim=32, graph_type=graph_type, heterophily=hetero, homophily=homo)
                    dataset, _, _ = graph.generate_graph()
                    split_edge = do_edge_split(dataset.clone(), True)
                    data = dataset
                    data.edge_index = split_edge['train']['edge'].t()
        
                    data = T.ToSparseTensor(remove_edge_index=False)(data)
                    data = data.to(device)
                
                    model = HLGNN(data, args).to(device)
                    
                    predictor = LinkPredictor(data.num_features, args.hidden_channels, 1, args.mlp_num_layers, args.dropout).to(device)
                    para_list = list(model.parameters()) + list(predictor.parameters())
                    total_params = sum(p.numel() for param in para_list for p in param)
                    total_params_print = f'Total number of model parameters is {total_params}'
                    print(total_params_print)
                    
                    evaluator = Evaluator(name='ogbl-collab')
                    
                    loggers = {
                        'Hits@10': Logger(args.runs, args),
                        'Hits@50': Logger(args.runs, args),
                        'Hits@100': Logger(args.runs, args),
                    }

                    beta_values = []
                    for run in range(args.runs):
                        model.reset_parameters()
                        predictor.reset_parameters()
                        optimizer = torch.optim.Adam(list(predictor.parameters()) + list(model.parameters()), lr=args.lr)
                        
                        start_time = time.time()
                        for epoch in range(1, 1 + args.epochs):
                            loss = train(model, predictor, data, split_edge, optimizer, args.batch_size, writer, epoch)
                            
                            # Save beta values along with their layer indices on the last epoch
                            if epoch == args.epochs:
                                beta_values_epoch = [(epoch, layer, value.item()) for layer, value in enumerate(model.temp.detach().cpu())]
                                beta_values.extend(beta_values_epoch)

                            if epoch % args.eval_steps == 0:
                                results = test(model, predictor, data, split_edge, evaluator, args.batch_size, writer, epoch)
                                for key, result in results.items():
                                    loggers[key].add_result(run, result)

                                if epoch % args.log_steps == 0:
                                    spent_time = time.time() - start_time
                                    for key, result in results.items():
                                        train_hits, valid_hits, test_hits = result
                                        print(key)
                                        print(f'Run: {run + 1:02d}, '
                                            f'Epoch: {epoch:02d}, '
                                            f'Loss: {loss:.4f}, '
                                            f'Train: {100 * train_hits:.2f}%, '
                                            f'Valid: {100 * valid_hits:.2f}%, '
                                            f'Test: {100 * test_hits:.2f}%')
                                        writer.add_scalar(f'{key}/Train', train_hits, epoch)
                                        writer.add_scalar(f'{key}/Valid', valid_hits, epoch)
                                        writer.add_scalar(f'{key}/Test', test_hits, epoch)
                                    print('---')
                                    print(f'Training Time Per Epoch: {spent_time / args.eval_steps: .4f} s')
                                    print('---')
                                    start_time = time.time()

                        for key in loggers.keys():
                            print(key)
                            loggers[key].print_statistics(run)
                    
                    with open('metrics_and_weights/results.txt', 'a') as f:
                        f.write(f"Type Heuristic:{args.init}, Dataset: {args.dataset}, Norm function: {args.norm_func}\n")

                    for key in loggers.keys():
                        print(key)
                        loggers[key].print_statistics()
                    
                    # Save beta values to a file
                    results_all[alpha] = beta_values
                    with open('metrics_and_weights/beta_values.txt', 'a') as f:
                        f.write(f"Type Heuristic:{args.init}, Dataset: {args.dataset}\n")
                        for epoch, layer, value in beta_values:
                            f.write(f'{epoch}\t{layer}\t{value}\n')
                    print(f"Type Heuristic:{args.init}, Dataset: {args.dataset}\n")
                    print(beta_values)
                    visualization_geom_fig(beta_values, graph_type, m, n, homo, hetero)
    visualization_beta(results_all)
    writer.close()
if __name__ == "__main__":
    main()
