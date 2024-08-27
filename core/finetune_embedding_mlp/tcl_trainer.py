import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse
import torch
import torch as th
from utils import *
from typing import Dict
from datasets import load_metric
from transformers import (
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    Trainer,
    IntervalStrategy
)
from utils import seed_everything

import wandb

from graphgps.utility.utils import (
    random_sampling,
    set_cfg,
    get_git_repo_root_path,
    custom_set_run_dir,
    set_printing,
    run_loop_settings,
    create_optimizer,
    config_device,
    create_logger,
    save_run_results_to_csv
)
from model import *
from data_utils.load import load_data_lp, load_graph_lp

METRICS = {  # metric -> metric_path
    'accuracy': 'hf_accuracy.py',
    'f1score': 'hf_f1.py',
    'precision': 'hf_precision.py',
    'recall': 'hf_recall.py',
    'spearmanr': 'hf_spearmanr.py',
    'pearsonr': 'hf_pearsonr.py',

}

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                                default='core/yamls/cora/lms/ft-llama.yaml',
                        help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='The number of starting seed.')
    # parser.add_argument('--device', dest='device', required=False,
    #                    help='device id')
    parser.add_argument('--downsampling', type=float, default=1,
                        help='Downsampling rate.')
    parser.add_argument('--epochs', dest='epoch', type=int, required=False,
                        default=1000)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()


class CLModel(PreTrainedModel):
    def __init__(self, PLM, dropout=0.0, cl_dim=128):
        super().__init__(PLM.config)
        self.dropout = nn.Dropout(dropout)
        hidden_dim = PLM.config.hidden_size
        self.text_encoder = PLM

        self.project = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cl_dim))

    def forward(self, input_ids=None, attention_mask=None, nb_input_ids=None, nb_attention_mask=None):
        # Getting Center Node text features and its neighbours feature
        center_node_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        center_node_emb = self.dropout(center_node_outputs['hidden_states'][-1]).permute(1, 0, 2)[0]

        toplogy_node_outputs = self.text_encoder(
            input_ids=nb_input_ids, attention_mask=nb_attention_mask, output_hidden_states=True
        )

        toplogy_emb = self.dropout(toplogy_node_outputs['hidden_states'][-1]).permute(1, 0, 2)[0]

        center_contrast_embeddings = self.project(center_node_emb)
        toplogy_contrast_embeddings = self.project(toplogy_emb)

        return center_contrast_embeddings, toplogy_contrast_embeddings

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        # forward pass
        center_contrast_embeddings, toplogy_contrast_embeddings = model(**inputs)
        # compute
        loss = infonce(center_contrast_embeddings, toplogy_contrast_embeddings)
        return  loss

class TCLTrainer():
    def __init__(self, cf):
        self.cf = cf
        # logging.set_verbosity_warning()
        from transformers import logging as trfm_logging
        trfm_logging.set_verbosity_error()

    @uf.time_logger
    def train_trainer(self):
        # ! Prepare data
        cf = self.cf
        self.train_data = None 
        # Finetune on dowstream tasks
        train_steps = len(num_train_samples) // cf.eq_batch_size + 1 
        warmup_steps = int(cf.warmup_epochs * train_steps)
        # ! Load Model for NP with no trainer
        PLM = AutoModel.from_pretrained(cf.hf_model) if cf.pretrain_path is None else AutoModel.from_pretrained(
            f'{cf.pretrain_path}')

        #! Freeze the model.encoder layer if cf.freeze is not None
        if cf.freeze is not None:
            for param in PLM.parameters():
                param.requires_grad = False
            if cf.local_rank <= 0:
                trainable_params = sum(
                    p.numel() for p in PLM.parameters() if p.requires_grad
                )
                assert trainable_params == 0
            for param in PLM.encoder.layer[-cf.freeze:].parameters():
                param.requires_grad = True
            if cf.local_rank <= 0:
                trainable_params = sum(
                    p.numel() for p in PLM.parameters() if p.requires_grad
                )
                print(f" Pass the freeze layer, the LM Encoder  parameters are {trainable_params}")

        self.model = CLModel(
                PLM,
                dropout=cf.cla_dropout,
                cl_dim=cf.cl_dim,
            )
        if cf.local_rank <= 0:
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f" LM Model parameters are {trainable_params}")
        if cf.model == 'Distilbert':
            self.model.config.dropout = cf.dropout
            self.model.config.attention_dropout = cf.att_dropout
        else:
            self.model.config.hidden_dropout_prob = cf.dropout
            self.model.config.attention_probs_dropout_prob = cf.att_dropout
        self.log(self.model.config)

        if cf.grad_steps is not None:
            cf.grad_acc_steps = cf.grad_steps
            cf.batch_size = cf.per_device_bsz

        training_args = TrainingArguments(
            output_dir=cf.out_dir,
            learning_rate=cf.lr, weight_decay=cf.weight_decay,
            gradient_accumulation_steps=cf.grad_acc_steps,
            save_total_limit=None,
            report_to='wandb' if cf.wandb_on else None,
            per_device_train_batch_size=cf.batch_size,
            per_device_eval_batch_size=cf.batch_size * 6 if cf.hf_model in {'distilbert-base-uncased',
                                                                            'google/electra-base-discriminator'} else cf.batch_size * 10,
            warmup_steps=warmup_steps,
            disable_tqdm=False,
            dataloader_drop_last=True,
            num_train_epochs=cf.epochs,
            local_rank=cf.local_rank,
            dataloader_num_workers=1,
            fp16=torch.cuda.is_available() # True,
        )

        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
        )
        self.trainer.train()

        if cf.local_rank <= 0:
            if cf.cache_dir is not None:
                print(f'Save the finnal cl model in {cf.cache_dir}')
                PLM.save_pretrained(cf.cache_dir)
                ckpt = f'{cf.cache_dir}{cf.model}.ckpt'
                th.save(self.model.state_dict(), uf.init_path(ckpt))
            else:
                PLM.save_pretrained(cf.out_dir)
                th.save(self.model.state_dict(), uf.init_path(cf.lm.ckpt))
        else:
            print('Dont save the model in the local_rank:', cf.local_rank)



if __name__ == '__main__':
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
   
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    torch.set_num_threads(cfg.num_threads)
    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
    start_ft = time.time()
    for run_id in range(args.repeat):
        seed = run_id + args.start_seed
        custom_set_run_dir(cfg, run_id)

        print_logger = set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        cfg = config_device(cfg)
        
        cfg.seed = seed
        trainer = TCLTrainer(cfg)
        trainer.train()
        start_inf = time.time()
        result_test = trainer.eval_and_save(trainer.test_dataset)
        eval_time = time.time() - start_inf
        
        result_valid = trainer.eval_and_save(trainer.val_dataset)
        result_train = trainer.eval_and_save(trainer.train_dataset)
        result_all = {
            key: (result_train[key], result_valid[key], result_test[key])
            for key in result_test.keys()
        }
        for key, result in result_all.items():
            loggers[key].add_result(run_id, result)

            trainer.tensorboard_writer.add_scalar(f"Metrics/Train/{key}", result[0])
            trainer.tensorboard_writer.add_scalar(f"Metrics/Valid/{key}", result[1])
            trainer.tensorboard_writer.add_scalar(f"Metrics/Test/{key}", result[2])

            train_hits, valid_hits, test_hits = result
            trainer.print_logger.info(
                f'Run: {run_id + 1:02d}, Key: {key}, '
                f'Train: {100 * train_hits:.2f}, Valid: {100 * valid_hits:.2f}, Test: {100 * test_hits:.2f}%')

        trainer.print_logger.info('---')
        save_run_results_to_csv(cfg, loggers, seed, run_id)

    print('All runs:')

    result_dict = {}
    for key in loggers:
        print(key)
        _, _, _, valid_test, _, _ = loggers[key].calc_all_stats()
        result_dict[key] = valid_test

    trainer.save_result(result_dict)

    print_logger.info(f"Results for: {cfg.model.type}")
    print_logger.info(f"Model Params: {trainer.trainable_params}")
    print_logger.info(f'Num parameters: {cfg.model.params}')
    print_logger.info(f"Inference time: {eval_time}")


