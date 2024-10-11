import json
from datetime import datetime
from data_provider.forecast_dataloader import ForecastDataset, de_normalized
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as torch_data
import os
import time
import warnings
import numpy as np

from utils.math_utils import evaluate

class Exp_StemGNN_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_StemGNN_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def save_model(self, model, model_dir, epoch=None):
        if model_dir is None:
            return
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        epoch = str(epoch) if epoch else ''
        file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
        with open(file_name, 'wb') as f:
            torch.save(model, f)


    def load_model(self, model_dir, epoch=None):
        if not model_dir:
            return
        epoch = str(epoch) if epoch else ''
        file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(file_name):
            return
        with open(file_name, 'rb') as f:
            model = torch.load(f)
        return model


    def inference(self, model, dataloader, device, node_cnt, window_size, horizon):
        forecast_set = []
        target_set = []
        model.eval()
        with torch.no_grad():
            for i, (inputs, target) in enumerate(dataloader):
                inputs = inputs.to(device)
                target = target.to(device)
                step = 0
                forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
                while step < horizon:
                    forecast_result, a = model(inputs)
                    len_model_output = forecast_result.size()[1]
                    if len_model_output == 0:
                        raise Exception('Get blank inference result')
                    inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                    :].clone()
                    inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                    forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                        forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                    step += min(horizon - step, len_model_output)
                forecast_set.append(forecast_steps)
                target_set.append(target.detach().cpu().numpy())
        return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)


    def validate(self, model, dataloader, device, normalize_method, statistic,
                node_cnt, window_size, horizon,
                result_file=None):
        start = datetime.now()
        forecast_norm, target_norm = self.inference(model, dataloader, device,
                                            node_cnt, window_size, horizon)
        if normalize_method and statistic:
            forecast = de_normalized(forecast_norm, normalize_method, statistic)
            target = de_normalized(target_norm, normalize_method, statistic)
        else:
            forecast, target = forecast_norm, target_norm
        score = evaluate(target, forecast)
        score_by_node = evaluate(target, forecast, by_node=True)
        end = datetime.now()

        score_norm = evaluate(target_norm, forecast_norm)
        print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
        print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
        if result_file:
            if not os.path.exists(result_file):
                os.makedirs(result_file)
            step_to_print = 0
            forcasting_2d = forecast[:, step_to_print, :]
            forcasting_2d_target = target[:, step_to_print, :]

            np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
            np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
            np.savetxt(f'{result_file}/predict_abs_error.csv',
                    np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
            np.savetxt(f'{result_file}/predict_ape.csv',
                    np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

        return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                    rmse=score[2], rmse_node=score_by_node[2])


    def train(self, setting, node_cnt, norm_method, optimizer):
        train_data, train_loader = self._get_data(flag='train')
        valid_data, valid_loader = self._get_data(flag='val')
        
        self.model.to(self.device)
        if len(train_data) == 0:
            raise Exception('Cannot organize enough training data')
        if len(valid_data) == 0:
            raise Exception('Cannot organize enough validation data')

        if norm_method == 'z_score':
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        elif norm_method == 'min_max':
            train_min = np.min(train_data, axis=0)
            train_max = np.max(train_data, axis=0)
            normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
        else:
            normalize_statistic = None
        # if normalize_statistic is not None:
        #     with open(os.path.join('result_file', 'norm_stat.json'), 'w') as f:
        #         json.dump(normalize_statistic, f)
        print('HERE')
        if optimizer == 'RMSProp':
            my_optim = torch.optim.RMSprop(params=self.model.parameters(), lr=setting.lr, eps=1e-08)
        else:
            my_optim = torch.optim.Adam(params=self.model.parameters(), lr=setting.lr, betas=(0.9, 0.999))
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=setting.decay_rate)

        # train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
        #                             normalize_method=args.norm_method, norm_statistic=normalize_statistic)
        # valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
        #                             normalize_method=args.norm_method, norm_statistic=normalize_statistic)
        # train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
        #                                     num_workers=0)
        # valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

        forecast_loss = nn.MSELoss(reduction='mean').to(self.device)
        print('TUT')
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Params: {total_params}")

        best_validate_mae = np.inf
        validate_score_non_decrease_count = 0
        performance_metrics = {}
        for epoch in range(setting.train_epochs):
            epoch_start_time = time.time()
            self.model.train()
            loss_total = 0
            cnt = 0
            for i, (inputs, target) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                self.model.zero_grad()
                forecast, _ = self.model(inputs)
                loss = forecast_loss(forecast, target)
                cnt += 1
                loss.backward()
                my_optim.step()
                loss_total += float(loss)
            print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                    time.time() - epoch_start_time), loss_total / cnt))
            self.save_model(self.model, 'result_file', epoch)
            if (epoch+1) % setting.exponential_decay_step == 0:
                my_lr_scheduler.step()
            if (epoch + 1) % setting.validate_freq == 0:
                is_best_for_now = False
                print('------ validate on data: VALIDATE ------')
                performance_metrics = \
                    self.validate(self.model, valid_loader, self.device, setting.norm_method, normalize_statistic,
                            node_cnt, setting.time_step, setting.horizon,
                            result_file='result_file')
                if best_validate_mae > performance_metrics['mae']:
                    best_validate_mae = performance_metrics['mae']
                    is_best_for_now = True
                    validate_score_non_decrease_count = 0
                else:
                    validate_score_non_decrease_count += 1
                # save model
                if is_best_for_now:
                    self.save_model(self.model, 'result_file')
            # early stop
            if setting.early_stop and validate_score_non_decrease_count >= setting.early_stop_step:
                break
        return performance_metrics, normalize_statistic


    def test(self, test_data, args, result_train_file, result_test_file):
        with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
            normalize_statistic = json.load(f)
        model = self.load_model(result_train_file)
        node_cnt = test_data.shape[1]
        test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
        test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                            shuffle=False, num_workers=0)
        performance_metrics = self.validate(model, test_loader, args.device, args.norm_method, normalize_statistic,
                        node_cnt, args.window_size, args.horizon,
                        result_file=result_test_file)
        mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
        print('Performance on test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape, mae, rmse))