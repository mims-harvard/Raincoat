import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import collections
import argparse

from dataloader.dataloader import data_generator, few_shot_data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from utils import AverageMeter


torch.backends.cudnn.benchmark = True  # to fasten TCN


class same_domain_Trainer(object):
    """
   This class contain the main training functions for our pretrainer
    """
    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.data_type = args.selected_dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device


        # Exp Description
        self.run_description = args.da_method + '_' + args.backbone
        self.experiment_description = args.selected_dataset

        # paths
        self.home_path = os.getcwd()
        self.save_dir = 'src_only_saved_models'
        self.data_path = os.path.join(args.data_path, self.data_type)
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels
        self.dataset_configs.final_out_channels = self.dataset_configs.lstm_hid if args.backbone == "LSTM" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.default_hparams = {**self.hparams_class.train_params}


    def train(self):
        run_name = f"{self.run_description}"
        self.hparams = config=self.default_hparams
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        copy_Files(self.exp_log_dir)  # save a copy of training files:

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.

        self.metrics = {'accuracy': [], 'f1_score': []}

        for i in scenarios:
            if self.da_method == 'Source_only':  # training on source and testing on target
                src_id = i[0]
                trg_id = i[1]
            elif self.da_method == 'Target_only':  # training on target and testing target
                src_id = i[1]
                trg_id = i[1]
            else:
                raise NotImplementedError("select the the base method")

            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.data_type, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)

                # Load data
                self.load_data(src_id, trg_id)

                # get algorithm
                algorithm_class = get_algorithm_class('Lower_Upper_bounds')
                backbone_fe = get_backbone_class(self.backbone)
                algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
                algorithm.to(self.device)

                # Average meters
                loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # training..
                for epoch in range(1, self.hparams["num_epochs"] + 1):
                    joint_loaders = enumerate(zip(self.src_train_dl, self.trg_train_dl))
                    len_dataloader = min(len(self.src_train_dl), len(self.trg_train_dl))
                    algorithm.train()

                    for step, ((src_x, src_y), (trg_x, _)) in joint_loaders:
                        src_x, src_y, trg_x = src_x.float().to(self.device), src_y.long().to(self.device), \
                                              trg_x.float().to(self.device)

                        losses = algorithm.update(src_x, src_y)

                        for key, val in losses.items():
                            loss_avg_meters[key].update(val, src_x.size(0))

                    # logging
                    self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                    for key, val in loss_avg_meters.items():
                        self.logger.debug(f'{key}\t: {val.avg:2.4f}')
                    self.logger.debug(f'-------------------------------------')

                self.algorithm = algorithm
                save_checkpoint(self.home_path, self.algorithm, scenarios, self.dataset_configs,
                                self.scenario_log_dir, self.hparams)

                self.evaluate()

                self.calc_results_per_run()

        # logging metrics
        self.calc_overall_results()
        average_metrics = {metric: np.mean(value) for (metric, value) in self.metrics.items()}


    def evaluate(self):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss_ = []

        self.trg_pred_labels = np.array([])
        self.trg_true_labels = np.array([])

        with torch.no_grad():
            for data, labels in self.trg_test_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss

                loss = F.cross_entropy(predictions, labels)

                total_loss_.append(loss.item())

                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.trg_pred_labels = np.append(self.trg_pred_labels, pred.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())

        self.trg_loss = torch.tensor(total_loss_).mean()  # average loss

    def get_configs(self):
        dataset_class = get_dataset_class(self.data_type)
        hparams_class = get_hparams_class(self.data_type)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs,
                                                             self.hparams)
        self.trg_train_dl, self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs,
                                                             self.hparams)
        self.few_shot_dl = few_shot_data_generator(self.trg_test_dl)

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def calc_results_per_run(self):
        '''
        Calculates the acc, f1 and risk values for each cross-domain scenario
        '''
        self.acc, self.f1 = _calc_metrics(self.trg_pred_labels, self.trg_true_labels, self.scenario_log_dir,
                                          self.home_path,
                                          self.dataset_configs.class_names)

        run_metrics = {'accuracy': self.acc, 'f1_score': self.f1}
        for (key, val) in run_metrics.items(): self.metrics[key].append(val)

        df = pd.DataFrame(columns=["acc", "f1"])
        df.loc[0] = [self.acc, self.f1]
        scores_save_path = os.path.join(self.home_path, self.scenario_log_dir, "scores.xlsx")
        df.to_excel(scores_save_path, index=False)
        self.results_df = df

    def calc_overall_results(self):
        exp = self.exp_log_dir


        results = pd.DataFrame(columns=["scenario", "acc", "f1"])

        scenarios_list = os.listdir(exp)
        scenarios_list = [i for i in scenarios_list if "_to_" in i]
        scenarios_list.sort()

        unique_scenarios_names = [f'{i}_to_{j}' for i, j in self.dataset_configs.scenarios]

        for scenario in scenarios_list:
            scenario_dir = os.path.join(exp, scenario)
            scores = pd.read_excel(os.path.join(scenario_dir, 'scores.xlsx'))
            scores.insert(0, 'scenario', '_'.join(scenario.split('_')[:-2]))
            results = pd.concat([results, scores])
        
        avg_results = results.groupby('scenario').mean()
        std_results = results.groupby('scenario').std()

        avg_results.loc[len(avg_results)] = avg_results.mean()
        print(avg_results)
        avg_results.insert(0, "scenario", list(unique_scenarios_names) + ['mean'], True)
        std_results.insert(0, "scenario", list(unique_scenarios_names), True)

        report_save_path_avg = os.path.join(exp, f"Average_results.xlsx")
        report_save_path_std = os.path.join(exp, f"std_results.xlsx")
        
        self.averages_results_df = avg_results
        self.std_results_df = std_results
        avg_results.to_excel(report_save_path_avg)
        std_results.to_excel(report_save_path_std)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='Source_only', type=str,
                        help='Source_only, Target_only')

    # ========= Select the DATASET ==============
    parser.add_argument('--selected_dataset', default='WISDM', type=str,
                        help='Dataset of choice: WISDM, EEG, HAR, HHAR_SA')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str,
                        help='Backbone of choice: CNN - RESNET18 - RESNET18_REDUCED - TCN')

    # ========= Experiment settings ===============
    parser.add_argument('--data_path', default='./data', type=str, help='Path containing dataset')
    parser.add_argument('--num_runs', default=3, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default='cuda:0', type=str, help='cpu or cuda')

    args = parser.parse_args()

    trainer = same_domain_Trainer(args)
    trainer.train()
