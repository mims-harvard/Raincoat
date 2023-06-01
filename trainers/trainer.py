import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import collections

from sklearn.metrics import classification_report, accuracy_score
from dataloader.dataloader import data_generator, few_shot_data_generator, generator_percentage_of_data
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics
from algorithms.utils import calc_dev_risk, calculate_risk
from algorithms.algorithms import get_algorithm_class
from algorithms.RAINCOAT import RAINCOAT
from models.models import get_backbone_class
from algorithms.utils import AverageMeter
from sklearn.metrics import f1_score
torch.backends.cudnn.benchmark = True  
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)        

class cross_domain_trainer(object):
    """
   This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        self.run_description = args.run_description
        self.experiment_description = args.experiment_description

        self.best_acc = 0
        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.default_hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}


    def train(self):

        run_name = f"{self.run_description}"
        self.hparams = self.default_hparams
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        df_a = pd.DataFrame(columns=['scenario','run_id','accuracy','f1','H-score'])
        self.trg_acc_list = []
        for i in scenarios:
            src_id = i[0]
            trg_id = i[1]
            loggers = {}
            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                self.fpath = os.path.join(self.home_path, self.scenario_log_dir, 'backbone.pth')
                self.cpath = os.path.join(self.home_path, self.scenario_log_dir, 'classifier.pth')
                self.best_acc = 0
                # Load data
                self.load_data(src_id, trg_id)
                
                # get algorithm
                
                backbone_fe = get_backbone_class(self.backbone)
                if self.da_method == 'RAINCOAT':
                    algorithm = RAINCOAT(self.dataset_configs, self.hparams, self.device)
                else:
                    algorithm_class = get_algorithm_class(self.da_method)
                    algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
                algorithm.to(self.device)
                self.algorithm = algorithm
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

                        if self.da_method in ["DANN", "CoDATS", "AdaMatch"]:
                            losses = algorithm.update(src_x, src_y, trg_x, step, epoch, len_dataloader)
                        else:
                            losses = algorithm.update(src_x, src_y, trg_x)

                        for key, val in losses.items():
                            loss_avg_meters[key].update(val, src_x.size(0))

                    # logging
                    self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                    if self.da_method =='RAINCOAT':
                        acc, f1 = self.eval()
                    else:
                        acc, f1 = self.evaluate()
                    if f1>=self.best_acc:
                        self.best_acc = f1
                        print(self.best_acc)
                        torch.save(self.algorithm.feature_extractor.state_dict(), self.fpath)
                        torch.save(self.algorithm.classifier.state_dict(), self.cpath)
                # self.algorithm = algorithm
                # save_checkpoint(self.home_path, self.algorithm, scenarios, self.dataset_configs,
                #                 self.scenario_log_dir, self.hparams)
                # acc, f1 = self.evaluate(final=True)
                # log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                # df_a = df_a.append(log, ignore_index=True)
            if  self.da_method == 'RAINCOAT':
                print("===== Correct ====")
                for epoch in range(1, self.hparams["num_epochs"] + 1):
                    joint_loaders = enumerate(zip(self.src_train_dl, self.trg_train_dl))
                    len_dataloader = min(len(self.src_train_dl), len(self.trg_train_dl))
                    algorithm.train()
                    for step, ((src_x, src_y), (trg_x, _)) in joint_loaders:
                        src_x, src_y, trg_x = src_x.float().to(self.device), src_y.long().to(self.device), \
                                              trg_x.float().to(self.device)
                        algorithm.correct(src_x, src_y, trg_x)
                    acc, f1 = self.eval()
                    if f1>=self.best_acc:
                        self.best_acc = f1
                        print(self.best_acc)
                        torch.save(self.algorithm.feature_extractor.state_dict(), self.fpath)
                        torch.save(self.algorithm.classifier.state_dict(), self.cpath)
                acc, f1 = self.eval(final=True)
                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                df_a = df_a.append(log, ignore_index=True)
        mean_acc, std_acc, mean_f1, std_f1 = self.avg_result(df_a,'average_align.csv')
        log = {'scenario':mean_acc,'run_id':std_acc,'accuracy':mean_f1,'f1':std_f1}
        df_a = df_a.append(log, ignore_index=True)
        print(df_a)
        path =  os.path.join(self.exp_log_dir, 'average_align.csv')
        df_a.to_csv(path,sep = ',')

    def visualize(self):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        feature_extractor.eval()
        self.trg_pred_labels = np.array([])

        self.trg_true_labels = np.array([])
        self.trg_all_features = []
        self.src_true_labels = np.array([])
        self.src_all_features = []
        with torch.no_grad():
            # for data, labels in self.trg_test_dl:
            for data, labels in self.trg_train_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)
                features = feature_extractor(data)
                self.trg_all_features.append(features.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())
        
            
            for data, labels in self.src_train_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)
                features = feature_extractor(data)
                self.src_all_features.append(features.cpu().numpy())
                self.src_true_labels = np.append(self.src_true_labels, labels.data.cpu().numpy())
            self.src_all_features = np.vstack(self.src_all_features)
            self.trg_all_features = np.vstack(self.trg_all_features)
        
    
    def evaluate(self, final=False):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        # self.algorithm.ema.apply_shadow()
        # # evaluate
        # self.algorithm.ema.restore()
        if final == True:
            feature_extractor.load_state_dict(torch.load(self.fpath))
            classifier.load_state_dict(torch.load(self.cpath))
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
        accuracy = accuracy_score(self.trg_true_labels, self.trg_pred_labels)
        f1 = f1_score(self.trg_pred_labels, self.trg_true_labels, pos_label=None, average="macro")
        return accuracy*100, f1

    def eval(self, final=False):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        if final == True:
            feature_extractor.load_state_dict(torch.load(self.fpath))
            classifier.load_state_dict(torch.load(self.cpath))
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
                features,_ = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.trg_pred_labels = np.append(self.trg_pred_labels, pred.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())
        accuracy = accuracy_score(self.trg_true_labels, self.trg_pred_labels)
        f1 = f1_score(self.trg_pred_labels, self.trg_true_labels, pos_label=None, average="macro")
        return accuracy*100, f1

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
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

    def avg_result(self, df, name):
        mean_acc = df.groupby('scenario', as_index=False, sort=False)['accuracy'].mean()
        mean_f1 = df.groupby('scenario', as_index=False, sort=False)['f1'].mean()
        std_acc = df.groupby('run_id', as_index=False, sort=False)['accuracy'].mean()
        std_f1 =  df.groupby('run_id', as_index=False, sort=False)['f1'].mean()
        return mean_acc.mean().values, std_acc.std().values, mean_f1.mean().values, std_f1.std().values

    
        