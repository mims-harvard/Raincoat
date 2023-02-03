import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)
np.seterr(all="ignore")
import copy
import diptest
from sklearn.cluster import KMeans
from algorithms.TFAC import TFAC
from sklearn.metrics import classification_report, accuracy_score
from dataloader.uni_dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics, calculate_risk
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
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

        self.run_description = args.experiment_description
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
        copy_Files(self.exp_log_dir)  # save a copy of training files:

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        self.trg_acc_list = []
        df_a = pd.DataFrame(columns=['scenario','run_id','accuracy','f1','H-score'])
        df_c = pd.DataFrame(columns=['scenario','run_id','accuracy','f1','H-score'])
        for i in scenarios:
            src_id = i[0]
            trg_id = i[1]
            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                self.fpath = os.path.join(self.home_path, self.scenario_log_dir, 'backbone.pth')
                self.cpath = os.path.join(self.home_path, self.scenario_log_dir, 'classifier.pth')
                self.best_acc = 0
                # Load data
                self.load_data(src_id, trg_id)
                if self.da_method =='DANCE':
                # get algorithm
                    algorithm_class = get_algorithm_class(self.da_method)
                    backbone_fe = get_backbone_class(self.backbone)
                    algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device, len(self.trg_train_dl.dataset))
                else:
                    algorithm = TFAC(self.dataset_configs, self.hparams, self.device)

                algorithm.to(self.device)
                self.algorithm = algorithm
                tar_uni_label_train, pri_class = self.preprocess_labels(self.src_train_dl, self.trg_train_dl)
                tar_uni_label_test, pri_class = self.preprocess_labels(self.src_train_dl, self.trg_test_dl)
                size_ltrain, size_ltest = len(tar_uni_label_train),len(tar_uni_label_test)
                for epoch in range(1, self.hparams["num_epochs"] + 1):
                    joint_loaders = enumerate(zip(self.src_train_dl, self.trg_train_dl))
                    len_dataloader = min(len(self.src_train_dl), len(self.trg_train_dl))
                    algorithm.train()
                    for step, ((src_x, src_y, _), (trg_x, _, trg_index)) in joint_loaders:
                        src_x, src_y, trg_x, trg_index = src_x.float().to(self.device), src_y.long().to(self.device), \
                                              trg_x.float().to(self.device), trg_index.to(self.device)
                        if self.da_method=='DANCE':
                            algorithm.update(src_x, src_y, trg_x, trg_index, step, epoch, len_dataloader)
                        else: 
                            algorithm.update(src_x, src_y, trg_x)
                    if self.da_method=='DANCE':
                        acc, f1, H = self.evaluate_dance(size_ltest)
                    else:
                        acc, f1, H = self.evaluate_tfac(self.trg_test_dl.dataset.y_data)
                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1,'H-score':H}
                df_a = df_a.append(log, ignore_index=True)
                # Step 2: correct
                

                if self.da_method=='TFAC':
                    print("===== Correct ====")
                    dis2proto_a = self.calc_distance(size_ltrain, self.trg_train_dl)
                    dis2proto_a_test = self.calc_distance(size_ltest, self.trg_test_dl)
                    for epoch in range(1, self.hparams["num_epochs"] + 1):
                        joint_loaders = enumerate(zip(self.src_train_dl, self.trg_train_dl))
                        len_dataloader = min(len(self.src_train_dl), len(self.trg_train_dl))
                        algorithm.train()
                        for step, ((src_x, src_y, _), (trg_x, _, trg_index)) in joint_loaders:
                            src_x, src_y, trg_x, trg_index = src_x.float().to(self.device), src_y.long().to(self.device), \
                                                trg_x.float().to(self.device), trg_index.to(self.device)
                            algorithm.correct(src_x, src_y, trg_x)
                            acc, f1, H = self.evaluate_tfac(self.trg_test_dl.dataset.y_data)
                    dis2proto_c = self.calc_distance(size_ltrain, self.trg_train_dl)
                    dis2proto_c_test = self.calc_distance(size_ltest, self.trg_test_dl)
                    c_list = self.learn_t(dis2proto_a, dis2proto_c)
                    print(c_list)
                    self.trg_true_labels = tar_uni_label_test
                    acc, f1, H = self.detect_private(dis2proto_a_test, dis2proto_c_test, tar_uni_label_test, c_list)
                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1,'H-score':H}
                df_c = df_c.append(log, ignore_index=True)
        self.save_result(df_a,'average_align.csv')
        self.save_result(df_c,'average_correct.csv')


    def detect_private(self, d1, d2, tar_uni_label, c_list):
        diff = np.abs(d2-d1)
        for i in range(6):
            cat = np.where(self.trg_pred_labels==i)
            cc = diff[cat]
            if cc.shape[0]>3:
                dip, pval = diptest.diptest(diff[cat])
                if dip < 0.05:
                    print("contain private")
                    # gm = GaussianMixture(n_components=2, random_state=0,max_iter=5000, n_init=50).fit(diff[cat].reshape(-1, 1))
                    # c =  max(gm.means_)
                    # kmeans = KMeans(n_clusters=2, random_state=0,max_iter=5000, n_init=50, init="random").fit(diff[cat].reshape(-1, 1))
                    # c = max(kmeans.cluster_centers_)
                    c = c_list[i]
                    m1 = np.where(diff>c)
                    m2 = np.where(self.trg_pred_labels==i)
                    mask = np.intersect1d(m1, m2)
                    # print(m1, m2, mask)
                    self.trg_pred_labels[mask] = -1
        accuracy = accuracy_score(tar_uni_label, self.trg_pred_labels)
        f1 = f1_score(self.trg_pred_labels, tar_uni_label, pos_label=None, average="macro")
        return accuracy*100, f1, self.H_score()

    def preprocess_labels(self, source_loader, target_loader):
        trg_y= copy.deepcopy(target_loader.dataset.y_data)
        src_y = source_loader.dataset.y_data
        pri_c = np.setdiff1d(trg_y, src_y)
        mask = np.isin(trg_y, pri_c)
        trg_y[mask] = -1
        return trg_y, pri_c


    def learn_t(self,d1,d2):
        diff = np.abs(d2-d1)
        c_list= []
        for i in range(6):
            cat = np.where(self.trg_train_dl.dataset.y_data==i)
            cc = diff[cat]
            if cc.shape[0]>3:
                dip, pval = diptest.diptest(diff[cat])
                print(i, dip)
                if dip < 0.05:
                    kmeans = KMeans(n_clusters=2, random_state=0,max_iter=5000, n_init=50, init="random").fit(diff[cat].reshape(-1, 1))
                    c = max(kmeans.cluster_centers_)
                else:
                    c = 1e10
            else: 
                c = 1e10
            c_list.append(c)
        return c_list

    def calc_distance(self, len_y, dataloader):
        feature_extractor = self.algorithm.encoder.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        feature_extractor.eval()
        classifier.eval()
        
        proto = classifier.logits.weight.data
        # norm = proto.norm(p=2, dim=1, keepdim=True)
        # proto = proto.div(norm.expand_as(norm))
        trg_drift = np.zeros(len_y)
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)

        with torch.no_grad():
            for data, labels, trg_index in dataloader:
                data = data.float().to('cuda')
                labels = labels.view((-1)).long().to('cuda')
                features,_ = feature_extractor(data)
                predictions = classifier(features.detach())
                pred_label = torch.argmax(predictions, dim=1)
                proto_M = torch.vstack([proto[l,:] for l in pred_label])
                angle_c = cos(features,proto_M)**2
                # dist = (torch.max(predictions,1).values).div(torch.log(angle_c))
                trg_drift[trg_index] = angle_c.cpu().numpy()
        return trg_drift

    def evaluate_dance(self, labels, threshold=1.6):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        feature_extractor.eval()
        classifier.eval()
        data = copy.deepcopy(self.trg_test_dl.dataset.x_data).float().to('cuda')
        labels = labels.view((-1)).long().to(self.device)
        features = feature_extractor(data)
        out_t = classifier(features)
        out_t = F.softmax(out_t,dim=-1)
        entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
        pred = out_t.argmax(dim=1)
        # pred_cls = pred.cpu().numpy()
        pred = pred.cpu().numpy()

        pred_unk = np.where(entr > threshold)
        pred[pred_unk[0]] = -1
        accuracy = accuracy_score(labels.cpu().numpy(), pred)
        f1 = f1_score(pred, labels.cpu().numpy(), pos_label=None, average="macro")
        self.trg_pred_labels = pred
        self.trg_true_labels = labels.cpu().numpy()
        return accuracy*100, f1, self.H_score()

    def evaluate_tfac(self, labels):
        feature_extractor = self.algorithm.encoder.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        feature_extractor.eval()
        classifier.eval()
        data = copy.deepcopy(self.trg_test_dl.dataset.x_data).float().to('cuda')
        labels = labels.view((-1)).long().to(self.device)
        features, _ = feature_extractor(data)
        predictions = classifier(features)
        pred = predictions.argmax(dim=1)
        pred = pred.cpu().numpy()
        accuracy = accuracy_score(labels.cpu().numpy(), pred)
        f1 = f1_score(pred, labels.cpu().numpy(), pos_label=None, average="macro")
        self.trg_pred_labels = pred
        self.trg_true_labels = labels.cpu().numpy()
        return accuracy*100, f1, self.H_score()

    def H_score(self):
        class_c = np.where(self.trg_true_labels!=-1)
        class_p = np.where(self.trg_true_labels==-1)
        label_c, pred_c = self.trg_true_labels[class_c], self.trg_pred_labels[class_c]
        label_p, pred_p = self.trg_true_labels[class_p], self.trg_pred_labels[class_p]
        acc_c = accuracy_score(label_c, pred_c)
        acc_p = accuracy_score(label_p, pred_p)
        # print(acc_c, acc_p)
        if acc_c ==0 or acc_p==0:
            H = 0
        else:
            H = 2*acc_c * acc_p/(acc_p+acc_c)
        return H

    def save_result(self, df, name):
        mean_acc = df.groupby('scenario', as_index=False, sort=False)['accuracy'].mean()
        mean_f1 = df.groupby('scenario', as_index=False, sort=False)['f1'].mean()
        mean_H = df.groupby('scenario', as_index=False, sort=False)['H-score'].mean()
        std_acc = df.groupby('scenario', as_index=False, sort=False)['accuracy'].std()
        std_f1 = df.groupby('scenario', as_index=False, sort=False)['f1'].std()
        std_H = df.groupby('scenario', as_index=False, sort=False)['H-score'].std()
        result = pd.concat(objs=(iDF.set_index('scenario') for iDF in (mean_acc, mean_f1, mean_H,std_acc,std_f1,std_H)),
                axis=1, join='inner').reset_index()
        print(result)
        path =  os.path.join(self.exp_log_dir, name)
        result.to_csv(path)  

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs,
                                                             self.hparams)
        self.trg_train_dl, self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs,
                                                             self.hparams)


    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)