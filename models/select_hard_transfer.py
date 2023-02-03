# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:48:14 2022

@author: Huan
"""

# -*- coding: utf-8 -*-

import numpy as np
import sys
sys.path.append('D:\\Transfer-Learning-Time-Series\\Transfer-Learning-Time-Series')
# sys.path.append('D:\\Transfer-Learning-Time-Series\\Transfer-Learning-Time-Series\\models')
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader.dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from sklearn.metrics import f1_score, accuracy_score
from models import classifier


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError

def evaluate( cnn, classifier, eval_loader):

    cnn.eval()
    classifier.eval()
    total_loss_ = []
    trg_pred_labels = np.array([])
    trg_true_labels = np.array([])
   
    all_features = []
    with torch.no_grad():
        for data, labels in eval_loader:
            data = data.float().to('cuda')
            labels = labels.view((-1)).long().to('cuda')
            # XX,r,p = feature_extractor(data)
            features = cnn(data)
            # print(features.shape)
            predictions = classifier(features.detach())
            loss = F.cross_entropy(predictions, labels)
            total_loss_.append(loss.item())
            pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

            trg_pred_labels = np.append(trg_pred_labels, pred.cpu().numpy())
            trg_true_labels = np.append(trg_true_labels, labels.data.cpu().numpy())
            all_features.append(features.cpu().numpy())
    trg_loss = torch.tensor(total_loss_).mean()  # average loss
    f1 = f1_score(trg_pred_labels, trg_true_labels, pos_label=None, average="weighted")
    all_features = np.vstack(all_features)
    labels = np.vstack(trg_true_labels)
    return trg_loss, accuracy_score(trg_true_labels, trg_pred_labels), f1, all_features, labels


class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()
        self.width = configs.input_channels
        self.channel = configs.input_channels
        self.fl =   configs.sequence_len
        self.fc0 = nn.Linear(self.channel, self.width) # input channel is 2: (a(x), x)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels*2, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels*2, configs.mid_channels*2 , kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels*2 ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels*2 , configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat
    
def get_configs():
    dataset_class = get_dataset_class('WISDM')
    hparams_class = get_hparams_class('WISDM')
    return dataset_class(), hparams_class()

class Deep_Coral(Algorithm):
    """
    Deep Coral: https://arxiv.org/abs/1607.01719
    """
    def __init__(self, cnn,  configs, hparams, device):
        super(Deep_Coral, self).__init__(configs)
        self.cnn = cnn(configs).to(device)
        self.classifier = classifier(configs).to(device)
        self.network = nn.Sequential(self.cnn, self.classifier)
        self.optimizer = torch.optim.Adadelta(
            self.network.parameters(),
            # lr=hparams["learning_rate"],
            weight_decay=1e-4
        )
        self.hparams = hparams
       
    def update(self, src_x, src_y):
        self.optimizer.zero_grad()
        src_pred = self.network(src_x)
        Loss = self.cross_entropy(src_pred, src_y)
        Loss.backward()
        self.optimizer.step()
        return Loss


data_path = 'D:\\Transfer-Learning-Time-Series\\Transfer-Learning-Time-Series\\data\\WISDM'

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  
numofruns = 1
num_e = 40
hparams = {"batch_size":32, \
           'learning_rate': 1e-3,    'src_cls_loss_wt': 1,   'coral_wt': 1}
import pandas as pd
import seaborn as sns


df = pd.DataFrame(columns=['run_id','source','target','src_accuracy','trg_accuracy','Gap'])
device = 'cuda'
for r in range(numofruns):
    for t in range(35):
        for s in range(35): 
            dataset_configs, hparams_class = get_configs()
            dataset_configs.final_out_channels = dataset_configs.final_out_channels
            src_train_dl, src_test_dl = data_generator(data_path, str(s) ,dataset_configs,hparams)
            trg_train_dl, trg_test_dl = data_generator(data_path, str(t), dataset_configs,hparams)
            algorithm = Deep_Coral(CNN, dataset_configs, hparams, device)
            for e in range(num_e):
                algorithm.train()
                joint_loaders = enumerate(zip(src_train_dl, trg_train_dl))
                for step, ((src_x, src_y), (trg_x, _)) in joint_loaders:
                        size = src_x.size(0)
                        src_x, src_y, trg_x = src_x.float().to(device), src_y.long().to(device), \
                                                  trg_x.float().to(device)
                        loss = algorithm.update(src_x, src_y)
        
                        # print(loss)
            lt, acc,f1,all_features, labels = evaluate(algorithm.cnn,algorithm.classifier, src_test_dl)
            lt, acc2,f1,all_features2, labels2 = evaluate(algorithm.cnn,algorithm.classifier, trg_test_dl)
            log = {'run_id':r,'source':s,'target':t,'src_accuracy':acc,\
                   'trg_accuracy':acc2,'Gap':acc-acc2}
            df = df.append(log, ignore_index=True)
            print(acc, acc2, acc-acc2)
            


