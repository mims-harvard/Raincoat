# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:48:14 2022

@author: Huan
"""

import numpy as np
import sys
sys.path.append('/home/huan/Documents/Transfer-Learning-Time-Series-master')
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader.dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from sklearn.metrics import f1_score, accuracy_score
from models import ResClassifier_MME
import pandas as pd
import seaborn as sns
import seaborn as sn
from sklearn.manifold import TSNE
from matplotlib import cm
import numpy as np
from scipy.stats import wasserstein_distance
from sinkhorn import SinkhornSolver
from pytorch_metric_learning import losses
from layers import SinkhornDistance
import ot
from scipy import stats
from sklearn.mixture import GaussianMixture
import copy
import diptest

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        # features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss
    
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


def evaluate(feature_extractor, classifier, eval_loader):
    feature_extractor.eval()
    classifier.eval()
    total_loss_ = []
    trg_pred_labels = np.array([])
    trg_true_labels = np.array([])
   
    all_features = []
    with torch.no_grad():
        for data, labels in eval_loader:
            data = data.float().to('cuda')
            labels = labels.view((-1)).long().to('cuda')

            features,_ = feature_extractor(data)

            # print(features.shape)
            predictions = classifier(features.detach())
            loss = F.cross_entropy(predictions, labels)
            total_loss_.append(loss.item())
            pred = predictions.detach().argmax(dim=1)  
            trg_pred_labels = np.append(trg_pred_labels, pred.cpu().numpy())
            trg_true_labels = np.append(trg_true_labels, labels.data.cpu().numpy())
            all_features.append(features.cpu().numpy())
    trg_loss = torch.tensor(total_loss_).mean()  # average loss
    f1 = f1_score(trg_pred_labels, trg_true_labels, pos_label=None, average="macro")
    all_features = np.vstack(all_features)
    labels = np.vstack(trg_true_labels)
    return trg_loss, accuracy_score(trg_true_labels, trg_pred_labels), f1, all_features, trg_pred_labels
    # return 0, 0, 0, all_features, labels






class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, norm='ortho')

        # Multiply relevant Fourier modes
        # perm = torch.randperm(self.modes1)
        # idx = perm[:self.modes1]
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        # out_ft[:, :, idx] = self.compl_mul1d(x_ft[:, :, idx], self.weights1) 
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1) 
        # x = torch.fft.irfft(out_ft, n=x.size(-1))
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle()
        return torch.concat([r,p],-1), out_ft


class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()
        self.width = configs.input_channels
        self.channel = configs.input_channels
        self.fl =   configs.sequence_len
        self.fc0 = nn.Linear(self.channel, self.width) # input channel is 2: (a(x), x)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.final_out_channels , kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels , configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat



class tf_encoder(nn.Module):
    def __init__(self, configs):
        super(tf_encoder, self).__init__()
        self.modes1 = 32
        self.width = configs.input_channels
        self.channel = configs.input_channels
        self.fl =   configs.sequence_len
        self.fc0 = nn.Linear(self.channel, self.width) # input channel is 2: (a(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.nn = nn.LayerNorm([128])
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.cnn = CNN(configs).to(device)
        self.con1 = nn.Conv1d(3, 1, kernel_size=configs.kernel_size,
                  stride=configs.stride, bias=False, padding=(configs.kernel_size // 2))
        
    def forward(self, x):

        ef, out_ft = self.conv0(x)
        ef = (F.gelu(self.con1(ef).squeeze()))
        # ef =  self.adaptive_pool(ef.permute(0,2,1)).squeeze()
        # print(ef.shape)
        # et = self.nn2(F.gelu(self.gate(x.reshape(x.shape[0], 3*128))))
        et = self.cnn(x)
        f = self.nn(torch.concat([ef,et],-1))
        return f, out_ft

class tf_decoder(nn.Module):
    def __init__(self, configs):
        super(tf_decoder, self).__init__()
        self.nn = nn.LayerNorm([3, 128],eps=1e-04)
        self.nn2 = nn.LayerNorm([3, 128],eps=1e-04)
        self.fc1 = nn.Linear(64, 3*128)
        self.convT = torch.nn.ConvTranspose1d(64, 128,3, stride=1)
        # self.gate = nn.Linear(3* 64, 128)
        # self.conv_block1 = nn.Sequential(
        #     nn.ConvTranspose1d(configs.final_out_channels, configs.mid_channels, kernel_size=3,
        #               stride=1),
        #     nn.BatchNorm1d(configs.mid_channels),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        #     # nn.Dropout(configs.dropout)
        # )

        # self.conv_block2 = nn.Sequential(
        #     nn.ConvTranspose1d(configs.mid_channels, 128 , \
        #                        kernel_size=1, stride=1,padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        # )
        
    def forward(self, f, out_ft):
        freq, time = f.chunk(2,dim=1)
        # _, time = f[:,0:64], f[:,64:]
        x_low = self.nn(torch.fft.irfft(out_ft, n=128, norm='ortho'))
        x_high = self.nn2(F.gelu((self.fc1(time).reshape(-1, 3, 128))))
        # print(x_low.shape, time.shape)
        # x_high = self.nn2(F.relu(self.convT(time.unsqueeze(2))).permute(0,2,1))
        # x_high = self.conv_block1(time.unsqueeze(2))
        # x_high = self.conv_block2(x_high).permute(0,2,1)
        return x_low + x_high
    
def get_configs():
    dataset_class = get_dataset_class('WISDM')
    hparams_class = get_hparams_class('WISDM')
    return dataset_class(), hparams_class()


def drift(feature_extractor, classifier, eval_loader):
    feature_extractor.eval()
    classifier.eval()
    proto = classifier.fc.weight.data
    trg_drift = np.array([])
    cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
    with torch.no_grad():
        # for data, labels in eval_loader:
        data = copy.deepcopy(trg_train_dl.dataset.x_data).float().to('cuda')
        # labels = copy.deepcopy(trg_train_dl.dataset.y_data.view((-1))).long().to('cuda')
            # data = data.float().to('cuda')
            # labels = labels.view((-1)).long().to('cuda')
        features,_ = feature_extractor(data)
        predictions = classifier(features.detach())
        pred_label = torch.argmax(predictions, dim=1)
        
        proto_M = torch.vstack([proto[l,:] for l in pred_label])
        dist = cos(features,proto_M)
        trg_drift = np.append(trg_drift, dist.cpu().numpy())
    return trg_drift

class AAC(Algorithm):
    """
    Deep Coral: https://arxiv.org/abs/1607.01719
    """
    def __init__(self, encoder,decoder, configs, hparams, device):
        super(AAC, self).__init__(configs)
        self.encoder = encoder(configs).to(device)
        self.decoder = decoder(configs).to(device)
        self.classifier = ResClassifier_MME(configs).to(device)
        self.classifier.weights_init()
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())+list(self.decoder.parameters())\
                +list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=1e-4
        )
        self.coptimizer = torch.optim.Adam(
            list(self.encoder.parameters())+list(self.decoder.parameters()),
        # self.encoder.parameters(),
            lr=1*hparams["learning_rate"],
            weight_decay=1e-4
        )
            
        self.hparams = hparams
        self.op_loss = OrthogonalProjectionLoss(gamma=0.5)
        self.mse = nn.L1Loss(reduction='sum').to(device)
        # self.mse = nn.MSELoss(reduction='sum').to(device)
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        # self.loss_func = losses.ContrastiveLoss()
        self.loss_func = losses.TripletMarginLoss()
        self.sink = SinkhornDistance(eps=0.001, max_iter=200)
        
    def align(self, src_x, src_y, trg_x, e):
  
        self.optimizer.zero_grad()
        # self.optimizer2.zero_grad()
        self.classifier.weight_norm()
        src_feat, out = self.encoder(src_x)
        trg_feat, out = self.encoder(trg_x)
        # src_recon = self.decoder(src_feat, out)
        # trg_recon = self.decoder(trg_feat, out)
        # recons = self.mse(src_recon, src_x)+self.mse(trg_recon, src_x)
       
        dr, _, _ = self.sink(src_feat, trg_feat)
        sink_loss = dr
        # sink_loss.backward(retain_graph=True)
        # recons.backward(retain_graph=True)
        # loss=  3*sink_loss +  recons
        # lossinner =  self.loss_func(src_feat, src_y) 
        # lossinner.backward(retain_graph=True)
        lossinner = self.op_loss (src_feat, src_y)
        # lossinner.backward(retain_graph=True)
        src_pred = self.classifier(src_feat)
        
        loss_cls = self.cross_entropy(src_pred, src_y) 
        # loss_cls.backward(retain_graph=True)
        loss = 1 * sink_loss+lossinner+loss_cls + 1*0
        loss.backward()
        self.optimizer.step()
        return {'Src_cls_loss': loss_cls.item(),'Sink': sink_loss.item(),'recon': 0}
    
    def correct(self, trg_x, e):
        self.coptimizer.zero_grad()
        trg_feat, out = self.encoder(trg_x)
        trg_recon = self.decoder(trg_feat, out)
        recons = 1*self.mse(trg_recon, trg_x)
        recons.backward()
        self.coptimizer.step()
        return {'recon': recons.item()}
    

    

data_paths = '/home/huan/Documents/Transfer-Learning-Time-Series-master/data/WISDM'
# data_patht = '/home/huan/Documents/Transfer-Learning-Time-Series-master/data/HHAR_SA'

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  
numofruns = 1
num_e = 40
seeds = [3,5,7,9, 11]
df = pd.DataFrame(columns=['run_id','epoch','loss_align','loss_dis','loss_cls','target_acc'])


for r in range(numofruns):
    hparams = {"batch_size":32, \
               'learning_rate': 5e-4,    'src_cls_loss_wt': 1,   'coral_wt': 1}

    # setup_seed(seeds[r])
    dataset_configs, hparams_class = get_configs()
    dataset_configs.final_out_channels = dataset_configs.final_out_channels
    src_train_dl, src_test_dl = data_generator(data_paths, '2' ,dataset_configs,hparams)
    trg_train_dl, trg_test_dl = data_generator(data_paths, '32', dataset_configs,hparams)
    # yy = copy.deepcopy(trg_train_dl.dataset.y_data)
    # class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
    # bike = np.where(yy==0)
    # sit = np.where(yy==1)
    # stand = np.where(yy==2)
    # walk = np.where(yy==3)
    # up = np.where(yy==4)
    # down = np.where(yy==5)
    
    # # ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
    # trg_train_dl.dataset.y_data[walk] = 0
    # trg_train_dl.dataset.y_data[sit] = 2
    # trg_train_dl.dataset.y_data[stand] = 3
    # trg_train_dl.dataset.y_data[up] = 4
    # trg_train_dl.dataset.y_data[down] = 5
    # trg_train_dl.dataset.y_data[bike] = 1
    
    device = 'cuda'
    algorithm = AAC(tf_encoder,tf_decoder, dataset_configs, hparams, device)
    for e in range(num_e):
        algorithm.train()
        # algorithm.classifier.weight_norm()
        joint_loaders = enumerate(zip(src_train_dl, trg_train_dl))
        loss_c = 0
        loss_r = 0
        loss_s = 0
        for step, ((src_x, src_y), (trg_x, _)) in joint_loaders:
                size = src_x.size(0)
                src_x, src_y, trg_x = src_x.float().to(device), src_y.long().to(device), \
                                          trg_x.float().to(device)

                loss = algorithm.align(src_x, src_y, trg_x, e)
                loss_c += loss['Src_cls_loss']*src_x.size(0)
                loss_s += loss['Sink']*src_x.size(0)
                loss_r += loss['recon']*src_x.size(0)
                # print(loss)
        lt, acc,f1,all_features, _ = evaluate(algorithm.encoder,algorithm.classifier, src_test_dl)
        lt, acc2,f1,all_features2_a, label_a = evaluate(algorithm.encoder,algorithm.classifier,trg_train_dl)
        
        loss_c = loss_c/len(src_train_dl.sampler)
        loss_s = loss_s/len(src_train_dl.sampler)
        log = {'run_id':r,'epoch':e,'loss_cls':loss_c,'target_acc':acc2,\
               'sinkhorn': loss_s, 'recon': loss_r }
        df = df.append(log, ignore_index=True)
        print(acc.item(),acc2.item())
    trg_drift = drift(algorithm.encoder,algorithm.classifier, trg_train_dl)
    print('=====correct====')
    
    for e in range(40):
        algorithm.train()
        joint_loaders = enumerate(zip(src_train_dl, trg_train_dl))
        loss_c = 0
        loss_r = 0
        loss_s = 0
        
        for step, ((src_x, src_y), (trg_x, _)) in joint_loaders:
                size = src_x.size(0)
                src_x, src_y, trg_x = src_x.float().to(device), src_y.long().to(device), \
                                          trg_x.float().to(device)
                loss = algorithm.correct(trg_x, e)
                loss_r += loss['recon']*size
                loss = algorithm.correct(src_x, e)
                loss_r += loss['recon']*size
                
            
            
        lt, acc,f1,all_features, _ = evaluate(algorithm.encoder,algorithm.classifier, src_test_dl)
        lt, acc2,f1,all_features, label_c = evaluate(algorithm.encoder,algorithm.classifier, trg_train_dl)
        
        loss_c = loss_c/len(src_train_dl.sampler)
        loss_s = loss_s/len(src_train_dl.sampler)
        print(acc.item(),acc2.item())
        # log = {'run_id':r,'epoch':num_e+e,'loss_cls':0,'target_acc':acc2,\
        #        'sinkhorn': 0, 'recon': loss_r }
        # df = df.append(log, ignore_index=True)
    trg_drift2 = drift(algorithm.encoder,algorithm.classifier, trg_train_dl)

    
def preprocess_labels(source_loader, target_loader):
    trg_y= copy.deepcopy(target_loader.dataset.y_data)
    src_y = source_loader.dataset.y_data
    pri_c = np.setdiff1d(trg_y, src_y)
    mask = np.isin(trg_y, pri_c)
    trg_y[mask] = -1
    return trg_y, pri_c

# trg_train_dl_p, trg_test_dl = data_generator(data_path, '16', dataset_configs,hparams)

# trg_train_dl_p.dataset.y_data = new_y
drift_diff = np.abs(trg_drift2-trg_drift)

feature_extractor = algorithm.encoder.eval()
classifier = algorithm.classifier.eval()
new_y, pri_c = preprocess_labels(src_train_dl, trg_train_dl)
print(pri_c)
data = copy.deepcopy(trg_train_dl.dataset.x_data).float().to('cuda')
with torch.no_grad():
        features,_ = feature_extractor(data)
        predictions = classifier(features.detach())
        pred_label = predictions.detach().argmax(dim=1).cpu().numpy()
pred_label +=1 
for i in range(1,7):
    cat = np.where(pred_label==i)
    dc = drift_diff[cat]
    print(i-1, dc.shape)
    if dc.shape[0]>3:
        dip, pval = diptest.diptest(dc)
        if dip < 0.05:
            print("contain private")
            gm = GaussianMixture(n_components=2, random_state=0).fit(dc.reshape(-1, 1))
            # labels = gm.predict(dc.reshape(-1, 1))
            # c1, c2 = gm.means_
            c1, c2 = min(gm.means_), max(gm.means_)
            labels = np.zeros(dc.shape[0])
            labels[np.where(dc>c2)] = -1
            labels[np.where(dc<=c2)] = 1
            # if c1<c2:
                # labels[labels==1] = -1 
                # labels[labels==0] = 1
            # else:
            #     labels[labels==0] = -1
            pred_label[cat] = np.multiply(pred_label[cat], labels)
pred_label -=1
pred_label[pred_label<0] = -1

# acc2 = inference(algorithm, trg_train_dl_p, drift_diff, pri_c)
acc2 = accuracy_score(new_y, pred_label)
print(acc2.item())

# new_y = preprocess_labels(src_train_dl, target_loader=trg_train_dl)
# drift_diff = np.abs(trg_drift2-trg_drift)
# acc1 = inference(algorithm, trg_train_dl, drift_diff, new_y)
# print(acc1)

# trg_private = np.where((labels2_c == 2) | (labels2_c == 3 ))[0]
# trg_private = np.where((labels2_c == 1) | (labels2_c == 4 ) | (labels2_c == 2 ) | (labels2_c == 3 ))[0]
# trg_private = np.where((labels2_c == 1))[0]
# fig, ax = plt.subplots(figsize=(7,5))
# drift_diff = np.abs(trg_drift2-trg_drift)
# pri_shift, common_shift = drift_diff[trg_private],drift_diff[~trg_private]
# sns.histplot(common_shift,color='blue', kde=True,label='Target Common ')
# sns.histplot(pri_shift,color='red', kde=True,label='Target Private ')
# plt.xlabel("drift")
# plt.ylabel("frequency")
# plt.legend(fontsize=15)


# print(df[df.epoch==num_e-1].target_acc.mean())
# sns.set_style('whitegrid')
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))
# fig.suptitle('Transfer from {} to {}'.format(2, 35))

# ss = sns.lineplot(x="epoch", y="target_acc", data=df, estimator=np.mean, ci=85,\
#              err_style='bars', ax=axes[0])
# sns.despine()
# axes[0].set_title('Target Accuracy')

# sns.lineplot(x="epoch", y="sinkhorn", data=df, estimator=np.mean, ci=85, \
#              err_style='bars',ax=axes[1])
# sns.despine()
# axes[1].set_title('sinkhorn Loss')

# sns.lineplot(x="epoch", y="recon", data=df, estimator=np.mean, ci=85, \
#              err_style='bars',ax=axes[2])
# sns.despine()
# axes[2].set_title('Recon Loss')




# src_f, src_y, trg_f, trg_y = all_features, labels, all_features2_a, labels2_a
# trg_fc, trg_yc = all_features2_c, labels2_c

# weight = algorithm.classifier.fc.weight.data
# lw = np.array([1,2,3,4,5,6])
# all_f = pd.DataFrame(np.vstack((src_f, trg_f, trg_fc, weight.detach().cpu().numpy())))
# all_y = np.asarray(np.vstack((labels, labels2_a, labels2_c)), dtype=np.float32).squeeze().tolist() \
#     + ['p1','p2','p3','p4','p5','p6']
# tsne = TSNE(n_components=2, verbose=1, perplexity=100, random_state=123)
# z = tsne.fit_transform(all_f)
# df = pd.DataFrame()
# df["y"] = all_y
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]

# df_s = df.iloc[0:len(src_f)]
# df_t = df.iloc[len(src_f):len(src_f)+len(trg_f)]
# df_tc = df.iloc[len(src_f)+len(trg_f):len(src_f)+len(trg_f)+len(trg_fc)]
# df_p = df.iloc[-6:]
# fig, ax = plt.subplots(figsize=(12,12))
# sns.scatterplot(x="comp-1", y="comp-2", hue=df_s.y.tolist(), marker = 'o',s=100,
#                 palette=sns.color_palette("hls", 4),
#                 data=df_s)
# # sns.scatterplot(x="comp-1", y="comp-2", hue=df_t.y.tolist(), marker = 'd',s=100,
# #                 palette=sns.color_palette("hls", 6),
# #                 data=df_t)
# # sns.scatterplot(x="comp-1", y="comp-2", hue=df_tc.y.tolist(), marker = '+',s=100,
# #                 palette=sns.color_palette("hls", 6),
# #                 data=df_tc)
# sns.scatterplot(x="comp-1", y="comp-2", hue=df_p.y.tolist(), marker = 'H',s=300,
#                 palette=sns.color_palette("hls", 6),
#                 data=df_p)
# plt.legend()



# tsne = TSNE(2, verbose=1,  perplexity=50)
# tsne_proj = tsne.fit_transform(all_f)
# tsne_proto = tsne_proj[-6:]
# # Plot those points as a scatter plot and label them based on the pred labels
# cmap = cm.get_cmap('tab20')
# cmap2 = cm.get_cmap('GnBu')
# fig, ax = plt.subplots(figsize=(12,12))
# num_categories = 6
# markers = ['o','d','s','P','*','h']
# for activity in range(1):
#     indices = np.squeeze(trg_y==activity)
#     indices2 = np.squeeze(trg_y==activity)
#     tsne_src, tsne_trg = tsne_proj[0:len(trg_f)], tsne_proj[len(trg_f):len(trg_f)+len(labels2_c)]
    
#     ax.scatter(tsne_src[indices2,0][0:10],tsne_src[indices2,1][0:10], s=50,\
#                 marker=markers[activity], label = activity)
#     ax.scatter(tsne_trg[indices,0][0:10],tsne_trg[indices,1][0:10],s=50,  \
#                 marker=markers[activity], label = activity)
# ax.scatter(tsne_proto[:,0],tsne_proto[:,1],marker='d',s=100, c='black')
# ax.legend(fontsize='large', markerscale=2)
# plt.show()

