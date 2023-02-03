# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:48:14 2022

@author: Huan
"""

# -*- coding: utf-8 -*-

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
from models import classifier
from loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss, NT_Xent
import seaborn as sn
from sklearn.manifold import TSNE
from matplotlib import cm
import numpy as np
from scipy.stats import wasserstein_distance
from sinkhorn import SinkhornSolver
from pytorch_metric_learning import losses
from layers import SinkhornDistance

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

def vis(pt, rt, ps, rs):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(pt, rt,color="red")
    ax.scatter(ps, rs,color="blue")
    # show_assignments(polar_s, polar_t, pi, ax)
    ax.set_rmax(0.5)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

def evaluate(feature_extractor, cnn, classifier, eval_loader):
    feature_extractor.eval()
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
            XX,r,p = feature_extractor(data)
            features = cnn(XX)
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
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle() + self.pi
        return r, p, x

class fno(nn.Module):
    def __init__(self, configs):
        super(fno, self).__init__()
        self.modes1 = 32
        self.width = configs.input_channels
        self.channel = configs.input_channels
        self.fl =   configs.sequence_len
        self.fc0 = nn.Linear(self.channel, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.nn = nn.LayerNorm([3, 128],eps=1e-03)
        self.nn2 = nn.LayerNorm([3, 128],eps=1e-03)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.gate = nn.Linear(3* 128, 3 * 128)
        
    def forward(self, x):
        t = x
        rl = []
        pl = []
        r1, p1,x1 = self.conv0(x)
        rl.append(r1)
        pl.append(p1)
        x2 = self.w0(x)
        # print(x.shape)
        gate = torch.sigmoid(self.gate(x.reshape(-1,3*128)).reshape(-1,3,128))
        x = gate.mul(x1) + (1-gate).mul(x2)
        # x = torch.cat([t,x],1)
        # x = self.nn(x)
        # x = F.gelu(x)
        # r2, p2, x1 = self.conv1(x)
        # rl.append(r2)
        # pl.append(p2)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = self.nn2(x)
        # x = F.relu(x)
        # r3, p3, x1 = self.conv2(x)
        # rl.append(r3)
        # pl.append(p3)
        # x2 = self.w2(x)
        # x = x1 + x2
        return x, rl, pl

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
            nn.Conv1d(configs.mid_channels, configs.mid_channels , kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels ),
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
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat
    
def get_configs():
    dataset_class = get_dataset_class('WISDM')
    hparams_class = get_hparams_class('WISDM')
    return dataset_class(), hparams_class()


# def welch()

def most_frequent(nums):
  return max(set(nums), key = nums.count) 

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


class Inner(nn.Module):
    def __init__(self):
        super(Inner, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse=nn.MSELoss()
    def forward(self, p: torch.tensor, q: torch.tensor):
        n = q.shape[0]
        loss = 0
        for i in range(n):
            loss += self.mse(p, q[i])
        return loss.mean()
    

class Deep_Coral(Algorithm):
    """
    Deep Coral: https://arxiv.org/abs/1607.01719
    """
    def __init__(self, freq, cnn,  configs, hparams, device):
        super(Deep_Coral, self).__init__(configs)

        self.coral = CORAL()

        self.FNO = freq(configs).to(device)
        self.cnn = cnn(configs).to(device)
        self.classifier = classifier(configs).to(device)
        
        self.network = nn.Sequential(self.cnn, self.classifier)
        
        self.optimizer1 = torch.optim.Adam(
            self.FNO.parameters(),
            lr=5*hparams["learning_rate"],
            weight_decay=1e-4
        )
        # self.optimizer1 =torch.optim.SGD( self.FNO.parameters(),lr=0.1, momentum=0.9)
        self.optimizer2 = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=1e-4
        )
        self.hparams = hparams
       
        self.mse = nn.MSELoss()
        self.coral = CORAL()
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.kl = nn.KLDivLoss()
        self.jsd = JSD()
        self.inner = Inner()
        # self.tri = nn.TripletMarginLoss(margin=1.0, p=2)
        self.r_s1 ={1: [],2:[],0:[]}
        # self.loss_func = losses.ContrastiveLoss()
        self.loss_func = losses.TripletMarginLoss()
        self.coral = CORAL()
        self.sink =SinkhornDistance(eps=0.1, max_iter=500)
        self.mmd = MMD_loss()
        
    def align(self, rs, rt):
        return 2*torch.log(rt/rs)+(rs**2-rt**2)/rt**2 

    
    def distance_p(self, p1,p2):
        return min(2 * self.pi- torch.abs(p1-p2),torch.abs(p1-p2))
    
    def welch(self, d):
        return 1 * (1- torch.exp(-1*d**2))
    
    def estimate_theta(self,r):
        return torch.sqrt((r**2).mean(0)/(2)).mean(1)

        
    def update(self, src_x, src_y, trg_x, e):
        
        self.optimizer2.zero_grad()
        # self.network.requires_grad_(False)
        # self.FNO.requires_grad_(True)
        # for i in range(1):
        self.optimizer1.zero_grad()
        src_X,rs,ps = self.FNO(src_x)
        trg_X,rt,pt = self.FNO(trg_x)
        

        # align_loss = 1 * sink_loss - 0 * lossinner
        # align_loss.backward()
        # self.optimizer1.step()        
        src_feat = self.cnn(src_X.detach())
        trg_feat = self.cnn(trg_X.detach())
        coral_loss = self.coral(src_feat, trg_feat)
        src_pred = self.classifier(src_feat)
        
        Loss2 = 1*self.cross_entropy(src_pred, src_y) + 1 * coral_loss
        Loss2.backward()
        self.optimizer2.step()
        lossr,lossp = 0,0
        lossinner = 0
        losss = 0
        # epsilon = 10**(-(2*1))
        # solver = SinkhornSolver(epsilon=epsilon, iterations=1000)
        dim = (1)
        for i in range(len(rs)):
            rs[i] = (rs[i]-rs[i].mean(dim,keepdim=True))/rs[i].std(dim,keepdim=True)
            rt[i] = (rt[i]-rt[i].mean(dim,keepdim=True))/rt[i].std(dim,keepdim=True)
            ps[i] = (ps[i]-ps[i].mean(dim,keepdim=True))/ps[i].std(dim,keepdim=True)
            pt[i] = (pt[i]-pt[i].mean(dim,keepdim=True))/pt[i].std(dim,keepdim=True)
            # with torch.no_grad():
            #     lossr += self.jsd(F.softmax(nrs, dim=0),F.softmax(nrt, dim=0))
            #     lossp += self.jsd(F.softmax(nps, dim=0),F.softmax(npt, dim=0))
            lossinner += 1 * self.loss_func(ps[i].flatten(1), src_y) +\
                1*self.loss_func(rs[i].flatten(1), src_y)
            # with torch.no_grad():
           
            dr, c, p = self.sink(rs[i].cpu(), rt[i].cpu())
            dp, c, p = self.sink(ps[i].cpu(), pt[i].cpu())
            losss += dp.mean() + dr.mean()
            # ds,c,p = self.sink(src_X.cpu(), trg_X.cpu())
            
            # losss += dr.mean()
            # losss += self.mmd(rs[i].flatten(1), rt[i].flatten(1)).mean() + self.mmd(ps[i].flatten(1), pt[i].flatten(1)).mean()
        sink_loss = losss.cuda()
        jsd_loss = 0
        align_loss = 0.1 * sink_loss + 10 * lossinner
        align_loss.backward()
        self.optimizer1.step()   
        # if e == 39:
        # vis(pt[i].mean([1,2]).cpu().detach().numpy(), rt[i].mean([1,2]).cpu().detach().numpy(), \
                # ps[i].mean([1,2]).cpu().detach().numpy(), rs[i].mean([1,2]).cpu().detach().numpy())
        return {'JSD_loss': 0,'Src_cls_loss': Loss2.item(),\
                'Inner':lossinner.item(), 'Sink': losss.item()}


    
data_path = '/home/huan/Documents/Transfer-Learning-Time-Series-master/data/WISDM'

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  
numofruns = 1
num_e = 40
import pandas as pd
import seaborn as sns
seeds = [3,5,7,9, 11]
df = pd.DataFrame(columns=['run_id','epoch','loss_align','loss_dis','loss_cls','target_acc'])
for r in range(numofruns):
    hparams = {"batch_size":32, \
               'learning_rate': 1e-4,    'src_cls_loss_wt': 1,   'coral_wt': 1}

    # setup_seed(seeds[r])
    dataset_configs, hparams_class = get_configs()
    dataset_configs.final_out_channels = dataset_configs.final_out_channels
    src_train_dl, src_test_dl = data_generator(data_path, '7' ,dataset_configs,hparams)
    trg_train_dl, trg_test_dl = data_generator(data_path, '30', dataset_configs,hparams)
    device = 'cuda'
    algorithm = Deep_Coral(fno, CNN, dataset_configs, hparams, device)
    
    for e in range(num_e):
        algorithm.train()
        joint_loaders = enumerate(zip(src_train_dl, trg_train_dl))
        loss_c = 0
        loss_d = 0
        loss_a = 0
        loss_s = 0
        for step, ((src_x, src_y), (trg_x, _)) in joint_loaders:
                size = src_x.size(0)
                src_x, src_y, trg_x = src_x.float().to(device), src_y.long().to(device), \
                                          trg_x.float().to(device)
                loss = algorithm.update(src_x, src_y, trg_x, e)
                loss_c += loss['Src_cls_loss']*src_x.size(0)
                loss_a += loss['Inner']*src_x.size(0)
                loss_d += loss['JSD_loss']*src_x.size(0)
                loss_s += loss['Sink']*src_x.size(0)
                # print(loss)
        lt, acc,f1,all_features, labels = evaluate(algorithm.FNO, algorithm.cnn,algorithm.classifier, src_test_dl)
        lt, acc2,f1,all_features2, labels2 = evaluate(algorithm.FNO, algorithm.cnn,algorithm.classifier, trg_test_dl)
        loss_c = loss_c/len(src_train_dl.sampler)
        loss_a = loss_a/len(src_train_dl.sampler)
        loss_d = loss_d/len(src_train_dl.sampler)
        loss_s = loss_s/len(src_train_dl.sampler)
        log = {'run_id':r,'epoch':e,'loss_align':loss_a,'loss_dis':loss_d,\
               'loss_cls':loss_c,'target_acc':acc2,'sinkhorn': loss_s}
        df = df.append(log, ignore_index=True)
        print(acc2.item(),f1)
    
print(df[df.epoch==num_e-1].target_acc.mean())
sns.set_style('whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Transfer from {} to {}'.format(2, 35))

sns.lineplot(x="epoch", y="target_acc", data=df, estimator=np.mean, ci=85,\
             err_style='bars', ax=axes[0])
sns.despine()
axes[0].set_title('Target Accuracy')

sns.lineplot(x="epoch", y="loss_align", data=df, estimator=np.mean, ci=85, \
             err_style='bars',ax=axes[1])
sns.despine()
axes[1].set_title('Inner Loss ')

sns.lineplot(x="epoch", y="sinkhorn", data=df, estimator=np.mean, ci=85, \
             err_style='bars',ax=axes[2])
sns.despine()
axes[2].set_title('sinkhorn Loss')

# src_f, src_y, trg_f, trg_y = all_features, labels, all_features2,labels2


# all_f = np.vstack((src_f, trg_f))

# tsne = TSNE(2, verbose=1)
# tsne_proj = tsne.fit_transform(all_f)

# # Plot those points as a scatter plot and label them based on the pred labels
# cmap = cm.get_cmap('tab20')
# cmap2 = cm.get_cmap('GnBu')
# fig, ax = plt.subplots(figsize=(8,8))
# num_categories = 6
# colors = ['red','red','red','red','red','red']
# for activity in range(num_categories):
#     indices = np.squeeze(trg_y==activity)
#     indices2 = np.squeeze(src_y==activity)
#     tsne_src, tsne_trg = tsne_proj[0:len(src_f)], tsne_proj[len(src_f):]
#     ax.scatter(tsne_src[indices2,0],tsne_src[indices2,1], s=50,\
#                 marker='o',c=np.array(cmap(activity)).reshape(1,4), label = activity)
#     ax.scatter(tsne_trg[indices,0],tsne_trg[indices,1],s=50,  \
#                 marker='s',c=np.array(cmap(activity)).reshape(1,4), label = activity)
# ax.legend(fontsize='large', markerscale=2)
# plt.show()

