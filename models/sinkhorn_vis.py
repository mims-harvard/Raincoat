# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch 
import sys
sys.path.append('D:\\Transfer-Learning-Time-Series\\Transfer-Learning-Time-Series')
from dataloader.dataloader import data_generator
sns.set_style("whitegrid")
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class

def show_assignments(a, b, P=None, ax=None): 
    if P is not None:
        norm_P = P/P.max()
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                ax.arrow(a[i, 0], a[i, 1], b[j, 0] - a[i, 0], b[j, 1] - a[i, 1],
                         alpha=norm_P[i,j].item(), color="k")

    ax = plt if ax is None else ax
    
    # ax.scatter(*a.t(), color="red")
    # ax.scatter(*b.t(), color="blue")


    
data_path = 'D:\\Transfer-Learning-Time-Series\\Transfer-Learning-Time-Series\\data\\WISDM'
hparams = {"batch_size":32, \
           'learning_rate': 0.01,    'src_cls_loss_wt': 1,   'coral_wt': 1}

def compl_mul1d( input, weights):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", input, weights)
    
def get_configs():
    dataset_class = get_dataset_class('WISDM')
    hparams_class = get_hparams_class('WISDM')
    return dataset_class(), hparams_class()
dataset_configs, hparams_class = get_configs()
dataset_configs.final_out_channels = dataset_configs.final_out_channels

src_train_dl, src_test_dl = data_generator(data_path, '7' ,dataset_configs,hparams)
trg_train_dl, trg_test_dl = data_generator(data_path, '18', dataset_configs,hparams)

f = plt.figure(figsize=(9, 6))
import matplotlib.cm as cm
n = 32
xs,ys = next(iter(src_train_dl))
xt,yt = next(iter(trg_train_dl))
xs_ft = torch.fft.rfft(xs, norm='ortho')
xt_ft = torch.fft.rfft(xt, norm='ortho')
outs_ft = torch.zeros(32, 3, xs.size(-1)//2 + 1,  device=xs.device, dtype=torch.cfloat)
weights = torch.rand(3,3,32,dtype=torch.cfloat)
outs_ft[:, :, :32] = compl_mul1d(xs_ft[:, :, :32], weights) 
rs = outs_ft[:, :, :32].abs().mean([1,2])
ps = outs_ft[:, :, :32].angle().mean([1,2])+torch.acos(torch.zeros(1)).item() * 2

outt_ft = torch.zeros(32, 3, xt.size(-1)//2 + 1,  device=xt.device, dtype=torch.cfloat)
# weights = torch.rand(3,3,32,dtype=torch.cfloat)
outt_ft[:, :, :32] = compl_mul1d(xt_ft[:, :, :32], weights) 
rt = outt_ft[:,:,:32].abs().mean([1,2])
pt = outt_ft[:,:,:32].angle().mean([1,2])+torch.acos(torch.zeros(1)).item() * 2

# sns.scatterplot(rs,ps,hue=ys,cmap=cm.jet,marker='d', s=80)
# sns.scatterplot(rt,pt,hue=yt,cmap=cm.jet,marker='o', s=80)
polar_s = torch.vstack([rs,ps]).t()
polar_t = torch.vstack([rt,pt]).t()
show_assignments(polar_s, polar_t)
from sinkhorn import SinkhornSolver
epsilon = 10**(-(2*2))
solver = SinkhornSolver(epsilon=epsilon, iterations=10000)
cost, pi = solver.forward(polar_s, polar_t)
# f, axarr = plt.subplots(figsize=(18, 9))
# show_assignments(polar_s, polar_t, pi, axarr)
# axarr.set_title("Epsilon: {0:}. Cost: {1:.2f}".format(epsilon, cost))

# cmap = axarr[1, i].imshow(pi)
# axarr.set_title("Probabilistic transport plan of 1 channel 1 freq",fontsize=18)
# cbar = plt.colorbar(cmap, ax=axarr)
# cbar.set_label("Probability mass")
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(pt, rt,color="red")
ax.scatter(ps, rs,color="blue")
# show_assignments(polar_s, polar_t, pi, ax)
# ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)
