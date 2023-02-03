# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 02:33:37 2022

@author: Huan
"""

from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def compute_probs(data, n=10): 
    h, e = np.histogram(data, n)
    p = h/data.shape[0]
    return e, p

def support_intersection(p, q): 
    sup_int = (
        list(
            filter(
                lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
            )
        )
    )
    return sup_int

def get_probs(list_of_tuples): 
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

def kl_divergence(p, q): 
    return np.sum(p*np.log(p/q))

def js_divergence(p, q):
    m = (1./2.)*(p + q)
    return (1./2.)*kl_divergence(p, m) + (1./2.)*kl_divergence(q, m)

def compute_kl_divergence(train_sample, test_sample, n_bins=100): 
    """
    Computes the KL Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    
    return kl_divergence(p, q)

def compute_js_divergence(train_sample, test_sample, n_bins=100): 
    """
    Computes the JS Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)
    
    list_of_tuples = support_intersection(p,q)
    p, q = get_probs(list_of_tuples)
    
    return js_divergence(p, q)

import torch.nn as nn
import torch

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
    
    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
jsd = JSD()

from scipy.stats import rayleigh
import seaborn as sns
import torch.nn.functional as F
from scipy import stats
from scipy.stats import wasserstein_distance
# from sinkhorn import SinkhornSolver
from layers import SinkhornDistance

def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# Create a Continuous Variable: 
X = stats.rayleigh(loc=0, scale=1)
x0 = X.rvs(size=1000, random_state=123)

# Adjust Distribution parameters
loc, scale = rayleigh.fit(x0) 

# Tabulate over sample range (PDF display):
xl = np.linspace(x0.min(), x0.max(), 100)
fig, axe = plt.subplots()
# axe.hist(x0, density=1, label="Sample")
axe.plot(xl, X.pdf(xl), label="Source Distribution, loc=0, scale=5")

js = []
kl = []
wass = []
sink = []
mmd = []
epsilon = 10**(-(2*2))
solver = SinkhornDistance(eps=0.1, max_iter=500)
for i in range(1, 10):
    # X2 = stats.rayleigh(loc=0, scale=i)
    X2 = stats.rayleigh(loc=i, scale=1)
    x1 = X2.rvs(size=1000, random_state=123)
    loc1, scale1 = rayleigh.fit(x1) 
    xl1 = np.linspace(x1.min(), x1.max(), 100)
    # js.append(jsd(F.softmax(torch.tensor(x0)), F.softmax(torch.tensor(x1))))
    js.append(compute_js_divergence(x0, x1))
    kl.append(compute_kl_divergence(x0, x1))
    wass.append(wasserstein_distance(x0,x1))
    cost, p,c = solver.forward(torch.tensor(x0).unsqueeze(1), torch.tensor(x1).unsqueeze(1))
    sink.append(cost)
    ml = mmd_rbf(x0.reshape(1, -1),x1.reshape(1, -1))
    mmd.append(ml)
    axe.plot(xl1, X2.pdf(xl1), label="Target Distribution, loc=0, scale=" + str(i))
axe.set_title("Distribution of Source and Target")
axe.set_xlabel("Variable, $x$")
axe.set_ylabel("Density, $f(x)$")
axe.legend()
axe.grid()

fig, axe = plt.subplots()
axe.semilogy(np.arange(1,10),wass,label='Was')
axe.semilogy(np.arange(1,10),kl,label='KL')
axe.semilogy(np.arange(1,10),js,label='JS')
axe.semilogy(np.arange(1,10),mmd,label='MMD')
axe.semilogy(np.arange(1,10),sink,label='Sinkhorn')
axe.set_title("Different Location")
axe.set_xlabel("Loc, $x$")
axe.set_ylabel("Loss, $f(x)$")
axe.legend()
axe.grid()
plt.ylim(0.1,100)



# X = stats.rayleigh(loc=0, scale=1)
# x0 = X.rvs(size=10000, random_state=123)

# # Adjust Distribution parameters
# loc, scale = rayleigh.fit(x0) 

# # Tabulate over sample range (PDF display):
# xl = np.linspace(x0.min(), x0.max(), 100)
# fig, axe = plt.subplots()
# # axe.hist(x0, density=1, label="Sample")
# axe.plot(xl, X.pdf(xl), label="Source Distribution, loc=0, scale=5")
# sink = []
# js = []
# kl = []
# wass = []
# epsilon = 10**(-(2*2))
# solver = SinkhornDistance(eps=0.1, max_iter=500)
# for i in range(1, 10):
#     X2 = stats.rayleigh(loc=i, scale=1)
#     x1 = X2.rvs(size=10000, random_state=123)
#     loc1, scale1 = rayleigh.fit(x1) 
#     xl1 = np.linspace(x1.min(), x1.max(), 100)
#     # js.append(jsd(F.softmax(torch.tensor(x0)), F.softmax(torch.tensor(x1))))
#     js.append(compute_js_divergence(x0, x1))
#     kl.append(compute_kl_divergence(x0, x1))
#     wass.append(wasserstein_distance(x0,x1))
#     cost, pi = solver.forward(torch.tensor(x0).unsqueeze(1), torch.tensor(x1).unsqueeze(1))
#     sink.append(cost)
#     axe.plot(xl1, X2.pdf(xl1), label="Target Distribution, loc=0, scale=" + str(i))
# axe.set_title("Distribution of Source and Target")
# axe.set_xlabel("Variable, $x$")
# axe.set_ylabel("Density, $f(x)$")
# axe.legend()
# axe.grid()

# fig, axe = plt.subplots()
# axe.plot(np.arange(1,10),wass,label='Was')
# axe.plot(np.arange(1,10),kl,label='KL')
# axe.plot(np.arange(1,10),js,label='JS')
# axe.plot(np.arange(1,10),sink,label='Sinkhorn')
# axe.set_title("Different Loc")
# axe.set_xlabel("Loc, $x$")
# axe.set_ylabel("Loss, $f(x)$")
# axe.legend()
# axe.grid()


