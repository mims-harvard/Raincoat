import torch
import numpy as np


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + torch.randn_like(x) * sigma


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, (x.shape[0],))

    ret = torch.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat

    return ret


def scaling(x, sigma=0.1):
    # TODO: this was coded based on https://github.com/mims-harvard/TFC-pretraining/blob/main/code/augmentations.py#L88-L95
    # however I have two concerns:
    #  1. should the mean should be 1, not 2, so that on average the magnitude of the data is not changed when sigma = 0?
    #  2. should the shape of the scale should probably be such that the scale remains the same over the time domain,
    #      e.g. torch.randn(1, 1, x.shape[2]) ?
    
    # one random scale per time step, mu = 2
    scale = torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * sigma + 2

    # one random scale per channel, mu = 1
    # scale = torch.randn(x.shape[0], x.shape[1], 1, device=x.device) * sigma + 1
    return x * scale