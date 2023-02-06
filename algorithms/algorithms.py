import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.models import classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, AdvSKM_Disc
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss
from models.augmentations import jitter, scaling, permutation
from torch.optim import SGD

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


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


class Lower_Upper_bounds(Algorithm):
    """
    Lower bound: train on source and test on target.
    Upper bound: train on target and test on target.
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(Lower_Upper_bounds, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        self.hparams = hparams

    def update(self, src_x, src_y):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        loss = src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Src_cls_loss': src_cls_loss.item()}


class DANCE(Algorithm):
    """
    Universal Domain Adaptation through Self-Supervision
    https://arxiv.org/abs/2002.07953
    Original code: https://github.com/VisionLearningGroup/DANCE
    """

    class LinearAverage(nn.Module):
        def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
            super().__init__()
            self.nLem = outputSize
            self.momentum = momentum
            self.register_buffer('params', torch.tensor([T, momentum]));
            self.register_buffer('memory', torch.zeros(outputSize, inputSize))
            self.flag = 0
            self.T = T
            # self.memory =  self.memory.cuda()
        def forward(self, x, y):
            out = torch.mm(x, self.memory.t())/self.T
            return out

        def update_weight(self, features, index):
            if not self.flag:
                weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
                weight_pos.mul_(0.0)
                weight_pos.add_(torch.mul(features.data, 1.0))

                w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_weight = weight_pos.div(w_norm)
                self.memory.index_copy_(0, index, updated_weight)
                self.flag = 1
            else:
                weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
                weight_pos.mul_(self.momentum)
                weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

                w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_weight = weight_pos.div(w_norm)
                self.memory.index_copy_(0, index, updated_weight)

            self.memory = F.normalize(self.memory)#.cuda()


        def set_weight(self, features, index):
            self.memory.index_copy_(0, index, features)


    @staticmethod
    def entropy(p):
        p = F.softmax(p,dim=-1)
        return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))

    @staticmethod
    def entropy_margin(p, value, margin=0.2, weight=None):

        def hinge(input, margin=0.2):
            return torch.clamp(input, min=margin)

        p = F.softmax(p, -1)
        return -torch.mean(hinge(torch.abs(-torch.sum(p * torch.log(p+1e-5), 1)-value), margin))


    def __init__(self, backbone_fe, configs, hparams, device, trg_train_size):
        super().__init__(configs)
        
        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)

        self.optimizer = torch.optim.SGD(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            momentum=hparams["momentum"],
            nesterov=True,
        )

        self.lemniscate = self.LinearAverage(configs.features_len * configs.final_out_channels, trg_train_size, hparams["temp"])
        self.device = device
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x, trg_index, step, epoch, len_dataloader):
        total_steps = self.hparams["num_epochs"] + 1 / len_dataloader
        current_step = step + epoch * len_dataloader

        # TODO: weight norm on feature extractor?

        src_feat = self.feature_extractor(src_x)
        src_logits = self.classifier(src_feat)
        src_loss = F.cross_entropy(src_logits, src_y)

        trg_feat = self.feature_extractor(trg_x)
        trg_logits = self.classifier(trg_feat)
        trg_feat = F.normalize(trg_feat)

        # calculate mini-batch x memory similarity
        feat_mat = self.lemniscate(trg_feat, trg_index)

        # do not use memory features present in mini-batch
        feat_mat[:, trg_index] = -1 / self.hparams["temp"]

        # calculate mini-batch x mini-batch similarity
        feat_mat2 = torch.matmul(trg_feat, trg_feat.t()) / self.hparams["temp"]

        mask = torch.eye(feat_mat2.shape[0], feat_mat2.shape[0]).bool().to(self.device)
    
        feat_mat2.masked_fill_(mask, -1 / self.hparams["temp"])

        loss_nc = self.hparams["eta"] * self.entropy(torch.cat([trg_logits, feat_mat, feat_mat2], 1))

        loss_ent = self.hparams["eta"] * self.entropy_margin(trg_logits, self.hparams["thr"], self.hparams["margin"])

        loss = src_loss + loss_nc + loss_ent

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.lemniscate.update_weight(trg_feat, trg_index)

        return {'total_loss': loss.item(), 'src_loss': src_loss.item(), 'loss_nc': loss_nc.item(), 'loss_ent': loss_nc.item()}


class OVANet(Algorithm):
    """
    OVANet https://arxiv.org/pdf/2104.03344v3.pdf
    Based on PyTorch implementation: https://github.com/VisionLearningGroup/OVANet
    """
    def __init__(self, backbone_fe, configs, hparams, device):
        super().__init__(configs)
        
        self.device = device
        self.hparams = hparams
        self.criterion = nn.CrossEntropyLoss()
        
        self.feature_extractor = backbone_fe(configs) # G
        self.classifier1 = classifier(configs) # C1
        
        configs2 = configs
        configs2.num_classes = configs.num_classes * 2
        
        self.classifier2 = classifier(configs2) # C2
        
        self.feature_extractor.to(device)
        self.classifier1.to(device)
        self.classifier2.to(device)
        
        self.opt_g = SGD(self.feature_extractor.parameters(), momentum=self.hparams['sgd_momentum'],
                         lr = self.hparams['learning_rate'], weight_decay=0.0005, nesterov=True)
        self.opt_c = SGD(list(self.classifier1.parameters()) + list(self.classifier2.parameters()), lr=1.0,
                           momentum=self.hparams['sgd_momentum'], weight_decay=0.0005,
                           nesterov=True)
        
        param_lr_g = []
        for param_group in self.opt_g.param_groups:
            param_lr_g.append(param_group["lr"])
        param_lr_c = []
        for param_group in self.opt_c.param_groups:
            param_lr_c.append(param_group["lr"])
        
        self.param_lr_g = param_lr_g
        self.param_lr_c = param_lr_c

    
    @staticmethod
    def _inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=10,
                     power=0.75, init_lr=0.001,weight_decay=0.0005,
                     max_iter=10000):
        #10000
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        #max_iter = 10000
        gamma = 10.0
        lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_lr[i]
            i+=1
        return lr

    def update(self, src_x, src_y, trg_x, step, epoch, len_train_source, len_train_target):
        
        # Applying classifier network => replacing G, C2 in paper
        self.feature_extractor.train()
        self.classifier1.train()
        self.classifier2.train()
        
        self._inv_lr_scheduler(self.param_lr_g, self.opt_g, step,
                         init_lr=self.hparams['learning_rate'],
                         max_iter=self.hparams['min_step'])
        self._inv_lr_scheduler(self.param_lr_c, self.opt_c, step,
                         init_lr=self.hparams['learning_rate'],
                         max_iter=self.hparams['min_step'])
        
        self.opt_g.zero_grad()
        self.opt_c.zero_grad()
        
#         self.classifier2.weight_norm()
        
        ## Source loss calculation
        out_s = self.classifier1(self.feature_extractor(src_x))
        out_open = self.classifier2(self.feature_extractor(src_x))

        ## source classification loss
        loss_s = self.criterion(out_s, src_y)
        ## open set loss for source
        out_open = out_open.view(out_s.size(0), 2, -1)
        open_loss_pos, open_loss_neg = ova_loss(out_open, src_y)
        ## b x 2 x C
        loss_open = 0.5 * (open_loss_pos + open_loss_neg)
        ## open set loss for target
        all = loss_s + loss_open
        
        # OEM - Open Entropy Minimization
        no_adapt = False
        if not no_adapt: # TODO: Figure out if this needs to be altered
            out_open_t = self.classifier2(self.feature_extractor(trg_x))
            out_open_t = out_open_t.view(trg_x.size(0), 2, -1)

            ent_open = open_entropy(out_open_t)
            all += self.hparams['multi'] * ent_open
        
        all.backward()
        
        self.opt_g.step()
        self.opt_c.step()
        self.opt_g.zero_grad()
        self.opt_c.zero_grad()

        return {'src_loss': loss_s.item(),
                'open_loss': loss_open.item(), 
                'open_src_pos_loss': open_loss_pos.item(),
                'open_src_neg_loss': open_loss_neg.item(),
                'open_trg_loss': ent_open.item()
               }

class AdaMatch(Algorithm):
    """
    AdaMatch https://arxiv.org/abs/2106.04732
    Based on PyTorch implementation: https://github.com/zysymu/AdaMatch-pytorch
    """
    def __init__(self, backbone_fe, configs, hparams, device):
        super().__init__(configs)
        
        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.device = device
        self.hparams = hparams

    @staticmethod
    def _enable_batchnorm_tracking(model):
        """start tracking running stats for batch norm"""
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)
        
    @staticmethod
    def _disable_batchnorm_tracking(model):
        """stop tracking running stats for batch norm"""
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)
        
    @staticmethod
    def _compute_src_loss(logits_weak, logits_strong, labels):
        loss_function = nn.CrossEntropyLoss()
        weak_loss = loss_function(logits_weak, labels)
        strong_loss = loss_function(logits_strong, labels)
        return (weak_loss + strong_loss) / 2

    @staticmethod
    def _compute_trg_loss(pseudolabels, logits_strong, mask):
        loss_function = nn.CrossEntropyLoss(reduction="none")
        pseudolabels = pseudolabels.detach()
        loss = loss_function(logits_strong, pseudolabels)
        return (loss * mask).mean()
    
    def augment_weak(self, x):
        return scaling(x, self.hparams["jitter_scale_ratio"])

    def augment_strong(self, x):
        return jitter(permutation(x, max_segments=self.hparams["max_segments"]), self.hparams["jitter_ratio"])
    
    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):

        total_steps = self.hparams["num_epochs"] + 1 / len_dataloader
        current_step = step + epoch * len_dataloader
    
        src_x_weak = self.augment_weak(src_x)
        src_x_strong = self.augment_strong(src_x)

        trg_x_weak = self.augment_weak(trg_x)
        trg_x_strong = self.augment_strong(trg_x)

        x_combined = torch.cat([src_x_weak, src_x_strong, trg_x_weak, trg_x_strong], dim=0)
        src_x_combined = torch.cat([src_x_weak, src_x_strong], dim=0)

        src_total = src_x_combined.shape[0]

        logits_combined = self.classifier(self.feature_extractor(x_combined))
        logits_source_p = logits_combined[:src_total]

        self._disable_batchnorm_tracking(self.feature_extractor)
        self._disable_batchnorm_tracking(self.classifier)
        logits_source_pp = self.classifier(self.feature_extractor(src_x_combined))
        self._enable_batchnorm_tracking(self.feature_extractor)
        self._enable_batchnorm_tracking(self.classifier)

        # random logit interpolation
        lambd = torch.rand_like(logits_source_p)
        final_logits_src = (lambd * logits_source_p) + ((1 - lambd) * logits_source_pp)

        # distribution alignment
        # softmax for logits of weakly augmented source timeseries
        logits_src_weak = final_logits_src[:src_x_weak.shape[0]]
        pseudolabels_src = F.softmax(logits_src_weak, dim=1)

        # softmax for logits of weakly augmented target timeseries
        logits_trg = logits_combined[src_total:]
        logits_trg_weak = logits_trg[:trg_x_weak.shape[0]]
        pseudolabels_trg = F.softmax(logits_trg_weak, dim=1)


        # align target label distribution to source label distribution
        expectation_ratio = (1e-6 + torch.mean(pseudolabels_src)) / (1e-6 + torch.mean(pseudolabels_trg))
        # l2 norm
        final_pseudolabels = F.normalize((pseudolabels_trg * expectation_ratio), p=2, dim=1)

        # relative confidence tresholding
        row_wise_max, _ = torch.max(pseudolabels_src, dim=1)
        final_sum = torch.mean(row_wise_max)

        # relative confidence threshold
        c_tau = self.hparams['tau'] * final_sum

        max_values, _ = torch.max(final_pseudolabels, dim=1)
        mask = (max_values >= c_tau).float()

        src_loss = self._compute_src_loss(logits_src_weak, final_logits_src[src_x_weak.shape[0]:], src_y)

        final_pseudolabels = torch.max(final_pseudolabels, 1)[1]
        trg_loss = self._compute_trg_loss(final_pseudolabels, logits_trg[trg_x_weak.shape[0]:], mask)

        pi = torch.tensor(np.pi, dtype=torch.float).to(self.device)
        mu = 0.5 - torch.cos(torch.minimum(pi, (2 * pi * current_step) / total_steps)) / 2
        loss = src_loss + (mu * trg_loss)
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'total_loss': loss.item(), 'src_loss': src_loss.item(), 'trg_loss': trg_loss.item(), "mu": mu.item(), "current_step": current_step}


class Deep_Coral(Algorithm):
    """
    Deep Coral: https://arxiv.org/abs/1607.01719
    """
    def __init__(self, backbone_fe, configs, hparams, device):
        super(Deep_Coral, self).__init__(configs)

        self.coral = CORAL()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        trg_feat = self.feature_extractor(trg_x)

        coral_loss = self.coral(src_feat, trg_feat)

        loss = self.hparams["coral_wt"] * coral_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class MMDA(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(MMDA, self).__init__(configs)

        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        trg_feat = self.feature_extractor(trg_x)

        coral_loss = self.coral(src_feat, trg_feat)
        mmd_loss = self.mmd(src_feat, trg_feat)
        cond_ent_loss = self.cond_ent(trg_feat)

        loss = self.hparams["coral_wt"] * coral_loss + \
               self.hparams["mmd_wt"] * mmd_loss + \
               self.hparams["cond_ent_wt"] * cond_ent_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                'cond_ent_wt': cond_ent_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class DANN(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DANN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier).to('cuda')

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=1*hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr= hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device
        self.ema = EMA2(self.network, 0.9)
        self.ema.register()
        
    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
               self.hparams["domain_loss_wt"] * domain_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()
        # self.ema.update()
        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class CDAN(Algorithm):
    """
    CDAN: https://arxiv.org/abs/1705.10667
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(CDAN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator_CDAN(configs)
        self.random_layer = RandomLayer([configs.features_len * configs.final_out_channels, configs.num_classes],
                                        configs.features_len * configs.final_out_channels)

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.device = device

    def update(self, src_x, src_y, trg_x):
        # prepare true domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        # source features and predictions
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # target features and predictions
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)
        pred_concat = torch.cat((src_pred, trg_pred), dim=0)

        # Domain classification loss
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # update Domain classification
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # Repeat predictions after updating discriminator
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        # loss of domain discriminator according to fake labels

        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # conditional entropy loss.
        loss_trg_cent = self.criterion_cond(trg_pred)

        # total loss
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["cond_ent_wt"] * loss_trg_cent

        # update feature extractor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'cond_ent_loss': loss_trg_cent.item()}


class DIRT(Algorithm):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DIRT, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams

        # criterion
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.vat_loss = VAT(self.network, device).to(device)

        # device for further usage
        self.device = device

    def update(self, src_x, src_y, trg_x):
        # prepare true domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # target features and predictions
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)

        # Domain classification loss
        disc_prediction = self.domain_classifier(feat_concat.detach())
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # update Domain classification
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # Repeat predictions after updating discriminator
        disc_prediction = self.domain_classifier(feat_concat)

        # loss of domain discriminator according to fake labels
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # conditional entropy loss.
        loss_trg_cent = self.criterion_cond(trg_pred)

        # Virual advariarial training loss
        loss_src_vat = self.vat_loss(src_x, src_pred)
        loss_trg_vat = self.vat_loss(trg_x, trg_pred)
        total_vat = loss_src_vat + loss_trg_vat
        # total loss
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat

        # # update exponential moving average
        # self.ema(self.network)

        # update feature extractor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'cond_ent_loss': loss_trg_cent.item()}


class DSAN(Algorithm):
    """
    DSAN: https://ieeexplore.ieee.org/document/9085896
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DSAN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.loss_LMMD = LMMD_loss(device=device, class_num=configs.num_classes).to(device)

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # calculate lmmd loss
        domain_loss = self.loss_LMMD.get_loss(src_feat, trg_feat, src_y, torch.nn.functional.softmax(trg_pred, dim=1))

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'LMMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class HoMM(Algorithm):
    """
    HoMM: https://arxiv.org/pdf/1912.11976.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(HoMM, self).__init__(configs)

        self.coral = CORAL()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.HoMM_loss = HoMM_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate lmmd loss
        domain_loss = self.HoMM_loss(src_feat, trg_feat)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class DDC(Algorithm):
    """
    DDC: https://arxiv.org/abs/1412.3474
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DDC, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.mmd_loss = MMD_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate mmd loss
        domain_loss = self.mmd_loss(src_feat, trg_feat)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class CoDATS(Algorithm):
    """
    CoDATS: https://arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(CoDATS, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        # we replace the original classifier with codats the classifier
        # remember to use same name of self.classifier, as we use it for the model evaluation
        self.classifier = codats_classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device

    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
               self.hparams["domain_loss_wt"] * domain_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class AdvSKM(Algorithm):
    """
    AdvSKM: https://www.ijcai.org/proceedings/2021/0378.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(AdvSKM, self).__init__(configs)
        self.AdvSKM_embedder = AdvSKM_Disc(configs).to(device)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_disc = torch.optim.Adam(
            self.AdvSKM_embedder.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.mmd_loss = MMD_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)

        source_embedding_disc = self.AdvSKM_embedder(src_feat.detach())
        target_embedding_disc = self.AdvSKM_embedder(trg_feat.detach())
        mmd_loss = - self.mmd_loss(source_embedding_disc, target_embedding_disc)
        mmd_loss.requires_grad = True

        # update discriminator
        self.optimizer_disc.zero_grad()
        mmd_loss.backward()
        self.optimizer_disc.step()

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # domain loss.
        source_embedding_disc = self.AdvSKM_embedder(src_feat)
        target_embedding_disc = self.AdvSKM_embedder(trg_feat)

        mmd_loss_adv = self.mmd_loss(source_embedding_disc, target_embedding_disc)
        mmd_loss_adv.requires_grad = True

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * mmd_loss_adv + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        # update optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'MMD_loss': mmd_loss_adv.item(), 'Src_cls_loss': src_cls_loss.item()}
