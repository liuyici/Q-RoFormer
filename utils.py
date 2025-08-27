import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import manifold
from typing import Optional
from Weight import Weight
import copy
import os
import argparse

import random
from pyriemann.tangentspace import TangentSpace
from centroid_align import centroid_align
from torch.autograd import Variable


def init_weights(model: nn.Module):
    """
    Network Parameters Initialization Function
    :param model: the model to initialize
    :return: None
    """
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight, 1.0, 0.02)
        nn.init.zeros_(model.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias)
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(model.weight)


def Entropylogits(input, redu='mean'):
    input_ = F.softmax(input, dim=1)
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if redu == 'mean':
        entropy = torch.mean(torch.sum(entropy, dim=1))
    elif redu == 'None':
        entropy = torch.sum(entropy, dim=1)
    return entropy


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def loss_adv(features, ad_net, logits=None):

    ad_out = ad_net(features, logits)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def cosine_matrix(x,y):
    x=F.normalize(x,dim=1)
    y=F.normalize(y,dim=1)
    xty=torch.sum(x.unsqueeze(1)*y.unsqueeze(0),2)
    return 1-xty

def SM(Xs, Xt, Ys, Yt, Cs_memory, Ct_memory, Wt=None, decay=0.3):
    # Clone memory
    Cs = Cs_memory.clone()
    Ct = Ct_memory.clone()

    r = torch.norm(Xs, dim=1)[0]
    Ct = r*Ct / (torch.norm(Ct, dim=1, keepdim=True)+1e-10)
    Cs = r*Cs / (torch.norm(Cs, dim=1, keepdim=True)+1e-10)

    K = Cs.size(0)
    # for each class
    for k in range(K):
        Xs_k = Xs[Ys==k]
        Xt_k = Xt[Yt==k]

        if len(Xs_k)==0:
            Cs_k = 0.0
        else:
            Cs_k = torch.mean(Xs_k,dim=0)

        if len(Xt_k) == 0:
            Ct_k = 0.0
        else:
            if Wt is None:
                Ct_k = torch.mean(Xt_k,dim=0)
            else:
                Wt_k = Wt[Yt==k]
                Ct_k = torch.sum(Wt_k.view(-1, 1) * Xt_k, dim=0) / (torch.sum(Wt_k) + 1e-5)

        Cs[k, :] = (1-decay) * Cs_memory[k, :] + decay * Cs_k
        Ct[k, :] = (1-decay) * Ct_memory[k, :] + decay * Ct_k

    Dist = cosine_matrix(Cs, Ct)

    return torch.sum(torch.diag(Dist)), Cs, Ct

def robust_pseudo_loss(output,label,weight,q=1.0):
    weight[weight<0.5] = 0.0
    one_hot_label=torch.zeros(output.size()).scatter_(1,label.cpu().view(-1,1),1).cuda()
    mask=torch.eq(one_hot_label,1.0)
    output=F.softmax(output,dim=1)
    mae=(1.0-torch.masked_select(output,mask)**q)/q
    return torch.sum(weight*mae)/(torch.sum(weight)+1e-10)

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)
    
    
def seed(seed: Optional[int] = 0):
    """
    fix all the random seed
    :param seed: random seed
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def temperature_normalization(x: torch.Tensor,
                              t: Optional[float] = 0.1) -> torch.Tensor:
    """
    L2 Normalization for bottleneck module output with temperature factor
    :param x: feature tensor
    :param t: temperature factor
    :return: t-normalized feature
    """
    n_feature = x.shape[1]
    norm = torch.norm(x, p=2, dim=1).view(-1, 1).repeat(1, n_feature)
    return x / (norm * t)



def augment(data):
    data_new = copy.deepcopy(data).cpu()
    ca = centroid_align(center_type='riemann', cov_type='lwf')
    Ca_noTral,new_data_x = ca.fit_transform(data_new)
    tanTrans = TangentSpace().fit(Ca_noTral)
    tan = tanTrans.transform(Ca_noTral)
    return Variable(torch.from_numpy(tan).type(torch.FloatTensor))



def centroid_A(data):
    data_new = copy.deepcopy(data)
    ca = centroid_align(center_type='riemann', cov_type='lwf')
    Ca_noTral,new_data_x = ca.fit_transform(data_new)
    new_X = np.transpose(new_data_x, (0, 2, 1))
    # tanTrans = TangentSpace().fit(Ca_noTral)
    # tan = tanTrans.transform(Ca_noTral)
    return Variable(torch.from_numpy(new_X).type(torch.FloatTensor))


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

def linear_mmd2( f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)#求矩阵的内积：即将矩阵内的元素依次点乘，然后再将所有的点乘结果相加，得到一位数的结果
        return loss

def marginal(source, target, kernel_type='rbf', kernel_mul=2.0, kernel_num=5,fix_sigma = None):
        if kernel_type == 'linear':
            return linear_mmd2(source, target)
        elif kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = guassian_kernel(
                source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

def conditional(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = Weight.cal_weight(
            s_label, t_label, type='visual')
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

        kernels = guassian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss
    
    
# 损失函数
def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim = 1)- F.softmax(out2,dim = 1)))



class SKLDivLoss(nn.Module):
    """
    Symmetric KL Divergence Loss for BAIT
    """

    def __init__(self):
        super(SKLDivLoss, self).__init__()

    def forward(self,
                out1: torch.Tensor,
                out2: torch.Tensor) -> torch.Tensor:
        out2_t = out2.clone()
        out2_t = out2_t.detach()
        out1_t = out1.clone()
        out1_t = out1_t.detach()
        return (F.kl_div(out1.log(), out2_t, reduction='none') +
                F.kl_div(out2.log(), out1_t, reduction='none')) / 2
        
class LabelSmooth(nn.Module):
    """
    Label smooth cross entropy loss

    Parameters:
        - **num_class** (int): num of classes
        - **alpha** Optional(float): the smooth factor
        - **device** Optional(str): the used device "cuda" or "cpu"
    """

    def __init__(self,
                 num_class: int,
                 alpha: Optional[float] = 0.1,
                 device: Optional[str] = "cuda"):
        super(LabelSmooth, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).to(self.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.alpha) * targets + self.alpha / self.num_class
        loss = (-targets * log_probs).sum(dim=1)
        return loss.mean()


def BNM(src, tar):
    """ Batch nuclear-norm maximization, CVPR 2020.
    tar: a tensor, softmax target output.
    NOTE: this does not require source domain data.
    """
    _, out, _ = torch.svd(tar)
    loss = -torch.mean(out)
    return loss

def CORAL_loss(source, target):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss



def Entropy(input: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy
    :param input: the softmax output
    :return: entropy
    """
    entropy = -input * torch.log(input + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class TEntropyLoss(nn.Module):
    """
    The Tsallis Entropy for Uncertainty Reduction

    Parameters:
        - **t** Optional(float): the temperature factor used in TEntropyLoss
        - **order** Optional(float): the order of loss function
    """

    def __init__(self,
                 t: Optional[float] = 2.0,
                 order: Optional[float] = 2.0):
        super(TEntropyLoss, self).__init__()
        self.t = t
        self.order = order

    def forward(self,
                output: torch.Tensor) -> torch.Tensor:
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        entropy_weight = entropy_weight.repeat(1, n_class)
        tentropy = torch.pow(softmax_out, self.order) * entropy_weight
        # weight_softmax_out=softmax_out*entropy_weight
        tentropy = tentropy.sum(dim=0) / softmax_out.sum(dim=0)
        loss = -torch.sum(tentropy) / (n_class * (self.order - 1.0))
        return loss