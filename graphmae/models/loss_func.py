from dgl.function.base import TargetCode
import torch
import torch.nn.functional as F


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def weighted_mse_loss(x, y, alpha=2, beta=1.5):                    # x是原数据，y是重构数据
    
    diff = torch.pow(x - y, alpha) 
    weight = torch.pow(x+1, beta)
    weight.requires_grad = False

    loss = (weight * diff).mean()
    return loss

def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss

def crossEntropyLoss(x, y):
    log_x = -torch.log(x)
    loss = (log_x @ y.T).sum()/x.shape[0]
    return loss

def dec_kl(x, target=None):
    if target==None:
        q = x
        weight = q**2/q.sum(0)
        target = (weight.T/weight.sum(1)).T
    loss = F.kl_div(torch.log(x), target)
    return loss