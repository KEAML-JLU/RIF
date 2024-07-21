import dgl
from dgl.function.message import _TARGET_MAP
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from typing import Optional
from itertools import chain
from functools import partial
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss, weighted_mse_loss
from graphmae.utils import create_norm, drop_edge


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            K: int,
            mask_gene_rate: float = 0.3,
            remask_rate: float = 0.5,
            num_remask: int = 3,
            spot_num = 0,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            beta_l: float = 1.5,        
            momentum: float = 0.996,
            warm_up: int = 100,
         ):
        super(PreModel, self).__init__()
        
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._num_classes = K
        
        self._mask_gene_rate = mask_gene_rate
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._num_remask = num_remask
        self._remask_rate = remask_rate
        self._momentum = momentum
        self._warm_up = warm_up

        # mask设置
        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        self.mask_token_gene = nn.Parameter(torch.zeros(1, spot_num))
        self.remask_token_spot = nn.Parameter(torch.zeros(1, dec_in_dim))

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1
       

        # 网络结构设置
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.encoder_rec = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.projector_rec = nn.Linear(enc_num_hidden, dec_in_dim, bias=False)
        self.projector_cls1 = nn.Linear(enc_num_hidden, dec_in_dim, bias=False)
        self.projector_cls2 = nn.Linear(enc_num_hidden, dec_in_dim, bias=False)
        self.cls1 = nn.Linear(dec_in_dim, K)
        self.cls2 = nn.Linear(dec_in_dim, K)
        self.cls_rec = nn.Linear(dec_in_dim, K)

        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        
        # 设置损失函数
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l, beta_l)

    def setup_loss_fn(self, loss_fn, alpha_l, beta_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        elif loss_fn == "weighted_mse":
            criterion = partial(weighted_mse_loss, alpha = alpha_l, beta = beta_l)
        else:
            raise NotImplementedError
        return criterion
    
    
    def forward(self, g, x, pseudo_label, certain_spot):           
        use_g, use_x, (mask_genes, _) = self.encoding_mask_noise(g, x["not_scaled"].T, self._mask_gene_rate)            # 该函数是在feat维度做mask，目的是为了让模型更关注基因之间的相关性   
        enc_rec = self.encoder_rec(use_g, use_x)                                                 # encoder相当于是GinCov(aggregae + MLP + BN + ACT + MLP + BN + ACT)的叠加

        # use_g, use_x, (mask_genes, _) = self.encoding_mask_noise(g, x["scaled"].T, self._mask_gene_rate)
        enc = self.encoder(g, x["scaled"])
        
        rep1 = self.projector_cls1(enc)
        rep_rec = self.projector_rec(enc_rec)      
        rep2 = self.projector_cls2(enc_rec)  

        pred1 = self.cls1(rep1)
        pred2 = self.cls2(rep2)
        pred_rec = self.cls_rec(rep_rec)
        
        loss_classify_scaled = nn.CrossEntropyLoss()(pred1[certain_spot["scaled"]], pseudo_label["scaled"][certain_spot["scaled"]])
        loss_classify_rec = nn.CrossEntropyLoss()(pred_rec[certain_spot["not_scaled"]], pseudo_label["not_scaled"][certain_spot["not_scaled"]])
        loss_classify_not_scaled = nn.CrossEntropyLoss()(pred2[certain_spot["not_scaled"]], pseudo_label["not_scaled"][certain_spot["not_scaled"]])

        loss_rec = 0
        for i in range(self._num_remask):
            rep_c = rep_rec.clone()
            rep_c, _, _ = self.random_remask(rep_c, self._remask_rate)
            x_rec = self.decoder(use_g, rep_c)

            x_init = x["not_scaled"][:, mask_genes]
            x_rec = x_rec[:, mask_genes]           
            loss_rec += self.criterion(x_init, x_rec)
        
        # random_dict = {"mask": mask_genes, "noise": noise, "noise_source": noise_source}
        return loss_rec, loss_classify_rec, loss_classify_scaled, loss_classify_not_scaled, pred1, pred2, pred_rec

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        # num_nodes = g.num_nodes()
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]
        
        if self._replace_rate > 0:                                                                        # 添加噪声分为两种类型：noise或者mask(token)
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: -int(self._replace_rate * num_mask_nodes)]]              # 选中添加mask的节点
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]               # 选中添加噪声的节点
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]             # 选择一些节点作为噪声来源

            out_x = x.clone()
            out_x[token_nodes] = 0.0                                                                      # mask掉被选中的mask节点
            out_x[noise_nodes] = x[noise_to_be_chosen]                                                    # 改变noise节点的值
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        out_x[token_nodes] += self.mask_token_gene

        use_g = g.clone()
        return use_g, out_x.T, (mask_nodes, keep_nodes)                                                   # 返回结果graph， 添加noise和token的节点特征矩阵，(所有有噪声的节点序号，所有保持原样的节点序号)


    def random_remask(self, rep, remask_rate=0.5):       
        num_nodes = rep.shape[0]
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes: ]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.remask_token_spot

        return rep, remask_nodes, rekeep_nodes

    def embed(self, g, x):
        use_g, use_x, (_, _) = self.encoding_mask_noise(g, x.T, self._mask_gene_rate) 
        enc_rec = self.encoder_rec(use_g, use_x)
        return enc_rec

    def embed_dec(self, g, x):
        rep = self.encoder(g, x)
        rep = self.projector1(rep)
        return rep

    def get_imputed(self, g, x):
        use_g, use_x, (_, _) = self.encoding_mask_noise(g, x.T, self._mask_gene_rate) 
        enc_rec = self.encoder_rec(use_g, use_x)
        rep_rec = self.projector_rec(enc_rec)
        rep_c, _, _ = self.random_remask(rep_rec, self._remask_rate)
        x_rec = self.decoder(use_g, rep_c)
        return x_rec

    def get_pred1(self, g, x, mask=None):
        enc = self.encoder(g, x, mask=mask)
        rep = self.projector_cls1(enc)
        pred = self.cls1(rep)
        return pred
    
    def get_pred2(self, g, x, random_state_dict=None):
        if isinstance(random_state_dict, dict):
            x = x.T
            x[random_state_dict["noise"]] = x[random_state_dict["noise_source"]]
            x[random_state_dict["mask"]] = 0
            x[random_state_dict["mask"]] += self.mask_token_gene
            x = x.T
        else:
            use_g, use_x, (_, _) = self.encoding_mask_noise(g, x.T, self._mask_gene_rate) 
        enc_rec = self.encoder_rec(use_g, use_x) 
        rep = self.projector_cls2(enc_rec)  
        pred = self.cls2(rep)
        return pred

    def explain_pred2(self, g, x, mask=None):
        enc_rec = self.encoder_rec(g, x, mask=mask) 
        rep = self.projector_cls2(enc_rec)  
        pred = self.cls2(rep)
        return pred

    def finetune(self, g, x):
        use_g, use_x, (mask_genes, _) = self.encoding_mask_noise(g, x.T, self._mask_gene_rate)            # 该函数是在feat维度做mask，目的是为了让模型更关注基因之间的相关性   
        enc_rec = self.encoder_rec(use_g, use_x)                                                 # encoder相当于是GinCov(aggregae + MLP + BN + ACT + MLP + BN + ACT)的叠加
        rep_rec = self.projector_rec(enc_rec)      

        loss_rec = 0
        for i in range(self._num_remask):
            rep_c = rep_rec.clone()
            rep_c, _, _ = self.random_remask(rep_c, self._remask_rate)
            x_rec = self.decoder(use_g, rep_c)

            x_init = x["not_scaled"][:, mask_genes]
            x_rec = x_rec[:, mask_genes]           
            loss_rec += self.criterion(x_init, x_rec)
        
        return loss_rec

    def predict(self, g, x):           
        use_g, use_x, (mask_genes, _) = self.encoding_mask_noise(g, x["not_scaled"].T, self._mask_gene_rate) 
        enc_rec = self.encoder_rec(use_g, use_x)                                                 

        enc = self.encoder(g, x["scaled"])
        
        rep1 = self.projector_cls1(enc)    
        rep2 = self.projector_cls2(enc_rec)  

        pred1 = self.cls1(rep1)
        pred2 = self.cls2(rep2)
        return pred1, pred2, enc_rec






class TransferModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            K: int,
            mask_gene_rate: float = 0.3,
            remask_rate: float = 0.5,
            num_remask: int = 3,
            spot_num = 0,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            balance_class = -1,
            alpha_l: float = 2,
            beta_l: float = 1.5,    
            gamma: float = 2,     
            momentum: float = 0.996,
            warm_up: int = 100,
         ):
        super(TransferModel, self).__init__()
        
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._num_classes = K
        
        self._mask_gene_rate = mask_gene_rate
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._num_remask = num_remask
        self._remask_rate = remask_rate
        self._momentum = momentum
        self._warm_up = warm_up

        # mask设置
        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        self.mask_token_gene = nn.Parameter(torch.zeros(1, spot_num))
        self.remask_token_spot = nn.Parameter(torch.zeros(1, dec_in_dim))

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1
       

        # 网络结构设置
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        self.batch_mlp = nn.Linear(spot_num, 1)

        self.projector_rec = nn.Linear(num_hidden, dec_in_dim, bias=False)
        self.projector_cls = nn.Linear(num_hidden, dec_in_dim, bias=False)
        self.cls = nn.Linear(dec_in_dim, K)

        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=False,
        )

        
        # 设置损失函数        
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l, beta_l)
        if isinstance(balance_class, torch.Tensor):
            self.cls_criterion = FocalLoss(alpha = balance_class, gamma = gamma, num_classes = K)
        else:
            self.cls_criterion = FocalLoss(alpha = torch.ones([K]), gamma = gamma, num_classes = K)

        if remask_rate == 0:
            self._num_remask = 1

    def setup_loss_fn(self, loss_fn, alpha_l = 4, beta_l = 2):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        elif loss_fn == "weighted_mse":
            criterion = partial(weighted_mse_loss, alpha = alpha_l, beta = beta_l)
        else:
            raise NotImplementedError
        return criterion
    
    def forward(self, g, x, label):           
        use_g, use_x, (mask_genes, _) = self.encoding_mask_noise(g, x.T, self._mask_gene_rate)            # 该函数是在feat维度做mask，目的是为了让模型更关注基因之间的相关性   
        if self._mask_gene_rate ==0:
            mask_genes = torch.arange(0, use_x.shape[1])
        gene_bias = self.batch_mlp(x.T)                                                 # encoder相当于是GinCov(aggregae + MLP + BN + ACT + MLP + BN + ACT)的叠加
        enc = self.encoder(g, use_x)
        
        rep_cls = self.projector_cls(enc)
        rep_rec = self.projector_rec(enc)       
        pred = self.cls(rep_cls)
        
        loss_cls = self.cls_criterion(pred, label)
        

        loss_rec = 0
        for i in range(self._num_remask):
            rep_c = rep_rec.clone()
            rep_c, _, _ = self.random_remask(rep_c, self._remask_rate)
            x_rec = self.decoder(use_g, rep_c)
            x_rec = x_rec + gene_bias.T
            if self._mask_gene_rate != 0:
                x_init = x[:, mask_genes]
                x_rec = x_rec[:, mask_genes]  
            else:
                x_init = x         
            loss_rec += self.criterion(x_init, x_rec)
        
        return loss_rec, loss_cls, pred

    def predict(self, g, x):
        use_g, use_x, (mask_genes, _) = self.encoding_mask_noise(g, x.T, self._mask_gene_rate)            # 该函数是在feat维度做mask，目的是为了让模型更关注基因之间的相关性   
        
        gene_bias = self.batch_mlp(x.T)                                                 # encoder相当于是GinCov(aggregae + MLP + BN + ACT + MLP + BN + ACT)的叠加
        enc = self.encoder(g, use_x)
        
        rep_cls = self.projector_cls(enc)
        rep_rec = self.projector_rec(enc)       
        pred = self.cls(rep_cls)
        
        rep_c, _, _ = self.random_remask(rep_rec, self._remask_rate)
        x_rec = self.decoder(use_g, rep_c)
        x_rec = x_rec + gene_bias.T        
        
        return pred, enc, x_rec


    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        # num_nodes = g.num_nodes()
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]
        
        if self._replace_rate > 0:                                                                        # 添加噪声分为两种类型：noise或者mask(token)
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: -int(self._replace_rate * num_mask_nodes)]]              # 选中添加mask的节点
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]               # 选中添加噪声的节点
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]             # 选择一些节点作为噪声来源

            out_x = x.clone()
            out_x[token_nodes] = 0.0                                                                      # mask掉被选中的mask节点
            out_x[noise_nodes] = x[noise_to_be_chosen]                                                    # 改变noise节点的值
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        out_x[token_nodes] += self.mask_token_gene

        use_g = g.clone()
        return use_g, out_x.T, (mask_nodes, keep_nodes)                                                   # 返回结果graph， 添加noise和token的节点特征矩阵，(所有有噪声的节点序号，所有保持原样的节点序号)


    def random_remask(self, rep, remask_rate=0.5):       
        num_nodes = rep.shape[0]
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes: ]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.remask_token_spot

        return rep, remask_nodes, rekeep_nodes

class DEC(nn.Module):
    def __init__(self, nout, K):
        super(DEC, self).__init__()

        self.K = K
        self.nout = nout   
        self.init = nn.Parameter(torch.zeros(size=(K, nout)))                                                                                      # 每轮初始值都一样，合理吗？可以用上一轮中心表示嘛？
        
    def init_center(self, center):
        self.init = nn.Parameter(center, requires_grad=True)

    def forward(self, embeds, alpha=1):                                               
        dist = 1 + ((embeds.unsqueeze(1) - self.init)**2).sum(2) / alpha
        dist = dist.pow(-(alpha + 1.0) / 2.0)     
        q = (dist.T / dist.sum(1)).T

        weight = q**2 / q.sum(0)
        p = (weight.T / weight.sum(1)).T
        loss = F.kl_div(torch.log(q), p)
        return loss, q                               
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):

        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, torch.Tensor):
            assert len(alpha)==num_classes   # α可以以torch方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = alpha
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss