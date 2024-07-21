""" 
    Adapted from GNNExplainer
    @misc{ying2019gnnexplainer,
    title={GNNExplainer: Generating Explanations for Graph Neural Networks},
    author={Rex Ying and Dylan Bourgeois and Jiaxuan You and Marinka Zitnik and Jure Leskovec},
    year={2019},
    eprint={1903.03894},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
"""

import math


import dgl
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import scipy.sparse as sp

from .utils import EarlyStopping_loss

class Explainer:
    def __init__(
        self,
        model,
        adj,
        feat,
        pred,
        cluster_id,
        args,
        seed=None,
    ):
        self.model = model
        self.model.eval()
        self.feat = feat
        self.args = args
        self.cluster_id = cluster_id
        self.n_hops = args.num_layers
        self.pred = pred
        self.adj = adj
        self.neighbors = self.get_neighbors_id(cluster_id)
        self.pred = self.pred[self.neighbors]
        self.pos_adj = self.get_positive_adj()

        if args.device==-1 or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = args.device
        self.adj = self.adj.to(self.device)
        self.neighbors = self.neighbors.to(self.device)
        self.pos_adj = self.pos_adj.to(self.device)
        self.pred = self.pred.to(self.device)
        self.seed = seed.to(self.device)


    # Main method
    def explain(self, unconstrained=False):
        # node_idx_new, sub_adj, sub_feat, sub_pred, neighbors = self.extract_neighborhood(node_idx)         
        explainer = ExplainModule(
            args = self.args,
            adj = self.adj,
            x = self.feat,
            model = self.model,
            device = self.device,
            seed_mask = self.seed,
            pos_adj = self.pos_adj,
            neighbors = self.neighbors,
        )
        explainer = explainer.to(self.device)  
        early_stop = EarlyStopping_loss()    

        explainer.train()
        self.model.eval()
        epoch_iter = tqdm(range(self.args.max_epoch))
        for epoch in epoch_iter:
            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            pred = explainer(unconstrained=unconstrained)
            loss = explainer.loss(pred[self.neighbors], self.pred)
            if early_stop(loss):
                break
            loss.backward(retain_graph = True)
            explainer.optimizer.step()

            feat_density = explainer.feat_density()
            epoch_iter.set_description(f"# Epoch {epoch}: loss: {loss.item():.2f}, p_feat: {feat_density.item():.2f}")

        influential_feats = self.influential_feats(explainer.feat_mask)    
        return influential_feats
    
    def get_neighbors_id(self, cluster_id):
        return torch.where(self.pred == cluster_id)[0]
    
    def get_positive_adj(self):
        row = self.neighbors.unsqueeze(1).repeat(1,self.neighbors.shape[0]).reshape(-1)
        col = self.neighbors.repeat(self.neighbors.shape[0])
        data = torch.ones_like(row)
        matrix = sp.coo_matrix((data, (row, col)), shape=(self.adj.shape[0], self.adj.shape[0]))
        return torch.tensor(matrix.todense().A)

    # Utilities
    def extract_neighborhood(self, node_idx):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[node_idx]
        node_idx_new = sum(neighbors_adj_row[:node_idx])                     # 查询节点在新子图中的序号
        neighbors = np.nonzero(neighbors_adj_row)[0]                         # 抽取子图的邻居序号
        sub_adj = self.adj[neighbors][:, neighbors]                          # 抽取有关联的节点所构成的子图          
        sub_feat = self.feat[neighbors]                                      # 抽取相关节点的特征
        sub_pred = self.pred[neighbors]                                      # 抽取相关节点的标签
        neighbors = torch.tensor(neighbors)
        return node_idx_new, sub_adj, sub_feat, sub_pred, neighbors
    

    def get_sub_adj(self, adj):
        """Returns the n_hops degree adjacency matrix adj."""
        adj = torch.tensor(adj, dtype=torch.float)
        adj = adj.to(self.device)
        hop_adj = power_adj = adj                           # adj--初始层, power_adj--当前层, hop_adj--累计邻居
        for i in range(self.n_hops - 1):                    # 迭代计算每一层的邻接矩阵
            power_adj = power_adj @ adj                     # 计算本层的邻接矩阵
            hop_adj = hop_adj + power_adj                   # 累计计算所有层的邻接矩阵
            hop_adj = (hop_adj > 0).float()                 # 二值化
        return hop_adj.cpu().numpy().astype(int)            # 可以理解为，如果每次聚合一阶邻居，第i行是在这么多层里i总共聚合到哪些邻居

    def influential_feats(self, feat_mask2): 
        if self.args.feat_max_num > 0:
            index = torch.argsort(feat_mask2)[-self.args.feat_max_num:]
        else:
            feat_mask2 = torch.sigmoid(feat_mask2)
            mean = torch.mean(feat_mask2)
            std = torch.std(feat_mask2)
            index = torch.where(feat_mask2 > mean + self.args.feat_threshold*std)[0]
        if len(index) < self.args.feat_min_num :
            index = torch.argsort(feat_mask2, descending=True)[:self.args.feat_min_num]
        feat_mask_np = feat_mask2.detach().cpu().numpy()
        np.savetxt("/home/wcy/code/pyFile/NewFolder/GSG_modified_DLPFH/output/SVG/1W_MOB_domain"+str(self.cluster_id)+"_filter50_c7.txt", feat_mask_np)
        return index.detach().cpu().numpy()
    



class ExplainModule(nn.Module):
    def __init__(
        self,
        adj,
        x,
        model,
        mask_graph = False,
        args = None,
        device = "cpu",
        use_sigmoid=True,
        seed_mask = None,
        pos_adj = None,
        neighbors = None,
    ):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.args = args
        self.model = model
        self.model.eval()
        self.device = device
        self.mask_graph = False
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        if (x.sum(1)==0).sum()>0:
            x = x + 1e-6
        self.x = x.to(self.device)
        self.pos_adj = pos_adj
        self.neighbors = neighbors

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.feat_mask = self.construct_feat_mask(x.size(-1), seed_mask, init_strategy="seed")                    # 创建可学习参数，特征的mask，由于所有节点要隐去的特征一样，所以是一维的
        params = [self.feat_mask]
        if mask_graph:
            self.mask, self.mask_bias = self.construct_edge_mask(num_nodes, init_strategy=init_strategy)                  # 创建可学习参数，邻接矩阵的mask
            params.append(self.mask)
            if self.mask_bias is not None:
                params.append(self.mask_bias)
            # For masking diagonal entries
            self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            self.diag_mask = self.diag_mask.to(device)
        self.optimizer = torch.optim.Adam(params, lr=args.lr)
        self.coeffs = {
            "mi": 1.,
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            # "feat_ent": 0.1,
            "feat_ent": 1,
            "grad": 0,
            "con": 1,
        }

    def construct_feat_mask(self, feat_dim, seed_mask=None, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        elif init_strategy == "seed":
            mask = nn.Parameter(seed_mask)
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

        

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2                         # 变成对称矩阵
        adj = self.adj.to(self.device)
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def feat_density(self):
        mask_sum = torch.sum(torch.abs(self.feat_mask)).cpu()
        feat_sum = self.feat_mask.shape[-1]
        return mask_sum / feat_sum

    def forward(self, unconstrained=False, mask_graph=False, marginalize=False):
        feat = {}
        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask        # 对邻接矩阵进行mask
        else:
            feat_mask = (                                                            # 对特征mask使用sigmoid
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
            if marginalize:
                std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                z = torch.normal(mean=mean_tensor, std=std_tensor)
                x = x + z * (1 - feat_mask)
            else:                                                  
                feat = self.x.detach() * feat_mask           # 对x进行mask
            if mask_graph:
                self.masked_adj = self._masked_adj()              

        if mask_graph:
            masked_adj = sp.csr_matrix(self.masked_adj.detach().cpu().numpy())  
        else:
            masked_adj = sp.csr_matrix(self.adj.detach().cpu().numpy())
        graph = dgl.from_scipy(masked_adj).to(self.device)
        pred = self.model.explain_pred2(graph, feat)
        return pred

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
        else:
            x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)
        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad

    def loss(self, pred, pred_label, mi_obj=False):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        if mi_obj:
            pred_loss = -torch.sum(pred_label * torch.log(pred))
        else:
            criterion = nn.CrossEntropyLoss()
            pred_loss = criterion(pred, pred_label.long())        # 交叉熵损失                                        
        pred_loss = self.coeffs["mi"] * pred_loss

        # size 
        feat_mask = torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)                    # p-1损失，限值选取的特征总数

        # entropy  
        feat_mask_ent = - feat_mask * torch.log(feat_mask) - (1-feat_mask) * torch.log(1-feat_mask)                                              
        ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)                  # 熵损失，使得feat_mask取值更趋向于二值化

        # contrast
        masked_x = self.x.detach() * feat_mask
        masked_x = (masked_x.T/torch.norm(masked_x, dim=1, p=2).T).T
        similarity_matrix = masked_x @ masked_x.T
        positive = torch.sum(torch.exp(self.pos_adj[self.neighbors] * similarity_matrix[self.neighbors]), dim=1)
        negative = torch.sum(torch.exp(similarity_matrix[self.neighbors]), dim=1)
        contrast_loss = self.coeffs["con"] * (-torch.mean(torch.log(positive/negative)))

        if self.mask_graph:
            graph_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            size_loss += self.coeffs["mask_size"] * torch.mean(graph_mask)

            graph_mask_ent = - graph_mask * torch.log(graph_mask) - (1-graph_mask) * torch.log(1-graph_mask)    
            ent_loss += self.coeffs["mask_ent"] * torch.mean(graph_mask_ent) 
                                                                            
        loss = pred_loss + contrast_loss + size_loss + ent_loss
        return loss