import os
from matplotlib.pyplot import xcorr
import yaml
import random
import logging
import argparse
from functools import partial

import dgl
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch import optim as optim
from tensorboardX import SummaryWriter
from sklearn.metrics import adjusted_rand_score




logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dataset", type=str, default="Stere-seq")
    parser.add_argument("--data_path", type=str, default="./data/151673.h5ad") # Modified by zpwu
    parser.add_argument("--image_row_name", type=str, default="imagerow",
                        help="row of spatial position") # Modified by zpwu
    parser.add_argument("--image_col_name", type=str, default="imagecol",
                        help="column of spatial position") # Modified by zpwu
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=500,
                        help="number of training epochs") # Modified by zpwu
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--threshold_num", type=int, default=120) # Modified by zpwu
    parser.add_argument("--feature_dim", type=str, default="HVG") # Modified by zpwu
    parser.add_argument("--feature_dim_num", type=int, default=3000) # Modified by zpwu
    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="number of hidden layers") # Modified by zpwu
    parser.add_argument("--num_hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--num_classes", type=int, default=8,
                        help="number of domains") # Modified by zpwu
    parser.add_argument("--groud_truth_column_name", type=str, default="",
                        help="column name of groud truth") # Modified by zpwu
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate") # Modified by zpwu
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="elu") # Modified by zpwu
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="byol")
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=False)
    
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)

    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
def pretrain(args, model, adata, graph, optimizer, scheduler=None):
    print("=============== Start training ===============")
    device = args.device if args.device >= 0 else "cpu"

    certain_spot = {}
    certain_spot["not_scaled"] = (adata.obs["uncertainty"]<args.confidence_threshold).values.nonzero()[0].tolist()
    if len(certain_spot["not_scaled"]) < args.min_pseudo_label:
        certain_spot["not_scaled"] = (adata.obs["uncertainty"] < 1).values.nonzero()[0].tolist()    
    certain_spot["scaled"] = (adata.obs["uncertainty_scaled"]<args.confidence_threshold).values.nonzero()[0].tolist()
    if len(certain_spot["scaled"]) < args.min_pseudo_label:
        certain_spot["scaled"] = (adata.obs["uncertainty_scaled"] < 1).values.nonzero()[0].tolist()
    if args.cluster_label != "":
        true_label = adata[~pd.isnull(adata.obs[args.cluster_label])].obs[args.cluster_label]

    model = model.to(device)
    graph = graph.to(device)
    x = {"not_scaled": graph.ndata["feat"].to(device), 
         "scaled": graph.ndata["feat_scaled"].to(device)}
    pseudo_label = {"not_scaled": torch.tensor(adata.obs["pseudo_label"].values-1, dtype = torch.long).to(device),
                    "scaled": torch.tensor(adata.obs["pseudo_label_scaled"].values-1, dtype = torch.long).to(device)}

    print("===================== Clustering =======================")
    epoch_iter = tqdm(range(args.max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_rec, loss_classify_rec, loss_classify_scaled, loss_classify_not_scaled, pred1, pred2, pred_rec = model(graph, x, pseudo_label, certain_spot)
        loss = 0.001*loss_rec + loss_classify_rec + loss_classify_scaled + loss_classify_not_scaled
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()   

        pred1_label = torch.argmax(pred1, dim=1).detach().cpu().numpy()
        pred2_label = torch.argmax(pred2, dim=1).detach().cpu().numpy()
        pred_rec = torch.argmax(pred_rec, dim=1).detach().cpu().numpy()
        if args.cluster_label != "":
            pred1_label = torch.argmax(pred1, dim=1).detach().cpu().numpy()
            pred_reduce1 = pred1_label[~pd.isnull(adata.obs[args.cluster_label])]
            ari1 = adjusted_rand_score(true_label, pred_reduce1)
            pred_reduce_rec = pred_rec[~pd.isnull(adata.obs[args.cluster_label])]
            ari_rec = adjusted_rand_score(true_label, pred_reduce_rec)
            pred_reduce2 = pred2_label[~pd.isnull(adata.obs[args.cluster_label])]
            ari2 = adjusted_rand_score(true_label, pred_reduce2)
            epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.2f}, ari: {ari1:.2f}, ari: {ari2:.2f}, ari: {ari_rec:.2f}")

    pred1 = nn.Softmax(dim=0)(pred1)
    pred2 = nn.Softmax(dim=0)(pred2)
    torch.save(model.state_dict(), args.output_folder + "/model/" + args.sample_name + ".pth")
    adata.obs["cluster_pred1"] = pred1_label
    adata.obs["cluster_pred2"] = pred2_label
    adata.obs["cluster_recon"] = pred_rec

    print("===================== Imputation =======================")
    epoch_iter = tqdm(range(300))
    for epoch in epoch_iter:
        model.train()
        loss_rec, _, _, _, _, _, _ = model(graph, x, pseudo_label, certain_spot)
        loss = 0.001*loss_rec
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()   

    return model, adata

def transferTrain(args, model, adata_list, graph_list, optimizer, scheduler=None):
    print("===================== Start training =======================")
    device = args.device if args.device >= 0 else "cpu"

    model = model.to(device)
    epoch_iter = tqdm(range(args.max_epoch))
    earlyStop = EarlyStopping(patience=10)
    for epoch in epoch_iter:
        model.train()
        loss_rec_epoch = 0
        loss_cls_epoch = 0
        inputs, obs_seq = batch_adatas(adata_list, graph_list, args.batch_node, device)
        pred_cat = torch.tensor([], device=device)

        for (graph, x, label) in zip(*inputs):
            loss_rec, loss_cls, pred = model(graph, x, label)
            loss_rec_epoch = loss_rec_epoch + loss_rec
            loss_cls_epoch = loss_cls_epoch + loss_cls
            pred_cat = torch.concat([pred_cat, torch.argmax(pred, 1)], 0)

        loss = 0.01*loss_rec_epoch + loss_cls_epoch
        if loss_cls_epoch<1:
            if earlyStop(pred_cat.detach(), obs_seq):
                break                 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.2f}, recon_loss: {loss_rec_epoch:.2f}, cls_loss: {loss_cls_epoch:.2f}") 
    torch.save(model.state_dict(),"/home/wcy/code/pyFile/NewFolder/GSG_modified_DLPFH/output/model/transfer/" + args.target_name + ".pth")
    return model, adata_list

def batch_adatas(adata_list, graph_list, batch_size, device="cpu"):
    sub_graph_list = []
    sub_feat_list = []
    sub_label_list = []
    obs_id_list = []
    slice_id_list = []
    
    for i in range(len(graph_list)):
        node_seq_list = []
        graph = graph_list[i].to(device)
        n_obs = list(range(graph.num_nodes()))
        random.shuffle(n_obs)
        last = 0
        while last + batch_size <= len(n_obs):
            node_index = n_obs[last: last + batch_size]
            node_seq_list = node_seq_list + node_index
            sub_graph = dgl.node_subgraph(graph, node_index)           
            sub_graph_list.append(sub_graph)
            sub_feat_list.append(sub_graph.ndata["feat"])
            sub_label_list.append(sub_graph.ndata['label'])
            obs_id_list.append(adata_list[i].obs_names[node_index].tolist())
            slice_id_list.append([str(i)+'-'] * batch_size)
            last = last + batch_size
            
        if last < len(n_obs):
            node_index = n_obs[-batch_size: ]
            node_seq_list = node_seq_list + node_index
            sub_graph = dgl.node_subgraph(graph, node_index)           
            sub_graph_list.append(sub_graph)
            sub_feat_list.append(sub_graph.ndata["feat"])
            sub_label_list.append(sub_graph.ndata['label'])
            obs_id_list.append(adata_list[i].obs_names[node_index].tolist())
            slice_id_list.append([str(i)+'-'] * batch_size)
    
    sample_seq = list(range(len(sub_graph_list)))
    random.shuffle(sample_seq)
    sub_graph_list = [sub_graph_list[i] for i in sample_seq]
    sub_feat_list = [sub_feat_list[i] for i in sample_seq]
    sub_label_list = [sub_label_list[i] for i in sample_seq]

    slice_list = []
    obs_list = []
    for i in sample_seq:
        slice_list = slice_list + slice_id_list[i]
        obs_list = obs_list + obs_id_list[i]

    df = pd.DataFrame({'slice_id': slice_list, 'obs_id': obs_list})
    obs_seq = (df['slice_id'] + df['obs_id']).values
    return (sub_graph_list, sub_feat_list, sub_label_list), obs_seq


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

# ------ logging ------

class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor
        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)
        sub = tensor - mean * self.mean_scale
        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


class EarlyStopping:
    def __init__(self, patience=10, delta=0.05, verbose=False):
        self.patience= patience
        self.pred = -1
        self.counter = 0
        self.delta = delta
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, pred, index):
        if not isinstance(self.pred, pd.DataFrame):
            self.pred = pd.DataFrame({'pre_pred': pred.cpu()}, index=index)
            self.pred = self.pred[~self.pred.index.duplicated(keep='first')]
        else:
            pred = pd.DataFrame({'pred': pred.cpu()}, index=index)
            pred = pred[~pred.index.duplicated(keep='first')]
            compare = pd.concat([self.pred, pred], axis=1)
            change = (compare['pre_pred'] != compare['pred']).values.sum()
            if change < self.delta * self.pred.shape[0]:  # 如果预测标签稳定
                self.counter += 1
                if self.counter > self.patience:
                    self.early_stop = True
            else:                                        
                self.pred = pred
                self.pred.columns = ['pre_pred']
                self.counter = 0
        return self.early_stop


class EarlyStopping_loss:
    def __init__(self, patience=10, delta=0.001, verbose=False):
        self.patience= patience
        self.min_loss = 1e10
        self.counter = 0
        self.delta = delta
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, loss):
        if self.min_loss == 1e10:
            self.min_loss = loss
        elif loss > self.min_loss - self.delta:             # 如果损失连续提升超过7次，那么提前停止
            self.counter += 1
            if self.min_loss > loss:
                self.min_loss = loss
            if self.counter > self.patience:
                self.early_stop = True
        else:                                                   # 如果损失降低那么保存当前模型    
            self.min_loss = loss
            self.counter = 0
        return self.early_stop