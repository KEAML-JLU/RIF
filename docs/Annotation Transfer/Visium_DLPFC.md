---
sort: 1
---

# 10X Visium DLPFC

Tutorial for annotation transfer for DLPFC slice 151674.

```python
import os
import warnings
import argparse

import scanpy as sc
import anndata
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

import Riff
os.environ['R_HOME'] = '/usr/lib/R'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")
```

## 1. Set parameters

```python
parser = argparse.ArgumentParser(description="GAT")
parser.add_argument("--seeds", type=int, default=0)
parser.add_argument("--device", type=int, default=4)
parser.add_argument("--warmup_steps", type=int, default=-1)
parser.add_argument("--num_heads", type=int, default=4, help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
parser.add_argument("--in_drop", type=float, default=0.2, help="input feature dropout")
parser.add_argument("--attn_drop", type=float, default=0.1, help="attention dropout")
parser.add_argument("--weight_decay", type=float, default=2e-4, help="weight decay")
parser.add_argument("--negative_slope", type=float, default=0.2, help="the negative slope of leaky relu for GAT")
parser.add_argument("--drop_edge_rate", type=float, default=0.0)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--lr_f", type=float, default=0.01, help="learning rate for evaluation")
parser.add_argument("--weight_decay_f", type=float, default=1e-4, help="weight decay for evaluation")
parser.add_argument("--linear_prob", action="store_true", default=True)
parser.add_argument("--load_model", action="store_true")
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--use_cfg", action="store_true")
parser.add_argument("--logging", action="store_true")
parser.add_argument("--scheduler", action="store_true", default=True)

# for graph classification
parser.add_argument("--pooling", type=str, default="mean")
parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
parser.add_argument("--batch_size", type=int, default=32)

# adjustable parameters
parser.add_argument("--encoder", type=str, default="gat")
parser.add_argument("--decoder", type=str, default="gat")
parser.add_argument("--num_hidden", type=int, default=64, help="number of hidden units")
parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
parser.add_argument("--activation", type=str, default="elu")
parser.add_argument("--max_epoch", type=int, default=50000, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `weighted_mse` loss")
parser.add_argument("--beta_l", type=float, default=1, help="`pow`inddex for `weighted_mse` loss")   
parser.add_argument("--loss_fn", type=str, default="weighted_mse")
parser.add_argument("--mask_gene_rate", type=float, default=0.3)
parser.add_argument("--replace_rate", type=float, default=0.05)
parser.add_argument("--remask_rate", type=float, default=0.)
parser.add_argument("--warm_up", type=int, default=50)
parser.add_argument("--norm", type=str, default="batchnorm")

# RIF parameter
parser.add_argument("--batch_node", type=int, default=4096)
parser.add_argument("--num_neighbors", type=int, default=7)
parser.add_argument("--num_features", type=int, default=3000)
parser.add_argument("--ref_name", type=list, default=["MouseOlfactoryBulb"])
parser.add_argument("--target_name", type=str, default="151507")
parser.add_argument("--cluster_label", type=str, default= "layer_guess")
parser.add_argument("--folder_name", type=str, default="/home/wcy/code/datasets/10X/")  
parser.add_argument("--num_classes", type=int, default=8, help = "The number of clusters")
parser.add_argument("--radius", type=int, default=7)

# read parameters
args = parser.parse_args(args=['--target_name', '151674',
                              '--ref_name', ["151507", "151508", "151509", "151510", '151673', '151675', '151676']]) 
args
```

<details>
<summary> </summary>
Namespace(activation='elu', alpha_l=2, attn_drop=0.1, batch_node=4096, batch_size=32, beta_l=1, cluster_label='layer_guess', decoder='gat', deg4feat=False, device=4, drop_edge_rate=0.0, encoder='gat', folder_name='/home/wcy/code/datasets/10X/', in_drop=0.2, linear_prob=True, load_model=False, logging=False, loss_fn='weighted_mse', lr=0.001, lr_f=0.01, mask_gene_rate=0.3, max_epoch=50000, negative_slope=0.2, norm='batchnorm', num_classes=8, num_features=3000, num_heads=4, num_hidden=64, num_layers=2, num_neighbors=7, num_out_heads=1, optimizer='adam', pooling='mean', radius=7, ref_name=['151507', '151508', '151509', '151510', '151673', '151675', '151676'], remask_rate=0.0, replace_rate=0.05, residual=False, save_model=False, scheduler=True, seeds=0, target_name='151674', use_cfg=False, warm_up=50, warmup_steps=-1, weight_decay=0.0002, weight_decay_f=0.0001)
</details>

## 2. Preprocessing
```python
adata_ref_list = []
for ref_name in args.ref_name:
    data_path = os.path.join(args.folder_name, ref_name)
    adata_ref = Riff.read_10X_Visium_with_label(data_path)

    num_classes = adata_ref.obs[args.cluster_label].nunique()
    adata_ref.obs[args.cluster_label] = adata_ref.obs[args.cluster_label].astype('category')
    adata_ref_list.append(adata_ref)

data_path = os.path.join(args.folder_name, args.target_name)
adata_target = Riff.read_10X_Visium_with_label(data_path)
        
adata_ref_list, adata_target, graph_ref_list, graph_target = Riff.transfer_preprocess(args, adata_ref_list, adata_target)
```

<details>
<summary> </summary>
=============== Contructing graph =================
</details>

## 3. Training and spatial domain identification

```python
adata_ref, adata_target = Riff.transfer_train(args, adata_ref_list, graph_ref_list, adata_target, graph_target, num_classes)
ground_truth = graph_target.ndata["label"][graph_target.ndata["label"] != -1]
pred = adata_target.obs["cluster_pred"].values[graph_target.ndata["label"] != -1]
acc = round(accuracy_score(ground_truth, pred), 4)
adata_target.obs['cluster_pred'] = adata_target.obs['cluster_pred'].astype(int).astype('category')
```
<details>
<summary> </summary>
=============== Building model =============== <br>
batch nodes change from 4096 to 3611. <br>
batch nodes change from 4096 to 3566. <br>
batch nodes change from 4096 to 3431. <br>
===================== Start training ======================= <br>
# Epoch 961: train_loss: 0.71, recon_loss: 5.39, cls_loss: 0.65:   2%|██▊                                                                                                                                             | 962/50000 [06:11<5:15:56,  2.59it/s] <br>
</details>

```python
map_dict = {}
cat = graph_ref_list[0].ndata['label'].unique()
for c in cat:
    index = np.where(graph_ref_list[0].ndata['label'] == c)[0][0]
    map_dict[int(c)] = str(adata_ref_list[0].obs["layer_guess"][index])

adata_target.obs['RIF Transfer'] = adata_target.obs['cluster_pred'].map(map_dict)
sc.pl.spatial(adata_target, color=[args.cluster_label, 'RIF Transfer'], title=['Manually annotation', 'RIF    ACC='+str(acc)])
```
<details>
<summary> </summary>
... storing 'subject' as categorical <br>
... storing 'subject_position' as categorical <br>
... storing 'Maynard' as categorical <br>
... storing 'Martinowich' as categorical <br>
... storing 'layer_guess' as categorical <br>
... storing 'layer_guess_reordered' as categorical <br>
... storing 'layer_guess_reordered_short' as categorical <br>
... storing 'feature_types' as categorical <br>
... storing 'genome' as categorical <br>
</details>

![](https://github.com/DDDoGGie/RIF/raw/gh-pages/docs/Figures/Transfer/Transfer_DLPFC_domain.png)

## 4. Batch effect removal

```python
for i in range(len(adata_ref)):
    adata_ref[i] = adata_ref[i][:, adata_target.var_names]
    
X = []
batch = []
cluster_pred = []
embedding = []
layer_guess = []

adata_ref.append(adata_target)
batch_list = args.ref_name + [args.target_name]
for i in range(len(adata_ref)):
    adata = adata_ref[i]
    X.append(adata.X.todense().A)
    batch = batch + [batch_list[i]]*adata.n_obs
    layer_guess = layer_guess + list(adata.obs['layer_guess'].values)
    cluster_pred = cluster_pred + list(adata.obs['cluster_pred'].values)
    embedding.append(adata.obsm['Riff_embedding'])
X = np.vstack(X)
embedding = np.vstack(embedding)
cluster_pred = np.array(cluster_pred)
layer_guess = np.array(layer_guess)
batch = np.array(batch)

adata = anndata.AnnData(X = X)
adata.obs['batch'] = batch
adata.obs['pred'] = cluster_pred
adata.obs['pred'] = adata.obs['pred'].astype(int).astype('category')
adata.obs['layer_guess'] = layer_guess
adata.obsm['embedding'] = embedding
adata
```
<details>
<summary> </summary>
AnnData object with n_obs × n_vars = 32266 × 3000 <br>
    obs: 'batch', 'pred', 'layer_guess' <br>
    obsm: 'embedding' <br>
</details>

```python
sc.pp.neighbors(adata, use_rep='embedding')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['layer_guess', 'batch'], title=['Manually Annotation', 'Batch'], frameon=False)
```
<details>
<summary> </summary>
... storing 'batch' as categorical <br>
... storing 'layer_guess' as categorical <br>
</details>

![](https://github.com/DDDoGGie/RIF/raw/gh-pages/docs/Figures/Transfer/Transfer_DLPFC_umap.png)