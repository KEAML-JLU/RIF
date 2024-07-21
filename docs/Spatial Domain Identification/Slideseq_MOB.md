---
sort: 5
---

# Slide-seq V2 MOB

Tutorial for spatial domain identification on Slide-seq V2 MOB. The raw dataset can be downloaded from [https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary](https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary). This tutorial used the STAGATE filtered data, which can be downloaded from [https://stagate.readthedocs.io/en/latest/T3_Slide-seqV2.html](https://stagate.readthedocs.io/en/latest/T3_Slide-seqV2.html).


```python
import os
import warnings
import warnings
import argparse

import scanpy as sc
import pandas as pd
from sklearn.metrics import adjusted_rand_score
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


parser.add_argument("--pooling", type=str, default="mean")
parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
parser.add_argument("--batch_size", type=int, default=32)


parser.add_argument("--encoder", type=str, default="gin")
parser.add_argument("--decoder", type=str, default="gin")
parser.add_argument("--num_hidden", type=int, default=64, help="number of hidden units")
parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
parser.add_argument("--activation", type=str, default="elu")
parser.add_argument("--max_epoch", type=int, default=200, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--alpha_l", type=float, default=4, help="`pow`inddex for `sce` loss")
parser.add_argument("--beta_l", type=float, default=2, help="`pow`inddex for `weighted_mse` loss")   
parser.add_argument("--loss_fn", type=str, default="weighted_mse")
parser.add_argument("--mask_gene_rate", type=float, default=0.8)
parser.add_argument("--replace_rate", type=float, default=0.05)
parser.add_argument("--remask_rate", type=float, default=0.5)
parser.add_argument("--warm_up", type=int, default=50)
parser.add_argument("--norm", type=str, default="batchnorm")

# RIF parameter
parser.add_argument("--num_neighbors", type=int, default=15)
parser.add_argument("--confidence_threshold", type=float, default=1)
parser.add_argument("--pre_aggregation", type=int, default=[1, 1]) 
parser.add_argument("--min_pseudo_label", type=int, default=3000)
parser.add_argument("--num_features", type=int, default=3000)
parser.add_argument("--seq_tech", type=str, default="SlideseqV2")
parser.add_argument("--sample_name", type=str, default="Puck_200127_15")
parser.add_argument("--cluster_label", type=str, default= "")
parser.add_argument("--folder_name", type=str, default="/home/wcy/code/datasets/SlideseqV2/")  
parser.add_argument("--num_classes", type=int, default=12, help = "The number of clusters")
parser.add_argument("--top_num", type=int, default=20)
parser.add_argument("--radius", type=int, default=15)

args = parser.parse_args(args=['--sample_name', 'Puck_200127_15']) 
args
```

<details>
<summary> </summary>
Namespace(activation='elu', alpha_l=4, attn_drop=0.1, batch_size=32, beta_l=2, cluster_label='', confidence_threshold=1, decoder='gin', deg4feat=False, device=4, drop_edge_rate=0.0, encoder='gin', folder_name='/home/wcy/code/datasets/SlideseqV2/', in_drop=0.2, linear_prob=True, load_model=False, logging=False, loss_fn='weighted_mse', lr=0.001, lr_f=0.01, mask_gene_rate=0.8, max_epoch=200, min_pseudo_label=3000, negative_slope=0.2, norm='batchnorm', num_classes=12, num_features=3000, num_heads=4, num_hidden=64, num_layers=2, num_neighbors=15, num_out_heads=1, optimizer='adam', pooling='mean', pre_aggregation=[1, 1], radius=15, remask_rate=0.5, replace_rate=0.05, residual=False, sample_name='Puck_200127_15', save_model=False, scheduler=True, seeds=0, seq_tech='SlideseqV2', top_num=15, use_cfg=False, warm_up=50, warmup_steps=-1, weight_decay=0.0002, weight_decay_f=0.0001)
</details>


## 2. Read data and preprocessing
```python
data_path = os.path.join(args.folder_name, args.sample_name)
adata = Riff.read_10X_Visium_with_label(data_path)
# adata = sc.read_h5ad('/home/wcy/code/datasets/SlideseqV2/Puck_200127_15.h5ad')
if(args.cluster_label == ""):
    num_classes = args.num_classes
else:
    num_classes = adata.obs[args.cluster_label].nunique()
    adata.obs[args.cluster_label] = adata.obs[args.cluster_label].astype('category')
    
adata, graph = Riff.build_graph(args, adata)
adata, num_classes
```
<details>
<summary> </summary>
=============== Contructing graph ================= <br>
2024-05-18 09:54:39,519 - INFO - cffi mode is CFFI_MODE.ANY <br>
2024-05-18 09:54:39,543 - INFO - R home found: /usr/lib/R <br>
2024-05-18 09:54:39,730 - INFO - R library path: /usr/lib/R/lib:/usr/lib/ x86_64-linux-gnu:/usr/lib/jvm/default-java/lib/server:/usr/local/cuda/lib64: 
2024-05-18 09:54:39,731 - INFO - LD_LIBRARY_PATH: /usr/local/cuda/lib64 <br>
2024-05-18 09:54:39,735 - INFO - Default options to initialize R: rpy2, --quiet, --no-save <br>
2024-05-18 09:54:39,919 - INFO - R is already initialized. No need to initialize. <br>
2024-05-18 09:54:39,936 - WARNING - R[write to console]:            <br>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp__ &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp__  <br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp____ ___&nbsp&nbsp_____/ /_&nbsp&nbsp_______/ /_  <br>
&nbsp&nbsp&nbsp&nbsp/ __ `__ \/ ___/ / / / / ___/ __/  <br>
&nbsp / / / / / / /__/ / /_/ (__  ) /_    <br>
/_/ /_/ /_/\___/_/\__,_/____/\__/   version 6.0.1 <br>
Type 'citation("mclust")' for citing this R package in publications. <br>

fitting ... <br>
  |======================================================================| 100% <br>
fitting ... <br>
  |======================================================================| 100% <br>
</details>
<details>
<summary> </summary>
(AnnData object with n_obs × n_vars = 20139 × 3000 <br>
     obs: 'imagerow', 'imagecol', 'pseudo_label', 'uncertainty', 'pseudo_label_scaled', 'uncertainty_scaled' <br>
     var: 'n_cells', 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm', 'mean', 'std' <br>
     uns: 'hvg', 'log1p' <br>
     obsm: 'spatial', 'emb_pca', 'mclust_prob', 'mclust_prob_scaled', <br>
 12)
</details>

## 3. Model training
```python
adata, _ = Riff.train(args, adata, graph, num_classes)
```
<details>
<summary> </summary>
=============== Building model =============== <br>
=============== Start training =============== <br>
===================== Clustering ======================= <br>
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:33<00:00,  5.93it/s] <br>
===================== Imputation ======================= <br>
100%| █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:47<00:00,  6.27it/s] <br>
</details>

## 4. Robust spatial domain identification
```python
adata.obs["combined"] = Riff.HBGF_fast(adata, ["cluster_pred1", "cluster_pred2"], pred_class=num_classes, combined_class=num_classes, top_num=args.top_num)
adata.obs['combined'] = adata.obs['combined'].astype(int).astype('category')
adata.obsm['spatial'] = (-1)*adata.obsm["spatial"]
sc.pl.embedding(adata, basis='spatial', color=['combined'], title=['RIF'], s=8, frameon=False, show=False)
plt.axis('equal')
```
<details>
<summary> </summary>
=================== Combining Result =================== <br>
fitting ... <br>
  |======================================================================| 100% <br>
</details>
<details>
<summary> </summary>
(-5108.555, -588.545, -5670.73, -847.6700000000001)
</details>

![](https://github.com/DDDoGGie/RIF/raw/gh-pages/docs/Figures/SDI/SDI_SlideseqV2_MOB_domain.png)