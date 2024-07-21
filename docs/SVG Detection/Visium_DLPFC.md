---
sort: 1
---

# 10X Visium DLPFC

Tutorial for spatial variablly gene (SVG) detection on Visium DLPFC slice 151674. 

**Note**: Before run RIF Explainer for SVG detection, please run spatial domain identification algorithm in RIF on DLPFC slice 151674, and save two things: <br>
(1) RIF model's parameters in `[custom defined output folder]/model/[sample name].pth`;   <br>
(2) The adata file with the domain information in `[custom defined output folder]/adata/[sample name].h5ad`. <br>

```python
import os
import warnings
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import Riff
os.environ['R_HOME'] = '/usr/lib/R'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")
```

## 1. Set parameters

```python
parser = argparse.ArgumentParser(description="GAT")
parser.add_argument("--seeds", type=int, default=42)
parser.add_argument("--device", type=int, default=3)
parser.add_argument("--encoder", type=str, default="gin")
parser.add_argument("--decoder", type=str, default="gin")
parser.add_argument("--num_hidden", type=int, default=64, help="number of hidden units")
parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
parser.add_argument("--activation", type=str, default="elu")
parser.add_argument("--num_heads", type=int, default=4, help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
parser.add_argument("--in_drop", type=float, default=0.2, help="input feature dropout")
parser.add_argument("--attn_drop", type=float, default=0.1, help="attention dropout")
parser.add_argument("--negative_slope", type=float, default=0.2, help="the negative slope of leaky relu for GAT")
parser.add_argument("--drop_edge_rate", type=float, default=0.0)
parser.add_argument("--alpha_l", type=float, default=4, help="`pow`inddex for `sce` loss")
parser.add_argument("--beta_l", type=float, default=2, help="`pow`inddex for `weighted_mse` loss")   
parser.add_argument("--loss_fn", type=str, default="weighted_mse")
parser.add_argument("--mask_gene_rate", type=float, default=0.8)
parser.add_argument("--replace_rate", type=float, default=0.05)
parser.add_argument("--remask_rate", type=float, default=0.5)
parser.add_argument("--warm_up", type=int, default=50)
parser.add_argument("--norm", type=str, default="batchnorm") 

# explain model parameters
parser.add_argument("--sample_num", type=int, default=1, help="number of nodes for explaination")
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--mask_act", type=str, default="sigmoid")
parser.add_argument("--mask_bias", action="store_true", default=True)
parser.add_argument("--scheduler", action="store_true", default=True)
parser.add_argument("--max_epoch", type=int, default=10000, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate for explaination")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay for evaluation")

# RIF parameter
parser.add_argument("--adj_max_num", type=int, default=3)
parser.add_argument("--feat_max_num", type=int, default=-1)
parser.add_argument("--feat_min_num", type=int, default=10)
parser.add_argument("--feat_threshold", type=float, default=2)
parser.add_argument("--num_neighbors", type=int, default=7)
parser.add_argument("--num_features", type=int, default=3000) 
parser.add_argument("--sample_name", type=str, default="151674")
parser.add_argument("--seq_tech", type=str, default="Visium")
parser.add_argument("--cluster_label", type=str, default="layer_guess")
parser.add_argument("--data_folder", type=str, default="/home/wcy/code/datasets/10X/")
parser.add_argument("--num_classes", type=int, default=7, help="The number of clusters")
parser.add_argument("--output_folder", type=str, default="/home/wcy/code/pyFile/NewFolder/GSG_modified_DLPFH/output/")

args = parser.parse_args(args=['--sample_name', '151674']) 
args
```

<details>
<summary> </summary>
Namespace(activation='elu', adj_max_num=3, alpha_l=4, attn_drop=0.1, beta_l=2, cluster_label='layer_guess', data_folder='/home/wcy/code/datasets/10X/', decoder='gin', device=3, drop_edge_rate=0.0, encoder='gin', feat_max_num=-1, feat_min_num=10, feat_threshold=2, in_drop=0.2, loss_fn='weighted_mse', lr=0.001, mask_act='sigmoid', mask_bias=True, mask_gene_rate=0.8, max_epoch=10000, negative_slope=0.2, norm='batchnorm', num_classes=7, num_features=3000, num_heads=4, num_hidden=64, num_layers=2, num_neighbors=7, num_out_heads=1, optimizer='adam', output_folder='/home/wcy/code/pyFile/NewFolder/GSG_modified_DLPFH/output/', remask_rate=0.5, replace_rate=0.05, residual=False, sample_name='151674', sample_num=1, scheduler=True, seeds=42, seq_tech='Visium', warm_up=50, weight_decay=0.0001)
</details>

## 2. Preprocessing

```python
Riff.set_random_seed(args.seeds)
data_path = args.data_folder + args.sample_name
adata = Riff.read_10X_Visium_with_label(data_path)

if(args.cluster_label == ""):
    num_classes = args.num_classes
else:
    num_classes = adata.obs[args.cluster_label].nunique()
adata, graph = Riff.build_graph(args, adata, need_preclust=False)

adata_path = os.path.join(args.output_folder, "adata/"+args.sample_name+".h5ad")
adata_imputed = sc.read_h5ad(adata_path)
```

<details>
<summary> </summary>
=============== Contructing graph =================
</details>

## 3. SVG detection

```python
selected_feats = set()
for i in range(num_classes):
    torch.cuda.empty_cache()
    print("Domain:" + str(i))
    selected_feat = Riff.find_influential_component(args, adata_imputed, graph, i)
    selected_feats = selected_feats.union(set(selected_feat))

selected_feats = list(selected_feats)
print(str(len(selected_feats)) + "SVG finded!")
svg_path = args.output_folder + "/SVG/" + str(args.sample_name) + ".txt"
f = open(svg_path, 'w')
for line in selected_feats:
    f.write(line+"\n")
f.close()
```

<details>
<summary> </summary>
Domain:0
# Epoch 10: loss: 1.73, p_feat: 0.06:   0%|▏                                                                                                                                                                             | 11/10000 [00:02<38:48,  4.29it/s] <br>
Domain:1
# Epoch 198: loss: 1.78, p_feat: 0.17:   2%|███▍                                                                                                                                                                        | 199/10000 [00:22<18:04,  9.04it/s] <br>
Domain:2
# Epoch 452: loss: 1.91, p_feat: 0.31:   5%|███████▊                                                                                                                                                                    | 453/10000 [00:58<20:36,  7.72it/s] <br>
Domain:3
# Epoch 509: loss: 1.94, p_feat: 0.32:   5%|████████▊                                                                                                                                                                   | 510/10000 [01:06<20:30,  7.71it/s] <br>
Domain:4
# Epoch 665: loss: 1.79, p_feat: 0.49:   7%|███████████▍                                                                                                                                                                | 666/10000 [01:27<20:25,  7.61it/s] <br>
Domain:5
# Epoch 10: loss: 1.59, p_feat: 0.13:   0%|▏                                                                                                                                                                             | 11/10000 [00:01<19:37,  8.48it/s] <br>
Domain:6
# Epoch 10: loss: 1.71, p_feat: 0.05:   0%|▏                                                                                                                                                                             | 11/10000 [00:01<19:56,  8.35it/s] <br>
575SVG finded!
</details>

```python
MoransI = Riff.compute_Moran_mean(adata, graph, svg_path)
print("Morans index: " + str(MoransI.round(4)))
```
<details>
<summary> </summary>
Morans index: 0.5806
</details>

## 4. Visualize some marker genes
```python
marker_gene_list = ['CNP', 'DIRAS2', 'PCP4', 'NEFM', 'HPCAL1', 'CXCL14']
for marker in marker_gene_list:
    if marker in selected_feats:
        sc.pl.spatial(adata, color=marker, alpha_img=0.3, s=12)
```

![](https://github.com/DDDoGGie/RIF/raw/gh-pages/docs/Figures/SVG/SVG_DLPFC_CNP.png)
![](https://github.com/DDDoGGie/RIF/raw/gh-pages/docs/Figures/SVG/SVG_DLPFC_DIRAS2.png)
![](https://github.com/DDDoGGie/RIF/raw/gh-pages/docs/Figures/SVG/SVG_DLPFC_PCP4.png)
![](https://github.com/DDDoGGie/RIF/raw/gh-pages/docs/Figures/SVG/SVG_DLPFC_NEFM.png)
![](https://github.com/DDDoGGie/RIF/raw/gh-pages/docs/Figures/SVG/SVG_DLPFC_HPCAL1.png)
![](https://github.com/DDDoGGie/RIF/raw/gh-pages/docs/Figures/SVG/SVG_DLPFC_CXCL14.png)