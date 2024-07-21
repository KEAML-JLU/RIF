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

def build_args():
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


    parser.add_argument("--sample_num", type=int, default=1, help="number of nodes for explaination")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--mask_act", type=str, default="sigmoid")
    parser.add_argument("--mask_bias", action="store_true", default=True)
    parser.add_argument("--scheduler", action="store_true", default=True)
    parser.add_argument("--max_epoch", type=int, default=10000, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for explaination")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay for evaluation")
     
    # Riff parameter
    parser.add_argument("--adj_max_num", type=int, default=3)
    parser.add_argument("--feat_max_num", type=int, default=-1)
    parser.add_argument("--feat_min_num", type=int, default=10)
    parser.add_argument("--feat_threshold", type=float, default=2.5)
    parser.add_argument("--num_neighbors", type=int, default=7)
    parser.add_argument("--num_features", type=int, default=3000) 
    parser.add_argument("--sample_name", type=str, default="151674")
    parser.add_argument("--seq_tech", type=str, default="Visium")
    parser.add_argument("--cluster_label", type=str, default="layer_guess")
    parser.add_argument("--data_folder", type=str, default="/home/wcy/code/datasets/10X/")
    parser.add_argument("--num_classes", type=int, default=7, help="The number of clusters")
    parser.add_argument("--output_folder", type=str, default="/home/wcy/code/pyFile/RIF/output/")


    args = parser.parse_args()
    return args

def main(args):
    Riff.set_random_seed(args.seeds)

    data_path = args.data_folder + args.sample_name
    # adata = Riff.read_Stereo_seq(data_path)
    # adata = Riff.read_slideseq_V2(data_path)
    adata = Riff.read_10X_Visium_with_label(data_path)
    # data_path = '/home/wcy/code/datasets/Stero-seq/MouseOlfactoryBulb/MouseOlfactoryBulb.h5ad'
    # adata = sc.read_h5ad(data_path)
    if(args.cluster_label == ""):
        num_classes = args.num_classes
    else:
        num_classes = adata.obs[args.cluster_label].nunique()
    adata, graph = Riff.build_graph(args, adata, need_preclust=False)

    adata_path = os.path.join(args.output_folder, "adata/"+args.sample_name+".h5ad")
    # adata_path = '/home/wcy/code/pyFile/NewFolder/GSG_modified_DLPFH/output/adata/MOB/MouseOlfactoryBulb_filter50_c7.h5ad'
    adata_imputed = sc.read_h5ad(adata_path)
    selected_feats = set()
    for i in range(num_classes):
    # for i in range(num_classes):
        torch.cuda.empty_cache()
        print("Domain:" + str(i))
        selected_feat = Riff.find_influential_component(args, adata_imputed, graph, i)
        selected_feats = selected_feats.union(set(selected_feat))

    
    selected_feats = list(selected_feats)
    print(str(len(selected_feats)) + "SVG finded!")
    svg_path = os.path.join(args.output_folder, "SVG/" + str(args.sample_name) + ".txt")
    f = open(svg_path, 'w')
    for line in selected_feats:
        f.write(line+"\n")
    f.close()
        
    MoransI = Riff.compute_Moran_mean(adata, graph, svg_path)
    print("Morans index: " + str(MoransI.round(4)))
    
    
        
        
if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)