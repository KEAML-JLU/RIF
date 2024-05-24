import os
import warnings
import warnings
import argparse

import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

import Riff
os.environ['R_HOME'] = '/usr/lib/R'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")


def build_args():
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

    # GSG parameter
    parser.add_argument("--batch_node", type=int, default=4096)
    parser.add_argument("--num_neighbors", type=int, default=15)
    parser.add_argument("--num_features", type=int, default=3000)
    parser.add_argument("--ref_name", type=list, default=["MouseOlfactoryBulb"])
    parser.add_argument("--target_name", type=str, default="151507")
    parser.add_argument("--cluster_label", type=str, default= "RIFF")
    parser.add_argument("--folder_name", type=str, default="/home/wcy/code/datasets/10X/")  
    parser.add_argument("--num_classes", type=int, default=8, help = "The number of clusters")
    parser.add_argument("--radius", type=int, default=7)

    # read parameters
    args = parser.parse_args()
    return args

def main(args):
    acc_list = []
    acc_refine_list = []
    # sample_name_list = ["151507", "151508", "151509", "151510", '151673', '151674', '151675', '151676']
    # sample_name_list = ['151669', '151670', '151671', '151672']
    # sample_name_list = ['V1_Breast_Cancer_Block_A_Section_2', 'V1_Breast_Cancer_Block_A_Section_1']
    # args.target_name = sample_name_list[i]
    # args.ref_name = sample_name_list[:i] + sample_name_list[i+1:]
    print("=============== " + args.target_name + " ===============")
    adata_ref_list = []
    for ref_name in args.ref_name:
        # data_path = os.path.join(args.folder_name, ref_name)
        # adata_ref = Riff.read_10X_Visium_with_label(data_path)
        adata_ref = Riff.read_slideseq_V2("/home/wcy/code/datasets/SlideseqV2/Puck_200127_15")
        cell_type = pd.read_csv("/home/wcy/code/datasets/SlideseqV2/Puck_200127_15/Riff.csv", index_col=0, header=None)
        adata_ref.obs['RIFF'] = np.squeeze(cell_type.values)
        # adata_ref = adata_ref[adata_ref.obs['RIFF']!=7]
        num_classes = adata_ref.obs[args.cluster_label].nunique()
        adata_ref.obs[args.cluster_label] = adata_ref.obs[args.cluster_label].astype('category')
        adata_ref_list.append(adata_ref)

    # data_path = os.path.join(args.folder_name, args.target_name)
    # adata_target = GSG.read_10X_Visium_with_label(data_path)
    adata_target = Riff.read_Stereo_seq("/home/wcy/code/datasets/Stero-seq/MouseOlfactoryBulb")
        
    adata_ref_list, adata_target, graph_ref_list, graph_target = Riff.transfer_preprocess(args, adata_ref_list, adata_target)
    adata_ref, adata_target = Riff.transfer_train(args, adata_ref_list, graph_ref_list, adata_target, graph_target, num_classes)

    ground_truth = graph_target.ndata["label"][graph_target.ndata["label"] != -1]
    pred = adata_target.obs["cluster_pred"].values[graph_target.ndata["label"] != -1]
    acc = accuracy_score(ground_truth, pred)
    acc_list.append(acc)
    if args.radius>0:
        adata_target.obs["cluster_pred"] = Riff.refine_label(adata_target, radius=args.radius, key='cluster_pred')
    pred = adata_target.obs["cluster_pred"].values[graph_target.ndata["label"] != -1]
    pred = [int(pred[i]) for i in range(pred.shape[0])]
    acc = accuracy_score(ground_truth, pred)
    acc_refine_list.append(acc)
    adata_target.write_h5ad("/home/wcy/code/pyFile/NewFolder/GSG_modified_DLPFH/output/adata/transfer/" + args.target_name + ".h5ad")

    


    
        
if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)