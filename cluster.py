import os
import warnings
import argparse

import pandas as pd
from sklearn.metrics import adjusted_rand_score

import Riff
os.environ['R_HOME'] = '/usr/lib/R'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, default=0)
    parser.add_argument("--device", type=int, default=3)
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
    parser.add_argument("--encoder", type=str, default="gin")
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--num_hidden", type=int, default=16, help="number of hidden units")
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
    parser.add_argument("--num_neighbors", type=int, default=7)
    parser.add_argument("--confidence_threshold", type=float, default=3e-3)
    parser.add_argument("--pre_aggregation", type=int, default=[1, 1])
    parser.add_argument("--min_pseudo_label", type=int, default=3000)
    parser.add_argument("--num_features", type=int, default=3000)
    parser.add_argument("--seq_tech", type=str, default="Visium")
    parser.add_argument("--sample_name", type=str, default="moffitt2018molecular")
    parser.add_argument("--cluster_label", type=str, default= "layer_guess")
    parser.add_argument("--folder_name", type=str, default="/home/wcy/code/datasets/10X/")  
    parser.add_argument("--output_folder", type=str, default="/home/wcy/code/pyFile/NewFolder/GSG_modified_DLPFH/output/")  
    parser.add_argument("--num_classes", type=int, default=7, help = "The number of clusters")
    parser.add_argument("--top_num", type=int, default=10)
    parser.add_argument("--radius", type=int, default=50)

    # read parameters
    args = parser.parse_args()
    return args

def main(args):
    sample_name_list = ['151670', '151507', '151508', '151509', '151510', '151669', '151671', '151672', '151673', '151674', '151675', '151676']
    # sample_name_list = ["V1_Mouse_Brain_Sagittal_Anterior", "V1_Mouse_Brain_Sagittal_Posterior"]
    # sample_name_list = ["mm_hippo"]
    # sample_name_list = ["mm_embryogenesis"]
    # sample_name_list = ["V1_Breast_Cancer_Block_A_Section_1", "V1_Breast_Cancer_Block_A_Section_2"]
    # sample_name_list = ['151507', '151508', '151509', '151510', '151673', '151674', '151675', '151676']
    # sample_name_list = ['MouseOlfactoryBulb']
    '''
    sample_name_list = ['mouse_hypothalamic_preoptic1', 'mouse_hypothalamic_preoptic2', 'mouse_hypothalamic_preoptic3', 
                        'mouse_hypothalamic_preoptic4', 'mouse_hypothalamic_preoptic5']
    
    sample_name_list = ['Zeng2023Integrative_13m-disease-replicate_2', 'Zeng2023Integrative_8m-disease-replicate_1', 'Zeng2023Integrative_8m-disease-replicate_2',
                        'Zeng2023Integrative_8m-control-replicate_1', 'Zeng2023Integrative_8m-control-replicate_2', 
                        'Zeng2023Integrative_13m-control-replicate_1', 'Zeng2023Integrative_13m-control-replicate_2',
                        'Zeng2023Integrative_13m-disease-replicate_1', ]
    '''
    # sample_name_list = ["somatosensory_cortex"]
    ari_list = []

    for i in sample_name_list:
        args.sample_name = i
        data_path = os.path.join(args.folder_name, args.sample_name)
        # adata = Riff.read_BLCA(data_path)
        # adata = Riff.read_Stereo_seq(data_path)
        # adata = sc.read_h5ad(data_path + ".h5ad")
        adata = Riff.read_10X_Visium_with_label(data_path)
        # adata = Riff.read_h5ad(data_path)
        # adata = Riff.read_slideseq_V2(data_path)
        if(args.cluster_label == ""):
            num_classes = args.num_classes
        else:
            num_classes = adata.obs[args.cluster_label].nunique()
            adata.obs[args.cluster_label] = adata.obs[args.cluster_label].astype('category')

        adata, graph = Riff.build_graph(args, adata)
        adata, _ = Riff.train(args, adata, graph, num_classes)
       
        adata.obs["pred1_refine"] = Riff.refine_label(adata, args.radius, key='cluster_pred1')
        adata.obs["pred2_refine"] = Riff.refine_label(adata, args.radius, key='cluster_pred2')
        adata.obs["combined"] = Riff.HBGF(adata, ["pred1_refine", "pred2_refine"], num_classes, top_num=args.top_num)
        adata.obs["combined_refine"] = Riff.refine_label(adata, args.radius, key='combined')
        adata_reduce = adata[~pd.isna(adata.obs[args.cluster_label])]
        ari = round(adjusted_rand_score(adata_reduce.obs[args.cluster_label], adata_reduce.obs["combined_refine"]), 4)
        ari_list.append(ari)
        adata.write_h5ad(args.output_folder+"adata/"+args.sample_name+".h5ad")


        '''
        adata, new_key = Riff.test_refine(adata, num_classes, max_neigh=args.radius, key='combined', refined_key='combined_refine')
         
        
        adata_reduce = adata[~pd.isnull(adata.obs[args.cluster_label])]
        ari_final = adjusted_rand_score(adata_reduce.obs[args.cluster_label], adata_reduce.obs["pred1_refine"])
        ari_dict["pred1"].append(ari_final)
        
        ari_final = adjusted_rand_score(adata_reduce.obs[args.cluster_label], adata_reduce.obs["pred2_refine"])          
        ari_dict["pred2"].append(ari_final)

        ari_final = adjusted_rand_score(adata_reduce.obs[args.cluster_label], adata_reduce.obs[new_key])
        ari_dict["combined"].append(ari_final)

        adata.uns['RIF_domain_key'] = new_key
        
        adata.obs["combined"] = Riff.HBGF(adata, ["cluster_pred1", "cluster_pred2"], num_classes, top_num=args.top_num)
        
        adata_reduce = adata[~pd.isnull(adata.obs[args.cluster_label])]
        ari1 = adjusted_rand_score(adata_reduce.obs[args.cluster_label], adata_reduce.obs["cluster_pred1"])
        ari2 = adjusted_rand_score(adata_reduce.obs[args.cluster_label], adata_reduce.obs["cluster_pred2"])
        ari3 = adjusted_rand_score(adata_reduce.obs[args.cluster_label], adata_reduce.obs["combined"])
        print(ari1, ari2, ari3)
        adata.write_h5ad("/home/wcy/code/pyFile/NewFolder/GSG_modified_DLPFH/output/adata/slide-tags/"+args.sample_name+".h5ad")
        
        ari_dict["n_obs"].append(adata.n_obs)
        
'''
    final_ari = pd.DataFrame({'ari':ari_list})
    final_ari = final_ari.round(4)
    final_ari.index = sample_name_list
    print(final_ari)

        
if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)