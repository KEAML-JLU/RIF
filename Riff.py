import os
import warnings
import numpy as np
import pandas as pd

import ot
import dgl
import torch
import scanpy as sc
import dgl.function as fn
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

from graphmae import explainer
from graphmae.models import build_model, build_Transfer_model
from graphmae.utils import (create_optimizer, set_random_seed, pretrain, transferTrain)
warnings.filterwarnings("ignore")


def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)            
		print("---  new folder...  ---")
		print("---  OK  ---")
	else:
		print("---  There is this folder!  ---")

def Read_Visium_with_label(path, 
                    genome=None,
                    count_file='filtered_feature_bc_matrix.h5', 
                    library_id=None, 
                    load_images=True, 
                    quality='hires',
                    image_path = None):
    adata = sc.read_visium(path, 
                        genome=genome,
                        count_file=count_file,
                        library_id=library_id,
                        load_images=load_images,)
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
    image_coor = adata.obsm["spatial"] * scale
    if(os.path.exists(path + "/metadata.tsv")):
        adata.obs = pd.read_table(path + "/metadata.tsv",sep="\t",index_col=0)
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.obsm['image_coor'] = image_coor
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata

def Read_10X_Visium_with_label(path):
    adata = sc.read_visium(path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata

def Read_Xenium(h5_path, obs_path):
    adata = sc.read_10x_h5(h5_path)
    adata.obs = pd.read_csv(obs_path, index_col=0)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    return adata

def Read_Stereoseq(path):
    count_path = os.path.join(path, "RNA_counts.tsv")
    pos_path = os.path.join(path, "position.tsv")
    spot_path = os.path.join(path, "used_barcodes.txt")
    counts = pd.read_csv(count_path, sep='\t', index_col=0)
    pos = pd.read_csv(pos_path, sep='\t')
    used_barcode = pd.read_csv(spot_path, sep='\t', header=None)

    counts.columns = ['Spot_'+str(x) for x in counts.columns]
    pos.index = pos['label'].map(lambda x: 'Spot_'+str(x))
    pos = pos.loc[:, ['x','y']]
    pos.columns = ["imagerow","imagecol"]

    adata = sc.AnnData(counts.T)
    adata.var_names_make_unique()
    pos = pos.loc[adata.obs_names, ['imagerow', 'imagecol']]
    adata.obsm["spatial"] = pos.to_numpy()
    adata.obs["imagerow"] = 0
    adata.obs["imagecol"] = 0
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 0]
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 1]
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    used_barcode = used_barcode[0]
    adata = adata[used_barcode,]
    return adata

def Read_SlideseqV2(path):
    count_path = os.path.join(path, "digital_expression.txt")
    pos_path = os.path.join(path, "bead_locations.csv")
    spot_path = os.path.join(path, "used_barcodes.txt")
    gene_expression = pd.read_table(count_path)
    position = pd.read_csv(pos_path, index_col=0)
    used_barcodes = pd.read_table(spot_path, header=None)

    obs_name = list(used_barcodes[0].values)
    var_name = list(gene_expression["GENE"].values)
    X = gene_expression[used_barcodes[0].values].T.values
    adata = sc.AnnData(X)
    adata.obs_names = obs_name
    adata.var_names = var_name

    position.index = position["barcode"]
    position = position.loc[obs_name]
    spatial = position[['xcoord', 'ycoord']].values
    adata.obsm["spatial"] = spatial

    adata.obs["imagerow"] = 0
    adata.obs["imagecol"] = 0
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 0]
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 1]
    return adata

def Preprocess_adata(args, adata):
    if args.seq_tech == 'Visium':
        sc.pp.filter_genes(adata, min_cells=5)
    elif args.seq_tech == 'Stereo-seq':
        sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=args.num_features)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    return adata

def Build_graph(args, adata, spatial_key="spatial"):
    print("=================== Contructing graph =====================")
    position = adata.obsm[spatial_key]
    spot_num = len(position)
    adjacent_head = torch.from_numpy(np.arange(spot_num).repeat(args.num_neighbors))
    NN_model = NearestNeighbors(n_neighbors=args.num_neighbors, algorithm="ball_tree").fit(position)
    _, adjacent_tail = NN_model.kneighbors(position)
    adjacent_tail = torch.from_numpy(adjacent_tail.flatten())
    graph = dgl.graph((adjacent_head,adjacent_tail))

    if isinstance(adata.X, sp.csr_matrix):
        graph.ndata["feat"] = torch.tensor(adata.X.todense().copy(), dtype=torch.float32)
    else:
        graph.ndata["feat"] = torch.tensor(adata.X.copy(), dtype=torch.float32)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    if isinstance(adata.X, sp.csr_matrix):
        graph.ndata["feat_scaled"] = torch.tensor(adata.X.todense().copy(), dtype=torch.float32)  
    else:
        graph.ndata["feat_scaled"] = torch.tensor(adata.X.copy(), dtype=torch.float32)
    return adata, graph

def Generate_pseudo_label(args, adata, graph):
    print("================== Generating pseudo-label ================")
    cluster_label, cluster_prob, cluster_uncertainty = preclust(adata, graph, args.num_classes, key="feat", pre_agg=args.pre_aggregation[0])
    adata.obs["pseudo_label"] = cluster_label
    adata.obsm["mclust_prob"] = cluster_prob
    adata.obs["uncertainty"] = cluster_uncertainty
    cluster_label, cluster_prob, cluster_uncertainty = preclust(adata, graph, args.num_classes, key="feat_scaled", pre_agg=args.pre_aggregation[1])
    adata.obs["pseudo_label_scaled"] = cluster_label
    adata.obsm["mclust_prob_scaled"] = cluster_prob
    adata.obs["uncertainty_scaled"] = cluster_uncertainty
    return adata

def transfer_preprocess(args, adata_ref_list, adata_target):
    print("=============== Contructing graph =================")
    
    position = adata_target.obs[["imagerow","imagecol"]].values
    spot_num = len(position)
    adjacent_head = torch.from_numpy(np.arange(spot_num).repeat(args.num_neighbors))
    NN_model = NearestNeighbors(n_neighbors=args.num_neighbors, algorithm="ball_tree").fit(position)
    _, adjacent_tail = NN_model.kneighbors(position)
    adjacent_tail = torch.from_numpy(adjacent_tail.flatten())
    graph_target = dgl.graph((adjacent_head,adjacent_tail))

    adata_target.var_names_make_unique()
    sc.pp.filter_genes(adata_target, min_cells=5)
    gene_common = set(adata_target.var_names.tolist())

    class_name = adata_ref_list[0].obs[args.cluster_label][~pd.isna(adata_ref_list[0].obs[args.cluster_label])].unique()
    map_dict = {value: index for index, value in enumerate(class_name)}
    graph_ref_list = []
    for i in range(len(adata_ref_list)):
        adata_ref = adata_ref_list[i]
        adata_ref.var_names_make_unique()
        adata_ref = adata_ref[~pd.isna(adata_ref.obs[args.cluster_label])]
        sc.pp.filter_genes(adata_ref, min_cells=5)
        gene_ref = set(adata_ref.var_names.tolist())
        gene_common = gene_common.intersection(gene_ref)
        adata_ref_list[i] = adata_ref

        position = adata_ref.obs[["imagerow","imagecol"]].values
        spot_num = len(position)
        adjacent_head = torch.from_numpy(np.arange(spot_num).repeat(args.num_neighbors))
        NN_model = NearestNeighbors(n_neighbors=args.num_neighbors, algorithm="ball_tree").fit(position)
        _, adjacent_tail = NN_model.kneighbors(position)
        adjacent_tail = torch.from_numpy(adjacent_tail.flatten())
        graph_ref = dgl.graph((adjacent_head,adjacent_tail))
        graph_ref_list.append(graph_ref)
        

    adata_target = adata_target[:, list(gene_common)]
    sc.pp.highly_variable_genes(adata_target, flavor="seurat_v3", n_top_genes=args.num_features)
    sc.pp.normalize_total(adata_target, target_sum=1e4)
    sc.pp.log1p(adata_target)
    adata_target = adata_target[:, adata_target.var['highly_variable']]
    gene_selected = adata_target.var_names
    if isinstance(adata_target.X, sp.csr_matrix):
        graph_target.ndata["feat"] = torch.tensor(adata_target.X.todense().copy(), dtype=torch.float32)
    else:
        graph_target.ndata["feat"] = torch.tensor(adata_target.X.copy(), dtype=torch.float32)
    if args.cluster_label in adata_target.obs.columns:
        adata_target.obs['label'] = adata_target.obs[args.cluster_label].map(map_dict)
        graph_target.ndata["label"] = torch.tensor(adata_target.obs['label'].values, dtype=torch.long)
    
    for i in range(len(adata_ref_list)):
        adata_ref = adata_ref_list[i]
        graph_ref = graph_ref_list[i]
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        adata_ref = adata_ref[:, gene_selected]
        if isinstance(adata_ref.X, sp.csr_matrix):
            graph_ref.ndata["feat"] = torch.tensor(adata_ref.X.todense().copy(), dtype=torch.float32)
        else:
            graph_ref.ndata["feat"] = torch.tensor(adata_ref.X.copy(), dtype=torch.float32)
        adata_ref.obs['label'] = adata_ref.obs[args.cluster_label].map(map_dict)
        graph_ref.ndata["label"] = torch.tensor(adata_ref.obs['label'].values, dtype=torch.long)

    return adata_ref_list, adata_target, graph_ref_list, graph_target

def Train(args, adata, graph):
    print("==================== Building model ====================")
    device = args.device if args.device >= 0 else "cpu"
    lr = args.lr
    optim_type = args.optimizer
    weight_decay = args.weight_decay
    set_random_seed(args.seeds)

    model = build_model(args, spot_num=adata.n_obs)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)
    model, adata = pretrain(args, model, adata, graph, optimizer)

    model.train(False)
    x = graph.ndata["feat"]
    embedding = model.embed(graph.to(device), x.to(device))  
    adata.obsm["Riff_embedding"] = embedding.cpu().detach().numpy() 
    adata.obsm["Riff_imputation"] = model.get_imputed(graph.to(device), x.to(device)).cpu().detach().numpy()
    return adata, model

def transfer_train(args, adata_ref, graph_ref, adata_target, graph_target, num_classes):
    print("=============== Building model ===============")
    device = args.device if args.device >= 0 else "cpu"
    lr = args.lr
    optim_type = args.optimizer
    weight_decay = args.weight_decay
    set_random_seed(args.seeds)
    
    batch_node = args.batch_node
    count = torch.zeros(size=[num_classes])
    for i in range(len(graph_ref)):
        label = graph_ref[i].ndata["label"]
        count = count + torch.unique(label, return_counts=True)[1]
        if batch_node > graph_ref[i].num_nodes():
            batch_node = graph_ref[i].num_nodes()
            print("batch nodes change from " + str(args.batch_node) + " to " + str(batch_node) + ".")
    args.balance_class = count.max()/count
    args.batch_node = batch_node

    model = build_Transfer_model(args, num_classes, spot_num=args.batch_node)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)
    model, adata_ref = transferTrain(args, model, adata_ref, graph_ref, optimizer)
    # model_path = "/home/wcy/code/pyFile/NewFolder/GSG_modified_DLPFH/output/model/transfer_500_4096.pth"
    # model_state_dict = torch.load(model_path)
    # model.load_state_dict(model_state_dict)
    # model.to(device)
    # model.train(False)
    graph_target = graph_target.to(device)

    last = 0
    spot_list = np.arange(adata_target.n_obs)
    np.random.shuffle(spot_list)
    pred_df = pd.DataFrame()
    embed_df = pd.DataFrame()
    impute_df = pd.DataFrame()
    while last + args.batch_node <= adata_target.n_obs:
        sub_spot = spot_list[last : last+args.batch_node]
        last = last + args.batch_node
        sub_graph = dgl.node_subgraph(graph_target, sub_spot)

        x = sub_graph.ndata["feat"]
        pred, embedding, imputed = model.predict(sub_graph, x)
        pred = pred.detach().cpu().numpy()
        embedding = embedding.detach().cpu().numpy()
        imputed = imputed.detach().cpu().numpy()
        df = pd.DataFrame(pred, index=sub_spot)
        pred_df = pd.concat([pred_df, df]) 
        df = pd.DataFrame(embedding, index=sub_spot)
        embed_df = pd.concat([embed_df, df])
        df = pd.DataFrame(imputed, index=sub_spot)
        impute_df = pd.concat([impute_df, df]) 

    if last < adata_target.n_obs:
        sub_spot = spot_list[- args.batch_node:]
        sub_graph = dgl.node_subgraph(graph_target, sub_spot)

        x = sub_graph.ndata["feat"]
        pred, embedding, imputed = model.predict(sub_graph, x)
        pred = pred.detach().cpu().numpy()
        embedding = embedding.detach().cpu().numpy()
        imputed = imputed.detach().cpu().numpy()
        pred = pred[-adata_target.n_obs%args.batch_node:]
        embedding = embedding[-adata_target.n_obs%args.batch_node:]
        imputed = imputed[-adata_target.n_obs%args.batch_node:]
        sub_spot = sub_spot[-adata_target.n_obs%args.batch_node:]
        df = pd.DataFrame(pred, index=sub_spot)
        pred_df = pd.concat([pred_df, df]) 
        df = pd.DataFrame(embedding, index=sub_spot)
        embed_df = pd.concat([embed_df, df])
        df = pd.DataFrame(imputed, index=sub_spot)
        impute_df = pd.concat([impute_df, df]) 

    pred_df = pred_df.sort_index()
    embed_df = embed_df.sort_index()
    impute_df = impute_df.sort_index()
    adata_target.obs["cluster_pred"] = np.argmax(pred_df.values, axis=1)
    adata_target.obsm["Riff_embedding"] = embed_df.values
    adata_target.obsm["Riff_imputed"] = impute_df.values
    return adata_ref, adata_target

def preclust(adata, graph, num_classes, key="feat", pre_agg=1):
    if pre_agg == 0:
        pca = PCA(n_components=20, random_state=42)
        embedding = pca.fit_transform(graph.ndata[key].numpy().copy())
        adata.obsm["emb_pca"] = embedding
        return mclust_R(adata, used_obsm="emb_pca", num_cluster=num_classes)
    
    graph.update_all(fn.copy_u(key, 'm'), fn.mean('m', 'feat_aggregated'))
    for i in range(pre_agg-1):
        graph.update_all(fn.copy_u('feat_aggregated', 'm'), fn.mean('m', 'feat_aggregated'))
    pca = PCA(n_components=20, random_state=42)
    embedding = pca.fit_transform(graph.ndata['feat_aggregated'].numpy().copy())
    adata.obsm["emb_pca"] = embedding
    return mclust_R(adata, used_obsm="emb_pca", num_cluster=num_classes)


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r('options(warn=-1)') 
    robjects.r('.libPaths(c("/home/wcy/R/x86_64-pc-linux-gnu-library/4.2", .libPaths()))')
    robjects.r('suppressPackageStartupMessages(library(mclust))')
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    
    mclust_prob = np.array(res[-3])
    mclust_result = np.array(res[-2])
    mclust_uncertainty = np.array(res[-1])
    return mclust_result, mclust_prob, mclust_uncertainty


def Ensemble_clustering(adata, radius, num_classes, top_num=10, cluster_key=['cluster_pred1', 'cluster_pred2'], store_key='combined_refine'):
    if radius > 0:
        adata.obs["pred1_refine"] = Refine_label(adata, radius, key=cluster_key[0])
        adata.obs["pred2_refine"] = Refine_label(adata, radius, key=cluster_key[1])
    else:
        adata.obs["pred1_refine"] = adata.obs[cluster_key[0]]
        adata.obs["pred2_refine"] = adata.obs[cluster_key[1]]
    adata.obs["combined"] = HBGF(adata, ["pred1_refine", "pred2_refine"], num_classes, top_num=top_num)
    if radius > 0:
        adata.obs[store_key] = Refine_label(adata, radius, key='combined')
    else:
        adata.obs[store_key] = adata.obs["combined"]
    adata, new_key = Test_refine(adata, num_classes, max_neigh=radius, key='combined', refined_key=store_key)
    return adata, new_key


def Refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    
    return new_type

def Test_refine(adata, num_classes, max_neigh=50, radius=6, key='label', refined_key='label_refine'):
    if len(np.unique(adata.obs[refined_key].values)) == num_classes:
        print("Nothing changed!")
        return adata, refined_key
    elif len(np.unique(adata.obs[key].values)) < num_classes:
        print("Nothing changed! Try to increase 'args.top_num'.")
        return adata, refined_key
    else:
        test_radius_list = [1, ]
        for i in range(100000):
            test_radius = (i+1)*radius + test_radius_list[-1]
            if test_radius < max_neigh:
                test_radius_list.append(test_radius)
            else:
                break
        test_radius_list.reverse()

        for test_radius in test_radius_list[:-1]:
            new_key = refined_key + "_" + str(test_radius)
            adata.obs[new_key] = Refine_label(adata, radius=test_radius, key=key)
            if len(np.unique(adata.obs[new_key].values)) == num_classes:
                print("Try new radius! Result is stored in '" + new_key + "'.")
                return adata, new_key
        print("Remove smoothing! Result is stored in '" + key + "'.")
        return adata, key

    

def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):
    pca = PCA(n_components=20, random_state=42) 
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding
    
    if method == 'mclust':
       adata.obs['mclust'] = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)[0]
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain'] 
       
    if refinement:  
       new_type = Refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type 

def HBGF(adata, keys, num_classes, top_num=10):
    print("=================== Combining Result ===================")
    cluster_onehot = np.eye(num_classes)
    A1 = cluster_onehot[adata.obs[keys[0]].values.astype(int)]
    A2 = cluster_onehot[adata.obs[keys[1]].values.astype(int)]
    A = np.concatenate((A1, A2), 1)
    A_left = np.concatenate((np.zeros((A.shape[1], A.shape[1])), A), 0)
    A_right = np.concatenate((A.T, np.zeros((A.shape[0], A.shape[0]))), 0)
    W = np.concatenate((A_left, A_right), 1)
    D = np.diag(1/W.sum(0))
    # D[np.isinf(D)] = 0
    L = D @ W
    L[np.isnan(L)] = 0

    eigenvalue, featurevector = np.linalg.eig(L)
    top_eigen = abs(eigenvalue).argsort()[-top_num:]
    top_vector = featurevector[:, top_eigen].real
    cluster = KMeans(n_clusters=num_classes, random_state=42, n_init=1)
    cluster = cluster.fit(top_vector)
    return cluster.labels_[2*num_classes:]

def HBGF_fast(adata, keys, pred_class, combined_class, random_seed=2020, top_num=10):
    print("=================== Combining Result ===================")
    cluster_onehot = np.eye(pred_class)
    A1 = cluster_onehot[adata.obs[keys[0]].values.astype(int)]
    A2 = cluster_onehot[adata.obs[keys[1]].values.astype(int)]
    A = np.concatenate((A1, A2), 1)
    A_left = np.concatenate((np.zeros((A.shape[1], A.shape[1])), A), 0)
    A_right = np.concatenate((A.T, np.zeros((A.shape[0], A.shape[0]))), 0)
    W = np.concatenate((A_left, A_right), 1)
    D = np.diag(1/W.sum(0))
    D[np.isinf(D)] = 0
    L = D @ W

    _, v = sp.linalg.eigsh(L, top_num)
    
    import rpy2.robjects as robjects
    robjects.r('.libPaths(c("/home/wcy/R/x86_64-pc-linux-gnu-library/4.2", .libPaths()))')
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(v), combined_class, "EEE")
    return res[-2][2*pred_class:]


def find_influential_component(args, adata, graph, clust_id):
    x = graph.ndata["feat"]
    spatial_adj = graph.adj().to_dense()
    if(args.cluster_label == ""):
        num_classes = args.num_classes
    else:
        num_classes = adata.obs[args.cluster_label].nunique()
    
    model = build_model(args, num_classes, spot_num=adata.n_obs)

    pred_res = torch.tensor(adata.obs["cluster_pred2"].astype(int).values)
    gene_exp = pd.DataFrame(x)
    gene_exp["label"] = pred_res
    gene_exp["label"] = (gene_exp["label"] != clust_id).astype(int)
    gene_mean = gene_exp.groupby("label").mean()
    gene_diff = (gene_mean.iloc[0] - gene_mean.iloc[1]).values
    gene_diff = torch.tensor(gene_diff)

    model_path = os.path.join(args.output_folder, "model/", args.sample_name + ".pth")
    # model_path = '/home/wcy/code/pyFile/NewFolder/GSG_modified_DLPFH/output/model/MouseOlfactoryBulb_c7.pth'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    explain = explainer.Explainer(model, spatial_adj, x, pred_res, clust_id, args, gene_diff)
    selected_feat = explain.explain()
    selected_feat = adata.var_names[selected_feat]
    return selected_feat

def find_influential_component_transfer(args, adata, graph, clust_id):
    x = graph.ndata["feat"]
    spatial_adj = graph.adj().to_dense()
    if(args.cluster_label == ""):
        num_classes = args.num_classes
    else:
        num_classes = adata.obs[args.cluster_label].nunique()
    
    args.balance_class = -1
    model = build_Transfer_model(args, num_classes, spot_num=adata.n_obs)

    pred_res = torch.tensor(adata.obs["cluster_pred"].astype(int).values)
    gene_exp = pd.DataFrame(adata.X)
    gene_exp["label"] = pred_res
    gene_exp["label"] = (gene_exp["label"] != clust_id).astype(int)
    gene_mean = gene_exp.groupby("label").mean()
    gene_diff = (gene_mean.iloc[0] - gene_mean.iloc[1]).values
    gene_diff = torch.tensor(gene_diff)

    model_path = os.path.join(args.output_folder, "model/transfer", args.sample_name+".pth")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    explain = explainer.Explainer(model, spatial_adj, x, pred_res, clust_id, args, gene_diff)
    selected_feat = explain.explain()
    selected_feat = adata.var_names[selected_feat]
    return selected_feat

def Moran(data, w):
    similarity = data*data.T
    numerator = (similarity * w).sum()
    denominator = (data * data).sum()
    coef = data.shape[0]/w.sum()
    return coef * numerator / denominator

def compute_Moran_mean(adata, graph, svg_path):
    moran_sum = 0
    with open(svg_path, "r") as f:
        svg = f.readlines()
        svg = [gene.rstrip("\n") for gene in svg]
    f.close()
    if isinstance(adata.X, sp.csr_matrix):
        adata.X = adata.X.todense().A
    w = graph.adj().to_dense().numpy()
    for gene in svg:
        data = adata[:, gene].X
        moran_sum += Moran(data, w)
    return moran_sum/len(svg)











def transferToClustDenseGraph(cluster_result):
    cluster_list = cluster_result.unique()
    cluster_affinity_matrix = np.zeros(shape=(cluster_result.shape[0], cluster_result.shape[0]))
    for cluster in cluster_list:
        index = np.where(cluster_result==cluster)[0]
        
        for i in range(len(index)):
            for j in range(i+1, len(index)):
                cluster_affinity_matrix[index[i], index[j]] = 1
    cluster_affinity_matrix = cluster_affinity_matrix + cluster_affinity_matrix.T
    return cluster_affinity_matrix

def buildClustAgainstGraph(spatial_adj, cluster_result, clust_name):
    index = torch.where(cluster_result==clust_name)[0]
    spatial_adj[index, :] = 0
    spatial_adj[:, index] = 0
    for i in range(len(index)):
        for j in range(i, len(index)):
            spatial_adj[index[i], index[j]] = 1
            spatial_adj[index[j], index[i]] = 1
    return spatial_adj

def searchThreshold(adata):
    spot_num = adata.shape[0]
    gap = spot_num//100
    threshold = 0.1
    last_threshold = 1
    uncertainty = torch.tensor(adata.obs['mclust_uncertainty'].values)
    certain_label = (uncertainty<threshold).nonzero().squeeze().tolist()
    while abs(len(certain_label)-spot_num*0.75)>gap:
        temp = threshold
        if len(certain_label) > spot_num*0.75:
            if threshold > last_threshold:
                threshold = (threshold+last_threshold)/2
            else:
                threshold = threshold/2
        else:
            if threshold > last_threshold:
                threshold = 2*threshold
            else:
                threshold = (threshold+last_threshold)/2
        last_threshold = temp
        certain_label = (uncertainty<threshold).nonzero().squeeze().tolist()
    return threshold, certain_label

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=0.6, increment=0.01):
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
            print('resolution={}, cluster number={}'.format(res, count_unique))
            
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!."      
    return res

def preclust_star_graph(adata, graph, num_classes):
    graph.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat_aggregated'))
    pca = PCA(n_components=20, random_state=42)
    embedding = pca.fit_transform(graph.ndata['feat_aggregated'].numpy().copy())
    adata.obsm["emb_pca"] = embedding
    
    adata, _, mclust_uncertainty = mclust_R(adata, used_obsm="emb_pca", num_cluster=num_classes)
    
    label = adata.obs['mclust'].values
    certain_indices = (mclust_uncertainty<1e-5).astype(np.int0).nonzero()[0]
    certain_label = label[certain_indices]
    certain_emb = adata.obsm["emb_pca"][certain_indices]
    certain_label = certain_label.astype('int')-1
    certain_label_onehot = np.eye(num_classes, dtype=np.uint8)[certain_label]
    mean_emb = certain_label_onehot.T @ certain_emb
    mean_emb = (mean_emb.T/certain_label_onehot.sum(0).T).T
    mean_emb = np.nan_to_num(mean_emb)
    
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(certain_emb)
    _, indices = nn.kneighbors(mean_emb)                                              # 每个cluster的代表spot
    certain_tail = indices[certain_label].squeeze()

    adj = graph.adj().to_dense()
    for i in range(len(certain_indices)):
        head = certain_indices[i]
        adj_tail = adj[head].nonzero().squeeze()
        for tail in adj_tail:
            if label[head] != label[tail]:
                adj[head][tail] = 0

        tail = certain_tail[i]
        if label[head] == label[tail]:        
            adj[head][tail] = 1

    adj = sp.csr_array(adj)
    new_graph = dgl.from_scipy(adj)
    new_graph.ndata["feat"] = torch.tensor(graph.ndata["feat"])
    return new_graph

def preclust_full_graph(adata, graph, num_classes):
    graph.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat_aggregated'))
    pca = PCA(n_components=20, random_state=42)
    embedding = pca.fit_transform(graph.ndata['feat_aggregated'].numpy().copy())
    adata.obsm["emb_pca"] = embedding
    
    adata, _, mclust_uncertainty = mclust_R(adata, used_obsm="emb_pca", num_cluster=num_classes)
    label = adata.obs['mclust'].values
    certain_indices = (mclust_uncertainty<1e-5).astype(np.int0).nonzero()[0]
    certain_label = label[certain_indices]
    
    adj = graph.adj().to_dense()
    for k in range(num_classes):
        cluter_label = (certain_label==k+1).nonzero()[0]
        cluter_label_false = (certain_label != k+1).nonzero()[0]
        for i in cluter_label:
            adj[i][cluter_label] = 1
            adj[i][cluter_label_false] = 0
    
    adj = sp.csr_array(adj)
    new_graph = dgl.from_scipy(adj)
    new_graph.ndata["feat"] = torch.tensor(graph.ndata["feat"])
    return new_graph


