#!/usr/bin/python
import numpy as np
import torch
from torch_geometric import utils
from torch_geometric.data import Data, ClusterData, ClusterLoader
from scipy import sparse as sp
from os.path import join
import gc

AGF_Folder = "AGF_Pipeline"

def load(adj, split):  
    mode, K, R, th_type, threshold = adj[0], adj[1], adj[2], adj[3], adj[4]
    
    adj = mode + ";" + str(K) + "," + str(R) 
    
#    print(split.upper())
    path_adj = join(AGF_Folder,"ADJs", split, mode)
    
    if th_type.upper() == "B":
#        print("Binary Threshold: " + str(threshold))
        adj_th = sp.coo_matrix(np.where(np.load(join(path_adj, adj + ".npy")) >= threshold, 1, 0))
    elif th_type.upper() == "T":
#        print("Threshold: " + str(threshold))
        a = np.load(join(path_adj, adj + ".npy")) 
        adj_th = sp.coo_matrix(np.where(a > threshold, a, 0))
        del a
    elif th_type.upper() == "N":         
        adj_th = sp.coo_matrix(np.load(join(path_adj, adj + ".npy")))


#    print("Computing PyG Graph")
    g_att = utils.from_scipy_sparse_matrix(adj_th)
    gc.collect()
    g = Data(edge_index = g_att[0], weight = g_att[1])

#    print("Adding features \n")
    features = np.load(join("FVS", split + "_" + mode + ".npy"))
    labels = np.load(join("FVS", "y_" + split + ".npy"))
    
    g['x'] = torch.from_numpy(np.array(features, dtype = 'f'))  
    g['y'] = torch.from_numpy(np.array(labels, dtype = 'f'))
    
    size = g.x.size(0)

    dataloader = ClusterLoader(ClusterData(g, 30), shuffle = True)
    
    return dataloader, size

