import networkx as nx
import torch
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
import random
import numpy as np
from fluidc import asyn_fluidc
import multiprocessing
import sys
from scipy import sparse as sp
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.utils.convert import from_networkx
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Planetoid,Coauthor
import time
import hashlib
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def positional_encoding(A, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    D = sp.diags(degree(A[0]).numpy() ** -0.5, dtype=float)
    A = to_dense_adj(A).squeeze(0).numpy()
    L = sp.eye(pos_enc_dim) - D * A * D

    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pe = torch.from_numpy(np.real(EigVec[:,1:pos_enc_dim+1])).float() 

    return lap_pe

def convert(string):
    return [ord(s) for s in string]

name = str(sys.argv[1])
if name in ["Cora"]:
    str_input = 40
    num_local = 8
elif name in ["Citeseer"]:
    str_input = 40
    num_local = 8
elif name in ["Pubmed"]:
    str_input = 50
    num_local = 10
elif name in ["photo", "computers", "cs", "physics"]:
    str_input = 60
    num_local = 12
elif name in ["collab"]:
    str_input = 60
    num_local = 12
elif name in ["ppa"]:
    str_input = 60 - 2 * 12
    num_local = 12
elif name in ["citation2"]:
    str_input = 60 - 2 * 12
    num_local = 12
k=int(sys.argv[2])

def get_com():
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    #dataset = PygLinkPropPredDataset(root="data", name='ogbl-collab')
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="data", name=name)
    elif name in ["photo", "computers"]:
        dataset = Amazon(root="data", name=name)
    elif name in ["cs", "physics"]:
        dataset = Coauthor(root="data", name=name)
    else:
        dataset = PygLinkPropPredDataset(root="data", name=f'ogbl-{name}')

    data = dataset[0]

    edge_index = data.edge_index
    #data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = T.ToSparseTensor()(data)
    data.full_adj_t = SparseTensor.from_edge_index(edge_index).t()
    data.full_adj_t = data.full_adj_t.to_symmetric()
    
    adj = data.full_adj_t.to_scipy()
    net = nx.from_scipy_sparse_array(adj)

    nets = sorted(nx.connected_components(net), key=len, reverse=True)
    if len(nets) > 1:
        lgst_net = nx.subgraph(net, nets[0])
        others = []
        for n in nets[1:]:
            others.extend(list(n))
        com = asyn_fluidc(lgst_net, k=num_local)
    else:
        com = asyn_fluidc(net, k=num_local)
    anchor_nodes = []
    node2cluster = {}
    node2localcom = {}
    nums = 0
    print(len(com))

    top_nodes_high = []  # 每个小社区 pagerank 前100
    top_nodes_low = []  # 每个小社区 pagerank 后100

    for i, c in enumerate(com):
        subnet = nx.subgraph(net, c)
        local_com = asyn_fluidc(subnet, k)
        for j, local_c in enumerate(local_com):
            start_time = time.time()
            local_subnet = nx.subgraph(subnet, local_c)
            #print("here")
            #度中心性
            #anchor_node = sorted(local_subnet.degree(), key=lambda x : x[1], reverse=True)[0][0]
            #介数中心性
            #anchor_node = max(nx.betweenness_centrality(local_subnet).items(), key=lambda x: x[1])[0]
            #接近中心性
            #anchor_node = max(nx.closeness_centrality(local_subnet).items(), key=lambda x: x[1])[0]
            #特征向量中心性
            #anchor_node = max(nx.eigenvector_centrality(local_subnet).items(), key=lambda x: x[1])[0]
            #PageRank中心性
            anchor_node = max(nx.pagerank(local_subnet).items(), key=lambda x: x[1])[0]
            #Katz中心性
            #anchor_node = max(nx.katz_centrality(local_subnet).items(), key=lambda x: x[1])[0]
            anchor_nodes.append(anchor_node)

            # 计算 PageRank
            pr = nx.pagerank(local_subnet)

            # 按 PageRank 值排序
            sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)

            # 记录前100和后100
            top_10_high = [node for node, _ in sorted_pr[:10]]
            top_10_low = [node for node, _ in sorted_pr[-10:]]

            top_nodes_high.extend(top_10_high)
            top_nodes_low.extend(top_10_low)

            for node in local_c:
                node2cluster[node] = i
                node2localcom[node] = i * k + j
            end_time = time.time()
            nums = nums + 1
            print(f"Computing {nums} local graph took {end_time - start_time:.4f}s")

    torch.save(top_nodes_high, name + '_top10_async_fluid.pt')
    torch.save(top_nodes_low, name + '_last10_com_async_fluid.pt')

    node2com = []
    membership = []
    node2anchor = []

    for i in range(len(net)):
        try:
            membership.append(node2cluster[i])
        except:
            membership.append(len(anchor_nodes))

    for i in range(len(net)):
        try:
            node2com.append(node2localcom[i])
        except:
            node2com.append(len(anchor_nodes))

    for i in range(len(net)):
        try:
            node2anchor.append(anchor_nodes[node2com[i]])
        except:
            node2anchor.append(i)

    torch.save(membership, name + '_node2com_async_fluid.pt')
    torch.save(node2com, name + '_node2local_com_async_fluid.pt')
    print(len(anchor_nodes))
    #print(anchor_nodes)
    #print(node2com)
    #print(node2anchor)
    torch.save(node2anchor, name + '_node2anchor_async_fluid.pt')

    num_centers = len(anchor_nodes)
    center_dists = torch.zeros((num_centers, num_centers), dtype=torch.long)
    for i in range(num_centers):
        for j in range(num_centers):
            if i == j:
                center_dists[i, j] = 0
            else:
                try:
                    dist = nx.shortest_path_length(net, anchor_nodes[i], anchor_nodes[j])
                except:
                    dist = 0
                center_dists[i, j] = dist
    torch.save(anchor_nodes, name + '_local_centers_async_fluid.pt')
    torch.save(center_dists, name + '_center_dists_async_fluid.pt')

    landmark_graph = nx.Graph()
    temperature=5.0
    for i in range(num_centers):
        for j in range(i+1, num_centers):
            dist = int(center_dists[i, j].item())
            weight = np.exp((-dist**2)/temperature)
            landmark_graph.add_edge(i, j, weight=weight)
            landmark_graph.add_edge(j, i, weight=weight)

    landmark_graph = from_networkx(landmark_graph)

    lap_pe = positional_encoding(landmark_graph.edge_index, k*num_local)
    lap_pe = np.concatenate([lap_pe, np.zeros(k*num_local-1).reshape(1,-1)], axis=0)

    lap_pe = lap_pe[membership]

    torch.save(lap_pe, name+'_lap_pe_async_fluid.pt')

    if len(nets) > 1:
        lgst_net = nx.subgraph(net, nets[0])
        com = asyn_fluidc(lgst_net, k=num_local)
    else:
        com = asyn_fluidc(net, k=num_local)
    torch.save(com, name+'_com_async_fluid.pt')
    torch.save(anchor_nodes, name+'_anchor_async_fluid.pt')

    return [i for i in range(data.num_nodes)], net, anchor_nodes, edge_index

def get_pe(net, anchor_nodes, nodes, map_ret):
    for node in nodes:
        pos_ret = dict()
        pos_enc = []
        for anchor_node in anchor_nodes:
            try:
                pos_enc.append(len(nx.algorithms.shortest_path(net, node, anchor_node)))
            except:
                pos_enc.append(0)
        pos_ret[node] = pos_enc
        map_ret.update(pos_ret)
        print(f"{node} node completed!!")

def main():
    mult_manager = multiprocessing.Manager()
    nodes, net, anchor_nodes, edge_index = get_com()
    WORKER_NUM = 128
    bs = int(len(nodes) / WORKER_NUM)
    if len(nodes) % WORKER_NUM != 0:
        bs += 1
    vec_process = []
    return_dict = mult_manager.dict()
    for pidx in range(WORKER_NUM):
        p = multiprocessing.Process(target=get_pe, args=(net, anchor_nodes, nodes[pidx * bs: min((pidx + 1) * bs, len(nodes))], return_dict))
        p.start()
        vec_process.append(p)
    for p in vec_process:
        p.join()

    ret = dict()
    ret.update(return_dict.copy())
    torch.save(ret, name+'_pe_dict.pt')
    pos_enc = []
    for i in range(len(nodes)):
        pos_enc.append(ret[i])
    pos_enc = np.array(pos_enc)
    max_d = 0
    for p in pos_enc:
        max_d = max(max_d, max(p))

    np.where(pos_enc == 0, max_d, pos_enc)
    for i, a in enumerate(anchor_nodes):
        pos_enc[a][i] = 0
        
    torch.save(pos_enc, name+'_pe_async_fluid.pt')

if __name__ == '__main__':
    main()
