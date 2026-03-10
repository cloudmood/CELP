import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_sparse import SparseTensor

from fluidc import asyn_fluidc
from train_utils import get_train_test
from models import GCN, MLP, SAGE, LinkPredictor, CELP
from node_label import get_two_hop_adj
from utils import ( get_dataset, data_summary, get_git_revision_short_hash,
                   set_random_seeds, str2bool, get_data_split, initial_embedding)
from scipy import sparse as sp
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.utils.convert import from_networkx
import random
import networkx as nx
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader
from node_label import spmdiff_
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib as mpl
mpl.use('tkagg')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

CELP_dict={
    "CELP": "combine",
    "CELP+": "prop_only",
}
def approximate_ppr(edge_index, num_nodes, alpha=0.15, topk=32, device="cpu"):
    """
    近似 Personalized PageRank (PPR) 矩阵，返回稀疏 COO Tensor
    edge_index: [2, E] torch.LongTensor
    num_nodes: 节点数
    alpha: teleport 参数 (默认0.15)
    topk: 每个节点保留 top-k PPR 概率
    device: 返回的稀疏矩阵存放的 device
    """
    # Step 1: 构建 scipy 稀疏邻接
    row, col = edge_index.cpu().numpy()
    data = np.ones(len(row))
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    # 无向图 -> 对称
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Step 2: 归一化邻接 (D^-1 A)
    deg = np.array(adj.sum(1)).flatten()
    deg_inv = 1.0 / np.maximum(deg, 1e-12)
    D_inv = sp.diags(deg_inv)
    trans = D_inv.dot(adj)  # row-normalized adjacency

    # Step 3: Power iteration 近似 PPR
    # PPR:  Pi = alpha * sum_{k=0}^∞ (1-alpha)^k * T^k
    # 用有限步迭代逼近
    ppr_rows, ppr_cols, ppr_vals = [], [], []
    for i in range(num_nodes):
        # 初始向量 e_i
        x = np.zeros(num_nodes)
        x[i] = 1.0
        p = np.zeros(num_nodes)  # 存储 PPR
        r = x.copy()

        for _ in range(10):  # 迭代次数可调
            p += alpha * r
            r = (1 - alpha) * trans.T.dot(r)

        # 只保留 top-k
        topk_idx = np.argpartition(-p, topk)[:topk]
        ppr_rows.extend([i] * len(topk_idx))
        ppr_cols.extend(topk_idx.tolist())
        ppr_vals.extend(p[topk_idx].tolist())

    # Step 4: 转 PyTorch 稀疏 COO
    indices = torch.tensor([ppr_rows, ppr_cols], dtype=torch.long, device=device)
    values = torch.tensor(ppr_vals, dtype=torch.float, device=device)
    Pi = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    return Pi.coalesce()

def build_constraint_matrix(node2local, edge_index, num_nodes, alpha=0.15, topk=32):
    """
    快速构建用于对比约束的稀疏矩阵 M。

    设计目标：
    - 同社区边作为正样本（权重为 1）
    - 不同社区边作为分母干扰项（权重为极小值 eps，不参与梯度但参与归一化）
    - 仅使用图的实际边（邻接），避免昂贵的 PPR 全图计算

    参数含义保持不变，但此实现忽略 PPR，仅基于邻接快速生成。
    返回: torch.sparse_coo_tensor(coalesced)
    """
    eps = 1e-6
    row, col = edge_index
    device = row.device

    node2local = node2local.to(device).long()
    pos_mask = (node2local[row] == node2local[col])
    neg_mask = ~pos_mask

    pos_row = row[pos_mask]
    pos_col = col[pos_mask]
    neg_row = row[neg_mask]
    neg_col = col[neg_mask]

    indices = torch.stack([
        torch.cat([pos_row, neg_row], dim=0),
        torch.cat([pos_col, neg_col], dim=0)
    ], dim=0)
    values = torch.cat([
        torch.ones(pos_row.size(0), device=device),
        torch.full((neg_row.size(0),), eps, device=device)
    ], dim=0)

    M = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()
    return M

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    # dataset setting
    parser.add_argument('--dataset', type=str, default='collab')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--use_valedges_as_input', type=str2bool, default='False', help='whether to use val edges as input')
    parser.add_argument('--year', type=int, default=-1)

    # CELP settings # adapted from MPLP
    parser.add_argument('--signature_dim', type=int, default=1024, help="the node signature dimension `F` in CELP")
    parser.add_argument('--minimum_degree_onehot', type=int, default=-1, help='the minimum degree of hubs with onehot encoding to reduce variance')
    parser.add_argument('--mask_target', type=str2bool, default='True', help='whether to mask the target edges to remove the shortcut')
    parser.add_argument('--use_degree', type=str, default='none', choices=["none","mlp","AA","RA"], help="rescale vector norm to facilitate weighted count")
    parser.add_argument('--signature_sampling', type=str, default='torchhd', help='whether to use torchhd to randomize vectors', choices=["torchhd","gaussian","onehot"])
    parser.add_argument('--fast_inference', type=str2bool, default='False', help='whether to enable a faster inference by caching the node vectors')
    parser.add_argument('--adj2', type=str2bool, default="False", help='Whether to use 2-hop adj for CELP+prop_only.')
    
    # model setting
    parser.add_argument('--predictor', type=str, default='CELP', choices=["inner","mlp","ENL",
    "CELP+","CELP"])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--xdp', type=float, default=0.2)
    parser.add_argument('--feat_dropout', type=float, default=0.5)
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_feature', type=str2bool, default='True', help='whether to use node features as input')
    parser.add_argument('--feature_combine', type=str, default='hadamard', choices=['hadamard','plus_minus'], help='how to represent a link with two nodes features')
    parser.add_argument('--jk', type=str2bool, default='True', help='whether to use Jumping Knowledge')
    parser.add_argument('--batchnorm_affine', type=str2bool, default='True', help='whether to use Affine in BatchNorm')
    parser.add_argument('--use_embedding', type=str2bool, default='False', help='whether to train node embedding')
    # parser.add_argument('--dgcnn', type=str2bool, default='False', help='whether to use DGCNN as the target edge pooling')

    # training setting
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--test_batch_size', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--metric', type=str, default='Hits@50', help='main evaluation metric')
    parser.add_argument('--global_label', type=str, default='tea', choices=['tea', 'stu'],
                        help='whether to use node features with global label')
    parser.add_argument('--local_label', type=str, default='tea', choices=['tea', 'stu'],
                        help='whether to use node features with local label')
    parser.add_argument('--complete', type=str, default='t', choices=['t', 'f'],
                        help='whether to use Structure Enhancement Module')
    # misc
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--data_split_only', type=str2bool, default='False')
    parser.add_argument('--print_summary', type=str, default='')

    args = parser.parse_args()

    name = args.dataset
    if name in ["Cora"]:
        str_input = 40
        #str_input = 27
        num_local = 8
    elif name in ["Citeseer"]:
        str_input = 40
        #str_input = 36
        num_local = 8
    elif name in ["Pubmed"]:
        str_input = 50
        #str_input = 41
        num_local = 10
    elif name in ["photo"]:
        str_input = 60
        #str_input = 15
        num_local = 12
    elif name in ["computers"]:
        str_input = 60 + 2 * 12
        #str_input = 19
        num_local = 12
    elif name in ["physics"]:
        str_input = 60 + 2*12
        num_local = 12
    elif name in ["cs"]:
        str_input = 60 + 2 * 12
        num_local = 12
    elif name in ["ogbl-collab"]:
        str_input = 60
        #str_input = 196
        num_local = 12
    elif name in ["ogbl-ppa"]:
        str_input = 60 - 2 * 12
        num_local = 12
    elif name in ["ogbl-citation2"]:
        str_input = 60 - 2 * 12
        num_local = 12


    # start time
    start_time = time.time()
    set_random_seeds(234)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # device = torch.device('cpu')

    data, split_edge = get_dataset(args.dataset_dir, args.dataset, args.use_valedges_as_input, args.year)
    if args.dataset == "ogbl-citation2":
        args.metric = "MRR"
    if data.x is None:
        args.use_feature = False

    if args.print_summary:
        data_summary(args.dataset, data, header='header' in args.print_summary, latex='latex' in args.print_summary);exit(0)
    else:
        print(args)
    data = data.to(device)
    final_log_path = Path(args.log_dir) / f"{args.dataset}_jobID_{os.getenv('JOB_ID','None')}_PID_{os.getpid()}_{int(time.time())}.log"    
    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    print('Command line input: ' + cmd_input + ' is saved.')

    
    train, test, evaluator, loggers, loggers2 = get_train_test(args)

    def build_custom_edge_index_by_degree(edge_index, num_nodes):
        # 计算每个节点的度
        row, col = edge_index
        deg = torch.bincount(torch.cat([row, col]), minlength=num_nodes)

        # 建立每个节点重复 k 次的列表（边不能随意构造，需要实际构造 k 条合法边）
        new_edges = []

        adj_dict = {i: set() for i in range(num_nodes)}
        for u, v in zip(row.tolist(), col.tolist()):
            adj_dict[u].add(v)
            adj_dict[v].add(u)

        for node in range(num_nodes):
            neighbors = list(adj_dict[node])
            count = deg[node].item()

            if len(neighbors) == 0 or count == 0:
                continue

            for _ in range(count):
                # 随机选择一个邻居构造一条边
                sampled_neighbor = random.choice(neighbors)
                new_edges.append((node, sampled_neighbor))

        if len(new_edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)

        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        return new_edge_index

    val_max = 0.0
    for run in range(args.runs):
        if run == 0:
            if not args.dataset.startswith('ogbl-'):
                data, split_edge = get_data_split(args.dataset_dir, args.dataset, args.val_ratio, args.test_ratio, run=run)
                data = T.ToSparseTensor(remove_edge_index=False)(data)
                # Use training + validation edges for inference on test set.
                if args.use_valedges_as_input:
                    val_edge_index = split_edge['valid']['edge'].t()
                    full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
                    data.full_adj_t = SparseTensor.from_edge_index(full_edge_index,
                                                            sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
                    data.full_adj_t = data.full_adj_t.to_symmetric()
                else:
                    data.full_adj_t = data.adj_t
                if args.data_split_only:
                    if run == args.runs - 1:
                        exit(0)
                    else:
                        continue
            #data.custom_edge_index = build_custom_edge_index_by_degree(data.edge_index, data.num_nodes)
            #pos_train_edge = split_edge['train']['edge'].to(device)
            #print('pos=',pos_train_edge)
            if args.adj2:
                print("Computing 2-hop adj. This may take a while.")
                start_time = time.time()
                adj_t_no_value = data.adj_t.clone().set_value_(None)
                _, adj2 = get_two_hop_adj(adj_t_no_value)
                end_time = time.time()
                print(f"Computing 2-hop adj took {end_time - start_time:.4f}s")
            else:
                adj2 = None
            if args.minimum_degree_onehot > 0:
                d_v = data.adj_t.sum(dim=0).to_dense()
                nodes_to_one_hot = d_v >= args.minimum_degree_onehot
                one_hot_dim = nodes_to_one_hot.sum()
                print(f"number of nodes to onehot: {int(one_hot_dim)}")
            #'''
            pe = torch.load(args.dataset.split("-")[-1]+'_pe_async_fluid.pt')
            lap_pe = torch.load(args.dataset.split("-")[-1]+'_lap_pe_async_fluid.pt')
            node2com = torch.load(args.dataset.split("-")[-1]+'_node2com_async_fluid.pt')
            node2local = torch.load(args.dataset.split("-")[-1] + '_node2local_com_async_fluid.pt')
            #top_10 = torch.load(args.dataset.split("-")[-1] + '_top10_async_fluid.pt')
            #last_10 = torch.load(args.dataset.split("-")[-1] + '_last10_com_async_fluid.pt')
            #node2anchor = torch.load(args.dataset.split("-")[-1] + '_node2anchor_async_fluid.pt')
            #lap_pe, com, anchor_nodes, pos_enc = get_com(data.adj_t.to_scipy(), data.num_nodes, 8)
            data['pos_enc'] = torch.tensor(pe, dtype=torch.float)
            data['lap_pe'] = torch.tensor(lap_pe, dtype=torch.float)
            #data['node2anchor'] = torch.tensor(node2anchor, dtype=torch.long)
            #data['ppr'] = approximate_ppr(data.edge_index, data.num_nodes, alpha=0.15, topk=2)


            data['node2com'] = torch.tensor(node2com, dtype=torch.float)
            data['node2local'] = torch.tensor(node2local, dtype=torch.float)
            try:
                center_dists = torch.load(args.dataset.split("-")[-1] + '_center_dists_async_fluid.pt')
                data['center_dis'] = torch.tensor(center_dists, dtype=torch.float)
            except:
                data['center_dis'] = torch.zeros((int(max(node2local)+1), int(max(node2local)+1)), dtype=torch.float)
            try:
                local_centers = torch.load(args.dataset.split("-")[-1] + '_local_centers_async_fluid.pt')
                data['local_centers'] = torch.tensor(local_centers, dtype=torch.long)
            except:
                data['local_centers'] = torch.arange(int(max(node2local)+1))
            data['M'] = build_constraint_matrix(data['node2local'], data.edge_index,
                                                data.num_nodes, alpha=0.15, topk=64)
            print('lap_pe=', data['lap_pe'].shape)
            print('pos_enc=', data['pos_enc'].shape)
            #print('node2com=', data['node2com'].shape)
            #print('node2local=', data['node2local'].shape)

            #data.x = torch.cat((data.x, lap_pe, pos_enc), dim=1).to(torch.float32)
            #data.num_features = data.x.shape[1]
            #print('new_data=', data.x.shape)
            data = data.to(device)


            if args.use_embedding:
                emb = initial_embedding(data, args.hidden_channels, device)
            else:
                emb = None
            if 'gcn' in args.encoder:
                encoder = GCN(data.num_features, args.hidden_channels,
                            args.hidden_channels, args.num_layers,
                            args.feat_dropout, args.xdp, args.use_feature, args.jk, args.encoder, emb,
                              str_input=str_input, num_local=num_local, mode=args.global_label).to(device)
            elif args.encoder == 'sage':
                encoder = SAGE(data.num_features, args.hidden_channels,
                            args.hidden_channels, args.num_layers,
                            args.feat_dropout, args.xdp, args.use_feature, args.jk, emb).to(device)
            elif args.encoder == 'mlp':
                encoder = MLP(args.num_layers, data.num_features,
                              args.hidden_channels, args.hidden_channels, args.dropout).to(device)

            predictor_in_dim = args.hidden_channels * int(args.use_feature or args.use_embedding)
                                # * (1 + args.jk * (args.num_layers - 1))
            if args.predictor in ['inner','mlp']:
                predictor = LinkPredictor(args.predictor, predictor_in_dim, args.hidden_channels, 1,
                                        args.num_layers, args.feat_dropout)
            # elif args.predictor == 'ENL':
            #     predictor = NaiveNodeLabelling(predictor_in_dim, args.hidden_channels,
            #                             args.num_layers, args.feat_dropout, args.num_hops,
            #                             dgcnn=args.dgcnn, use_degree=args.use_degree).to(device)
            elif 'CELP' in args.predictor:
                prop_type = CELP_dict[args.predictor]
                predictor = CELP(predictor_in_dim, args.hidden_channels,
                                        args.num_layers, args.feat_dropout, args.label_dropout, args.num_hops,
                                        prop_type=prop_type, signature_sampling=args.signature_sampling,
                                        use_degree=args.use_degree, signature_dim=args.signature_dim,
                                        minimum_degree_onehot=args.minimum_degree_onehot, batchnorm_affine=args.batchnorm_affine,
                                        feature_combine=args.feature_combine, adj2=args.adj2, use_local=args.local_label)
            predictor = predictor.to(device)

            encoder.reset_parameters()
            predictor.reset_parameters()
            parameters = list(encoder.parameters()) + list(predictor.parameters())
            optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
            total_params = sum(p.numel() for param in parameters for p in param)
            print(f'Total number of parameters is {total_params}')

            cnt_wait = 0
            best_val = 0.0

            for epoch in range(1, 1 + args.epochs):
                loss = train(encoder, predictor, data, split_edge,
                             optimizer, args.batch_size, args.mask_target, args.dataset,
                             num_neg=args.num_neg, adj2=adj2, mode=args.global_label)

                results = test(encoder, predictor, data, split_edge,
                                evaluator, args.test_batch_size, args.use_valedges_as_input, args.fast_inference, adj2=adj2,
                                mode=args.global_label)

                if results[args.metric][0] >= best_val:
                    best_val = results[args.metric][0]
                    best_encoder = encoder
                    cnt_wait = 0
                else:
                    cnt_wait +=1

                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        valid_hits, test_hits = result
                        to_print = (f'Run: {run + 1:02d}, ' +
                                f'Epoch: {epoch:02d}, '+
                                f'Loss: {loss:.4f}, '+
                                f'Valid: {100 * valid_hits:.2f}%, '+
                                f'Test: {100 * test_hits:.2f}%')
                        print(key)
                        print(to_print)
                        with open(final_log_path, 'a') as f:
                            print(key, file=f)
                            print(to_print, file=f)
                    print('---')

                if cnt_wait >= args.patience:
                    break
                # import json
                # with open("threshold.json") as f:
                #     threshold = json.load(f).get(args.dataset,[]) # threshold is a list [(epoch, performance)]
                # for t_epoch, t_value in threshold:
                #     if epoch >= t_epoch and results[args.metric][1]*100 < t_value:
                #         print(f"Discard due to low test performance {results[args.metric][1]} < {t_value} after epoch {t_epoch}")
                #         break
                # else:
                #     continue
                # break

            for key in loggers.keys():
                print(key)
                loggers[key].print_statistics(run)
                with open(final_log_path, 'a') as f:
                    print(key,file=f)
                    loggers[key].print_statistics(run=run, file=f)

        #--------------------------

            '''
            if args.complete == 't':
                pos_train_edge = split_edge['train']['edge'].to(device)
                with torch.no_grad():
                    adj_t = data.adj_t
                    edge_index = data.edge_index
                    num_nodes = adj_t.size(0)
    
                    # 计算每个节点的度
                    degrees = adj_t.sum(dim=1).to(torch.float)
                    _, top_deg_nodes = torch.topk(degrees, 100)
                    top_deg_set = set(top_deg_nodes.tolist())
    
                    # 构建原始连接对集合（便于判断是否为负样本）
                    existing_edges = set(map(tuple, edge_index.t().tolist()))
    
                    # 构建候选集：没有边连接的 (u,v)，且 u 或 v 的度在 top 100 中
                    candidate_edges = []
                    for u in range(num_nodes):
                        for v in range(u + 1, num_nodes):  # 只考虑 u < v 保持无向图对称性
                            if (u, v) not in existing_edges and (v, u) not in existing_edges:
                                if u in top_deg_set or v in top_deg_set:
                                    if data['node2local'][u] == data['node2local'][v]:
                                        candidate_edges.append((u, v))
                    print(f'Candidate edges size: {len(candidate_edges)}')
    
                    # 转为 tensor
                    candidate_edges = torch.tensor(candidate_edges, dtype=torch.long).t().to(device)  # shape: [2, N]
    
                    # 节点编码
                    if args.global_label == 'tea':
                        h = best_encoder(data.x, data.adj_t, data['pos_enc'], data['lap_pe'], mode='tea')
                    else:
                        h = best_encoder(data.x, data.adj_t, mode='stu')
    
                    # 预测候选边得分
                    out = predictor(h, data.adj_t, candidate_edges, node2local=data['node2local'], adj2=adj2).squeeze()
    
                    # 选取 top 10% 边
                    k = int(out.size(0) * 0.05)
                    _, topk_mask = torch.topk(out, k=k, largest=True)
                    add_edge = candidate_edges[:, topk_mask]  # shape: [2, K]
                    print('add_size =', add_edge.shape)
    
                    # 更新 edge_index 和 adj_t
                    data.edge_index = torch.cat([data.edge_index, add_edge], dim=-1)
                    add_adj_t = SparseTensor.from_edge_index(add_edge,
                                                             sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
                    data.adj_t = data.adj_t + add_adj_t
                    data.full_adj_t = data.full_adj_t + add_adj_t
            '''



            if args.complete == 't':
                pos_train_edge = split_edge['train']['edge'].to(device)
                '''
                top_10_set = set(top_10)
                last_10_set = set(last_10)
    
                # 全部验证边（valid）
                pos_test_edge = split_edge['test']['edge'].to(device)  # [N1, 2]
                neg_test_edge = split_edge['test']['edge_neg'].to(device)  # [N2, 2]
    
                # 辅助函数：筛选边的两端是否都在指定集合中
                def mask_edges(edge, node_set):
                    src, dst = edge[:, 0], edge[:, 1]
                    return torch.tensor(
                        [(u.item() in node_set and v.item() in node_set) for u, v in zip(src, dst)],
                        device=edge.device,
                        dtype=torch.bool
                    )
    
                # 分别筛选 top10 和 last10 的正负样本边
                mask_top10_pos = mask_edges(pos_test_edge, top_10_set)
                mask_top10_neg = mask_edges(neg_test_edge, top_10_set)
    
                mask_last10_pos = mask_edges(pos_test_edge, last_10_set)
                mask_last10_neg = mask_edges(neg_test_edge, last_10_set)
    
                edge_top10 = pos_test_edge[mask_top10_pos]
                edge_top10_neg = neg_test_edge[mask_top10_neg]
                edge_last10 = pos_test_edge[mask_last10_pos]
                edge_last10_neg = neg_test_edge[mask_last10_neg]
                label_top10 = torch.ones(mask_top10_pos.sum(), device=device)
                label_last10 = torch.ones(mask_last10_pos.sum(), device=device)
                '''
                def predict_edges(edge_subset):
                    with torch.no_grad():
                        if args.mask_target:
                            adj_t = data.adj_t
                            undirected_edges = torch.cat((edge_subset.t(), edge_subset.t().flip(0)), dim=-1)
                            target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
                            adj_t = spmdiff_(adj_t, target_adj, keep_val=True)
                        else:
                            adj_t = data.adj_t

                        if args.global_label == 'tea':
                            h = best_encoder(data.x, adj_t, data['pos_enc'], data['lap_pe'], mode='tea')
                        else:
                            h = best_encoder(data.x, adj_t, mode='stu')

                        preds = predictor(h, adj_t, edge_subset.t(), node2local=data['node2local'], center_dis=data.get('center_dis', None), adj2=adj2).squeeze()
                        #auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())

                    return preds
                '''
                #preds_top10, auc_top10 = predict_edges(edge_top10, label_top10)
                #preds_last10, auc_last10 = predict_edges(edge_last10, label_last10)
                preds_top10 = predict_edges(edge_top10)
                preds_last10 = predict_edges(edge_last10)
                preds_top10_neg = predict_edges(edge_top10_neg)
                preds_last10_neg = predict_edges(edge_last10_neg)
                #print(f"AUC on top_10 edges: {auc_top10:.4f}")
                #print(f"AUC on last_10 edges: {auc_last10:.4f}")
    
                plt.figure(figsize=(10, 6))
                sns.kdeplot(preds_top10.cpu().numpy(), label='Top-10 Pos', linewidth=2, color='red')
                sns.kdeplot(preds_last10.cpu().numpy(), label='Last-10 Pos', linewidth=2, color='blue')
                sns.kdeplot(preds_top10_neg.cpu().numpy(), label='Top-10 Neg', linewidth=2)
                sns.kdeplot(preds_last10_neg.cpu().numpy(), label='Last-10 Neg', linewidth=2)
    
                plt.title("Predicted Link Probabilities")
                plt.xlabel("Prediction Score")
                plt.ylabel("Density")
                plt.legend(loc='best')
                plt.tight_layout()
                plt.show()
                '''
                with torch.no_grad():
                    remove_edges_batch = []

                    for perm in DataLoader(range(pos_train_edge.size(0)), args.batch_size):
                        edge = pos_train_edge[perm].t()

                        if args.mask_target:
                            adj_t = data.adj_t
                            undirected_edges = torch.cat((edge, edge.flip(0)), dim=-1)
                            target_adj = SparseTensor.from_edge_index(
                                undirected_edges, sparse_sizes=adj_t.sizes()
                            )
                            adj_t = spmdiff_(adj_t, target_adj, keep_val=True)
                        else:
                            adj_t = data.adj_t

                        if args.global_label == 'tea':
                            h = best_encoder(data.x, adj_t, data['pos_enc'], data['lap_pe'], mode='tea')
                        else:
                            h = best_encoder(data.x, adj_t, mode='stu')

                        out = predictor(h, adj_t, edge, node2local=data['node2local'], adj2=adj2).squeeze()
                        # 去掉5%最不可靠的正例边
                        k = max(1, int(out.size(0) * 0.05), )
                        #k = 0
                        _, mask1 = torch.topk(out, k, largest=False)
                        mask1 = mask1.to(edge.device)
                        remove_edges_batch.append(edge[:, mask1])

                    remove_edge = torch.cat(remove_edges_batch, dim=1)  # shape: [2, total_to_remove]
                    print('remove_size =', remove_edge.shape)

                    ### 从 adj_t 和 full_adj_t 中删除边 ###
                    remove_adj_t = SparseTensor.from_edge_index(
                        remove_edge, sparse_sizes=(data.num_nodes, data.num_nodes)
                    ).coalesce()
                    data.adj_t = spmdiff_(data.adj_t, remove_adj_t)
                    data.full_adj_t = spmdiff_(data.full_adj_t, remove_adj_t)

                    ### 从 edge_index 中删除边 ###
                    edge_set = set(map(tuple, data.edge_index.t().tolist()))
                    remove_set = set(map(tuple, remove_edge.t().tolist()))
                    new_edges = [e for e in edge_set if e not in remove_set]
                    data.edge_index = torch.tensor(new_edges, dtype=torch.long, device=data.edge_index.device).t()

                    ### 从 split_edge['train']['edge'] 中删除边 ###
                    train_edge = split_edge['train']['edge']
                    train_set = set(map(tuple, train_edge.tolist()))
                    new_train_edges = [e for e in train_set if e not in remove_set]
                    split_edge['train']['edge'] = torch.tensor(new_train_edges, dtype=torch.long,
                                                               device=train_edge.device)

            if args.complete == 't':
                pos_train_edge = split_edge['train']['edge'].to(device)
                add_edges_batch = []

                with torch.no_grad():
                    for perm in DataLoader(range(pos_train_edge.size(0)), args.batch_size):
                        pos_edge = pos_train_edge[perm].t()

                        if args.mask_target:
                            adj_t = data.adj_t
                            undirected_edges = torch.cat((pos_edge, pos_edge.flip(0)), dim=-1)
                            target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
                            adj_t = spmdiff_(adj_t, target_adj, keep_val=True)
                        else:
                            adj_t = data.adj_t

                        if args.global_label == 'tea':
                            h = best_encoder(data.x, adj_t, data['pos_enc'], data['lap_pe'], mode='tea')
                        else:
                            h = best_encoder(data.x, adj_t, mode='stu')

                        # ✅ 按当前 batch 动态负采样：数量 = batch_size * args.num_neg
                        neg_edge = negative_sampling(
                            data.edge_index,
                            num_nodes=data.num_nodes,
                            num_neg_samples=pos_edge.size(1) * args.num_neg
                        ).to(device)

                        out = predictor(h, adj_t, neg_edge, node2local=data['node2local'], adj2=adj2).squeeze()

                        # 选得分最高的5%负边作为伪阳性边加入图中
                        k = max(1, int(out.size(0) * 0.2),)
                        #k = 0
                        _, mask1 = torch.topk(out, k=k, largest=True)
                        selected_neg_edge = neg_edge[:, mask1]  # shape [2, k]

                        add_edges_batch.append(selected_neg_edge)

                # ✅ 合并所有 batch 中的高得分负边
                add_edge = torch.cat(add_edges_batch, dim=1)  # shape [2, total_selected]
                print('add_size =', add_edge.shape)

                # ✅ 加入图结构
                data.edge_index = torch.cat([data.edge_index, add_edge], dim=-1).to(device)

                add_adj_t = SparseTensor.from_edge_index(
                    add_edge, sparse_sizes=(data.num_nodes, data.num_nodes)
                ).coalesce()

                data.adj_t = data.adj_t + add_adj_t
                data.full_adj_t = data.full_adj_t + add_adj_t
                #data.custom_edge_index = build_custom_edge_index_by_degree(data.edge_index, data.num_nodes).to(device)


        #encoder.reset_parameters()
        #predictor.reset_parameters()
        #parameters = list(encoder.parameters()) + list(predictor.parameters())


        optimizer2 = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
        # 1. 保存旧优化器的状态字典
        old_state_dict = optimizer.state_dict()
        # 2. 更新新优化器的状态字典（保留学习率、动量等）
        optimizer2.load_state_dict(old_state_dict)

        cnt_wait = 0
        best_val = 0.0
        best_test = 0.0

        for epoch in range(1, 1 + args.epochs):
            loss = train(encoder, predictor, data, split_edge,
                         optimizer2, args.batch_size, args.mask_target, args.dataset,
                         num_neg=args.num_neg, adj2=adj2, mode=args.global_label)

            results = test(encoder, predictor, data, split_edge,
                            evaluator, args.test_batch_size, args.use_valedges_as_input, args.fast_inference,
                           adj2=adj2, mode=args.global_label)

            if results[args.metric][0] >= best_val and results[args.metric][1] < best_test:
                a=1
            else:
                for key, result in results.items():
                    loggers2[key].add_result(run, result)

            if results[args.metric][0] >= best_val:
                best_val = results[args.metric][0]
                cnt_wait = 0
            else:
                cnt_wait +=1

            if results[args.metric][1] >= best_test:
                best_test = results[args.metric][1]


            if epoch % args.log_steps == 0:
                for key, result in results.items():
                    valid_hits, test_hits = result
                    to_print = (f'2Run: {run + 1:02d}, ' +
                            f'Epoch: {epoch:02d}, '+
                            f'Loss: {loss:.4f}, '+
                            f'Valid: {100 * valid_hits:.2f}%, '+
                            f'Test: {100 * test_hits:.2f}%')
                    print(key)
                    print(to_print)
                    with open(final_log_path, 'a') as f:
                        print(key, file=f)
                        print(to_print, file=f)
                print('---')

            if cnt_wait >= args.patience:
                break
            # import json
            # with open("threshold.json") as f:
            #     threshold = json.load(f).get(args.dataset,[]) # threshold is a list [(epoch, performance)]
            # for t_epoch, t_value in threshold:
            #     if epoch >= t_epoch and results[args.metric][1]*100 < t_value:
            #         print(f"Discard due to low test performance {results[args.metric][1]} < {t_value} after epoch {t_epoch}")
            #         break
            # else:
            #     continue
            # break

        for key in loggers2.keys():
            print(key)
            loggers2[key].print_statistics(run)
            with open(final_log_path, 'a') as f:
                print(key,file=f)
                loggers2[key].print_statistics(run=run, file=f)


        print('##############################################################')

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(final_log_path, 'a') as f:
            print(key,file=f)
            loggers[key].print_statistics(file=f)

    print('##############################################################')

    if args.complete == 't':
        for key in loggers2.keys():
            print(key)
            loggers2[key].print_statistics()
            with open(final_log_path, 'a') as f:
                print(key,file=f)
                loggers2[key].print_statistics(file=f)

    # end time
    end_time = time.time()
    with open(final_log_path, 'a') as f:
        print(f"Total time: {end_time - start_time:.4f}s", file=f)




if __name__ == "__main__":
    main()
