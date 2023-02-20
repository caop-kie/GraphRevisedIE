import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, knn_graph
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from .model_utils import GCNConv_diag, GCNConv_dense, EOS
import torch_sparse as ts


class GRCN(torch.nn.Module):
    # 512, 512, device, 1024, 512, 512, dense
    def __init__(self, num_features, num_classes, device, node_graph_hid_dim1,
                 node_graph_hid_dim2, feature_dim, layer_type):
        super(GRCN, self).__init__()
        self.num_features = num_features
        self.node_graph_nhid = node_graph_hid_dim1   # 1024
        self.node_graph_nhid2 = node_graph_hid_dim2
        self.nhid = feature_dim
        self.conv1 = GCNConv_dense(num_features, self.nhid)
        self.conv2 = GCNConv_dense(self.nhid, num_classes)
        if layer_type == "dense":
            self.node_graph = GCNConv_dense(num_features, self.node_graph_nhid)
            self.node_graph2 = GCNConv_dense(self.node_graph_nhid, self.node_graph_nhid2)
        elif layer_type == "diag":
            self.node_graph = GCNConv_diag(num_features, device)
            self.node_graph2 = GCNConv_diag(num_features, device)
        else:
            exit("wrong layer type")
        self.F = torch.relu
        self.F_graph = torch.tanh
        self.dropout = 0.5
        self.K = 10
        self.mask = None
        self.Adj_new = None
        self._normalize = True
        self.device = device
        self.reduce = 'knn'
        # self.sparse = False
        self.norm_mode = "sym"

    def init_para(self):
        self.conv1.init_para()
        self.conv2.init_para()
        self.conv_graph.init_para()
        self.conv_graph2.init_para()

    def graph_parameters(self):
        return list(self.conv_graph.parameters()) + list(self.conv_graph2.parameters())

    def base_parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())

    def cal_similarity_graph(self, node_embeddings):
        # similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
        similarity_graph = torch.mm(node_embeddings[:, :int(self.num_features/2)], node_embeddings[:, :int(self.num_features/2)].t())
        similarity_graph += torch.mm(node_embeddings[:, int(self.num_features/2):], node_embeddings[:, int(self.num_features/2):].t())
        return similarity_graph

    def normalize(self, adj, mode="sym"):
        sparse = False
        if not sparse:
            if mode == "sym":
                inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
                return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
            elif mode == "row":
                inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
                return inv_degree[:, None] * adj
            else:
                exit("wrong norm mode")
        else:
            adj = adj.coalesce()
            if mode == "sym":
                inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + EOS)
                D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
            elif mode == "row":
                inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
                D_value = inv_degree[adj.indices()[0]]
            else:
                exit("wrong norm mode")
            new_values = adj.values() * D_value
            return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).to(self.device)

    def _sparse_graph(self, raw_graph, K):
        sparse = False
        if self.reduce == "knn":
            values, indices = raw_graph.topk(k=int(K), dim=-1)
            # print(values, indices)
            assert torch.sum(torch.isnan(values)) == 0
            assert torch.max(indices) < raw_graph.shape[1]
            if not sparse:
                self.mask = torch.zeros(raw_graph.shape).to(self.device)
                self.mask[torch.arange(raw_graph.shape[0]).view(-1,1), indices] = 1.
                self.mask[indices, torch.arange(raw_graph.shape[1]).view(-1,1)] = 1.
            else:
                inds = torch.stack([torch.arange(raw_graph.shape[0]).view(-1,1).expand(-1,int(K)).contiguous().view(1,-1)[0].to(self.device),
                                     indices.view(1,-1)[0]])
                inds = torch.cat([inds, torch.stack([inds[1], inds[0]])], dim=1)
                values = torch.cat([values.view(1,-1)[0], values.view(1,-1)[0]])
                return inds, values
        else:
            exit("wrong sparsification method")
        self.mask.requires_grad = False
        sparse_graph = raw_graph * self.mask
        return sparse_graph

    def _node_embeddings(self, input, Adj):
        norm_Adj = self.normalize(Adj, self.norm_mode)
        if self.F_graph != "identity":
            node_embeddings = self.F_graph(self.node_graph(input, norm_Adj))
            node_embeddings = self.node_graph2(node_embeddings, norm_Adj)
        else:
            node_embeddings = self.node_graph(input, norm_Adj)
            node_embeddings = self.node_graph2(node_embeddings, norm_Adj)
        if self._normalize:
            node_embeddings = F.normalize(node_embeddings, dim=1, p=2)
        return node_embeddings

    def forward(self, input, Adj):
        B, N, D = input.shape
        x = torch.zeros((B, N, D), device=input.device)
        Adj.requires_grad = False
        for i in range(B):
            node_embeddings = self._node_embeddings(input[i], Adj[i])
            Adj_new = self.cal_similarity_graph(node_embeddings)

            Adj_new = self._sparse_graph(Adj_new, self.K)
            Adj_new = self.normalize(Adj[i] + Adj_new, self.norm_mode)

            conv1_out = self.conv1(input[i], Adj_new)
            drop_out = F.dropout(self.F(conv1_out), training=self.training, p=self.dropout)
            x[i] = self.conv2(drop_out, Adj_new)

        return x