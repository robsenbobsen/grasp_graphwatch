import torch
from torch_scatter import scatter_max
from torch_sparse import SparseTensor
from torch_cluster import knn_graph
from torch_spline_conv import spline_conv
from torch_geometric.nn import GCNConv


def main():

    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())

    src = torch.tensor([2.0, 1.0, 5.0, 3.0])
    idx = torch.tensor([0, 0, 1, 1])
    print("scatter_max:", scatter_max(src, idx)[0].tolist())

    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    adj = SparseTensor(row=ei[0], col=ei[1], sparse_sizes=(3, 3))
    print("sparse matmul:", adj.matmul(torch.ones(3, 2)).sum().item())

    x = torch.randn(10, 3)
    print("knn_graph edges:", knn_graph(x, k=3).shape)
    pseudo = torch.rand(3, 1)                      # [num_edges, dim], values in [0, 1]
    kernel = torch.tensor([5])                     # LongTensor, one entry per pseudo dim
    is_open = torch.ones(1, dtype=torch.uint8)     # ByteTensor
    w = torch.rand(5, 3, 4)                        # [prod(kernel_size), in_ch, out_ch]
    print("spline_conv:", spline_conv(torch.randn(3, 3), ei, pseudo, w, kernel, is_open, 1).shape)
