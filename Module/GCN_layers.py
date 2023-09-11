import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False, use_sparse_matrix=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.use_sparse_matrix = use_sparse_matrix
        self.adj_cache = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input @ self.weight
        if self.use_sparse_matrix:
            # [batch_size, num_node, hidden_size]
            batch_size,num_node,hidden_size=support.size()
            if self.adj_cache is None:
                self.adj_cache = adj.to_sparse()
            support = support.transpose(0,1).reshape(num_node,-1)
            output = torch.spmm(self.adj_cache,support)
            output = output.reshape(num_node,batch_size,hidden_size)
            output = output.transpose(0,1)
        else:
            output = adj @ support
        if self.bias is not None:
            return output + self.bias
        else:
            return output


