import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_prob = dropout_prob
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.leaky_relu = nn.LeakyReLU()
        
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
    def forward(self, x, adj):
        h = torch.matmul(x, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)

# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, dropout_prob=0.1, nheads=4):
#         super(GAT, self).__init__()
#         self.dropout_prob = dropout_prob

#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout_prob) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#         self.out_att = GraphAttentionLayer(nhid*nheads, nhid*nheads, dropout_prob)
#         self.dropout = nn.Dropout(p=self.dropout_prob)
#         self.leaky_relu = nn.LeakyReLU()
        
#     def forward(self, x, adj):
#         x = self.dropout(x)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = self.dropout(x)
#         x = F.elu(self.out_att(x, adj))
#         return x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout_prob=0.1, nheads=4):
        super(GAT, self).__init__()
        self.attn_layer = GraphAttentionLayer(nhid, nhid, dropout_prob)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, x, adj):
        x = F.elu(self.attn_layer(x, adj))
        return x


if __name__ == '__main__':
    gat = GAT(100,100)
    feats = torch.zeros(34,100)
    adj = torch.randn(34,34)
    adj[adj>0.5] = 1
    adj[adj<=0.5] = 0
    out = gat(feats,adj)
    print(out.size())