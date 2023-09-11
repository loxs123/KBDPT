# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.model_name = 'CAML'
        self.embedding = nn.Embedding(30522, 768, padding_idx=0)
        self.embed_drop = nn.Dropout(opt.dropout_rate)
        self.conv = nn.Conv1d(768,768,kernel_size=3,padding = 1)
        
        self.U = nn.Linear(768, opt.class_num)
        self.final = nn.Linear(768, opt.class_num)


    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2) # [batch_size, embed_size ,seq_len]

        #apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1,2)) # [batch_size, seq_len, hidden_size]
        #apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2) # [batch_size,num_class,seq_len]
        #document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x) # [batch_size, num_class, hidden_size]
        #final layer classification
        y_hat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return y_hat
