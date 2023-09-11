# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(30552, 768, padding_idx=0)

        self.lstm = nn.LSTM(768, 512, 1,bidirectional=True, batch_first=True, dropout=opt.dropout_rate)
        self.fc = nn.Linear(1024+768 ,opt.class_num)

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out,_ = torch.max(out,dim=2)
        out = self.fc(out)
        return out
