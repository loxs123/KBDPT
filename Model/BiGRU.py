# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(30522, 768, padding_idx=0)
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.gru = nn.GRU(768, 512, 1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024,opt.class_num)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out