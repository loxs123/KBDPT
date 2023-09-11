# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
import json
from math import floor
import numpy as np

class WordRep(nn.Module):
    def __init__(self, opt):
        super(WordRep, self).__init__()

        self.embedding = nn.Embedding(30522, 768, padding_idx=0)
        self.feature_size = self.embedding.embedding_dim

        self.embed_drop = nn.Dropout(p=opt.dropout_rate)

        self.conv_dict = {1: [self.feature_size, 256],
                     2: [self.feature_size, 100, 256],
                     3: [self.feature_size, 150, 100, 256],
                     4: [self.feature_size, 200, 150, 100, 256]
                     }


    def forward(self, x):

        features = [self.embedding(x)]

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        return x

class OutputLayer(nn.Module):
    def __init__(self, opt,input_size):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, opt.class_num)
        xavier_uniform(self.U.weight)


        self.final = nn.Linear(input_size, opt.class_num)
        xavier_uniform(self.final.weight)

    def forward(self, x):

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=1)
        # origin code dim = 2 but 1 is better for our dataset
        # [batch_size,label_num,seq_len]

        m = alpha.matmul(x)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        return y


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        self.word_rep = WordRep(opt)

        self.conv_layer=2

        self.conv = nn.ModuleList()
        filter_sizes = [3,5,7]

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[self.conv_layer]
            for idx in range(self.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    opt.dropout_rate)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer(opt, self.filter_num * 256)


    def forward(self, x):

        x = self.word_rep(x)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        y = self.output_layer(x)
        return y
        