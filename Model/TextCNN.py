from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class TextCNN(nn.Module):
    def __init__(self,opt):
        super(TextCNN, self).__init__()
        self.model_name = 'TextCNN'
        self.word_embedding = nn.Embedding(30522,768) # 和bert相同
        self.embedding_dim = opt.embedding_dim
        self.class_num = opt.class_num
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.filter_sizes = (3, 4, 5)
        self.num_filters = 256
        self.convs = nn.ModuleList(
                [nn.Conv1d(768, self.num_filters, k) for k in self.filter_sizes])
        self.classifier = nn.Linear(self.embedding_dim, self.class_num)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def maxpool(self,X):
        max_x,_ = X.max(dim = 2) # [batch_size, 256]
        return max_x

    def forward(self, sentence):
        embed = self.word_embedding(sentence).transpose(1,2)

        cnn_out = torch.cat([self.maxpool(conv(embed)) for conv in self.convs], 1)

        return cnn_out # [batch_size, hidden_size]

class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()

        self.class_num = opt.class_num
        self.textcnn_layer = TextCNN(opt)

        self.cls_layer1 = nn.Linear(opt.embedding_dim, 1024)
        nn.init.xavier_normal_(self.cls_layer1.weight)
        self.cls_layer2 = nn.Linear(1024, opt.class_num)
        nn.init.xavier_normal_(self.cls_layer2.weight)

    def forward(self,sentence):
        textcnn_vec = self.textcnn_layer(sentence) # [batch_size, hidden_size]
        hidden_vec = torch.relu(self.cls_layer1(textcnn_vec))
        y_hat = self.cls_layer2(hidden_vec)
        return y_hat
        

