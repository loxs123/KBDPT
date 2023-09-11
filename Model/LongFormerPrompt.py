from abc import ABC

import torch
import torch.nn as nn

from transformers import AutoModel
from fengshen import LongformerModel

import numpy as np


class LongFormerPrompt(nn.Module, ABC):
    def __init__(self, opt):
        super(LongFormerPrompt, self).__init__()
        self.model_name = 'LongformerModel'
        self.word_embedding = LongformerModel.from_pretrained(opt.bert_path)
        for param in self.word_embedding.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, X, masks, prompt_mask_idx):
        batch_size,seq_len = masks.size()
        global_attention_mask = np.zeros((batch_size,seq_len))
        global_attention_mask[:,0] = 1
        global_attention_mask[np.arange(batch_size),prompt_mask_idx] = 1 # 
        global_attention_mask = torch.tensor(global_attention_mask,device=X.device)
        embed = self.word_embedding(X, attention_mask=masks,global_attention_mask=global_attention_mask).last_hidden_state
        embed = self.dropout(embed)
        # [batch_size, seq_len,hidden_size]
        pooled1 = embed[:,0]# [batch_size ,hidden_size]
        pooled2 = embed[np.arange(batch_size),prompt_mask_idx]# [batch_size ,hidden_size]
        return pooled1,pooled2

class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()

        self.class_num = opt.class_num
        self.longformer_layer = LongFormerPrompt(opt)

        self.cls_layer1 = nn.Linear(opt.embedding_dim, 1024)
        nn.init.xavier_normal_(self.cls_layer1.weight)
        self.cls_layer2 = nn.Linear(1024, opt.class_num)
        nn.init.xavier_normal_(self.cls_layer2.weight)

        self.prompt_cls_layer1 = nn.Linear(opt.embedding_dim, 256)
        nn.init.xavier_normal_(self.cls_layer1.weight)
        self.prompt_cls_layer2 = nn.Linear(256, 2)
        nn.init.xavier_normal_(self.cls_layer2.weight)

    def forward(self,sentence,mask,prompt_mask_idx):
        longformer_vec,longformer_prompt_vec = self.longformer_layer(sentence,mask,prompt_mask_idx) # [batch_size, hidden_size]
        hidden_vec = torch.relu(self.cls_layer1(longformer_vec))
        y_hat = self.cls_layer2(hidden_vec)

        hidden_vec = torch.relu(self.prompt_cls_layer1(longformer_prompt_vec))
        y_hat_prompt = self.prompt_cls_layer2(hidden_vec)
        return y_hat,y_hat_prompt
        

