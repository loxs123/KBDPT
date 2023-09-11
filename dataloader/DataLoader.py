"""
   使用pytorch中的Dataloader 
"""
import random

import torch
import json
import torch.utils.data as data
import numpy as np
import pickle as pk

import os
from transformers import AutoTokenizer
import copy
import json

from utils import get_age,edit_distance

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'


class Dataset(data.Dataset):

    def __init__(self, filename, opt):
        super(Dataset, self).__init__()
        self.data_dir = filename  # data_path
        self.label_idx_path = opt.label_idx_path
        self.label_smooth_lambda = opt.label_smooth_lambda
        self.label2id = opt.label2id
        self.entity2id = opt.entity2id

        self.bert_path = opt.bert_path
        self.class_num = opt.class_num
        self.data = []
        self.ent_label_matrix = opt.ent_label_matrix # [ent_num,class_num]
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.max_length = opt.max_length
        # 大小为 [(实体数量+疾病数量) * (实体数量+疾病数量)]
        self._preprocess()

    def _min_distance_entity_id(self,entity):
        min_distrance = 1e3

        min_entity_id = -1
        for (e,i) in self.entity2id.items():
            cur_d = edit_distance(e,entity)
            if cur_d < min_distrance:
                min_distrance = cur_d
                min_entity_id = i
        return min_entity_id

    def _preprocess(self):
        print("Loading data file...")
        tokenizer = AutoTokenizer.from_pretrained(self.bert_path)

        with open(self.data_dir, 'r', encoding='UTF-8')as f:
            dicts = json.load(f)
        count = 0
        for dic in dicts:
            if '主诉' not in dic["主诉"]:
                dic['主诉'] = '主诉：'+dic['主诉']
            if '现病史' not in dic["现病史"]:
                dic['现病史'] = '现病史：'+dic['现病史']
            if '既往史' not in dic["既往史"]:
                dic['既往史'] = '既往史：'+dic['既往史']
            chief_complaint = '性别：' + dic['性别'] + '；年龄：'+get_age(dic['年龄']) + '；'+ dic["主诉"]
            now_history, past_history = dic["现病史"], dic["既往史"]
            doc = chief_complaint + '[SEP]' + now_history + '[SEP]' + past_history

            doc = tokenizer(doc)
            # item_entities = set(dic['疾病实体']) | set(dic['治疗实体']) | set(dic['检查实体']) | set(dic['症状实体']) | set(dic['检查结果实体'])
            entities = dic['实体']
            
            # 疾病个数+实体个数
            entities = np.concatenate((np.arange(self.class_num),\
                                       np.array([self.entity2id[entity]+self.class_num for entity in entities if entity in self.entity2id])),axis=0)
            #print(entities.sum())
            # 抽取子图
            if len(entities) != self.class_num:
                ent_label_matrix = self.ent_label_matrix[entities[self.class_num:]-self.class_num] # 抽取子图 [entity_num, class_num]
                adj_matrix = np.zeros((len(entities),len(entities)))
                # adj_matrix[self.class_num:,:self.class_num] = ent_label_matrix
                adj_matrix[:self.class_num,self.class_num:] = ent_label_matrix.T
            else:
                adj_matrix = np.zeros((len(entities),len(entities)))
                

            #print(adj_matrix.sum())
            # adj_matrix[np.arange(len(entities)),np.arange(len(entities))] = 1
            if isinstance(dic['出院诊断'],str):
                label = np.array([self.label_smooth_lambda if label not in dic['出院诊断'].split(';') else 1-self.label_smooth_lambda \
                                  for label in self.label2id])
            else:
                label = np.array([self.label_smooth_lambda if label not in dic['出院诊断'] else 1-self.label_smooth_lambda \
                                  for label in self.label2id])
            count += 1
            # break
            if count % 1000 == 0:
                print(count,'完成')


            self.data.append((doc['input_ids'],doc['attention_mask'],entities, adj_matrix, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc_idxes, doc_mask, entities, adj_matrix, label = self.data[idx]
        
        entities = torch.tensor(entities, dtype=torch.float32).unsqueeze(0) #[1,num_class+num_entities]
        adj_matrix=torch.tensor(adj_matrix,dtype=torch.float32) #[num_class+num_entities,\
                                                                            #num_class+num_entities]
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        if len(doc_idxes) > self.max_length-1:
            doc_idxes = doc_idxes[:self.max_length-1] + [self.tokenizer.sep_token_id]
            doc_mask = doc_mask[:self.max_length-1] + [1]

        return doc_idxes, doc_mask, entities, adj_matrix, label


def collate_fn(X):
    X = list(zip(*X))
    doc_idxes, doc_mask, entities, adj_matrixs, labels = X

    # 最长pad
    idxs = [doc_idxes]
    masks = [doc_mask]
    for j,(idx,mask) in enumerate(zip(idxs,masks)):
        max_len = max([len(t) for t in idx])
        for i in range(len(idx)):
            idx[i].extend([0 for _ in range(max_len - len(idx[i]))])  # pad
            mask[i].extend([0 for _ in range(max_len - len(mask[i]))])
        idxs[j] = torch.tensor(idx,dtype = torch.long)
        masks[j] = torch.tensor(mask,dtype = torch.long)

    labels = torch.cat(labels, 0)

    return (idxs[0],),(masks[0],), entities,adj_matrixs, labels

def data_loader(data_file, opt, shuffle, num_workers=0):
    dataset = Dataset(data_file, opt)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)


    return loader