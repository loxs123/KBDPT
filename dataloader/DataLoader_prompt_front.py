"""
   [CLS] prompt [MASK] doc [SEP]
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


class DatasetTrain(data.Dataset):

    def __init__(self, filename, opt):
        super(DatasetTrain, self).__init__()
        self.data_dir = filename  # data_path
        self.label_idx_path = opt.label_idx_path
        self.label_smooth_lambda = opt.label_smooth_lambda
        self.label2id = opt.label2id
        self.entity2id = opt.entity2id

        self.bert_path = opt.bert_path
        self.class_num = opt.class_num
        self.data = []
        self.ent_label_matrix = opt.ent_label_matrix # [ent_num,class_num]
        # 大小为 [(实体数量+疾病数量) * (实体数量+疾病数量)]
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.max_length = opt.max_length
        self.sample_radio = opt.sample_radio


        with open(self.data_dir, 'r', encoding='UTF-8')as f:
            self.dicts = json.load(f)
        with open(opt.path_path,'r',encoding='utf-8')as f:
            self.id2path = json.load(f)

        self.data = []
        self._preprocess()

    def _preprocess(self):
        """
        tokenize
        """
        print("Loading data file...")
        count = 0
        for dic in self.dicts:
            item = {}
            chief_complaint = '性别：' + dic['性别'] + '；年龄：'+get_age(dic['年龄']) + '；'+ dic["主诉"]
            now_history, past_history = dic["现病史"], dic["既往史"]
            doc =  '[CLS]' + chief_complaint + '[SEP]' + now_history + '[SEP]' + past_history + '[SEP]'
            item['doc'] = self.tokenizer(doc,add_special_tokens = False)
            
            item['nodes'] = np.concatenate((np.arange(self.class_num),\
                            np.array([self.entity2id[entity]+self.class_num for entity in dic['实体'] if entity in self.entity2id])),axis=0)
            if len(item['nodes'] ) != self.class_num:

                ent_label_matrix = self.ent_label_matrix[item['nodes'][self.class_num:]-self.class_num]
                item['adj_matrix'] = np.zeros((len(item['nodes']),len(item['nodes'])))
                item['adj_matrix'][:self.class_num,self.class_num:] = ent_label_matrix.T
            else:
                item['adj_matrix'] = np.zeros((len(item['nodes']),len(item['nodes'])))
            
            item['pos_path'] = dic['path_T']
            item['neg_path'] = dic['path_F']

            if isinstance(dic['出院诊断'],str):
                item['label'] = np.array([self.label_smooth_lambda if label not in dic['出院诊断'].split(';') else 1-self.label_smooth_lambda \
                                  for label in self.label2id])
            else:
                item['label'] = np.array([self.label_smooth_lambda if label not in dic['出院诊断'] else 1-self.label_smooth_lambda \
                                  for label in self.label2id])
            self.data.append(item)
            
            count += 1
            # break
            if count % 1000 == 0:
                print(count,'完成')
        del self.dicts


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_id = idx
        item = self.data[item_id]

        pos_len = len(item['pos_path'])
        neg_len = len(item['neg_path'])
        
        prompt_idx = random.randint(0,pos_len+neg_len-1)
        if prompt_idx < pos_len:
            prompt = self.id2path[item['pos_path'][prompt_idx]].replace('\t','')
            positive = 1
        else:
            prompt = self.id2path[item['neg_path'][prompt_idx-pos_len]].replace('\t','')
            positive = 0

        prompt = self.tokenizer(''.join(prompt),add_special_tokens = False)

        if len(item['doc']['input_ids']) + 1 + len(prompt['input_ids']) + 1 > self.max_length:
            item['doc']['input_ids'] = item['doc']['input_ids'][:self.max_length - 1 - len(prompt['input_ids']) - 1] + [self.tokenizer.sep_token_id]
            item['doc']['attention_mask'] = item['doc']['attention_mask'][:self.max_length - 1 - len(prompt['attention_mask']) - 1] + [1]

        mask_idx = len(prompt['input_ids']) + 1
        doc_idx = [self.tokenizer.cls_token_id] +prompt['input_ids'] + [self.tokenizer.mask_token_id] + item['doc']['input_ids'][1:] +  [self.tokenizer.sep_token_id]
        doc_mask = [1] + prompt['attention_mask'] + [1] + item['doc']['attention_mask'][1:] + [1]

        nodes = torch.tensor(item['nodes']).unsqueeze(0) #[1,num_class+num_entities]
        adj_matrix = torch.tensor(item['adj_matrix'],dtype=torch.float32)
        mask_idx = np.array([mask_idx])

        label = torch.tensor(item['label'], dtype=torch.float32).unsqueeze(0)
        label_prompt = np.zeros((2))
        label_prompt[positive] = 1
        label_prompt = torch.tensor(label_prompt).unsqueeze(0)

        return doc_idx,doc_mask,nodes, adj_matrix,mask_idx, label, label_prompt

class DatasetTest(data.Dataset):

    def __init__(self, filename, opt):
        super(DatasetTest, self).__init__()
        self.data_dir = filename  # data_path
        self.label_idx_path = opt.label_idx_path
        self.label_smooth_lambda = opt.label_smooth_lambda
        self.label2id = opt.label2id
        self.entity2id = opt.entity2id

        self.bert_path = opt.bert_path
        self.class_num = opt.class_num
        self.data = []
        self.ent_label_matrix = opt.ent_label_matrix # [ent_num,class_num]
        # 大小为 [(实体数量+疾病数量) * (实体数量+疾病数量)]
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.max_length = opt.max_length
        with open(self.data_dir, 'r', encoding='UTF-8')as f:
            self.dicts = json.load(f)
        with open(opt.path_path,'r',encoding='utf-8')as f:
            self.id2path = json.load(f)
        self.data = []
        self._preprocess()

    def _preprocess(self):
        """
        tokenize
        """
        print("Loading data file...")
        count = 0
        for dic in self.dicts:
            item = {}
            chief_complaint = '性别：' + dic['性别'] + '；年龄：'+get_age(dic['年龄']) + '；'+ dic["主诉"]
            now_history, past_history = dic["现病史"], dic["既往史"]
            doc =  '[CLS]' + chief_complaint + '[SEP]' + now_history + '[SEP]' + past_history + '[SEP]'
            item['doc'] = self.tokenizer(doc,add_special_tokens = False)
            
            item['nodes'] = np.concatenate((np.arange(self.class_num),\
                            np.array([self.entity2id[entity]+self.class_num for entity in dic['实体'] if entity in self.entity2id])),axis=0)
            if len(item['nodes']) != self.class_num:
                ent_label_matrix = self.ent_label_matrix[item['nodes'][self.class_num:]-self.class_num]
                item['adj_matrix'] = np.zeros((len(item['nodes']),len(item['nodes'])))
                item['adj_matrix'][:self.class_num,self.class_num:] = ent_label_matrix.T
            else:
                item['adj_matrix'] = np.zeros((len(item['nodes']),len(item['nodes'])))
            
            label2path_t = {}
            for path in dic['path_T']:
                label = self.id2path[path].split('\t')[-1]
                if label not in label2path_t:
                    label2path_t[label] = []
                label2path_t[label].append(path)

            label2path_f = {}
            for path in dic['path_F']:
                label = self.id2path[path].split('\t')[-1]
                if label not in label2path_f:
                    label2path_f[label] = []
                label2path_f[label].append(path)

            item['pos_path'] = label2path_t
            item['neg_path'] = label2path_f

            if isinstance(dic['出院诊断'],str):
                item['label'] = np.array([self.label_smooth_lambda if label not in dic['出院诊断'].split(';') else 1-self.label_smooth_lambda \
                                  for label in self.label2id])
            else:
                item['label'] = np.array([self.label_smooth_lambda if label not in dic['出院诊断'] else 1-self.label_smooth_lambda \
                                  for label in self.label2id])

            self.data.append(item)
            
            count += 1
            # break
            if count % 1000 == 0:
                print(count,'完成')

        del self.dicts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_id = idx
        item = self.data[item_id]
        
        all_doc_idx = []
        all_doc_mask = []
        all_nodes = []
        all_adj_matrix = []
        all_mask_idx = []
        all_label = []
        all_label_prompt = []

        path_labels = list(item['pos_path'].keys()) + list(item['neg_path'].keys())
        
        all_path = []
        for label in path_labels:
            if label in item['pos_path']:
                for path_id in item['pos_path'][label]:
                    all_path.append(path_id)
            else:
                for path_id in item['neg_path'][label]:
                    all_path.append(path_id)

        for path_id in all_path:
            prompt =self.id2path[path_id].replace('\t','')
            positive = 0
            
            prompt = self.tokenizer(''.join(prompt),add_special_tokens = False)
            raw_doc = item['doc']['input_ids']
            raw_mask = item['doc']['attention_mask']

            if len(raw_doc) + 1 + len(prompt['input_ids']) + 1 > self.max_length:
                raw_doc = raw_doc[:self.max_length - 1 - len(prompt['input_ids']) - 1] + [self.tokenizer.sep_token_id]
                raw_mask = raw_mask[:self.max_length - 1 - len(prompt['attention_mask']) - 1] + [1]

            mask_idx = len(prompt['input_ids']) + 1
            mask_idx = np.array([mask_idx])


            doc_idx = [self.tokenizer.cls_token_id] +prompt['input_ids'] + [self.tokenizer.mask_token_id] + raw_doc[1:] +  [self.tokenizer.sep_token_id]
            doc_mask = [1] + prompt['attention_mask'] + [1] + raw_mask[1:] + [1]

            nodes = torch.tensor(item['nodes']).unsqueeze(0) #[1,num_class+num_entities]
            adj_matrix = torch.tensor(item['adj_matrix'],dtype=torch.float32)
            

            label = torch.tensor(item['label'], dtype=torch.float32).unsqueeze(0)
            label_prompt = np.zeros((2))
            label_prompt[positive] = 1
            label_prompt = torch.tensor(label_prompt).unsqueeze(0)

            all_doc_idx.append(doc_idx)
            all_doc_mask.append(doc_mask)
            all_nodes.append(nodes)
            all_adj_matrix.append(adj_matrix)
            all_mask_idx.append(mask_idx)
            all_label.append(label)
            all_label_prompt.append(label_prompt)

        return all_doc_idx,all_doc_mask,all_nodes, all_adj_matrix,all_mask_idx, all_label, all_label_prompt


# 去掉没有的
def collate_train_fn(X):
    X = list(zip(*X))
    doc_idx, doc_mask, nodes, adj_matrix,mask_idx, label, label_prompt = X
    #doc_idx, doc_mask, nodes, adj_matrix,mask_idx, label, label_prompt = \
    #doc_idx[0],doc_mask[0], nodes[0],adj_matrix[0],mask_idx[0],label[0],label_prompt[0]

    # 最长pad
    idxs = [doc_idx]
    masks = [doc_mask]
    for j,(idx,mask) in enumerate(zip(idxs,masks)):
        max_len = max([len(t) for t in idx])
        for i in range(len(idx)):
            idx[i].extend([0 for _ in range(max_len - len(idx[i]))])  # pad
            mask[i].extend([0 for _ in range(max_len - len(mask[i]))])
        idxs[j] = torch.tensor(idx,dtype = torch.long)
        masks[j] = torch.tensor(mask,dtype = torch.long)
    
    mask_idx = np.concatenate(mask_idx, axis = 0)
    label = torch.cat(label, 0)
    label_prompt = torch.cat(label_prompt, 0)

    return (idxs[0],),(masks[0],),nodes,adj_matrix,mask_idx, label,label_prompt


def collate_test_fn(X):
    X = list(zip(*X))
    doc_idx, doc_mask, nodes, adj_matrix,mask_idx, label, label_prompt = X
    #print(X)
    doc_idx, doc_mask, nodes, adj_matrix,mask_idx, label, label_prompt = \
    doc_idx[0],doc_mask[0], nodes[0],adj_matrix[0],mask_idx[0],label[0],label_prompt[0]
    # 最长pad
    idxs = [doc_idx]
    masks = [doc_mask]
    for j,(idx,mask) in enumerate(zip(idxs,masks)):
        max_len = max([len(t) for t in idx])
        for i in range(len(idx)):
            idx[i].extend([0 for _ in range(max_len - len(idx[i]))])  # pad
            mask[i].extend([0 for _ in range(max_len - len(mask[i]))])
        idxs[j] = torch.tensor(idx,dtype = torch.long)
        masks[j] = torch.tensor(mask,dtype = torch.long)
    
    mask_idx = np.concatenate(mask_idx, axis = 0)
    label = torch.cat(label, 0)
    label_prompt = torch.cat(label_prompt, 0)

    return (idxs[0],),(masks[0],),nodes,adj_matrix,mask_idx, label,label_prompt

def data_loader(data_file, opt, shuffle, num_workers=0):
    if 'train' in data_file:
        dataset = DatasetTrain(data_file, opt)
        loader = data.DataLoader(dataset=dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=shuffle,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_train_fn)
    else:
        dataset = DatasetTest(data_file, opt)
        loader = data.DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=shuffle,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_test_fn)

    return loader

