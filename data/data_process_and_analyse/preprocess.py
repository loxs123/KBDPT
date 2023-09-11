# -*- coding:utf-8 -*- 
import json
import random
import os
import numpy as np
import copy
# from utils import GenerateEmbedding
from transformers import AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np
import csv


data_path = r"data/chinese"
train_path=os.path.join(data_path,'train.json')
label_path=os.path.join(data_path,'label2id.txt')
label_key = '出院诊断'
ent_key = '实体'
bert_path='../recommend-huawei/pretrain_language_models/Longformer/'
cuda=0

"""
    预处理过程一：构造药物-实体相关矩阵
"""

entity2id = {}
label2id = {}
entity2num = {}
label2num = {}
entity_label_arr = []
label_label_arr = np.zeros((len(label2id),len(label2id)))
with open(label_path,'r',encoding='utf-8') as f:
    labels = f.read().split('\n')
    labels = [label.split(' ')[0] for label in labels]
zeros = [0 for _ in range(len(labels))]
for label in labels:
    label2num[label] = 0
    label2id[label] = len(label2id)

label_label_arr = np.zeros((len(label2id),len(label2id)))


with open(train_path,'r',encoding='utf-8') as f:
    train = json.load(f)

    for item in train:
        entities = item[ent_key]
        for entity in entities:
            if entity not in entity2num:
                entity2num[entity] = 0
            entity2num[entity] += 1
        for label in set(item[label_key]):
            if label not in label2num:
                continue
            label2num[label] += 1
            for label2 in set(item[label_key]):
                if label2 not in label2num:
                    continue
                label_label_arr[label2id[label], label2id[label2]] += 1

    for item in train:
        entities = item[ent_key]
        for entity in entities:
            if entity not in entity2id:
                entity2id[entity] = len(entity2id)
                entity_label_arr.append(copy.copy(zeros))

            for label in set(item[label_key]):
                if label not in labels:
                    continue
                entity_label_arr[entity2id[entity]][labels.index(label)] += 1

entity_label_arr = np.array(entity_label_arr,dtype=np.float32) # entity_num * label_num
label_label_arr = np.array(label_label_arr,dtype=np.float32) # entity_num * label_num

for entity,entity_id in entity2id.items():
    for label,label_id in label2id.items():
        if entity_label_arr[entity_id,label_id] > 0:
            entity_label_arr[entity_id,label_id] = 1.0*entity_label_arr[entity_id,label_id]\
                                                   / (entity2num[entity]+label2num[label])

for label1,label_id1 in label2id.items():
    for label2,label_id2 in label2id.items():
        label_label_arr[label_id1,label_id2] = 1.0 * label_label_arr[label_id1,label_id2] \
                                                    / (label2num[label1] + label2num[label2])

print()


max_value = entity_label_arr.max()
entity_label_arr1 = entity_label_arr.copy()
entity_label_arr1[entity_label_arr<max_value/10] = 0
entity_label_arr1[entity_label_arr>=max_value/10] = 1
np.save(os.path.join(data_path,'ent_label_matrix1.npy'),entity_label_arr1) # [ent,med]

entity_label_arr2 = entity_label_arr.copy()
entity_label_arr2[entity_label_arr<max_value/100] = 0
entity_label_arr2[entity_label_arr>=max_value/100] = 1
np.save(os.path.join(data_path,'ent_label_matrix2.npy'),entity_label_arr2) # [ent,med]

entity_label_arr3 = entity_label_arr.copy()
entity_label_arr3[entity_label_arr<max_value/1000] = 0
entity_label_arr3[entity_label_arr>=max_value/1000] = 1
np.save(os.path.join(data_path,'ent_label_matrix3.npy'),entity_label_arr3) # [ent,med]

entity_label_arr4 = entity_label_arr.copy()
entity_label_arr4[entity_label_arr<max_value/10000] = 0
entity_label_arr4[entity_label_arr>=max_value/10000] = 1
np.save(os.path.join(data_path,'ent_label_matrix4.npy'),entity_label_arr4) # [ent,med]

entity_label_arr5 = entity_label_arr.copy()
entity_label_arr5[entity_label_arr<max_value/100000] = 0
entity_label_arr5[entity_label_arr>=max_value/100000] = 1
np.save(os.path.join(data_path,'ent_label_matrix5.npy'),entity_label_arr5) # [ent,med]

with open(os.path.join(data_path,'entity2id.txt'),'w',encoding='utf-8') as f:
    f.write('\n'.join([entity for entity in entity2id]))

"""
    预处理过程二：构造实体-字符矩阵
"""
# bert_path = '../longformer'
tokenizer = AutoTokenizer.from_pretrained(bert_path)
entities = []
for label in labels:
    entity_ids = tokenizer(label,add_special_tokens=False)['input_ids']
    if len(entity_ids) < 6:
        entity_ids.extend([0 for _ in range(6-len(entity_ids))])
    else:
        entity_ids = entity_ids[:6]
    entities.append(entity_ids)
for entity in entity2id:
    entity_ids = tokenizer(entity,add_special_tokens=False)['input_ids']
    if len(entity_ids) < 6:
        entity_ids.extend([0 for _ in range(6-len(entity_ids))])
    else:
        entity_ids = entity_ids[:6]
    entities.append(entity_ids)
entities = np.array(entities)
np.save(os.path.join(data_path,'ent_tokens_matrix.npy'),entities) # [entity_num,6]
