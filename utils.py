import numpy as np
import torch
import re
import copy
import torch

import transformers


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, dim=1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx

def normalize_adj(adj):
    """D^(-1/2)AD^(-1/2)"""
    # D = torch.diag(torch.sum(adj > 1e-5, dim=1)).float()
    # D_2 = torch.pow(D, -0.5)
    # D_2[torch.isinf(D_2)] = 0.
    # ans = torch.mm(D_2, adj)
    # ans = torch.mm(ans, D_2)
    adj = adj / (adj.sum(dim = 1,keepdim=True)+1e-5)
    return adj

def normalize_features(mx):
    rowsum = mx.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx


def union_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)


def intersect_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)


def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)


def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (union_size(yhatmic, ymic, 0) + 1e-10)


def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (yhatmic.sum(axis=0) + 1e-10)


def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (ymic.sum(axis=0) + 1e-10)


def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic,
                                                                                                                ymic)


def all_metrics(y_hat, y):
    """
    :param y_hat:
    :param y:

    :return:
    """
    names = ['acc', 'prec', 'rec', 'f1']
    macro_metrics = all_macro(y_hat, y)

    y_mic = y.ravel()
    y_hat_mic = y_hat.ravel()
    micro_metrics = all_micro(y_hat_mic, y_mic)

    metrics = {names[i] + "_macro": macro_metrics[i] for i in range(len(macro_metrics))}
    metrics.update({names[i] + '_micro': micro_metrics[i] for i in range(len(micro_metrics))})

    return metrics



# 使用pytorch计算top5准确率的函数[^2^][2]
def topk_accuracy(logits, target, topk=(1,5,10)):
    indices = np.argsort(logits, axis=-1)
    batch_size,class_num = logits.shape
    ans = []
    for k in topk:
        predict = np.zeros((batch_size,class_num))
        for i in range(batch_size):
            predict[i,indices[i,-k:]] = 1
        ans.append(np.sum(predict*target) / (batch_size*k))
    return ans

def print_metrics(metrics_test):
    print("\n[MACRO] accuracy, precision, recall, f-measure")
    print("%.4f, %.4f, %.4f, %.4f" %
          (metrics_test["acc_macro"], metrics_test["prec_macro"], metrics_test["rec_macro"], metrics_test["f1_macro"]))

    print("[MICRO] accuracy, precision, recall, f-measure")
    print("%.4f, %.4f, %.4f, %.4f" %
          (metrics_test["acc_micro"], metrics_test["prec_micro"], metrics_test["rec_micro"], metrics_test["f1_micro"]))


def write_result(report, result_path):
    with open(result_path, "w", encoding="UTF-8")as f:
        f.write(report)

def get_age(raw_age):
    if '岁' in raw_age or '月' in raw_age or '日' in raw_age or '天' in raw_age:
        year = re.search(r'(\d*?)岁',raw_age)
        month = re.search(r'(\d*?)月',raw_age)
        day = re.search(r'(\d*?)日',raw_age)
        day2 = re.search(r'(\d*?)天',raw_age)

        ans = 0
        if year is None or year.group(1)=='': ans += 0
        else: ans += int(year.group(1))*365
        if month is None or month.group(1)=='': ans += 0
        else: ans += int(month.group(1))*30
        if day is None or day.group(1)=='': ans += 0
        else: ans += int(day.group(1))
        if day2 is None or day2.group(1)=='': ans += 0
        else: ans += int(day2.group(1))
        ans = ans // 365
    else:
        if 'Y' in raw_age:
            raw_age = raw_age.replace('Y','')
        try:
            ans = int(raw_age)
        except:
            ans = -1
    if ans < 0:
        return ''
    elif ans >= 0 and ans < 1:
        return '婴儿'
    elif ans >= 1 and ans <= 6:
        return '童年'
    elif ans >=7 and ans <= 18:
        return '少年'
    elif ans >= 19 and ans <= 30:
        return '青年'
    elif ans >= 31 and ans <= 40:
        return '壮年' 
    elif ans >= 41 and ans <= 55:
        return '中年'
    else:
        return '老年'

def format(entity):
    entity = entity.replace('+','\+').replace('*','\*').replace('.','\.')\
                   .replace('(','\(').replace(')','\)').replace('[','\[')\
                   .replace(']','\[')
    return entity

def remove_neg_entities(document, entities):
    entities = list(set(entities))
    for entity in entities:
        index = document.index(entity)
        if '无' in document[min(0,index-20):index] or '否认' in document[min(0,index-20):index]:
            entities.remove(entity)

        # if re.search(r'(无|(否认))(.{0,10}(、|及|，))*?.{0,5}'+format(entity),document) is not None:
        #     entities.remove(entity)
    return entities

import os
import re


from transformers import AutoModel,AutoTokenizer
import torch
import numpy as np

class GenerateEmbedding:
    def __init__(self,bert_path,cuda):
        self.cuda = cuda
        self.bert = AutoModel.from_pretrained(bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        if torch.cuda.is_available():
            self.bert = self.bert.cuda(cuda)
    def generate(self,entity):
        """
            生成实体嵌入向量
        """
        entity = '#' + entity
        tokens = self.tokenizer(entity,return_tensors = 'pt')
        with torch.no_grad():
            if torch.cuda.is_available():
                tokens['input_ids'] = tokens['input_ids'].cuda(self.cuda)
                tokens['attention_mask'] = tokens['attention_mask'].cuda(self.cuda)
            output = self.bert(tokens['input_ids'],tokens['attention_mask']).last_hidden_state[:,2:]
        return output.squeeze(0).mean(dim = 0).cpu().numpy()

    def similarity(self,vec1,vec2):
        """
            计算余弦相似度
        """
        return np.sum(vec1 * vec2) / (np.sqrt(np.sum(np.power(vec1,2))) + np.sqrt(np.sum(np.power(vec2,2))))


from sko.GA import GA
def best_threshold1(Y,Y_hat,prec=0.01):
    """
        通过验证集确定最佳阈值
        Y     : [batch_size, class_num] ∈ {0,1}
        Y_hat : [batch_size, class_num] ∈ [0,1]
        prec  : float ∈ [0,1] 精度
        return: [class_num] ∈ [0,1] 最佳阈值
    """
    def func(threshold):
        threshold = np.expand_dims(threshold, axis=0)
        y_hat = Y_hat.copy()
        y_hat[y_hat>threshold] = 1
        y_hat[y_hat<=threshold] = 0
        return -micro_f1(y_hat.ravel(),Y.ravel()) # + 0.2 * np.mean(np.abs(threshold-0.5)) # 加入正则

    ga = GA(func=func, n_dim=Y.shape[1], size_pop=1000, max_iter=500, prob_mut=0.01,
            lb=[0.35]*Y.shape[1], ub=[0.65]*Y.shape[1], precision=[prec]*Y.shape[1])

    best_x, best_y = ga.run()
    return best_x

def best_threshold(Y,Y_hat,prec=0.01):
    """
        通过验证集确定最佳阈值
        Y     : [batch_size, class_num] ∈ {0,1}
        Y_hat : [batch_size, class_num] ∈ [0,1]
        prec  : float ∈ [0,1] 精度
        return: [class_num] ∈ [0,1] 最佳阈值
    """
    def func(threshold):
        # threshold = threshold.repeat(Y.shape[1]).reshape(1,-1)
        y_hat = Y_hat.copy()
        y_hat[y_hat>threshold] = 1
        y_hat[y_hat<=threshold] = 0
        return -micro_f1(y_hat.ravel(),Y.ravel()) # + 0.2 * np.mean(np.abs(threshold-0.5)) # 加入正则

    ga = GA(func=func, n_dim=Y.shape[1], size_pop=100, max_iter=500, prob_mut=0.01,
            lb=[0.35]*Y.shape[1], ub=[0.65]*Y.shape[1], precision=[prec]*Y.shape[1])

    best_x, best_y = ga.run()
    return best_x
def edit_distance(str1, str2):
    """
    python 实现编辑距离
    """
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    return dp[m][n]

def model_optimizer(model,model_name,opt):
    param_optimizer = None
    if 'LongFormer' in model_name:
        bert_params = set(model.longformer_layer.word_embedding.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = [
            {'params': [p for n, p in model.longformer_layer.word_embedding.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': opt.bert_lr,
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.longformer_layer.word_embedding.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': 0.0,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': opt.other_lr,
             'weight_decay': 0}
        ]
    elif 'Ernie' in model_name:
        bert_params = set(model.ernie_layer.word_embedding.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = [
            {'params': [p for n, p in model.ernie_layer.word_embedding.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': opt.bert_lr,
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.ernie_layer.word_embedding.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': 0.0,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': opt.other_lr,
             'weight_decay': 0}
        ]
    elif 'Bert' in model_name:
        bert_params = set(model.bert_layer.word_embedding.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = [
            {'params': [p for n, p in model.bert_layer.word_embedding.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': opt.bert_lr,
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.bert_layer.word_embedding.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': 0.0,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': opt.other_lr,
             'weight_decay': 0}
        ]
    elif 'Auto' in model_name:
        bert_params = set(model.base_layer.word_embedding.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = [
            {'params': [p for n, p in model.base_layer.word_embedding.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': opt.bert_lr,
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.base_layer.word_embedding.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': 0.0,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': opt.other_lr,
             'weight_decay': 0}
        ]
    if param_optimizer is not None:
        optimizer = transformers.AdamW(param_optimizer, lr=opt.other_lr, weight_decay=0.0)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr = opt.other_lr)
    return optimizer


if __name__ == '__main__':
    logits = np.random.randn(2,50)

    y = np.zeros((2,50))
    ans = topk_accuracy(logits,y)
    print(ans)
