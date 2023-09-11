# -*- coding: utf-8 -*-
from importlib import import_module

import torch
import torch.nn as nn
import os
import argparse

from transformers import AutoTokenizer
import transformers
import warnings
import numpy as np
from sklearn import metrics
import random
import time
from tqdm import tqdm
import json
import datetime
import copy

from config import DefaultConfig

import utils
from utils import all_metrics, print_metrics, write_result,topk_accuracy,model_optimizer
import re

import wandb

def softmax(x,axis):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x,axis=axis,keepdims=True)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
warnings.filterwarnings('ignore')

best_threshold = None

MODEL_TYPE1 = [
    'ExperientialKnowledgeModel',
    'ExperientialKnowledgeGATModel',
    'ExperientialKnowledgeLastLayerAttnModel',
    'ExperientialKnowledgeGATLastLayerAttnModel'
]

MODEL_TYPE2 = [
    'CAML',
    'MultiResCNN',
    'BiGRU',
    'DPCNN',
    'RCNN',
    'RNNAttn',
    'TextCNN'
]

MODEL_TYPE3 = [
    'TextCNNFuseModel',
    'TextCNNFuseGATModel',
    'TextCNNFuseLastLayerAttnModel',
    'TextCNNFuseGATLastLayerAttnModel'
]

MODEL_TYPE4 = [
    'LongFormer',
    'AutoModel'
]

MODEL_TYPE5 = [
    'LongFormerFuseModel',
    'LongFormerFusePreTrainAsyModel',
    'LongFormerFusePreTrainSynModel',
    'LongFormerFuseGATModel'
]

MODEL_TYPE6 = [
    'LongFormerPrompt',
    'ErniePrompt',
    'BertPrompt',
    'ErniePrompt2',
    'BertPrompt2'
]

MODEL_TYPE7 = [
    'LongFormerPromptBase',
    'AutoPromptBase',
    'AutoPTuning',
    'LongFormerPTuning'
]
MODEL_TYPE8 = [
    'BertPromptField',
    'ErniePromptField'
]
MODEL_TYPE9 = [
    'AutoModelField'
]


def train(opt, train_data_loader, dev_data_loader,test_data_loader):
    global adv_model, K
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    model = import_module('Model.' + opt.model_name).Model(opt)
    if torch.cuda.is_available():
        model = model.cuda(opt.gpu)
    train_num = len(train_data_loader) * opt.batch_size

    # 损失函数
    loss_func = nn.BCEWithLogitsLoss()
    loss_func2 = nn.BCEWithLogitsLoss()

    optimizer = model_optimizer(model,opt.model_name,opt)

    updates_total = len(train_data_loader) // (opt.accumulation_steps) * opt.epochs
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=opt.warmup_rate * updates_total,
                                                             num_training_steps=updates_total)

    max_micro_f1 = -1.0  # the best micro F1
    max_scores = [-1 for _ in range(opt.class_num)]
    no_imp_valid = 0  # patience no improvement
    for epoch in range(opt.epochs):
        print("\n=== Epoch %d train ===" % epoch)
        for i, data in enumerate(tqdm(train_data_loader)):
            model.train()
            if 'Prompt' not in opt.model_name and 'PTuning' not in opt.model_name and 'yangyang' not in opt.data_loader:
                (sentence,), (mask,), nodes, adj_matrixs, labels = data
                if torch.cuda.is_available():
                    sentence = sentence.cuda(opt.gpu)
                    mask = mask.cuda(opt.gpu)
                    labels =  labels.cuda(opt.gpu)
                    for i in range(len(nodes)):
                        nodes[i] = nodes[i].cuda(opt.gpu)
                        adj_matrixs[i] = adj_matrixs[i].cuda(opt.gpu)
            elif 'yangyang' in opt.data_loader:
                (sentence1,sentence2,sentence3), (mask1,mask2,mask3),(mask_idx1,mask_idx2,mask_idx3),labels,label_prompts = data
                if torch.cuda.is_available():
                    sentence1 = sentence1.cuda(opt.gpu)
                    sentence2 = sentence2.cuda(opt.gpu)
                    sentence3 = sentence3.cuda(opt.gpu)

                    mask1 = mask1.cuda(opt.gpu)
                    mask2 = mask2.cuda(opt.gpu)
                    mask3 = mask3.cuda(opt.gpu)

                    labels =  labels.cuda(opt.gpu)
                    label_prompts = label_prompts.cuda(opt.gpu)

            else:
                (sentence,), (mask,),nodes,adj_matrixs,mask_idx, labels,label_prompts = data
                if torch.cuda.is_available():
                    sentence = sentence.cuda(opt.gpu)
                    mask = mask.cuda(opt.gpu)
                    labels =  labels.cuda(opt.gpu)
                    label_prompts = label_prompts.cuda(opt.gpu)
                    for i in range(len(nodes)):
                        nodes[i] = nodes[i].cuda(opt.gpu)
                        adj_matrixs[i] = adj_matrixs[i].cuda(opt.gpu)

            if opt.model_name in MODEL_TYPE1 :
                output = model(nodes,adj_matrixs)
                loss = (loss_func(output, labels)) / opt.accumulation_steps
            elif opt.model_name in MODEL_TYPE2:
                output = model(sentence)
                loss = (loss_func(output, labels)) / opt.accumulation_steps
            elif opt.model_name in MODEL_TYPE3:
                output = model(sentence,nodes,adj_matrixs)
                loss = (loss_func(output, labels)) / opt.accumulation_steps
            elif opt.model_name in MODEL_TYPE4:
                output = model(sentence,mask)
                loss = (loss_func(output, labels)) / opt.accumulation_steps
            elif opt.model_name in MODEL_TYPE5:
                output = model(sentence,mask,nodes,adj_matrixs)
                loss = (loss_func(output, labels)) / opt.accumulation_steps
            elif opt.model_name in MODEL_TYPE6:
                output,output_prompt = model(sentence,mask,mask_idx)
                loss = (loss_func(output, labels) + loss_func2(output_prompt, label_prompts)) / opt.accumulation_steps
            elif opt.model_name in MODEL_TYPE7:
                _,output_prompt = model(sentence,mask,mask_idx)
                loss = (loss_func2(output_prompt, labels)) / opt.accumulation_steps
            elif opt.model_name in MODEL_TYPE8:
                output,output_prompt = model(sentence1,mask1,mask_idx1,sentence2,mask2,mask_idx2,sentence3,mask3,mask_idx3)
                loss = (loss_func(output, labels) + loss_func2(output_prompt, label_prompts)) / opt.accumulation_steps
            elif opt.model_name in MODEL_TYPE9:
                output = model(sentence1,mask1,sentence2,mask2,sentence3,mask3)
                loss = (loss_func(output, labels)) / opt.accumulation_steps
            else:
                raise Exception
            loss.backward()
            if opt.use_wandb:
                # log metrics to wandb
                wandb.log({"loss": loss})

            if (i+1) % opt.accumulation_steps == 0: 
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # if (i+1) % 600 == 0:
            #     valid_macro_f1, report,scores = inference(model, dev_data_loader,opt,k_fold,batch_no = i//opt.accumulation_steps + updates_total//opt.epochs*epoch)
            #     if valid_macro_f1 > max_macro_f1:
            #         max_macro_f1 = valid_macro_f1
            #         torch.save(model.state_dict(),opt.save_model_path)
            #         print("目前最优验证集结果:{:.5f}".format(max_macro_f1))

        print(f'epochs {epoch} end')
        if (epoch + 1) % opt.test_freq == 0:
            valid_micro_f1, valid_report = inference(model, dev_data_loader,opt)
            test_micro_f1, test_report,_ = inference(model, test_data_loader,opt,test_set=True)
    
    
            print("\n验证集micro f1: {:.5f}".format(valid_micro_f1))
            print("\n测试集micro f1: {:.5f}".format(test_micro_f1))
            if opt.use_wandb:
                wandb.log({"dev-micro-f1":valid_micro_f1})
                wandb.log({"test-micro-f1":test_micro_f1})
            if valid_micro_f1 > max_micro_f1:
                max_micro_f1 = valid_micro_f1
    
                torch.save({'dict':model.state_dict()},opt.save_model_path)
    
            print("目前最优验证集结果:{:.5f}".format(max_micro_f1))
            print("\n=== Epoch %d end ===" % epoch)


def inference(model, data_loader, opt,test_set = False):
    global best_threshold
    """validation"""
    model.eval()
    y, y_hat1 = [], []
    y_hat2 = []
    y_hat3 = []
    y_hat4 = []
    y_hat5 = []
    y_hat6 = []

    predict_ans = []
    tokenizer = AutoTokenizer.from_pretrained(opt.bert_path)


    id2labels = []
    with open(opt.label_idx_path, "r", encoding="utf-8") as f:
        for line in f:
            lin = line.strip().split()
            id2labels.append(lin[0])

    with torch.no_grad():
        for ii, data in enumerate(data_loader):

            if 'Prompt' not in opt.model_name and 'PTuning' not in opt.model_name and 'yangyang' not in opt.data_loader:
                (sentence,), (mask,), nodes, adj_matrixs, labels = data
                if torch.cuda.is_available():
                    sentence = sentence.cuda(opt.gpu)
                    mask = mask.cuda(opt.gpu)
                    labels =  labels.cuda(opt.gpu)
                    for i in range(len(nodes)):
                        nodes[i] = nodes[i].cuda(opt.gpu)
                        adj_matrixs[i] = adj_matrixs[i].cuda(opt.gpu)
            elif 'yangyang' in opt.data_loader:
                (sentence1,sentence2,sentence3), (mask1,mask2,mask3),(mask_idx1,mask_idx2,mask_idx3),labels,label_prompts = data
                if torch.cuda.is_available():
                    sentence1 = sentence1.cuda(opt.gpu)
                    sentence2 = sentence2.cuda(opt.gpu)
                    sentence3 = sentence3.cuda(opt.gpu)

                    mask1 = mask1.cuda(opt.gpu)
                    mask2 = mask2.cuda(opt.gpu)
                    mask3 = mask3.cuda(opt.gpu)

                    labels =  labels.cuda(opt.gpu)
                    label_prompts = label_prompts.cuda(opt.gpu)
            else:
                (sentence,), (mask,),nodes,adj_matrixs,mask_idx, labels,label_prompts = data
                if torch.cuda.is_available():
                    sentence = sentence.cuda(opt.gpu)
                    mask = mask.cuda(opt.gpu)
                    labels =  labels.cuda(opt.gpu)
                    label_prompts = label_prompts.cuda(opt.gpu)
                    for i in range(len(nodes)):
                        nodes[i] = nodes[i].cuda(opt.gpu)
                        adj_matrixs[i] = adj_matrixs[i].cuda(opt.gpu)

            if opt.model_name in MODEL_TYPE1:
                raw_output = model(nodes,adj_matrixs)
            elif opt.model_name in MODEL_TYPE2:
                raw_output = model(sentence)
            elif opt.model_name in MODEL_TYPE3:
                raw_output = model(sentence,nodes,adj_matrixs)
            elif opt.model_name in MODEL_TYPE4:
                raw_output = model(sentence,mask)
            elif opt.model_name in MODEL_TYPE5:
                raw_output = model(sentence,mask,nodes,adj_matrixs)
            elif opt.model_name in MODEL_TYPE6:
                raw_output,output_prompt = model(sentence,mask,mask_idx)
            elif opt.model_name in MODEL_TYPE7:
                _,raw_output = model(sentence,mask,mask_idx)
            elif opt.model_name in MODEL_TYPE8:
                raw_output,output_prompt = model(sentence1,mask1,mask_idx1,sentence2,mask2,mask_idx2,sentence3,mask3,mask_idx3)
            elif opt.model_name in MODEL_TYPE9:
                raw_output = model(sentence1,mask1,sentence2,mask2,sentence3,mask3)
            else:
                raise Exception
            
            output = torch.sigmoid(raw_output).data.cpu().numpy() # [正负例]
            raw_output = raw_output.data.cpu().numpy()

            labels = labels.data.cpu().numpy()
            if opt.model_name in MODEL_TYPE6 or opt.model_name in MODEL_TYPE8:
                output_prompt1 = torch.sigmoid(output_prompt).data.cpu().numpy()
                output_prompt2 = torch.softmax(output_prompt,dim = 0).data.cpu().numpy()
                output_vote = np.argmax(output,axis = 1) # [路径标签数]
                y.append(labels[0]) # 同一个batch都是一样的标签
                # ----------------------- 方案一 ----------------------
                y_hati = np.zeros((labels.shape[1]),dtype=np.float32)
                for vote_label in output_vote:
                    y_hati[vote_label] += 1
                y_hati = y_hati / (np.sum(y_hati) + 1e-4)
                y_hati[np.argmax(y_hati)] = 1 # 票数最多的被预测出来
                y_hat1.append(y_hati)
                # ----------------------- 方案二 ----------------------
                y_hati = np.zeros((labels.shape[1]),dtype=np.float32)
                for i,vote_label in enumerate(output_vote):
                    if output_prompt1[i][1] > 0.5:
                        y_hati[vote_label] += 1
                y_hati = y_hati / (np.sum(y_hati) + 1e-4)
                y_hati[np.argmax(y_hati)] = 1 # 票数最多的被预测出来
                y_hat2.append(y_hati)
                # ----------------------- 方案三 ----------------------
                y_hati = np.zeros((labels.shape[1]),dtype=np.float32)
                for i,vote_label in enumerate(output_vote):
                    y_hati[vote_label] += output_prompt1[i][1] # 加入预测确信度
                y_hati = y_hati / (np.sum(y_hati) + 1e-4)
                y_hati[np.argmax(y_hati)] = 1 # 票数最多的被预测出来
                y_hat3.append(y_hati)
                
                if test_set and opt.test_only:
                    # case study
                    batch_text = tokenizer.batch_decode(sentence.tolist())
                    for batch_text_i in range(len(batch_text)):
                        batch_text[batch_text_i] = batch_text[batch_text_i].replace(tokenizer.pad_token,'')
                    item_dic = {}
                    item = []
                    for text,prompt_prob,cls_prob,vote_label in zip(batch_text,output_prompt1[:,1],output,output_vote):
                        item_item = {}
                        item_item['text'] = text
                        item_item['prompt_prob'] = float(prompt_prob)
                        item_item['cls_prob'] = {}
                        for label_id in range(len(id2labels)):
                            item_item['cls_prob'][id2labels[label_id]] = float(cls_prob[label_id])
                        item_item['vote_label'] = id2labels[vote_label]
                        item.append(item_item)
                    item_dic['info'] = item
                    item_dic['info'].sort(key = lambda x:x['prompt_prob'],reverse=True)
                    item_dic['label'] = id2labels[np.argmax(labels[0])]
                    item_dic['pred'] = {}
                    for label_id in range(len(id2labels)):
                        item_dic['pred'][id2labels[label_id]] = float(y_hati[label_id])
                    if np.argmax(labels[0]) == np.argmax(y_hati):
                        item_dic['ans'] = '正确'
                    else:
                        item_dic['ans'] = '错误'
                    predict_ans.append(item_dic)

                # ----------------------- 方案四 ----------------------
                y_hati = np.zeros((labels.shape[1]),dtype=np.float32)
                for i,vote_label in enumerate(output):
                    y_hati += vote_label * output_prompt2[i][1] # 加入预测确信度
                y_hat4.append(y_hati)
                # ----------------------- 方案五 ----------------------
                y_hati = np.zeros((labels.shape[1]),dtype=np.float32)
                for i,vote_label in enumerate(raw_output):
                    y_hati += vote_label * output_prompt1[i][1] # 加入预测确信度
                y_hat5.append(y_hati)
                # ----------------------- 方案六 ----------------------
                y_hati = np.zeros((labels.shape[1]),dtype=np.float32)
                for i,vote_label in enumerate(output):
                    y_hati += (vote_label>0.5) * output_prompt2[i][1] # 加入预测确信度
                y_hat6.append(y_hati)
            else:
                y.append(labels)
                y_hat1.append(output)

    if opt.model_name in MODEL_TYPE6 or opt.model_name in MODEL_TYPE8:
        y = np.stack(y, axis=0)
        y_hat1 = np.stack(y_hat1, axis=0)
        y_hat2 = np.stack(y_hat2, axis=0)
        y_hat3 = np.stack(y_hat3, axis=0)
        y_hat4 = np.stack(y_hat4, axis=0)
        y_hat5 = np.stack(y_hat5, axis=0)
        y_hat6 = np.stack(y_hat5, axis=0)

        topk1 = topk_accuracy(y_hat1,y)
        topk2 = topk_accuracy(y_hat2,y)
        topk3 = topk_accuracy(y_hat3,y)
        topk4 = topk_accuracy(y_hat4,y)
        topk5 = topk_accuracy(y_hat5,y)
        topk6 = topk_accuracy(y_hat6,y)
        #topk1 = 0
        #topk2 = 0
        #topk3 = 0
        #topk4 = 0
        #topk5 = 0
        #topk6 = 0
        

        if best_threshold is None:
            best_threshold = utils.best_threshold(y,y_hat4)

        print(best_threshold)

        y_hat4_raw = y_hat4.copy()
        y_hat8 = softmax(y_hat4_raw,axis = 1)

        y_hat1[y_hat1>0.5] = 1
        y_hat1[y_hat1<=0.5] = 0
        y_hat2[y_hat2>0.5] = 1
        y_hat2[y_hat2<=0.5] = 0
        y_hat3[y_hat3>0.5] = 1
        y_hat3[y_hat3<=0.5] = 0

        y_hat5[y_hat5>0.0] = 1
        y_hat5[y_hat5<=0.0] = 0
        y_hat6[y_hat6>0.5] = 1
        y_hat6[y_hat6<=0.5] = 0
        y_hat7 = np.zeros((y.shape[0],y.shape[1]))
        y_hat7[y_hat4>best_threshold] = 1
        y_hat4[y_hat4>0.5] = 1
        y_hat4[y_hat4<=0.5] = 0

        print(np.mean(y_hat8))
        print(np.max(y_hat8))

        print('-'*10,'方案1','-'*10)
        metrics_test = all_metrics(y_hat1, y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, y_hat1, digits=4,target_names = id2labels)
        print(report)
        print(topk1)

        print('-'*10,'方案2','-'*10)
        metrics_test = all_metrics(y_hat2, y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, y_hat2, digits=4,target_names = id2labels)
        print(report)
        print(topk2)

        print('-'*10,'方案3','-'*10)
        metrics_test = all_metrics(y_hat3, y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, y_hat3, digits=4,target_names = id2labels)
        print(report)
        print(topk3)

        print('-'*10,'方案4','-'*10)
        metrics_test = all_metrics(y_hat4, y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, y_hat4, digits=4,target_names = id2labels)
        print(report)
        print(topk4)

        print('-'*10,'方案5','-'*10)
        metrics_test = all_metrics(y_hat5, y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, y_hat5, digits=4,target_names = id2labels)
        print(report)
        print(topk5)

        print('-'*10,'方案6','-'*10)
        metrics_test = all_metrics(y_hat6, y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, y_hat6, digits=4,target_names = id2labels)
        print(report)
        print(topk6)

        print('-'*10,'方案7','-'*10)
        metrics_test = all_metrics(y_hat7, y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, y_hat7, digits=4,target_names = id2labels)
        print(report)
        print(topk6)

        print('-'*10,'方案8','-'*10)
        metrics_test = all_metrics((y_hat8>0.01).astype(int), y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, (y_hat8>0.01).astype(int), digits=4,target_names = id2labels)


        print('-'*10,'方案9','-'*10)
        metrics_test = all_metrics((y_hat8>0.02).astype(int), y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, (y_hat8>0.02).astype(int), digits=4,target_names = id2labels)


        print('-'*10,'方案10','-'*10)
        metrics_test = all_metrics((y_hat8>0.05).astype(int), y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, (y_hat8>0.05).astype(int), digits=4,target_names = id2labels)


        print('-'*10,'方案11','-'*10)
        metrics_test = all_metrics((y_hat8>0.015).astype(int), y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, (y_hat8>0.015).astype(int), digits=4,target_names = id2labels)

        print('-'*10,'方案12','-'*10)
        metrics_test = all_metrics((y_hat8>0.025).astype(int), y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, (y_hat8>0.025).astype(int), digits=4,target_names = id2labels)

        print('-'*10,'方案13','-'*10)
        metrics_test = all_metrics((y_hat8>0.035).astype(int), y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, (y_hat8>0.035).astype(int), digits=4,target_names = id2labels)
        
        print('-'*10,'方案14','-'*10)
        metrics_test = all_metrics((y_hat8>0.04).astype(int), y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, (y_hat8>0.04).astype(int), digits=4,target_names = id2labels)
        
        print('-'*10,'方案15','-'*10)
        metrics_test = all_metrics((y_hat8>0.45).astype(int), y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, (y_hat8>0.45).astype(int), digits=4,target_names = id2labels)

        print('-'*10,'方案16','-'*10)
        metrics_test = all_metrics((y_hat8>0.5).astype(int), y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, (y_hat8>0.5).astype(int), digits=4,target_names = id2labels)
        
        print('-'*10,'方案17','-'*10)
        metrics_test = all_metrics((y_hat4_raw>0.45).astype(int), y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, (y_hat4_raw>0.45).astype(int), digits=4,target_names = id2labels)

        print('-'*10,'方案18','-'*10)
        metrics_test = all_metrics((y_hat4_raw>0.55).astype(int), y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, (y_hat4_raw>0.55).astype(int), digits=4,target_names = id2labels)

    else:
        y = np.concatenate(y,axis=0)
        y_hat = np.concatenate(y_hat1,axis=0)
        topk = topk_accuracy(y_hat,y)
        y_hat[y_hat>0.5] = 1
        y_hat[y_hat<=0.5] = 0
        print('-'*20)
        metrics_test = all_metrics(y_hat, y)
        print_metrics(metrics_test)
        report = metrics.classification_report(y, y_hat, digits=4,target_names = id2labels)
        print(report)
        print(topk)


    if not test_set:
        return metrics_test["f1_micro"], report
    else:
        return metrics_test["f1_micro"], report, json.dumps(predict_ans,ensure_ascii=False,indent=4) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--model_name', type=str, default='BertCNN_v1')
    parser.add_argument('--bert_path', type=str, default='bert_chinese', help='pretrained path')
    parser.add_argument('--data_path', type=str, default="data/electronic-medical-record")
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--accumulation_steps',type=int ,default = 1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--other_lr', type=float, default=5e-4)
    parser.add_argument('--warmup_rate', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--result_path', type=str, default="result")
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--graph', type=int, default=5)
    parser.add_argument('--data_version', type=str, default='')
    parser.add_argument('--use_wandb',action="store_true", default=False)
    parser.add_argument('--data_loader',type=str,default='DataLoader_short')
    parser.add_argument('--max_length',type=int,default=510)
    parser.add_argument('--test_freq',type=int,default=1)
    parser.add_argument('--test_only',action="store_true", default=False)
    parser.add_argument('--test_model_path',type=str,default='')

    # 隐藏数
    parser.add_argument('--hidden_size',type=int,default=768)
    # 标签平滑
    parser.add_argument('--label_smooth_lambda',type=float,default=0)
    parser.add_argument('--sample_radio',type=int,default=2)

    args = parser.parse_args()

    save_model_names = [args.bert_path.split('/')[-1], args.model_name, "seed", str(args.seed),
                        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")]
    save_model_path = os.path.join("checkpoints", '_'.join(save_model_names) + ".pth")     # best model path
    result_path = os.path.join("result", '_'.join(save_model_names) + ".txt")         # the report of test dataset path
    error_path = os.path.join("result", '_'.join(save_model_names) + "err.txt")         # the report of test dataset path
    score_path = os.path.join("result",'_'.join(save_model_names) + "score.txt")      # 保存分数获取结果
    correct_path = os.path.join(args.data_path,'train_correct.json')                  # the correct of test dataset path

    opt = DefaultConfig(args, save_model_path)
    print(opt.use_wandb)
    if opt.use_wandb:
        save_opt = copy.copy(opt)
        save_opt.entity2id = None
        save_opt.label2id = None
        save_opt.adj_matrixs = None
        save_opt.ent_tokens_matrix = None
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="diagnose",
            # track hyperparameters and run metadata
            config=save_opt
        )

    import dataloader
    data_loader = eval('dataloader.' + opt.data_loader)

    print(opt)

    train_data_loader = data_loader(opt.train_path, opt, shuffle=True)
    dev_data_loader = data_loader(opt.dev_path, opt, shuffle=False)
    test_data_loader = data_loader(opt.test_path, opt, shuffle=False)

    if opt.test_only:
        opt.save_model_path = opt.test_model_path
    else:
        train(opt, train_data_loader, dev_data_loader,test_data_loader)
    model = import_module('Model.' + opt.model_name).Model(opt)
    if torch.cuda.is_available():
        model = model.cuda(opt.gpu)
    save_dict = torch.load(opt.save_model_path,map_location=torch.device('cuda:%d'%opt.gpu))
    model.load_state_dict(save_dict['dict'])
    #micro_f1, report = inference(model, dev_data_loader, opt)
    micro_f1, report,error_report = inference(model, test_data_loader, opt,test_set=True)
    print('测试集micro-f1',micro_f1)
    write_result(report, result_path)
    write_result(error_report, error_path)
    if opt.use_wandb:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()

    print("==============Finish==============")

