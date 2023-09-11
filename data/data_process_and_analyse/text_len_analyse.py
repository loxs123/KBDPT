#  
import json

text_len = []
label_num = []
train_num = 0
dev_num = 0
test_num = 0


label2id = {}
with open('../mimic3-50/label2id.txt','r',encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        label2id[line] = 0

with open('../mimic3-50/train.json','r',encoding='utf-8') as f:
    data = json.load(f)
    train_num = len(data)
    for item in data:
        text_len.append(len(item['doc'])+len(item['现病史'])+len(item['既往史']))
        label_num.append(len(item['出院诊断'].split(';')))
        for label in set(item['出院诊断'].split(';')):
            if label in label2id:
                label2id[label] += 1


with open('../mimic3-50/test.json','r',encoding='utf-8') as f:
    data = json.load(f)
    dev_num = len(data)
    for item in data:
        text_len.append(len(item['主诉'])+len(item['现病史'])+len(item['既往史']))
        label_num.append(len(item['出院诊断'].split(';')))


with open('../mimic3-50/dev.json','r',encoding='utf-8') as f:
    data = json.load(f)
    test_num = len(data)
    for item in data:
        text_len.append(len(item['主诉'])+len(item['现病史'])+len(item['既往史']))
        label_num.append(len(item['出院诊断'].split(';')))

# print(train_num)
# print(dev_num)
# print(test_num)
# print(sum(text_len)/len(text_len))
# print(sum(label_num))
# print(sum(label_num)/len(label_num))

for label in label2id:
    print(label)
for label in label2id:
    print(label2id[label])


#  
import json

text_len = []
label_num = []
train_num = 0
dev_num = 0
test_num = 0
with open('../electronic-medical-record/train.json','r',encoding='utf-8') as f:
    data = json.load(f)
    train_num = len(data)
    for item in data:
        text_len.append(len(item['主诉'])+len(item['现病史'])+len(item['既往史']))
        label_num.append(len(item['出院诊断']))

with open('../electronic-medical-record/test.json','r',encoding='utf-8') as f:
    data = json.load(f)
    dev_num = len(data)
    for item in data:
        text_len.append(len(item['主诉'])+len(item['现病史'])+len(item['既往史']))
        label_num.append(len(item['出院诊断']))


with open('../electronic-medical-record/dev.json','r',encoding='utf-8') as f:
    data = json.load(f)
    test_num = len(data)
    for item in data:
        text_len.append(len(item['主诉'])+len(item['现病史'])+len(item['既往史']))
        label_num.append(len(item['出院诊断']))

# print(train_num)
# print(dev_num)
# print(test_num)
# print(sum(text_len)/len(text_len))
# print(sum(label_num))
# print(sum(label_num)/len(label_num))










