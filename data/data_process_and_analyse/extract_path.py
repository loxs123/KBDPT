import json

def update_store(file):
    global path2id
    with open(file,'r',encoding='utf-8') as f:
        data = json.load(f)
    new_data = []

    for item in data:
        new_path_T = []
        for path_t in item['path_T']:
            path_t = '\t'.join(path_t)
            if path_t not in path2id:
                path2id[path_t] = len(path2id)
            new_path_T.append(path2id[path_t])
        new_path_F = []
        for path_f in item['path_F']:
            path_f = '\t'.join(path_f)
            if path_f not in path2id:
                path2id[path_f] = len(path2id)
            new_path_F.append(path2id[path_f])
        item['path_T'] = new_path_T
        item['path_F'] = new_path_F
        new_data.append(item)
    with open(file,'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=4)


update_store('../chinese/train.json')
update_store('../chinese/dev.json')
update_store('../chinese/test.json')
path2id = {}
id2path = []
for path in path2id:
    id2path.append(path)
with open('../chinese/path.json','w',encoding='utf-8') as f:
    json.dump(id2path,f,ensure_ascii=False,indent=4)

update_store('../chinese-small/train.json')
update_store('../chinese-small/dev.json')
update_store('../chinese-small/test.json')
path2id = {}
id2path = []
for path in path2id:
    id2path.append(path)
with open('../chinese-small/path.json','w',encoding='utf-8') as f:
    json.dump(id2path,f,ensure_ascii=False,indent=4)
