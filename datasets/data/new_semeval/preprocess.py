import os
import json
import random
random.seed(2020)
import collections

def format_file(path):
    data = json.load(open(path))
    with open(path.replace('json', 'txt'), 'w',encoding='utf-8') as f:
        for x in data:
            new_line = {}
            new_line['token'] = x['tokens']
            new_line['relation'] = x['label']
            pos1 = x['entities'][0]
            pos2 = x['entities'][1]
            new_line['h'] = {'pos':pos1}
            new_line['t'] = {'pos':pos2}
            f.writelines(str(new_line)+'\n')

def split_train(path, out1, out2):
    data = json.load(open(path))
    random.shuffle(data)
    length = len(data)
    sp = length-1500
    train = data[:sp]
    dev = data[sp:]
    with open(out1, 'w', encoding='utf-8') as f1:
        json.dump(train, f1)
    with open(out2, 'w', encoding='utf-8') as f2:
        json.dump(dev, f2)




def reltoid(path):
    data = json.load(open(path))
    with open(path.replace('train', 'srel2id'), 'w',encoding='utf-8') as f:
        rel2id = {}
        for x in data:
            nums = len(rel2id)
            if x['label'] in rel2id:
                pass
            else:
                rel2id[x['label']] = nums
        json.dump(rel2id, f)


if __name__ == '__main__':
    path = 'new_semeval'
    format_file(os.path.join(path, 'test.json'))
    tpath = os.path.join(path, 'train.json')
    out1 = os.path.join(path, 'new_train.json')
    out2 = os.path.join(path, 'new_dev.json')
    split_train(tpath, out1, out2)
    format_file(os.path.join(path, 'new_dev.json')) # new_dev -> dev 
    format_file(os.path.join(path, 'new_train.json'))# new_train -> train
    reltoid(os.path.join(path, 'train.json'))
