import json
import pandas as pd
import time


def json_from(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


senti = json_from('senti_res-2021-11-04-15-46-24.json')
ner = json_from('ner_res-2021-11-05-22-12-05.json')
merge = {}
data = {'id': [], 'BIO_anno': [], 'class': []}
for _id in senti:
    merge[_id] = [ner[_id], senti[_id]]
for _id in merge:
    data['id'].append(_id)
    data['BIO_anno'].append(merge[_id][0])
    data['class'].append(merge[_id][1])
submit_df = pd.DataFrame(data)
submit_df.to_csv(f'./results/submit-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.csv', index=False, sep=',')
