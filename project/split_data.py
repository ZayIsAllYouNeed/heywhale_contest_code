import pandas as pd
import json
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_in_dir",default='./data/contest_data',type=str)
parser.add_argument("--data_out_dir",default='./data/tmp_data',type=str)

args = parser.parse_args()

random.seed(2023)

data = pd.read_csv('../train.csv', header=None, )
print(len(data))
data = data.sample(frac=1, random_state=2023)

train = data[:17000]#.to_csv('./data/train.csv', index=None, header=None)
val = data[17000:]#.to_csv('./data/valid.csv', index=None, header=None)
testA =  pd.read_csv(f'{args.data_in_dir}/preliminary_a_test.csv', header=None)
testB = pd.read_csv(f'{args.data_in_dir}/preliminary_b_test.csv', header=None)

js_train = [{'report_ID':x[0], 'description':x[1], 'diagnosis':x[2]} for x in train.to_dict('records')]
js_val = [{'report_ID':x[0], 'description':x[1], 'diagnosis':x[2]} for x in val.to_dict('records')]
js_testA = [{'report_ID':x[0], 'description':x[1], 'diagnosis':"0 0"} for x in testA.to_dict('records')]
js_testB = [{'report_ID':x[0], 'description':x[1], 'diagnosis':"0 0"} for x in testB.to_dict('records')]

label, unlabelB, unlabelA = js_val+js_train, js_testB, js_testA
len_val = 2000
for i in range(10):
    dat = []
    if not os.path.exists(f'{args.data_out_dir}/{i}'):
        os.mkdir(f'{args.data_out_dir}/{i}')
    for x in label[:i*len_val]+label[(i+1)*len_val:]:
        d = dict()
        d['id'] = x['report_ID']
        d['article'] = x['description'].strip()
        d['summarization'] = x['diagnosis'].strip()
        dat.append(d)
    json.dump(dat, open(f'{args.data_out_dir}/{i}/train.json', 'w'))
    
    dat = []
    for x in label[i*len_val: (i+1)*len_val]:
        d = dict()
        d['id'] = x['report_ID']
        d['article'] = x['description'].strip()
        d['summarization'] = x['diagnosis'].strip()
        dat.append(d)
    json.dump(dat, open(f'{args.data_out_dir}/{i}/dev.json', 'w'))

    dat = []
    for x in unlabelB:
        d = dict()
        d['id'] = x['report_ID']
        d['article'] = x['description'].strip()
        d['summarization'] = x['diagnosis'].strip()
        dat.append(d)
    json.dump(dat, open(f'{args.data_out_dir}/{i}/test.json', 'w'))

    # dat = []
    # for x in unlabelA+unlabelB:
    #     d = dict()
    #     d['id'] = x['report_ID']
    #     d['article'] = x['description'].strip()
    #     d['summarization'] = x['diagnosis'].strip()
    #     dat.append(d)
    # json.dump(dat, open(f'./data/{i}/test_AB.json', 'w'))