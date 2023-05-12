import argparse
import json
import logging
import os
import random
import sys
import nltk
import pandas as pd
import numpy as np
from datasets import load_metric,Dataset
from utils import DataTrainingArguments, ModelArguments, load_json
from glob import glob
import torch
import transformers
from transformers.trainer_utils import is_main_process
from transformers import (BartForConditionalGeneration, BertTokenizer,
                          HfArgumentParser,DataCollatorForSeq2Seq,Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
# Metric
# from rouge import Rouge 
# rouge = Rouge()

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    while '' in preds:
        idx=preds.index('')
        preds[idx]= '10' #'。'
    return preds, labels

def compute_metrics(eval_preds, tokenizer, data_args):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
# zay
    from bleu_metric import Metric
    metric = Metric(None)
    metric.hyps = [pred.split() for pred in decoded_preds]
    metric.refs = [[label.split()] for label in decoded_labels]
    bleu_score = metric.calc_bleu_k(4)

    # from utils_hw import Smoother # 来自 heywhale 的 baseline 的文件
    from evaluate_hw import CiderD
    # metrics_hw = Smoother(100)
    res, gts = [], {}
    for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        res.append({'image_id':i, 'caption': [pred]})
        gts[i] = [label]
    CiderD_scorer = CiderD(df='corpus', sigma=15)
    cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
    
    hw_score = (cider_score*2.0 + bleu_score)/3.0
    result = {"cider": cider_score, "bleu": bleu_score, "hw_score": hw_score}#metrics_hw.value()
# zay
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    # zay
    # print(result)
    return result

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_function(examples, tokenizer, data_args, text_column='article', summary_column='summarization', max_target_length=80, padding=False):
    inputs = examples[text_column]
    targets = examples[summary_column]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def inference(model_paths, input_data_path, output_data_path):
    print('start inference...')
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    parser = argparse.ArgumentParser()
    print('16...')
    parser.add_argument("--model_path",default=model_paths,type=str)
    parser.add_argument("--lr",default=2e-5,type=float)
    parser.add_argument("--batch_size",default='50',type=str)
    parser.add_argument("--eval_batch_size",default='500',type=str)
    parser.add_argument("--epoch",default='5',type=str)
    parser.add_argument("--load_state_files",default="./data/best_model.*.bin",type=str)
    parser.add_argument("--dataset", default="lcsts",type=str)
    parser.add_argument("--data_dir",default="./data/contest_data/preliminary_b_test.csv",type=str)
    # parser.add_argument("--output_dir",default="./data/submission",type=str)
    parser.add_argument("--output_dir",default="./data",type=str)
    print('11...')
    #args = parser.parse_args()
    #print('12...')
    #arg_dict=args.__dict__
    arg_dict = {"model_path":model_paths, "lr":2e-5, "batch_size":'50', "eval_batch_size":'100', "epoch":'5', "load_state_files":"./data/best_model.*.bin", "dataset":"heywhale", "data_dir":"./data/contest_data/preliminary_b_test.csv", "output_dir":"./data"}
    print('13...')
    logger = logging.getLogger(__name__)
    print('14...')
    dataset_name=arg_dict['dataset']
    print('15...')
    outdir = arg_dict['output_dir']
    print('1...')
    if not os.path.exists(arg_dict['output_dir']):
        os.mkdir(arg_dict['output_dir'])
    print('2...')
    seed=len(os.listdir(outdir))+1
    outdir=outdir+'/'+str(seed)
    length_map={'lcsts':'30','csl':'50','adgen':'128', 'heywhale':'85'}
    print('3...')

    args=[
        '--model_name_or_path',arg_dict['model_path'],
        '--do_train','--do_eval','--do_predict',
        '--train_file',os.path.join(arg_dict['data_dir'],'train.json'),
        '--validation_file',os.path.join(arg_dict['data_dir'],'dev.json'),
        '--test_file',input_data_path,#os.path.join(arg_dict['data_dir'],'test.json'),
        '--output_dir',outdir,
        '--per_device_train_batch_size',arg_dict['batch_size'],
        '--per_device_eval_batch_size',arg_dict['eval_batch_size'],
        '--overwrite_output_dir',
        '--max_source_length=186',#512
        '--val_max_target_length='+length_map[arg_dict['dataset']],
        '--predict_with_generate=1',
        '--seed',str(1000*seed),
        '--num_train_epochs',arg_dict['epoch'],
        '--save_strategy','no',
        '--evaluation_strategy','epoch',
        '--learning_rate',str(arg_dict['lr']),
        # zay
        ## '--preprocessing_num_workers', '4',
    ]

    print('#'*100)
    print(sys.argv)
    if sys.argv[0] == '':
        sys.argv[0] = 'predict.py'
    print(sys.argv)
    print('4...')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)
##############################################################################################################################
    print('5...')
    set_seed(training_args.seed)
    datasets={}
    data_files = {}
    print('6...')
    testB = pd.read_csv(data_args.test_file, header=None).fillna('')
    print('7...')
    if len(testB.to_dict('records')[0]) == 3:#
        # 原来是'diagnosis':"0 0",不在字典里，会跳过，相当于""空字符串，所以报错
        js_testB = [{'report_ID':x[0], 'description':x[1], 'diagnosis':"10", 'clinical': x[2]} for x in testB.to_dict('records')]
    else:
        js_testB = [{'report_ID':x[0], 'description':x[1], 'diagnosis':x[2], 'clinical': x[3]} for x in testB.to_dict('records')]
    testB = []
    for x in js_testB:
        d = dict()
        d['id'] = x['report_ID']
        d['article'] = x['description'].strip()
        if x['clinical']:
            d['article'] += ' ' + x['clinical'].strip()
        d['summarization'] = x['diagnosis'].strip()
        testB.append(d)
    print('8...')
    json.dump(testB, open('./data/test.json', 'w'))
    data_files["test"] = './data/test.json'
    datasets['test']= load_json(data_files["test"]) # testB #
    # zay
    # data_args.preprocessing_num_workers = 2
    training_args.load_best_model_at_end=True
    training_args.metric_for_best_model='eval_cider'
    training_args.greater_is_better=True
    # training_args.save_steps=500
    training_args.save_strategy='epoch' #no
    training_args.save_total_limit=8 #None
    # training_args.resume_from_checkpoint=None
    training_args.lr_scheduler_type='cosine'#linear
    training_args.load_best_model_at_end=True#False
    # training_args.label_smoothing_factor=0.0
    # training_args.generation_max_length=None
    training_args.generation_num_beams=6 #None
    training_args.dataloader_num_workers=8 #0
    training_args._n_gpu=1 #使用的 GPU 数量
    training_args.do_train = False #取消训练
    training_args.do_eval = False #取消验证
    
    column_names = datasets["test"].column_names
    max_target_length = data_args.val_max_target_length
    padding=False
##############################################################################################################################
    tokenizer=BertTokenizer.from_pretrained(model_args.model_name_or_path)#, use_fast=True)#use_fast=True
    model=BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    model.config.max_length=data_args.val_max_target_length
    # model.cuda()
    if "test" not in datasets:
        raise ValueError("--do_predict requires a test dataset")
    test_dataset = datasets["test"]
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer=tokenizer, data_args=data_args, max_target_length=max_target_length, padding=padding),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    print('9...')
    # Initialize our Trainer  
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        # train_dataset=train_dataset if training_args.do_train else None,
        # eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer=tokenizer, data_args=data_args) if training_args.predict_with_generate else None,
    )
    print('10...')
    predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
    test_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True,
    )
    print(metrics)
    test_preds = [pred.strip() for pred in test_preds]
    data = [{'id':i, 'diagnosis':s} for i, s in enumerate(test_preds)]
    pd.DataFrame(data).to_csv(output_data_path, index=None, header=None)
    print('10...')
    
if __name__ == '__main__':
    inference('./my_model', './data/test.csv', './data/pred.csv')