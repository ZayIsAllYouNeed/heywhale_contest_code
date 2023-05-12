import argparse
import json
import logging
import os
import random
import sys
import nltk


import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModel, BertTokenizer,BertForTokenClassification,
                          DataCollatorForTokenClassification, HfArgumentParser,DataCollatorForSeq2Seq,Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, Trainer, TrainerCallback,AutoModelForSeq2SeqLM)
from transformers.trainer_utils import is_main_process
from datasets import load_metric,Dataset
from utils import DataTrainingArguments, ModelArguments, load_json

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# from modeling_cpt import CPTModel, CPTForConditionalGeneration
from transformers import BartForConditionalGeneration, EarlyStoppingCallback

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",default='/path/to/model',type=str)
parser.add_argument("--dataset", default="lcsts",type=str)
parser.add_argument("--lr",default=2e-5,type=float)
parser.add_argument("--batch_size",default='50',type=str)
parser.add_argument("--eval_batch_size",default='100',type=str)
parser.add_argument("--epoch",default='5',type=str)
parser.add_argument("--data_dir",default="/path/to/dataset/",type=str)
parser.add_argument("--output_dir",default="/path/to/dataset/",type=str)

args = parser.parse_args()
arg_dict=args.__dict__

logger = logging.getLogger(__name__)

dataset_name=arg_dict['dataset']
outdir_1=arg_dict['output_dir'] #'output'
if not os.path.exists(outdir_1):
    os.mkdir(outdir_1)
outdir=outdir_1
# outdir=outdir_1+'/'+dataset_name
# if not os.path.exists(outdir):
#     os.mkdir(outdir)

seed=len(os.listdir(outdir))+1
outdir=outdir+'/'+str(seed)
length_map={'lcsts':'30','csl':'50','adgen':'128', 'heywhale':'80'}


args=[
    '--model_name_or_path',arg_dict['model_path'],
    '--do_train','--do_eval','--do_predict',
    '--train_file',os.path.join(arg_dict['data_dir'],'train.json'),
    '--validation_file',os.path.join(arg_dict['data_dir'],'dev.json'),
    '--test_file',os.path.join(arg_dict['data_dir'],'test.json'),
    '--output_dir',outdir,
    '--per_device_train_batch_size',arg_dict['batch_size'],
    '--per_device_eval_batch_size',arg_dict['eval_batch_size'],
    '--overwrite_output_dir',
    '--max_source_length=150',#512
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
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
set_seed(training_args.seed)

datasets={}
data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file
if data_args.test_file is not None:
    data_files["test"] = data_args.test_file
for key in data_files:
    print(key)
    datasets[key]=load_json(data_files[key])

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
logger.info("Training/evaluation parameters %s", training_args)

tokenizer=BertTokenizer.from_pretrained(model_args.model_name_or_path)#use_fast=True
model=BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

# zay
pretrained_pt_path = ''#'/mnt/nlp01/usr/zhengayong/heywhale/CPT/pretrain_large_3/checkpoints/bart-large/last/mp_rank_00_000/model_optim_rng.pt'
#'/mnt/nlp01/usr/zhengayong/heywhale/CPT/pretrain_large/checkpoints/bart-large/last/mp_rank_00_000/model_optim_rng.pt'
# pretrained_pt_path = '/mnt/nlp01/usr/zhengayong/heywhale/CPT/finetune/generation_3/output/heywhale/5/pytorch_model.bin'
if pretrained_pt_path:
    pt = torch.load(pretrained_pt_path)
    model.load_state_dict(pt['model']['language_model'])
    # model.load_state_dict(pt)#['model']['language_model'])
    print('*'*50)
    print('Loaded pretrained pt parameters ^_^ !')

model.config.max_length=data_args.val_max_target_length

# zay ##############################################################################################################################
print('#'*50)
print(data_args.preprocessing_num_workers)
data_args.preprocessing_num_workers = 8
print(data_args.preprocessing_num_workers)
print('#'*50)

# zay
training_args.load_best_model_at_end=True
training_args.metric_for_best_model='eval_cider'
training_args.greater_is_better=True
# training_args.save_steps=500
training_args.save_strategy='epoch' #no
training_args.save_total_limit=10 #8 #None
# training_args.resume_from_checkpoint=None
training_args.lr_scheduler_type='cosine'#linear
training_args.load_best_model_at_end=True#False
# training_args.label_smoothing_factor=0.0
# training_args.generation_max_length=None
training_args.generation_num_beams=4 #None
training_args.dataloader_num_workers=2 #0
training_args._n_gpu=1 #1


text_column='article'
summary_column='summarization'
column_names = datasets["train"].column_names
max_target_length = data_args.val_max_target_length
padding=False

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[summary_column]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)


    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



if training_args.do_train:
    train_dataset = datasets["train"]
    if "train" not in datasets:
        raise ValueError("--do_train requires a train dataset")
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

if training_args.do_eval:
    max_target_length = data_args.val_max_target_length
    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = datasets["validation"]
    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

if training_args.do_predict:
    max_target_length = data_args.val_max_target_length
    if "test" not in datasets:
        raise ValueError("--do_predict requires a test dataset")
    test_dataset = datasets["test"]
    if data_args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_test_samples))
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )


max_eval_num=30000
if len(eval_dataset)>max_eval_num:
    eval_dataset=Dataset.from_dict(eval_dataset[:max_eval_num])
print(len(eval_dataset))


# Data collator
label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)



# Metric
from rouge import Rouge 
rouge = Rouge()

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # # rougeLSum expects newline after each sentence
    # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    while '' in preds:
        idx=preds.index('')
        preds[idx]='。'

    return preds, labels

def compute_metrics(eval_preds):
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
    scores = rouge.get_scores(decoded_preds, decoded_labels,avg=True)
    for key in scores:
        scores[key]=scores[key]['f']*100

    result=scores
# zay
    from utils_hw import Smoother # 来自 heywhale 的 baseline 的文件
    from evaluate_hw import CiderD
    metrics_hw = Smoother(100)
    res, gts = [], {}
    for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        res.append({'image_id':i, 'caption': [pred]})
        gts[i] = [label]
    CiderD_scorer = CiderD(df='corpus', sigma=15)
    cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
    metrics_hw.update(cider = cider_score)
    # print('hw'+'*'*100)
    # print(metrics_hw.value())
    # print('hw'+'*'*100)
    result_1 = metrics_hw.value()
    result_1.update(result)
    result = result_1
    # result.update(metrics_hw.value())
# zay
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    
    # zay
    print(result)
    
    return result

class TestCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        metrics['epoch']=state.epoch
        state.log_history.append(metrics)




print('model_args'+'*'*100)
print(model_args)
print('data_args'+'*'*100)
print(data_args)
print('training_args'+'*'*100)
print(training_args)
# exit()

early_stop = EarlyStoppingCallback(early_stopping_patience = 5, early_stopping_threshold = 0.001)# 7

# Initialize our Trainer

from transformers import AdamW

weight_decay = 1e-2

decay_parameters = [name for name, _ in model.named_parameters() if "bias" not in name]
optimizer_grouped_parameters = [
    # zay
    {
        "params": [p for n, p in model.named_parameters() if (n in decay_parameters and "shared" in n)],
        "weight_decay": weight_decay,
        "lr": training_args.learning_rate * 5,
    },
    {
        "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and "shared" in n)],
        "weight_decay": 0.0,
        "lr": training_args.learning_rate * 5,
    },
    # zay
    {
        "params": [p for n, p in model.named_parameters() if (n in decay_parameters and "shared" not in n)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and "shared" not in n)],
        "weight_decay": 0.0,
    },
]
# 在这个里面设置每一层的学习率
# optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n], 'lr': 1e-5},\
#                                 {'params': [p for n, p in model.named_parameters() if 'encoder' not in n], 'lr': 5e-5}]

optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
# lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 0,)

class CustomTrainer(Seq2SeqTrainer):
    # pass
    def training_step(self, model, inputs):
        '''
        The tensor with training loss on this batch.

        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.
        '''
        loss = super().training_step(model, inputs)
        # 对抗训练
        fgm.attack()  # （#2）在embedding上添加对抗扰动
        loss += super().training_step(model, inputs)
        fgm.restore()  # （#5）恢复embedding参数
        return loss

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}  # 用于保存模型扰动前的参数

    def attack(self, epsilon=0.5, emb_name='shared'):  # emb_name表示模型中embedding的参数名 # 'word_embeddings' # bart里是shared
        '''
        生成扰动和对抗样本
        '''
        for name, param in self.model.named_parameters():  # 遍历模型的所有参数
            if param.requires_grad and emb_name in name:  # 只取word embedding层的参数
                self.backup[name] = param.data.clone()  # 保存参数值
                norm = torch.norm(param.grad)  # 对参数梯度进行二范式归一化
                if norm != 0 and not torch.isnan(norm):  # 计算扰动，并在输入参数值上添加扰动
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='shared'):  # emb_name表示模型中embedding的参数名 'word_embeddings' # bart里是shared
        '''
        恢复添加扰动的参数
        '''
        for name, param in self.model.named_parameters():  # 遍历模型的所有参数
            if param.requires_grad and emb_name in name:  # 只取word embedding层的参数
                assert name in self.backup
                param.data = self.backup[name]  # 重新加载保存的参数值
        self.backup = {}
        
fgm = FGM(model)




trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    # zay
    callbacks=[early_stop],#[TestCallback],
    optimizers=(optimizer, None),
)


# Training
if training_args.do_train:
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if trainer.is_world_process_zero():
    if training_args.predict_with_generate:
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        test_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True,
        )
        test_preds = [pred.strip() for pred in test_preds]
        output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
        with open(output_test_preds_file, "w",encoding='UTF-8') as writer:
            writer.write("\n".join(test_preds))
            
        import pandas as pd
        with open(output_test_preds_file) as f:
            data = f.readlines()
        data = [{'id':i, 'diagnosis':s} for i, s in enumerate(data)]
        submit_path = os.path.join(training_args.output_dir, "pred.csv")
        pd.DataFrame(data).to_csv(submit_path, index=None, header=None)
