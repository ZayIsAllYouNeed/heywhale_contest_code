# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 10:16
# @Author  : binbin
# @File    : get_best_predict.py
# @Description : 这个s文件的作用是模型融合预测
import csv
import numpy as np
import argparse


def read_file(file_paths):
    res = []
    for i in range(len(file_paths)):
        with open(file_paths[i], 'r') as fp:
            reader = csv.reader(fp)
            data = [row for row in reader]
        res.append(data.copy())
    return res


def save_file(file_path, data):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for idx in range(len(data)):
            writer.writerow([str(idx), data[idx].strip()])


def get_score(target_sen, others, weight):
    # 计算目标句子和其他句子的分数
    # 猜测可以在这里设置每个模型的预测权重
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    score = 0
    smooth = SmoothingFunction().method1
    for i, sen in enumerate(others):
        # 分别计算每两个句子之间的分数，这里的计算方式可以更换
        # 使用cider也可以
        score += sentence_bleu([sen], target_sen, smoothing_function=smooth, weights=(0.25, 0.25, 0.25, 0.25)) * weight[i]
    return score

weight = []

def merge_save_best_result(results, res_path):
    global weight
    weight = [1/len(results)]*(len(results))
    
    data = []
    for result in results:
        dat = [[x['id'], x['diagnosis']] for x in result]
        data.append(dat)
    
    results = []
    for i in range(len(data[0])):  # 遍历每一个样本
        currow = [data[j][i][1] for j in range(len(data))]
        results.append(currow)
    print('load data. Begin to cal...')
    best_results = []
    for idx in range(len(data[0])):
        answers = results[idx]
        scores = []
        for i in range(len(answers)):
            raw_ans = answers[i]
            ref_answers = answers[:i] + answers[1 + i:]
            ref_weight = weight[:i] + weight[1 + i:]
            sc = 0.0001 + get_score(raw_ans, ref_answers, ref_weight)  # 计算目标句子和其他句子的F1分数
            scores.append(sc)
        r_ans = answers[np.argmax(scores)]
        best_results.append(r_ans)

    # 保存最终的结果
    save_file(res_path, best_results)
    print('Done!')    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_results_dir",default='./data/tmp_data/all_results',type=str)
    parser.add_argument("--output_result_dir",default='./data/submission',type=str)
    args = parser.parse_args()

    # 假设有三个模型的预测结果
    from glob import glob
    model_output_file = []
    for f in glob(args.all_results_dir+'/*.csv'):
        model_output_file.append(f)
    print(model_output_file)
    # model_output_file = ['./output/mydataset/13/test_generations.csv', './output/mydataset/12/test_generations.csv', './output/mydataset/11/test_generations.csv']
    # 设置每个模型的权重,要个上面的模型个数对应好
    weight = [1/len(model_output_file)]*(len(model_output_file))
    # 融合后的结果保存的位置
    best_output_file = args.output_result_dir+'/results.csv'

    data = read_file(model_output_file)
    results = []
    for i in range(len(data[0])):  # 遍历每一个样本
        currow = [data[j][i][1] for j in range(len(data))]
        results.append(currow)
    print('load data. Begin to cal...')
    best_results = []
    for idx in range(len(data[0])):
        answers = results[idx]
        scores = []
        for i in range(len(answers)):
            raw_ans = answers[i]
            ref_answers = answers[:i] + answers[1 + i:]
            ref_weight = weight[:i] + weight[1 + i:]
            sc = 0.0001 + get_score(raw_ans, ref_answers, ref_weight)  # 计算目标句子和其他句子的F1分数
            scores.append(sc)
        r_ans = answers[np.argmax(scores)]
        best_results.append(r_ans)

    # 保存最终的结果
    save_file(best_output_file, best_results)
    print('Done!')
