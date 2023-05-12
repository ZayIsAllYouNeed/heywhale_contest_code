import csv
import random
import os
from predict import inference
'''
本文件是一个基本的demo，随机生成N条10-999之间的数据作为伪结果文件。
N 应该和标准结果中要求的数据行数相同。
'''
min_num = 10
max_num = 999

# 定义每条数据的长度。行数应该和测试输入文件一致
line_length = 5


def invoke(input_data_path, output_data_path):
    print('hello^^^^')
    model_path='./my_model'
    #'/nlp/zhengayong/heywhale/CPT/finetune/generation_8/output/heywhale/64/epoch-12.0'
    # './my_model'#'/nlp/zhengayong/heywhale/project_0_problem/my_model'#64/epoch-12.0
    print(model_path)
    inference(model_path, input_data_path, output_data_path)
    print('#'*200)
    return

    # 从输入地址 'input_data_path'读入待测试数据
    num_lines_gen = 0
    with open(input_data_path, 'r') as file:
        # 注意提供的csv文件没有表头（header）
        # ...
        reader = csv.reader(file)
        num_lines_gen = len(list(reader))

    # 生成预定义数据样本
    numbers = [[random.randint(min_num, max_num) for _ in range(line_length)] for _ in range(num_lines_gen)]

    # 将选手程序生成的结果写入目的地地址文件中
    with open(output_data_path, 'w') as file:
        writer = csv.writer(file)
        for idx, number in enumerate(numbers):
            writer.writerow([idx, ' '.join(map(str, number))])
    

if __name__ == '__main__':
    # invoke('./data/test.csv', './data/pred.csv')
    pass