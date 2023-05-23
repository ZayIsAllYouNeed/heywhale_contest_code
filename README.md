# heywhale_contest_code
- 模型文件置于路径[my_model](./project/my_model)下即可推理。
  - 如需多模型结果投票，可将另外 **2** 个以上 *bin* 文件模型（名字以"pytorch_model"起头，如"pytorch_model_1.bin"）置于路径[project/my_model/other_models](./project/my_model/other_models)下，即可自动多模型结果投票。
  - 如需半精度推理，只需要取消注释[predict.py](./project/predict.py)中的 *model.half()* 指令即可。
- 如需在本地测试，可以运行以下几行命令：
```
cd project
import index
index.invoke('./data/test.csv', './data/pred.csv')
```
# 提交审核README

# 代码说明  

## 环境配置  
本项目训练使用的算力资源型号为「单卡 V100 GPU 资源」，使用基于官方镜像生成的自定义镜像“啊啊啊+复现镜像”(提交审核的那个版本写成了"啊啊啊复现镜像"，🤦)
包括以下额外的安装包：  
- datasets==2.10.1  
- nltk==3.8.1  
- deepspeed==0.8.3  
- argparse==1.1  
- opencv-python==4.5.5.64  

## 预训练模型  
使用了“fnlp/bart-large”预训练模型，可以通过运行以下 linux 命令方式获得：  
```
# Make sure you have git-lfs installed (https://git-lfs.com)  
git lfs install  
git clone https://huggingface.co/fnlp/bart-large-chinese  
```
## 算法  

### 整体思路介绍（必选）  

- 预训练：预训练模型的嵌入层 **shared layer**（即所谓的 word embedding layer）大小进行调整，并对除 special tokens 以外的 tokens 对应的 embedding 向量进行随机初始化。然后，在初赛 + 复赛的数据上做随机 n gram mask 预训练。  
- 微调：采用 k fold 来交叉验证，训练得到 n (此处为3)个模型。使用一些 trick 提升模型的泛化性。  
- 推理：用微调阶段得到的 n 个模型进行推理得到各自的输出 sentence 结果，然后采用投票的方式抉择出最优的 sentence（详见“方法的创新点”节）。  

### 方法的创新点（可选）  
 - 在预训练阶段随机选择 **n gram tokens span** 进行 **mask**，每个被 mask 的 span 用 1 个 [MASK] 来代替，让模型根据上下文去从 1 个 [MASK] 中还原出原来的 n gram tokens span，从而学到生成能力。  
 - 使用 **R-Drop**(Regularized Dropout )来消除模型训练阶段和推理阶段的**GAP**：因为模型在推理阶段是关闭 dropout 的，与训练阶段不同，采用 R-Drop 可以使得模型对 dropout 的干扰更加鲁棒，从而在推理阶段保持较好的性能。  
 - 使用 **label smoothing** 使得模型不要过度自信，可以一定程度较少过拟合，提升泛化性。  
 - 使用 **FGM** (fast gradient method)对抗训练，在 finetune 阶段对嵌入层进行扰动，增加模型鲁棒性。  
 - 使用 **分层学习率**，增大嵌入层的学习率，让模型加入收敛。  
 - 多个模型结果 **投票**，在得到多个模型的推理结果后，让它们的结果之间互相投票：  
 > 1. 给定一条 input 数据，当前模型预测的 sentence 将分别以其他几个模型的预测的 sentence 作为伪 ground truth 来计算 BLEU_4 score，求和作为当前模型预测的 sentence 的分值。  
 > 2. 其他几个模型预测的 sentence 同样按步骤 1 方式计算自己的得分。  
 > 3. 最后取得分最高的 sentence 最为最后提交的结果。  
 - 推理时施加重复输出 token 惩罚，避免模型反复输出同样的 token；推理中防止 5 gram tokens 及更长的 n gram tokens 出现 2 次。  

### 算法的其他细节（可选）  
- 预训练必不可少：预训练不仅可以加速 finetune 阶段的收敛，而且可以提高 finetune 的指标上限  
- 预训练的 step 需要适中：太少会欠拟合，太多过拟合。经实验，采用本项目的 pretrain 代码，可以 1e-6 ～ 2e-5 learning rate、256 batch size 训练 10w steps，得到较不错的预训练参数  
- 一个可以提高数据训练利用率的方式：先划分出 dev 来 pretrain，根据 pretrain 过程中 dev 上的指标变化判断能够避免模型过拟合的合理 step 数 N，然后再用所有的数据 pretrain N steps 左右。同理，在 finetune 阶段也可以采用类似操作，从而避免模型在全量数据训练中过拟合。  
- 一个减少模型大小和显存占用的方法：经统计，初赛 + 复赛数据的总 token 数为 ***1628***，加上 6 个 special tokens，(e.g.: [CLS]、[EOS]、[MASK]、[PAD]、[SEP]、[UNK]) ，共 ***1634*** 个 token，将 预训练模型嵌入层也初始化为 ***1634***。该操作使得嵌入层向量数从 5w+ 减少为 1634。  

## 训练流程  
在「训练」notebook 中逐 cell 运行即可  
notebook 包括 变量设置、预处理数据、DEBUG阶段、正式训练阶段、保存模型阶段：  
> 变量设置：将 python 文件路径加入 os.path，从而让 notebook 来 import function  
> 预处理数据：将 初赛 + 复赛 数据处理为训练脚本需要的格式（包括将复赛数据中的 clinical 与 description 拼接），用于 k fold cross validation 以及 pretrain  
> DEBUG阶段：用于调试代码，**已被注释**，若需要，可以取消注释用以调试。（因为 10w 级数据量做文本生成用 V100 训练速度太慢，所以该环节使用少量的 debug 数据代替每个 fold 数据来分别训练 1 epoch，得到对应的模型，确保整个流程能够跑通）  
> 正式训练阶段：分别 finetune fold1、fold2、fold3 的数据，用于模型结果的投票  
> 保存模型阶段：将每个 fold 数据 finetune 出的的模型保存于 best_model 路径下对应的位置，用于推理阶段使用  

## 推理流程  
在「推理」notebook 中逐 cell 运行即可  
notebook 包括 变量设置、推理DEBUG阶段、正式推理阶段：  
> 变量设置：将 python 文件路径加入 os.path，从而让 notebook 来 import function  
> 推理DEBUG阶段：用于调试代码，**已被注释**，若需要，可以取消注释用以调试。（其中包括 debug 数据处理以及预测 debug 输入数据的结果，生成 debug results 文件 于 temp 临时空间下  ）  
> 正式推理阶段：调用训练流程保存的 n (此处为 3)个模型来推理输入的测试集的结果，然后对多个结果进行投票以得到最终的结果，输出到目标路径下。  
> - 细节 1：设置 no_repeat_ngram_size=5，在推理中让 5 gram 或以上的 span 最多重复 1 次。  
> - 细节 2：设置 repetition_penalty=1.2，施加 重复输出 惩罚，减小模型输出 token 的重复率。  

## 其他注意事项  
数据的划分需要按照 notebook 的流程来 逐cell 运行，其中已经设定好了 seed 以保证每次从头运行得到相同的数据划分结果。
```
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"]=os.environ["NVIDIA_VISIBLE_DEVICES"]
torch.cuda.get_device_name(0), torch.cuda.is_available()  # 确认 gpu 正常使用
temp_save_dir = '/home/mw/temp/'

import os
if not os.path.exists(temp_save_dir):  # 个别机器可能没有 temp 路径
    os.mkdir(temp_save_dir) # 生成用于保存另外几个fold的 best model 的路径
```