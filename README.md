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