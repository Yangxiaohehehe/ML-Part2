# ML part2

主要三部分 数据预处理 训练 评估

## 数据预处理
将 Math500 数据集中的“问题”和“解答”格式化为 Qwen3 模型能识别的对话格式。
使用 Qwen3 提供的 tokenizer 对格式化后的数据进行 tokenize。
对 prompt 部分进行 mask（不计算 loss），只对 assistant 的解答部分计算 loss。


## 训练
三种优化器训练
完善loss图等

finetune.py


## 评估
补全 rollout.py 中的模板部分（与数据预处理一致）

在 500 条测试数据上的得分
比较前后的得分 

评估过程中分别测试了不同maxtokens 以及 温度的效果实验


### 评估部分的内容
out_jsonl是利用rollout运行完的答案
out_score是利用eval文件跑出分数后的jsonl结果

更改了eval文件可以读取整个文件夹并生成文件夹结果：eval.py


### 评估过程中问题报错
各种包的问题 需要根据缺什么 去github找到对应的 pip即可
