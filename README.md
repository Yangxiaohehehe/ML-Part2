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

