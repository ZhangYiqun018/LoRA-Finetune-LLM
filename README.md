# LoRA-Finetune-LLM

一个用LoRA方式训练large language model的方案，基于huggingface的框架。

目前存在的问题:

- [ ] 开lora时并行训练有问题
- [x] train, evaluate时显存爆炸
    设置`predict_with_generate = True`可以解决这个问题
- [ ] evaluate模式不能到底
  - [ ] `IndexError: piece id is out of range. `暂时不知道错误原因
- [x] 本地模型训练bug
  - [x] 本地训练时 use fast tokenizer

TODO LIST:
- [ ] 测试8bit训练
- [ ] 完成test部分
- [ ] 完成metrics部分 


实验记录
- a6000(单卡)
  - small
  - base
  - lagre
    - batch size 8
    - accumulation 4
    - lr 5e-5
    - 8bit false
    - 显存 42
  - xl
    - batchsize 2
    - accumulation 16 
    - lr 5e-5
    - 8bit false
  - xxl
    - batchsize 8
    - accumulation 4
    - lr 5e-5
    - 8bit true