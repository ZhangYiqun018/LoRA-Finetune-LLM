# LoRA-Finetune-LLM

一个用LoRA方式训练large language model的方案，基于huggingface的框架。

目前存在的问题:

- [ ] 开lora时并行训练有问题
- [x] evaluate时显存爆炸
    设置`predict_with_generate = True`可以解决这个问题
- [ ] evaluate模式不能到底

TODO LIST:
- [ ] 测试8bit训练
- [ ] 完成test部分
- [ ] 完成metrics部分 
