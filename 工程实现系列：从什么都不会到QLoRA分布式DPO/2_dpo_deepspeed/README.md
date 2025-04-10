# 完整的基于DeepSpeed和trl的DPO训练

## 区别

和1_dpo_deepspeed相比，只多了3行代码

```python
local_rank = int(os.getenv('LOCAL_RANK', '0'))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

```

`local_rank` 是当前进程的本地rank，`torch.cuda.set_device(local_rank)` 设置当前进程的GPU设备，`deepspeed.init_distributed()` 初始化分布式训练。
其中`deepspeed.init_distributed()` 是DeepSpeed的分布式训练初始化函数，需要传入一个配置文件，这里我们使用默认的配置文件。
