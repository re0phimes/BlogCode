from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import wandb
import torch
from datetime import datetime
import os
import deepspeed


local_rank = int(os.getenv('LOCAL_RANK', '0'))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,  # 自动选择计算类型
    bnb_4bit_use_double_quant=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"



# wandb
date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb.init(project="qwen", name=f"{model_name}_{date_time}")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config)
try:    
    print(model.model.layers[0].self_attn.q_proj.weight.dtype)
    print(model.model.layers[0].self_attn.q_proj.weight.quant_state)
    print(model.model.layers[0].self_attn.q_proj.weight.quant_type)
    print(type(model.model.layers[0].self_attn.q_proj)) 
except:
    print("no quant")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备模型
model = prepare_model_for_kbit_training(model)

# Lora 设置
LoraConfig = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "o_proj", "v_proj", "gate_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# 应用 Lora
model = get_peft_model(model, LoraConfig)


# 加载数据集
dataset = load_dataset("phimes/DPO-bad-boy-chinese-for-Qwen2.5-extended")

train_dataset = dataset["train"]
test_dataset = dataset["test"]

# 训练配置
training_args = DPOConfig(
    output_dir=f"./output",
    
    num_train_epochs=1,
    
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    
    learning_rate=1e-5,
    
    evaluation_strategy="steps",
    eval_steps=200,
    
    optim="paged_adamw_8bit",
    
    logging_dir="./logs",
    logging_steps=100,
    
    bf16=False,
    fp16=True,
    
    report_to=["wandb"],
    run_name="my_first_dpo",
    
    deepspeed="./ds_config.json",
    
)

dpo_trainer = DPOTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    args=training_args,    
)


if __name__ == "__main__":
    dpo_trainer.train()
    wandb.finish()