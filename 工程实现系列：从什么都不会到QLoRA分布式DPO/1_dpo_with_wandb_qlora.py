from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import wandb
import torch
from datetime import datetime


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,  # 自动选择计算类型
    bnb_4bit_use_double_quant=True,
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"


# wandb
date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb.init(project="qwen", name=f"{model_name}_{date_time}")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config, use_cache=False)
try:    
    print(model.model.layers[0].self_attn.q_proj.weight.dtype)
    print(model.model.layers[0].self_attn.q_proj.weight.quant_state)
    print(model.model.layers[0].self_attn.q_proj.weight.quant_type)
    print(type(model.model.layers[0].self_attn.q_proj)) 
except:
    print("no quant")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 量化模型的工具方法，确保模型是量化后，且配置正确
# model = prepare_model_for_kbit_training(model)

# Lora 设置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "o_proj", "v_proj", "gate_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# 应用 Lora
model = get_peft_model(model, lora_config)

# for name, param in model.named_parameters():
#     if "lora" in name:
#         print(f"{name}: requires_grad={param.requires_grad}")

# 加载数据集
dataset = load_dataset("phimes/DPO-bad-boy-chinese-for-Qwen2.5-extended")


train_dataset = dataset["train"]
test_dataset = dataset["test"]

# 训练配置
training_args = DPOConfig(
    output_dir=f"./output", # 输出目录
    
    num_train_epochs=3, # 训练轮数，1轮
    
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    
    gradient_accumulation_steps=4, 
    gradient_checkpointing=True, 
    
    learning_rate=1e-5, 
    
    eval_strategy="steps", 
    eval_steps=20, 
    
    optim="paged_adamw_8bit",
    
    logging_steps=20, 
    
    bf16=False,
    fp16=True,
    
    report_to=["wandb"],         # 启用wandb记录，其他还可以tensorboard等工具，但是这里只用wandb。
    run_name=f"{model_name}_{date_time}",     # wandb运行名称，将保存在本地，和init的name可以不一致,
    
)

dpo_trainer = DPOTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    args=training_args,    
)


if __name__ == "__main__":
    try:
        dpo_trainer.train()
    except Exception as e:
        print(e)
    finally:
        wandb.finish()
