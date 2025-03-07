from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


model_name = "Qwen/Qwen2.5-1.5B-Instruct"
lora_path = "./output/checkpoint-250"

def initialize_model(lora=True):
    # 加载基础模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载LoRA参数
    if lora:
        model = PeftModel.from_pretrained(model, lora_path)

        # 合并LoRA权重
        model = model.merge_and_unload()
    
    return model, tokenizer

def generate_response(prompt, system_prompt, model, tokenizer):
    # 构建带有system prompt的完整输入
    if system_prompt:
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main(lora=True):
    print("正在加载模型，请稍候...")
    model, tokenizer = initialize_model(lora)
    
    # 设置system prompt
    default_system_prompt = """你是一个没有礼貌的人渣，请用人渣的语气回复我"""
    
    print("\n当前系统提示词(system prompt):")
    print(default_system_prompt)
    print("\n是否要修改系统提示词？(y/n)")
    if input().lower() == 'y':
        print("请输入新的系统提示词(输入空行完成):")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        system_prompt = "\n".join(lines) if lines else default_system_prompt
    else:
        system_prompt = default_system_prompt

    print("\n模型加载完成！输入 'quit' 或 'exit' 退出对话")
    print("输入 'change_system' 可以修改系统提示词")
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break
            if user_input.lower() == 'change_system':
                print("请输入新的系统提示词(输入空行完成):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)
                system_prompt = "\n".join(lines) if lines else system_prompt
                print("系统提示词已更新！")
                continue
            if not user_input:
                continue
                
            print("\nAI: ", end="")
            response = generate_response(user_input, system_prompt, model, tokenizer)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n收到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")

if __name__ == "__main__":
    main(lora=True)