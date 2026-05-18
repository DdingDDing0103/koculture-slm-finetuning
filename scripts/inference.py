# scripts/inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "./output/koculture-lora-final"  # 또는 HF Hub 경로

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

def chat(prompt, max_new_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

if __name__ == "__main__":
    print(chat("친구가 게임에서 봉산탈춤 추고 있다는데 뭔 뜻이야?"))