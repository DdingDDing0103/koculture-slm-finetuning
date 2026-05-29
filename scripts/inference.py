"""학습된 KoCulture 어댑터로 추론
실행: python scripts/inference.py "친구가 게임에서 봉산탈춤 춘대"
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE    = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "DdingDDing0103/koculture-qwen2.5-3b-lora"   # HF Hub에서 자동 다운로드

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE)
base  = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def chat(prompt, max_new_tokens=200):
    msgs = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, return_tensors="pt", add_generation_prompt=True).to(model.device)
    out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "친구가 게임에서 봉산탈춤 춘대"
    print(chat(q))