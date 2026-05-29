"""KoCulture QLoRA 파인튜닝 (Colab T4 권장)
실행: python scripts/train.py
"""
import torch
from datasets import load_dataset
from datasets.builder import VerificationMode
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID   = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./output/koculture-lora-final"
HF_REPO    = "DdingDDing0103/koculture-qwen2.5-3b-lora"   # push_to_hub 대상

# 1) 데이터 로드 & 전처리 (9:1 분할)
ds = load_dataset("huggingface-KREW/KoCulture-Dialogues", split="train",
                  verification_mode=VerificationMode.NO_CHECKS)
ds = ds.map(lambda ex: {"messages": [
        {"role": "user", "content": ex["question"]},
        {"role": "assistant", "content": ex["answer"]},
    ]}, remove_columns=ds.column_names)
ds = ds.train_test_split(test_size=0.1, seed=42)

# 2) 4-bit 양자화 모델 로드
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
model = prepare_model_for_kbit_training(model)

# 3) LoRA 설정
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)

# 4) 학습 설정
sft_config = SFTConfig(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
    bf16=False, fp16=True,                      # T4는 bf16 미지원
    optim="paged_adamw_8bit", gradient_checkpointing=True,
    logging_steps=20, save_strategy="steps", save_steps=200, save_total_limit=2,
    eval_strategy="epoch", max_seq_length=512, packing=False,
    report_to="none", seed=42,
)

# 5) 학습 & 저장
trainer = SFTTrainer(model=model, args=sft_config,
                     train_dataset=ds["train"], eval_dataset=ds["test"],
                     peft_config=lora_config, tokenizer=tokenizer)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 6) (선택) HF Hub 업로드
# trainer.model.push_to_hub(HF_REPO); tokenizer.push_to_hub(HF_REPO)
print("✅ 완료:", OUTPUT_DIR)