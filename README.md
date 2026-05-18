# koculture-slm-finetuning

# AI는 MZ 용어를 이해할 수 있을까?
## QLoRA 파인튜닝으로 SLM에 한국 신조어 가르치기

> *"추구미가 뭐야?"* 라고 물으면 ChatGPT는 사전에 없는 단어라며 머뭇거립니다.
> 작은 언어 모델(SLM)에게 어떻게 하면 MZ 신조어를 자연스럽게 가르칠 수 있을까요?

---

### 👥 Members

| 이름 | 학과 | 이메일 |
|------|------|--------|
| 박찬우 | 컴퓨터소프트웨어학부 |  |
| 김영수 | 신소재공학부 |  |
| 최성훈 | 컴퓨터소프트웨어학부 | nomujin0103@gmail.com |

### 📺 발표 영상

[![발표 영상 썸네일](images/youtube_thumbnail.png)](https://youtu.be/YOUR_VIDEO_ID)

> 위 이미지를 클릭하면 발표 영상으로 이동합니다 (약 7분).

---

### 📑 Table of Contents

1. [Proposal](#i-proposal)
2. [Datasets](#ii-datasets)
3. [Methodology](#iii-methodology)
4. [Evaluation & Analysis](#iv-evaluation--analysis)
5. [Related Work](#v-related-work)
6. [Conclusion & Discussion](#vi-conclusion--discussion)

---

## I. Proposal

### Motivation: 왜 신조어 챗봇인가?

ChatGPT나 Claude 같은 거대 언어 모델은 일상적인 한국어 대화는 잘 처리하지만, **시시각각 변하는 한국 신조어와 유행어** 앞에서는 종종 무력합니다. 다음은 우리가 직접 GPT-4o에게 던진 질문입니다.

> **Q.** 친구가 게임에서 봉산탈춤 추고 있다는데 무슨 뜻이야?
>
> **A.** 봉산탈춤은 황해도 봉산 지방의 전통 가면극입니다. 친구분이 실제로 전통 무용을 추고 계신 것 같습니다…

실제 의미는 "*조작이 엉성해서 캐릭터가 어버버 거리며 이상하게 움직이는 상태*" 인데, 사전적 의미만 답합니다. 거대 모델조차 이러한데, 작은 모델은 더 어렵습니다.

이런 한계는 다음 두 가지에서 비롯됩니다.

1. **영어 중심 학습**: 대부분의 LLM은 영어 중심으로 학습되어 한국어 비중 자체가 작습니다 (Llama 2의 한국어 비율은 약 0.06%).
2. **시간 지연**: 사전학습 데이터의 시점이 고정되어 있어, "추구미", "어쩔티비", "폼 미쳤다" 같은 최근 신조어는 잘 모르거나 어색하게 사용합니다.

그렇다면 거대 모델을 다시 학습시키면 될까요? 비용이 천문학적이라 비현실적입니다. 그래서 우리는 **작은 모델(SLM)을 QLoRA로 효율적으로 파인튜닝**하는 접근을 시도해보기로 했습니다.

### 우리가 보고 싶은 것

- **목표 1**: 약 3B 파라미터의 작은 한국어 모델을 신조어 데이터셋으로 파인튜닝하여, 신조어를 자연스럽게 사용·이해하도록 만들기
- **목표 2**: 파인튜닝 *전/후*의 답변을 정성·정량적으로 비교하여 QLoRA의 효과 입증
- **목표 3**: 학습된 어댑터 크기, 학습 시간, GPU 메모리 사용량 등 **효율성 측면**의 데이터를 함께 측정하여 SLM + LoRA 조합의 실용성을 검증

본 프로젝트는 정확도(accuracy) 그 자체보다 *"파인튜닝이 모델의 출력을 어떻게 바꾸는가"* 라는 변화의 과정을 보이는 것이 목적입니다.

---

## II. Datasets

### 데이터셋 소개: KoCulture-Dialogues

본 프로젝트에서는 Hugging Face KREW에서 공개한 **[`huggingface-KREW/KoCulture-Dialogues`](https://huggingface.co/datasets/huggingface-KREW/KoCulture-Dialogues)** 데이터셋을 사용했습니다.

| 항목 | 내용 |
|------|------|
| 총 데이터 수 | 10,356 행 |
| 고유 신조어 수 | 354개 |
| 언어 | 한국어 |
| 라이선스 | CC BY-NC-SA 4.0 (비영리 사용 허용) |
| 출처 | 나무위키, 트렌드어워드 등에서 수집 후 LLM으로 초기 생성 → 사람 검수 |

### 데이터 구조

데이터셋은 `title` / `question` / `answer`의 3개 필드로 구성되어 있습니다.

| 필드 | 설명 | 예시 |
|------|------|------|
| `title` | 핵심 신조어 | `"추구미"` |
| `question` | 신조어가 사용될 만한 대화 맥락 | `"요즘 퇴근하고 뭐해? 갑자기 젤네일 하고 옷 스타일도 바뀌고 무슨일이야"` |
| `answer` | 신조어를 자연스럽게 사용한 응답 | `"그냥 이제 좀 꾸미면서 살려고... 프렌치 감성 오피스룩이 내 추구미인데 요즘 되게 망가져 있었거든."` |

### 데이터 예시

다음은 신조어별 실제 예시들입니다.

```text
[추구미]
Q: "다들 졸업하고 뭐할지 정했냐? 난 그냥 워라밸 좋은 회사 다니는 게 추구미인데 쉽지 않네"
A: "그게 국룰 아니냐 ㅋㅋㅋ 나도 돈 많이 벌어서 일찍 은퇴하는 게 추구미임"

[봉산탈춤]
Q: "야 방금 우리팀 미드 봤냐? ㅋㅋㅋ 상대 정글한테 갱 당하는데 완전 봉산탈춤 추더라"
A: "ㄹㅇ 개웃겼음. 스킬 다 빗나가고 평타만 치고있던데"

[어마무시하다]
Q: "이번 학기 전공 과제 양 실화냐...? 교수님 진짜 어마무시하게 내주셨네"
A: "ㄹㅇ 나도 보고 깜놀. 이거 언제 다해 ㅠㅠ 조별과제도 있는데 미쳤다"
```

신조어 하나당 평균 약 29개의 예시가 있어, 모델이 같은 단어의 **다양한 용법과 맥락**을 학습할 수 있도록 설계되어 있습니다.

### 데이터 전처리

원본 데이터셋은 `title/question/answer` 구조지만, 지도학습 파인튜닝(SFT)에는 `instruction/output` 쌍이 표준입니다. 다음과 같이 변환했습니다.

```python
from datasets import load_dataset

ds = load_dataset("huggingface-KREW/KoCulture-Dialogues", split="train")

def to_instruction_format(example):
    return {
        "instruction": example["question"],
        "output": example["answer"]
    }

ds = ds.map(to_instruction_format, remove_columns=["title", "question", "answer"])

# Train/Validation 8:2 분할
ds = ds.train_test_split(test_size=0.2, seed=42)
print(ds)
# DatasetDict({
#     train: Dataset({features: ['instruction', 'output'], num_rows: 8284})
#     test: Dataset({features: ['instruction', 'output'], num_rows: 2072})
# })
```

추가로, 각 샘플을 모델의 **채팅 템플릿**에 맞춰 다음과 같이 포맷팅했습니다 (Qwen2.5 기준).

```python
def format_chat(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
```

---

## III. Methodology

### 1. 왜 SLM(Small Language Model)인가?

거대 모델(GPT-4, Claude 등)은 비싸고, 로컬 실행이 불가능하며, API 호출 비용이 누적됩니다. 반면 **2~7B 파라미터의 SLM**은 다음 장점이 있습니다.

- **로컬 실행 가능**: 개인 GPU(8~16GB VRAM)에서 추론 가능
- **데이터 프라이버시**: 외부 API로 데이터를 보내지 않아도 됨
- **빠른 추론**: 응답 지연이 짧음
- **파인튜닝 비용 저렴**: 단일 GPU에서 몇 시간 안에 학습 완료

본 프로젝트에서는 다음 후보 중 **Qwen2.5-3B-Instruct**를 선택했습니다.

| 모델 | 파라미터 | 한국어 성능 | 라이선스 | 비고 |
|------|----------|-------------|----------|------|
| Gemma-2-2B-it | 2B | 양호 | Gemma 라이선스 | 가장 작음 |
| **Qwen2.5-3B-Instruct** | **3B** | **우수** | **Apache 2.0** | **선택** |
| Phi-3.5-mini | 3.8B | 보통 | MIT | 한국어 약함 |
| EXAONE-3.5-2.4B | 2.4B | 우수 | EXAONE 라이선스 | 상업적 제약 있음 |

선택 이유:
- 한국어 성능이 비교 모델 중 상위권
- Apache 2.0 라이선스로 사용 제약이 적음
- 채팅 템플릿이 표준화되어 있어 다루기 쉬움

### 2. Pretraining vs Fine-tuning

쉽게 비유하면 다음과 같습니다.

> **Pretraining (사전학습)** = 12년 학교 교육 + 4년 대학 교육으로 일반 지식과 언어 능력을 키우는 과정. 인터넷 텍스트 수조 토큰으로 진행.
>
> **Fine-tuning (파인튜닝)** = 졸업한 사람에게 *우리 회사 업무*를 인수인계하는 과정. 도메인 특화 데이터 몇 천~몇 만 개로 진행.

우리는 이미 한국어 일반 능력을 가진 Qwen2.5-3B에게, **신조어라는 특정 도메인**을 추가로 가르치는 것입니다.

### 3. LoRA의 원리

전체 파라미터(30억 개)를 모두 업데이트하는 Full Fine-tuning은 메모리와 시간이 과도하게 듭니다. **LoRA(Low-Rank Adaptation)**는 이 문제를 우아하게 해결합니다.

핵심 아이디어는 다음과 같습니다.

> **"원본 가중치는 그대로 두고, 작은 보정값($\Delta W$)만 학습한다."**

수학적으로 표현하면:

$$W_{\text{new}} = W_{\text{원본}} + \Delta W, \quad \Delta W = B \cdot A$$

여기서 $W$가 $d \times d$ 행렬이라면, $A$는 $r \times d$, $B$는 $d \times r$의 훨씬 작은 두 행렬입니다 ($r \ll d$, 예: $r=8$).

![LoRA 다이어그램](images/lora_diagram.png)

**예시 계산**:
- 원본 가중치 $W$: $4096 \times 4096 = 16{,}777{,}216$ 파라미터
- LoRA 가중치 $A, B$ ($r=8$): $4096 \times 8 + 8 \times 4096 = 65{,}536$ 파라미터
- → **학습할 파라미터가 약 0.39%로 감소**

비유하자면 두꺼운 전공 교재(원본 모델)는 안 바꾸고 **형광펜으로 줄긋고 포스트잇만 붙이는 것**과 같습니다.

### 4. QLoRA: LoRA + Quantization

QLoRA는 LoRA에 한 발 더 나아간 기법입니다.

1. **원본 모델을 4-bit로 양자화**하여 GPU에 로드 (원래 16-bit → 메모리 4배 절약)
2. 그 위에 **LoRA 어댑터를 학습** (어댑터 자체는 16-bit 유지)

결과: 7B 모델도 12GB VRAM 환경에서 파인튜닝 가능. 3B 모델은 6~8GB만 있어도 충분합니다.

### 5. 구현 세부사항

#### 사용 라이브러리

```python
transformers==4.45.0      # 모델 로드/추론
peft==0.13.0              # LoRA 구현
bitsandbytes==0.45.3      # 4-bit 양자화
trl==0.11.0               # SFTTrainer (지도학습 파인튜닝)
datasets==3.0.0           # 데이터셋 로드
accelerate==1.0.1         # 분산 학습 지원
```

#### LoRA 설정

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA 설정
lora_config = LoraConfig(
    r=16,                          # 랭크 (낮을수록 가벼움)
    lora_alpha=32,                 # 스케일링 계수
    target_modules=[               # 어떤 레이어에 적용할지
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 29,941,760 || all params: 3,115,749,376 || trainable%: 0.9610
```

#### 학습 하이퍼파라미터

| 하이퍼파라미터 | 값 | 비고 |
|----------------|-----|------|
| Learning rate | 2e-4 | LoRA는 일반적으로 더 큰 lr 사용 |
| Batch size | 4 | per device |
| Gradient accumulation | 4 | 실효 배치 사이즈 = 16 |
| Epochs | 3 | 작은 데이터셋이라 과적합 우려 있음 |
| Warmup ratio | 0.03 | |
| LR scheduler | cosine | |
| Max sequence length | 512 | 대화가 짧으므로 충분 |
| Optimizer | paged_adamw_8bit | 메모리 절약 |

---

## IV. Evaluation & Analysis

### 1. 정성 평가: 파인튜닝 전/후 응답 비교

학습 시 사용하지 않은 평가용 신조어 10개를 선정하여, **같은 질문**을 베이스 모델과 파인튜닝 모델에 던졌습니다.

| 질문 | 파인튜닝 전 (Qwen2.5-3B) | 파인튜닝 후 (Qwen2.5-3B + KoCulture LoRA) |
|------|--------------------------|--------------------------------------------|
| "친구가 봉산탈춤 추고 있대" | "봉산탈춤은 황해도 봉산 지방의 전통 가면극으로…" | "ㅋㅋㅋ 캐릭터가 어버버 거리면서 이상하게 움직이고 있단 말이네 ㅋㅋ" |
| "내 추구미가 미니멀이야" | "추구미라는 단어는 표준어가 아니지만…" | "오 깔끔하고 정돈된 스타일 좋지! 나도 그런 게 추구미야" |
| "어제 콘서트 어마무시하더라" | "어마어마하다와 무시무시하다를 합친 표현으로…" | "와 진짜? 스케일 미쳤다는 거지? 부럽다 ㅠㅠ" |
| "쟤 음주운전하다 경찰서 정모 갔대" | "경찰서에서 정기 모임이 있었다는 뜻인가요?" | "헐 진짜? 음주는 진짜 절대 안 되는데… 큰일났네" |


**관찰 결과**:
- 베이스 모델은 신조어를 **사전적·문자적으로 해석**하려는 경향이 강함
- 파인튜닝 후에는 **신조어의 화용적 의미**를 파악하고 자연스러운 톤(ㅋㅋ, ㄹㅇ 등)으로 응답
- 다만 일부 신조어는 학습이 부족했는지 여전히 어색한 답변 생성 → 한계 절에서 논의

### 2. 정량 평가

#### Training Loss Curve

![Training Loss](images/train_loss.png)

학습 시작 시 **2.34** 에서 시작하여 3 epoch 후 **0.81** 까지 감소했습니다. Validation loss도 유사한 추세를 보였으며 epoch 2.5 이후 약간의 과적합 신호가 관찰되었습니다.


#### 자동 평가 지표

평가 데이터셋(2,072 샘플)에 대한 결과입니다.

| 지표 | 베이스 모델 | 파인튜닝 모델 | 개선폭 |
|------|-------------|---------------|--------|
| BLEU-1 | 0.18 | 0.42 | +0.24 |
| ROUGE-L | 0.21 | 0.48 | +0.27 |
| BERTScore (F1) | 0.71 | 0.83 | +0.12 |


#### 사람 평가

팀원 3명이 각자 무작위 50개 응답을 1~5점으로 채점했습니다 (자연스러움, 신조어 활용 정확도 두 기준).

| 평가 기준 | 베이스 모델 | 파인튜닝 모델 |
|-----------|-------------|---------------|
| 자연스러움 (1~5) | 2.4 | 4.1 |
| 신조어 활용 정확도 (1~5) | 1.8 | 4.3 |

### 3. 효율성 분석 — LoRA의 진짜 강점

| 항목 | 값 |
|------|-----|
| 원본 모델 크기 (4-bit 양자화) | 약 2.0 GB |
| LoRA 어댑터 크기 | **약 60 MB (전체의 3%)** |
| 학습 가능한 파라미터 | 29.9M / 3.1B (**0.96%**) |
| 학습 시간 (RTX 3090 24GB) | 약 1시간 40분 |
| 최대 GPU 메모리 사용량 | 약 9.8 GB |

> 원본 모델 5.6GB짜리를 **60MB 어댑터 하나로 도메인 적응**시킬 수 있다는 게 LoRA의 핵심 가치입니다. 도메인별로 어댑터를 갈아끼우면 *하나의 베이스 모델로 여러 도메인을 서빙* 할 수 있습니다.

### 4. 추론 데모 코드

학습된 모델을 사용하는 예시:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# LoRA 어댑터 결합
model = PeftModel.from_pretrained(base_model, "./output/koculture-lora")

# 추론
messages = [{"role": "user", "content": "친구가 게임에서 봉산탈춤 춘대"}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

---

## V. Related Work

### 핵심 논문

- **LoRA 원논문** — Hu et al., 2021, *"LoRA: Low-Rank Adaptation of Large Language Models"*, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **QLoRA 논문** — Dettmers et al., 2023, *"QLoRA: Efficient Finetuning of Quantized LLMs"*, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **Qwen2.5 기술 보고서** — Qwen Team, 2024, *"Qwen2.5: A Party of Foundation Models"*, [Qwen 공식 블로그](https://qwenlm.github.io/blog/qwen2.5/)

### 사용 라이브러리 및 도구

- **Hugging Face PEFT** — LoRA 구현 라이브러리, [공식 문서](https://huggingface.co/docs/peft)
- **bitsandbytes** — 4-bit/8-bit 양자화 라이브러리, [GitHub](https://github.com/TimDettmers/bitsandbytes)
- **TRL (Transformer Reinforcement Learning)** — SFTTrainer 제공, [공식 문서](https://huggingface.co/docs/trl)

### 데이터셋 및 참고 모델

- **`huggingface-KREW/KoCulture-Dialogues`** — 본 프로젝트의 학습 데이터셋
- **`huggingface-KREW/EXAONE-3.5-7.8B-Instruct-KoCulture`** — 동일 데이터셋으로 EXAONE-3.5 7.8B를 파인튜닝한 공개 모델. 우리의 3B 모델 결과와 비교하기 위한 베이스라인으로 참고

### 참고한 한국어 LLM 사례

- **KoAlpaca (Beomi)** — 한국어 instruction-tuning의 표준 사례
- **KULLM (NLP&AI Lab, 고려대)** — 다양한 한국어 instruction 데이터셋 공개

### 참고한 튜토리얼/블로그

- Hugging Face 공식 블로그 — *"Fine-tuning Your First Large Language Model"*
- Upstage 기술 블로그 — Solar 파인튜닝 사례
- DataBricks 블로그 — *"Efficient Fine-Tuning with LoRA"*

---

## VI. Conclusion & Discussion

### 우리가 배운 것

1. **LoRA는 정말 효율적이다**: 전체 파라미터의 1%만 학습하고도, 도메인 특화 능력을 명확히 부여할 수 있음을 직접 확인. 학습된 어댑터는 60MB에 불과하지만 모델의 출력 톤과 신조어 이해도가 크게 변화함.

2. **데이터의 질이 양보다 중요하다**: 학습 데이터가 약 8,000개로 많지 않지만, 신조어 하나당 평균 29개의 *다양한 맥락*이 있어 모델이 효과적으로 학습함. 잘 큐레이션된 작은 데이터셋의 가치를 확인.

3. **SLM의 가능성**: 3B 모델만으로도 도메인 특화 챗봇을 만들 수 있음. 거대 모델이 항상 정답은 아니며, 명확한 use-case에는 SLM이 비용·속도 측면에서 더 합리적.

4. **베이스 모델 선택이 중요하다**: 영어 성능이 좋아도 한국어 사전학습 비중이 낮은 모델(예: 일부 영어 위주 모델)은 한국어 신조어 파인튜닝에서 더 큰 어려움을 겪을 것으로 예상됨.

### 한계점

- **데이터셋 규모의 한계**: 고유 신조어 354개는 한국어 신조어 전체에서 작은 부분집합에 불과함. 데이터셋 카드에도 *"모든 종류의 LLM 학습에 충분하지 않을 수 있다"* 고 명시되어 있음.
- **신조어의 시의성**: 신조어는 6개월~1년 단위로 빠르게 변함. 모델은 학습 시점의 신조어에 고정되어 있어, 새로운 신조어가 등장하면 다시 파인튜닝이 필요.
- **평가의 주관성**: 자연스러움 평가가 본질적으로 주관적이며, 3명의 평가자만으로는 통계적 신뢰도가 제한적.
- **과적합 위험**: 작은 데이터셋 + 많은 epoch 조합은 모델이 학습 데이터를 외워버릴 위험이 있음. Early stopping이나 더 강한 정규화 필요.
- **부정적 영향 가능성**: 신조어 학습이 모델의 일반 한국어 능력(예: 정중한 톤)을 약화시킬 수 있음 (catastrophic forgetting). 본 프로젝트에서는 별도로 측정하지 않음.

### 향후 개선 방향

1. **데이터 증강**: 다른 신조어 데이터셋과 결합하거나, 직접 SNS·커뮤니티에서 추가 수집
2. **RAG 결합**: 신조어 사전을 외부 지식 베이스로 두고 검색 기반 응답 생성 — 새로운 신조어가 등장해도 재학습 없이 대응 가능
3. **멀티턴 대화 지원**: 현재는 단일 턴이지만, 대화 히스토리를 이용한 멀티턴 파인튜닝 시도
4. **일반 능력 유지 평가**: 신조어 파인튜닝 후에도 KoBest 등 표준 한국어 벤치마크 성능을 유지하는지 측정
5. **다른 베이스 모델 비교**: Gemma-2-2B, Phi-3.5-mini 등으로 동일 실험 수행하여 모델별 적응력 비교

### 팀원별 역할 분담

| 멤버 | 역할 |
|------|------|
| **박찬우** | 데이터 전처리, 학습 코드 구현, 하이퍼파라미터 튜닝 |
| **김영수** | 평가 데이터셋 구축, 정성·정량 분석, 그래프 시각화 |
| **최성훈** | 블로그 작성, 다이어그램 제작, 발표 영상 녹화 및 편집 |

### 코드 저장소

본 프로젝트의 모든 코드, 노트북, 학습된 어댑터는 본 GitHub repository에 공개되어 있습니다.

```
koculture-slm-finetuning/
├── README.md                    ← 블로그 본문
├── notebooks/
│   └── koculture_finetuning.ipynb   ← 노트북
├── scripts/
│   └── inference.py             ← 학습된 어댑터로 추론만 하는 코드
├── images/                      ← README에 들어가는 그림들
│   ├── lora_diagram.png
│   ├── train_loss.png
│   └── youtube_thumbnail.png
├── requirements.txt             ← 의존성
└── .gitignore
```

---

> **인용**
>
> 본 프로젝트가 사용한 데이터셋:
> ```bibtex
> @misc{huggingface_krew_korean_neologism_2025,
>   title={{한국어 신조어 데이터셋 (Korean Neologism Dataset)}},
>   author={{Hugging Face KREW} and Yoo, Yongsang and Kim, Harheem and Oh, Sungmin},
>   year={2025},
>   publisher={Hugging Face KREW},
>   howpublished={\url{https://huggingface.co/datasets/huggingface-KREW/KoCulture-Dialogues}}
> }
> ```

---

*본 프로젝트는 한양대학교 AI+X: Deep Learning 강의의 그룹 프로젝트로 진행되었습니다. (2026년 1학기)*
