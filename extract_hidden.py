
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# ── 配置 ──────────────────────────────────────────────────
MODEL_PATH = "/opt/models/Llama-3-8B"
SAVE_DIR   = "./results"
DEVICE     = "cuda:0"
MAX_SAMPLES = 800   # TruthfulQA共817条，全跑
BATCH_SIZE  = 1
# ─────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map=DEVICE,
    output_hidden_states=True,
)
model.eval()
print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

# ── 加载TruthfulQA ────────────────────────────────────────
print("Loading TruthfulQA...")
dataset = load_dataset("truthful_qa", "generation", split="validation")
dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
print(f"Samples: {len(dataset)}")

# ── 提取hidden states ─────────────────────────────────────
# 对每条样本，分别提取：
#   - question本身（中性）
#   - best_answer（事实正确）
#   - incorrect_answers[0]（幻觉）
# 取最后一个token的所有层hidden states

def get_hidden_states(text, max_length=128):
    """返回 shape: [num_layers+1, hidden_size]，取最后token"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (batch, seq_len, hidden_size), len = num_layers+1
    # 取最后一个token位置
    hidden = torch.stack([h[0, -1, :] for h in outputs.hidden_states])
    # shape: [num_layers+1, hidden_size]
    return hidden.float().cpu().numpy()

records = []

print("Extracting hidden states...")
for i, sample in enumerate(tqdm(dataset)):
    question = sample["question"]
    correct  = sample["best_answer"]
    wrongs   = sample["incorrect_answers"]

    if not correct or not wrongs:
        continue

    wrong = wrongs[0]

    # 构造prompt格式
    q_text       = f"Q: {question}\nA:"
    correct_text = f"Q: {question}\nA: {correct}"
    wrong_text   = f"Q: {question}\nA: {wrong}"

    try:
        h_correct = get_hidden_states(correct_text)  # [33, 4096]
        h_wrong   = get_hidden_states(wrong_text)    # [33, 4096]

        records.append({
            "id"       : i,
            "question" : question,
            "correct"  : correct,
            "wrong"    : wrong,
            "h_correct": h_correct,   # factual
            "h_wrong"  : h_wrong,     # hallucination
        })
    except Exception as e:
        print(f"Sample {i} error: {e}")
        continue

print(f"\nExtracted {len(records)} valid samples")

# ── 保存 ──────────────────────────────────────────────────
save_path = os.path.join(SAVE_DIR, "hidden_states.npz")

h_correct_all = np.stack([r["h_correct"] for r in records])  # [N, 33, 4096]
h_wrong_all   = np.stack([r["h_wrong"]   for r in records])  # [N, 33, 4096]

np.savez_compressed(
    save_path,
    h_correct = h_correct_all,
    h_wrong   = h_wrong_all,
)

# 保存文本元信息
meta = [{"id": r["id"], "question": r["question"],
         "correct": r["correct"], "wrong": r["wrong"]} for r in records]
with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Saved to {save_path}")
print(f"Shape: correct={h_correct_all.shape}, wrong={h_wrong_all.shape}")