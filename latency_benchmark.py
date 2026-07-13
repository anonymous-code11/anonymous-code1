
import os
import gc
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/opt/models/Llama-3-8B"
DEVICE      = "cuda:0"
BEST_LAYER  = 16
N_SAMPLES   = 200       # 评测问题数
N_REPEATS   = 3         # 每种方法重复次数
SCG_K_LIST  = [5, 10]  # SelfCheckGPT 的采样次数

SAVE_DIR = "./results/latency"
FIG_DIR  = "./figures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── 加载方向向量 ──────────────────────────────────────────────────────────────
print("Loading wfact and building full probe...")
data = np.load("./results/hidden_states.npz")
hc   = data["h_correct"]
hw   = data["h_wrong"]
N    = hc.shape[0]
X    = np.concatenate([hc[:, BEST_LAYER, :], hw[:, BEST_LAYER, :]])
y    = np.array([1]*N + [0]*N)

pca_model = PCA(n_components=128, random_state=42)
Xp = pca_model.fit_transform(X)
clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
clf.fit(Xp, y)

# Sfact 方向
w_fact_np = pca_model.components_.T @ clf.coef_[0]
w_fact_np = w_fact_np / (np.linalg.norm(w_fact_np) + 1e-8)
w_fact_t  = torch.tensor(w_fact_np, dtype=torch.float16)

# PCA components（用于 Full Probe online 部分）
pca_components = torch.tensor(pca_model.components_, dtype=torch.float32)  # [128, d]
clf_coef       = torch.tensor(clf.coef_[0], dtype=torch.float32)           # [128]
clf_intercept  = torch.tensor(clf.intercept_[0], dtype=torch.float32)      # []

print(f"wfact shape: {w_fact_np.shape}")

# ── 加载评测问题 ──────────────────────────────────────────────────────────────
print("Loading TruthfulQA questions...")
tqa = load_dataset("truthful_qa", "generation", split="validation")
prompts = [f"Q: {s['question']}\nA:" for s in list(tqa)[:N_SAMPLES]]
print(f"Loaded {len(prompts)} prompts")

# ── 加载模型 ──────────────────────────────────────────────────────────────────
print("\nLoading Llama-3-8B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16,
    device_map=DEVICE, output_hidden_states=True,
)
model.eval()
w_fact_dev = w_fact_t.to(DEVICE)
print("Model loaded.")

# GPU warm-up
_ = tokenizer("warm up", return_tensors="pt").to(DEVICE)
with torch.no_grad():
    __ = model(**_, output_hidden_states=True)
torch.cuda.synchronize()

# ── 计时函数 ──────────────────────────────────────────────────────────────────

def time_sfact(prompts, n_repeats=N_REPEATS):
    """
    单次前向传播 + 点积
    返回：总时间 (s)，单样本均值 (ms)
    """
    times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=128,
                               padding=False).to(DEVICE)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            h_last = out.hidden_states[BEST_LAYER][0, -1, :]  # [d]
            _ = torch.dot(h_last, w_fact_dev).item()          # Sfact
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times), np.median(times) / len(prompts) * 1000


def time_full_probe(prompts, n_repeats=N_REPEATS):
    """
    单次前向传播 + PCA(d→128) + LR 决策函数
    """
    pca_dev = pca_components.to(DEVICE)  # [128, d]
    coef_dev = clf_coef.to(DEVICE)
    bias_dev = clf_intercept.to(DEVICE)

    times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=128,
                               padding=False).to(DEVICE)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            h_last = out.hidden_states[BEST_LAYER][0, -1, :].float()  # [d]
            # mean-center (近似，略去训练均值)
            h_pca  = (pca_dev @ h_last)        # [128]
            score  = torch.dot(h_pca, coef_dev) + bias_dev  # scalar
            _ = score.item()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times), np.median(times) / len(prompts) * 1000


def time_selfcheck_style(prompts, k=5, n_repeats=N_REPEATS):
    """
    k 次前向传播（模拟 SelfCheckGPT 的多采样开销）
    注意：这里不做实际采样，只是重复 k 次前向传播来量化开销。
    """
    times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=128,
                               padding=False).to(DEVICE)
            for _ in range(k):
                with torch.no_grad():
                    _ = model(**inputs)   # no hidden states needed
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times), np.median(times) / len(prompts) * 1000


# ── 运行基准测试 ──────────────────────────────────────────────────────────────
results = {}

print(f"\n{'='*60}")
print(f"Benchmarking on {N_SAMPLES} prompts, {N_REPEATS} repeats each")
print('='*60)

print("\n[1] Sfact (1 forward pass + 1 dot product)...")
total, per_sample = time_sfact(prompts)
results["Sfact"] = {"total_s": total, "per_sample_ms": per_sample,
                    "fwd_passes": 1}
print(f"    Median total: {total:.2f}s | Per sample: {per_sample:.1f}ms")

print("\n[2] Full Probe (1 forward pass + PCA + LR)...")
total, per_sample = time_full_probe(prompts)
results["Full_Probe"] = {"total_s": total, "per_sample_ms": per_sample,
                         "fwd_passes": 1}
print(f"    Median total: {total:.2f}s | Per sample: {per_sample:.1f}ms")

for k in SCG_K_LIST:
    label = f"SelfCheck_k{k}"
    print(f"\n[3] SelfCheckGPT-style ({k} forward passes)...")
    total, per_sample = time_selfcheck_style(prompts, k=k)
    results[label] = {"total_s": total, "per_sample_ms": per_sample,
                      "fwd_passes": k}
    print(f"    Median total: {total:.2f}s | Per sample: {per_sample:.1f}ms")

# ── 计算相对加速比 ────────────────────────────────────────────────────────────
baseline_time = results["Sfact"]["per_sample_ms"]
for name, r in results.items():
    r["speedup_vs_sfact"] = r["per_sample_ms"] / baseline_time

# ── 打印汇总表格 ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"{'Method':<22} {'Fwd Passes':>12} {'Per Sample (ms)':>18} {'Relative Cost':>16}")
print('-'*70)
for name, r in results.items():
    print(f"{name:<22} {r['fwd_passes']:>12} {r['per_sample_ms']:>18.1f} "
          f"{r['speedup_vs_sfact']:>15.1f}×")
print('='*70)

# ── 保存结果 ──────────────────────────────────────────────────────────────────
with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {os.path.join(SAVE_DIR, 'results.json')}")

# ── 绘图 ──────────────────────────────────────────────────────────────────────
methods = list(results.keys())
times_ms = [results[m]["per_sample_ms"] for m in methods]
fwd_passes = [results[m]["fwd_passes"] for m in methods]

display_names = {
    "Sfact":          "$S_{\\mathrm{fact}}$ (ours)\n1 fwd pass + dot product",
    "Full_Probe":     "Full Probe\n1 fwd pass + PCA + LR",
    "SelfCheck_k5":   "SelfCheckGPT\n(k=5 fwd passes)",
    "SelfCheck_k10":  "SelfCheckGPT\n(k=10 fwd passes)",
}
colors = {
    "Sfact":         "steelblue",
    "Full_Probe":    "mediumseagreen",
    "SelfCheck_k5":  "tomato",
    "SelfCheck_k10": "firebrick",
}

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    [display_names.get(m, m) for m in methods],
    times_ms,
    color=[colors.get(m, "gray") for m in methods],
    alpha=0.85, edgecolor="white", width=0.55
)

for bar, val in zip(bars, times_ms):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            f"{val:.0f}ms", ha="center", va="bottom",
            fontsize=12, fontweight="bold")

ax.set_ylabel("Median latency per sample (ms)", fontsize=13)
ax.set_title("Inference Latency Comparison\n"
             f"Llama-3-8B, RTX 3090, n={N_SAMPLES} prompts",
             fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, max(times_ms) * 1.25)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "latency_benchmark.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path}")

# 卸载模型
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
print("\nDone.")
