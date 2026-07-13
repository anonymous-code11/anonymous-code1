
import os
import gc
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# ── 配置 ──────────────────────────────────────────────────
MODEL_PATH    = "/opt/models/Llama-3-8B"
DEVICE        = "cuda:0"
BEST_LAYER    = 16
FLUENCY_LAYER = 3
TRAIN_SIZE    = 400       # 训练w_fact用
MAX_TOTAL     = 5000      # 总样本数（400 train + 4600 test）
N_RANDOM      = 100       # 随机方向baseline次数

SAVE_DIR = "./results/halueval_v2"
FIG_DIR  = "./figures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── 工具函数 ──────────────────────────────────────────────
def get_hidden(text, model, tokenizer, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length, padding=False).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = torch.stack([layer[0, -1, :] for layer in out.hidden_states])
    return h.float().cpu().numpy()


def build_w_fact(h_correct, h_wrong, layer=BEST_LAYER):
    N     = h_correct.shape[0]
    X     = np.concatenate([h_correct[:, layer, :],
                             h_wrong[:,   layer, :]], axis=0)
    y     = np.array([1]*N + [0]*N)
    pca   = PCA(n_components=128, random_state=42)
    X_pca = pca.fit_transform(X)
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X_pca)
    clf   = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(X_sc, y)
    w = pca.components_.T @ clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-8)


def build_w_flu(h_correct, h_wrong, layer=FLUENCY_LAYER):
    mean_c = h_correct[:, layer, :].mean(0)
    mean_w = h_wrong[:,   layer, :].mean(0)
    w = mean_c - mean_w
    return w / (np.linalg.norm(w) + 1e-8)


def compute_dsd_auc(h_correct, h_wrong, w_fact, w_flu, layer=BEST_LAYER):
    N  = h_correct.shape[0]
    y  = np.array([1]*N + [0]*N)
    dc = h_correct[:, layer, :] @ w_fact - h_correct[:, layer, :] @ w_flu
    dw = h_wrong[:,   layer, :] @ w_fact - h_wrong[:,   layer, :] @ w_flu
    return float(roc_auc_score(y, np.concatenate([dc, dw])))


# ════════════════════════════════════════════════════════
# STEP 1: 加载HaluEval，提取hidden states
# ════════════════════════════════════════════════════════
print("Loading HaluEval QA...")
halueval = load_dataset("pminervini/HaluEval", "qa", split="data")
print(f"Total: {len(halueval)}")

samples = []
for s in halueval:
    q  = s.get("question", "")
    ca = s.get("right_answer", "")
    wa = s.get("hallucinated_answer", "")
    if q and ca and wa:
        samples.append((
            f"Q: {q}\nA: {ca}",
            f"Q: {q}\nA: {wa}",
        ))

samples = samples[:MAX_TOTAL]
print(f"Using {len(samples)} samples "
      f"(train={TRAIN_SIZE}, test={len(samples)-TRAIN_SIZE})")

# 检查缓存
cache_path = os.path.join(SAVE_DIR, f"halueval_hidden_{MAX_TOTAL}.npz")
if os.path.exists(cache_path):
    print(f"Loading cached hidden states from {cache_path}...")
    d = np.load(cache_path)
    h_correct_all = d["h_correct"]
    h_wrong_all   = d["h_wrong"]
    print(f"Shape: {h_correct_all.shape}")
else:
    print("\nLoading Llama-3-8B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map=DEVICE, output_hidden_states=True,
    )
    model.eval()
    print(f"Loaded. Layers={model.config.num_hidden_layers}")

    print("Extracting hidden states...")
    h_correct_list, h_wrong_list = [], []
    for correct_text, wrong_text in tqdm(samples, desc="HaluEval"):
        try:
            hc = get_hidden(correct_text, model, tokenizer)
            hw = get_hidden(wrong_text,   model, tokenizer)
            h_correct_list.append(hc)
            h_wrong_list.append(hw)
        except Exception as e:
            print(f"Error: {e}")
            continue

    h_correct_all = np.stack(h_correct_list)
    h_wrong_all   = np.stack(h_wrong_list)
    print(f"Shape: {h_correct_all.shape}")

    np.savez_compressed(cache_path,
                        h_correct=h_correct_all,
                        h_wrong=h_wrong_all)
    print(f"Saved to {cache_path}")

    del model, tokenizer
    torch.cuda.empty_cache(); gc.collect()
    print("Model unloaded.")

# ════════════════════════════════════════════════════════
# STEP 2: 严格split
# ════════════════════════════════════════════════════════
N_total = h_correct_all.shape[0]
N_test  = N_total - TRAIN_SIZE

h_c_train = h_correct_all[:TRAIN_SIZE]
h_w_train = h_wrong_all[:TRAIN_SIZE]
h_c_test  = h_correct_all[TRAIN_SIZE:]
h_w_test  = h_wrong_all[TRAIN_SIZE:]

print(f"\nSplit: train={h_c_train.shape[0]}, test={h_c_test.shape[0]}")

# ════════════════════════════════════════════════════════
# STEP 3: 训练HaluEval的w_fact（只用train split）
# ════════════════════════════════════════════════════════
print("\n[1] Building w_fact from HaluEval TRAIN split (400 samples)...")
w_fact_halu = build_w_fact(h_c_train, h_w_train, layer=BEST_LAYER)
w_flu_halu  = build_w_flu(h_c_train,  h_w_train, layer=FLUENCY_LAYER)

# ════════════════════════════════════════════════════════
# STEP 4: 加载TruthfulQA的w_fact
# ════════════════════════════════════════════════════════
print("[2] Building w_fact from TruthfulQA (for zero-transfer)...")
tqa_data    = np.load("./results/hidden_states.npz")
h_c_tqa     = tqa_data["h_correct"]
h_w_tqa     = tqa_data["h_wrong"]
w_fact_tqa  = build_w_fact(h_c_tqa, h_w_tqa, layer=BEST_LAYER)
w_flu_tqa   = build_w_flu(h_c_tqa,  h_w_tqa, layer=FLUENCY_LAYER)

# ════════════════════════════════════════════════════════
# STEP 5: 在TEST split上评估
# ════════════════════════════════════════════════════════
print(f"\n[3] Evaluating on TEST split ({N_test} unseen samples)...")

# Within-domain（HaluEval train→test）
dsd_auc_within = compute_dsd_auc(
    h_c_test, h_w_test, w_fact_halu, w_flu_halu, layer=BEST_LAYER)

# Zero-transfer（TQA→HaluEval test）
dsd_auc_transfer = compute_dsd_auc(
    h_c_test, h_w_test, w_fact_tqa, w_flu_tqa, layer=BEST_LAYER)

# ════════════════════════════════════════════════════════
# STEP 6: 随机方向baseline
# ════════════════════════════════════════════════════════
print(f"[4] Random direction baseline ({N_RANDOM} runs)...")
hidden_size = h_c_test.shape[2]
N_test_half = h_c_test.shape[0]
y_test      = np.array([1]*N_test_half + [0]*N_test_half)

random_aucs = []
np.random.seed(42)
for _ in tqdm(range(N_RANDOM), desc="Random baseline"):
    w_rand = np.random.randn(hidden_size)
    w_rand = w_rand / (np.linalg.norm(w_rand) + 1e-8)
    w_flu_rand = np.random.randn(hidden_size)
    w_flu_rand = w_flu_rand / (np.linalg.norm(w_flu_rand) + 1e-8)
    dc = h_c_test[:, BEST_LAYER, :] @ w_rand - h_c_test[:, BEST_LAYER, :] @ w_flu_rand
    dw = h_w_test[:, BEST_LAYER, :] @ w_rand - h_w_test[:, BEST_LAYER, :] @ w_flu_rand
    auc = float(roc_auc_score(y_test, np.concatenate([dc, dw])))
    random_aucs.append(auc)

random_mean = float(np.mean(random_aucs))
random_std  = float(np.std(random_aucs))

# 方向稳定性
cos_wfact = float(np.dot(w_fact_tqa, w_fact_halu))
cos_wflu  = float(np.dot(w_flu_tqa,  w_flu_halu))

# ════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════
print("\n" + "="*62)
print("HALUEVAL STRICT VALIDATION (no data leakage)")
print(f"Train: {TRAIN_SIZE} samples | Test: {N_test} samples")
print("="*62)
print(f"{'Metric':<45} {'Value':>12}")
print("-"*62)
print(f"{'Within-domain DSD AUC (HaluEval train→test)':<45} {dsd_auc_within:>12.4f}")
print(f"{'Zero-transfer DSD AUC (TQA→HaluEval test)':<45} {dsd_auc_transfer:>12.4f}")
print(f"{'Random baseline AUC (mean ± std)':<45} {random_mean:.4f} ± {random_std:.4f}")
print(f"{'cos(w_fact_TQA, w_fact_HaluEval)':<45} {cos_wfact:>12.4f}")
print(f"{'cos(w_flu_TQA,  w_flu_HaluEval)':<45} {cos_wflu:>12.4f}")
print("="*62)

# 解读
gap_within   = dsd_auc_within   - random_mean
gap_transfer = dsd_auc_transfer - random_mean
print(f"\nWithin-domain gap above random:   +{gap_within:.4f}")
print(f"Zero-transfer gap above random:   +{gap_transfer:.4f}")

# 保存
results = {
    "train_size"            : TRAIN_SIZE,
    "test_size"             : N_test,
    "dsd_auc_within"        : dsd_auc_within,
    "dsd_auc_zero_transfer" : dsd_auc_transfer,
    "random_auc_mean"       : random_mean,
    "random_auc_std"        : random_std,
    "cos_wfact_tqa_halu"    : cos_wfact,
    "cos_wflu_tqa_halu"     : cos_wflu,
    "gap_within_vs_random"  : gap_within,
    "gap_transfer_vs_random": gap_transfer,
}
with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ── 绘图 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左图：三个AUC对比条形图
ax = axes[0]
labels = ["Within-domain\n(HaluEval\ntrain→test)",
          "Zero-transfer\n(TQA→HaluEval)",
          f"Random\nbaseline\n(mean±std)"]
vals   = [dsd_auc_within, dsd_auc_transfer, random_mean]
colors = ["steelblue", "tomato", "gray"]
bars   = ax.bar(labels, vals, color=colors, alpha=0.75,
                edgecolor="white", width=0.5)

# 随机baseline误差棒
ax.errorbar(2, random_mean, yerr=random_std*2,
            fmt="none", color="black", capsize=6, linewidth=2)

for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11,
            fontweight="bold")

ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Chance (0.5)")
ax.set_ylabel("DSD AUC", fontsize=12)
ax.set_title("HaluEval Strict Validation\n(No Data Leakage)", fontsize=12)
ax.set_ylim(0.3, 1.05)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

# 右图：随机baseline分布
ax2 = axes[1]
ax2.hist(random_aucs, bins=20, color="gray", alpha=0.7,
         edgecolor="white", label="Random directions (n=100)")
ax2.axvline(dsd_auc_within,   color="steelblue", lw=2.5,
            label=f"Within-domain ({dsd_auc_within:.3f})")
ax2.axvline(dsd_auc_transfer, color="tomato",    lw=2.5,
            label=f"Zero-transfer ({dsd_auc_transfer:.3f})")
ax2.axvline(random_mean,      color="black",     lw=1.5,
            linestyle="--", label=f"Random mean ({random_mean:.3f})")
ax2.set_xlabel("DSD AUC", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title("Random Direction Baseline Distribution\n"
              "vs Actual DSD Performance", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle("HaluEval QA Strict Validation — Dual Space Divergence\n"
             f"Llama-3-8B | Train={TRAIN_SIZE}, Test={N_test}",
             fontsize=13, y=1.01)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "halueval_strict.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {fig_path}")