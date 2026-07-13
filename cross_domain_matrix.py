
import os
import gc
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/opt/models/Llama-3-8B"
DEVICE      = "cuda:0"
BEST_LAYER  = 16
FLUENCY_LAYER = 3
N_TRAIN     = 400      # 每个域用于估计 wfact 的样本数
MAX_LEN     = 192      # tokenizer 最大长度（摘要域需要更长）
PCA_DIM     = 128

SAVE_DIR    = "./results/cross_domain"
FIG_DIR     = "./figures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def get_hidden(text, model, tokenizer, layer_idx=BEST_LAYER,
               max_length=MAX_LEN, device=DEVICE):
    """返回指定层最后 token 的 hidden state，shape: [num_layers+1, hidden_size]"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length, padding=False).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # stack 所有层最后 token 表示
    h = torch.stack([layer[0, -1, :] for layer in out.hidden_states])
    return h.float().cpu().numpy()   # [L+1, hidden_size]


def build_wfact(h_correct, h_wrong, layer=BEST_LAYER, pca_dim=PCA_DIM):
    """PCA-128 + LR back-projection → 单位化 wfact"""
    N  = h_correct.shape[0]
    X  = np.concatenate([h_correct[:, layer, :], h_wrong[:, layer, :]], axis=0)
    y  = np.array([1]*N + [0]*N)
    pca = PCA(n_components=pca_dim, random_state=42)
    Xp  = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(Xp, y)
    w = pca.components_.T @ clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-8)


def compute_sfact_auc(h_correct, h_wrong, wfact, layer=BEST_LAYER):
    """计算 Sfact（单点积）在给定数据上的 AUC"""
    N   = h_correct.shape[0]
    y   = np.array([1]*N + [0]*N)
    sc  = np.concatenate([h_correct[:, layer, :] @ wfact,
                          h_wrong[:,   layer, :] @ wfact])
    return float(roc_auc_score(y, sc))


def extract_and_cache(domain_name, pairs, model, tokenizer,
                      n_train=N_TRAIN, max_len=MAX_LEN):
    """
    pairs: list of (correct_text, wrong_text)
    返回 h_correct, h_wrong: [n_train, L+1, hidden_size]
    自动利用磁盘缓存。
    """
    cache_path = os.path.join(SAVE_DIR, f"{domain_name}_hidden.npz")
    if os.path.exists(cache_path):
        print(f"  [cache] Loading {domain_name} from {cache_path}")
        d = np.load(cache_path)
        return d["h_correct"][:n_train], d["h_wrong"][:n_train]

    print(f"  Extracting {domain_name} ({n_train} pairs)...")
    h_c_list, h_w_list = [], []
    for correct_text, wrong_text in tqdm(pairs[:n_train], desc=domain_name):
        try:
            hc = get_hidden(correct_text, model, tokenizer, max_length=max_len)
            hw = get_hidden(wrong_text,   model, tokenizer, max_length=max_len)
            h_c_list.append(hc)
            h_w_list.append(hw)
        except Exception as e:
            print(f"    skip: {e}")
            continue
    h_c = np.stack(h_c_list[:n_train])
    h_w = np.stack(h_w_list[:n_train])
    np.savez_compressed(cache_path, h_correct=h_c, h_wrong=h_w)
    print(f"  Saved {cache_path}, shape={h_c.shape}")
    return h_c, h_w


# ── 构造各域 (correct_text, wrong_text) 列表 ─────────────────────────────────

print("Loading datasets...")

# D1 TruthfulQA
tqa_raw  = load_dataset("truthful_qa", "generation", split="validation")
tqa_pairs = [
    (f"Q: {s['question']}\nA: {s['best_answer']}",
     f"Q: {s['question']}\nA: {s['incorrect_answers'][0]}")
    for s in tqa_raw if s["best_answer"] and s["incorrect_answers"]
]
print(f"TQA pairs: {len(tqa_pairs)}")

# D2 HaluEval-QA
halu_qa  = load_dataset("pminervini/HaluEval", "qa", split="data")
halu_qa_pairs = [
    (f"Q: {s['question']}\nA: {s['right_answer']}",
     f"Q: {s['question']}\nA: {s['hallucinated_answer']}")
    for s in halu_qa if s.get("question") and s.get("right_answer")
]
print(f"HaluEval-QA pairs: {len(halu_qa_pairs)}")

# D3 HaluEval-Dialogue
halu_dial = load_dataset("pminervini/HaluEval", "dialogue", split="data")
halu_dial_pairs = [
    (f"Dialogue: {s['dialogue_history']}\nResponse: {s['right_response']}",
     f"Dialogue: {s['dialogue_history']}\nResponse: {s['hallucinated_response']}")
    for s in halu_dial
    if s.get("dialogue_history") and s.get("right_response") and s.get("hallucinated_response")
]
print(f"HaluEval-Dialogue pairs: {len(halu_dial_pairs)}")

# D4 HaluEval-Summarization
halu_sum  = load_dataset("pminervini/HaluEval", "summarization", split="data")
halu_sum_pairs = [
    (f"Article: {s['document'][:400]}\nSummary: {s['right_summary']}",
     f"Article: {s['document'][:400]}\nSummary: {s['hallucinated_summary']}")
    for s in halu_sum
    if s.get("document") and s.get("right_summary") and s.get("hallucinated_summary")
]
print(f"HaluEval-Summary pairs: {len(halu_sum_pairs)}")

# ── 域定义 ────────────────────────────────────────────────────────────────────
DOMAINS = {
    "TruthfulQA":         tqa_pairs,
    "HaluEval-QA":        halu_qa_pairs,
    "HaluEval-Dialogue":  halu_dial_pairs,
    "HaluEval-Summary":   halu_sum_pairs,
}
DOMAIN_NAMES = list(DOMAINS.keys())
N_DOMAINS    = len(DOMAIN_NAMES)

# ── 对已缓存的域跳过模型加载 ─────────────────────────────────────────────────
# 检查哪些域需要提取
cached_map = {
    "TruthfulQA":    "./results/hidden_states.npz",          # 已有
    "HaluEval-QA":   "./results/halueval_v2/halueval_hidden_5000.npz",  # 已有
    "HaluEval-Dialogue":  os.path.join(SAVE_DIR, "HaluEval-Dialogue_hidden.npz"),
    "HaluEval-Summary":   os.path.join(SAVE_DIR, "HaluEval-Summary_hidden.npz"),
}

need_model = any(
    not os.path.exists(cached_map[d])
    for d in DOMAIN_NAMES
)

# ── 加载模型（如需要） ────────────────────────────────────────────────────────
model, tokenizer = None, None

if need_model:
    print("\nLoading Llama-3-8B for extraction...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map=DEVICE, output_hidden_states=True,
    )
    model.eval()
    print("Model loaded.")

# ── 提取各域 hidden states ────────────────────────────────────────────────────
domain_hidden = {}  # domain_name -> (h_correct, h_wrong)

for dname, pairs in DOMAINS.items():
    print(f"\n{'='*55}")
    print(f"Domain: {dname}")
    cache = cached_map[dname]

    if os.path.exists(cache):
        print(f"  Loading from existing cache: {cache}")
        d = np.load(cache)
        hc = d["h_correct"][:N_TRAIN]
        hw = d["h_wrong"][:N_TRAIN]
        print(f"  Shape: {hc.shape}")
    else:
        hc, hw = extract_and_cache(dname, pairs, model, tokenizer,
                                   n_train=N_TRAIN)

    domain_hidden[dname] = (hc, hw)

# 卸载模型
if model is not None:
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("\nModel unloaded.")

# ── 构建 wfact 向量 ───────────────────────────────────────────────────────────
print("\nBuilding wfact for each domain...")
wfact_dict = {}
for dname, (hc, hw) in domain_hidden.items():
    w = build_wfact(hc, hw, layer=BEST_LAYER)
    wfact_dict[dname] = w
    print(f"  {dname}: wfact shape={w.shape}, norm={np.linalg.norm(w):.4f}")

# ── 计算 N×N 余弦相似度矩阵 ───────────────────────────────────────────────────
print("\nComputing pairwise cosine similarity matrix...")
cos_matrix = np.zeros((N_DOMAINS, N_DOMAINS))
for i, d1 in enumerate(DOMAIN_NAMES):
    for j, d2 in enumerate(DOMAIN_NAMES):
        cos_matrix[i, j] = float(np.dot(wfact_dict[d1], wfact_dict[d2]))

print("\nCosine Similarity Matrix (wfact):")
print(f"{'':>22}", end="")
for dname in DOMAIN_NAMES:
    print(f"  {dname[:12]:>12}", end="")
print()
for i, d1 in enumerate(DOMAIN_NAMES):
    print(f"{d1:>22}", end="")
    for j in range(N_DOMAINS):
        print(f"  {cos_matrix[i,j]:>12.4f}", end="")
    print()

# ── 计算 AUC 矩阵（cross-domain transfer） ───────────────────────────────────
print("\nComputing cross-domain AUC matrix (zero-transfer)...")
auc_matrix = np.zeros((N_DOMAINS, N_DOMAINS))
for i, d_src in enumerate(DOMAIN_NAMES):
    wf = wfact_dict[d_src]
    for j, d_tgt in enumerate(DOMAIN_NAMES):
        hc, hw = domain_hidden[d_tgt]
        auc_matrix[i, j] = compute_sfact_auc(hc, hw, wf, layer=BEST_LAYER)

print("\nAUC Matrix (row=wfact source, col=test domain):")
print(f"{'':>22}", end="")
for dname in DOMAIN_NAMES:
    print(f"  {dname[:12]:>12}", end="")
print()
for i, d1 in enumerate(DOMAIN_NAMES):
    print(f"{d1:>22}", end="")
    for j in range(N_DOMAINS):
        marker = " *" if i == j else "  "
        print(f"  {auc_matrix[i,j]:>10.4f}{marker}", end="")
    print()

# ── 绘图 ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# -- 左图：余弦相似度热力图 --
ax1 = axes[0]
short_names = [d.replace("HaluEval-", "Halu-") for d in DOMAIN_NAMES]

im1 = ax1.imshow(cos_matrix, vmin=-0.5, vmax=1.0,
                  cmap="RdBu_r", aspect="auto")
plt.colorbar(im1, ax=ax1, shrink=0.85)
ax1.set_xticks(range(N_DOMAINS))
ax1.set_yticks(range(N_DOMAINS))
ax1.set_xticklabels(short_names, rotation=30, ha="right", fontsize=10)
ax1.set_yticklabels(short_names, fontsize=10)
ax1.set_title("Pairwise Cosine Similarity of $w_{\\mathrm{fact}}$\n"
              "(near-zero off-diagonal → no universal direction)",
              fontsize=11)

# 填写数值
for i in range(N_DOMAINS):
    for j in range(N_DOMAINS):
        val = cos_matrix[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax1.text(j, i, f"{val:.3f}", ha="center", va="center",
                 fontsize=11, color=color, fontweight="bold")

# -- 右图：AUC 热力图 --
ax2 = axes[1]
im2 = ax2.imshow(auc_matrix, vmin=0.45, vmax=1.0,
                  cmap="YlOrRd", aspect="auto")
plt.colorbar(im2, ax=ax2, shrink=0.85)
ax2.set_xticks(range(N_DOMAINS))
ax2.set_yticks(range(N_DOMAINS))
ax2.set_xticklabels(short_names, rotation=30, ha="right", fontsize=10)
ax2.set_yticklabels(short_names, fontsize=10)
ax2.set_xlabel("Test domain", fontsize=11)
ax2.set_ylabel("$w_{\\mathrm{fact}}$ source domain", fontsize=11)
ax2.set_title("Cross-Domain Transfer AUC\n"
              "(diagonal = within-domain; off-diagonal ≈ zero-transfer)",
              fontsize=11)

for i in range(N_DOMAINS):
    for j in range(N_DOMAINS):
        val = auc_matrix[i, j]
        color = "white" if val > 0.85 else "black"
        marker = "★" if i == j else ""
        ax2.text(j, i, f"{val:.3f}{marker}", ha="center", va="center",
                 fontsize=10, color=color, fontweight="bold")

plt.suptitle("Cross-Domain Factuality Direction Analysis\n"
             "Llama-3-8B, 400 calibration samples per domain",
             fontsize=13, y=1.02)
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "cross_domain_matrix.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {fig_path}")

# ── 保存结果 ──────────────────────────────────────────────────────────────────
results = {
    "domain_names":  DOMAIN_NAMES,
    "n_train":       N_TRAIN,
    "layer":         BEST_LAYER,
    "cos_matrix":    cos_matrix.tolist(),
    "auc_matrix":    auc_matrix.tolist(),
    "pairwise_cos":  {},
    "pairwise_auc":  {},
}

for i, d1 in enumerate(DOMAIN_NAMES):
    for j, d2 in enumerate(DOMAIN_NAMES):
        if i != j:
            key = f"{d1}→{d2}"
            results["pairwise_cos"][key] = float(cos_matrix[i, j])
            results["pairwise_auc"][key] = float(auc_matrix[i, j])

with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ── 打印摘要 ──────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY: Off-diagonal cosine similarities")
print("(near-zero → no universal factuality direction)")
print("="*65)
off_diag = [(DOMAIN_NAMES[i], DOMAIN_NAMES[j], cos_matrix[i,j])
            for i in range(N_DOMAINS) for j in range(N_DOMAINS) if i != j]
for d1, d2, cos in sorted(off_diag, key=lambda x: abs(x[2])):
    print(f"  cos({d1:>22}, {d2:<22}) = {cos:+.4f}")

print(f"\nMean |off-diagonal|: {np.mean([abs(c) for _,_,c in off_diag]):.4f}")
print(f"Max  |off-diagonal|: {np.max([abs(c) for _,_,c in off_diag]):.4f}")

print("\nSUMMARY: Within-domain vs Zero-transfer AUC")
for i, d in enumerate(DOMAIN_NAMES):
    within = auc_matrix[i, i]
    others = [auc_matrix[i, j] for j in range(N_DOMAINS) if j != i]
    print(f"  {d:<22}: within={within:.4f}, "
          f"zero-transfer={np.mean(others):.4f} (mean)")
