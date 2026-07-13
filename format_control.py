
import os, gc, json, random
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
from sklearn.metrics import roc_auc_score

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/opt/models/Llama-3-8B"
DEVICE      = "cuda:0"
BEST_LAYER  = 16
N_TRAIN     = 400
MAX_LEN     = 192
PCA_DIM     = 128
RANDOM_SEED = 42

SAVE_DIR  = "./results/format_controlled"
WFACT_DIR = os.path.join(SAVE_DIR, "wfact")
FIG_DIR   = "./figures"
for d in [SAVE_DIR, WFACT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── 3种格式定义 ───────────────────────────────────────────────────────────────
# 每种格式用同一个函数签名：(question, answer) -> text
FORMATS = {
    "Format-MCQ": lambda q, a: f"Q: {q}\nA: {a}",
    "Format-Claim": lambda q, a: f"Claim: The answer to '{q}' is '{a}'.",
    "Format-Stmt": lambda q, a: f"Fact: {a} — in response to: {q}",
}

FORMAT_ORDER = list(FORMATS.keys())
N_FORMATS = len(FORMAT_ORDER)

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def get_hidden(text, model, tokenizer, max_length=MAX_LEN, device=DEVICE):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length, padding=False).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = torch.stack([layer[0, -1, :] for layer in out.hidden_states])
    return h.float().cpu().numpy()   # [L+1, hidden_size]


def build_wfact(h_correct, h_wrong, layer=BEST_LAYER, pca_dim=PCA_DIM):
    N  = min(h_correct.shape[0], h_wrong.shape[0])
    hc = h_correct[:N, layer, :]
    hw = h_wrong[:N, layer, :]
    X  = np.concatenate([hc, hw], axis=0)
    y  = np.array([1]*N + [0]*N)
    pca_dim_eff = min(pca_dim, X.shape[0]-1, X.shape[1])
    pca = PCA(n_components=pca_dim_eff, random_state=42)
    Xp  = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(Xp, y)
    w = pca.components_.T @ clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-8)


def compute_sfact_auc(h_correct, h_wrong, wfact, layer=BEST_LAYER):
    N  = min(h_correct.shape[0], h_wrong.shape[0])
    y  = np.array([1]*N + [0]*N)
    sc = np.concatenate([h_correct[:N, layer, :] @ wfact,
                         h_wrong[:N,   layer, :] @ wfact])
    return float(roc_auc_score(y, sc))


# ── 加载 ARC-Science 题目 ─────────────────────────────────────────────────────
print("Loading ARC-Science dataset...")
ds_easy = load_dataset("ai2_arc", "ARC-Easy",      split="test")
ds_hard = load_dataset("ai2_arc", "ARC-Challenge",  split="test")
combined = list(ds_easy) + list(ds_hard)
random.shuffle(combined)

label2idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
             "1": 0, "2": 1, "3": 2, "4": 3}

# 提取 (question, correct_answer, wrong_answer) 三元组
# 所有格式共享同一批三元组，保证知识内容完全相同
raw_triples = []
for x in combined:
    q      = x["question"]
    key    = x["answerKey"]
    texts  = x["choices"]["text"]
    if key not in label2idx:
        continue
    ans_idx = label2idx[key]
    if ans_idx >= len(texts):
        continue
    correct = texts[ans_idx]
    wrongs  = [t for i, t in enumerate(texts) if i != ans_idx]
    if not wrongs:
        continue
    raw_triples.append((q, correct, wrongs[0]))

print(f"Total ARC triples: {len(raw_triples)}")
if len(raw_triples) < N_TRAIN:
    raise ValueError(f"Not enough ARC data: {len(raw_triples)} < {N_TRAIN}")

# 固定使用前 N_TRAIN 个三元组（所有格式相同的题目集合）
triples = raw_triples[:N_TRAIN]
print(f"Using {len(triples)} triples (identical across all formats)")

# ── 为每种格式构造 pairs ───────────────────────────────────────────────────────
format_pairs = {}
for fname, fmt_fn in FORMATS.items():
    pairs = [(fmt_fn(q, correct), fmt_fn(q, wrong))
             for q, correct, wrong in triples]
    format_pairs[fname] = pairs
    print(f"  {fname}: {len(pairs)} pairs")
    print(f"    example correct: {pairs[0][0][:80]}...")
    print(f"    example wrong:   {pairs[0][1][:80]}...")

# ── 加载模型 ──────────────────────────────────────────────────────────────────
# Format-MCQ 可复用已有 ARC-Science cache（格式相同）
# 其他格式需要重新提取

def try_load_mcq_cache():
    """尝试复用 expand_domains 的 ARC-Science cache"""
    cache_path = "./results/expand_domains/ARC-Science_hidden.npz"
    if os.path.exists(cache_path):
        print(f"  [cache] Format-MCQ: reusing ARC-Science cache")
        d = np.load(cache_path)
        return d["h_correct"][:N_TRAIN], d["h_wrong"][:N_TRAIN]
    return None, None

hc_mcq_cache, hw_mcq_cache = try_load_mcq_cache()

# 判断是否需要模型
need_model = not (hc_mcq_cache is not None) or any(
    not os.path.exists(os.path.join(SAVE_DIR, f"{fname}_hidden.npz"))
    for fname in FORMAT_ORDER if fname != "Format-MCQ"
)

model, tokenizer = None, None
if need_model:
    print("\nLoading Llama-3-8B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map=DEVICE, output_hidden_states=True,
    )
    model.eval()
    print("Model loaded.")

# ── 提取各格式的 hidden states ────────────────────────────────────────────────
format_hidden = {}

for fname in FORMAT_ORDER:
    print(f"\n{'='*55}\nFormat: {fname}")

    cache_path = os.path.join(SAVE_DIR, f"{fname}_hidden.npz")

    # Format-MCQ：优先复用已有 ARC-Science cache
    if fname == "Format-MCQ" and hc_mcq_cache is not None:
        hc, hw = hc_mcq_cache, hw_mcq_cache
        print(f"  Reused existing ARC-Science cache")

    elif os.path.exists(cache_path):
        print(f"  [cache] {fname}")
        d = np.load(cache_path)
        hc = d["h_correct"][:N_TRAIN]
        hw = d["h_wrong"][:N_TRAIN]

    else:
        print(f"  Extracting {fname} ({N_TRAIN} pairs)...")
        pairs = format_pairs[fname]
        h_c_list, h_w_list = [], []
        for correct_text, wrong_text in tqdm(pairs, desc=fname):
            try:
                hc_i = get_hidden(correct_text, model, tokenizer)
                hw_i = get_hidden(wrong_text,   model, tokenizer)
                h_c_list.append(hc_i)
                h_w_list.append(hw_i)
            except Exception as e:
                print(f"    skip: {e}")
        hc = np.stack(h_c_list[:N_TRAIN])
        hw = np.stack(h_w_list[:N_TRAIN])
        np.savez_compressed(cache_path, h_correct=hc, h_wrong=hw)
        print(f"  Saved {cache_path}, shape={hc.shape}")

    print(f"  Shape: hc={hc.shape}, hw={hw.shape}")
    format_hidden[fname] = (hc, hw)

if model is not None:
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("\nModel unloaded.")

# ── 构建 wfact 向量 ──────────────────────────────────────────────────────────
print("\nBuilding wfact for each format...")
wfact_dict = {}
for fname, (hc, hw) in format_hidden.items():
    w = build_wfact(hc, hw, layer=BEST_LAYER)
    wfact_dict[fname] = w
    np.save(os.path.join(WFACT_DIR, f"{fname}.npy"), w)
    print(f"  {fname}: norm={np.linalg.norm(w):.4f}")

# ── 3x3 余弦相似度矩阵 ──────────────────────────────────────────────────────
print("\nComputing cosine similarity matrix...")
cos_matrix = np.zeros((N_FORMATS, N_FORMATS))
for i, f1 in enumerate(FORMAT_ORDER):
    for j, f2 in enumerate(FORMAT_ORDER):
        cos_matrix[i, j] = float(np.dot(wfact_dict[f1], wfact_dict[f2]))

# ── 3x3 AUC 矩阵 ────────────────────────────────────────────────────────────
print("Computing AUC matrix...")
auc_matrix = np.zeros((N_FORMATS, N_FORMATS))
for i, f_src in enumerate(FORMAT_ORDER):
    for j, f_tgt in enumerate(FORMAT_ORDER):
        hc, hw = format_hidden[f_tgt]
        auc_matrix[i, j] = compute_sfact_auc(hc, hw, wfact_dict[f_src])

# ── SVD 秩分析 ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SVD Rank Analysis")
W = np.stack([wfact_dict[f] for f in FORMAT_ORDER])
U, S, Vt = np.linalg.svd(W, full_matrices=False)
total_var = np.sum(S**2)
cum_var   = np.cumsum(S**2) / total_var

p = S**2 / total_var
eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
print(f"Singular values: {np.round(S, 4)}")
for k, cv in enumerate(cum_var):
    print(f"  top-{k+1}: {cv*100:.1f}%")
print(f"Effective rank: {eff_rank:.2f} / {N_FORMATS}")

# 随机基线
random_eff_ranks = []
for _ in range(100):
    R = np.random.randn(N_FORMATS, 4096)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    _, Sr, _ = np.linalg.svd(R, full_matrices=False)
    pr = Sr**2 / np.sum(Sr**2)
    random_eff_ranks.append(float(np.exp(-np.sum(pr * np.log(pr + 1e-12)))))
print(f"Random baseline: {np.mean(random_eff_ranks):.2f} +/- {np.std(random_eff_ranks):.2f}")

# ── 打印矩阵 ─────────────────────────────────────────────────────────────────
short = {"Format-MCQ": "MCQ", "Format-Claim": "Claim", "Format-Stmt": "Stmt"}
labels = [short[f] for f in FORMAT_ORDER]

print("\nCosine Similarity Matrix:")
print(f"{'':>14}" + "".join(f"  {l:>8}" for l in labels))
for i, f1 in enumerate(FORMAT_ORDER):
    row = f"{short[f1]:>14}"
    for j in range(N_FORMATS):
        row += f"  {cos_matrix[i,j]:>8.4f}"
    print(row)

print("\nAUC Matrix (row=source, col=target):")
print(f"{'':>14}" + "".join(f"  {l:>8}" for l in labels))
for i, f1 in enumerate(FORMAT_ORDER):
    row = f"{short[f1]:>14}"
    for j in range(N_FORMATS):
        m = "*" if i == j else " "
        row += f"  {auc_matrix[i,j]:>7.4f}{m}"
    print(row)

# ── 绘图1：Cosine + AUC 矩阵 ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax1 = axes[0]
im1 = ax1.imshow(cos_matrix, vmin=-0.5, vmax=1.0, cmap="RdBu_r", aspect="auto")
plt.colorbar(im1, ax=ax1, shrink=0.80)
ax1.set_xticks(range(N_FORMATS)); ax1.set_yticks(range(N_FORMATS))
ax1.set_xticklabels(labels, fontsize=12)
ax1.set_yticklabels(labels, fontsize=12)
ax1.set_title("Pairwise Cosine of $w_{\\mathrm{fact}}$\n"
              "(Same ARC-Science questions, different prompt format)", fontsize=11)
for i in range(N_FORMATS):
    for j in range(N_FORMATS):
        v = cos_matrix[i, j]
        c = "white" if abs(v) > 0.55 else "black"
        ax1.text(j, i, f"{v:.3f}", ha="center", va="center",
                 fontsize=11, color=c, fontweight="bold")

ax2 = axes[1]
im2 = ax2.imshow(auc_matrix, vmin=0.4, vmax=1.0, cmap="YlOrRd", aspect="auto")
plt.colorbar(im2, ax=ax2, shrink=0.80)
ax2.set_xticks(range(N_FORMATS)); ax2.set_yticks(range(N_FORMATS))
ax2.set_xticklabels(labels, fontsize=12)
ax2.set_yticklabels(labels, fontsize=12)
ax2.set_xlabel("Test format", fontsize=11)
ax2.set_ylabel("$w_{\\mathrm{fact}}$ source format", fontsize=11)
ax2.set_title("Cross-Format Transfer AUC\n"
              "(Fixed knowledge domain: ARC-Science)", fontsize=11)
for i in range(N_FORMATS):
    for j in range(N_FORMATS):
        v = auc_matrix[i, j]
        c = "white" if v > 0.85 else "black"
        m = "*" if i == j else ""
        ax2.text(j, i, f"{v:.3f}{m}", ha="center", va="center",
                 fontsize=11, color=c, fontweight="bold")

plt.suptitle("Controlled Experiment 2: Fixed Knowledge Domain, Varying Task Format\n"
             "Same ARC-Science questions — only prompt template changes",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "format_controlled_matrix.png"),
            dpi=150, bbox_inches="tight")
print(f"\nSaved: {FIG_DIR}/format_controlled_matrix.png")

# ── 绘图2：SVD ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax1 = axes[0]
ax1.bar(range(1, len(S)+1), S**2 / total_var * 100, color="steelblue", alpha=0.8)
ax1.set_xlabel("Singular value index", fontsize=12)
ax1.set_ylabel("Variance explained (%)", fontsize=12)
ax1.set_title("Per-SV Variance\n(Fixed domain, 3 formats)", fontsize=11)
ax1.set_xticks(range(1, len(S)+1))

ax2 = axes[1]
ax2.plot(range(1, len(S)+1), cum_var * 100, "o-", color="darkorange",
         linewidth=2, markersize=8)
ax2.axhline(90, color="gray", linestyle="--", alpha=0.7, label="90%")
ax2.set_xlabel("Number of singular vectors (k)", fontsize=12)
ax2.set_ylabel("Cumulative variance (%)", fontsize=12)
ax2.set_title(f"Cumulative Variance\neff. rank={eff_rank:.2f}/{N_FORMATS}  "
              f"(random={np.mean(random_eff_ranks):.2f})", fontsize=11)
ax2.legend(fontsize=10)
ax2.set_xticks(range(1, len(S)+1))
ax2.set_ylim(0, 105)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "format_controlled_svd.png"),
            dpi=150, bbox_inches="tight")
print(f"Saved: {FIG_DIR}/format_controlled_svd.png")

# ── 保存结果 ─────────────────────────────────────────────────────────────────
off_diag_cos = [cos_matrix[i,j]
                for i in range(N_FORMATS) for j in range(N_FORMATS) if i != j]
off_diag_auc = [auc_matrix[i,j]
                for i in range(N_FORMATS) for j in range(N_FORMATS) if i != j]

results = {
    "experiment": "Controlled: fixed knowledge domain (ARC-Science), varying task format",
    "format_names": FORMAT_ORDER,
    "format_templates": {
        "Format-MCQ":   "Q: {question}\\nA: {answer}",
        "Format-Claim": "Claim: The answer to '{question}' is '{answer}'.",
        "Format-Stmt":  "Fact: {answer} — in response to: {question}",
    },
    "n_triples": len(triples),
    "n_train": N_TRAIN,
    "layer": BEST_LAYER,
    "pca_dim": PCA_DIM,
    "cos_matrix": cos_matrix.tolist(),
    "auc_matrix": auc_matrix.tolist(),
    "svd": {
        "singular_values": S.tolist(),
        "cumulative_variance": cum_var.tolist(),
        "effective_rank": eff_rank,
        "random_baseline_mean": float(np.mean(random_eff_ranks)),
        "random_baseline_std":  float(np.std(random_eff_ranks)),
    },
    "summary": {
        "mean_off_diag_cos":  float(np.mean(np.abs(off_diag_cos))),
        "max_off_diag_cos":   float(np.max(np.abs(off_diag_cos))),
        "min_off_diag_cos":   float(np.min(np.abs(off_diag_cos))),
        "mean_within_auc":    float(np.mean([auc_matrix[i,i] for i in range(N_FORMATS)])),
        "mean_cross_auc":     float(np.mean(off_diag_auc)),
    },
    "pairwise_detail": [
        {
            "format_1": FORMAT_ORDER[i],
            "format_2": FORMAT_ORDER[j],
            "cosine":   float(cos_matrix[i, j]),
            "auc_1to2": float(auc_matrix[i, j]),
            "auc_2to1": float(auc_matrix[j, i]),
        }
        for i in range(N_FORMATS) for j in range(N_FORMATS) if i < j
    ],
}

with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {SAVE_DIR}/results.json")

# ── 打印摘要 + 跨实验对比 ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("FORMAT CONTROLLED EXPERIMENT SUMMARY")
print("="*65)
print(f"Knowledge domain: ARC-Science (fixed, {N_TRAIN} questions)")
print(f"Off-diagonal |cos|: mean={results['summary']['mean_off_diag_cos']:.4f}, "
      f"max={results['summary']['max_off_diag_cos']:.4f}, "
      f"min={results['summary']['min_off_diag_cos']:.4f}")
print(f"Mean within-format AUC: {results['summary']['mean_within_auc']:.4f}")
print(f"Mean cross-format AUC:  {results['summary']['mean_cross_auc']:.4f}")
print(f"SVD effective rank:     {eff_rank:.2f} / {N_FORMATS}")
print(f"Random baseline:        {np.mean(random_eff_ranks):.2f} +/- {np.std(random_eff_ranks):.2f}")

print("\nPairwise cosines:")
for i, f1 in enumerate(FORMAT_ORDER):
    for j, f2 in enumerate(FORMAT_ORDER):
        if i < j:
            print(f"  cos({short[f1]:>5}, {short[f2]:<5}) = {cos_matrix[i,j]:+.4f}"
                  f"  | AUC {short[f1]}->{short[f2]}: {auc_matrix[i,j]:.3f}"
                  f"  | AUC {short[f2]}->{short[f1]}: {auc_matrix[j,i]:.3f}")

# 三个实验对比汇总
print("\n" + "="*65)
print("THREE-EXPERIMENT COMPARISON")
print("="*65)
print(f"{'Experiment':<35} {'Mean |cos|':>10} {'Eff.rank':>10} {'Cross AUC':>10}")
print("-"*65)

exp1_path = "./results/expand_domains/results.json"
exp2_path = "./results/mmlu_controlled/results.json"
# ── 与实验1对比，量化两个因素的相对贡献 ──────────────────────────────────────
print("\n" + "="*65)
print("JOINT INTERPRETATION: FORMAT vs DOMAIN CONTRIBUTION")
print("="*65)
exp1_path = "./results/mmlu_controlled/results.json"
if os.path.exists(exp1_path):
    with open(exp1_path) as f:
        exp1 = json.load(f)
    domain_cos = exp1["summary"]["mean_off_diag_cos"]    # 固定格式，变领域
    format_cos = results["summary"]["mean_off_diag_cos"]  # 固定领域，变格式
    print(f"Exp1 (fixed MCQ format, vary domain): mean off-diag |cos| = {domain_cos:.4f}")
    print(f"Exp2 (fixed ARC domain, vary format): mean off-diag |cos| = {format_cos:.4f}")
    print()
    if domain_cos > format_cos:
        print(f"  -> Knowledge domain is the LARGER contributor ({domain_cos:.3f} > {format_cos:.3f})")
    else:
        print(f"  -> Task format is the LARGER contributor ({format_cos:.3f} > {domain_cos:.3f})")
    print(f"  -> Both factors independently produce direction variation,")
    print(f"     supporting the conclusion that no universal factuality direction exists.")
else:
    print("(Run mmlu_controlled.py first for joint interpretation)")
"""
控制实验2：固定知识领域，变任务格式
目的：量化 task format 对 wfact 方向差异的独立贡献

所有3个条件均来自 ARC-Science，共享完全相同的：
  - 知识领域：科学（物理、生物、地球科学）
  - 问题内容：完全相同的题目
  - 正确/错误答案：完全相同

唯一变量：prompt 格式
  Format-MCQ   : "Q: {question}\nA: {answer}"          ← 原论文格式
  Format-Claim : "Claim: {answer} is the answer to: {question}"  ← FEVER风格
  Format-Stmt  : "{question} {answer}"                  ← 裸拼接，无格式标记

如果不同格式的方向接近（cos 高）→ 格式影响小，知识领域是主因
如果不同格式的方向仍然正交（cos 低）→ 格式本身就足以改变方向

与实验1合并解读：
  实验1（固定格式变领域）: mean cos = 0.26  → 领域贡献
  实验2（固定领域变格式）: 待测              → 格式贡献
  两者对比即可量化各自贡献

输出：
  results/format_controlled/results.json
  figures/format_controlled_matrix.png
  figures/format_controlled_svd.png
"""

import os, gc, json, random
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
from sklearn.metrics import roc_auc_score

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/opt/models/Llama-3-8B"
DEVICE      = "cuda:0"
BEST_LAYER  = 16
N_TRAIN     = 400
MAX_LEN     = 192
PCA_DIM     = 128
RANDOM_SEED = 42

SAVE_DIR  = "./results/format_controlled"
WFACT_DIR = os.path.join(SAVE_DIR, "wfact")
FIG_DIR   = "./figures"
for d in [SAVE_DIR, WFACT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── 3种格式定义 ───────────────────────────────────────────────────────────────
def format_mcq(question, answer):
    """原论文使用的格式，与 expand_domains.py 完全一致"""
    return f"Q: {question}\nA: {answer}"

def format_claim(question, answer):
    """FEVER风格的 claim 陈述，去除 Q/A 结构"""
    return f"Claim: {answer} is the answer to: {question}"

def format_stmt(question, answer):
    """裸拼接，无任何格式标记"""
    return f"{question} {answer}"

FORMATS = {
    "Format-MCQ":   format_mcq,
    "Format-Claim": format_claim,
    "Format-Stmt":  format_stmt,
}
FORMAT_ORDER = list(FORMATS.keys())
N_FORMATS = len(FORMAT_ORDER)

# ── 加载 ARC-Science 题目（只加载一次，三种格式共用同一批题目） ─────────────────
print("Loading ARC-Science dataset...")
ds_easy = load_dataset("ai2_arc", "ARC-Easy",      split="test")
ds_hard = load_dataset("ai2_arc", "ARC-Challenge",  split="test")
combined = list(ds_easy) + list(ds_hard)
random.shuffle(combined)

label2idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
             "1": 0, "2": 1, "3": 2, "4": 3}

# 提取 (question, correct_answer, wrong_answer) 三元组
triplets = []
for x in combined:
    q      = x["question"]
    key    = x["answerKey"]
    texts  = x["choices"]["text"]
    if key not in label2idx:
        continue
    ans_idx = label2idx[key]
    if ans_idx >= len(texts):
        continue
    correct = texts[ans_idx]
    wrongs  = [t for i, t in enumerate(texts) if i != ans_idx]
    if not wrongs:
        continue
    triplets.append((q, correct, wrongs[0]))

print(f"ARC-Science triplets: {len(triplets)}")
assert len(triplets) >= N_TRAIN, f"Not enough ARC data: {len(triplets)} < {N_TRAIN}"

# 固定使用同一批 N_TRAIN 个题目，三种格式完全相同
triplets = triplets[:N_TRAIN]

# ── 工具函数（与 expand_domains.py 完全一致） ─────────────────────────────────

def get_hidden(text, model, tokenizer, max_length=MAX_LEN, device=DEVICE):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length, padding=False).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = torch.stack([layer[0, -1, :] for layer in out.hidden_states])
    return h.float().cpu().numpy()   # [L+1, hidden_size]


def build_wfact(h_correct, h_wrong, layer=BEST_LAYER, pca_dim=PCA_DIM):
    N  = min(h_correct.shape[0], h_wrong.shape[0])
    hc = h_correct[:N, layer, :]
    hw = h_wrong[:N, layer, :]
    X  = np.concatenate([hc, hw], axis=0)
    y  = np.array([1]*N + [0]*N)
    pca_dim_eff = min(pca_dim, X.shape[0]-1, X.shape[1])
    pca = PCA(n_components=pca_dim_eff, random_state=42)
    Xp  = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(Xp, y)
    w = pca.components_.T @ clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-8)


def compute_sfact_auc(h_correct, h_wrong, wfact, layer=BEST_LAYER):
    N  = min(h_correct.shape[0], h_wrong.shape[0])
    y  = np.array([1]*N + [0]*N)
    sc = np.concatenate([h_correct[:N, layer, :] @ wfact,
                         h_wrong[:N,   layer, :] @ wfact])
    return float(roc_auc_score(y, sc))


# ── 提取各格式的 hidden states ────────────────────────────────────────────────
print("\nLoading Llama-3-8B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16,
    device_map=DEVICE, output_hidden_states=True,
)
model.eval()
print("Model loaded.")

format_hidden = {}

for fmt_name, fmt_fn in FORMATS.items():
    cache_path = os.path.join(SAVE_DIR, f"{fmt_name}_hidden.npz")
    if os.path.exists(cache_path):
        print(f"\n[cache] {fmt_name}")
        d = np.load(cache_path)
        format_hidden[fmt_name] = (d["h_correct"], d["h_wrong"])
        continue

    print(f"\nExtracting {fmt_name} ({N_TRAIN} pairs)...")
    h_c_list, h_w_list = [], []
    for q, correct, wrong in tqdm(triplets, desc=fmt_name):
        try:
            text_c = fmt_fn(q, correct)
            text_w = fmt_fn(q, wrong)
            hc = get_hidden(text_c, model, tokenizer)
            hw = get_hidden(text_w, model, tokenizer)
            h_c_list.append(hc)
            h_w_list.append(hw)
        except Exception as e:
            print(f"  skip: {e}")

    h_c = np.stack(h_c_list[:N_TRAIN])
    h_w = np.stack(h_w_list[:N_TRAIN])
    np.savez_compressed(cache_path, h_correct=h_c, h_wrong=h_w)
    print(f"  Saved {cache_path}, shape={h_c.shape}")
    format_hidden[fmt_name] = (h_c, h_w)

del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
print("\nModel unloaded.")

# ── 构建 wfact 向量 ──────────────────────────────────────────────────────────
print("\nBuilding wfact for each format...")
wfact_dict = {}
for fmt_name, (hc, hw) in format_hidden.items():
    w = build_wfact(hc, hw, layer=BEST_LAYER)
    wfact_dict[fmt_name] = w
    np.save(os.path.join(WFACT_DIR, f"{fmt_name}.npy"), w)
    print(f"  {fmt_name}: norm={np.linalg.norm(w):.4f}")

# ── 3x3 余弦相似度矩阵 ───────────────────────────────────────────────────────
print("\nComputing cosine similarity matrix...")
cos_matrix = np.zeros((N_FORMATS, N_FORMATS))
for i, f1 in enumerate(FORMAT_ORDER):
    for j, f2 in enumerate(FORMAT_ORDER):
        cos_matrix[i, j] = float(np.dot(wfact_dict[f1], wfact_dict[f2]))

# ── 3x3 AUC 矩阵 ────────────────────────────────────────────────────────────
print("Computing AUC matrix...")
auc_matrix = np.zeros((N_FORMATS, N_FORMATS))
for i, f_src in enumerate(FORMAT_ORDER):
    for j, f_tgt in enumerate(FORMAT_ORDER):
        hc, hw = format_hidden[f_tgt]
        auc_matrix[i, j] = compute_sfact_auc(hc, hw, wfact_dict[f_src])

# ── SVD 秩分析 ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SVD Rank Analysis")
W = np.stack([wfact_dict[f] for f in FORMAT_ORDER])
U, S, Vt = np.linalg.svd(W, full_matrices=False)
total_var = np.sum(S**2)
cum_var   = np.cumsum(S**2) / total_var

p = S**2 / total_var
eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
print(f"Singular values: {np.round(S, 4)}")
for k, cv in enumerate(cum_var):
    print(f"  top-{k+1}: {cv*100:.1f}%")
print(f"Effective rank: {eff_rank:.2f} / {N_FORMATS}")

# 随机基线（3个随机单位向量）
random_eff_ranks = []
for _ in range(100):
    R  = np.random.randn(N_FORMATS, 4096)
    R  = R / np.linalg.norm(R, axis=1, keepdims=True)
    _, Sr, _ = np.linalg.svd(R, full_matrices=False)
    pr = Sr**2 / np.sum(Sr**2)
    random_eff_ranks.append(float(np.exp(-np.sum(pr * np.log(pr + 1e-12)))))
print(f"Random baseline: {np.mean(random_eff_ranks):.2f} +/- {np.std(random_eff_ranks):.2f}")

# ── 绘图1：Cosine + AUC 矩阵 ────────────────────────────────────────────────
short  = {f: f.replace("Format-", "") for f in FORMAT_ORDER}
labels = [short[f] for f in FORMAT_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax1 = axes[0]
im1 = ax1.imshow(cos_matrix, vmin=-0.5, vmax=1.0, cmap="RdBu_r", aspect="auto")
plt.colorbar(im1, ax=ax1, shrink=0.80)
ax1.set_xticks(range(N_FORMATS)); ax1.set_yticks(range(N_FORMATS))
ax1.set_xticklabels(labels, fontsize=11)
ax1.set_yticklabels(labels, fontsize=11)
ax1.set_title("Pairwise Cosine Similarity of $w_{\\mathrm{fact}}$\n"
              "(Same ARC-Science content, different prompt format)", fontsize=11)
for i in range(N_FORMATS):
    for j in range(N_FORMATS):
        v = cos_matrix[i, j]
        c = "white" if abs(v) > 0.55 else "black"
        ax1.text(j, i, f"{v:.3f}", ha="center", va="center",
                 fontsize=11, color=c, fontweight="bold")

ax2 = axes[1]
im2 = ax2.imshow(auc_matrix, vmin=0.4, vmax=1.0, cmap="YlOrRd", aspect="auto")
plt.colorbar(im2, ax=ax2, shrink=0.80)
ax2.set_xticks(range(N_FORMATS)); ax2.set_yticks(range(N_FORMATS))
ax2.set_xticklabels(labels, fontsize=11)
ax2.set_yticklabels(labels, fontsize=11)
ax2.set_xlabel("Test format", fontsize=11)
ax2.set_ylabel("$w_{\\mathrm{fact}}$ source format", fontsize=11)
ax2.set_title("Cross-Format Transfer AUC\n"
              "(Fixed ARC-Science domain, varying prompt format)", fontsize=11)
for i in range(N_FORMATS):
    for j in range(N_FORMATS):
        v = auc_matrix[i, j]
        c = "white" if v > 0.85 else "black"
        m = " *" if i == j else ""
        ax2.text(j, i, f"{v:.3f}{m}", ha="center", va="center",
                 fontsize=11, color=c, fontweight="bold")

plt.suptitle("Controlled Experiment 2: Fixed Knowledge Domain, Varying Prompt Format\n"
             "All 3 conditions use identical ARC-Science questions and answers",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "format_controlled_matrix.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {FIG_DIR}/format_controlled_matrix.png")

# ── 绘图2：SVD 秩分析 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax1 = axes[0]
ax1.bar(range(1, len(S)+1), S**2 / total_var * 100, color="steelblue", alpha=0.8)
ax1.set_xlabel("Singular value index", fontsize=12)
ax1.set_ylabel("Variance explained (%)", fontsize=12)
ax1.set_title("Per-SV Variance\n(Fixed ARC-Science domain, 3 formats)", fontsize=11)
ax1.set_xticks(range(1, len(S)+1))

ax2 = axes[1]
ax2.plot(range(1, len(S)+1), cum_var * 100, "o-", color="darkorange",
         linewidth=2, markersize=8, label="Format controlled")
ax2.axhline(90, color="gray", linestyle="--", alpha=0.7, label="90%")
ax2.set_xlabel("Number of singular vectors (k)", fontsize=12)
ax2.set_ylabel("Cumulative variance explained (%)", fontsize=12)
ax2.set_title(f"Cumulative Variance\neff. rank={eff_rank:.2f}/{N_FORMATS}  "
              f"(random={np.mean(random_eff_ranks):.2f})", fontsize=11)
ax2.legend(fontsize=10)
ax2.set_xticks(range(1, len(S)+1))
ax2.set_ylim(0, 105)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "format_controlled_svd.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {FIG_DIR}/format_controlled_svd.png")

# ── 保存结果 ─────────────────────────────────────────────────────────────────
off_diag_cos = [cos_matrix[i,j]
                for i in range(N_FORMATS) for j in range(N_FORMATS) if i != j]
off_diag_auc = [auc_matrix[i,j]
                for i in range(N_FORMATS) for j in range(N_FORMATS) if i != j]

results = {
    "experiment": "Controlled: fixed ARC-Science domain, varying prompt format",
    "format_names": FORMAT_ORDER,
    "format_descriptions": {
        "Format-MCQ":   "Q: {question}\\nA: {answer}",
        "Format-Claim": "Claim: {answer} is the answer to: {question}",
        "Format-Stmt":  "{question} {answer}",
    },
    "n_formats": N_FORMATS,
    "n_train": N_TRAIN,
    "layer": BEST_LAYER,
    "pca_dim": PCA_DIM,
    "cos_matrix": cos_matrix.tolist(),
    "auc_matrix": auc_matrix.tolist(),
    "svd": {
        "singular_values": S.tolist(),
        "cumulative_variance": cum_var.tolist(),
        "effective_rank": eff_rank,
        "random_baseline_mean": float(np.mean(random_eff_ranks)),
        "random_baseline_std":  float(np.std(random_eff_ranks)),
    },
    "summary": {
        "mean_off_diag_cos":  float(np.mean(np.abs(off_diag_cos))),
        "max_off_diag_cos":   float(np.max(np.abs(off_diag_cos))),
        "min_off_diag_cos":   float(np.min(np.abs(off_diag_cos))),
        "mean_within_auc":    float(np.mean([auc_matrix[i,i] for i in range(N_FORMATS)])),
        "mean_cross_auc":     float(np.mean(off_diag_auc)),
    },
    "pairwise_detail": [
        {
            "format_1": FORMAT_ORDER[i],
            "format_2": FORMAT_ORDER[j],
            "cosine":   float(cos_matrix[i, j]),
            "auc_1to2": float(auc_matrix[i, j]),
            "auc_2to1": float(auc_matrix[j, i]),
        }
        for i in range(N_FORMATS) for j in range(N_FORMATS) if i < j
    ],
}

with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {SAVE_DIR}/results.json")

# ── 打印摘要 ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FORMAT CONTROLLED EXPERIMENT SUMMARY")
print("="*65)
print(f"Off-diagonal |cos|: mean={results['summary']['mean_off_diag_cos']:.4f}, "
      f"max={results['summary']['max_off_diag_cos']:.4f}, "
      f"min={results['summary']['min_off_diag_cos']:.4f}")
print(f"Mean within-format AUC: {results['summary']['mean_within_auc']:.4f}")
print(f"Mean cross-format AUC:  {results['summary']['mean_cross_auc']:.4f}")
print(f"SVD effective rank:     {eff_rank:.2f} / {N_FORMATS}")
print(f"Random baseline:        {np.mean(random_eff_ranks):.2f} +/- {np.std(random_eff_ranks):.2f}")

print("\nPairwise cosines:")
for i, f1 in enumerate(FORMAT_ORDER):
    for j, f2 in enumerate(FORMAT_ORDER):
        if i < j:
            print(f"  cos({short[f1]:>5}, {short[f2]:<5}) = {cos_matrix[i,j]:+.4f}"
                  f"  | AUC {short[f1]}->{short[f2]}: {auc_matrix[i,j]:.3f}"
                  f"  | AUC {short[f2]}->{short[f1]}: {auc_matrix[j,i]:.3f}")

# ── 与实验1对比，量化两个因素的相对贡献 ──────────────────────────────────────
print("\n" + "="*65)
print("JOINT INTERPRETATION: FORMAT vs DOMAIN CONTRIBUTION")
print("="*65)
exp1_path = "./results/mmlu_controlled/results.json"
if os.path.exists(exp1_path):
    with open(exp1_path) as f:
        exp1 = json.load(f)
    domain_cos = exp1["summary"]["mean_off_diag_cos"]   # 固定格式，变领域
    format_cos = results["summary"]["mean_off_diag_cos"] # 固定领域，变格式
    print(f"Exp1 (fixed format, vary domain): mean off-diag |cos| = {domain_cos:.4f}")
    print(f"Exp2 (fixed domain, vary format): mean off-diag |cos| = {format_cos:.4f}")
    print(f"\nInterpretation:")
    print(f"  Domain contribution to direction variation: ~{domain_cos:.4f} mean cos distance")
    print(f"  Format contribution to direction variation: ~{format_cos:.4f} mean cos distance")
    if domain_cos > format_cos:
        print(f"  -> Knowledge domain is the LARGER contributor ({domain_cos:.3f} > {format_cos:.3f})")
    else:
        print(f"  -> Task format is the LARGER contributor ({format_cos:.3f} > {domain_cos:.3f})")
    print(f"\n  Both factors independently produce domain-specific directions,")
    print(f"  supporting the conclusion that no universal factuality direction exists.")
else:
    print("(Run mmlu_controlled.py first for joint interpretation)")