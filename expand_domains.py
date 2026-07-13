
import os, gc, json, random
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
MODEL_PATH   = "/opt/models/Llama-3-8B"
DEVICE       = "cuda:0"
BEST_LAYER   = 16
N_TRAIN      = 400
MAX_LEN      = 192
PCA_DIM      = 128
RANDOM_SEED  = 42

SAVE_DIR  = "./results/expand_domains"
WFACT_DIR = os.path.join(SAVE_DIR, "wfact")
FIG_DIR   = "./figures"
for d in [SAVE_DIR, WFACT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def get_hidden(text, model, tokenizer, layer_idx=BEST_LAYER,
               max_length=MAX_LEN, device=DEVICE):
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


def build_wfact_all_layers(h_correct, h_wrong, n_layers=33, pca_dim=PCA_DIM):
    """返回每层的 wfact，shape: [n_layers, hidden_size]"""
    wfacts = []
    N = min(h_correct.shape[0], h_wrong.shape[0])
    for l in range(n_layers):
        hc = h_correct[:N, l, :]
        hw = h_wrong[:N, l, :]
        X  = np.concatenate([hc, hw], axis=0)
        y  = np.array([1]*N + [0]*N)
        pca_dim_eff = min(pca_dim, X.shape[0]-1, X.shape[1])
        pca = PCA(n_components=pca_dim_eff, random_state=42)
        Xp  = pca.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(Xp, y)
        w = pca.components_.T @ clf.coef_[0]
        w = w / (np.linalg.norm(w) + 1e-8)
        wfacts.append(w)
    return np.stack(wfacts)   # [n_layers, hidden_size]


def compute_sfact_auc(h_correct, h_wrong, wfact, layer=BEST_LAYER):
    N  = min(h_correct.shape[0], h_wrong.shape[0])
    y  = np.array([1]*N + [0]*N)
    sc = np.concatenate([h_correct[:N, layer, :] @ wfact,
                         h_wrong[:N,   layer, :] @ wfact])
    return float(roc_auc_score(y, sc))


def extract_and_cache(domain_name, pairs, model, tokenizer,
                      n_train=N_TRAIN, max_len=MAX_LEN, save_dir=SAVE_DIR):
    cache_path = os.path.join(save_dir, f"{domain_name}_hidden.npz")
    if os.path.exists(cache_path):
        print(f"  [cache] {domain_name}")
        d = np.load(cache_path)
        return d["h_correct"][:n_train], d["h_wrong"][:n_train]
    print(f"  Extracting {domain_name} ({n_train} pairs)...")
    h_c_list, h_w_list = [], []
    for correct_text, wrong_text in tqdm(pairs[:n_train * 2], desc=domain_name):
        if len(h_c_list) >= n_train:
            break
        try:
            hc = get_hidden(correct_text, model, tokenizer, max_length=max_len)
            hw = get_hidden(wrong_text,   model, tokenizer, max_length=max_len)
            h_c_list.append(hc)
            h_w_list.append(hw)
        except Exception as e:
            print(f"    skip: {e}")
    h_c = np.stack(h_c_list[:n_train])
    h_w = np.stack(h_w_list[:n_train])
    np.savez_compressed(cache_path, h_correct=h_c, h_wrong=h_w)
    print(f"  Saved {cache_path}, shape={h_c.shape}")
    return h_c, h_w


# ── 构造各域 pairs ─────────────────────────────────────────────────────────────

print("Loading datasets...")

# D5: FEVER（Wikipedia事实核验，非GPT生成）
def load_fever_pairs():
    ds = load_dataset("pietrolesci/nli_fever", split="train")
    # label: 0=SUPPORTS, 1=REFUTES, 2=NOT ENOUGH INFO
    supports = [x for x in ds if x["label"] == 0 and x["hypothesis"].strip()]
    refutes  = [x for x in ds if x["label"] == 1 and x["hypothesis"].strip()]
    random.shuffle(supports)
    random.shuffle(refutes)
    n = min(len(supports), len(refutes), 600)
    pairs = [
        (f"Claim: {supports[i]['hypothesis']}",
         f"Claim: {refutes[i]['hypothesis']}")
        for i in range(n)
    ]
    print(f"FEVER pairs: {len(pairs)} (from {len(supports)} supports, {len(refutes)} refutes)")
    return pairs

# D6: MMLU-Medical（医学知识多选，非GPT生成）
MEDICAL_SUBJECTS = {
    "anatomy", "clinical_knowledge", "medical_genetics",
    "college_medicine", "professional_medicine", "college_biology",
    "high_school_biology", "nutrition",
}
def load_mmlu_medical_pairs():
    ds = load_dataset("cais/mmlu", "all", split="test")
    medical = [x for x in ds if x["subject"] in MEDICAL_SUBJECTS]
    random.shuffle(medical)
    pairs = []
    for x in medical:
        q      = x["question"]
        ans    = x["answer"]   # int 0-3
        ch     = x["choices"]  # list of 4 strings
        if not (0 <= ans < len(ch)):
            continue
        correct = ch[ans]
        wrongs  = [c for i, c in enumerate(ch) if i != ans]
        if not wrongs:
            continue
        wrong = wrongs[0]
        pairs.append((
            f"Q: {q}\nA: {correct}",
            f"Q: {q}\nA: {wrong}"
        ))
    print(f"MMLU-Medical pairs: {len(pairs)}")
    return pairs

# D7: ARC-Science（科学推理，非GPT生成）
def load_arc_pairs():
    # Use both Easy and Challenge for more samples
    ds_easy  = load_dataset("ai2_arc", "ARC-Easy",      split="test")
    ds_hard  = load_dataset("ai2_arc", "ARC-Challenge",  split="test")
    combined = list(ds_easy) + list(ds_hard)
    random.shuffle(combined)
    pairs = []
    label2idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
                 "1": 0, "2": 1, "3": 2, "4": 3}
    for x in combined:
        q       = x["question"]
        key     = x["answerKey"]
        texts   = x["choices"]["text"]
        labels  = x["choices"]["label"]
        if key not in label2idx:
            continue
        ans_idx = label2idx[key]
        if ans_idx >= len(texts):
            continue
        correct = texts[ans_idx]
        wrongs  = [t for i, t in enumerate(texts) if i != ans_idx]
        if not wrongs:
            continue
        wrong = wrongs[0]
        pairs.append((
            f"Q: {q}\nA: {correct}",
            f"Q: {q}\nA: {wrong}"
        ))
    print(f"ARC-Science pairs: {len(pairs)}")
    return pairs

fever_pairs  = load_fever_pairs()
mmlu_pairs   = load_mmlu_medical_pairs()
arc_pairs    = load_arc_pairs()

# ── 域定义（包含已有缓存路径） ─────────────────────────────────────────────────

# 旧域使用已有缓存
OLD_CACHE = {
    "TruthfulQA":        "./results/hidden_states.npz",
    "HaluEval-QA":       "./results/halueval_v2/halueval_hidden_5000.npz",
    "HaluEval-Dialogue": "./results/cross_domain/HaluEval-Dialogue_hidden.npz",
    "HaluEval-Summary":  "./results/cross_domain/HaluEval-Summary_hidden.npz",
}

NEW_DOMAINS = {
    "FEVER":        fever_pairs,
    "MMLU-Medical": mmlu_pairs,
    "ARC-Science":  arc_pairs,
}

DOMAIN_ORDER = [
    "TruthfulQA",
    "FEVER",
    "MMLU-Medical",
    "ARC-Science",
    "HaluEval-QA",
    "HaluEval-Dialogue",
    "HaluEval-Summary",
]
N_DOMAINS = len(DOMAIN_ORDER)

# ── 检查哪些域需要提取 ─────────────────────────────────────────────────────────
need_model = any(
    not os.path.exists(OLD_CACHE.get(d, os.path.join(SAVE_DIR, f"{d}_hidden.npz")))
    for d in DOMAIN_ORDER
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

# ── 提取各域 hidden states ────────────────────────────────────────────────────
domain_hidden = {}

for dname in DOMAIN_ORDER:
    print(f"\n{'='*55}")
    print(f"Domain: {dname}")

    if dname in OLD_CACHE:
        cache = OLD_CACHE[dname]
        print(f"  Loading from existing cache: {cache}")
        d = np.load(cache)
        hc = d["h_correct"][:N_TRAIN]
        hw = d["h_wrong"][:N_TRAIN]
    else:
        pairs = NEW_DOMAINS[dname]
        hc, hw = extract_and_cache(dname, pairs, model, tokenizer)

    print(f"  Shape: hc={hc.shape}, hw={hw.shape}")
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
    # 保存到文件
    np.save(os.path.join(WFACT_DIR, f"{dname}.npy"), w)
    print(f"  {dname}: norm={np.linalg.norm(w):.4f}")

# ── 计算 N×N 余弦相似度矩阵 ───────────────────────────────────────────────────
print("\nComputing cosine similarity matrix...")
cos_matrix = np.zeros((N_DOMAINS, N_DOMAINS))
for i, d1 in enumerate(DOMAIN_ORDER):
    for j, d2 in enumerate(DOMAIN_ORDER):
        cos_matrix[i, j] = float(np.dot(wfact_dict[d1], wfact_dict[d2]))

print("\nCosine Similarity Matrix:")
header = f"{'':>22}" + "".join(f"  {d[:12]:>12}" for d in DOMAIN_ORDER)
print(header)
for i, d1 in enumerate(DOMAIN_ORDER):
    row = f"{d1:>22}"
    for j in range(N_DOMAINS):
        row += f"  {cos_matrix[i,j]:>12.4f}"
    print(row)

# ── 计算 AUC 矩阵 ─────────────────────────────────────────────────────────────
print("\nComputing AUC matrix...")
auc_matrix = np.zeros((N_DOMAINS, N_DOMAINS))
for i, d_src in enumerate(DOMAIN_ORDER):
    wf = wfact_dict[d_src]
    for j, d_tgt in enumerate(DOMAIN_ORDER):
        hc, hw = domain_hidden[d_tgt]
        auc_matrix[i, j] = compute_sfact_auc(hc, hw, wf, layer=BEST_LAYER)

print("\nAUC Matrix (row=source, col=target):")
print(header)
for i, d1 in enumerate(DOMAIN_ORDER):
    row = f"{d1:>22}"
    for j in range(N_DOMAINS):
        m = "★" if i == j else " "
        row += f"  {auc_matrix[i,j]:>10.4f}{m} "
    print(row)

# ── 绘图：2×1（cosine + AUC） ─────────────────────────────────────────────────
short = {
    "TruthfulQA":       "TruthfulQA",
    "FEVER":            "FEVER",
    "MMLU-Medical":     "MMLU-Med",
    "ARC-Science":      "ARC-Sci",
    "HaluEval-QA":      "Halu-QA",
    "HaluEval-Dialogue":"Halu-Dial",
    "HaluEval-Summary": "Halu-Sum",
}
labels = [short[d] for d in DOMAIN_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

# Cosine
ax1 = axes[0]
im1 = ax1.imshow(cos_matrix, vmin=-0.5, vmax=1.0, cmap="RdBu_r", aspect="auto")
plt.colorbar(im1, ax=ax1, shrink=0.80)
ax1.set_xticks(range(N_DOMAINS)); ax1.set_yticks(range(N_DOMAINS))
ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax1.set_yticklabels(labels, fontsize=9)
ax1.set_title("Pairwise Cosine Similarity of $w_{\\mathrm{fact}}$\n"
              "(near-zero off-diagonal → domain-specific directions)", fontsize=11)
for i in range(N_DOMAINS):
    for j in range(N_DOMAINS):
        v = cos_matrix[i,j]
        c = "white" if abs(v) > 0.55 else "black"
        ax1.text(j, i, f"{v:.3f}", ha="center", va="center",
                 fontsize=8, color=c, fontweight="bold")

# AUC
ax2 = axes[1]
im2 = ax2.imshow(auc_matrix, vmin=0.4, vmax=1.0, cmap="YlOrRd", aspect="auto")
plt.colorbar(im2, ax=ax2, shrink=0.80)
ax2.set_xticks(range(N_DOMAINS)); ax2.set_yticks(range(N_DOMAINS))
ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax2.set_yticklabels(labels, fontsize=9)
ax2.set_xlabel("Test domain", fontsize=11)
ax2.set_ylabel("$w_{\\mathrm{fact}}$ source domain", fontsize=11)
ax2.set_title("Cross-Domain Transfer AUC\n"
              "(diagonal★ = within-domain; off-diagonal = zero-transfer)", fontsize=11)
for i in range(N_DOMAINS):
    for j in range(N_DOMAINS):
        v = auc_matrix[i,j]
        c = "white" if v > 0.85 else "black"
        m = "★" if i == j else ""
        ax2.text(j, i, f"{v:.3f}{m}", ha="center", va="center",
                 fontsize=8, color=c, fontweight="bold")

# 标注域分组
for ax in axes:
    # 竖线/横线区分 clean vs HaluEval
    ax.axvline(x=3.5, color='blue', lw=1.5, linestyle='--', alpha=0.5)
    ax.axhline(y=3.5, color='blue', lw=1.5, linestyle='--', alpha=0.5)

plt.suptitle("7-Domain Factuality Direction Analysis — Llama-3-8B\n"
             "Left of blue line: clean domains (non-GPT-generated); Right: HaluEval",
             fontsize=12, y=1.01)
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "expand_domains_matrix.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nSaved figure: {fig_path}")

# ── 保存结果 ──────────────────────────────────────────────────────────────────
off_diag_cos = [cos_matrix[i,j]
                for i in range(N_DOMAINS) for j in range(N_DOMAINS) if i != j]
# Clean-only off-diagonal (first 4 domains: TQA, FEVER, MMLU-Med, ARC)
N_CLEAN = 4
off_diag_clean = [cos_matrix[i,j]
                  for i in range(N_CLEAN) for j in range(N_CLEAN) if i != j]

results = {
    "domain_names":       DOMAIN_ORDER,
    "n_clean_domains":    N_CLEAN,
    "n_train":            N_TRAIN,
    "layer":              BEST_LAYER,
    "cos_matrix":         cos_matrix.tolist(),
    "auc_matrix":         auc_matrix.tolist(),
    "summary": {
        "mean_off_diag_cos_all":   float(np.mean(np.abs(off_diag_cos))),
        "max_off_diag_cos_all":    float(np.max(np.abs(off_diag_cos))),
        "mean_off_diag_cos_clean": float(np.mean(np.abs(off_diag_clean))),
        "max_off_diag_cos_clean":  float(np.max(np.abs(off_diag_clean))),
        "mean_within_auc":         float(np.mean([auc_matrix[i,i] for i in range(N_DOMAINS)])),
        "clean_cross_auc_mean":    float(np.mean([auc_matrix[i,j]
                                                   for i in range(N_CLEAN)
                                                   for j in range(N_CLEAN) if i != j])),
    }
}

with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {os.path.join(SAVE_DIR, 'results.json')}")

# ── 打印摘要 ──────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
print(f"All off-diagonal |cos|:   mean={results['summary']['mean_off_diag_cos_all']:.4f}, "
      f"max={results['summary']['max_off_diag_cos_all']:.4f}")
print(f"Clean-only |cos|:         mean={results['summary']['mean_off_diag_cos_clean']:.4f}, "
      f"max={results['summary']['max_off_diag_cos_clean']:.4f}")
print(f"Mean within-domain AUC:   {results['summary']['mean_within_auc']:.4f}")
print(f"Clean cross-domain AUC:   {results['summary']['clean_cross_auc_mean']:.4f}")

print("\nPairwise clean-domain cosines:")
clean_domains = DOMAIN_ORDER[:N_CLEAN]
for i, d1 in enumerate(clean_domains):
    for j, d2 in enumerate(clean_domains):
        if i < j:
            print(f"  cos({d1}, {d2}) = {cos_matrix[i,j]:.4f}")
