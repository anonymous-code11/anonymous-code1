
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
MODEL_PATH   = "/opt/models/Llama-3-8B"
DEVICE       = "cuda:0"
BEST_LAYER   = 16
N_TRAIN      = 400
MAX_LEN      = 192
PCA_DIM      = 128
RANDOM_SEED  = 42

SAVE_DIR  = "./results/mmlu_controlled"
WFACT_DIR = os.path.join(SAVE_DIR, "wfact")
FIG_DIR   = "./figures"
for d in [SAVE_DIR, WFACT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── 5个MMLU知识领域（完全相同MCQ格式） ────────────────────────────────────────
MMLU_DOMAINS = {
    "MMLU-Medical": {
        "anatomy", "clinical_knowledge", "medical_genetics",
        "college_medicine", "professional_medicine",
    },
    "MMLU-Law": {
        "jurisprudence", "international_law", "professional_law",
        "professional_accounting", "business_ethics",
    },
    "MMLU-History": {
        "high_school_us_history", "high_school_world_history",
        "high_school_european_history", "prehistory",
    },
    "MMLU-CS": {
        "computer_security", "machine_learning",
        "college_computer_science", "high_school_computer_science",
    },
    "MMLU-Psychology": {
        "high_school_psychology", "professional_psychology",
        "moral_scenarios",
    },
}

DOMAIN_ORDER = list(MMLU_DOMAINS.keys())
N_DOMAINS = len(DOMAIN_ORDER)

# ── 工具函数（与 expand_domains.py 完全一致） ─────────────────────────────────

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


# ── 加载MMLU数据集 ────────────────────────────────────────────────────────────
print("Loading MMLU dataset...")
ds = load_dataset("cais/mmlu", "all", split="test")

def load_mmlu_domain_pairs(ds, subjects):
    items = [x for x in ds if x["subject"] in subjects]
    random.shuffle(items)
    pairs = []
    for x in items:
        q   = x["question"]
        ans = x["answer"]       # int 0-3
        ch  = x["choices"]
        if not (0 <= ans < len(ch)):
            continue
        correct = ch[ans]
        wrongs  = [c for i, c in enumerate(ch) if i != ans]
        if not wrongs:
            continue
        wrong = wrongs[0]       # 统一取第一个错误选项
        pairs.append((
            f"Q: {q}\nA: {correct}",
            f"Q: {q}\nA: {wrong}"
        ))
    return pairs

domain_pairs = {}
for dname, subjects in MMLU_DOMAINS.items():
    pairs = load_mmlu_domain_pairs(ds, subjects)
    domain_pairs[dname] = pairs
    status = "OK" if len(pairs) >= N_TRAIN else f"WARNING: only {len(pairs)}"
    print(f"  {dname}: {len(pairs)} pairs [{status}]")

# ── 可选：复用已有 MMLU-Medical cache ─────────────────────────────────────────
EXISTING_CACHE = {
    "MMLU-Medical": "./results/expand_domains/MMLU-Medical_hidden.npz",
}

# ── 加载模型 ──────────────────────────────────────────────────────────────────
need_model = any(
    not os.path.exists(EXISTING_CACHE.get(d, os.path.join(SAVE_DIR, f"{d}_hidden.npz")))
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

# ── 提取 hidden states ────────────────────────────────────────────────────────
domain_hidden = {}
for dname in DOMAIN_ORDER:
    print(f"\n{'='*55}\nDomain: {dname}")
    if dname in EXISTING_CACHE and os.path.exists(EXISTING_CACHE[dname]):
        print(f"  Loading from existing cache: {EXISTING_CACHE[dname]}")
        d = np.load(EXISTING_CACHE[dname])
        hc = d["h_correct"][:N_TRAIN]
        hw = d["h_wrong"][:N_TRAIN]
    else:
        hc, hw = extract_and_cache(dname, domain_pairs[dname], model, tokenizer)
    print(f"  Shape: hc={hc.shape}, hw={hw.shape}")
    domain_hidden[dname] = (hc, hw)

if model is not None:
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("\nModel unloaded.")

# ── 构建 wfact 向量 ──────────────────────────────────────────────────────────
print("\nBuilding wfact for each domain...")
wfact_dict = {}
for dname, (hc, hw) in domain_hidden.items():
    w = build_wfact(hc, hw, layer=BEST_LAYER)
    wfact_dict[dname] = w
    np.save(os.path.join(WFACT_DIR, f"{dname}.npy"), w)
    print(f"  {dname}: norm={np.linalg.norm(w):.4f}")

# ── 5x5 余弦相似度矩阵 ───────────────────────────────────────────────────────
print("\nComputing cosine similarity matrix...")
cos_matrix = np.zeros((N_DOMAINS, N_DOMAINS))
for i, d1 in enumerate(DOMAIN_ORDER):
    for j, d2 in enumerate(DOMAIN_ORDER):
        cos_matrix[i, j] = float(np.dot(wfact_dict[d1], wfact_dict[d2]))

# ── 5x5 AUC 矩阵 ────────────────────────────────────────────────────────────
print("Computing AUC matrix...")
auc_matrix = np.zeros((N_DOMAINS, N_DOMAINS))
for i, d_src in enumerate(DOMAIN_ORDER):
    for j, d_tgt in enumerate(DOMAIN_ORDER):
        hc, hw = domain_hidden[d_tgt]
        auc_matrix[i, j] = compute_sfact_auc(hc, hw, wfact_dict[d_src])

# ── SVD 秩分析 ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SVD Rank Analysis")
W = np.stack([wfact_dict[d] for d in DOMAIN_ORDER])
U, S, Vt = np.linalg.svd(W, full_matrices=False)
total_var = np.sum(S**2)
cum_var   = np.cumsum(S**2) / total_var

p = S**2 / total_var
eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
print(f"Singular values: {np.round(S, 4)}")
for k, cv in enumerate(cum_var):
    print(f"  top-{k+1}: {cv*100:.1f}%")
print(f"Effective rank: {eff_rank:.2f} / {N_DOMAINS}")

# 随机基线
random_eff_ranks = []
for _ in range(100):
    R = np.random.randn(N_DOMAINS, 4096)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    _, Sr, _ = np.linalg.svd(R, full_matrices=False)
    pr = Sr**2 / np.sum(Sr**2)
    random_eff_ranks.append(float(np.exp(-np.sum(pr * np.log(pr + 1e-12)))))
print(f"Random baseline: {np.mean(random_eff_ranks):.2f} +/- {np.std(random_eff_ranks):.2f}")

# ── 绘图1：Cosine + AUC 矩阵 ────────────────────────────────────────────────
short  = {d: d.replace("MMLU-", "") for d in DOMAIN_ORDER}
labels = [short[d] for d in DOMAIN_ORDER]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

ax1 = axes[0]
im1 = ax1.imshow(cos_matrix, vmin=-0.5, vmax=1.0, cmap="RdBu_r", aspect="auto")
plt.colorbar(im1, ax=ax1, shrink=0.80)
ax1.set_xticks(range(N_DOMAINS)); ax1.set_yticks(range(N_DOMAINS))
ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
ax1.set_yticklabels(labels, fontsize=10)
ax1.set_title("Pairwise Cosine Similarity of $w_{\\mathrm{fact}}$\n"
              "(All domains: identical MCQ format, different knowledge domain)", fontsize=11)
for i in range(N_DOMAINS):
    for j in range(N_DOMAINS):
        v = cos_matrix[i, j]
        c = "white" if abs(v) > 0.55 else "black"
        ax1.text(j, i, f"{v:.3f}", ha="center", va="center",
                 fontsize=9, color=c, fontweight="bold")

ax2 = axes[1]
im2 = ax2.imshow(auc_matrix, vmin=0.4, vmax=1.0, cmap="YlOrRd", aspect="auto")
plt.colorbar(im2, ax=ax2, shrink=0.80)
ax2.set_xticks(range(N_DOMAINS)); ax2.set_yticks(range(N_DOMAINS))
ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
ax2.set_yticklabels(labels, fontsize=10)
ax2.set_xlabel("Test domain", fontsize=11)
ax2.set_ylabel("$w_{\\mathrm{fact}}$ source domain", fontsize=11)
ax2.set_title("Cross-Domain Transfer AUC\n"
              "(Fixed MCQ format, varying knowledge domain)", fontsize=11)
for i in range(N_DOMAINS):
    for j in range(N_DOMAINS):
        v = auc_matrix[i, j]
        c = "white" if v > 0.85 else "black"
        m = " *" if i == j else ""
        ax2.text(j, i, f"{v:.3f}{m}", ha="center", va="center",
                 fontsize=9, color=c, fontweight="bold")

plt.suptitle("Controlled Experiment: Fixed MCQ Format, Varying Knowledge Domain\n"
             "5 MMLU domains — identical prompt template and label construction",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "mmlu_controlled_matrix.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {FIG_DIR}/mmlu_controlled_matrix.png")

# ── 绘图2：SVD 秩分析 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax1 = axes[0]
ax1.bar(range(1, len(S)+1), S**2 / total_var * 100, color="steelblue", alpha=0.8)
ax1.set_xlabel("Singular value index", fontsize=12)
ax1.set_ylabel("Variance explained (%)", fontsize=12)
ax1.set_title("Per-SV Variance (Fixed MCQ, 5 MMLU domains)", fontsize=11)
ax1.set_xticks(range(1, len(S)+1))

ax2 = axes[1]
ax2.plot(range(1, len(S)+1), cum_var * 100, "o-", color="darkorange",
         linewidth=2, markersize=8, label="MMLU controlled")
ax2.axhline(90, color="gray", linestyle="--", alpha=0.7, label="90%")
ax2.axhline(95, color="gray", linestyle=":",  alpha=0.7, label="95%")
ax2.set_xlabel("Number of singular vectors (k)", fontsize=12)
ax2.set_ylabel("Cumulative variance explained (%)", fontsize=12)
ax2.set_title(f"Cumulative Variance\neff. rank={eff_rank:.2f}/{N_DOMAINS}  "
              f"(random baseline={np.mean(random_eff_ranks):.2f})", fontsize=11)
ax2.legend(fontsize=10)
ax2.set_xticks(range(1, len(S)+1))
ax2.set_ylim(0, 105)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "mmlu_controlled_svd.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {FIG_DIR}/mmlu_controlled_svd.png")

# ── 保存结果 ─────────────────────────────────────────────────────────────────
off_diag_cos = [cos_matrix[i,j]
                for i in range(N_DOMAINS) for j in range(N_DOMAINS) if i != j]
off_diag_auc = [auc_matrix[i,j]
                for i in range(N_DOMAINS) for j in range(N_DOMAINS) if i != j]

results = {
    "experiment": "Controlled: fixed MCQ format, varying knowledge domain",
    "domain_names": DOMAIN_ORDER,
    "domain_subjects": {d: sorted(s) for d, s in MMLU_DOMAINS.items()},
    "n_domains": N_DOMAINS,
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
        "k_for_80pct": int(np.searchsorted(cum_var, 0.80)) + 1,
        "k_for_90pct": int(np.searchsorted(cum_var, 0.90)) + 1,
    },
    "summary": {
        "mean_off_diag_cos":  float(np.mean(np.abs(off_diag_cos))),
        "max_off_diag_cos":   float(np.max(np.abs(off_diag_cos))),
        "min_off_diag_cos":   float(np.min(np.abs(off_diag_cos))),
        "mean_within_auc":    float(np.mean([auc_matrix[i,i] for i in range(N_DOMAINS)])),
        "mean_cross_auc":     float(np.mean(off_diag_auc)),
        "n_pairs_below_0.1":  sum(1 for c in off_diag_cos if abs(c) < 0.1),
        "n_pairs_total":      len(off_diag_cos),
    },
    "pairwise_detail": [
        {
            "domain_1": DOMAIN_ORDER[i],
            "domain_2": DOMAIN_ORDER[j],
            "cosine":   float(cos_matrix[i, j]),
            "auc_1to2": float(auc_matrix[i, j]),
            "auc_2to1": float(auc_matrix[j, i]),
        }
        for i in range(N_DOMAINS) for j in range(N_DOMAINS) if i < j
    ],
}

with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {SAVE_DIR}/results.json")

# ── 打印摘要 ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("CONTROLLED EXPERIMENT SUMMARY")
print("="*65)
print(f"Off-diagonal |cos|: mean={results['summary']['mean_off_diag_cos']:.4f}, "
      f"max={results['summary']['max_off_diag_cos']:.4f}, "
      f"min={results['summary']['min_off_diag_cos']:.4f}")
print(f"Pairs with |cos| < 0.1: {results['summary']['n_pairs_below_0.1']}/{results['summary']['n_pairs_total']}")
print(f"Mean within-domain AUC: {results['summary']['mean_within_auc']:.4f}")
print(f"Mean cross-domain AUC:  {results['summary']['mean_cross_auc']:.4f}")
print(f"SVD effective rank:     {eff_rank:.2f} / {N_DOMAINS}")
print(f"Random baseline:        {np.mean(random_eff_ranks):.2f} +/- {np.std(random_eff_ranks):.2f}")
print("\nPairwise cosines:")
for i, d1 in enumerate(DOMAIN_ORDER):
    for j, d2 in enumerate(DOMAIN_ORDER):
        if i < j:
            print(f"  cos({short[d1]:>10}, {short[d2]:<10}) = {cos_matrix[i,j]:+.4f}"
                  f"  | AUC {short[d1]}->{short[d2]}: {auc_matrix[i,j]:.3f}"
                  f"  | AUC {short[d2]}->{short[d1]}: {auc_matrix[j,i]:.3f}")

# ── 与原始实验对比 ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("COMPARISON WITH ORIGINAL 7-DOMAIN RESULTS")
print("="*65)
orig_path = "./results/expand_domains/results.json"
if os.path.exists(orig_path):
    with open(orig_path) as f:
        orig = json.load(f)
    print(f"Original 7-domain (mixed format):")
    print(f"  Mean off-diag |cos| (clean): {orig['summary']['mean_off_diag_cos_clean']:.4f}")
    print(f"  Max off-diag |cos| (clean):  {orig['summary']['max_off_diag_cos_clean']:.4f}")
    print(f"  SVD effective rank: 6.73 / 7")
    print(f"\nThis controlled experiment (fixed MCQ):")
    print(f"  Mean off-diag |cos|: {results['summary']['mean_off_diag_cos']:.4f}")
    print(f"  Max off-diag |cos|:  {results['summary']['max_off_diag_cos']:.4f}")
    print(f"  SVD effective rank:  {eff_rank:.2f} / {N_DOMAINS}")
    if results['summary']['mean_off_diag_cos'] < 0.15:
        print("\n  -> Directions remain near-orthogonal with identical format.")
        print("  -> Knowledge domain is the primary driver, not task format.")
    else:
        print("\n  -> Directions show higher alignment with identical format.")
        print("  -> Task format contributes significantly to direction differences.")
else:
    print("(Run expand_domains.py first to enable comparison)")