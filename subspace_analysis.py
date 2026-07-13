
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

WFACT_DIR = "./results/expand_domains/wfact"
SAVE_DIR  = "./results/subspace_analysis"
FIG_DIR   = "./figures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── 域顺序（与 expand_domains.py 一致） ───────────────────────────────────────
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

# ── 加载 wfact 向量 ──────────────────────────────────────────────────────────
print("Loading wfact vectors...")
wfact_list = []
for dname in DOMAIN_ORDER:
    p = os.path.join(WFACT_DIR, f"{dname}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing wfact: {p}. Run expand_domains.py first.")
    w = np.load(p)
    wfact_list.append(w)
    print(f"  {dname}: shape={w.shape}")

W = np.stack(wfact_list)   # [N_DOMAINS, hidden_size]
print(f"\nW matrix shape: {W.shape}")

# 加载 expand_domains 的 cos_matrix（用于分析3）
expand_results_path = "./results/expand_domains/results.json"
if os.path.exists(expand_results_path):
    with open(expand_results_path) as f:
        expand_res = json.load(f)
    cos_matrix = np.array(expand_res["cos_matrix"])
    print(f"Loaded cos_matrix from expand_domains results.")
else:
    # 自己算
    cos_matrix = np.zeros((N_DOMAINS, N_DOMAINS))
    for i in range(N_DOMAINS):
        for j in range(N_DOMAINS):
            cos_matrix[i, j] = float(np.dot(wfact_list[i], wfact_list[j]))

# ══════════════════════════════════════════════════════════════════════════════
# 分析1：SVD 秩分析
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Analysis 1: SVD Rank Analysis")
print("="*60)

U, S, Vt = np.linalg.svd(W, full_matrices=False)
total_var = np.sum(S**2)
cum_var   = np.cumsum(S**2) / total_var

print("Singular values:", np.round(S, 4))
print("Cumulative variance explained:")
for k, cv in enumerate(cum_var):
    print(f"  top-{k+1}: {cv*100:.1f}%")

# 有效秩（entropy-based）
p    = S**2 / total_var
ent  = -np.sum(p * np.log(p + 1e-12))
eff_rank = float(np.exp(ent))
print(f"\nEffective rank (exp of entropy): {eff_rank:.2f}")

# 需要几个SVs来解释 90% / 95% 方差
for thresh in [0.80, 0.90, 0.95]:
    k = int(np.searchsorted(cum_var, thresh)) + 1
    k = min(k, len(S))
    print(f"SVs needed for {thresh*100:.0f}% variance: {k}")

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax1 = axes[0]
ax1.bar(range(1, len(S)+1), S**2 / total_var * 100, color="steelblue", alpha=0.8)
ax1.set_xlabel("Singular value index", fontsize=12)
ax1.set_ylabel("Variance explained (%)", fontsize=12)
ax1.set_title("Variance Explained by Each Singular Value\n"
              "of $W_{\\mathrm{fact}}$ Matrix", fontsize=11)
ax1.set_xticks(range(1, len(S)+1))

ax2 = axes[1]
ax2.plot(range(1, len(S)+1), cum_var * 100, "o-", color="darkorange",
         linewidth=2, markersize=8)
ax2.axhline(90, color="gray", linestyle="--", alpha=0.7, label="90%")
ax2.axhline(95, color="gray", linestyle=":",  alpha=0.7, label="95%")
ax2.set_xlabel("Number of singular vectors (k)", fontsize=12)
ax2.set_ylabel("Cumulative variance explained (%)", fontsize=12)
ax2.set_title("Cumulative Variance — Does a Low-Rank\n"
              "Factuality Subspace Exist?", fontsize=11)
ax2.legend(fontsize=10)
ax2.set_xticks(range(1, len(S)+1))
ax2.set_ylim(0, 105)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "subspace_svd.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {FIG_DIR}/subspace_svd.png")

# ══════════════════════════════════════════════════════════════════════════════
# 分析2：层级 cosine 演化
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Analysis 2: Layer-wise Cross-Domain Cosine Evolution")
print("="*60)

# 加载已有隐藏状态（已有4个域的all-layer缓存）
OLD_CACHE = {
    "TruthfulQA":        "./results/hidden_states.npz",
    "HaluEval-QA":       "./results/halueval_v2/halueval_hidden_5000.npz",
    "HaluEval-Dialogue": "./results/cross_domain/HaluEval-Dialogue_hidden.npz",
    "HaluEval-Summary":  "./results/cross_domain/HaluEval-Summary_hidden.npz",
}
NEW_CACHE_DIR = "./results/expand_domains"

N_TRAIN = 400
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def build_wfact_layer(hc, hw, layer, pca_dim=128):
    N = min(hc.shape[0], hw.shape[0])
    X = np.concatenate([hc[:N, layer, :], hw[:N, layer, :]], axis=0)
    y = np.array([1]*N + [0]*N)
    dim = min(pca_dim, X.shape[0]-1, X.shape[1])
    pca = PCA(n_components=dim, random_state=42)
    Xp  = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(Xp, y)
    w = pca.components_.T @ clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-8)

# 只用能快速加载的域做层级分析（4个干净域 + Dialogue）
LAYER_DOMAINS = ["TruthfulQA", "HaluEval-QA", "HaluEval-Dialogue"]

hidden_all = {}
for dname in LAYER_DOMAINS:
    if dname in OLD_CACHE:
        d = np.load(OLD_CACHE[dname])
    else:
        d = np.load(os.path.join(NEW_CACHE_DIR, f"{dname}_hidden.npz"))
    hidden_all[dname] = (d["h_correct"][:N_TRAIN], d["h_wrong"][:N_TRAIN])
    print(f"  Loaded {dname}: shape={hidden_all[dname][0].shape}")

n_layers = hidden_all["TruthfulQA"][0].shape[1]  # 33
print(f"\nComputing wfact at each of {n_layers} layers...")

wfact_by_layer = {}   # domain -> [n_layers, hidden_size]
for dname in LAYER_DOMAINS:
    hc, hw = hidden_all[dname]
    print(f"  {dname}...", end="", flush=True)
    ws = []
    for l in range(n_layers):
        ws.append(build_wfact_layer(hc, hw, l))
    wfact_by_layer[dname] = np.stack(ws)
    print(" done")

# 计算每层的跨域 cosine
domain_pairs = [
    ("TruthfulQA", "HaluEval-QA"),
    ("TruthfulQA", "HaluEval-Dialogue"),
    ("HaluEval-QA", "HaluEval-Dialogue"),
]
colors = ["steelblue", "darkorange", "green"]

fig, ax = plt.subplots(figsize=(9, 4.5))
for (d1, d2), c in zip(domain_pairs, colors):
    cos_per_layer = [
        float(np.dot(wfact_by_layer[d1][l], wfact_by_layer[d2][l]))
        for l in range(n_layers)
    ]
    short1 = d1.replace("HaluEval-", "Halu-")
    short2 = d2.replace("HaluEval-", "Halu-")
    ax.plot(range(n_layers), cos_per_layer, "o-", color=c, linewidth=2,
            markersize=5, label=f"{short1} vs {short2}")

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(16, color="gray", linewidth=1, linestyle=":", alpha=0.7, label="Layer 16 (l*)")
ax.set_xlabel("Layer index", fontsize=12)
ax.set_ylabel("Cosine similarity", fontsize=12)
ax.set_title("Cross-Domain $w_{\\mathrm{fact}}$ Cosine vs. Layer\n"
             "(convergence to ~0 in middle layers → domain specialization)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
ax.set_xlim(-0.5, n_layers - 0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "layer_cosine_evolution.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {FIG_DIR}/layer_cosine_evolution.png")

# ══════════════════════════════════════════════════════════════════════════════
# 分析3：语义相似度 vs wfact cosine 相关性
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Analysis 3: Semantic Similarity vs Direction Cosine")
print("="*60)

# 手动定义域间语义相似度（0-1）
# 基于任务类型和知识领域：
#   TruthfulQA & FEVER: 都是事实核验 → 高 (0.8)
#   MMLU-Medical & ARC-Science: 都是科学/医学QA → 中高 (0.6)
#   TruthfulQA & MMLU-Medical: 都是知识QA但领域不同 → 中 (0.4)
#   TruthfulQA & ARC-Science: 知识QA vs 科学推理 → 中低 (0.35)
#   FEVER & MMLU-Medical: 事实核验 vs 医学QA → 低中 (0.3)
#   FEVER & ARC-Science: 事实核验 vs 科学推理 → 低中 (0.3)
#   HaluEval-QA vs TruthfulQA: 相似任务但不同来源 → 中 (0.45)
#   HaluEval-QA vs FEVER: QA vs 事实核验 → 低中 (0.3)
#   HaluEval-QA vs MMLU-Medical: 通用QA vs 医学QA → 低 (0.2)
#   HaluEval-QA vs ARC-Science: 通用QA vs 科学QA → 低 (0.2)
#   HaluEval-Dialogue vs others: 对话场景差异大 → 很低 (0.1)
#   HaluEval-Summary vs others: 摘要场景差异最大 → 很低 (0.05)

SEMANTIC_SIM = {
    ("TruthfulQA", "FEVER"):            0.80,
    ("TruthfulQA", "MMLU-Medical"):     0.40,
    ("TruthfulQA", "ARC-Science"):      0.35,
    ("TruthfulQA", "HaluEval-QA"):      0.45,
    ("TruthfulQA", "HaluEval-Dialogue"):0.10,
    ("TruthfulQA", "HaluEval-Summary"): 0.05,
    ("FEVER", "MMLU-Medical"):          0.30,
    ("FEVER", "ARC-Science"):           0.30,
    ("FEVER", "HaluEval-QA"):           0.30,
    ("FEVER", "HaluEval-Dialogue"):     0.08,
    ("FEVER", "HaluEval-Summary"):      0.05,
    ("MMLU-Medical", "ARC-Science"):    0.60,
    ("MMLU-Medical", "HaluEval-QA"):    0.20,
    ("MMLU-Medical", "HaluEval-Dialogue"):0.08,
    ("MMLU-Medical", "HaluEval-Summary"):0.05,
    ("ARC-Science", "HaluEval-QA"):     0.20,
    ("ARC-Science", "HaluEval-Dialogue"):0.08,
    ("ARC-Science", "HaluEval-Summary"):0.05,
    ("HaluEval-QA", "HaluEval-Dialogue"):0.20,
    ("HaluEval-QA", "HaluEval-Summary"):0.15,
    ("HaluEval-Dialogue", "HaluEval-Summary"):0.12,
}

sem_vals, cos_vals = [], []
pair_labels = []
for i, d1 in enumerate(DOMAIN_ORDER):
    for j, d2 in enumerate(DOMAIN_ORDER):
        if i >= j:
            continue
        key = (d1, d2) if (d1, d2) in SEMANTIC_SIM else (d2, d1)
        if key not in SEMANTIC_SIM:
            continue
        sem = SEMANTIC_SIM[key]
        cos = cos_matrix[i, j]
        sem_vals.append(sem)
        cos_vals.append(cos)
        pair_labels.append(f"{d1[:6]}-{d2[:6]}")

sem_arr = np.array(sem_vals)
cos_arr = np.array(cos_vals)
rho, pval = spearmanr(sem_arr, cos_arr)
print(f"Spearman correlation: rho={rho:.3f}, p={pval:.4f}")
print(f"n pairs: {len(sem_vals)}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(sem_arr, cos_arr, s=80, alpha=0.7, color="steelblue", zorder=3)
# 趋势线
z = np.polyfit(sem_arr, cos_arr, 1)
p = np.poly1d(z)
xs = np.linspace(min(sem_arr), max(sem_arr), 50)
ax.plot(xs, p(xs), "r--", linewidth=1.5, alpha=0.8, label=f"Linear fit")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Domain semantic similarity (manually rated)", fontsize=12)
ax.set_ylabel("$w_{\\mathrm{fact}}$ cosine similarity", fontsize=12)
ax.set_title(f"Semantic Proximity → Direction Alignment\n"
             f"Spearman $\\rho$ = {rho:.3f} (p={pval:.3f})", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "semantic_vs_direction.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {FIG_DIR}/semantic_vs_direction.png")

# ── 保存结果 ──────────────────────────────────────────────────────────────────
results = {
    "svd": {
        "singular_values": S.tolist(),
        "cumulative_variance": cum_var.tolist(),
        "effective_rank": eff_rank,
        "k_for_80pct": int(np.searchsorted(cum_var, 0.80)) + 1,
        "k_for_90pct": int(np.searchsorted(cum_var, 0.90)) + 1,
        "k_for_95pct": int(np.searchsorted(cum_var, 0.95)) + 1,
    },
    "semantic_gradient": {
        "spearman_rho": float(rho),
        "spearman_p":   float(pval),
        "n_pairs":      len(sem_vals),
        "pairs":        [
            {"d1": DOMAIN_ORDER[i], "d2": DOMAIN_ORDER[j],
             "sem_sim": SEMANTIC_SIM.get((DOMAIN_ORDER[i], DOMAIN_ORDER[j]),
                         SEMANTIC_SIM.get((DOMAIN_ORDER[j], DOMAIN_ORDER[i]), None)),
             "cos_sim": float(cos_matrix[i,j])}
            for i in range(N_DOMAINS) for j in range(N_DOMAINS)
            if i < j and (
                (DOMAIN_ORDER[i], DOMAIN_ORDER[j]) in SEMANTIC_SIM or
                (DOMAIN_ORDER[j], DOMAIN_ORDER[i]) in SEMANTIC_SIM
            )
        ]
    }
}
with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {os.path.join(SAVE_DIR, 'results.json')}")

print("\n" + "="*60)
print("SUBSPACE SUMMARY")
print("="*60)
print(f"Effective rank: {eff_rank:.2f} (out of {N_DOMAINS} domains)")
print(f"Top-1 SV explains {cum_var[0]*100:.1f}% variance")
print(f"Semantic gradient: rho={rho:.3f}, p={pval:.3f}")
if rho > 0.3 and pval < 0.1:
    print("  → SIGNIFICANT positive correlation: semantically similar domains have more aligned directions")
else:
    print("  → Correlation not significant at p<0.1")
