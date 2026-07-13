
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RESULTS_DIR = "./results"
FIG_DIR     = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────
print("Loading hidden states...")
data      = np.load(os.path.join(RESULTS_DIR, "hidden_states.npz"))
h_correct = data["h_correct"]   # [N, 33, 4096]
h_wrong   = data["h_wrong"]     # [N, 33, 4096]
N         = h_correct.shape[0]

BEST_LAYER = 16   # 从上一步实验得到

# ── 方向1：事实方向 w_fact ────────────────────────────────
# 在best layer上训练probing分类器，提取权重向量
print(f"\n[1] Computing w_fact from Layer {BEST_LAYER} probing classifier...")

X_layer = np.concatenate([h_correct[:, BEST_LAYER, :],
                           h_wrong[:,   BEST_LAYER, :]], axis=0)  # [2N, 4096]
y       = np.array([1]*N + [0]*N)

# PCA降维
pca = PCA(n_components=128, random_state=42)
X_pca = pca.fit_transform(X_layer)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
clf.fit(X_scaled, y)

# 权重向量投影回原始空间
# clf.coef_: [1, 128] in PCA space
# 投影回4096维
w_fact_pca = clf.coef_[0]                          # [128]
w_fact = pca.components_.T @ w_fact_pca            # [4096]
w_fact = w_fact / (np.linalg.norm(w_fact) + 1e-8)

print(f"   w_fact norm (normalized): {np.linalg.norm(w_fact):.4f}")
print(f"   Classifier train ACC: {clf.score(X_scaled, y):.4f}")

# ── 方向2：流畅性方向 w_fluency ───────────────────────────
# 定义：correct 和 wrong 在same layer的均值差
# correct样本流畅性更高（训练数据中的标准表达），wrong样本表达更"奇怪"
# 所以均值差方向 = 流畅性方向的近似
print(f"\n[2] Computing w_fluency from mean difference at Layer {BEST_LAYER}...")

mean_correct = h_correct[:, BEST_LAYER, :].mean(axis=0)  # [4096]
mean_wrong   = h_wrong[:,   BEST_LAYER, :].mean(axis=0)  # [4096]

# 注意：这里用early layer（Layer 3）的均值差作为流畅性方向
# 因为early layer主要编码surface/syntactic信息，事实知识尚未激活
FLUENCY_LAYER = 3
print(f"   (Using Layer {FLUENCY_LAYER} for fluency direction — early layer encodes surface features)")

mean_correct_early = h_correct[:, FLUENCY_LAYER, :].mean(axis=0)
mean_wrong_early   = h_wrong[:,   FLUENCY_LAYER, :].mean(axis=0)

w_fluency = mean_correct_early - mean_wrong_early
w_fluency = w_fluency / (np.linalg.norm(w_fluency) + 1e-8)

print(f"   w_fluency norm (normalized): {np.linalg.norm(w_fluency):.4f}")

# ── 核心计算：余弦相似度 ──────────────────────────────────
cos_sim = float(np.dot(w_fact, w_fluency))
print(f"\n[3] cos(w_fact, w_fluency) = {cos_sim:.4f}")

if abs(cos_sim) < 0.15:
    verdict = "NEAR ORTHOGONAL ✓ — Dual subspace hypothesis SUPPORTED"
elif abs(cos_sim) < 0.3:
    verdict = "WEAKLY CORRELATED — Partial support"
else:
    verdict = "CORRELATED ✗ — Hypothesis needs revision"
print(f"    Verdict: {verdict}")

# ── 多层余弦相似度扫描 ────────────────────────────────────
# 对每一层计算 cos(w_fact@layer_i, w_fluency@layer_3)
# 观察：随着层加深，事实方向是否逐渐与流畅性方向分离
print("\n[4] Scanning cosine similarity across layers...")

cos_per_layer = []
for layer_idx in range(h_correct.shape[1]):
    mean_c = h_correct[:, layer_idx, :].mean(axis=0)
    mean_w = h_wrong[:,   layer_idx, :].mean(axis=0)
    diff = mean_c - mean_w
    diff = diff / (np.linalg.norm(diff) + 1e-8)
    cos = float(np.dot(diff, w_fluency))
    cos_per_layer.append(cos)
    print(f"  Layer {layer_idx:2d}: cos={cos:.4f}")

# ── 图：多层余弦相似度 ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

layers = list(range(len(cos_per_layer)))
axes[0].plot(layers, cos_per_layer, "purple", marker="o", markersize=4, linewidth=2)
axes[0].axhline(0, color="gray", linestyle="--", alpha=0.6)
axes[0].axvline(BEST_LAYER, color="red", linestyle="--", alpha=0.6,
                label=f"Best probing layer ({BEST_LAYER})")
axes[0].axvline(FLUENCY_LAYER, color="blue", linestyle="--", alpha=0.6,
                label=f"Fluency ref layer ({FLUENCY_LAYER})")
axes[0].set_xlabel("Layer Index", fontsize=13)
axes[0].set_ylabel("cos(mean_diff_layer_i, w_fluency)", fontsize=11)
axes[0].set_title("Cosine Similarity: Mean-Diff Direction vs Fluency Direction\n"
                  "Approaching 0 = orthogonal = dual subspace", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 图2：w_fact 和 w_fluency 的PCA投影对比（2D）
X_best = np.concatenate([h_correct[:, BEST_LAYER, :],
                          h_wrong[:,   BEST_LAYER, :]], axis=0)
pca2d = PCA(n_components=2, random_state=42)
pca2d.fit(X_best)

# 投影两个方向向量到2D
def project_vec(v, pca):
    return pca.components_ @ v   # [2]

wf_2d  = project_vec(w_fact,    pca2d)
wfl_2d = project_vec(w_fluency, pca2d)

wf_2d  = wf_2d  / (np.linalg.norm(wf_2d)  + 1e-8)
wfl_2d = wfl_2d / (np.linalg.norm(wfl_2d) + 1e-8)

ax2 = axes[1]
X_2d = pca2d.transform(X_best)
ax2.scatter(X_2d[:N, 0],  X_2d[:N, 1],  c="steelblue", alpha=0.3, s=10, label="Factual")
ax2.scatter(X_2d[N:, 0],  X_2d[N:, 1],  c="tomato",    alpha=0.3, s=10, label="Hallucination")

scale = 30
ax2.annotate("", xy=(wf_2d[0]*scale,  wf_2d[1]*scale),  xytext=(0,0),
             arrowprops=dict(arrowstyle="->", color="navy", lw=2.5))
ax2.annotate("", xy=(wfl_2d[0]*scale, wfl_2d[1]*scale), xytext=(0,0),
             arrowprops=dict(arrowstyle="->", color="darkred", lw=2.5))
ax2.text(wf_2d[0]*scale*1.1,  wf_2d[1]*scale*1.1,
         f"w_fact\n(Layer {BEST_LAYER})", color="navy", fontsize=10)
ax2.text(wfl_2d[0]*scale*1.1, wfl_2d[1]*scale*1.1,
         f"w_fluency\n(Layer {FLUENCY_LAYER})", color="darkred", fontsize=10)

ax2.set_title(f"Dual Direction Visualization (PCA 2D)\n"
              f"cos(w_fact, w_fluency) = {cos_sim:.4f}", fontsize=12)
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "orthogonality.png")
plt.savefig(fig_path, dpi=150)
print(f"\nSaved: {fig_path}")

# ── 保存结果 ──────────────────────────────────────────────
orth_results = {
    "best_layer"          : BEST_LAYER,
    "fluency_layer"       : FLUENCY_LAYER,
    "cos_fact_fluency"    : cos_sim,
    "cos_per_layer"       : cos_per_layer,
    "verdict"             : verdict,
}
with open(os.path.join(RESULTS_DIR, "orthogonality.json"), "w") as f:
    json.dump(orth_results, f, indent=2)

print("\n=== Orthogonality Summary ===")
print(f"cos(w_fact @ Layer {BEST_LAYER}, w_fluency @ Layer {FLUENCY_LAYER}) = {cos_sim:.4f}")
print(f"Verdict: {verdict}")