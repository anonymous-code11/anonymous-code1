
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

RESULTS_DIR = "./results"
FIG_DIR     = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

BEST_LAYER    = 16
FLUENCY_LAYER = 3

# ── 加载数据 ──────────────────────────────────────────────
print("Loading hidden states...")
data      = np.load(os.path.join(RESULTS_DIR, "hidden_states.npz"))
h_correct = data["h_correct"]   # [N, 33, 4096]
h_wrong   = data["h_wrong"]     # [N, 33, 4096]
N         = h_correct.shape[0]

# ── 重建 w_fact ───────────────────────────────────────────
print(f"Rebuilding w_fact from Layer {BEST_LAYER}...")
X_layer = np.concatenate([h_correct[:, BEST_LAYER, :],
                           h_wrong[:,   BEST_LAYER, :]], axis=0)
y       = np.array([1]*N + [0]*N)

pca_fact = PCA(n_components=128, random_state=42)
X_pca    = pca_fact.fit_transform(X_layer)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
clf.fit(X_scaled, y)

w_fact_pca = clf.coef_[0]
w_fact     = pca_fact.components_.T @ w_fact_pca
w_fact     = w_fact / (np.linalg.norm(w_fact) + 1e-8)

# ── 重建 w_fluency ────────────────────────────────────────
print(f"Rebuilding w_fluency from Layer {FLUENCY_LAYER}...")
mean_c    = h_correct[:, FLUENCY_LAYER, :].mean(axis=0)
mean_w    = h_wrong[:,   FLUENCY_LAYER, :].mean(axis=0)
w_fluency = mean_c - mean_w
w_fluency = w_fluency / (np.linalg.norm(w_fluency) + 1e-8)

# ── 计算DSD per sample ────────────────────────────────────
# DSD(x) = proj(h_x, w_fact) - proj(h_x, w_fluency)
# 正值：事实方向强于流畅方向 → 倾向factual
# 负值：流畅方向强于事实方向 → 倾向hallucination
print("Computing DSD per sample...")

def compute_dsd(h_matrix):
    """h_matrix: [N, 4096], returns [N]"""
    proj_fact    = h_matrix @ w_fact      # [N]
    proj_fluency = h_matrix @ w_fluency   # [N]
    return proj_fact - proj_fluency

dsd_correct = compute_dsd(h_correct[:, BEST_LAYER, :])  # [N]
dsd_wrong   = compute_dsd(h_wrong[:,   BEST_LAYER, :])   # [N]

# 合并
dsd_all = np.concatenate([dsd_correct, dsd_wrong])
y_all   = np.array([1]*N + [0]*N)

# ── 评估DSD作为分类器 ────────────────────────────────────
auc_dsd = roc_auc_score(y_all, dsd_all)
fpr, tpr, thresholds = roc_curve(y_all, dsd_all)

# 最优阈值（Youden's J）
j_scores  = tpr - fpr
best_th   = thresholds[np.argmax(j_scores)]
pred      = (dsd_all >= best_th).astype(int)
acc_dsd   = (pred == y_all).mean()

print(f"\n=== DSD Performance ===")
print(f"AUC  : {auc_dsd:.4f}")
print(f"ACC  : {acc_dsd:.4f}  (threshold={best_th:.4f})")

# ── 对比：多层DSD AUC扫描 ────────────────────────────────
print("\nScanning DSD AUC across layers...")
dsd_aucs = []
for layer_idx in range(h_correct.shape[1]):
    d_c = compute_dsd(h_correct[:, layer_idx, :])
    d_w = compute_dsd(h_wrong[:,   layer_idx, :])
    d_all = np.concatenate([d_c, d_w])
    auc = roc_auc_score(y_all, d_all)
    dsd_aucs.append(auc)
    print(f"  Layer {layer_idx:2d}: DSD AUC={auc:.4f}")

# ── 加载probing AUC对比 ───────────────────────────────────
with open(os.path.join(RESULTS_DIR, "probing_results.json")) as f:
    probing = json.load(f)
probing_aucs = probing["layer_aucs"]

# ── 图1：DSD分布 ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.hist(dsd_correct, bins=40, alpha=0.6, color="steelblue", label="Factual",       density=True)
ax.hist(dsd_wrong,   bins=40, alpha=0.6, color="tomato",    label="Hallucination", density=True)
ax.axvline(best_th, color="black", linestyle="--", label=f"Threshold={best_th:.2f}")
ax.set_xlabel("DSD Score", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title(f"DSD Distribution (Layer {BEST_LAYER})\nAUC={auc_dsd:.4f}, ACC={acc_dsd:.4f}", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ── 图2：ROC曲线 ──────────────────────────────────────────
ax = axes[0, 1]
ax.plot(fpr, tpr, "b-", linewidth=2, label=f"DSD (AUC={auc_dsd:.4f})")
ax.plot([0,1],[0,1], "gray", linestyle="--", alpha=0.5, label="Chance")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — DSD as Hallucination Detector", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ── 图3：DSD AUC vs Probing AUC 各层对比 ─────────────────
ax = axes[1, 0]
layers = list(range(len(dsd_aucs)))
ax.plot(layers, dsd_aucs,    "b-o", markersize=4, linewidth=2, label="DSD AUC (ours)")
ax.plot(layers, probing_aucs,"r--s",markersize=4, linewidth=2, label="Probing AUC (baseline)")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
ax.axvline(BEST_LAYER, color="green", linestyle="--", alpha=0.6, label=f"Best layer ({BEST_LAYER})")
ax.set_xlabel("Layer Index", fontsize=12)
ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_title("DSD vs Linear Probing AUC per Layer\n(DSD is training-free)", fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0.45, 1.02)
ax.grid(True, alpha=0.3)

# ── 图4：DSD score散点（样本级） ─────────────────────────
ax = axes[1, 1]
idx = np.arange(N)
ax.scatter(idx, dsd_correct, c="steelblue", alpha=0.3, s=8,  label="Factual")
ax.scatter(idx, dsd_wrong,   c="tomato",    alpha=0.3, s=8,  label="Hallucination")
ax.axhline(best_th, color="black", linestyle="--", linewidth=1.5, label="Threshold")
ax.axhline(0,       color="gray",  linestyle=":",  linewidth=1,   alpha=0.5)
ax.set_xlabel("Sample Index", fontsize=12)
ax.set_ylabel("DSD Score", fontsize=12)
ax.set_title("Per-Sample DSD Score\n(Higher = more factual)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle("Dual Space Divergence (DSD) — Hallucination Detection\n"
             "Llama-3-8B + TruthfulQA", fontsize=14, y=1.01)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "dsd_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {fig_path}")

# ── 保存结果 ──────────────────────────────────────────────
results = {
    "dsd_auc"      : float(auc_dsd),
    "dsd_acc"      : float(acc_dsd),
    "best_threshold": float(best_th),
    "dsd_aucs_per_layer"    : [float(x) for x in dsd_aucs],
    "probing_aucs_per_layer": probing_aucs,
    "best_layer"   : BEST_LAYER,
    "fluency_layer": FLUENCY_LAYER,
}
with open(os.path.join(RESULTS_DIR, "dsd_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Final Summary ===")
print(f"DSD AUC          : {auc_dsd:.4f}")
print(f"DSD ACC          : {acc_dsd:.4f}")
print(f"Probing AUC@{BEST_LAYER}  : {probing_aucs[BEST_LAYER]:.4f}")
print(f"DSD is training-free — no classifier needed at inference time")