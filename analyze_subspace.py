
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "./results"
FIG_DIR     = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────
print("Loading hidden states...")
data = np.load(os.path.join(RESULTS_DIR, "hidden_states.npz"))
h_correct = data["h_correct"]  # [N, num_layers, hidden_size]
h_wrong   = data["h_wrong"]    # [N, num_layers, hidden_size]

N, num_layers, hidden_size = h_correct.shape
print(f"N={N}, layers={num_layers}, hidden_size={hidden_size}")

# ── 构造分类数据集 ────────────────────────────────────────
# X: hidden states, y: 1=factual, 0=hallucination
X_all = np.concatenate([h_correct, h_wrong], axis=0)  # [2N, L, H]
y_all = np.array([1]*N + [0]*N)

# ── 逐层线性probing ───────────────────────────────────────
print("Running linear probing per layer...")
layer_accs  = []
layer_aucs  = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for layer_idx in range(num_layers):
    X_layer = X_all[:, layer_idx, :]  # [2N, H]

    # 降维加速（保留95%方差）
    pca = PCA(n_components=min(128, X_layer.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_layer)

    accs, aucs = [], []
    for train_idx, val_idx in cv.split(X_pca, y_all):
        X_tr, X_val = X_pca[train_idx], X_pca[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_tr, y_tr)

        acc = clf.score(X_val, y_val)
        prob = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob)

        accs.append(acc)
        aucs.append(auc)

    layer_accs.append(np.mean(accs))
    layer_aucs.append(np.mean(aucs))
    print(f"  Layer {layer_idx:2d}: acc={np.mean(accs):.4f}, auc={np.mean(aucs):.4f}")

# ── 核心图：各层probing准确率 ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

layers = list(range(num_layers))

axes[0].plot(layers, layer_accs, "b-o", markersize=4, linewidth=2)
axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance (0.5)")
axes[0].set_xlabel("Layer Index", fontsize=13)
axes[0].set_ylabel("5-Fold CV Accuracy", fontsize=13)
axes[0].set_title("Linear Probing: Factual vs Hallucination\nper Layer (Llama-3-8B, TruthfulQA)", fontsize=13)
axes[0].legend()
axes[0].set_ylim(0.45, 1.02)
axes[0].grid(True, alpha=0.3)

axes[1].plot(layers, layer_aucs, "r-o", markersize=4, linewidth=2)
axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance (0.5)")
axes[1].set_xlabel("Layer Index", fontsize=13)
axes[1].set_ylabel("AUC-ROC", fontsize=13)
axes[1].set_title("AUC-ROC per Layer", fontsize=13)
axes[1].legend()
axes[1].set_ylim(0.45, 1.02)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "probing_per_layer.png")
plt.savefig(fig_path, dpi=150)
print(f"\nSaved: {fig_path}")

# ── PCA可视化（选最佳层） ─────────────────────────────────
best_layer = int(np.argmax(layer_aucs))
print(f"\nBest layer: {best_layer} (AUC={layer_aucs[best_layer]:.4f})")

X_best = X_all[:, best_layer, :]
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_best)

fig2, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_2d[y_all==1, 0], X_2d[y_all==1, 1],
           c="steelblue", alpha=0.5, s=20, label="Factual")
ax.scatter(X_2d[y_all==0, 0], X_2d[y_all==0, 1],
           c="tomato",    alpha=0.5, s=20, label="Hallucination")
ax.set_title(f"PCA Visualization — Layer {best_layer} (Best)\nLlama-3-8B, TruthfulQA", fontsize=13)
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

fig2_path = os.path.join(FIG_DIR, "pca_best_layer.png")
plt.savefig(fig2_path, dpi=150)
print(f"Saved: {fig2_path}")

# ── 保存数值结果 ──────────────────────────────────────────
results = {
    "layer_accs": layer_accs,
    "layer_aucs": layer_aucs,
    "best_layer": best_layer,
    "best_auc"  : layer_aucs[best_layer],
    "best_acc"  : layer_accs[best_layer],
}
with open(os.path.join(RESULTS_DIR, "probing_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Summary ===")
print(f"Best Layer : {best_layer}")
print(f"Best ACC   : {layer_accs[best_layer]:.4f}")
print(f"Best AUC   : {layer_aucs[best_layer]:.4f}")
print(f"Chance AUC : 0.5000")
print(f"Gap        : +{layer_aucs[best_layer]-0.5:.4f}")