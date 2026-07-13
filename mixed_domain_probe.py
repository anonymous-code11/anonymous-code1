"""Multi-domain mixed probe: train on a balanced mixture of all clean domains and compare with per-domain probes."""

import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis_utils import (
    load_hidden,
    build_wfact,
    BEST_LAYER,
    PCA_DIM,
    RANDOM_SEED,
    FOUR_DOMAIN_ORDER,
)

# ── 配置 ──────────────────────────────────────────────────────────────────────
N_PER_DOMAIN = 100    # 每域贡献 100 对到混合池 (共 400 对，与 per-domain 一致)
N_EVAL = 400          # 评估时每域用全部 400 对
LAYER = BEST_LAYER
SAVE_DIR = "./results/mixed_domain_probe"
os.makedirs(SAVE_DIR, exist_ok=True)

DOMAINS = FOUR_DOMAIN_ORDER  # ["TruthfulQA", "FEVER", "MMLU-Medical", "ARC-Science"]


def compute_auc(h_correct, h_wrong, wfact, layer=LAYER):
    """计算 wfact 在给定数据上的 Sfact AUC"""
    n = min(h_correct.shape[0], h_wrong.shape[0])
    y = np.array([1] * n + [0] * n)
    scores = np.concatenate([
        h_correct[:n, layer, :] @ wfact,
        h_wrong[:n, layer, :] @ wfact,
    ])
    return float(roc_auc_score(y, scores))


def build_wfact_from_flat(X, y, pca_dim=PCA_DIM):
    """从已经 concat 好的 X, y 构建 wfact（不需要 h_correct/h_wrong 分开传）"""
    dim = min(pca_dim, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=dim, random_state=RANDOM_SEED)
    Xp = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)
    clf.fit(Xp, y)
    w = pca.components_.T @ clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-8)


def main():
    print("=" * 60)
    print("Multi-Domain Mixed Probe")
    print("=" * 60)

    # ── Step 1: 加载所有域的 hidden states ────────────────────────────────────
    domain_data = {}
    for dname in DOMAINS:
        print(f"Loading {dname}...")
        hc, hw = load_hidden(dname, limit=N_EVAL)
        domain_data[dname] = (hc, hw)
        print(f"  shape: correct={hc.shape}, wrong={hw.shape}")

    # ── Step 2: 训练 per-domain probes（baseline） ────────────────────────────
    print("\n--- Per-domain probes (baseline) ---")
    per_domain_wfact = {}
    per_domain_within_auc = {}
    for dname in DOMAINS:
        hc, hw = domain_data[dname]
        w = build_wfact(hc, hw, layer=LAYER, pca_dim=PCA_DIM)
        per_domain_wfact[dname] = w
        auc = compute_auc(hc, hw, w, layer=LAYER)
        per_domain_within_auc[dname] = auc
        print(f"  {dname}: within-domain AUC = {auc:.4f}")

    # ── Step 3: 构建混合训练集 ────────────────────────────────────────────────
    print(f"\n--- Building mixed training set ({N_PER_DOMAIN} pairs x {len(DOMAINS)} domains) ---")
    rng = np.random.RandomState(RANDOM_SEED)

    mixed_X_list = []
    mixed_y_list = []
    for dname in DOMAINS:
        hc, hw = domain_data[dname]
        n_avail = min(hc.shape[0], hw.shape[0])
        indices = rng.choice(n_avail, size=min(N_PER_DOMAIN, n_avail), replace=False)
        hc_sub = hc[indices, LAYER, :]
        hw_sub = hw[indices, LAYER, :]
        mixed_X_list.append(hc_sub)
        mixed_X_list.append(hw_sub)
        mixed_y_list.append(np.ones(len(indices)))
        mixed_y_list.append(np.zeros(len(indices)))
        print(f"  {dname}: contributed {len(indices)} pairs")

    mixed_X = np.concatenate(mixed_X_list, axis=0)
    mixed_y = np.concatenate(mixed_y_list, axis=0)
    print(f"  Total mixed training set: {mixed_X.shape[0]} samples")

    # ── Step 4: 训练 mixed probe ──────────────────────────────────────────────
    print("\n--- Training mixed-domain probe ---")
    w_mixed = build_wfact_from_flat(mixed_X, mixed_y, pca_dim=PCA_DIM)
    print(f"  w_mixed shape: {w_mixed.shape}, norm: {np.linalg.norm(w_mixed):.4f}")

    # ── Step 5: 评估 mixed probe 在所有域上的 AUC ─────────────────────────────
    print("\n--- Evaluation: mixed probe vs. per-domain probe ---")
    print(f"{'Domain':<18} {'Per-domain AUC':>15} {'Mixed AUC':>12} {'Zero-xfer AUC':>15} {'Δ(mix-xfer)':>12}")
    print("-" * 75)

    results = {"config": {
        "n_per_domain_mix": N_PER_DOMAIN,
        "n_eval": N_EVAL,
        "layer": LAYER,
        "pca_dim": PCA_DIM,
        "domains": DOMAINS,
    }}

    mixed_aucs = {}
    zero_transfer_aucs = {}  # 用 TruthfulQA 的 wfact 评估其他域（典型 zero-transfer）

    w_tqa = per_domain_wfact["TruthfulQA"]

    for dname in DOMAINS:
        hc, hw = domain_data[dname]
        # Per-domain (within)
        auc_per = per_domain_within_auc[dname]
        # Mixed probe
        auc_mix = compute_auc(hc, hw, w_mixed, layer=LAYER)
        mixed_aucs[dname] = auc_mix
        # Zero-transfer from TruthfulQA
        auc_zero = compute_auc(hc, hw, w_tqa, layer=LAYER)
        zero_transfer_aucs[dname] = auc_zero

        delta = auc_mix - auc_zero
        print(f"{dname:<18} {auc_per:>15.4f} {auc_mix:>12.4f} {auc_zero:>15.4f} {delta:>+12.4f}")

    # ── Step 6: 完整 cross-domain AUC 矩阵（per-domain vs mixed） ─────────────
    print("\n--- Full cross-domain AUC matrix (per-domain probes) ---")
    auc_matrix_per = {}
    for src in DOMAINS:
        for tgt in DOMAINS:
            hc, hw = domain_data[tgt]
            auc = compute_auc(hc, hw, per_domain_wfact[src], layer=LAYER)
            auc_matrix_per[f"{src}->{tgt}"] = auc

    # ── Step 7: Cosine similarity: mixed vs per-domain ────────────────────────
    print("\n--- Cosine similarity: w_mixed vs. per-domain wfact ---")
    cos_mixed_vs_per = {}
    for dname in DOMAINS:
        cos = float(np.dot(w_mixed, per_domain_wfact[dname]))
        cos_mixed_vs_per[dname] = cos
        print(f"  cos(w_mixed, w_{dname}) = {cos:.4f}")

    # ── Step 8: 汇总结果 ──────────────────────────────────────────────────────
    print("\n--- Summary ---")
    mean_per_domain = np.mean(list(per_domain_within_auc.values()))
    mean_mixed = np.mean(list(mixed_aucs.values()))
    mean_zero = np.mean([zero_transfer_aucs[d] for d in DOMAINS if d != "TruthfulQA"])

    print(f"  Mean per-domain within AUC:  {mean_per_domain:.4f}")
    print(f"  Mean mixed-probe AUC:        {mean_mixed:.4f}")
    print(f"  Mean zero-transfer AUC:      {mean_zero:.4f}")
    print(f"  → Mixed probe {'improves' if mean_mixed > mean_zero else 'does not improve'} over zero-transfer")
    print(f"  → Per-domain {'outperforms' if mean_per_domain > mean_mixed else 'does not outperform'} mixed probe")

    results["per_domain_within_auc"] = per_domain_within_auc
    results["mixed_probe_auc"] = mixed_aucs
    results["zero_transfer_from_tqa"] = zero_transfer_aucs
    results["auc_matrix_per_domain"] = auc_matrix_per
    results["cos_mixed_vs_per_domain"] = cos_mixed_vs_per
    results["summary"] = {
        "mean_per_domain_auc": round(mean_per_domain, 4),
        "mean_mixed_probe_auc": round(mean_mixed, 4),
        "mean_zero_transfer_auc": round(mean_zero, 4),
    }

    save_path = os.path.join(SAVE_DIR, "results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()