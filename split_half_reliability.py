"""Split-half direction reliability: measure within-domain estimation stability vs cross-domain orthogonality."""

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
N_SAMPLES = 400       # 每域总样本数
N_SPLITS = 10         # 随机分半次数
LAYER = BEST_LAYER
SAVE_DIR = "./results/split_half_reliability"
os.makedirs(SAVE_DIR, exist_ok=True)

DOMAINS = FOUR_DOMAIN_ORDER  # ["TruthfulQA", "FEVER", "MMLU-Medical", "ARC-Science"]


def build_wfact_subset(hc, hw, indices, layer=LAYER, pca_dim=PCA_DIM):
    """从指定 indices 的子集构建 wfact"""
    hc_sub = hc[indices]
    hw_sub = hw[indices]
    n = len(indices)
    X = np.concatenate([hc_sub[:, layer, :], hw_sub[:, layer, :]], axis=0)
    y = np.array([1] * n + [0] * n)
    dim = min(pca_dim, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=dim, random_state=42)
    Xp = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(Xp, y)
    w = pca.components_.T @ clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-8)


def main():
    print("=" * 60)
    print("Split-Half Direction Reliability")
    print("=" * 60)

    # ── Step 1: 加载所有域 ────────────────────────────────────────────────────
    domain_data = {}
    for dname in DOMAINS:
        print(f"Loading {dname}...")
        hc, hw = load_hidden(dname, limit=N_SAMPLES)
        domain_data[dname] = (hc, hw)
        n = min(hc.shape[0], hw.shape[0])
        print(f"  available pairs: {n}")

    # ── Step 2: Split-half within each domain ─────────────────────────────────
    print(f"\n--- Split-half reliability ({N_SPLITS} random splits) ---")

    within_cosines = {}  # domain -> list of cosines across splits

    for dname in DOMAINS:
        hc, hw = domain_data[dname]
        n = min(hc.shape[0], hw.shape[0])
        cosines = []

        for seed in range(N_SPLITS):
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n)
            half = n // 2
            idx_a = perm[:half]
            idx_b = perm[half:2 * half]

            w_a = build_wfact_subset(hc, hw, idx_a)
            w_b = build_wfact_subset(hc, hw, idx_b)
            cos = float(np.dot(w_a, w_b))
            cosines.append(cos)

        within_cosines[dname] = cosines
        mean_cos = np.mean(cosines)
        std_cos = np.std(cosines)
        print(f"  {dname}: split-half cos = {mean_cos:.4f} ± {std_cos:.4f}  (range: [{min(cosines):.4f}, {max(cosines):.4f}])")

    # ── Step 3: Cross-domain cosine (full 400 pairs, for comparison) ──────────
    print(f"\n--- Cross-domain cosine (full {N_SAMPLES} pairs, for contrast) ---")

    full_wfact = {}
    for dname in DOMAINS:
        hc, hw = domain_data[dname]
        full_wfact[dname] = build_wfact(hc, hw, layer=LAYER, pca_dim=PCA_DIM)

    cross_cosines = {}
    for i, d1 in enumerate(DOMAINS):
        for j, d2 in enumerate(DOMAINS):
            if i < j:
                cos = float(np.dot(full_wfact[d1], full_wfact[d2]))
                cross_cosines[f"{d1} vs {d2}"] = cos
                print(f"  {d1} vs {d2}: cos = {cos:.4f}")

    # ── Step 4: AUC of split-half directions ──────────────────────────────────
    print(f"\n--- Split-half AUC consistency ---")
    split_half_aucs = {}

    for dname in DOMAINS:
        hc, hw = domain_data[dname]
        n = min(hc.shape[0], hw.shape[0])

        # Use seed=0 split
        rng = np.random.RandomState(0)
        perm = rng.permutation(n)
        half = n // 2
        idx_a = perm[:half]
        idx_b = perm[half:2 * half]

        w_a = build_wfact_subset(hc, hw, idx_a)
        w_b = build_wfact_subset(hc, hw, idx_b)

        # Evaluate w_a on half B (and vice versa)
        n_b = len(idx_b)
        y_b = np.array([1] * n_b + [0] * n_b)
        scores_b = np.concatenate([
            hc[idx_b, LAYER, :] @ w_a,
            hw[idx_b, LAYER, :] @ w_a,
        ])
        auc_a_on_b = float(roc_auc_score(y_b, scores_b))

        n_a = len(idx_a)
        y_a = np.array([1] * n_a + [0] * n_a)
        scores_a = np.concatenate([
            hc[idx_a, LAYER, :] @ w_b,
            hw[idx_a, LAYER, :] @ w_b,
        ])
        auc_b_on_a = float(roc_auc_score(y_a, scores_a))

        split_half_aucs[dname] = {
            "auc_a_on_b": auc_a_on_b,
            "auc_b_on_a": auc_b_on_a,
            "mean": (auc_a_on_b + auc_b_on_a) / 2,
        }
        print(f"  {dname}: AUC(w_A→B)={auc_a_on_b:.4f}, AUC(w_B→A)={auc_b_on_a:.4f}, mean={split_half_aucs[dname]['mean']:.4f}")

    # ── Step 5: Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    mean_within = np.mean([np.mean(v) for v in within_cosines.values()])
    mean_cross = np.mean(list(cross_cosines.values()))

    print(f"  Mean within-domain split-half cosine:  {mean_within:.4f}")
    print(f"  Mean cross-domain cosine:              {mean_cross:.4f}")
    print(f"  Ratio (within/cross):                  {mean_within/max(mean_cross, 1e-8):.1f}x")
    print()

    if mean_within > 0.5 and mean_cross < 0.15:
        print("  INTERPRETATION: Directions are reliably estimated within each domain")
        print("  (high split-half cosine) but genuinely orthogonal across domains")
        print("  (low cross-domain cosine). This rules out the 'noisy direction' explanation.")
    else:
        print("  INTERPRETATION: Check results — pattern differs from expectation.")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "config": {
            "n_samples": N_SAMPLES,
            "n_splits": N_SPLITS,
            "layer": LAYER,
            "pca_dim": PCA_DIM,
            "domains": DOMAINS,
        },
        "within_domain_split_half_cosine": {
            dname: {
                "mean": round(float(np.mean(cosines)), 4),
                "std": round(float(np.std(cosines)), 4),
                "all_splits": [round(c, 4) for c in cosines],
            }
            for dname, cosines in within_cosines.items()
        },
        "cross_domain_cosine": cross_cosines,
        "split_half_aucs": split_half_aucs,
        "summary": {
            "mean_within_cosine": round(mean_within, 4),
            "mean_cross_cosine": round(mean_cross, 4),
        },
    }

    save_path = os.path.join(SAVE_DIR, "results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()