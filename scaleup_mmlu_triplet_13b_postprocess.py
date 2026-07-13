import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


SAVE_DIR = "./results/scaleup_13b_triplet"
FIG_DIR = "./figures"
DOMAIN_ORDER = ["MMLU-Medical", "MMLU-Law", "MMLU-CS"]
RANDOM_SEED = 42
PCA_DIM = 128
N_PERMUTATIONS = 1000


def normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-8)


def build_wfact(h_correct, h_wrong, layer):
    n = min(h_correct.shape[0], h_wrong.shape[0])
    x = np.concatenate([h_correct[:n, layer, :], h_wrong[:n, layer, :]], axis=0)
    y = np.array([1] * n + [0] * n)
    dim = min(PCA_DIM, x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=dim, random_state=RANDOM_SEED)
    x_proj = pca.fit_transform(x)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)
    clf.fit(x_proj, y)
    return normalize(pca.components_.T @ clf.coef_[0])


def compute_auc(h_correct, h_wrong, wfact, layer):
    n = min(h_correct.shape[0], h_wrong.shape[0])
    y = np.array([1] * n + [0] * n)
    scores = np.concatenate([h_correct[:n, layer, :] @ wfact, h_wrong[:n, layer, :] @ wfact])
    return float(roc_auc_score(y, scores))


def effective_rank(matrix):
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    probs = s**2
    probs = probs / probs.sum()
    return float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))


def select_best_layer(h_correct, h_wrong):
    n = min(h_correct.shape[0], h_wrong.shape[0])
    x_all = np.concatenate([h_correct[:n], h_wrong[:n]], axis=0)
    y_all = np.array([1] * n + [0] * n)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    layer_aucs = []

    for layer_idx in range(x_all.shape[1]):
        x_layer = x_all[:, layer_idx, :]
        dim = min(PCA_DIM, x_layer.shape[0] - 1, x_layer.shape[1])
        pca = PCA(n_components=dim, random_state=RANDOM_SEED)
        x_proj = pca.fit_transform(x_layer)
        aucs = []
        for tr_idx, te_idx in cv.split(x_proj, y_all):
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)
            clf.fit(x_proj[tr_idx], y_all[tr_idx])
            probs = clf.predict_proba(x_proj[te_idx])[:, 1]
            aucs.append(roc_auc_score(y_all[te_idx], probs))
        layer_aucs.append(float(np.mean(aucs)))
        if (layer_idx + 1) % 10 == 0 or layer_idx == x_all.shape[1] - 1:
            print(f"[layer-select] finished {layer_idx + 1}/{x_all.shape[1]} layers")

    best_layer = int(np.argmax(layer_aucs))
    return best_layer, layer_aucs


def permutation_null(hidden, layer):
    rng = np.random.default_rng(RANDOM_SEED)
    null_values = []
    prepared = {}
    for domain_name, (h_correct, h_wrong) in hidden.items():
        n = min(h_correct.shape[0], h_wrong.shape[0])
        x = np.concatenate([h_correct[:n, layer, :], h_wrong[:n, layer, :]], axis=0)
        y = np.array([1] * n + [0] * n)
        dim = min(PCA_DIM, x.shape[0] - 1, x.shape[1])
        pca = PCA(n_components=dim, random_state=RANDOM_SEED)
        x_proj = pca.fit_transform(x)
        prepared[domain_name] = (x_proj, pca.components_, y)

    for idx in range(N_PERMUTATIONS):
        wfacts = []
        for domain_name in DOMAIN_ORDER:
            x_proj, components, labels = prepared[domain_name]
            permuted = labels.copy()
            rng.shuffle(permuted)
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)
            clf.fit(x_proj, permuted)
            wfacts.append(normalize(components.T @ clf.coef_[0]))
        null_values.append(effective_rank(np.stack(wfacts)))
        if (idx + 1) % 100 == 0 or idx == N_PERMUTATIONS - 1:
            print(f"[permutation] finished {idx + 1}/{N_PERMUTATIONS}")
    return np.array(null_values)


def load_hidden():
    hidden = {}
    for domain_name in DOMAIN_ORDER:
        path = os.path.join(SAVE_DIR, f"{domain_name}_hidden.npz")
        data = np.load(path)
        hidden[domain_name] = (data["h_correct"], data["h_wrong"])
        print(f"[cache] {domain_name}: {data['h_correct'].shape}")
    return hidden


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    hidden = load_hidden()
    best_layer, layer_aucs = select_best_layer(*hidden["MMLU-Medical"])
    print(f"Selected best layer on MMLU-Medical: {best_layer} (AUC={max(layer_aucs):.4f})")

    wfacts = {}
    for domain_name in DOMAIN_ORDER:
        wfacts[domain_name] = build_wfact(*hidden[domain_name], layer=best_layer)

    cos_matrix = np.zeros((len(DOMAIN_ORDER), len(DOMAIN_ORDER)))
    auc_matrix_raw = np.zeros_like(cos_matrix)
    auc_matrix_abs = np.zeros_like(cos_matrix)
    for i, src in enumerate(DOMAIN_ORDER):
        for j, tgt in enumerate(DOMAIN_ORDER):
            cos_matrix[i, j] = float(np.dot(wfacts[src], wfacts[tgt]))
            raw_auc = compute_auc(*hidden[tgt], wfacts[src], layer=best_layer)
            auc_matrix_raw[i, j] = raw_auc
            auc_matrix_abs[i, j] = max(raw_auc, 1.0 - raw_auc)

    stacked = np.stack([wfacts[name] for name in DOMAIN_ORDER])
    observed_rank = effective_rank(stacked)
    null_ranks = permutation_null(hidden, best_layer)
    p_value = float((1 + np.sum(null_ranks <= observed_rank)) / (len(null_ranks) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = [name.replace("MMLU-", "") for name in DOMAIN_ORDER]
    im1 = axes[0].imshow(cos_matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_yticklabels(labels)
    axes[0].set_title("13B controlled domain cosine")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[0].text(j, i, f"{cos_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    im2 = axes[1].imshow(auc_matrix_abs, vmin=0.45, vmax=0.95, cmap="YlOrRd")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_yticklabels(labels)
    axes[1].set_title("13B cross-domain transfer AUC")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1].text(j, i, f"{auc_matrix_abs[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "scaleup_13b_triplet.png")
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")

    output = {
        "domains": DOMAIN_ORDER,
        "best_layer": best_layer,
        "layer_aucs_medical": layer_aucs,
        "cos_matrix": cos_matrix.tolist(),
        "auc_matrix_raw": auc_matrix_raw.tolist(),
        "auc_matrix_abs": auc_matrix_abs.tolist(),
        "summary": {
            "mean_off_diag_cos": float(np.mean([cos_matrix[i, j] for i in range(3) for j in range(3) if i != j])),
            "mean_off_diag_abs_auc": float(
                np.mean([auc_matrix_abs[i, j] for i in range(3) for j in range(3) if i != j])
            ),
            "effective_rank": observed_rank,
            "null_mean": float(null_ranks.mean()),
            "null_ci95_low": float(np.percentile(null_ranks, 2.5)),
            "null_ci95_high": float(np.percentile(null_ranks, 97.5)),
            "permutation_p_value": p_value,
        },
    }
    out_path = os.path.join(SAVE_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"Saved {out_path}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
