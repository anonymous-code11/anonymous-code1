import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


SAVE_DIR = "./results/minimal_calibration"
FIG_DIR = "./figures"
N_LIST = [10, 20, 50, 100, 200, 400]
N_SEEDS = 10
BEST_LAYER = 16
PCA_DIM = 128
RANDOM_SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


HIDDEN_PATHS = {
    "TruthfulQA": "./results/hidden_states.npz",
    "FEVER": "./results/expand_domains/FEVER_hidden.npz",
    "MMLU-Medical": "./results/expand_domains/MMLU-Medical_hidden.npz",
    "ARC-Science": "./results/expand_domains/ARC-Science_hidden.npz",
}


def build_wfact(h_correct, h_wrong, layer=BEST_LAYER, pca_dim=PCA_DIM):
    n = min(h_correct.shape[0], h_wrong.shape[0])
    x = np.concatenate([h_correct[:n, layer, :], h_wrong[:n, layer, :]], axis=0)
    y = np.array([1] * n + [0] * n)
    dim = min(pca_dim, x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=dim, random_state=42)
    x_proj = pca.fit_transform(x)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(x_proj, y)
    wfact = pca.components_.T @ clf.coef_[0]
    return wfact / (np.linalg.norm(wfact) + 1e-8)


def compute_auc(h_correct, h_wrong, wfact, layer=BEST_LAYER):
    n = min(h_correct.shape[0], h_wrong.shape[0])
    y = np.array([1] * n + [0] * n)
    scores = np.concatenate(
        [h_correct[:n, layer, :] @ wfact, h_wrong[:n, layer, :] @ wfact]
    )
    return float(roc_auc_score(y, scores))


def kfold_splits(n_items, n_folds, rng):
    indices = rng.permutation(n_items)
    folds = np.array_split(indices, n_folds)
    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])
        yield train_idx, test_idx


def load_hidden(path):
    data = np.load(path)
    return data["h_correct"], data["h_wrong"]


def summarize(values):
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "values": [float(v) for v in arr],
    }


def compute_tqa_curve():
    h_correct, h_wrong = load_hidden(HIDDEN_PATHS["TruthfulQA"])
    n_total = min(h_correct.shape[0], h_wrong.shape[0])
    n_eval = 100
    pool_size = n_total - n_eval
    h_correct_eval = h_correct[n_total - n_eval : n_total]
    h_wrong_eval = h_wrong[n_total - n_eval : n_total]
    h_correct_pool = h_correct[:pool_size]
    h_wrong_pool = h_wrong[:pool_size]

    results = {}
    for n in N_LIST:
        values = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(RANDOM_SEED + seed)
            idx = rng.choice(pool_size, n, replace=False)
            wfact = build_wfact(h_correct_pool[idx], h_wrong_pool[idx])
            values.append(compute_auc(h_correct_eval, h_wrong_eval, wfact))
        results[n] = summarize(values)
        print(f"TQA N={n:>3}: {results[n]['mean']:.4f} +/- {results[n]['std']:.4f}")
    return results


def compute_cv_n400(domain_name):
    h_correct, h_wrong = load_hidden(HIDDEN_PATHS[domain_name])
    h_correct = h_correct[:400]
    h_wrong = h_wrong[:400]
    values = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(RANDOM_SEED + seed)
        fold_aucs = []
        for train_idx, test_idx in kfold_splits(400, 5, rng):
            wfact = build_wfact(h_correct[train_idx], h_wrong[train_idx])
            fold_aucs.append(compute_auc(h_correct[test_idx], h_wrong[test_idx], wfact))
        values.append(float(np.mean(fold_aucs)))
    summary = summarize(values)
    print(f"{domain_name} N=400 CV: {summary['mean']:.4f} +/- {summary['std']:.4f}")
    return summary


def main():
    with open("./results/fewshot_adaptation/results.json") as handle:
        fewshot_results = json.load(handle)

    combined = {
        "TruthfulQA": {},
        "FEVER": {int(n): stats for n, stats in fewshot_results["FEVER"]["from_scratch"].items()},
        "MMLU-Medical": {
            int(n): stats for n, stats in fewshot_results["MMLU-Medical"]["from_scratch"].items()
        },
        "ARC-Science": {
            int(n): stats for n, stats in fewshot_results["ARC-Science"]["from_scratch"].items()
        },
    }

    combined["TruthfulQA"] = compute_tqa_curve()
    for domain in ["FEVER", "MMLU-Medical", "ARC-Science"]:
        combined[domain][400] = compute_cv_n400(domain)

    threshold_summary = {}
    for domain, curve in combined.items():
        max_mean = max(curve[n]["mean"] for n in N_LIST)
        threshold = 0.9 * max_mean
        n90 = next(n for n in N_LIST if curve[n]["mean"] >= threshold)
        threshold_summary[domain] = {
            "max_mean_auc": max_mean,
            "threshold_auc_90pct": threshold,
            "min_n_for_90pct": n90,
        }
        print(
            f"{domain:<14} max={max_mean:.4f} threshold={threshold:.4f} min N={n90}"
        )

    colors = {
        "TruthfulQA": "#1f77b4",
        "FEVER": "#d62728",
        "MMLU-Medical": "#2ca02c",
        "ARC-Science": "#9467bd",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), sharey=True)
    for ax, domain in zip(axes.flat, ["TruthfulQA", "FEVER", "MMLU-Medical", "ARC-Science"]):
        means = [combined[domain][n]["mean"] for n in N_LIST]
        stds = [combined[domain][n]["std"] for n in N_LIST]
        threshold = threshold_summary[domain]["threshold_auc_90pct"]
        n90 = threshold_summary[domain]["min_n_for_90pct"]

        ax.errorbar(
            N_LIST,
            means,
            yerr=stds,
            fmt="o-",
            linewidth=2.0,
            markersize=7,
            capsize=4,
            color=colors[domain],
        )
        ax.axhline(threshold, color="#444444", linestyle="--", linewidth=1.0)
        ax.axvline(n90, color="#444444", linestyle=":", linewidth=1.0)
        ax.text(n90, threshold + 0.01, f"N90={n90}", fontsize=9)
        ax.set_title(
            f"{domain}\nmax mean AUC={threshold_summary[domain]['max_mean_auc']:.3f}",
            fontsize=11,
        )
        ax.set_xlabel("Labeled pairs (N)", fontsize=10)
        ax.set_ylabel("AUC", fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(N_LIST)
        ax.set_xticklabels([str(n) for n in N_LIST])
        ax.set_ylim(0.45, 1.0)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "minimal_calibration.png")
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")

    output = {
        "n_list": N_LIST,
        "n_seeds": N_SEEDS,
        "curves": combined,
        "threshold_summary": threshold_summary,
        "notes": {
            "protocol": (
                "FEVER/MMLU-Medical/ARC-Science use the existing 10-seed fixed-eval results for "
                "N<=200 from fewshot_adaptation.py; the N=400 point is added via 10 repeated "
                "5-fold CV over the 400 cached pairs. TruthfulQA uses the same fixed-eval protocol "
                "with the 800-example cache (last 100 examples held out)."
            )
        },
    }
    out_path = os.path.join(SAVE_DIR, "summary_from_existing.json")
    with open(out_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
