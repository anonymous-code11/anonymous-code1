import json
import os
from itertools import permutations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from analysis_utils import (
    FOUR_DOMAIN_ORDER,
    build_wfact,
    compute_auc,
    ensure_dir,
    load_hidden,
)


SAVE_DIR = "./results/minimal_calibration"
FIG_DIR = "./figures"
N_LIST = [10, 20, 50, 100, 200, 400]
N_SEEDS = 10
RANDOM_SEED = 42
LAMBDA = 1.0

ensure_dir(SAVE_DIR)
ensure_dir(FIG_DIR)


def adapt_direction(wfact_src, h_correct_train, h_wrong_train, layer=16, lam=LAMBDA):
    delta = h_correct_train[:, layer, :].mean(axis=0) - h_wrong_train[:, layer, :].mean(axis=0)
    delta = delta / (np.linalg.norm(delta) + 1e-8)
    adapted = wfact_src + lam * delta
    return adapted / (np.linalg.norm(adapted) + 1e-8)


def kfold_splits(n_items, n_folds, rng):
    indices = rng.permutation(n_items)
    folds = np.array_split(indices, n_folds)
    splits = []
    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])
        splits.append((train_idx, test_idx))
    return splits


def run_domain_curves(h_correct, h_wrong, seeds):
    total = min(h_correct.shape[0], h_wrong.shape[0])
    curves = {n: [] for n in N_LIST}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(total)
        for n in N_LIST:
            if n < total:
                train_idx = perm[:n]
                test_idx = perm[n:]
                wfact = build_wfact(h_correct[train_idx], h_wrong[train_idx])
                auc = compute_auc(h_correct[test_idx], h_wrong[test_idx], wfact)
            else:
                fold_aucs = []
                for train_idx, test_idx in kfold_splits(total, n_folds=5, rng=rng):
                    wfact = build_wfact(h_correct[train_idx], h_wrong[train_idx])
                    fold_aucs.append(compute_auc(h_correct[test_idx], h_wrong[test_idx], wfact))
                auc = float(np.mean(fold_aucs))
            curves[n].append(float(auc))
    return curves


def summarize_curves(curves):
    summary = {}
    for n, values in curves.items():
        arr = np.array(values, dtype=float)
        summary[n] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "values": [float(v) for v in arr],
        }
    max_mean_auc = max(summary[n]["mean"] for n in N_LIST)
    threshold_auc = 0.9 * max_mean_auc
    n90 = next(n for n in N_LIST if summary[n]["mean"] >= threshold_auc)
    return summary, max_mean_auc, threshold_auc, n90


def source_informed_curve(source_wfact, h_correct, h_wrong, seeds):
    total = min(h_correct.shape[0], h_wrong.shape[0])
    curves = {n: [] for n in N_LIST}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(total)
        for n in N_LIST:
            if n < total:
                train_idx = perm[:n]
                test_idx = perm[n:]
                adapted = adapt_direction(source_wfact, h_correct[train_idx], h_wrong[train_idx])
                auc = compute_auc(h_correct[test_idx], h_wrong[test_idx], adapted)
            else:
                fold_aucs = []
                for train_idx, test_idx in kfold_splits(total, n_folds=5, rng=rng):
                    adapted = adapt_direction(source_wfact, h_correct[train_idx], h_wrong[train_idx])
                    fold_aucs.append(compute_auc(h_correct[test_idx], h_wrong[test_idx], adapted))
                auc = float(np.mean(fold_aucs))
            curves[n].append(float(auc))
    return curves


def main():
    seeds = [RANDOM_SEED + offset for offset in range(N_SEEDS)]
    domain_hidden = {
        domain: load_hidden(domain, limit=400 if domain != "TruthfulQA" else 400)
        for domain in FOUR_DOMAIN_ORDER
    }
    source_wfacts = {
        domain: build_wfact(*domain_hidden[domain])
        for domain in FOUR_DOMAIN_ORDER
    }

    print("Running within-domain calibration curves...")
    within_domain_results = {}
    for domain in FOUR_DOMAIN_ORDER:
        h_correct, h_wrong = domain_hidden[domain]
        curves = run_domain_curves(h_correct, h_wrong, seeds)
        summary, max_mean_auc, threshold_auc, n90 = summarize_curves(curves)
        within_domain_results[domain] = {
            "summary_by_n": summary,
            "max_mean_auc": max_mean_auc,
            "threshold_auc_90pct": threshold_auc,
            "min_n_for_90pct": n90,
        }
        print(
            f"  {domain:<14} max-mean AUC={max_mean_auc:.4f} "
            f"| 90% threshold={threshold_auc:.4f} | min N={n90}"
        )

    print("\nRunning source-informed adaptation sweeps...")
    with open("./results/domain_transferability/results.json") as handle:
        predictor_results = json.load(handle)
    pair_lookup = {item["pair"]: item for item in predictor_results["pair_records"]}

    pair_efficiency_records = []
    for source, target in permutations(FOUR_DOMAIN_ORDER, 2):
        h_correct, h_wrong = domain_hidden[target]
        curves = source_informed_curve(source_wfacts[source], h_correct, h_wrong, seeds)
        summary, _, _, n90 = summarize_curves(curves)
        pair_name = " <> ".join(sorted((source, target)))
        similarities = pair_lookup[pair_name]
        pair_efficiency_records.append(
            {
                "source": source,
                "target": target,
                "min_n_for_90pct": n90,
                "delta_vs_from_scratch": int(n90 - within_domain_results[target]["min_n_for_90pct"]),
                "embedding_similarity": similarities["embedding_similarity"],
                "token_jaccard": similarities["token_jaccard"],
                "format_similarity": similarities["format_similarity"],
                "summary_by_n": summary,
            }
        )
        print(
            f"  {source:<14} -> {target:<14} min N={n90:>3} "
            f"(scratch target N={within_domain_results[target]['min_n_for_90pct']})"
        )

    pair_metric_summary = {}
    for metric_name in ["embedding_similarity", "token_jaccard", "format_similarity"]:
        similarities = [record[metric_name] for record in pair_efficiency_records]
        min_ns = [record["min_n_for_90pct"] for record in pair_efficiency_records]
        deltas = [record["delta_vs_from_scratch"] for record in pair_efficiency_records]
        rho_min_n, p_min_n = spearmanr(similarities, min_ns)
        rho_delta, p_delta = spearmanr(similarities, deltas)
        pair_metric_summary[metric_name] = {
            "spearman_rho_vs_source_informed_min_n": float(rho_min_n),
            "spearman_p_vs_source_informed_min_n": float(p_min_n),
            "spearman_rho_vs_delta_vs_scratch": float(rho_delta),
            "spearman_p_vs_delta_vs_scratch": float(p_delta),
        }
        print(
            f"\nPair-level efficiency: {metric_name}"
            f"\n  min-N rho={rho_min_n:.3f}, p={p_min_n:.4f}"
            f"\n  delta-vs-scratch rho={rho_delta:.3f}, p={p_delta:.4f}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), sharey=True)
    colors = {
        "TruthfulQA": "#1f77b4",
        "FEVER": "#d62728",
        "MMLU-Medical": "#2ca02c",
        "ARC-Science": "#9467bd",
    }

    for ax, domain in zip(axes.flat, FOUR_DOMAIN_ORDER):
        summary = within_domain_results[domain]["summary_by_n"]
        means = [summary[n]["mean"] for n in N_LIST]
        stds = [summary[n]["std"] for n in N_LIST]
        threshold = within_domain_results[domain]["threshold_auc_90pct"]
        n90 = within_domain_results[domain]["min_n_for_90pct"]

        ax.errorbar(
            N_LIST,
            means,
            yerr=stds,
            fmt="o-",
            color=colors[domain],
            linewidth=2.0,
            markersize=7,
            capsize=4,
        )
        ax.axhline(threshold, color="#444444", linestyle="--", linewidth=1.0)
        ax.axvline(n90, color="#444444", linestyle=":", linewidth=1.0)
        ax.text(
            n90,
            threshold + 0.01,
            f"N90={n90}",
            fontsize=9,
            ha="left",
            va="bottom",
        )
        ax.set_title(
            f"{domain}\nmax mean AUC={within_domain_results[domain]['max_mean_auc']:.3f}",
            fontsize=11,
        )
        ax.set_xlabel("Labeled pairs (N)", fontsize=10)
        ax.set_ylabel("AUC", fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(N_LIST)
        ax.set_xticklabels([str(n) for n in N_LIST])
        ax.grid(alpha=0.25)
        ax.set_ylim(0.45, 1.0)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "minimal_calibration.png")
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")

    output = {
        "domains": FOUR_DOMAIN_ORDER,
        "n_list": N_LIST,
        "n_seeds": N_SEEDS,
        "within_domain": within_domain_results,
        "pair_efficiency": pair_efficiency_records,
        "pair_efficiency_metric_summary": pair_metric_summary,
        "notes": {
            "protocol": (
                "For N < 400, each seed trains on N labeled pairs and evaluates on the remaining "
                "pairs within the 400-example cache. For N = 400, we estimate generalization with "
                "seeded 5-fold cross-validation because no held-out remainder exists."
            )
        },
    }
    out_path = os.path.join(SAVE_DIR, "results.json")
    with open(out_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
