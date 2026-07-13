import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import (
    DOMAIN_ORDER,
    effective_rank,
    ensure_dir,
    load_hidden,
    precompute_pca_inputs,
    build_wfact_from_precomputed,
)


SAVE_DIR = "./results/effective_rank_permutation"
FIG_DIR = "./figures"
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

ensure_dir(SAVE_DIR)
ensure_dir(FIG_DIR)

MMLU_ORDER = [
    "MMLU-Medical",
    "MMLU-Law",
    "MMLU-History",
    "MMLU-CS",
    "MMLU-Psychology",
]

MMLU_HIDDEN_PATHS = {
    "MMLU-Medical": "./results/expand_domains/MMLU-Medical_hidden.npz",
    "MMLU-Law": "./results/mmlu_controlled/MMLU-Law_hidden.npz",
    "MMLU-History": "./results/mmlu_controlled/MMLU-History_hidden.npz",
    "MMLU-CS": "./results/mmlu_controlled/MMLU-CS_hidden.npz",
    "MMLU-Psychology": "./results/mmlu_controlled/MMLU-Psychology_hidden.npz",
}
SEVEN_DOMAIN_WFACT_DIR = "./results/expand_domains/wfact"
MMLU_WFACT_DIR = "./results/mmlu_controlled/wfact"


def load_mmlu_hidden(domain_name):
    data = np.load(MMLU_HIDDEN_PATHS[domain_name])
    return data["h_correct"], data["h_wrong"]


def permutation_p_value(null_distribution, observed_value):
    null_distribution = np.asarray(null_distribution)
    return float((1 + np.sum(null_distribution <= observed_value)) / (len(null_distribution) + 1))


def summarize_distribution(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def run_permutation_test(name, domain_names, hidden_loader, observed_effective_rank):
    rng = np.random.default_rng(RANDOM_SEED)
    precomputed = {}

    print(f"\nPreparing PCA features for {name}...")
    for domain in domain_names:
        h_correct, h_wrong = hidden_loader(domain)
        precomputed[domain] = precompute_pca_inputs(h_correct, h_wrong)
        print(
            f"  {domain:<18} Xp={precomputed[domain]['x_proj'].shape} "
            f"labels={precomputed[domain]['labels'].shape}"
        )

    null_effective_ranks = []
    for perm_idx in range(N_PERMUTATIONS):
        wfacts = []
        for domain in domain_names:
            labels = precomputed[domain]["labels"].copy()
            rng.shuffle(labels)
            wfacts.append(
                build_wfact_from_precomputed(
                    precomputed[domain]["x_proj"],
                    precomputed[domain]["components"],
                    labels,
                )
            )
        stacked = np.stack(wfacts)
        null_effective_ranks.append(effective_rank(stacked))
        if (perm_idx + 1) % 100 == 0:
            print(f"  {name}: {perm_idx + 1}/{N_PERMUTATIONS}")

    stats = summarize_distribution(null_effective_ranks)
    p_value = permutation_p_value(null_effective_ranks, observed_effective_rank)
    print(
        f"\n{name}: observed={observed_effective_rank:.4f} | "
        f"null mean={stats['mean']:.4f} [{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}] | "
        f"one-sided p={p_value:.4f}"
    )
    return {
        "domain_names": domain_names,
        "observed_effective_rank": observed_effective_rank,
        "null_effective_ranks": [float(v) for v in null_effective_ranks],
        "null_summary": stats,
        "one_sided_p_value": p_value,
    }


def load_observed_effective_rank(wfact_dir, domain_names):
    wfacts = [np.load(os.path.join(wfact_dir, f"{domain}.npy")) for domain in domain_names]
    return effective_rank(np.stack(wfacts))


def main():
    with open("./results/expand_domains/results.json") as handle:
        expand_results = json.load(handle)
    with open("./results/mmlu_controlled/results.json") as handle:
        mmlu_results = json.load(handle)

    seven_observed = load_observed_effective_rank(SEVEN_DOMAIN_WFACT_DIR, DOMAIN_ORDER)
    mmlu_observed = load_observed_effective_rank(MMLU_WFACT_DIR, MMLU_ORDER)

    seven_domain_result = run_permutation_test(
        name="seven_domain",
        domain_names=DOMAIN_ORDER,
        hidden_loader=lambda name: load_hidden(name, limit=400),
        observed_effective_rank=seven_observed,
    )

    mmlu_result = run_permutation_test(
        name="mmlu_controlled",
        domain_names=MMLU_ORDER,
        hidden_loader=load_mmlu_hidden,
        observed_effective_rank=mmlu_observed,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (title, result) in zip(
        axes,
        [
            ("Seven domains", seven_domain_result),
            ("Five-domain MMLU", mmlu_result),
        ],
    ):
        values = np.array(result["null_effective_ranks"])
        ax.hist(values, bins=30, color="#4c72b0", alpha=0.8, density=False)
        ax.axvline(result["observed_effective_rank"], color="#c44e52", linewidth=2.2)
        ax.set_title(
            f"{title}\nobs={result['observed_effective_rank']:.2f}, "
            f"null 95% CI=[{result['null_summary']['ci95_low']:.2f}, {result['null_summary']['ci95_high']:.2f}], "
            f"p={result['one_sided_p_value']:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("Effective rank under label permutation", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "effective_rank_permutation.png")
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")

    output = {
        "seven_domain": seven_domain_result,
        "mmlu_controlled": mmlu_result,
        "expand_domain_summary": expand_results["summary"],
        "mmlu_summary": mmlu_results["summary"],
    }
    out_path = os.path.join(SAVE_DIR, "results.json")
    with open(out_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
