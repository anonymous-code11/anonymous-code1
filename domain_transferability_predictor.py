import json
import os
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

from analysis_utils import (
    DOMAIN_ORDER,
    FORMAT_FEATURES,
    cosine_similarity,
    ensure_dir,
    load_prompt_texts,
    pair_key,
    vocab_jaccard,
)


SAVE_DIR = "./results/domain_transferability"
FIG_DIR = "./figures"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS_PER_DOMAIN = 400
BATCH_SIZE = 32

ensure_dir(SAVE_DIR)
ensure_dir(FIG_DIR)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    return masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)


def encode_domain_centroid(
    texts,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu().numpy())
    stacked = np.concatenate(embeddings, axis=0)
    centroid = stacked.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    return centroid


def save_scatter(ax, x_vals, y_vals, labels, title, xlabel):
    ax.scatter(x_vals, y_vals, s=55, alpha=0.85, color="#2368a2")
    if len(np.unique(x_vals)) > 1:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        xs = np.linspace(min(x_vals), max(x_vals), 100)
        ax.plot(xs, slope * xs + intercept, color="#c44e52", linewidth=1.8, alpha=0.8)
    for label, x_val, y_val in zip(labels, x_vals, y_vals):
        if label == "ARC-Science <> MMLU-Medical":
            ax.annotate(
                "MMLU-Med <> ARC-Sci",
                (x_val, y_val),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Pairwise $w_{\\mathrm{fact}}$ cosine", fontsize=10)
    ax.grid(alpha=0.25)


def main():
    with open("./results/expand_domains/results.json") as handle:
        expand_results = json.load(handle)

    cos_matrix = np.array(expand_results["cos_matrix"])
    auc_matrix = np.array(expand_results["auc_matrix"])
    domain_to_idx = {domain: idx for idx, domain in enumerate(DOMAIN_ORDER)}

    print("Loading prompt texts...")
    prompt_texts = {
        domain: load_prompt_texts(domain, limit=TEXTS_PER_DOMAIN)
        for domain in DOMAIN_ORDER
    }
    for domain, texts in prompt_texts.items():
        print(f"  {domain:<18} {len(texts)} prompts")

    print("\nLoading MiniLM encoder from local cache...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True).to(device)
    model.eval()

    print("Encoding domain centroids...")
    domain_centroids = {}
    for domain in DOMAIN_ORDER:
        domain_centroids[domain] = encode_domain_centroid(
            prompt_texts[domain],
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
        print(f"  {domain:<18} done")

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    pair_records = []
    embedding_vals = []
    jaccard_vals = []
    format_vals = []
    cosine_vals = []
    mean_transfer_vals = []
    pair_labels = []

    for domain_a, domain_b in combinations(DOMAIN_ORDER, 2):
        idx_a = domain_to_idx[domain_a]
        idx_b = domain_to_idx[domain_b]

        embedding_sim = cosine_similarity(domain_centroids[domain_a], domain_centroids[domain_b])
        jaccard_sim = vocab_jaccard(prompt_texts[domain_a], prompt_texts[domain_b])
        format_sim = cosine_similarity(FORMAT_FEATURES[domain_a], FORMAT_FEATURES[domain_b])
        wfact_cos = float(cos_matrix[idx_a, idx_b])
        mean_transfer_auc = float((auc_matrix[idx_a, idx_b] + auc_matrix[idx_b, idx_a]) / 2.0)

        record = {
            "pair": pair_key(domain_a, domain_b),
            "domain_a": domain_a,
            "domain_b": domain_b,
            "embedding_similarity": embedding_sim,
            "token_jaccard": jaccard_sim,
            "format_similarity": format_sim,
            "wfact_cosine": wfact_cos,
            "mean_transfer_auc": mean_transfer_auc,
        }
        pair_records.append(record)
        embedding_vals.append(embedding_sim)
        jaccard_vals.append(jaccard_sim)
        format_vals.append(format_sim)
        cosine_vals.append(wfact_cos)
        mean_transfer_vals.append(mean_transfer_auc)
        pair_labels.append(record["pair"])

    metrics = {
        "embedding_similarity": np.array(embedding_vals),
        "token_jaccard": np.array(jaccard_vals),
        "format_similarity": np.array(format_vals),
    }

    metric_summary = {}
    print("\nSpearman correlations against pairwise wfact cosine:")
    for metric_name, metric_values in metrics.items():
        rho, p_val = spearmanr(metric_values, cosine_vals)
        auc_rho, auc_p = spearmanr(metric_values, mean_transfer_vals)
        metric_summary[metric_name] = {
            "spearman_rho_vs_wfact_cosine": float(rho),
            "spearman_p_vs_wfact_cosine": float(p_val),
            "spearman_rho_vs_mean_transfer_auc": float(auc_rho),
            "spearman_p_vs_mean_transfer_auc": float(auc_p),
        }
        print(
            f"  {metric_name:<20} rho={rho:>6.3f}, p={p_val:.4f} "
            f"| mean-transfer rho={auc_rho:>6.3f}, p={auc_p:.4f}"
        )

    print("\nBest-source recommendations by metric:")
    recommendations = {}
    for target in DOMAIN_ORDER:
        target_idx = domain_to_idx[target]
        actual_best_source = None
        actual_best_auc = -1.0
        metric_predictions = {}
        for metric_name in metrics:
            best_source = None
            best_score = -1e9
            for source in DOMAIN_ORDER:
                if source == target:
                    continue
                pair = pair_key(source, target)
                record = next(item for item in pair_records if item["pair"] == pair)
                score = record[metric_name]
                if score > best_score:
                    best_score = score
                    best_source = source
            metric_predictions[metric_name] = {
                "predicted_source": best_source,
                "predicted_similarity": best_score,
            }

        for source in DOMAIN_ORDER:
            if source == target:
                continue
            source_idx = domain_to_idx[source]
            transfer_auc = float(auc_matrix[source_idx, target_idx])
            if transfer_auc > actual_best_auc:
                actual_best_auc = transfer_auc
                actual_best_source = source

        recommendations[target] = {
            "actual_best_source_by_transfer_auc": actual_best_source,
            "actual_best_transfer_auc": actual_best_auc,
            **metric_predictions,
        }
        predicted = ", ".join(
            f"{name}={info['predicted_source']}" for name, info in metric_predictions.items()
        )
        print(
            f"  {target:<18} actual={actual_best_source:<18} "
            f"AUC={actual_best_auc:.3f} | {predicted}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    pretty_names = {
        "embedding_similarity": "Sentence-Embedding Similarity",
        "token_jaccard": "Token Jaccard Similarity",
        "format_similarity": "Format-Feature Similarity",
    }
    for ax, metric_name in zip(axes, metrics):
        rho = metric_summary[metric_name]["spearman_rho_vs_wfact_cosine"]
        p_val = metric_summary[metric_name]["spearman_p_vs_wfact_cosine"]
        save_scatter(
            ax,
            metrics[metric_name],
            cosine_vals,
            pair_labels,
            title=f"{pretty_names[metric_name]}\nSpearman $\\rho$={rho:.3f}, $p$={p_val:.4f}",
            xlabel=pretty_names[metric_name],
        )

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "domain_transferability_predictor.png")
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")

    output = {
        "domain_order": DOMAIN_ORDER,
        "texts_per_domain": TEXTS_PER_DOMAIN,
        "pair_records": pair_records,
        "metric_summary": metric_summary,
        "recommendations": recommendations,
    }
    out_path = os.path.join(SAVE_DIR, "results.json")
    with open(out_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
