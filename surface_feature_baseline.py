"""Surface-feature baseline: compare w_fact AUC against a classifier trained on surface features only."""

import os
import re
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset, Dataset

from analysis_utils import (
    load_hidden,
    build_wfact,
    BEST_LAYER,
    PCA_DIM,
    FOUR_DOMAIN_ORDER,
    MMLU_MEDICAL_SUBJECTS,
    ARROW_PATHS,
)

# ── 配置 ──────────────────────────────────────────────────────────────────────
N_SAMPLES = 400
LAYER = BEST_LAYER
SAVE_DIR = "./results/surface_feature_baseline"
os.makedirs(SAVE_DIR, exist_ok=True)

DOMAINS = FOUR_DOMAIN_ORDER


# ── Surface feature extraction ───────────────────────────────────────────────

KEYWORDS = [
    "therefore", "thus", "because", "if", "then",
    "hence", "so", "since", "implies", "conclude",
]

def extract_surface_features(text):
    """提取 6 维 surface features"""
    words = text.split()
    word_tokens = re.findall(r"[A-Za-z']+", text)
    lowered = text.lower()

    length = len(words)
    keyword_count = sum(lowered.count(kw) for kw in KEYWORDS)
    punctuation_count = sum(ch in set(",.;:!?()[]{}-") for ch in text)
    number_count = len(re.findall(r"\d+", text))
    lexical_richness = len(set(w.lower() for w in word_tokens)) / max(len(word_tokens), 1)
    avg_word_length = sum(len(w) for w in word_tokens) / max(len(word_tokens), 1)

    return np.array([
        length, keyword_count, punctuation_count,
        number_count, lexical_richness, avg_word_length,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "length", "keyword_count", "punctuation_count",
    "number_count", "lexical_richness", "avg_word_length",
]


# ── 加载各域的 (correct_text, wrong_text) pairs ──────────────────────────────

def load_domain_text_pairs(domain_name, limit=400):
    """加载每个域的文本 pairs"""

    if domain_name == "TruthfulQA":
        ds = load_dataset("truthful_qa", "generation", split="validation")
        pairs = []
        for s in ds:
            if s["best_answer"] and s["incorrect_answers"]:
                pairs.append((
                    f"Q: {s['question']}\nA: {s['best_answer']}",
                    f"Q: {s['question']}\nA: {s['incorrect_answers'][0]}",
                ))
                if len(pairs) >= limit:
                    break
        return pairs

    elif domain_name == "FEVER":
        ds = Dataset.from_file(ARROW_PATHS["FEVER"])
        supports, refutes = [], []
        for row in ds:
            label = row.get("label", -1)
            hyp = (row.get("hypothesis") or "").strip()
            evi = (row.get("premise") or "").strip()
            if not hyp or not evi:
                continue
            text = f"Evidence: {evi}\nClaim: {hyp}\nLabel:"
            if label == 0:
                supports.append(text)
            elif label == 1:
                refutes.append(text)
        n = min(len(supports), len(refutes), limit)
        return list(zip(
            [f"{s} SUPPORTS" for s in supports[:n]],
            [f"{r} REFUTES" for r in refutes[:n]],
        ))

    elif domain_name == "MMLU-Medical":
        ds = Dataset.from_file(ARROW_PATHS["MMLU-Medical"])
        pairs = []
        for row in ds:
            if row["subject"] not in MMLU_MEDICAL_SUBJECTS:
                continue
            q = row["question"]
            choices = row["choices"]
            answer = int(row["answer"])
            correct_choice = choices[answer]
            wrong_choices = [c for i, c in enumerate(choices) if i != answer]
            if not wrong_choices:
                continue
            prompt = f"Q: {q}\nA:"
            pairs.append((
                f"{prompt} {correct_choice}",
                f"{prompt} {wrong_choices[0]}",
            ))
            if len(pairs) >= limit:
                break
        return pairs

    elif domain_name == "ARC-Science":
        ds = Dataset.from_file(ARROW_PATHS["ARC-Science"])
        pairs = []
        for row in ds:
            q = row["question"]
            choices_text = row["choices"]["text"]
            choices_label = row["choices"]["label"]
            answer_key = row["answerKey"]
            correct_idx = None
            for i, lab in enumerate(choices_label):
                if lab == answer_key:
                    correct_idx = i
                    break
            if correct_idx is None:
                continue
            correct = choices_text[correct_idx]
            wrongs = [c for i, c in enumerate(choices_text) if i != correct_idx]
            if not wrongs:
                continue
            prompt = f"Q: {q}\nA:"
            pairs.append((
                f"{prompt} {correct}",
                f"{prompt} {wrongs[0]}",
            ))
            if len(pairs) >= limit:
                break
        return pairs

    raise KeyError(f"Unknown domain: {domain_name}")


def main():
    print("=" * 60)
    print("Surface-Feature Baseline")
    print("=" * 60)

    results = {"domains": {}}

    for dname in DOMAINS:
        print(f"\n{'='*50}")
        print(f"Domain: {dname}")
        print(f"{'='*50}")

        # ── Load text pairs and extract surface features ──────────────────
        print("  Loading text pairs...")
        pairs = load_domain_text_pairs(dname, limit=N_SAMPLES)
        n = len(pairs)
        print(f"  Got {n} pairs")

        correct_feats = np.array([extract_surface_features(c) for c, w in pairs])
        wrong_feats = np.array([extract_surface_features(w) for c, w in pairs])

        X_surf = np.concatenate([correct_feats, wrong_feats], axis=0)
        y_surf = np.array([1] * n + [0] * n)

        # ── Train surface-only classifier ─────────────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_surf)
        clf_surf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        clf_surf.fit(X_scaled, y_surf)
        surf_probs = clf_surf.predict_proba(X_scaled)[:, 1]
        surface_auc = float(roc_auc_score(y_surf, surf_probs))

        # ── Feature importance ────────────────────────────────────────────
        coef = clf_surf.coef_[0]
        feat_importance = {
            FEATURE_NAMES[i]: round(float(coef[i]), 4)
            for i in range(len(FEATURE_NAMES))
        }

        # ── w_fact AUC (from hidden states) ───────────────────────────────
        print("  Loading hidden states for w_fact comparison...")
        hc, hw = load_hidden(dname, limit=N_SAMPLES)
        n_h = min(hc.shape[0], hw.shape[0], n)
        w = build_wfact(hc[:n_h], hw[:n_h], layer=LAYER, pca_dim=PCA_DIM)
        y_h = np.array([1] * n_h + [0] * n_h)
        scores_h = np.concatenate([
            hc[:n_h, LAYER, :] @ w,
            hw[:n_h, LAYER, :] @ w,
        ])
        wfact_auc = float(roc_auc_score(y_h, scores_h))

        # ── Report ────────────────────────────────────────────────────────
        gap = wfact_auc - surface_auc
        print(f"  Surface-only AUC:  {surface_auc:.4f}")
        print(f"  w_fact AUC:        {wfact_auc:.4f}")
        print(f"  Gap (w_fact - surface):  {gap:+.4f}")
        print(f"  Feature importance: {feat_importance}")

        results["domains"][dname] = {
            "n_pairs": n,
            "surface_only_auc": round(surface_auc, 4),
            "wfact_auc": round(wfact_auc, 4),
            "gap": round(gap, 4),
            "feature_importance": feat_importance,
        }

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Domain':<18} {'Surface AUC':>12} {'w_fact AUC':>12} {'Gap':>8}")
    print("-" * 52)

    gaps = []
    for dname in DOMAINS:
        d = results["domains"][dname]
        print(f"{dname:<18} {d['surface_only_auc']:>12.4f} {d['wfact_auc']:>12.4f} {d['gap']:>+8.4f}")
        gaps.append(d["gap"])

    mean_surf = np.mean([results["domains"][d]["surface_only_auc"] for d in DOMAINS])
    mean_wfact = np.mean([results["domains"][d]["wfact_auc"] for d in DOMAINS])
    mean_gap = np.mean(gaps)

    print("-" * 52)
    print(f"{'Mean':<18} {mean_surf:>12.4f} {mean_wfact:>12.4f} {mean_gap:>+8.4f}")

    results["summary"] = {
        "mean_surface_auc": round(mean_surf, 4),
        "mean_wfact_auc": round(mean_wfact, 4),
        "mean_gap": round(mean_gap, 4),
    }

    if mean_gap > 0.05:
        print(f"\n  w_fact captures signal BEYOND surface features (mean gap = {mean_gap:+.4f})")
        print("  This argues against the pure spurious-correlation interpretation.")
    else:
        print(f"\n  Gap is small ({mean_gap:+.4f}); surface features may explain most of the signal.")

    save_path = os.path.join(SAVE_DIR, "results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()