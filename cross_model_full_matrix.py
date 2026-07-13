"""Full cross-domain matrix for a given model: cosine similarity, transfer AUC, mixed probe, and effective rank."""

import argparse
import gc
import json
import os
import re

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis_utils import (
    ARROW_PATHS,
    MMLU_MEDICAL_SUBJECTS,
    PCA_DIM,
    RANDOM_SEED,
)

# ── 配置 ──────────────────────────────────────────────────────────────────────
N_TRAIN = 400
MAX_LEN = 192
DEVICE = "cuda:0"

CLEAN_DOMAINS = ["TruthfulQA", "FEVER", "MMLU-Medical", "ARC-Science"]

KEYWORDS = [
    "therefore", "thus", "because", "if", "then",
    "hence", "so", "since", "implies", "conclude",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--best_layer", type=int, default=None,
                        help="Override best layer; default = auto-select ~0.5L")
    return parser.parse_args()


# ── Hidden state extraction ──────────────────────────────────────────────────

def get_hidden(text, model, tokenizer, max_length=MAX_LEN, device=DEVICE):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length, padding=False).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = torch.stack([layer[0, -1, :] for layer in out.hidden_states])
    return h.float().cpu().numpy()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_domain_pairs(domain_name, limit=400):
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
            text_base = f"Evidence: {evi}\nClaim: {hyp}\nLabel:"
            if label == 0:
                supports.append(f"{text_base} SUPPORTS")
            elif label == 1:
                refutes.append(f"{text_base} REFUTES")
        n = min(len(supports), len(refutes), limit)
        return list(zip(supports[:n], refutes[:n]))

    elif domain_name == "MMLU-Medical":
        ds = Dataset.from_file(ARROW_PATHS["MMLU-Medical"])
        pairs = []
        for row in ds:
            if row["subject"] not in MMLU_MEDICAL_SUBJECTS:
                continue
            q = row["question"]
            choices = row["choices"]
            answer = int(row["answer"])
            wrong_choices = [c for i, c in enumerate(choices) if i != answer]
            if not wrong_choices:
                continue
            prompt = f"Q: {q}\nA:"
            pairs.append((f"{prompt} {choices[answer]}", f"{prompt} {wrong_choices[0]}"))
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
            correct_idx = next((i for i, l in enumerate(choices_label) if l == answer_key), None)
            if correct_idx is None:
                continue
            wrongs = [c for i, c in enumerate(choices_text) if i != correct_idx]
            if not wrongs:
                continue
            prompt = f"Q: {q}\nA:"
            pairs.append((f"{prompt} {choices_text[correct_idx]}", f"{prompt} {wrongs[0]}"))
            if len(pairs) >= limit:
                break
        return pairs

    raise KeyError(f"Unknown domain: {domain_name}")


# ── Core analysis functions ──────────────────────────────────────────────────

def build_wfact(h_correct, h_wrong, layer, pca_dim=PCA_DIM):
    n = min(h_correct.shape[0], h_wrong.shape[0])
    X = np.concatenate([h_correct[:n, layer, :], h_wrong[:n, layer, :]], axis=0)
    y = np.array([1] * n + [0] * n)
    dim = min(pca_dim, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=dim, random_state=RANDOM_SEED)
    Xp = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)
    clf.fit(Xp, y)
    w = pca.components_.T @ clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-8)


def compute_auc(h_correct, h_wrong, wfact, layer):
    n = min(h_correct.shape[0], h_wrong.shape[0])
    y = np.array([1] * n + [0] * n)
    scores = np.concatenate([
        h_correct[:n, layer, :] @ wfact,
        h_wrong[:n, layer, :] @ wfact,
    ])
    return float(roc_auc_score(y, scores))


def find_best_layer(domain_hidden, num_layers):
    """Auto-select best layer by probing AUC on TruthfulQA"""
    hc, hw = domain_hidden["TruthfulQA"]
    best_auc = 0
    best_l = num_layers // 2
    for l in range(1, num_layers + 1):
        try:
            w = build_wfact(hc, hw, layer=l)
            auc = compute_auc(hc, hw, w, layer=l)
            if auc > best_auc:
                best_auc = auc
                best_l = l
        except Exception:
            continue
    return best_l


# ── Surface features ─────────────────────────────────────────────────────────

def extract_surface_features(text):
    words = text.split()
    word_tokens = re.findall(r"[A-Za-z']+", text)
    lowered = text.lower()
    return np.array([
        len(words),
        sum(lowered.count(kw) for kw in KEYWORDS),
        sum(ch in set(",.;:!?()[]{}-") for ch in text),
        len(re.findall(r"\d+", text)),
        len(set(w.lower() for w in word_tokens)) / max(len(word_tokens), 1),
        sum(len(w) for w in word_tokens) / max(len(word_tokens), 1),
    ], dtype=np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    model_name = args.model_name

    save_dir = f"./results/cross_model_full/{model_name}"
    wfact_dir = os.path.join(save_dir, "wfact")
    os.makedirs(wfact_dir, exist_ok=True)

    print("=" * 60)
    print(f"Experiment D: Full Cross-Domain Matrix for {model_name}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map=DEVICE, output_hidden_states=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded. Layers: {num_layers}")

    # ── Extract hidden states for all 4 domains ──────────────────────────
    domain_hidden = {}
    domain_pairs_text = {}

    for dname in CLEAN_DOMAINS:
        cache_path = os.path.join(save_dir, f"{dname}_hidden.npz")

        if os.path.exists(cache_path):
            print(f"\n[cache] Loading {dname} from {cache_path}")
            d = np.load(cache_path)
            domain_hidden[dname] = (d["h_correct"][:N_TRAIN], d["h_wrong"][:N_TRAIN])
        else:
            print(f"\nExtracting {dname}...")
            pairs = load_domain_pairs(dname, limit=N_TRAIN)
            domain_pairs_text[dname] = pairs
            print(f"  {len(pairs)} pairs")

            h_c_list, h_w_list = [], []
            for correct_text, wrong_text in tqdm(pairs[:N_TRAIN], desc=dname):
                try:
                    hc = get_hidden(correct_text, model, tokenizer)
                    hw = get_hidden(wrong_text, model, tokenizer)
                    h_c_list.append(hc)
                    h_w_list.append(hw)
                except Exception as e:
                    print(f"  skip: {e}")
                    continue

            hc_arr = np.stack(h_c_list[:N_TRAIN])
            hw_arr = np.stack(h_w_list[:N_TRAIN])
            np.savez_compressed(cache_path, h_correct=hc_arr, h_wrong=hw_arr)
            domain_hidden[dname] = (hc_arr, hw_arr)
            print(f"  Saved {cache_path}, shape={hc_arr.shape}")

    # ── Unload model ──────────────────────────────────────────────────────
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("\nModel unloaded.")

    # ── Select best layer ─────────────────────────────────────────────────
    if args.best_layer is not None:
        best_layer = args.best_layer
    else:
        best_layer = find_best_layer(domain_hidden, num_layers)
    print(f"\nBest layer: {best_layer} (of {num_layers})")

    # ── Build w_fact for each domain ──────────────────────────────────────
    print("\nBuilding w_fact...")
    wfact_dict = {}
    for dname in CLEAN_DOMAINS:
        hc, hw = domain_hidden[dname]
        w = build_wfact(hc, hw, layer=best_layer)
        wfact_dict[dname] = w
        np.save(os.path.join(wfact_dir, f"{dname}.npy"), w)
        within_auc = compute_auc(hc, hw, w, layer=best_layer)
        print(f"  {dname}: within-AUC={within_auc:.4f}")

    # ── Cosine matrix ─────────────────────────────────────────────────────
    print("\n--- Cosine Similarity Matrix ---")
    n_d = len(CLEAN_DOMAINS)
    cos_matrix = np.zeros((n_d, n_d))
    for i, d1 in enumerate(CLEAN_DOMAINS):
        for j, d2 in enumerate(CLEAN_DOMAINS):
            cos_matrix[i, j] = float(np.dot(wfact_dict[d1], wfact_dict[d2]))

    print(f"{'':>18}", end="")
    for d in CLEAN_DOMAINS:
        print(f"  {d[:12]:>12}", end="")
    print()
    for i, d1 in enumerate(CLEAN_DOMAINS):
        print(f"{d1:>18}", end="")
        for j in range(n_d):
            print(f"  {cos_matrix[i,j]:>12.4f}", end="")
        print()

    # ── AUC transfer matrix ───────────────────────────────────────────────
    print("\n--- Cross-Domain Transfer AUC ---")
    auc_matrix = np.zeros((n_d, n_d))
    for i, d_src in enumerate(CLEAN_DOMAINS):
        for j, d_tgt in enumerate(CLEAN_DOMAINS):
            hc, hw = domain_hidden[d_tgt]
            auc_matrix[i, j] = compute_auc(hc, hw, wfact_dict[d_src], layer=best_layer)

    header = "src \\ tgt"
    print(f"{header:>18}", end="")
    for d in CLEAN_DOMAINS:
        print(f"  {d[:12]:>12}", end="")
    print()
    for i, d1 in enumerate(CLEAN_DOMAINS):
        print(f"{d1:>18}", end="")
        for j in range(n_d):
            print(f"  {auc_matrix[i,j]:>12.4f}", end="")
        print()

    # ── Mixed probe ───────────────────────────────────────────────────────
    print("\n--- Mixed-Domain Probe ---")
    rng = np.random.RandomState(RANDOM_SEED)
    mixed_X, mixed_y = [], []
    n_mix = 100
    for dname in CLEAN_DOMAINS:
        hc, hw = domain_hidden[dname]
        n_avail = min(hc.shape[0], hw.shape[0])
        idx = rng.choice(n_avail, size=min(n_mix, n_avail), replace=False)
        mixed_X.append(hc[idx, best_layer, :])
        mixed_X.append(hw[idx, best_layer, :])
        mixed_y.extend([1] * len(idx))
        mixed_y.extend([0] * len(idx))

    mixed_X = np.concatenate(mixed_X, axis=0)
    mixed_y = np.array(mixed_y)

    dim = min(PCA_DIM, mixed_X.shape[0] - 1, mixed_X.shape[1])
    pca = PCA(n_components=dim, random_state=RANDOM_SEED)
    Xp = pca.fit_transform(mixed_X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)
    clf.fit(Xp, mixed_y)
    w_mixed = pca.components_.T @ clf.coef_[0]
    w_mixed = w_mixed / (np.linalg.norm(w_mixed) + 1e-8)

    mixed_aucs = {}
    for dname in CLEAN_DOMAINS:
        hc, hw = domain_hidden[dname]
        mixed_aucs[dname] = compute_auc(hc, hw, w_mixed, layer=best_layer)

    print(f"{'Domain':<18} {'Per-domain':>12} {'Mixed':>12} {'Gap':>8}")
    for dname in CLEAN_DOMAINS:
        pd_auc = auc_matrix[CLEAN_DOMAINS.index(dname), CLEAN_DOMAINS.index(dname)]
        mx_auc = mixed_aucs[dname]
        print(f"{dname:<18} {pd_auc:>12.4f} {mx_auc:>12.4f} {mx_auc - pd_auc:>+8.4f}")

    # ── Effective rank ────────────────────────────────────────────────────
    W = np.stack([wfact_dict[d] for d in CLEAN_DOMAINS])
    _, s, _ = np.linalg.svd(W, full_matrices=False)
    p = s ** 2 / np.sum(s ** 2)
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    print(f"\nEffective rank: {eff_rank:.2f} / {n_d}")
    print(f"Singular values: {s.round(4).tolist()}")

    # ── Summary statistics ────────────────────────────────────────────────
    off_diag_cos = []
    off_diag_auc = []
    for i in range(n_d):
        for j in range(n_d):
            if i != j:
                off_diag_cos.append(abs(cos_matrix[i, j]))
                off_diag_auc.append(auc_matrix[i, j])

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "model": model_name,
        "best_layer": best_layer,
        "num_layers": num_layers,
        "n_train": N_TRAIN,
        "domains": CLEAN_DOMAINS,
        "cos_matrix": cos_matrix.tolist(),
        "auc_matrix": auc_matrix.tolist(),
        "mixed_probe_aucs": mixed_aucs,
        "effective_rank": round(eff_rank, 4),
        "singular_values": s.tolist(),
        "summary": {
            "mean_within_auc": round(float(np.mean(np.diag(auc_matrix))), 4),
            "mean_off_diag_cos": round(float(np.mean(off_diag_cos)), 4),
            "mean_off_diag_auc": round(float(np.mean(off_diag_auc)), 4),
            "mean_mixed_auc": round(float(np.mean(list(mixed_aucs.values()))), 4),
        },
    }

    save_path = os.path.join(save_dir, "results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY for {model_name}")
    print(f"{'='*60}")
    print(f"  Best layer:          {best_layer}")
    print(f"  Mean within-AUC:     {results['summary']['mean_within_auc']}")
    print(f"  Mean off-diag |cos|: {results['summary']['mean_off_diag_cos']}")
    print(f"  Mean off-diag AUC:   {results['summary']['mean_off_diag_auc']}")
    print(f"  Mixed probe AUC:     {results['summary']['mean_mixed_auc']}")
    print(f"  Effective rank:      {eff_rank:.2f} / {n_d}")
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()