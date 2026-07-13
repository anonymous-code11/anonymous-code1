import gc
import json
import os
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers.modeling_utils as modeling_utils
from datasets import Dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/models/Llama-2-13b-hf")
MODEL_NAME = os.path.basename(MODEL_PATH.rstrip("/"))
MODEL_TAG = os.environ.get("MODEL_TAG", MODEL_NAME.lower().replace(".", "p").replace("-", "_"))
DEVICE = "cuda:0"
BEST_LAYER = int(os.environ.get("BEST_LAYER", "16"))
N_TRAIN = 100
MAX_LEN = 192
PCA_DIM = 128
RANDOM_SEED = 42

SAVE_DIR = os.environ.get("SAVE_DIR", f"./results/{MODEL_TAG}_triplet")
WFACT_DIR = os.path.join(SAVE_DIR, "wfact")
FIG_DIR = "./figures"
for directory in [SAVE_DIR, WFACT_DIR, FIG_DIR]:
    os.makedirs(directory, exist_ok=True)

MMLU_ARROW = (
    "/home/pzh/.cache/huggingface/datasets/cais___mmlu/all/0.0.0/"
    "c30699e8356da336a370243923dbaf21066bb9fe/mmlu-test.arrow"
)

MMLU_DOMAINS = {
    "MMLU-Medical": {
        "anatomy",
        "clinical_knowledge",
        "medical_genetics",
        "college_medicine",
        "professional_medicine",
    },
    "MMLU-Law": {
        "jurisprudence",
        "international_law",
        "professional_law",
        "professional_accounting",
        "business_ethics",
    },
    "MMLU-CS": {
        "computer_security",
        "machine_learning",
        "college_computer_science",
        "high_school_computer_science",
    },
}
DOMAIN_ORDER = list(MMLU_DOMAINS.keys())
N_DOMAINS = len(DOMAIN_ORDER)


def patch_single_gpu_bnb_dispatch():
    original_dispatch = modeling_utils.dispatch_model
    if getattr(original_dispatch, "_codex_bnb_single_gpu_patch", False):
        return

    def move_buffers_to_device(module, device):
        for submodule in module.modules():
            for name, buf in list(submodule._buffers.items()):
                if buf is not None and buf.device.type != "cuda":
                    submodule._buffers[name] = buf.to(device)

    def patched_dispatch(model, **kwargs):
        device_map = kwargs.get("device_map")
        try:
            return original_dispatch(model, **kwargs)
        except ValueError as exc:
            msg = str(exc)
            single_nonoffload = (
                isinstance(device_map, dict)
                and len(set(device_map.values())) == 1
                and all(v not in ("cpu", "disk") for v in device_map.values())
            )
            if single_nonoffload and "bitsandbytes models" in msg and "not supported" in msg:
                only_device = list(device_map.values())[0]
                if isinstance(only_device, int):
                    only_device = f"cuda:{only_device}"
                move_buffers_to_device(model, only_device)
                model.hf_device_map = dict(device_map)
                return model
            raise

    patched_dispatch._codex_bnb_single_gpu_patch = True
    modeling_utils.dispatch_model = patched_dispatch


def normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-8)


def get_hidden(text, model, tokenizer, max_length=MAX_LEN):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = torch.stack([layer[0, -1, :] for layer in outputs.hidden_states])
    return hidden.float().cpu().numpy()


def build_wfact(h_correct, h_wrong, layer=BEST_LAYER):
    n = min(h_correct.shape[0], h_wrong.shape[0])
    x = np.concatenate([h_correct[:n, layer, :], h_wrong[:n, layer, :]], axis=0)
    y = np.array([1] * n + [0] * n)
    pca_dim_eff = min(PCA_DIM, x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=pca_dim_eff, random_state=RANDOM_SEED)
    x_proj = pca.fit_transform(x)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)
    clf.fit(x_proj, y)
    return normalize(pca.components_.T @ clf.coef_[0])


def compute_auc(h_correct, h_wrong, wfact, layer=BEST_LAYER):
    n = min(h_correct.shape[0], h_wrong.shape[0])
    y = np.array([1] * n + [0] * n)
    scores = np.concatenate(
        [
            h_correct[:n, layer, :] @ wfact,
            h_wrong[:n, layer, :] @ wfact,
        ]
    )
    return float(roc_auc_score(y, scores))


def effective_rank(matrix):
    _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    probs = singular_values**2
    probs = probs / probs.sum()
    return float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))


def load_mmlu_domain_pairs():
    ds = Dataset.from_file(MMLU_ARROW)
    random.seed(RANDOM_SEED)
    domain_pairs = {}
    for domain_name, subjects in MMLU_DOMAINS.items():
        items = [row for row in ds if row["subject"] in subjects]
        random.shuffle(items)
        pairs = []
        for row in items:
            answer = row["answer"]
            choices = row["choices"]
            if not (0 <= answer < len(choices)):
                continue
            correct = choices[answer]
            wrongs = [choice for idx, choice in enumerate(choices) if idx != answer]
            if not wrongs:
                continue
            pairs.append(
                (
                    f"Q: {row['question']}\nA: {correct}",
                    f"Q: {row['question']}\nA: {wrongs[0]}",
                )
            )
        domain_pairs[domain_name] = pairs
    return domain_pairs


def extract_and_cache(domain_name, pairs, model, tokenizer):
    cache_path = os.path.join(SAVE_DIR, f"{domain_name}_hidden.npz")
    if os.path.exists(cache_path):
        print(f"  [cache] {domain_name}")
        data = np.load(cache_path)
        return data["h_correct"][:N_TRAIN], data["h_wrong"][:N_TRAIN]

    print(f"  Extracting {domain_name} ({N_TRAIN} pairs)...")
    h_correct_list = []
    h_wrong_list = []
    for correct_text, wrong_text in tqdm(pairs[: N_TRAIN * 2], desc=domain_name):
        if len(h_correct_list) >= N_TRAIN:
            break
        try:
            h_correct_list.append(get_hidden(correct_text, model, tokenizer))
            h_wrong_list.append(get_hidden(wrong_text, model, tokenizer))
        except Exception as exc:
            print(f"    skip: {exc}")

    h_correct = np.stack(h_correct_list[:N_TRAIN])
    h_wrong = np.stack(h_wrong_list[:N_TRAIN])
    np.savez_compressed(cache_path, h_correct=h_correct, h_wrong=h_wrong)
    print(f"  Saved {cache_path}, shape={h_correct.shape}")
    return h_correct, h_wrong


def load_tokenizer_and_model():
    patch_single_gpu_bnb_dispatch()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        output_hidden_states=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Loading MMLU domain pairs...")
    domain_pairs = load_mmlu_domain_pairs()
    for domain_name in DOMAIN_ORDER:
        count = len(domain_pairs[domain_name])
        status = "OK" if count >= N_TRAIN else f"WARNING: only {count}"
        print(f"  {domain_name}: {count} pairs [{status}]")

    print(f"\nLoading {MODEL_NAME} in 8-bit...")
    tokenizer, model = load_tokenizer_and_model()
    print("Model loaded.")

    domain_hidden = {}
    for domain_name in DOMAIN_ORDER:
        print(f"\n{'=' * 55}\nDomain: {domain_name}")
        h_correct, h_wrong = extract_and_cache(domain_name, domain_pairs[domain_name], model, tokenizer)
        print(f"  Shape: hc={h_correct.shape}, hw={h_wrong.shape}")
        domain_hidden[domain_name] = (h_correct, h_wrong)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("\nModel unloaded.")

    print("\nBuilding wfact for each domain...")
    wfact_dict = {}
    for domain_name, (h_correct, h_wrong) in domain_hidden.items():
        wfact = build_wfact(h_correct, h_wrong, layer=BEST_LAYER)
        wfact_dict[domain_name] = wfact
        np.save(os.path.join(WFACT_DIR, f"{domain_name}.npy"), wfact)
        print(f"  {domain_name}: norm={np.linalg.norm(wfact):.4f}")

    print("\nComputing cosine similarity matrix...")
    cos_matrix = np.zeros((N_DOMAINS, N_DOMAINS))
    for i, d1 in enumerate(DOMAIN_ORDER):
        for j, d2 in enumerate(DOMAIN_ORDER):
            cos_matrix[i, j] = float(np.dot(wfact_dict[d1], wfact_dict[d2]))

    print("Computing AUC matrix...")
    auc_matrix = np.zeros((N_DOMAINS, N_DOMAINS))
    for i, src_domain in enumerate(DOMAIN_ORDER):
        for j, tgt_domain in enumerate(DOMAIN_ORDER):
            h_correct, h_wrong = domain_hidden[tgt_domain]
            auc_matrix[i, j] = compute_auc(h_correct, h_wrong, wfact_dict[src_domain], layer=BEST_LAYER)

    print("\nRunning SVD rank analysis...")
    wfact_stack = np.stack([wfact_dict[d] for d in DOMAIN_ORDER])
    _, singular_values, _ = np.linalg.svd(wfact_stack, full_matrices=False)
    total_var = np.sum(singular_values**2)
    cumulative_var = np.cumsum(singular_values**2) / total_var
    eff_rank = effective_rank(wfact_stack)

    random_eff_ranks = []
    hidden_size = wfact_stack.shape[1]
    for _ in range(100):
        random_matrix = np.random.randn(N_DOMAINS, hidden_size)
        random_matrix = random_matrix / np.linalg.norm(random_matrix, axis=1, keepdims=True)
        random_eff_ranks.append(effective_rank(random_matrix))
    random_eff_ranks = np.array(random_eff_ranks)

    short = {domain: domain.replace("MMLU-", "") for domain in DOMAIN_ORDER}
    labels = [short[domain] for domain in DOMAIN_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1 = axes[0]
    im1 = ax1.imshow(cos_matrix, vmin=-0.5, vmax=1.0, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    ax1.set_xticks(range(N_DOMAINS))
    ax1.set_yticks(range(N_DOMAINS))
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_yticklabels(labels)
    ax1.set_title(f"{MODEL_NAME} controlled cosine")
    for i in range(N_DOMAINS):
        for j in range(N_DOMAINS):
            value = cos_matrix[i, j]
            color = "white" if abs(value) > 0.55 else "black"
            ax1.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    ax2 = axes[1]
    im2 = ax2.imshow(auc_matrix, vmin=0.4, vmax=1.0, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.set_xticks(range(N_DOMAINS))
    ax2.set_yticks(range(N_DOMAINS))
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_yticklabels(labels)
    ax2.set_title(f"{MODEL_NAME} cross-domain transfer AUC")
    for i in range(N_DOMAINS):
        for j in range(N_DOMAINS):
            value = auc_matrix[i, j]
            color = "white" if value > 0.85 else "black"
            mark = " *" if i == j else ""
            ax2.text(j, i, f"{value:.3f}{mark}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"{MODEL_TAG}_triplet.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")

    off_diag_cos = [cos_matrix[i, j] for i in range(N_DOMAINS) for j in range(N_DOMAINS) if i != j]
    off_diag_auc = [auc_matrix[i, j] for i in range(N_DOMAINS) for j in range(N_DOMAINS) if i != j]
    results = {
        "experiment": f"{MODEL_NAME} controlled: fixed MCQ format, varying knowledge domain",
        "model_path": MODEL_PATH,
        "quantization": "8bit",
        "model_tag": MODEL_TAG,
        "domain_names": DOMAIN_ORDER,
        "n_domains": N_DOMAINS,
        "n_train": N_TRAIN,
        "layer": BEST_LAYER,
        "pca_dim": PCA_DIM,
        "cos_matrix": cos_matrix.tolist(),
        "auc_matrix": auc_matrix.tolist(),
        "svd": {
            "singular_values": singular_values.tolist(),
            "cumulative_variance": cumulative_var.tolist(),
            "effective_rank": eff_rank,
            "random_baseline_mean": float(random_eff_ranks.mean()),
            "random_baseline_std": float(random_eff_ranks.std()),
        },
        "summary": {
            "mean_off_diag_cos": float(np.mean(np.abs(off_diag_cos))),
            "max_off_diag_cos": float(np.max(np.abs(off_diag_cos))),
            "min_off_diag_cos": float(np.min(np.abs(off_diag_cos))),
            "mean_within_auc": float(np.mean([auc_matrix[i, i] for i in range(N_DOMAINS)])),
            "mean_cross_auc": float(np.mean(off_diag_auc)),
        },
        "pairwise_detail": [
            {
                "domain_1": DOMAIN_ORDER[i],
                "domain_2": DOMAIN_ORDER[j],
                "cosine": float(cos_matrix[i, j]),
                "auc_1to2": float(auc_matrix[i, j]),
                "auc_2to1": float(auc_matrix[j, i]),
            }
            for i in range(N_DOMAINS)
            for j in range(N_DOMAINS)
            if i < j
        ],
    }

    results_path = os.path.join(SAVE_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")

    print("\n" + "=" * 60)
    print(f"{MODEL_NAME} CONTROLLED EXPERIMENT SUMMARY")
    print("=" * 60)
    print(
        f"Off-diagonal |cos|: mean={results['summary']['mean_off_diag_cos']:.4f}, "
        f"max={results['summary']['max_off_diag_cos']:.4f}, "
        f"min={results['summary']['min_off_diag_cos']:.4f}"
    )
    print(f"Mean within-domain AUC: {results['summary']['mean_within_auc']:.4f}")
    print(f"Mean cross-domain AUC:  {results['summary']['mean_cross_auc']:.4f}")
    print(f"SVD effective rank:     {eff_rank:.2f} / {N_DOMAINS}")
    print(
        f"Random baseline:        {results['svd']['random_baseline_mean']:.2f} +/- "
        f"{results['svd']['random_baseline_std']:.2f}"
    )


if __name__ == "__main__":
    main()
