
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DEVICE      = "cuda:0"
MAX_SAMPLES = 800
SAVE_DIR    = "./results/cross_model"
FIG_DIR     = "./figures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── 工具函数 ──────────────────────────────────────────────

def get_hidden_states(text, model, tokenizer, max_length=128):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=max_length, padding=False,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = torch.stack([h[0, -1, :] for h in outputs.hidden_states])
    return hidden.float().cpu().numpy()


def extract_hidden_states(samples, model, tokenizer):
    h_correct_list, h_wrong_list = [], []
    for question, correct, wrongs in tqdm(samples):
        wrong = wrongs[0]
        try:
            hc = get_hidden_states(f"Q: {question}\nA: {correct}", model, tokenizer)
            hw = get_hidden_states(f"Q: {question}\nA: {wrong}",   model, tokenizer)
            h_correct_list.append(hc)
            h_wrong_list.append(hw)
        except Exception:
            continue
    return np.stack(h_correct_list), np.stack(h_wrong_list)


def run_probing(h_correct, h_wrong):
    N     = h_correct.shape[0]
    L     = h_correct.shape[1]
    X_all = np.concatenate([h_correct, h_wrong], axis=0)
    y_all = np.array([1]*N + [0]*N)
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs  = []
    for layer_idx in range(L):
        X_layer = X_all[:, layer_idx, :]
        pca = PCA(n_components=min(128, X_layer.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X_layer)
        layer_aucs = []
        for tr, val in cv.split(X_pca, y_all):
            sc  = StandardScaler()
            Xtr = sc.fit_transform(X_pca[tr])
            Xval= sc.transform(X_pca[val])
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(Xtr, y_all[tr])
            prob = clf.predict_proba(Xval)[:, 1]
            layer_aucs.append(roc_auc_score(y_all[val], prob))
        aucs.append(float(np.mean(layer_aucs)))
    return aucs


def compute_dsd_aucs(h_correct, h_wrong, best_layer, fluency_layer):
    N     = h_correct.shape[0]
    L     = h_correct.shape[1]
    y_all = np.array([1]*N + [0]*N)

    X_layer = np.concatenate([h_correct[:, best_layer, :],
                               h_wrong[:,   best_layer, :]], axis=0)
    pca = PCA(n_components=128, random_state=42)
    X_pca = pca.fit_transform(X_layer)
    sc  = StandardScaler()
    X_sc= sc.fit_transform(X_pca)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(X_sc, y_all)
    w_fact = pca.components_.T @ clf.coef_[0]
    w_fact = w_fact / (np.linalg.norm(w_fact) + 1e-8)

    mean_c = h_correct[:, fluency_layer, :].mean(0)
    mean_w = h_wrong[:,   fluency_layer, :].mean(0)
    w_flu  = mean_c - mean_w
    w_flu  = w_flu / (np.linalg.norm(w_flu) + 1e-8)

    dsd_aucs = []
    for li in range(L):
        dc = h_correct[:, li, :] @ w_fact - h_correct[:, li, :] @ w_flu
        dw = h_wrong[:,   li, :] @ w_fact - h_wrong[:,   li, :] @ w_flu
        auc = roc_auc_score(y_all, np.concatenate([dc, dw]))
        dsd_aucs.append(float(auc))
    return dsd_aucs

# ── 读取已有结果 ──────────────────────────────────────────
print("Loading Llama-3-8B cached results...")
with open("./results/probing_results.json") as f:
    llama_probe = json.load(f)
with open("./results/dsd_results.json") as f:
    llama_dsd = json.load(f)

all_results = {
    "Llama-3-8B": {
        "probing_aucs": llama_probe["layer_aucs"],
        "dsd_aucs"    : llama_dsd["dsd_aucs_per_layer"],
        "best_probing": max(llama_probe["layer_aucs"]),
        "best_dsd"    : max(llama_dsd["dsd_aucs_per_layer"]),
        "num_layers"  : len(llama_probe["layer_aucs"]),
    }
}

# Qwen结果（从上次输出直接写入）
print("Loading Qwen2.5-7B-Instruct cached npz...")
qwen_npz = np.load(os.path.join(SAVE_DIR, "Qwen2.5-7B-Instruct.npz"))
h_c_q = qwen_npz["h_correct"]
h_w_q = qwen_npz["h_wrong"]
qwen_probe_aucs = run_probing(h_c_q, h_w_q)
qwen_best_layer = int(np.argmax(qwen_probe_aucs))
qwen_flu_layer  = max(1, int(h_c_q.shape[1] * 0.1))
qwen_dsd_aucs   = compute_dsd_aucs(h_c_q, h_w_q, qwen_best_layer, qwen_flu_layer)

all_results["Qwen2.5-7B"] = {
    "probing_aucs": qwen_probe_aucs,
    "dsd_aucs"    : qwen_dsd_aucs,
    "best_probing": max(qwen_probe_aucs),
    "best_dsd"    : max(qwen_dsd_aucs),
    "num_layers"  : len(qwen_probe_aucs),
}
print(f"  Qwen best probing: {max(qwen_probe_aucs):.4f}")
print(f"  Qwen best DSD    : {max(qwen_dsd_aucs):.4f}")

# ── 跑 Mistral ────────────────────────────────────────────
print("\nLoading TruthfulQA...")
dataset = load_dataset("truthful_qa", "generation", split="validation")
dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
samples = [(s["question"], s["best_answer"], s["incorrect_answers"])
           for s in dataset if s["best_answer"] and s["incorrect_answers"]]

print("\n[Mistral-7B-v0.2] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(
    "/opt/models/mistral-7B-v0.2",
    use_fast=False,          # 避免protobuf问题
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "/opt/models/mistral-7B-v0.2",
    torch_dtype=torch.float16,
    device_map=DEVICE,
    output_hidden_states=True,
    trust_remote_code=True,
)
model.eval()
num_layers  = model.config.num_hidden_layers
print(f"Loaded. Layers={num_layers}, HiddenSize={model.config.hidden_size}")

print("Extracting hidden states...")
h_c_m, h_w_m = extract_hidden_states(samples, model, tokenizer)
np.savez_compressed(os.path.join(SAVE_DIR, "Mistral-7B-v0.2.npz"),
                    h_correct=h_c_m, h_wrong=h_w_m)

del model
torch.cuda.empty_cache()

print("Running probing...")
mistral_probe_aucs = run_probing(h_c_m, h_w_m)
mistral_best_layer = int(np.argmax(mistral_probe_aucs))
mistral_flu_layer  = max(1, int(h_c_m.shape[1] * 0.1))
mistral_dsd_aucs   = compute_dsd_aucs(h_c_m, h_w_m, mistral_best_layer, mistral_flu_layer)

all_results["Mistral-7B"] = {
    "probing_aucs": mistral_probe_aucs,
    "dsd_aucs"    : mistral_dsd_aucs,
    "best_probing": max(mistral_probe_aucs),
    "best_dsd"    : max(mistral_dsd_aucs),
    "num_layers"  : len(mistral_probe_aucs),
}
print(f"  Mistral best probing: {max(mistral_probe_aucs):.4f}")
print(f"  Mistral best DSD    : {max(mistral_dsd_aucs):.4f}")

# ── 绘图 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax_idx, (model_name, res) in enumerate(all_results.items()):
    ax = axes[ax_idx]
    L  = res["num_layers"]
    # 归一化x轴到相对层位置
    x = [i / (L-1) for i in range(L)]

    ax.plot(x, res["probing_aucs"][:L], "r--s", markersize=3,
            linewidth=1.5, label=f"Probing (max={res['best_probing']:.3f})")
    ax.plot(x, res["dsd_aucs"][:L],     "b-o",  markersize=3,
            linewidth=1.5, label=f"DSD     (max={res['best_dsd']:.3f})")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Relative Layer Depth", fontsize=11)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title(model_name, fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.02)
    ax.grid(True, alpha=0.3)

plt.suptitle("Cross-Model Validation: Probing AUC vs DSD AUC (training-free)\n"
             "TruthfulQA", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cross_model.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {FIG_DIR}/cross_model.png")

# ── 汇总表 ────────────────────────────────────────────────
print("\n" + "="*58)
print(f"{'Model':<20} {'Probing AUC':>12} {'DSD AUC':>10} {'Retain%':>8}")
print("-"*58)
for name, res in all_results.items():
    retain = res["best_dsd"] / res["best_probing"] * 100
    print(f"{name:<20} {res['best_probing']:>12.4f} {res['best_dsd']:>10.4f} {retain:>7.1f}%")
print("="*58)

with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items()
                   if not isinstance(vv, list)}
               for k, v in all_results.items()}, f, indent=2)