"""
Negation sensitivity: 3 models x 3 datasets.
Models: Llama-3-8B, Qwen2.5-7B-Instruct, Mistral-7B-v0.2
Datasets: CounterFact, FM Queries, PopQA
"""

import gc, json, os, random
from collections import defaultdict
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    ("/opt/models/Llama-3-8B", "Llama-3-8B"),
    ("/opt/models/Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
    ("/opt/models/Mistral-7B-v0.2", "Mistral-7B"),
]
DEVICE = "cuda:0"
BEST_LAYER = 16
PCA_DIM = 128
N_TRAIN = 1000
N_TEST = 2000
N_BOOTSTRAP = 1000
SAVE_DIR = "./results/negation_sensitivity"
os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(42)
np.random.seed(42)


# ── Prepare all three datasets once ──────────────────────────────────────────

def prepare_counterfact():
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
    quads = []
    for row in ds:
        p = row["prompt"].strip()
        tt = row["target_true"].strip()
        tf = row["target_false"].strip()
        if not p or not tt or not tf or tt == tf:
            continue
        if not any(p.lower().endswith(v) for v in (" is", " was", " are", " were")):
            continue
        quads.append({
            "true": f"{p} {tt}",
            "false": f"{p} {tf}",
            "neg_true": f"{p} not {tt}",
            "neg_false": f"{p} not {tf}",
        })
    random.shuffle(quads)
    return quads

def prepare_fm():
    ds = load_dataset("coastalcph/fm_queries", split="train")
    relation_answers = defaultdict(set)
    candidates = []
    for row in ds:
        query = row["query"]
        answers = row["answer"]
        relation = row["relation"]
        if not answers or not query.rstrip().endswith("_X_."):
            continue
        prefix = query.replace("_X_.", "").rstrip()
        if not prefix.endswith(" is"):
            continue
        ans_name = answers[0]["name"]
        relation_answers[relation].add(ans_name)
        candidates.append({"prefix": prefix, "answer": ans_name, "relation": relation})
    quads = []
    seen = set()
    for c in candidates:
        if c["prefix"] in seen:
            continue
        seen.add(c["prefix"])
        others = list(relation_answers[c["relation"]] - {c["answer"]})
        if not others:
            continue
        wrong = random.choice(others)
        if len(c["answer"].split()) > 5 or len(wrong.split()) > 5:
            continue
        quads.append({
            "true": f"{c['prefix']} {c['answer']}.",
            "false": f"{c['prefix']} {wrong}.",
            "neg_true": f"{c['prefix']} not {c['answer']}.",
            "neg_false": f"{c['prefix']} not {wrong}.",
        })
    random.shuffle(quads)
    return quads

def prepare_popqa():
    ds = load_dataset("akariasai/PopQA", split="test")
    prop_objects = defaultdict(set)
    rows = []
    for row in ds:
        subj, prop, obj = row["subj"], row["prop"], row["obj"]
        if not subj or not prop or not obj or len(obj.split()) > 4:
            continue
        prop_objects[prop].add(obj)
        rows.append({"subj": subj, "prop": prop, "obj": obj})
    quads = []
    for r in rows:
        others = list(prop_objects[r["prop"]] - {r["obj"]})
        if not others:
            continue
        wrong = random.choice(others)
        prefix = f"The {r['prop']} of {r['subj']} is"
        quads.append({
            "true": f"{prefix} {r['obj']}.",
            "false": f"{prefix} {wrong}.",
            "neg_true": f"{prefix} not {r['obj']}.",
            "neg_false": f"{prefix} not {wrong}.",
        })
    random.shuffle(quads)
    return quads

print("Preparing datasets...")
datasets = {
    "CounterFact": prepare_counterfact(),
    "FM Queries": prepare_fm(),
    "PopQA": prepare_popqa(),
}
for name, quads in datasets.items():
    print(f"  {name}: {len(quads)} quadruples")


# ── Helper functions ─────────────────────────────────────────────────────────

def train_wfact(h_true_list, h_false_list):
    n = len(h_true_list)
    X = np.concatenate([np.stack(h_true_list), np.stack(h_false_list)])
    y = np.array([1]*n + [0]*n)
    dim = min(PCA_DIM, X.shape[0]-1, X.shape[1])
    pca = PCA(n_components=dim, random_state=42)
    Xp = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(Xp, y)
    w = pca.components_.T @ clf.coef_[0]
    w = w / (np.linalg.norm(w) + 1e-8)
    auc = roc_auc_score(y, X @ w)
    return w, auc

def run_negation_test(train_quads, test_quads, get_hidden_fn, label):
    # Train
    ht, hf = [], []
    for q in tqdm(train_quads, desc=f"{label} train"):
        try:
            ht.append(get_hidden_fn(q["true"]))
            hf.append(get_hidden_fn(q["false"]))
        except:
            continue
    w, train_auc = train_wfact(ht, hf)
    print(f"  Train AUC: {train_auc:.4f} (n={len(ht)})")

    # Test
    h_t, h_f, h_nt, h_nf = [], [], [], []
    for q in tqdm(test_quads, desc=f"{label} test"):
        try:
            h_t.append(get_hidden_fn(q["true"]))
            h_f.append(get_hidden_fn(q["false"]))
            h_nt.append(get_hidden_fn(q["neg_true"]))
            h_nf.append(get_hidden_fn(q["neg_false"]))
        except:
            continue

    n = len(h_t)
    if n < 10:
        return None
    s_t = np.stack(h_t) @ w
    s_f = np.stack(h_f) @ w
    s_nt = np.stack(h_nt) @ w
    s_nf = np.stack(h_nf) @ w

    auc_tf = roc_auc_score([1]*n+[0]*n, np.concatenate([s_t, s_f]))
    neg_dec = float(np.mean(s_t > s_nt))
    neg_inc = float(np.mean(s_nf > s_f))
    interaction = float(np.mean((s_nf - s_f) - (s_nt - s_t)))

    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.randint(0, n, size=n)
        boot.append(float(np.mean((s_nf[idx]-s_f[idx]) - (s_nt[idx]-s_t[idx]))))
    ci_lo, ci_hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    print(f"  n={n}, AUC={auc_tf:.4f}, Neg dec true={neg_dec*100:.1f}%, Neg inc false={neg_inc*100:.1f}%, Interaction={interaction:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "n": n, "train_auc": round(train_auc, 4),
        "auc_true_vs_false": round(auc_tf, 4),
        "neg_dec_true": round(neg_dec, 4),
        "neg_inc_false": round(neg_inc, 4),
        "interaction": round(interaction, 4),
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    }


# ── Run 3 models x 3 datasets ───────────────────────────────────────────────

all_results = {}

for model_path, model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map=DEVICE, output_hidden_states=True,
    )
    model.eval()

    def get_hidden(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=128, padding=False).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[BEST_LAYER][0, -1, :].float().cpu().numpy()

    for ds_name, quads in datasets.items():
        label = f"{model_name} / {ds_name}"
        print(f"\n--- {label} ---")
        train_q = quads[:N_TRAIN]
        test_q = quads[N_TRAIN:N_TRAIN+N_TEST]
        res = run_negation_test(train_q, test_q, get_hidden, label)
        if res:
            all_results[f"{model_name}_{ds_name}"] = res

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\n{model_name} unloaded.")


# ── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("FULL 3x3 RESULTS")
print("="*70)

ds_names = ["CounterFact", "FM Queries", "PopQA"]
model_names = [m[1] for m in MODELS]

# Header
header = f"{'Dataset':<15}"
for mn in model_names:
    header += f" | {mn:>28}"
print(header)
print("-" * len(header))

for ds_name in ds_names:
    row = f"{ds_name:<15}"
    for mn in model_names:
        key = f"{mn}_{ds_name}"
        if key in all_results:
            r = all_results[key]
            ci = r["ci_95"]
            row += f" | {r['interaction']:+.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
        else:
            row += f" | {'N/A':>28}"
    print(row)

print("\nDetailed:")
print(f"{'Model':<14} {'Dataset':<14} {'n':>5} {'AUC':>6} {'Neg↓T':>6} {'Neg↑F':>6} {'Inter':>7} {'95% CI':<18}")
print("-"*82)
for key, r in all_results.items():
    parts = key.split("_", 1)
    mn, dn = parts[0], parts[1]
    ci = r["ci_95"]
    sig = "*" if ci[0] > 0 else ""
    print(f"{mn:<14} {dn:<14} {r['n']:>5} {r['auc_true_vs_false']:>6.3f} {r['neg_dec_true']*100:>5.1f}% {r['neg_inc_false']*100:>5.1f}% {r['interaction']:>+7.4f} [{ci[0]:.4f},{ci[1]:.4f}]{sig}")

# Save
save_path = os.path.join(SAVE_DIR, "full_3x3_results.json")
with open(save_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {save_path}")