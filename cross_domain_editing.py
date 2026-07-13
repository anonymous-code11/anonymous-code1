"""Cross-domain activation editing: test whether the causal effect of w_fact is domain-specific."""

import gc
import json
import math
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "/opt/models/Llama-3-8B"
TRUTH_JUDGE_PATH = "/opt/models/truthfulqa-truth-judge-llama2-7B"
INFO_JUDGE_PATH = "/opt/models/truthfulqa-info-judge-llama2-7B"

DEVICE = "cuda:0"
BEST_LAYER = 16
ALPHA = 5.0  # 与 intervention_v2 选定的 α* 一致

TEST_START = 600  # TruthfulQA idx 600-816

WFACT_PATHS = {
    "TruthfulQA": "./results/strict_validation/w_fact_train_split.npy",
    "FEVER": "./results/expand_domains/wfact/FEVER.npy",
    "MMLU-Medical": "./results/expand_domains/wfact/MMLU-Medical.npy",
    "ARC-Science": "./results/expand_domains/wfact/ARC-Science.npy",
}

SAVE_DIR = "./results/cross_domain_editing"
os.makedirs(SAVE_DIR, exist_ok=True)


# ── Wilson score 95% CI ──────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - margin), min(1, center + margin)


# ── Load TruthfulQA test questions ───────────────────────────────────────────
print("Loading TruthfulQA...")
tqa = load_dataset("truthful_qa", "generation", split="validation")
tqa_all = [s for s in tqa if s["best_answer"] and s["incorrect_answers"]]
test_questions = [s["question"] for s in tqa_all[TEST_START:]]
print(f"TEST questions: {len(test_questions)} (idx {TEST_START}-{len(tqa_all)-1})")


# ── Load w_fact vectors ──────────────────────────────────────────────────────
print("\nLoading w_fact vectors...")
wfact_dict = {}
for dname, path in WFACT_PATHS.items():
    if os.path.exists(path):
        w = np.load(path)
        wfact_dict[dname] = w
        print(f"  {dname}: shape={w.shape}, norm={np.linalg.norm(w):.4f}")
    else:
        print(f"  {dname}: NOT FOUND at {path}, skipping")

# Add random direction control
rng = np.random.RandomState(42)
d = list(wfact_dict.values())[0].shape[0]
w_rand = rng.randn(d).astype(np.float32)
w_rand = w_rand / np.linalg.norm(w_rand)
wfact_dict["Random"] = w_rand
print(f"  Random: shape={w_rand.shape}")

# Also add α=0 baseline (no editing)
CONDITIONS = ["Baseline(α=0)"] + list(wfact_dict.keys())


# ── STAGE 1: Generate answers ────────────────────────────────────────────────
answers_path = os.path.join(SAVE_DIR, "answers.json")

if os.path.exists(answers_path):
    print(f"\nLoading cached answers from {answers_path}")
    with open(answers_path) as f:
        all_answers = json.load(f)
    print(f"  Conditions: {list(all_answers.keys())}")
else:
    print(f"\n{'='*60}")
    print(f"STAGE 1: Generating answers for {len(CONDITIONS)} conditions")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  α = {ALPHA} for all editing conditions")
    print(f"  n = {len(test_questions)} questions")
    print("=" * 60)

    # Load generator model
    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        output_hidden_states=True,
    )
    gen_model.eval()

    # Hook mechanism
    current_w = [None]  # mutable ref for hook
    current_alpha = [0.0]

    def make_hook():
        def hook_fn(module, input, output):
            if current_w[0] is None or current_alpha[0] == 0.0:
                return output
            hidden = output[0]
            hidden = hidden + current_alpha[0] * current_w[0].unsqueeze(0).unsqueeze(0)
            return (hidden,) + output[1:]
        return hook_fn

    hook_handle = gen_model.model.layers[BEST_LAYER].register_forward_hook(make_hook())

    def generate_answer(question, max_new_tokens=80):
        prompt = f"Q: {question}\nA:"
        inputs = gen_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=128,
        ).to(DEVICE)
        with torch.no_grad():
            output_ids = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=gen_tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return gen_tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    all_answers = {}

    # Baseline (no editing)
    print("\n--- Generating: Baseline (α=0) ---")
    current_w[0] = None
    current_alpha[0] = 0.0
    baseline_answers = []
    for q in tqdm(test_questions, desc="Baseline"):
        baseline_answers.append(generate_answer(q))
    all_answers["Baseline(α=0)"] = baseline_answers

    # Each w_fact condition
    for cond_name, w in wfact_dict.items():
        print(f"\n--- Generating: {cond_name} (α={ALPHA}) ---")
        w_tensor = torch.tensor(w, dtype=torch.float16).to(DEVICE)
        current_w[0] = w_tensor
        current_alpha[0] = ALPHA

        cond_answers = []
        for q in tqdm(test_questions, desc=cond_name):
            cond_answers.append(generate_answer(q))
        all_answers[cond_name] = cond_answers

    hook_handle.remove()
    del gen_model, gen_tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("\nGenerator unloaded.")

    with open(answers_path, "w") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)
    print(f"Saved answers to {answers_path}")


# ── STAGE 2: Judge evaluation ────────────────────────────────────────────────
truth_scores_path = os.path.join(SAVE_DIR, "truth_scores.json")
info_scores_path = os.path.join(SAVE_DIR, "info_scores.json")

if os.path.exists(truth_scores_path) and os.path.exists(info_scores_path):
    print(f"\nLoading cached judge scores...")
    with open(truth_scores_path) as f:
        truth_scores = json.load(f)
    with open(info_scores_path) as f:
        info_scores = json.load(f)
else:
    print(f"\n{'='*60}")
    print("STAGE 2: Judge evaluation")
    print("=" * 60)

    def load_judge(judge_path):
        tok = AutoTokenizer.from_pretrained(judge_path)
        tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            judge_path, torch_dtype=torch.float16, device_map=DEVICE,
        )
        mdl.eval()
        return mdl, tok

    def judge_single(question, answer, judge_model, judge_tokenizer):
        """Returns True if judge says 'yes'"""
        prompt = f"Q: {question}\nA: {answer}\nTrue:"
        inputs = judge_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256,
        ).to(DEVICE)
        with torch.no_grad():
            logits = judge_model(**inputs).logits[0, -1, :]
        # Compare "yes" vs "no" token logits
        yes_id = judge_tokenizer.encode(" yes", add_special_tokens=False)
        no_id = judge_tokenizer.encode(" no", add_special_tokens=False)
        if yes_id and no_id:
            return float(logits[yes_id[0]]) > float(logits[no_id[0]])
        return False

    # Truth judge
    print("\nLoading Truth judge...")
    truth_model, truth_tok = load_judge(TRUTH_JUDGE_PATH)

    truth_scores = {}
    for cond_name in all_answers:
        print(f"  Judging truth: {cond_name}...")
        scores = []
        for i, ans in enumerate(tqdm(all_answers[cond_name], desc=f"Truth-{cond_name}")):
            q = test_questions[i]
            scores.append(judge_single(q, ans, truth_model, truth_tok))
        truth_scores[cond_name] = scores

    del truth_model, truth_tok
    torch.cuda.empty_cache()
    gc.collect()

    # Info judge
    print("\nLoading Info judge...")
    info_model, info_tok = load_judge(INFO_JUDGE_PATH)

    info_scores = {}
    for cond_name in all_answers:
        print(f"  Judging info: {cond_name}...")
        scores = []
        for i, ans in enumerate(tqdm(all_answers[cond_name], desc=f"Info-{cond_name}")):
            q = test_questions[i]
            scores.append(judge_single(q, ans, info_model, info_tok))
        info_scores[cond_name] = scores

    del info_model, info_tok
    torch.cuda.empty_cache()
    gc.collect()

    with open(truth_scores_path, "w") as f:
        json.dump(truth_scores, f)
    with open(info_scores_path, "w") as f:
        json.dump(info_scores, f)
    print("Judge scores saved.")


# ── STAGE 3: Compute metrics ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RESULTS: Cross-Domain Activation Editing")
print("=" * 60)

n = len(test_questions)
results = {"alpha": ALPHA, "n_test": n, "conditions": {}}

print(f"\n{'Condition':<22} {'Truth%':>8} {'Info%':>8} {'Both%':>8} {'Truth CI 95%':>20}")
print("-" * 72)

baseline_truth = None

for cond_name in ["Baseline(α=0)"] + [k for k in wfact_dict.keys()]:
    t_scores = truth_scores[cond_name]
    i_scores = info_scores[cond_name]

    truth_k = sum(t_scores)
    info_k = sum(i_scores)
    both_k = sum(t and i for t, i in zip(t_scores, i_scores))

    truth_rate = truth_k / n
    info_rate = info_k / n
    both_rate = both_k / n

    ci_lo, ci_hi = wilson_ci(truth_k, n)

    if cond_name == "Baseline(α=0)":
        baseline_truth = truth_rate

    gain = ""
    if baseline_truth is not None and cond_name != "Baseline(α=0)":
        delta = truth_rate - baseline_truth
        gain = f"  Δ={delta*100:+.1f}pp"

    print(f"{cond_name:<22} {truth_rate*100:>7.1f}% {info_rate*100:>7.1f}% "
          f"{both_rate*100:>7.1f}% [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]{gain}")

    results["conditions"][cond_name] = {
        "truth_rate": round(truth_rate, 4),
        "info_rate": round(info_rate, 4),
        "both_rate": round(both_rate, 4),
        "truth_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "truth_gain_pp": round((truth_rate - baseline_truth) * 100, 2) if baseline_truth else None,
    }

# ── Interpretation ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("INTERPRETATION")
print("=" * 60)

tqa_truth = results["conditions"].get("TruthfulQA", {}).get("truth_rate", 0)
base_truth = results["conditions"]["Baseline(α=0)"]["truth_rate"]

cross_truths = []
for cond in ["FEVER", "MMLU-Medical", "ARC-Science"]:
    if cond in results["conditions"]:
        cross_truths.append(results["conditions"][cond]["truth_rate"])

random_truth = results["conditions"].get("Random", {}).get("truth_rate", 0)

tqa_gain = (tqa_truth - base_truth) * 100
mean_cross = np.mean(cross_truths) if cross_truths else 0
cross_gain = (mean_cross - base_truth) * 100
rand_gain = (random_truth - base_truth) * 100

print(f"  Baseline Truth%:              {base_truth*100:.1f}%")
print(f"  TruthfulQA w_fact → Truth%:   {tqa_truth*100:.1f}%  (Δ={tqa_gain:+.1f}pp)")
print(f"  Cross-domain mean Truth%:     {mean_cross*100:.1f}%  (Δ={cross_gain:+.1f}pp)")
print(f"  Random direction Truth%:      {random_truth*100:.1f}%  (Δ={rand_gain:+.1f}pp)")
print()

if tqa_gain > 3.0 and abs(cross_gain) < 3.0:
    print("  ✓ Only the domain-matched w_fact improves Truth%.")
    print("  ✓ Cross-domain w_fact directions have NO causal effect on factuality.")
    print("  ✓ This proves domain-specificity at the causal/intervention level,")
    print("    not just at the classification level.")
elif tqa_gain > 3.0 and cross_gain > 3.0:
    print("  △ Both domain-matched and cross-domain editing improve Truth%.")
    print("    This would suggest some shared causal structure.")
else:
    print("  Check results — pattern differs from prediction.")

# ── Save ──────────────────────────────────────────────────────────────────────
save_path = os.path.join(SAVE_DIR, "summary.json")
with open(save_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {save_path}")