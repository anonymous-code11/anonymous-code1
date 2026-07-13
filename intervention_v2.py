
import os
import gc
import json
import math
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH       = "/opt/models/Llama-3-8B"
TRUTH_JUDGE_PATH = "/opt/models/truthfulqa-truth-judge-llama2-7B"
INFO_JUDGE_PATH  = "/opt/models/truthfulqa-info-judge-llama2-7B"

DEVICE      = "cuda:0"
BEST_LAYER  = 16
ALPHAS      = [0.0, 2.0, 5.0, 10.0, 20.0]

TRAIN_SIZE  = 400      # questions 0-399  → wfact 估计
VAL_START   = 400      # questions 400-599 → α 选取（复用缓存）
VAL_END     = 600
TEST_START  = 600      # questions 600-816 → 最终报告

SAVE_DIR = "./results/intervention_v2"
FIG_DIR  = "./figures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── Wilson score 95% CI ───────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    """返回 (low, high) 比例置信区间"""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - margin), min(1, center + margin)

# ── 加载 TQA ─────────────────────────────────────────────────────────────────
print("Loading TruthfulQA...")
tqa = load_dataset("truthful_qa", "generation", split="validation")
tqa_all = [s for s in tqa if s["best_answer"] and s["incorrect_answers"]]
print(f"Total valid TQA: {len(tqa_all)}")

val_questions  = [s["question"] for s in tqa_all[VAL_START:VAL_END]]
test_questions = [s["question"] for s in tqa_all[TEST_START:]]
print(f"VAL  questions: {len(val_questions)}  (idx {VAL_START}-{VAL_END-1})")
print(f"TEST questions: {len(test_questions)}  (idx {TEST_START}-{len(tqa_all)-1})")

# ── 加载 wfact（从已缓存的 strict_validation） ───────────────────────────────
wfact_path = "./results/strict_validation/w_fact_train_split.npy"
wflu_path  = "./results/strict_validation/w_flu_train_split.npy"

if os.path.exists(wfact_path):
    print(f"\nLoading precomputed wfact from {wfact_path}")
    w_fact_np = np.load(wfact_path)
    print(f"  wfact shape: {w_fact_np.shape}")
else:
    print("\nRecomputing wfact from TQA train split (0-399)...")
    data = np.load("./results/hidden_states.npz")
    hc   = data["h_correct"][:TRAIN_SIZE]
    hw   = data["h_wrong"][:TRAIN_SIZE]
    N    = hc.shape[0]
    X    = np.concatenate([hc[:, BEST_LAYER, :], hw[:, BEST_LAYER, :]])
    y    = np.array([1]*N + [0]*N)
    pca  = PCA(n_components=128, random_state=42)
    Xp   = pca.fit_transform(X)
    clf  = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(Xp, y)
    w_fact_np = pca.components_.T @ clf.coef_[0]
    w_fact_np = w_fact_np / (np.linalg.norm(w_fact_np) + 1e-8)
    np.save(wfact_path, w_fact_np)
    print(f"  Saved: {wfact_path}")

# ── α 选取：复用已有 val set 结果 ────────────────────────────────────────────
# strict_validation 已生成 questions 400-599 的答案和 judge 分数
val_answers_path = "./results/strict_validation/answers_test_split.json"
val_summary_path = "./results/strict_validation/summary.json"

print("\nLoading VAL set results from strict_validation...")
with open(val_answers_path) as f:
    val_answers = json.load(f)   # dict: alpha_str -> [answer, ...]
with open(val_summary_path) as f:
    val_summary = json.load(f)

# 从 val summary 中找最优 α（基于 Both%）
val_alphas   = sorted(val_summary["intervention"].keys(),
                      key=lambda a: float(a))
val_best_alpha = max(
    val_alphas,
    key=lambda a: val_summary["intervention"][a]["both_rate"]
)
print(f"  VAL set α selection (maximize Both%):")
for a in val_alphas:
    d = val_summary["intervention"][a]
    marker = " ← selected" if a == val_best_alpha else ""
    print(f"    α={float(a):5.1f}  Truth%={d['truth_rate']*100:.1f}  "
          f"Both%={d['both_rate']*100:.1f}{marker}")

SELECTED_ALPHA = float(val_best_alpha)
print(f"\nSelected α = {SELECTED_ALPHA} (from VAL set)")

# ── STAGE 1：生成 TEST set 答案 ───────────────────────────────────────────────
test_answers_path = os.path.join(SAVE_DIR, "test_answers.json")

if os.path.exists(test_answers_path):
    print(f"\nLoading cached TEST answers from {test_answers_path}")
    with open(test_answers_path) as f:
        test_answers = json.load(f)
    print(f"  Cached alphas: {list(test_answers.keys())}")
else:
    print(f"\n{'='*55}")
    print(f"STAGE 1: Generating TEST answers (n={len(test_questions)})")
    print(f"  Alphas: {ALPHAS}")
    print('='*55)

    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map=DEVICE, output_hidden_states=True,
    )
    gen_model.eval()

    w_fact_t = torch.tensor(w_fact_np, dtype=torch.float16).to(DEVICE)
    alpha_ref = [0.0]

    def make_hook():
        def hook_fn(module, input, output):
            if alpha_ref[0] == 0.0:
                return output
            hidden = output[0]
            hidden = hidden + alpha_ref[0] * w_fact_t.unsqueeze(0).unsqueeze(0)
            return (hidden,) + output[1:]
        return hook_fn

    hook_handle = gen_model.model.layers[BEST_LAYER].register_forward_hook(make_hook())

    def generate_answer(question, alpha, max_new_tokens=80):
        alpha_ref[0] = alpha
        prompt = f"Q: {question}\nA:"
        inputs = gen_tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=128,
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

    test_answers = {str(a): [] for a in ALPHAS}
    for question in tqdm(test_questions, desc="TEST generation"):
        for alpha in ALPHAS:
            ans = generate_answer(question, alpha)
            test_answers[str(alpha)].append(ans)

    hook_handle.remove()
    del gen_model, gen_tokenizer, w_fact_t
    torch.cuda.empty_cache()
    gc.collect()
    print("Generator unloaded.")

    with open(test_answers_path, "w") as f:
        json.dump(test_answers, f, ensure_ascii=False, indent=2)
    print(f"Saved: {test_answers_path}")

# ── 工具：judge 评分 ──────────────────────────────────────────────────────────
def judge_all(questions, answers_by_alpha, judge_model, judge_tokenizer,
              label="judge", alphas=None):
    """
    返回 dict: alpha_str -> list of bool (is_yes)
    """
    if alphas is None:
        alphas = list(answers_by_alpha.keys())
    scores = {a: [] for a in alphas}
    for alpha in alphas:
        for q, a in tqdm(zip(questions, answers_by_alpha[alpha]),
                         total=len(questions), desc=f"{label} α={alpha}"):
            prompt  = f"Q: {q}\nA: {a}\nTrue:"
            inputs  = judge_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(DEVICE)
            with torch.no_grad():
                logits = judge_model(**inputs).logits[0, -1, :]
            yes_id = judge_tokenizer(" yes", add_special_tokens=False).input_ids[0]
            no_id  = judge_tokenizer(" no",  add_special_tokens=False).input_ids[0]
            is_yes = logits[yes_id].item() > logits[no_id].item()
            scores[alpha].append(bool(is_yes))
    return scores

# ── STAGE 2：Truth judge ───────────────────────────────────────────────────────
truth_scores_path = os.path.join(SAVE_DIR, "test_truth_scores.json")

if os.path.exists(truth_scores_path):
    print(f"\nLoading cached truth scores from {truth_scores_path}")
    with open(truth_scores_path) as f:
        test_truth = json.load(f)
else:
    print(f"\n{'='*55}")
    print("STAGE 2: Truth judge scoring")
    print('='*55)

    t_tok = AutoTokenizer.from_pretrained(TRUTH_JUDGE_PATH)
    t_mdl = AutoModelForCausalLM.from_pretrained(
        TRUTH_JUDGE_PATH, torch_dtype=torch.float16, device_map=DEVICE)
    t_mdl.eval()

    test_truth = judge_all(test_questions, test_answers, t_mdl, t_tok,
                            label="Truth")
    del t_mdl, t_tok
    torch.cuda.empty_cache(); gc.collect()

    with open(truth_scores_path, "w") as f:
        json.dump(test_truth, f, indent=2)
    print(f"Saved: {truth_scores_path}")

# ── STAGE 3：Info judge ────────────────────────────────────────────────────────
info_scores_path = os.path.join(SAVE_DIR, "test_info_scores.json")

if os.path.exists(info_scores_path):
    print(f"\nLoading cached info scores from {info_scores_path}")
    with open(info_scores_path) as f:
        test_info = json.load(f)
else:
    print(f"\n{'='*55}")
    print("STAGE 3: Info judge scoring")
    print('='*55)

    i_tok = AutoTokenizer.from_pretrained(INFO_JUDGE_PATH)
    i_mdl = AutoModelForCausalLM.from_pretrained(
        INFO_JUDGE_PATH, torch_dtype=torch.float16, device_map=DEVICE)
    i_mdl.eval()

    test_info = judge_all(test_questions, test_answers, i_mdl, i_tok,
                           label="Info")
    del i_mdl, i_tok
    torch.cuda.empty_cache(); gc.collect()

    with open(info_scores_path, "w") as f:
        json.dump(test_info, f, indent=2)
    print(f"Saved: {info_scores_path}")

# ── 汇总 TEST 结果 ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"TEST SET RESULTS  (α selected on VAL set, α* = {SELECTED_ALPHA})")
print(f"n = {len(test_questions)} questions (TQA idx {TEST_START}-{len(tqa_all)-1})")
print('='*70)
print(f"{'α':>8}  {'Truth%':>8}  {'95% CI':>16}  {'Info%':>8}  {'Both%':>8}  {'95% CI':>16}")
print('-'*70)

summary_test = {}
n = len(test_questions)
for alpha in ALPHAS:
    a_str = str(alpha)
    truth_list = test_truth[a_str]
    info_list  = test_info[a_str]
    both_list  = [t and i for t, i in zip(truth_list, info_list)]

    truth_rate = sum(truth_list) / n
    info_rate  = sum(info_list)  / n
    both_rate  = sum(both_list)  / n

    t_lo, t_hi = wilson_ci(sum(truth_list), n)
    b_lo, b_hi = wilson_ci(sum(both_list),  n)

    ci_truth = f"[{t_lo*100:.1f}, {t_hi*100:.1f}]"
    ci_both  = f"[{b_lo*100:.1f}, {b_hi*100:.1f}]"

    marker = " ←α*" if abs(alpha - SELECTED_ALPHA) < 0.01 else ""
    print(f"{alpha:>8.1f}  {truth_rate*100:>7.1f}%  {ci_truth:>16}  "
          f"{info_rate*100:>7.1f}%  {both_rate*100:>7.1f}%  {ci_both:>16}{marker}")

    summary_test[a_str] = {
        "truth_rate": truth_rate,
        "info_rate":  info_rate,
        "both_rate":  both_rate,
        "truth_ci_95": [t_lo, t_hi],
        "both_ci_95":  [b_lo, b_hi],
        "n": n,
    }
print('='*70)

# 计算 α* 相对 baseline 的 gain
base  = summary_test["0.0"]
best  = summary_test[str(SELECTED_ALPHA)]
truth_gain = best["truth_rate"] - base["truth_rate"]
both_gain  = best["both_rate"]  - base["both_rate"]

# gain 的 95% CI（两比例之差的 Wilson 近似）
def diff_ci(p1, p2, n):
    se = math.sqrt(p1*(1-p1)/n + p2*(1-p2)/n)
    diff = p1 - p2
    return diff - 1.96*se, diff + 1.96*se

truth_gain_lo, truth_gain_hi = diff_ci(
    best["truth_rate"], base["truth_rate"], n)
both_gain_lo,  both_gain_hi  = diff_ci(
    best["both_rate"],  base["both_rate"],  n)

print(f"\nα* = {SELECTED_ALPHA} vs baseline (α=0):")
print(f"  Truth% gain: +{truth_gain*100:.1f}pp  "
      f"95% CI [{truth_gain_lo*100:.1f}, {truth_gain_hi*100:.1f}]pp")
print(f"  Both%  gain: +{both_gain*100:.1f}pp  "
      f"95% CI [{both_gain_lo*100:.1f}, {both_gain_hi*100:.1f}]pp")
print(f"  n = {n}  (vs original n=200)")

# ── 保存结果 ──────────────────────────────────────────────────────────────────
final = {
    "val_alpha_selection": {
        "source":        "strict_validation (TQA 400-599)",
        "selected_alpha": SELECTED_ALPHA,
        "criterion":     "maximize Both%",
    },
    "test_set": {
        "source":     f"TQA idx {TEST_START}-{len(tqa_all)-1}",
        "n":          n,
    },
    "test_results": summary_test,
    "gain_at_best_alpha": {
        "alpha":          SELECTED_ALPHA,
        "truth_gain_pp":  truth_gain * 100,
        "both_gain_pp":   both_gain  * 100,
        "truth_gain_ci95_pp": [truth_gain_lo*100, truth_gain_hi*100],
        "both_gain_ci95_pp":  [both_gain_lo*100,  both_gain_hi*100],
    },
}
with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
    json.dump(final, f, indent=2)
print(f"\nSaved: {os.path.join(SAVE_DIR, 'summary.json')}")

# ── 绘图 ──────────────────────────────────────────────────────────────────────
truth_rates = [summary_test[str(a)]["truth_rate"] * 100 for a in ALPHAS]
info_rates  = [summary_test[str(a)]["info_rate"]  * 100 for a in ALPHAS]
both_rates  = [summary_test[str(a)]["both_rate"]  * 100 for a in ALPHAS]

# Truth% 的 95% CI（Wilson）
truth_lo = [summary_test[str(a)]["truth_ci_95"][0] * 100 for a in ALPHAS]
truth_hi = [summary_test[str(a)]["truth_ci_95"][1] * 100 for a in ALPHAS]
truth_err_lo = [t - lo for t, lo in zip(truth_rates, truth_lo)]
truth_err_hi = [hi - t  for t, hi in zip(truth_rates, truth_hi)]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(ALPHAS, truth_rates, "b-o", linewidth=2, markersize=7,
        label="Truthful%")
ax.fill_between(ALPHAS, truth_lo, truth_hi, alpha=0.15, color="blue",
                label="Truth 95% CI")
ax.plot(ALPHAS, info_rates,  "g-s", linewidth=2, markersize=7,
        label="Informative%")
ax.plot(ALPHAS, both_rates,  "r-^", linewidth=2, markersize=7,
        label="Both%")

# 标注 α*
ax.axvline(SELECTED_ALPHA, color="purple", linestyle="--", alpha=0.5,
           label=f"α* = {SELECTED_ALPHA} (selected on VAL)")

ax.set_xlabel("Intervention Strength α", fontsize=13)
ax.set_ylabel("Score (%)", fontsize=13)
ax.set_title(f"Activation Editing Results (TEST set, n={n})\n"
             f"α* selected on independent VAL set (n=200), "
             f"Llama-3-8B + TruthfulQA",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "intervention_v2.png")
plt.savefig(fig_path, dpi=150)
print(f"Saved: {fig_path}")

print("\nDone.")
