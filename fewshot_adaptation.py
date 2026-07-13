

import os, json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SAVE_DIR = "./results/fewshot_adaptation"
FIG_DIR  = "./figures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

BEST_LAYER = 16
PCA_DIM    = 128
N_TRIALS   = 10     # 每个 N 重复多少次随机采样
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

N_ADAPT_LIST = [10, 20, 50, 100, 200]
N_EVAL_FIXED = 100   # 固定100个样本作为eval集

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def build_wfact_from_arrays(hc, hw, layer=BEST_LAYER, pca_dim=PCA_DIM):
    N  = min(hc.shape[0], hw.shape[0])
    if N < 4:
        return None
    X  = np.concatenate([hc[:N, layer, :], hw[:N, layer, :]], axis=0)
    y  = np.array([1]*N + [0]*N)
    dim = min(pca_dim, X.shape[0]-1, X.shape[1])
    if dim < 1:
        return None
    pca = PCA(n_components=dim, random_state=42)
    Xp  = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(Xp, y)
    w = pca.components_.T @ clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-8)


def compute_auc(hc, hw, wfact, layer=BEST_LAYER):
    N  = min(hc.shape[0], hw.shape[0])
    y  = np.array([1]*N + [0]*N)
    sc = np.concatenate([hc[:N, layer, :] @ wfact,
                         hw[:N, layer, :] @ wfact])
    try:
        return float(roc_auc_score(y, sc))
    except Exception:
        return 0.5


def adapt_direction(wfact_src, hc_target, hw_target, layer=BEST_LAYER, lam=1.0):
    """Condition C: src direction + target mean-diff correction"""
    N = min(hc_target.shape[0], hw_target.shape[0])
    delta = (hc_target[:N, layer, :].mean(axis=0) -
             hw_target[:N, layer, :].mean(axis=0))
    norm = np.linalg.norm(delta)
    if norm < 1e-8:
        return wfact_src
    delta = delta / norm
    w_adapted = wfact_src + lam * delta
    return w_adapted / (np.linalg.norm(w_adapted) + 1e-8)


# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("Loading hidden states...")

OLD_CACHE = {
    "TruthfulQA":   "./results/hidden_states.npz",
}
NEW_CACHE_DIR = "./results/expand_domains"

def load_domain(dname):
    if dname in OLD_CACHE:
        d = np.load(OLD_CACHE[dname])
    else:
        d = np.load(os.path.join(NEW_CACHE_DIR, f"{dname}_hidden.npz"))
    return d["h_correct"], d["h_wrong"]

# 源域
hc_src, hw_src = load_domain("TruthfulQA")
print(f"Source (TruthfulQA): hc={hc_src.shape}")

# 构建源域 wfact（用全部400样本）
wfact_src = build_wfact_from_arrays(hc_src[:400], hw_src[:400])
print(f"Source wfact: norm={np.linalg.norm(wfact_src):.4f}")

# 目标域
TARGET_DOMAINS = ["FEVER", "MMLU-Medical", "ARC-Science"]
target_data = {}
for dname in TARGET_DOMAINS:
    hc, hw = load_domain(dname)
    target_data[dname] = (hc, hw)
    print(f"Target {dname}: hc={hc.shape}")

# ── 主实验循环 ────────────────────────────────────────────────────────────────
print("\nRunning few-shot adaptation experiment...")
print(f"N_ADAPT_LIST = {N_ADAPT_LIST}")
print(f"N_TRIALS per N = {N_TRIALS}")

results_by_domain = {}

for dname in TARGET_DOMAINS:
    hc_tgt, hw_tgt = target_data[dname]
    N_tgt = min(hc_tgt.shape[0], hw_tgt.shape[0])
    N_eval = min(N_EVAL_FIXED, N_tgt // 4)   # 最多用1/4作eval
    pool_size = N_tgt - N_eval

    if pool_size < 10:
        print(f"  {dname}: insufficient data ({N_tgt}), skipping")
        continue

    print(f"\n  {dname} (total={N_tgt}, eval={N_eval}, pool={pool_size})")

    # eval 集：最后 N_eval 个样本
    hc_eval = hc_tgt[N_tgt - N_eval : N_tgt]
    hw_eval = hw_tgt[N_tgt - N_eval : N_tgt]
    hc_pool   = hc_tgt[:pool_size]
    hw_pool   = hw_tgt[:pool_size]

    domain_results = {
        "zero_shot_auc":   None,
        "from_scratch":    {},   # N -> [auc1, auc2, ...]
        "src_adapt_lam1":  {},
        "src_adapt_lam2":  {},
    }

    # Condition A: Zero-shot
    z_auc = compute_auc(hc_eval, hw_eval, wfact_src)
    domain_results["zero_shot_auc"] = z_auc
    print(f"    Zero-shot AUC = {z_auc:.4f}")

    for N_adapt in N_ADAPT_LIST:
        if N_adapt > pool_size:
            print(f"    N={N_adapt}: not enough pool samples ({pool_size}), skip")
            continue

        aucs_scratch = []
        aucs_lam1    = []
        aucs_lam2    = []

        for trial in range(N_TRIALS):
            # 随机采样 N_adapt 个样本
            idx = np.random.choice(pool_size, N_adapt, replace=False)
            hc_adapt = hc_pool[idx]
            hw_adapt = hw_pool[idx]

            # Condition B: From scratch
            wf_scratch = build_wfact_from_arrays(hc_adapt, hw_adapt)
            if wf_scratch is not None:
                auc_s = compute_auc(hc_eval, hw_eval, wf_scratch)
                aucs_scratch.append(auc_s)

            # Condition C: Src + adapt (λ=1.0)
            wf_adapt1 = adapt_direction(wfact_src, hc_adapt, hw_adapt, lam=1.0)
            auc_a1 = compute_auc(hc_eval, hw_eval, wf_adapt1)
            aucs_lam1.append(auc_a1)

            # Condition C: Src + adapt (λ=2.0)
            wf_adapt2 = adapt_direction(wfact_src, hc_adapt, hw_adapt, lam=2.0)
            auc_a2 = compute_auc(hc_eval, hw_eval, wf_adapt2)
            aucs_lam2.append(auc_a2)

        domain_results["from_scratch"][N_adapt]   = aucs_scratch
        domain_results["src_adapt_lam1"][N_adapt] = aucs_lam1
        domain_results["src_adapt_lam2"][N_adapt] = aucs_lam2

        mean_s  = np.mean(aucs_scratch)  if aucs_scratch else float('nan')
        mean_a1 = np.mean(aucs_lam1)
        mean_a2 = np.mean(aucs_lam2)
        print(f"    N={N_adapt:3d}: scratch={mean_s:.4f} | adapt(λ=1)={mean_a1:.4f} | adapt(λ=2)={mean_a2:.4f}")

    results_by_domain[dname] = domain_results

# ── 绘图 ──────────────────────────────────────────────────────────────────────
n_target = len(results_by_domain)
fig, axes = plt.subplots(1, n_target, figsize=(5.5 * n_target, 5), sharey=True)
if n_target == 1:
    axes = [axes]

for ax, dname in zip(axes, results_by_domain.keys()):
    dr = results_by_domain[dname]
    z  = dr["zero_shot_auc"]

    valid_Ns = sorted([N for N in N_ADAPT_LIST
                       if N in dr["from_scratch"] and dr["from_scratch"][N]])

    # From scratch
    means_s = [np.mean(dr["from_scratch"][N])   for N in valid_Ns]
    stds_s  = [np.std(dr["from_scratch"][N])    for N in valid_Ns]
    # Src+adapt λ=1
    means_a1 = [np.mean(dr["src_adapt_lam1"][N]) for N in valid_Ns]
    stds_a1  = [np.std(dr["src_adapt_lam1"][N])  for N in valid_Ns]
    # Src+adapt λ=2
    means_a2 = [np.mean(dr["src_adapt_lam2"][N]) for N in valid_Ns]
    stds_a2  = [np.std(dr["src_adapt_lam2"][N])  for N in valid_Ns]

    ax.axhline(z, color="red", linewidth=2, linestyle="--", label=f"Zero-shot ({z:.3f})")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, label="Chance")

    ax.errorbar(valid_Ns, means_s,  yerr=stds_s,  fmt="o-",  color="steelblue",
                linewidth=2, markersize=7, capsize=4, label="From scratch")
    ax.errorbar(valid_Ns, means_a1, yerr=stds_a1, fmt="s--", color="darkorange",
                linewidth=2, markersize=7, capsize=4, label="Src+adapt (λ=1)")
    ax.errorbar(valid_Ns, means_a2, yerr=stds_a2, fmt="^:",  color="green",
                linewidth=2, markersize=7, capsize=4, label="Src+adapt (λ=2)")

    ax.set_xscale("log")
    ax.set_xticks(valid_Ns)
    ax.set_xticklabels([str(N) for N in valid_Ns])
    ax.set_xlabel("# target domain samples", fontsize=11)
    ax.set_ylabel("AUC" if ax == axes[0] else "", fontsize=11)
    ax.set_title(f"Target: {dname}", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.3, 1.05)

plt.suptitle("Few-Shot Domain Adaptation: TruthfulQA → Target Domain\n"
             "Does the source wfact help with limited target samples?",
             fontsize=12, y=1.02)
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "fewshot_adaptation.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {fig_path}")

# ── 保存结果 ──────────────────────────────────────────────────────────────────
output = {}
for dname, dr in results_by_domain.items():
    output[dname] = {
        "zero_shot_auc": dr["zero_shot_auc"],
        "from_scratch":  {N: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                          for N, v in dr["from_scratch"].items() if v},
        "src_adapt_lam1":{N: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                          for N, v in dr["src_adapt_lam1"].items() if v},
        "src_adapt_lam2":{N: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                          for N, v in dr["src_adapt_lam2"].items() if v},
    }

with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved: {os.path.join(SAVE_DIR, 'results.json')}")

print("\n" + "="*60)
print("FEW-SHOT ADAPTATION SUMMARY")
print("="*60)
for dname, dr in results_by_domain.items():
    z = dr["zero_shot_auc"]
    print(f"\n{dname}:")
    print(f"  Zero-shot AUC: {z:.4f}")
    for N in N_ADAPT_LIST:
        if N in dr["from_scratch"] and dr["from_scratch"][N]:
            ms = np.mean(dr["from_scratch"][N])
            ma1 = np.mean(dr["src_adapt_lam1"][N])
            delta = ma1 - ms
            print(f"  N={N:3d}: scratch={ms:.4f} | src+adapt(λ=1)={ma1:.4f} | Δ={delta:+.4f}")
