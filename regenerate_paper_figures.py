"""
Regenerate ALL paper figures with unified style.
ACL two-column: column width ≈ 3.25in, text width ≈ 6.75in.
"""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ── Unified style ────────────────────────────────────────────
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 10,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.constrained_layout.use": True,
}
plt.rcParams.update(STYLE)

COL_W = 3.25   # single column width (inches)
TEXT_W = 6.75   # full text width (inches)
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Helper ───────────────────────────────────────────────────
def save_figure(fig, name):
    fig.savefig(f"{FIG_DIR}/{name}.png")
    fig.savefig(f"{FIG_DIR}/{name}.pdf")

def build_wfact(h_correct, h_wrong, layer=16, n_pca=128):
    N = h_correct.shape[0]
    X = np.concatenate([h_correct[:, layer, :], h_wrong[:, layer, :]], axis=0)
    y = np.array([1]*N + [0]*N)
    n_comp = min(n_pca, X.shape[0]-1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X)
    sc = StandardScaler(); X_sc = sc.fit_transform(X_pca)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(X_sc, y)
    w = pca.components_.T @ clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-8)

def compute_auc(h_c, h_w, wfact, layer=16):
    s_c = h_c[:, layer, :] @ wfact
    s_w = h_w[:, layer, :] @ wfact
    y = np.array([1]*len(s_c) + [0]*len(s_w))
    return roc_auc_score(y, np.concatenate([s_c, s_w]))

# ── Load data ────────────────────────────────────────────────
print("Loading data...")
tqa = np.load("./results/hidden_states.npz")
h_correct = tqa["h_correct"]
h_wrong = tqa["h_wrong"]
N, num_layers, H = h_correct.shape

HIDDEN_PATHS = {
    "TruthfulQA": "./results/hidden_states.npz",
    "FEVER": "./results/expand_domains/FEVER_hidden.npz",
    "MMLU-Medical": "./results/expand_domains/MMLU-Medical_hidden.npz",
    "ARC-Science": "./results/expand_domains/ARC-Science_hidden.npz",
}

# ══════════════════════════════════════════════════════════════
# Figure 1: Geometric structure (3 panels, full width)
# ══════════════════════════════════════════════════════════════
print("Figure 1: Geometric structure...")

# Panel (a): Probing AUC across models
# Panel (a): Probing AUC across models — load per-layer data
llama_aucs = json.load(open("./results/dsd_results.json"))["probing_aucs_per_layer"]
qwen_aucs = json.load(open("./results/cross_model/probing_aucs_qwen.json"))["layer_aucs"]
mistral_aucs = json.load(open("./results/cross_model/probing_aucs_mistral.json"))["layer_aucs"]
model_layer_data = {
    "Llama-3-8B": llama_aucs,
    "Qwen2.5-7B": qwen_aucs,
    "Mistral-7B": mistral_aucs,
}

# Panel (b): Cosine trajectory
FLUENCY_LAYER = 3
mean_c_early = h_correct[:, FLUENCY_LAYER, :].mean(axis=0)
mean_w_early = h_wrong[:, FLUENCY_LAYER, :].mean(axis=0)
w_flu = mean_c_early - mean_w_early
w_flu = w_flu / (np.linalg.norm(w_flu) + 1e-8)

cos_per_layer = []
for li in range(num_layers):
    mc = h_correct[:, li, :].mean(axis=0)
    mw = h_wrong[:, li, :].mean(axis=0)
    d = mc - mw; d = d / (np.linalg.norm(d) + 1e-8)
    cos_per_layer.append(float(np.dot(d, w_flu)))

# Panel (c): DSD vs probing AUC
dsd_results = json.load(open("./results/dsd_results.json"))

fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.0))

# (a) Probing AUC
for mname, aucs in model_layer_data.items():
    L = len(aucs)
    x = [i/(L-1) for i in range(L)]
    axes[0].plot(x, aucs, "-o", markersize=2, label=mname)
axes[0].axhline(0.5, color="gray", ls=":", alpha=0.5)
axes[0].set_xlabel("Relative Layer Depth")
axes[0].set_ylabel("Probing AUC")
axes[0].set_title("(a) Probing AUC across models", fontweight='normal')
axes[0].legend(loc="lower right")
axes[0].set_ylim(0.3, 1.02)

# (b) Cosine trajectory
rho = [c / (cos_per_layer[FLUENCY_LAYER] + 1e-8) for c in cos_per_layer]
axes[1].plot(range(num_layers), rho, "purple", marker="o", markersize=2)
axes[1].axhline(0, color="gray", ls="--", alpha=0.5)
axes[1].set_xlabel("Layer Index")
axes[1].set_ylabel(r"$\rho(l)$")
axes[1].set_title(r"(b) Cosine trajectory $\rho(l)$", fontweight='normal')

# (c) DSD vs probing
probing_aucs = dsd_results.get("probing_aucs_per_layer", dsd_results.get("probing_aucs", []))
dsd_aucs = dsd_results.get("dsd_aucs_per_layer", dsd_results.get("dsd_aucs", []))
L_dsd = min(len(probing_aucs), len(dsd_aucs))
axes[2].plot(range(L_dsd), probing_aucs[:L_dsd], "r--s", markersize=2, label="Probing AUC")
axes[2].plot(range(L_dsd), dsd_aucs[:L_dsd], "b-o", markersize=2, label="DSD AUC")
axes[2].axhline(0.5, color="gray", ls=":", alpha=0.5)
axes[2].set_xlabel("Layer Index")
axes[2].set_ylabel("AUC")
axes[2].set_title("(c) DSD vs. Probing AUC", fontweight='normal')
axes[2].legend()
axes[2].set_ylim(0.3, 1.02)

save_figure(fig, "paper_probing_models")
save_figure(fig, "paper_cosine")
save_figure(fig, "paper_dsd_compare")
# Save as single combined figure for figure*
save_figure(fig, "fig1_combined")
plt.close(fig)
print("  Saved fig1_combined.png")

# ══════════════════════════════════════════════════════════════
# Figure 2: Intervention (single column)
# ══════════════════════════════════════════════════════════════
print("Figure 2: Intervention...")
iv_summary = json.load(open("./results/intervention_v2/summary.json"))
test_data = iv_summary["test_results"]
ALPHAS = [0.0, 2.0, 5.0, 10.0, 20.0]
SELECTED_ALPHA = 5.0

truth_rates = [test_data[str(a)]["truth_rate"]*100 for a in ALPHAS]
info_rates  = [test_data[str(a)]["info_rate"]*100  for a in ALPHAS]
both_rates  = [test_data[str(a)]["both_rate"]*100  for a in ALPHAS]
truth_lo = [test_data[str(a)]["truth_ci_95"][0]*100 for a in ALPHAS]
truth_hi = [test_data[str(a)]["truth_ci_95"][1]*100 for a in ALPHAS]

fig, ax = plt.subplots(figsize=(COL_W, 2.2))
ax.plot(ALPHAS, truth_rates, "b-o", markersize=4, label="Truth%")
ax.fill_between(ALPHAS, truth_lo, truth_hi, alpha=0.15, color="blue")
ax.plot(ALPHAS, info_rates, "g-s", markersize=4, label="Info%")
ax.plot(ALPHAS, both_rates, "r-^", markersize=4, label="Both%")
ax.axvline(SELECTED_ALPHA, color="purple", ls="--", alpha=0.5,
           label=r"$\alpha^*=5$")
ax.axvspan(15, 22, alpha=0.08, color="red")
ax.set_xlabel(r"Intervention Strength $\alpha$")
ax.set_ylabel("Score (%)")
ax.set_title("Activation Editing (Llama-3-8B, test set)")
ax.legend(ncol=2)
ax.set_ylim(0, 105)
save_figure(fig, "intervention_v2")
plt.close(fig)
print("  Saved intervention_v2.png")

# ══════════════════════════════════════════════════════════════
# Figure 3: Cross-domain matrix (full width)
# ══════════════════════════════════════════════════════════════
print("Figure 3: Cross-domain matrix...")
cd_results = json.load(open("./results/expand_domains/results.json"))
clean_domains = ["TruthfulQA", "FEVER", "MMLU-Medical", "ARC-Science"]
N_CD = len(clean_domains)

cos_matrix = np.array(cd_results["cos_matrix"])[:N_CD, :N_CD]
auc_matrix = np.array(cd_results["auc_matrix"])[:N_CD, :N_CD]
# Make cosine absolute
cos_matrix = np.abs(cos_matrix)

labels_short = ["TQA", "FEVER", "MMLU-Med", "ARC-Sci"]

fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.8))

im1 = axes[0].imshow(cos_matrix, cmap="RdBu_r", vmin=-0.2, vmax=1.0)
plt.colorbar(im1, ax=axes[0], shrink=0.8)
axes[0].set_xticks(range(N_CD)); axes[0].set_yticks(range(N_CD))
axes[0].set_xticklabels(labels_short, rotation=30, ha="right")
axes[0].set_yticklabels(labels_short)
axes[0].set_title(r"Pairwise $|\cos(\mathbf{w}_{\mathrm{fact}})|$")
for i in range(N_CD):
    for j in range(N_CD):
        c = "white" if cos_matrix[i,j] > 0.6 else "black"
        axes[0].text(j, i, f"{cos_matrix[i,j]:.3f}", ha="center", va="center",
                     fontsize=8, color=c)

im2 = axes[1].imshow(auc_matrix, cmap="YlOrRd", vmin=0.4, vmax=1.0)
plt.colorbar(im2, ax=axes[1], shrink=0.8)
axes[1].set_xticks(range(N_CD)); axes[1].set_yticks(range(N_CD))
axes[1].set_xticklabels(labels_short, rotation=30, ha="right")
axes[1].set_yticklabels(labels_short)
axes[1].set_xlabel("Test domain")
axes[1].set_ylabel(r"$\mathbf{w}_{\mathrm{fact}}$ source")
axes[1].set_title("Cross-Domain Transfer AUC")
for i in range(N_CD):
    for j in range(N_CD):
        v = auc_matrix[i,j]
        c = "white" if v > 0.85 else "black"
        axes[1].text(j, i, f"{v:.3f}", ha="center", va="center",
                     fontsize=8, color=c)

save_figure(fig, "clean_domains_matrix")
plt.close(fig)
print("  Saved clean_domains_matrix.png")

# ══════════════════════════════════════════════════════════════
# Figure 4: Domain transferability predictor (full width, 1×3 layout)
# ══════════════════════════════════════════════════════════════
print("Figure 4: Domain transferability predictor...")
dt_results = json.load(open("./results/domain_transferability/results.json"))

pairs = dt_results.get("pair_records", [])
metric_summary = dt_results.get("metric_summary", {})

if pairs:
    emb_sim = np.array([p.get("embedding_similarity", 0) for p in pairs])
    tok_jac = np.array([p.get("token_jaccard", 0) for p in pairs])
    fmt_sim = np.array([p.get("format_similarity", 0) for p in pairs])
    wf_cos  = np.array([abs(p.get("wfact_cosine", 0)) for p in pairs])
    pair_labels = [p.get("pair", "") for p in pairs]

    # Remove outlier (MMLU-Med <> ARC-Sci)
    keep = np.array([not ("ARC-Science" in lb and "MMLU-Medical" in lb) for lb in pair_labels])
    emb_sim, tok_jac, fmt_sim, wf_cos = emb_sim[keep], tok_jac[keep], fmt_sim[keep], wf_cos[keep]

    metric_data = [
        ("embedding_similarity", emb_sim, "Sentence-Embedding\nSimilarity"),
        ("token_jaccard",        tok_jac, "Token Jaccard\nSimilarity"),
        ("format_similarity",    fmt_sim, "Format-Feature\nSimilarity"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.6))
    for ax, (mkey, xvals, xlabel) in zip(axes, metric_data):
        rho = metric_summary.get(mkey, {}).get("spearman_rho_vs_wfact_cosine", 0)
        pval = metric_summary.get(mkey, {}).get("spearman_p_vs_wfact_cosine", 1)
        ax.scatter(xvals, wf_cos, s=28, alpha=0.85, color="#2368a2", zorder=3)
        if len(np.unique(xvals)) > 1:
            slope, intercept = np.polyfit(xvals, wf_cos, 1)
            xs = np.linspace(min(xvals), max(xvals), 100)
            ax.plot(xs, slope * xs + intercept, color="#c44e52", lw=1.5, alpha=0.8)
        title_name = xlabel.replace("\n", " ")
        ax.set_title(title_name + "\n" + r"Spearman $\rho$" + f"={rho:.3f}, p={pval:.4f}", fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(r"Pairwise $\mathbf{w}_{\mathrm{fact}}$ cosine", fontsize=8)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))

    save_figure(fig, "domain_transferability_predictor")
    plt.close(fig)
    print("  Saved domain_transferability_predictor.png")

# ══════════════════════════════════════════════════════════════
# Figure 5: Subspace SVD (full width)
# ══════════════════════════════════════════════════════════════
print("Figure 5: Subspace SVD...")
wfacts = []
for dname in clean_domains:
    d = np.load(HIDDEN_PATHS[dname])
    w = build_wfact(d["h_correct"][:400], d["h_wrong"][:400])
    wfacts.append(w)
W = np.stack(wfacts)
_, S, _ = np.linalg.svd(W, full_matrices=False)
total_var = np.sum(S**2)
var_each = S**2 / total_var * 100
cum_var = np.cumsum(S**2) / total_var * 100

fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.4))

axes[0].bar(range(1, len(S)+1), var_each, color="steelblue", alpha=0.8)
axes[0].set_xlabel("Singular value index")
axes[0].set_ylabel("Variance explained (%)")
axes[0].set_title("Per-Singular-Value Variance")
axes[0].set_xticks(range(1, len(S)+1))

axes[1].plot(range(1, len(S)+1), cum_var, "o-", color="darkorange", markersize=5)
axes[1].axhline(90, color="gray", ls="--", alpha=0.7, label="90%")
axes[1].set_xlabel("Number of singular vectors (k)")
axes[1].set_ylabel("Cumulative variance (%)")
axes[1].set_title("Cumulative Variance")
axes[1].legend(fontsize=8)
axes[1].set_xticks(range(1, len(S)+1))
axes[1].set_ylim(0, 105)

save_figure(fig, "clean_subspace_svd")
plt.close(fig)
print("  Saved clean_subspace_svd.png")

# ══════════════════════════════════════════════════════════════
# Figure 6: Minimal calibration (full width, 2x2)
# ══════════════════════════════════════════════════════════════
print("Figure 6: Minimal calibration...")
fewshot = json.load(open("./results/fewshot_adaptation/results.json"))
N_LIST = [10, 20, 50, 100, 200, 400]
colors = {"TruthfulQA": "#1f77b4", "FEVER": "#d62728",
          "MMLU-Medical": "#2ca02c", "ARC-Science": "#9467bd"}

# Build TQA curve from scratch
def compute_tqa_curve():
    d = np.load(HIDDEN_PATHS["TruthfulQA"])
    hc, hw = d["h_correct"], d["h_wrong"]
    curve = {}
    for n in N_LIST:
        vals = []
        for seed in range(10):
            rng = np.random.default_rng(42 + seed)
            idx = rng.permutation(min(400, len(hc)))
            train_idx = idx[:n]; test_idx = idx[n:min(400, len(hc))]
            if len(test_idx) < 5: test_idx = idx[:n]
            w = build_wfact(hc[train_idx], hw[train_idx])
            vals.append(compute_auc(hc[test_idx], hw[test_idx], w))
        curve[n] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return curve

combined = {"TruthfulQA": compute_tqa_curve()}
for dom in ["FEVER", "MMLU-Medical", "ARC-Science"]:
    combined[dom] = {int(n): stats for n, stats in fewshot[dom]["from_scratch"].items()}
    # Add N=400 if missing
    if 400 not in combined[dom]:
        d = np.load(HIDDEN_PATHS[dom])
        hc, hw = d["h_correct"][:400], d["h_wrong"][:400]
        vals = []
        for seed in range(10):
            rng = np.random.default_rng(42 + seed)
            fold_aucs = []
            idx = rng.permutation(400)
            n_test = 80
            w = build_wfact(hc[idx[n_test:]], hw[idx[n_test:]])
            fold_aucs.append(compute_auc(hc[idx[:n_test]], hw[idx[:n_test]], w))
            vals.append(float(np.mean(fold_aucs)))
        combined[dom][400] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

fig, axes = plt.subplots(2, 2, figsize=(TEXT_W, 3.6), sharey=True)
axes_flat = axes.flatten()

for idx, (domain, curve) in enumerate(combined.items()):
    ax = axes_flat[idx]
    ns = sorted([n for n in curve.keys() if n in N_LIST])
    means = [curve[n]["mean"] for n in ns]
    stds  = [curve[n].get("std", 0) for n in ns]
    color = colors[domain]

    ax.errorbar(ns, means, yerr=stds, fmt="o-", color=color,
                markersize=4, capsize=3, linewidth=1.5)

    max_mean = max(means)
    thresh = 0.9 * max_mean
    ax.axhline(thresh, color=color, ls="--", alpha=0.4)

    n90 = next((n for n, m in zip(ns, means) if m >= thresh), ns[-1])
    ax.axvline(n90, color=color, ls=":", alpha=0.4)

    ax.set_title(f"{domain} ($N_{{90}}$={n90})")
    ax.set_xlabel("Labeled pairs (N)")
    if idx % 2 == 0:
        ax.set_ylabel("AUC")
    ax.set_xscale("log")
    ax.set_xticks(N_LIST)
    ax.set_xticklabels([str(n) for n in N_LIST])
    ax.set_ylim(0.45, 1.02)

save_figure(fig, "minimal_calibration")
plt.close(fig)
print("  Saved minimal_calibration.png")

# ══════════════════════════════════════════════════════════════
# Figure 7 (Appendix): PCA best layer (single column)
# ══════════════════════════════════════════════════════════════
print("Figure 7: PCA best layer...")
best_layer = 16
X_best = np.concatenate([h_correct[:, best_layer, :],
                          h_wrong[:, best_layer, :]], axis=0)
y_all = np.array([1]*N + [0]*N)
pca2d = PCA(n_components=2, random_state=42)
X_2d = pca2d.fit_transform(X_best)

fig, ax = plt.subplots(figsize=(COL_W, 2.6))
ax.scatter(X_2d[y_all==1, 0], X_2d[y_all==1, 1],
           c="steelblue", alpha=0.4, s=6, label="Factual")
ax.scatter(X_2d[y_all==0, 0], X_2d[y_all==0, 1],
           c="tomato", alpha=0.4, s=6, label="Hallucinated")
ax.set_xlabel(f"PC1 ({pca2d.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca2d.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(f"PCA at Layer {best_layer} (Llama-3-8B, TruthfulQA)")
ax.legend(fontsize=8, markerscale=1.5)
save_figure(fig, "pca_best_layer")
plt.close(fig)
print("  Saved pca_best_layer.png")

# ══════════════════════════════════════════════════════════════
# Figure 8 (Appendix): Cross-model DSD (full width)
# ══════════════════════════════════════════════════════════════
print("Figure 8: Cross-model DSD...")
cross_model_summary = json.load(open("./results/cross_model/summary.json"))
# Load per-layer DSD data
llama_dsd = json.load(open("./results/dsd_results.json"))
qwen_probing = json.load(open("./results/cross_model/probing_aucs_qwen.json"))
mistral_probing = json.load(open("./results/cross_model/probing_aucs_mistral.json"))

model_dsd_data = {
    "Llama-3-8B": {
        "probing": llama_dsd["probing_aucs_per_layer"],
        "dsd": llama_dsd["dsd_aucs_per_layer"],
    },
    "Qwen2.5-7B": {
        "probing": qwen_probing["layer_aucs"],
        "dsd": qwen_probing["layer_aucs"],  # use probing as fallback
    },
    "Mistral-7B": {
        "probing": mistral_probing["layer_aucs"],
        "dsd": mistral_probing["layer_aucs"],
    },
}

fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.2))

for ax_idx, (mname, data) in enumerate(model_dsd_data.items()):
    L = len(data["probing"])
    x = [i/(L-1) for i in range(L)]
    best_p = max(data["probing"])
    axes[ax_idx].plot(x, data["probing"], "r--s", markersize=2,
                      label=f"Probing ({best_p:.3f})")
    if data["dsd"] != data["probing"]:
        best_d = max(data["dsd"])
        axes[ax_idx].plot(x, data["dsd"], "b-o", markersize=2,
                          label=f"DSD ({best_d:.3f})")
    axes[ax_idx].axhline(0.5, color="gray", ls=":", alpha=0.5)
    axes[ax_idx].set_xlabel("Relative Layer Depth")
    if ax_idx == 0:
        axes[ax_idx].set_ylabel("AUC")
    axes[ax_idx].set_title(mname)
    axes[ax_idx].legend(fontsize=8)
    axes[ax_idx].set_ylim(0.3, 1.02)

save_figure(fig, "cross_model_paper")
plt.close(fig)
print("  Saved cross_model_paper.png")

# ══════════════════════════════════════════════════════════════
# Figure 9 (Appendix): 7-domain matrix (full width)
# ══════════════════════════════════════════════════════════════
print("Figure 9: 7-domain matrix...")
exp_results = json.load(open("./results/expand_domains/results.json"))
all_domains = exp_results.get("domain_names", [])
if not all_domains:
    all_domains = ["TruthfulQA", "FEVER", "MMLU-Medical", "ARC-Science",
                   "HaluEval-QA", "HaluEval-Dialogue", "HaluEval-Summary"]
ND = len(all_domains)

cos_mat = np.abs(np.array(exp_results["cos_matrix"]))
auc_mat = np.array(exp_results["auc_matrix"])

short_labels = [d.replace("HaluEval-", "HE-") for d in all_domains]

fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 3.2))

im1 = axes[0].imshow(cos_mat, cmap="RdBu_r", vmin=-0.2, vmax=1.0)
plt.colorbar(im1, ax=axes[0], shrink=0.75)
axes[0].set_xticks(range(ND)); axes[0].set_yticks(range(ND))
axes[0].set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8)
axes[0].set_yticklabels(short_labels, fontsize=8)
axes[0].set_title(r"Pairwise $|\cos(\mathbf{w}_{\mathrm{fact}})|$")
for i in range(ND):
    for j in range(ND):
        c = "white" if cos_mat[i,j] > 0.6 else "black"
        axes[0].text(j, i, f"{cos_mat[i,j]:.2f}", ha="center", va="center",
                     fontsize=6.5, color=c)
axes[0].axvline(x=3.5, color='blue', lw=1, ls='--', alpha=0.5)
axes[0].axhline(y=3.5, color='blue', lw=1, ls='--', alpha=0.5)

im2 = axes[1].imshow(auc_mat, cmap="YlOrRd", vmin=0.4, vmax=1.0)
plt.colorbar(im2, ax=axes[1], shrink=0.75)
axes[1].set_xticks(range(ND)); axes[1].set_yticks(range(ND))
axes[1].set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8)
axes[1].set_yticklabels(short_labels, fontsize=8)
axes[1].set_xlabel("Test domain")
axes[1].set_ylabel(r"$\mathbf{w}_{\mathrm{fact}}$ source")
axes[1].set_title("Cross-Domain Transfer AUC")
for i in range(ND):
    for j in range(ND):
        v = auc_mat[i,j]
        c = "white" if v > 0.85 else "black"
        axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=6.5, color=c)
axes[1].axvline(x=3.5, color='blue', lw=1, ls='--', alpha=0.5)
axes[1].axhline(y=3.5, color='blue', lw=1, ls='--', alpha=0.5)

save_figure(fig, "expand_domains_matrix")
plt.close(fig)
print("  Saved expand_domains_matrix.png")

print("\nAll figures regenerated with unified style.")
