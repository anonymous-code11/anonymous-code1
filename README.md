# Code

## Setup

```bash
conda create -n factdir python=3.10
conda activate factdir
pip install torch transformers datasets scikit-learn numpy matplotlib seaborn tqdm
```

Models used (update paths in scripts as needed):
- Llama-3-8B
- Qwen2.5-7B-Instruct
- Mistral-7B-v0.2
- Llama-2-13B (scale-up)
- Qwen2.5-14B-Instruct (scale-up)
- truthfulqa-truth-judge-llama2-7B
- truthfulqa-info-judge-llama2-7B

## Repository Structure

### Core Utilities

| File | Description |
|------|-------------|
| `analysis_utils.py` | Shared utilities: `build_wfact`, `load_hidden`, `compute_auc`, dataset loading, domain definitions |

### Main Paper Experiments (§4–§5)

| File | Paper Section | Description |
|------|---------------|-------------|
| `extract_hidden.py` | §4.1 | Extract hidden states from TruthfulQA for Llama-3-8B |
| `expand_domains.py` | §5.1 | Extract hidden states for FEVER, MMLU-Medical, ARC-Science; compute 4-domain cross-domain matrix (Figure 3) |
| `cross_domain_matrix.py` | §5.3 | Compute cross-domain cosine and AUC matrix including HaluEval (Figure 9) |
| `mmlu_controlled.py` | §5.2 Exp 1 | Controlled MMLU experiment: 5 sub-domains, same MCQ format, varying knowledge area |
| `format_control.py` | §5.2 Exp 2 | Format control experiment: ARC-Science in 3 surface formats |
| `orthogonality.py` | §5.3 | Pairwise cosine similarity analysis |
| `subspace_analysis.py` | §5.3 | SVD subspace rank analysis (Table 6, Figure 5) |
| `effective_rank_permutation.py` | §5.3 | Permutation test for effective rank significance |
| `minimal_calibration.py` | §5.4 | Calibration curves with varying N (Figure 6) |
| `intervention_v2.py` | §4.4 | Activation editing with three-way split protocol (Table 3, Figure 2) |
| `strict_validation.py` | §4.4 | Strict validation split for activation editing |
| `dsd_metric.py` | §4.2, App C | Dual-space diagnostic score analysis |
| `domain_transferability_predictor.py` | §5.3 | Domain transferability predictor using embedding/format similarity (Figure 4) |
| `halueval_validation.py` | App G | HaluEval artifact analysis |
| `latency_benchmark.py` | §4.4 | Inference cost benchmark |

### Cross-Model Experiments (§4.3, §5.3)

| File | Description |
|------|-------------|
| `cross_model.py` | Within-domain probing across Llama-3-8B, Qwen2.5-7B, Mistral-7B (Table 2) |
| `cross_model_full_matrix.py` | Full 4×4 cross-domain matrix for Qwen2.5-7B and Mistral-7B |
| `scaleup_mmlu_triplet_13b.py` | Scale-up check on Llama-2-13B and Qwen2.5-14B (Table 5) |
| `scaleup_mmlu_triplet_13b_postprocess.py` | Postprocessing for scale-up results |

### Additional Experiments

| File | Description |
|------|-------------|
| `cross_domain_editing.py` | Cross-domain activation editing: use w_fact from each domain to edit TruthfulQA generation |
| `mixed_domain_probe.py` | Multi-domain mixed probe: train on mixture of 4 domains, compare with per-domain |
| `split_half_reliability.py` | Split-half direction reliability: within-domain stability vs cross-domain orthogonality |
| `surface_feature_baseline.py` | Surface-feature baseline: compare w_fact AUC with surface-only classifier |
| `extract_qualitative_examples.py` | Extract qualitative output examples at α=0 vs α*=5 |
| `fewshot_adaptation.py` | Few-shot domain adaptation experiments |
| `fewshot_equal_sample.py` | Equal-sample few-shot comparison |
| `minimal_calibration_summary.py` | Summary statistics for calibration results |
| `analyze_subspace.py` | Additional subspace analysis |
| `negation_3x3.py` | Negation sensitivity test: 3 models × 3 datasets (CounterFact, FM Queries, PopQA) with bootstrap CI |

### Scripts

| File | Description |
|------|-------------|
| `run.sh` | Run core experiments |
| `run_new_experiments.sh` | Run expanded domain experiments |
| `run_expanded_experiments.sh` | Run full experiment pipeline |
| `regenerate_paper_figures.py` | Regenerate all paper figures from cached results |
| `check_datasets.py` | Verify dataset availability and format |

## Reproducing Results

**Step 1: Extract hidden states**

```bash
python extract_hidden.py          # TruthfulQA (Llama-3-8B)
python expand_domains.py          # FEVER, MMLU-Medical, ARC-Science
```

**Step 2: Main paper analyses**

```bash
python mmlu_controlled.py         # §5.2: controlled MMLU experiment
python format_control.py          # §5.2: format control experiment
python subspace_analysis.py       # §5.3: SVD rank analysis
python minimal_calibration.py     # §5.4: calibration curves
python intervention_v2.py         # §4.4: activation editing
```

**Step 3: Cross-model replication**

```bash
python cross_model_full_matrix.py --model /path/to/Qwen2.5-7B-Instruct --model_name Qwen2.5-7B-Instruct
python cross_model_full_matrix.py --model /path/to/mistral-7B-v0.2 --model_name Mistral-7B-v0.2
```

**Step 4: Additional experiments**

```bash
python mixed_domain_probe.py          # multi-domain mixed probe
python split_half_reliability.py      # split-half reliability
python surface_feature_baseline.py    # surface-feature baseline
python cross_domain_editing.py        # cross-domain activation editing
python negation_3x3.py                # negation sensitivity (3 models x 3 datasets)
```

