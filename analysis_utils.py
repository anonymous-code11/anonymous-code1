import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from datasets import Dataset
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


BEST_LAYER = 16
PCA_DIM = 128
RANDOM_SEED = 42

DOMAIN_ORDER = [
    "TruthfulQA",
    "FEVER",
    "MMLU-Medical",
    "ARC-Science",
    "HaluEval-QA",
    "HaluEval-Dialogue",
    "HaluEval-Summary",
]

FOUR_DOMAIN_ORDER = [
    "TruthfulQA",
    "FEVER",
    "MMLU-Medical",
    "ARC-Science",
]

MMLU_MEDICAL_SUBJECTS = {
    "anatomy",
    "clinical_knowledge",
    "medical_genetics",
    "college_medicine",
    "professional_medicine",
    "college_biology",
    "high_school_biology",
    "nutrition",
}

ARROW_PATHS = {
    "TruthfulQA": (
        "/home/pzh/.cache/huggingface/datasets/truthful_qa/generation/0.0.0/"
        "741b8276f2d1982aa3d5b832d3ee81ed3b896490/truthful_qa-validation.arrow"
    ),
    "FEVER": (
        "/home/pzh/.cache/huggingface/datasets/pietrolesci___nli_fever/default/0.0.0/"
        "1eddac63112eee1fdf1966e0bca27a5ff248c772/nli_fever-train.arrow"
    ),
    "MMLU-Medical": (
        "/home/pzh/.cache/huggingface/datasets/cais___mmlu/all/0.0.0/"
        "c30699e8356da336a370243923dbaf21066bb9fe/mmlu-test.arrow"
    ),
    "ARC-Science": (
        "/home/pzh/.cache/huggingface/datasets/ai2_arc/ARC-Challenge/0.0.0/"
        "210d026faf9955653af8916fad021475a3f00453/ai2_arc-test.arrow"
    ),
    "HaluEval-QA": (
        "/home/pzh/.cache/huggingface/datasets/pminervini___halu_eval/qa/0.0.0/"
        "12a856119f03975a94509091e8cada3e6be6ead7/halu_eval-data.arrow"
    ),
    "HaluEval-Dialogue": (
        "/home/pzh/.cache/huggingface/datasets/pminervini___halu_eval/dialogue/0.0.0/"
        "12a856119f03975a94509091e8cada3e6be6ead7/halu_eval-data.arrow"
    ),
    "HaluEval-Summary": (
        "/home/pzh/.cache/huggingface/datasets/pminervini___halu_eval/summarization/0.0.0/"
        "12a856119f03975a94509091e8cada3e6be6ead7/halu_eval-data.arrow"
    ),
}

HIDDEN_PATHS = {
    "TruthfulQA": "./results/hidden_states.npz",
    "FEVER": "./results/expand_domains/FEVER_hidden.npz",
    "MMLU-Medical": "./results/expand_domains/MMLU-Medical_hidden.npz",
    "ARC-Science": "./results/expand_domains/ARC-Science_hidden.npz",
    "HaluEval-QA": "./results/halueval_v2/halueval_hidden_5000.npz",
    "HaluEval-Dialogue": "./results/cross_domain/HaluEval-Dialogue_hidden.npz",
    "HaluEval-Summary": "./results/cross_domain/HaluEval-Summary_hidden.npz",
}

FORMAT_FEATURES = {
    "TruthfulQA": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "FEVER": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "MMLU-Medical": np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "ARC-Science": np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "HaluEval-QA": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    "HaluEval-Dialogue": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
    "HaluEval-Summary": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_arrow_dataset(domain_name: str) -> Dataset:
    return Dataset.from_file(ARROW_PATHS[domain_name])


def load_hidden(domain_name: str, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(HIDDEN_PATHS[domain_name])
    h_correct = data["h_correct"]
    h_wrong = data["h_wrong"]
    if limit is not None:
        h_correct = h_correct[:limit]
        h_wrong = h_wrong[:limit]
    return h_correct, h_wrong


def format_mcq_prompt(question: str, choices: Sequence[str]) -> str:
    labels = ["A", "B", "C", "D", "E", "F"]
    rendered = []
    for idx, choice in enumerate(choices):
        label = labels[idx] if idx < len(labels) else str(idx)
        rendered.append(f"({label}) {choice}")
    return f"Q: {question}\nChoices: {' | '.join(rendered)}"


def load_prompt_texts(domain_name: str, limit: int = 400) -> List[str]:
    ds = load_arrow_dataset(domain_name)
    texts: List[str] = []

    if domain_name == "TruthfulQA":
        for row in ds:
            if row["best_answer"] and row["incorrect_answers"]:
                texts.append(f"Q: {row['question']}")
                if len(texts) >= limit:
                    break
        return texts

    if domain_name == "FEVER":
        for row in ds:
            if row["label"] in {0, 1} and row["hypothesis"].strip():
                texts.append(f"Claim: {row['hypothesis']}")
                if len(texts) >= limit:
                    break
        return texts

    if domain_name == "MMLU-Medical":
        for row in ds:
            if row["subject"] in MMLU_MEDICAL_SUBJECTS:
                texts.append(format_mcq_prompt(row["question"], row["choices"]))
                if len(texts) >= limit:
                    break
        return texts

    if domain_name == "ARC-Science":
        for row in ds:
            texts.append(format_mcq_prompt(row["question"], row["choices"]["text"]))
            if len(texts) >= limit:
                break
        return texts

    if domain_name == "HaluEval-QA":
        for row in ds:
            if row["question"] and row["right_answer"]:
                texts.append(f"Q: {row['question']}")
                if len(texts) >= limit:
                    break
        return texts

    if domain_name == "HaluEval-Dialogue":
        for row in ds:
            if row["dialogue_history"] and row["right_response"]:
                texts.append(f"Dialogue: {row['dialogue_history']}")
                if len(texts) >= limit:
                    break
        return texts

    if domain_name == "HaluEval-Summary":
        for row in ds:
            if row["document"] and row["right_summary"]:
                texts.append(f"Article: {row['document'][:400]}")
                if len(texts) >= limit:
                    break
        return texts

    raise KeyError(f"Unknown domain: {domain_name}")


def build_wfact(
    h_correct: np.ndarray,
    h_wrong: np.ndarray,
    layer: int = BEST_LAYER,
    pca_dim: int = PCA_DIM,
    labels: Optional[np.ndarray] = None,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    n = min(h_correct.shape[0], h_wrong.shape[0])
    x = np.concatenate([h_correct[:n, layer, :], h_wrong[:n, layer, :]], axis=0)
    y = np.array([1] * n + [0] * n) if labels is None else np.asarray(labels)
    dim = min(pca_dim, x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=dim, random_state=random_state)
    x_proj = pca.fit_transform(x)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
    clf.fit(x_proj, y)
    wfact = pca.components_.T @ clf.coef_[0]
    return normalize_vector(wfact)


def precompute_pca_inputs(
    h_correct: np.ndarray,
    h_wrong: np.ndarray,
    layer: int = BEST_LAYER,
    pca_dim: int = PCA_DIM,
    random_state: int = RANDOM_SEED,
) -> Dict[str, np.ndarray]:
    n = min(h_correct.shape[0], h_wrong.shape[0])
    x = np.concatenate([h_correct[:n, layer, :], h_wrong[:n, layer, :]], axis=0)
    y = np.array([1] * n + [0] * n)
    dim = min(pca_dim, x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=dim, random_state=random_state)
    x_proj = pca.fit_transform(x)
    return {
        "x_proj": x_proj,
        "components": pca.components_,
        "labels": y,
    }


def build_wfact_from_precomputed(
    x_proj: np.ndarray,
    components: np.ndarray,
    labels: np.ndarray,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
    clf.fit(x_proj, labels)
    wfact = components.T @ clf.coef_[0]
    return normalize_vector(wfact)


def compute_auc(
    h_correct: np.ndarray,
    h_wrong: np.ndarray,
    wfact: np.ndarray,
    layer: int = BEST_LAYER,
) -> float:
    n = min(h_correct.shape[0], h_wrong.shape[0])
    y = np.array([1] * n + [0] * n)
    scores = np.concatenate(
        [h_correct[:n, layer, :] @ wfact, h_wrong[:n, layer, :] @ wfact]
    )
    return float(roc_auc_score(y, scores))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-8)


def effective_rank(matrix: np.ndarray) -> float:
    _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    probs = singular_values ** 2
    probs = probs / probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(np.exp(entropy))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
    return float(np.dot(vec_a, vec_b) / denom)


def normalize_tokens(text: str) -> List[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 2 and token not in ENGLISH_STOP_WORDS
    ]


def vocab_jaccard(texts_a: Iterable[str], texts_b: Iterable[str]) -> float:
    vocab_a = {token for text in texts_a for token in normalize_tokens(text)}
    vocab_b = {token for text in texts_b for token in normalize_tokens(text)}
    if not vocab_a and not vocab_b:
        return 0.0
    return float(len(vocab_a & vocab_b) / len(vocab_a | vocab_b))


def pair_key(name_a: str, name_b: str) -> str:
    return " <> ".join(sorted((name_a, name_b)))
