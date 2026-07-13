"""Extract qualitative output examples comparing baseline vs edited generation."""

import json
import os
import numpy as np

ANSWERS_PATH = "./results/intervention_v2/test_answers.json"
TRUTH_SCORES_PATH = "./results/intervention_v2/test_truth_scores.json"
META_PATH = "./results/meta.json"  # TruthfulQA 元信息
SAVE_DIR = "./results/qualitative_examples"
os.makedirs(SAVE_DIR, exist_ok=True)


def truncate(text, max_words=80):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + " [...]"
    return text


def main():
    print("=" * 60)
    print("Extracting Qualitative Examples (α=0 vs α*=5)")
    print("=" * 60)

    with open(ANSWERS_PATH) as f:
        answers = json.load(f)

    baseline_answers = answers["0.0"]    # α = 0
    edited_answers = answers["5.0"]      # α* = 5
    degen_answers = answers["20.0"]      # α = 20 (degeneration)

    n = len(baseline_answers)
    print(f"Total test samples: {n}")

    # Load truth scores if available
    truth_scores = None
    if os.path.exists(TRUTH_SCORES_PATH):
        with open(TRUTH_SCORES_PATH) as f:
            truth_scores = json.load(f)
        print(f"Truth scores loaded: keys = {list(truth_scores.keys())[:5]}")

    # Load meta for questions
    meta = None
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        print(f"Meta loaded: {len(meta)} entries")

    # ── Identify flipped cases ────────────────────────────────────────────
    # We look for answers where α=0 looks wrong and α=5 looks improved
    examples = []

    for i in range(min(n, len(baseline_answers), len(edited_answers))):
        base = baseline_answers[i]
        edit = edited_answers[i]

        # Simple heuristics to identify interesting cases:
        # - baseline has repetition (degenerate) or clearly wrong content
        # - edited version is more substantive
        base_words = base.split()
        edit_words = edit.split()

        # Check for repetition in baseline
        base_has_repetition = False
        if len(base_words) > 10:
            # Check if any 5-word sequence repeats
            for j in range(len(base_words) - 10):
                chunk = " ".join(base_words[j:j+5])
                rest = " ".join(base_words[j+5:])
                if chunk in rest:
                    base_has_repetition = True
                    break

        # Check if answers are meaningfully different
        different = base.strip()[:100] != edit.strip()[:100]

        if different:
            q_text = ""
            if meta and i + 600 < len(meta):  # test set starts at idx 600
                q_text = meta[i + 600].get("question", "")

            examples.append({
                "index": i,
                "question": q_text,
                "baseline_answer": truncate(base),
                "edited_answer": truncate(edit),
                "baseline_has_repetition": base_has_repetition,
                "baseline_length": len(base_words),
                "edited_length": len(edit_words),
            })

    print(f"Found {len(examples)} cases where α=0 and α=5 differ meaningfully")

    # ── Select best examples ──────────────────────────────────────────────
    # Prioritize: (1) short, clear answers (2) baseline has repetition (3) different content
    good_examples = sorted(
        examples,
        key=lambda x: (
            -int(x["baseline_has_repetition"]),  # prefer repetition cases
            abs(x["baseline_length"] - 40),       # prefer medium-length
        ),
    )[:15]

    # ── Also get degeneration examples (α=20) ─────────────────────────────
    degen_examples = []
    for i in range(min(5, len(degen_answers))):
        degen_examples.append({
            "index": i,
            "question": meta[i + 600].get("question", "") if meta and i + 600 < len(meta) else "",
            "degen_answer": truncate(degen_answers[i], max_words=40),
        })

    # ── Print examples ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SELECTED EXAMPLES (α=0 vs α*=5)")
    print("=" * 60)

    for j, ex in enumerate(good_examples[:8]):
        print(f"\n--- Example {j+1} (test idx {ex['index']}) ---")
        if ex["question"]:
            print(f"  Q: {ex['question']}")
        print(f"  α=0:  {ex['baseline_answer']}")
        print(f"  α=5:  {ex['edited_answer']}")
        print(f"  [repetition in baseline: {ex['baseline_has_repetition']}]")

    print("\n" + "=" * 60)
    print("DEGENERATION EXAMPLES (α=20)")
    print("=" * 60)

    for j, ex in enumerate(degen_examples[:3]):
        print(f"\n--- Degen Example {j+1} ---")
        if ex["question"]:
            print(f"  Q: {ex['question']}")
        print(f"  α=20: {ex['degen_answer']}")

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "n_test": n,
        "n_different": len(examples),
        "selected_examples": good_examples[:15],
        "degeneration_examples": degen_examples,
    }

    save_path = os.path.join(SAVE_DIR, "qualitative_examples.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()