"""Compare a complete fresh 10-model panel against the archived vote CSVs."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


FIRST = [
    "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMo-2-1124-7B-Instruct", "microsoft/Phi-3.5-mini-instruct",
]
SECOND = [
    "aaditya/Llama3-OpenBioLLM-8B", "m42-health/Llama3-Med42-8B",
    "01-ai/Yi-1.5-9B-Chat", "ibm-granite/granite-3.1-8b-instruct",
    "tiiuae/Falcon3-7B-Instruct", "microsoft/Phi-4-mini-instruct",
]
PARTS = [FIRST + SECOND[:1], SECOND[1:]]


def check(ref, new, models, each, kind):
    keys = ["model", "concept", "template", "order"]
    if kind != "choice":
        keys.append("diag")
    values = [c for c in ref if c.startswith("p_")]
    ref = ref.loc[ref.model.isin(models)].copy()
    expected = each * len(models)
    for label, frame in (("archive", ref), ("fresh", new)):
        if len(frame) != expected or set(frame.model) != set(models):
            raise ValueError(f"{label}: expected {expected} rows for {models}, got {len(frame)}")
        if frame.groupby("model").size().to_dict() != dict.fromkeys(models, each):
            raise ValueError(f"{label}: incorrect row count for a model")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label}: duplicate prompt keys")
        if not np.isfinite(frame[values + ["letter_mass"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{label}: non-finite probabilities")
        if not np.allclose(frame[values].sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"{label}: probabilities do not sum to one")
    joined = ref.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    if len(joined) != expected:
        raise ValueError(f"Prompt mismatch: matched {len(joined)} of {expected}")
    difference = np.abs(joined[[c + "_old" for c in values]].to_numpy() -
                        joined[[c + "_new" for c in values]].to_numpy())
    return expected, difference.mean(), difference.max()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, default=Path("fresh_runs/full_panel"))
    args = parser.parse_args()
    checks = [
        ("derm", "llm_votes.csv", FIRST, 540, "votes"),
        ("derm", "llm_votes2.csv", SECOND, 540, "votes"),
    ]
    for dataset, per_comp, per_choice in (("derm", 540, 135), ("lidc", 252, 126)):
        for part, models in enumerate(PARTS):
            checks.extend([
                (dataset, f"llm_comp_{part}.csv", models, per_comp, "comp"),
                (dataset, f"llm_choice_{part}.csv", models, per_choice, "choice"),
            ])
    total = 0
    for dataset, name, models, each, kind in checks:
        original = Path("results" if dataset == "derm" else "results_lidc") / name
        fresh = args.fresh / dataset / name
        count, mean, maximum = check(pd.read_csv(original), pd.read_csv(fresh), models, each, kind)
        total += count
        print(f"{dataset}/{name}: {count} answers matched; mean difference {mean:.4f}; largest {maximum:.4f}")
    print(f"PASS: all {total} answers from the final 10 models are present and aligned.")


if __name__ == "__main__":
    main()
