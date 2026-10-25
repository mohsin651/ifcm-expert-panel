"""Check freshly generated LIDC LLM votes against the archived panel votes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare(reference: Path, fresh: Path, model: str) -> None:
    for style, expected, values in (
        ("comp", 14 * 2 * 3 * 3, ["p_up", "p_down", "p_none", "p_unsure"]),
        ("choice", 14 * 3 * 3, ["p_MAL", "p_BEN", "p_unsure"]),
    ):
        keys = ["model", "concept", "template", "order"]
        if style == "comp":
            keys.append("diag")
        new = pd.read_csv(fresh / f"llm_{style}_0.csv")
        old = pd.read_csv(reference / f"llm_{style}_0.csv")
        old = old.loc[old.model == model]
        if len(new) != expected or set(new.model) != {model} or new.duplicated(keys).any():
            raise ValueError(f"{style}: fresh output lacks the expected {expected} unique prompts")
        if len(old) != expected:
            raise ValueError(f"{style}: reference lacks the expected {expected} prompts")
        if not np.isfinite(new[values + ["letter_mass"]].to_numpy()).all():
            raise ValueError(f"{style}: non-finite vote or letter mass")
        if not np.allclose(new[values].sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"{style}: probabilities do not sum to one")
        joined = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
        if len(joined) != expected:
            raise ValueError(f"{style}: fresh and reference prompts do not align")
        original = joined[[v + "_old" for v in values]].to_numpy()
        reproduced = joined[[v + "_new" for v in values]].to_numpy()
        print(f"{style}: {expected} prompts; mean absolute probability difference "
              f"{np.abs(original - reproduced).mean():.4f}; "
              f"fresh mean letter mass {new.letter_mass.mean():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("fresh", type=Path)
    parser.add_argument("model")
    args = parser.parse_args()
    compare(args.reference, args.fresh, args.model)


if __name__ == "__main__":
    main()
