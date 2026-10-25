"""Summarize the fold metrics actually printed by the supplied CNN Kaggle jobs.

The archive has scripts and logs, but not per-nodule predictions. Therefore this
cannot calculate pooled AUROC, diagnosis AUROC, or hesitancy correlations.
"""

from __future__ import annotations

import argparse
import re
import statistics
import zipfile
from pathlib import Path


FOLD_AUC = re.compile(r"^fold ([0-4]): mean finding AUROC ([0-9]+\.[0-9]+)", re.M)


def summarize(archive: Path) -> list[tuple[str, int, float]]:
    rows = []
    with zipfile.ZipFile(archive) as bundle:
        for profile in ("lidc-cnn", "lidc-cnn2"):
            for part in range(3):
                path = f"cnn_notebooks/ifcm-{profile}-full-{part}/log.txt"
                log = bundle.read(path).decode("utf-8")
                found = FOLD_AUC.findall(log)
                if [int(fold) for fold, _ in found] != list(range(5)) or "saved; total" not in log:
                    raise ValueError(f"Incomplete CNN log: {path}")
                rows.append((profile, part, statistics.mean(float(auc) for _, auc in found)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="cnn_notebooks.zip supplied by the collaborator")
    args = parser.parse_args()
    for profile, part, mean in summarize(args.archive):
        print(f"{profile} fold-seed {part}: mean of five printed fold AUROCs = {mean:.4f}")


if __name__ == "__main__":
    main()
