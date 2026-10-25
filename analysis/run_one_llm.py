"""Rerun one final-panel LLM on both datasets and compare its votes to the archive."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import pandas as pd

from compare_full_llm_panel import FIRST, SECOND, PARTS, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=FIRST + SECOND)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    manifest = json.loads((root / "fresh_runs/models_manifest.json").read_text())
    snapshot = Path(manifest[args.model]["snapshot"])
    if not (snapshot / "config.json").is_file():
        parser.error(f"Local model files missing: {snapshot}")
    cache = Path("/var/tmp/ifcm-llm-cache-mimam")
    env = dict(os.environ, HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
               IFCM_MODEL_MANIFEST=str(root / "fresh_runs/models_manifest.json"),
               IFCM_MODELS=args.model)
    part = 0 if args.model in PARTS[0] else 1
    vote_job = "llm-panel" if args.model in FIRST else "llm-panel2"
    folder = root / "fresh_runs" / "one_model" / args.model.replace("/", "--")
    for job, dataset, job_part in ((vote_job, "derm", None),
                                   ("llm-panel3", "derm", part),
                                   ("llm-lidc", "lidc", part)):
        out = folder / dataset
        out.mkdir(parents=True, exist_ok=True)
        command = ["./kaggle/push.sh", job, "full"]
        if job_part is not None:
            command.append(str(job_part))
        print(f"Building {job} for {args.model}", flush=True)
        subprocess.run(command, env=dict(env, IFCM_BUILD_ONLY="1"), check=True)
        built = root / "kaggle/build" / (f"ifcm-{job}-full" +
                                         (f"-{job_part}" if job_part is not None else "")) / "run.py"
        print(f"Running {job} for {args.model}", flush=True)
        subprocess.run([os.sys.executable, str(built)],
                       env=dict(env, IFCM_OUT=str(out)), check=True)
        names = (["llm_votes.csv" if job == "llm-panel" else "llm_votes2.csv"]
                 if job_part is None else
                 [f"llm_comp_{part}.csv", f"llm_choice_{part}.csv"])
        for name in names:
            original = root / ("results" if dataset == "derm" else "results_lidc") / name
            kind = "choice" if name.startswith("llm_choice") else "comp" if name.startswith("llm_comp") else "votes"
            each = {"votes": 540, "comp": 540 if dataset == "derm" else 252,
                    "choice": 135 if dataset == "derm" else 126}[kind]
            count, mean, maximum = check(pd.read_csv(original), pd.read_csv(out / name),
                                         [args.model], each, kind)
            print(f"{dataset}/{name}: {count} answers; mean difference {mean:.4f}; "
                  f"largest {maximum:.4f}", flush=True)
    print(f"PASS: all archived question rows matched for {args.model}; "
          "review the differences above before deciding whether to use the archive.")


if __name__ == "__main__":
    main()
