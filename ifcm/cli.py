"""Run the published experiment definitions from one reproducible entry point.

The numerical implementations live in analysis/ and kaggle/. This module fixes
their arguments, validates inputs, and saves stdout alongside machine-readable
summaries. It intentionally does not launch Kaggle work implicitly.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DERM = ROOT / "results"
LIDC = ROOT / "results_lidc"
META = ROOT / "meta"

DERM_INPUTS = ["llm_votes.csv", "llm_votes2.csv", "llm_comp_0.csv", "llm_comp_1.csv",
               "llm_choice_0.csv", "llm_choice_1.csv", "emb_monet.npz", "emb_biomedclip.npz"]
LIDC_INPUTS = ["lidc_ratings.csv", "llm_comp_0.csv", "llm_comp_1.csv",
               "llm_choice_0.csv", "llm_choice_1.csv", "lidc_cnn.npz",
               "lidc_emb_biomedclip.npz", "lidc_emb_dinov3.npz"]


def check_paths(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing inputs:\n  " + "\n  ".join(missing))


def run(script: str, args: list[str], log: Path | None = None) -> None:
    command = [sys.executable, str(ROOT / script), *args]
    print("$ " + " ".join(command), flush=True)
    if log is None:
        subprocess.run(command, cwd=ROOT, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            output.write(line)
        if process.wait():
            raise SystemExit(f"Analysis failed; see {log}")
    print(f"Saved {log.relative_to(ROOT)}")


def derm_panel() -> str:
    """Use the ten models present in the final comparative/choice panel."""
    with (DERM / "llm_choice_0.csv").open(newline="") as first, \
         (DERM / "llm_choice_1.csv").open(newline="") as second:
        models = sorted({row["model"] for file in (first, second) for row in csv.DictReader(file)})
    if len(models) != 10:
        raise SystemExit(f"Expected 10 Derm7pt panel models; found {len(models)}")
    return ",".join(models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["check", "derm", "lidc", "figures", "report", "verify", "all"])
    parser.add_argument("--encoder", choices=["monet", "biomedclip", "dinov3", "cnn"],
                        help="Run one encoder for derm or lidc; omit for the published set")
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    if args.task in {"check", "all"}:
        check_paths([META / name for name in ("meta.csv", "train_indexes.csv", "valid_indexes.csv", "test_indexes.csv")]
                    + [DERM / name for name in DERM_INPUTS] + [LIDC / name for name in LIDC_INPUTS])
        print("Analysis inputs found. This check does not validate the medical datasets or Kaggle jobs.")
        if args.task == "check":
            return

    if args.task in {"derm", "all"}:
        encoders = [args.encoder] if args.encoder else ["monet", "biomedclip"]
        if any(enc not in {"monet", "biomedclip"} for enc in encoders):
            parser.error("Derm encoders: monet, biomedclip")
        for enc in encoders:
            check_paths([DERM / f"emb_{enc}.npz", META / "meta.csv"]
                        + [DERM / name for name in DERM_INPUTS if name.endswith(".csv")])
            members = derm_panel()
            styles = ["orig", "comp", "choice", "orig+comp+choice"] if enc == "monet" else ["orig+comp+choice"]
            for panel in styles:
                run("analysis/pilot.py", ["results", "meta", enc, "0", members, panel],
                    DERM / f"log_{enc}_{panel}.txt")
            panel = "orig+comp+choice"
            run("analysis/why_test.py", [members, panel, enc], DERM / f"why_{enc}_{panel}.txt")
            if enc == "monet":
                run("analysis/review_sim.py", [members, panel], DERM / "review_sim.txt")

    if args.task in {"lidc", "all"}:
        encoders = [args.encoder] if args.encoder else ["biomedclip", "dinov3", "cnn"]
        if any(enc not in {"biomedclip", "dinov3", "cnn"} for enc in encoders):
            parser.error("LIDC encoders: biomedclip, dinov3, cnn")
        for enc in encoders:
            input_name = "lidc_cnn.npz" if enc == "cnn" else f"lidc_emb_{enc}.npz"
            check_paths([LIDC / input_name] + [LIDC / name for name in LIDC_INPUTS if name.endswith(".csv")])
            run("analysis/lidc.py", ["results_lidc", enc, "results_lidc/lidc_cnn.npz", str(args.split_seed)],
                LIDC / f"log_{enc}.txt")

    if args.task in {"figures", "all"}:
        check_paths([DERM / "report_monet_orig+comp+choice.json", DERM / "edges_mid.csv",
                     DERM / "edges_hesitancy.csv", DERM / "why_monet_orig+comp+choice.txt",
                     DERM / "why_biomedclip_orig+comp+choice.txt", LIDC / "log_cnn.txt",
                     LIDC / "log_biomedclip.txt", LIDC / "log_dinov3.txt",
                     LIDC / "lidc_crops_3.npz", LIDC / "lidc_ratings.csv"])
        run("analysis/figures.py", [])
        run("ifcm/reporting.py", [])

    if args.task in {"report", "all"}:
        if args.task == "report":
            run("ifcm/reporting.py", [])
        if shutil.which("latexmk") is None:
            raise SystemExit("latexmk is required to build report/report.pdf")
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                       cwd=ROOT / "report", check=True)
        print("Built report/report.pdf")

    if args.task in {"verify", "all"}:
        run("ifcm/verify.py", [])


if __name__ == "__main__":
    main()
