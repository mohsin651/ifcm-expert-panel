"""Create paper-ready artifacts from verified saved evaluation tables only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ifcm.b1 import _json_new, _json_read, run_dir
from ifcm.safe_cnn import sha


REQUIRED = ("metrics.csv", "paired_comparisons.csv", "attribution.csv",
            "attribution_paired.csv", "calibration.csv", "reader_correlations.csv",
            "ifs_diagnostics.csv", "edge_quality.csv", "finding_auroc.csv")
PAPER_FOLDER = "paper_v4"


def _inputs(run_id: str, dataset: str):
    parent = run_dir(run_id).parent
    eval_dir = parent / "evaluation_v2"
    manifest = _json_read(eval_dir / "evaluation_manifest.json")
    if manifest["dataset"] != dataset:
        raise ValueError("Dataset does not match evaluation manifest")
    for name in REQUIRED:
        if sha(eval_dir / name) != manifest["output_hashes"][name]:
            raise ValueError(f"Saved evaluation changed: {name}")
    tables = {}
    for name in REQUIRED:
        try:
            tables[name] = pd.read_csv(eval_dir / name)
        except pd.errors.EmptyDataError:
            if name != "reader_correlations.csv" or dataset != "derm":
                raise
            tables[name] = pd.DataFrame(columns=["measure", "n", "n_patients", "spearman", "interpretation"])
    supplement = parent / "supplementary_v1"
    proof = _json_read(supplement / "supplementary_manifest.json")
    if proof["source_evaluation_sha256"] != sha(eval_dir / "evaluation_manifest.json"):
        raise ValueError("Supplement does not match evaluation")
    for name, digest in proof["outputs"].items():
        if sha(supplement / name) != digest:
            raise ValueError(f"Supplement changed: {name}")
        tables[name] = pd.read_csv(supplement / name)
    tables["dataset_splits.csv"] = pd.DataFrame([{
        "dataset": dataset, "protocol": manifest["protocol"],
        "fit_calibration_rows": int(tables["ablations.csv"].n_fit_calibration.iloc[0]),
        "all_test_rows": manifest["n_all_test"], "diagnosis_auroc_rows": manifest["n_diagnosis"],
        "four_class_accuracy_rows": manifest["n_four_class"],
        "bootstrap_unit": "patient" if dataset == "lidc" else "case",
    }])
    return parent, eval_dir, manifest, tables


def _save(fig, directory: Path, stem: str):
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", bbox_inches="tight", dpi=250)
    plt.close(fig)


def _latex_table(frame: pd.DataFrame) -> str:
    """Minimal tabular export without Pandas' optional Jinja dependency."""
    def cell(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            value = f"{value:.4f}"
        result = str(value)
        for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"),
                         ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                         ("$", r"\$")):
            result = result.replace(old, new)
        return result
    lines = [r"\begin{tabular}{" + "l" * len(frame.columns) + "}", r"\hline",
             " & ".join(map(cell, frame.columns)) + " \\\\", r"\hline"]
    lines.extend(" & ".join(cell(value) for value in row) + r" \\"
                 for row in frame.itertuples(index=False, name=None))
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _architecture(directory):
    fig, ax = plt.subplots(figsize=(13, 4.7))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.2, 3.4, 2.1, 0.9, "CT / derm image\nno LLM access", "#D9EBF5"),
        (2.8, 3.4, 2.1, 0.9, "CNN / image encoder\nfinding probabilities", "#D9EBF5"),
        (0.2, 0.6, 2.1, 0.9, "General clinical\nquestions", "#EDE3F6"),
        (2.8, 0.6, 2.1, 0.9, "10 LLMs\nsigned edges", "#EDE3F6"),
        (5.5, 2.0, 1.8, 0.9, "Concept graph\nfinding → dx", "#ECECEC"),
        (8.0, 3.6, 2.6, 0.8, "A: weighted sum", "#DBEDDA"),
        (8.0, 2.1, 2.6, 0.8, "B1: finding IFS", "#FFF0CC"),
        (8.0, 0.6, 2.6, 0.8, "B2: finding + edge IFS", "#F9DDDD"),
        (11.0, 2.0, 2.2, 0.9, "Sigmoid calibration\nDiagnosis prediction", "#E8E8E8"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    arrows = [((2.3, 3.85), (2.8, 3.85)), ((2.3, 1.05), (2.8, 1.05)),
              ((4.9, 3.85), (5.55, 2.8)), ((4.9, 1.05), (5.55, 2.1)),
              ((7.3, 2.5), (8.0, 4.0)), ((7.3, 2.5), (8.0, 2.5)),
              ((7.3, 2.5), (8.0, 1.0)),
              ((10.6, 4.0), (11.0, 2.7)), ((10.6, 2.5), (11.0, 2.5)),
              ((10.6, 1.0), (11.0, 2.2))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.6})
    ax.text(6.7, 0.2, "B1/B2 are experimental reasoning variants; no improvement is assumed.",
            ha="center", fontsize=9, style="italic")
    _save(fig, directory, "figure1_architecture")


def _diagnosis(directory, metrics, paired, dataset):
    names = ["a", "b1", "b2"]
    row = metrics.set_index("model").loc[names]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(names, row.auroc, color=["#597DA8", "#D99A32", "#B85D5D"])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Held-out diagnostic AUROC")
    ax.set_title(f"{dataset.upper()}: matched A / B1 / B2 predictions")
    for bar, value in zip(bars, row.auroc):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=11)
    n = int(row.n_auc.iloc[0])
    a_b1 = paired[(paired.left == "a") & (paired.right == "b1")].iloc[0]
    a_b2 = paired[(paired.left == "a") & (paired.right == "b2")].iloc[0]
    fig.subplots_adjust(bottom=0.21)
    fig.text(0.13, 0.03,
            f"n={n}; A–B1 {a_b1.difference:+.3f} [{a_b1.ci_low:+.3f}, {a_b1.ci_high:+.3f}]\n"
            f"A–B2 {a_b2.difference:+.3f} [{a_b2.ci_low:+.3f}, {a_b2.ci_high:+.3f}]",
            fontsize=9)
    _save(fig, directory, "figure2_diagnosis_auroc")


def _attribution(directory, attribution, paired, metrics, dataset):
    selected = attribution[(attribution.reference == "fixed_a_errors") &
                           (attribution.score.isin(["rank_perc_bern_minus_know_flip",
                                                     "rank_perc_flip_minus_know_flip",
                                                     "one_minus_confidence", "diagnosis_pi"]))]
    selected = selected.loc[~selected.duplicated(["model", "score"])]
    labels = [f"{r.model.upper()}\n" + ("flip–know" if r.score.startswith("rank_") else
              "1-confidence" if r.score == "one_minus_confidence" else "IFS π")
              for r in selected.itertuples()]
    fig, ax = plt.subplots(figsize=(max(7.5, len(labels) * 1.3), 4.6))
    bars = ax.bar(np.arange(len(selected)), selected.auroc, color="#628B8C")
    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(selected)), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Attribution AUROC on fixed A errors")
    n_wrong = int(metrics.loc[metrics.model == "a", "n_errors"].iloc[0])
    n_perc = int(metrics.loc[metrics.model == "a", "n_perception_errors"].iloc[0])
    ax.set_title(f"{dataset.upper()}: {n_wrong} A errors; {n_perc} operational perception errors")
    for bar, value in zip(bars, selected.auroc):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    row = paired[(paired.left == "a_flip_minus_know") &
                 (paired.right == "a_one_minus_confidence")].iloc[0]
    fig.subplots_adjust(bottom=0.23)
    fig.text(0.13, 0.025,
            f"A flip score – 1-confidence: {row.difference:+.3f} "
            f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}]",
            fontsize=8)
    _save(fig, directory, "figure3_attribution")


def _edges(directory, table, dataset):
    data = table.reindex(table.mean_edge.abs().sort_values(ascending=False).index).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.7, 5.3))
    y = np.arange(len(data))
    ax.barh(y, data.mean_edge, color=np.where(data.mean_edge >= 0, "#4E83A5", "#BD786A"))
    ax.errorbar(data.mean_edge, y, xerr=data.raw_uncertainty,
                fmt="none", ecolor="#333333", capsize=2, linewidth=1)
    ax.set_yticks(y, [f"{c} → {d}" for c, d in zip(data.concept, data.diagnosis)])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("Mean signed edge ± heuristic raw uncertainty")
    ax.set_title(f"{dataset.upper()}: illustrative high-magnitude LLM edges")
    ax.text(0.02, 0.02, "Uncertainty is not a confidence interval; edges need clinical review.",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    _save(fig, directory, "figure4_edges")


def _summaries(directory, manifest, tables):
    m = tables["metrics.csv"].set_index("model")
    pairs = tables["paired_comparisons.csv"]
    attr = tables["attribution_paired.csv"]
    point = tables["ablations.csv"].iloc[0]
    dataset = manifest["dataset"]
    a_b1 = pairs[(pairs.left == "a") & (pairs.right == "b1")].iloc[0]
    a_b2 = pairs[(pairs.left == "a") & (pairs.right == "b2")].iloc[0]
    text = (f"# Results summary — {dataset}\n\n"
            f"Protocol: `{manifest['protocol']}`. New results, distinct from the historical report. "
            f"Diagnosis AUROC uses {manifest['n_diagnosis']} held-out cases; "
            f"Derm four-class accuracy uses {manifest['n_four_class']} cases.\n\n"
            f"A/B1/B2 diagnosis AUROC: {m.loc['a','auroc']:.3f} / "
            f"{m.loc['b1','auroc']:.3f} / {m.loc['b2','auroc']:.3f}. "
            f"A–B1 paired difference {a_b1.difference:+.3f} "
            f"[{a_b1.ci_low:+.3f}, {a_b1.ci_high:+.3f}]; "
            f"A–B2 {a_b2.difference:+.3f} "
            f"[{a_b2.ci_low:+.3f}, {a_b2.ci_high:+.3f}]. "
            f"Intervals come from {int(a_b1.n_requested):,} "
            f"{'patient-clustered' if dataset == 'lidc' else 'case-level'} resamples.\n\n"
            f"A had {int(m.loc['a','n_errors'])} errors, including "
            f"{int(m.loc['a','n_perception_errors'])} operational perception errors. "
            f"Attribution scores and paired intervals are in the saved CSV tables; "
            f"they are exploratory and not clinical error explanations.\n\n"
            f"Prespecified B1 zero-finding-hesitancy ablation AUROC: {point.auroc:.3f}; "
            "the B2 zero-edge-width limit exactly reproduces B1. "
            "See the ablation and edge-quality tables for denominators.\n\n"
            "A paired interval crossing zero does not establish a difference. "
            "The three variants were not selected by these held-out scores.\n")
    (directory / "results_summary.md").write_text(text, encoding="utf-8")
    (directory / "methods_summary.md").write_text(
        "# Methods summary\n\nThe image pathway predicts clinical findings from CT or dermoscopic images. "
        "Ten LLMs answer general finding-to-diagnosis questions without patient images. "
        "Model A uses one weighted sum followed by independent diagnosis sigmoids. "
        "Experimental B1 maps finding-draw spread to intuitionistic triples and applies a "
        "single-step signed support/opposition operator; B2 additionally uses heuristic "
        "LLM-member edge intervals. Neither is an iterative FCM. All three use the same "
        "cases, finding means, full-precision panel mean and calibration capacity. "
        "LIDC uses patient-grouped strict CNN fitting and patient-clustered bootstrap; "
        "Derm uses saved official index splits and case-level bootstrap.\n", encoding="utf-8")
    (directory / "limitations.md").write_text(
        "# Limitations\n\nThe signed IFS projection is a proposed operator, not a validated clinical rule. "
        "Five CNN seeds and 30 Derm probes do not provide calibrated uncertainty intervals. "
        "B2 edge widths combine unsure mass and member disagreement heuristically. "
        "Operational perception/knowledge errors use an oracle-finding counterfactual, not "
        "clinician-annotated causal explanations. Historical pilot results were known before "
        "this method design, limiting claims of prospective pre-registration. Derm embeddings "
        "have positional alignment only and no patient IDs, so its case bootstrap may understate "
        "dependence. Reader agreement is descriptive and does not establish clinical validity. "
        "Small perception-error counts make attribution intervals wide.\n", encoding="utf-8")
    (directory / "reviewer_risks.md").write_text(
        "# Reviewer risks\n\n1. Do not call the reasoning iterative: it is a single-step graph.\n"
        "2. Do not conflate formal IFS hesitation with decision flips or calibrated confidence.\n"
        "3. Separate new strict LIDC results from the older exploratory CNN pilot and expanded-cohort fold means.\n"
        "4. State the A parity scope: equation/input alignment is checked, but strict CNN "
        "predictions are new, so historical report AUROC need not match.\n"
        "5. Give both denominator and paired CI for every claimed improvement; report negative results.\n"
        "6. Manual clinical review of example edges remains necessary.\n", encoding="utf-8")


def build(args):
    parent, eval_dir, manifest, tables = _inputs(args.run_id, args.dataset)
    target = parent / PAPER_FOLDER
    if target.exists():
        raise FileExistsError(f"Paper output already exists: {target}")
    figures, table_dir = target / "figures", target / "tables"
    figures.mkdir(parents=True, exist_ok=False)
    table_dir.mkdir()
    for name, frame in tables.items():
        frame.to_csv(table_dir / name, index=False)
        (table_dir / name.replace(".csv", ".tex")).write_text(
            _latex_table(frame), encoding="utf-8")
    _architecture(figures)
    _diagnosis(figures, tables["metrics.csv"], tables["paired_comparisons.csv"], args.dataset)
    _attribution(figures, tables["attribution.csv"], tables["attribution_paired.csv"],
                 tables["metrics.csv"], args.dataset)
    _edges(figures, tables["edge_quality.csv"], args.dataset)
    _summaries(target, manifest, tables)
    all_files = sorted(path for path in target.rglob("*") if path.is_file())
    _json_new(target / "paper_manifest.json", {
        "dataset": args.dataset, "source_evaluation": str(eval_dir),
        "source_manifest_sha256": sha(eval_dir / "evaluation_manifest.json"),
        "supplement_manifest_sha256": sha(parent / "supplementary_v1/supplementary_manifest.json"),
        "outputs": {str(path.relative_to(target)): sha(path) for path in all_files},
        "note": "Figures/tables generated only from saved verified evaluation results."})
    print(f"Paper outputs saved: {target}; files={len(all_files)}")


def verify(args):
    parent, eval_dir, manifest, tables = _inputs(args.run_id, args.dataset)
    target = parent / PAPER_FOLDER
    proof = _json_read(target / "paper_manifest.json")
    if proof["source_manifest_sha256"] != sha(eval_dir / "evaluation_manifest.json"):
        raise ValueError("Paper source changed")
    if proof["supplement_manifest_sha256"] != sha(parent / "supplementary_v1/supplementary_manifest.json"):
        raise ValueError("Paper supplement changed")
    for relative, digest in proof["outputs"].items():
        if sha(target / relative) != digest:
            raise ValueError(f"Paper output changed: {relative}")
    print(json.dumps({"status": "PASS", "dataset": args.dataset,
                      "files": len(proof["outputs"]), "source": str(eval_dir)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("lidc", "derm"))
    parser.add_argument("--run-id", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "build", "verify"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    try:
        if args.dry_run:
            parent, source, manifest, tables = _inputs(args.run_id, args.dataset)
            print(json.dumps({"dataset": args.dataset, "source": str(source),
                              "output": str(parent / PAPER_FOLDER),
                              "n_diagnosis": manifest["n_diagnosis"], "tables": list(tables)}, indent=2))
        elif args.build:
            build(args)
        else:
            verify(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
