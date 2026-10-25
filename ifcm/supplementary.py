"""Prespecified ablation and descriptive supplement; no test-set tuning."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score

from ifcm.b1 import _json_new, apply_calibration, fit_calibration, run_dir
from ifcm.b2 import _labels
from ifcm.evaluate_comparison import _read
from ifcm.intuitionistic_reasoning import point_ifs, reason_b1
from ifcm.safe_cnn import sha


def _base(args):
    parent, base, arrays, edge, ids, patients, eligible, y, findings, fractions, n_readers = _read(args)
    source = parent / "evaluation_v2"
    with (source / "evaluation_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for name, digest in manifest["output_hashes"].items():
        if sha(source / name) != digest:
            raise ValueError(f"Evaluation source changed: {name}")
    return parent, base, arrays, edge, ids, patients, eligible, y, findings, fractions, n_readers


def build(args):
    parent, base, arrays, edge, ids, patients, eligible, y, findings, fractions, n_readers = _base(args)
    target = parent / "supplementary_v1"
    if target.exists():
        raise FileExistsError(f"Supplement already exists: {target}")
    with np.load(parent / "b1/fit_predictions.npz", allow_pickle=False) as z:
        fit_ids, fit_mean = z["ids"].astype(str), z["finding_mean"]
    mask, labels = _labels(args.dataset, args.encoder, fit_ids, base["cnn_run_id"])
    point_fit = reason_b1(point_ifs(fit_mean), edge).score
    cal = fit_calibration(point_fit[mask], labels)
    point_test = reason_b1(point_ifs(arrays["b1"]["finding_mean"]), edge).score
    prob, decision, p_mal = apply_calibration(point_test, cal, args.dataset)
    if args.dataset == "lidc":
        auc = roc_auc_score(y[eligible], p_mal[eligible])
        acc = accuracy_score(y[eligible], decision[eligible])
    else:
        auc = roc_auc_score(y[:, 0], prob[:, 0])
        four = y.sum(axis=1) == 1
        acc = accuracy_score(y[four].argmax(axis=1), decision[four])
    metrics = pd.read_csv(parent / "evaluation_v2/metrics.csv").set_index("model")
    ablations = pd.DataFrame([
        {"ablation": "B1_zero_finding_hesitancy_refit_calibration", "auroc": auc,
         "accuracy": acc, "n_fit_calibration": int(mask.sum()),
         "n_test_auroc": int(eligible.sum()), "notes": "Point IFS from identical finding means; fit-only sigmoid refit"},
        {"ablation": "B2_zero_edge_hesitancy_equals_B1", "auroc": float(metrics.loc["b1", "auroc"]),
         "accuracy": float(metrics.loc["b1", "accuracy"]),
         "n_fit_calibration": int(mask.sum()), "n_test_auroc": int(eligible.sum()),
         "notes": "Exact operator identity, separately verified in B2 CLI"},
    ])
    edge_table = pd.read_csv(parent / "evaluation_v2/edge_quality.csv")
    strong = edge_table[edge_table.strong_fit_association.astype(bool)]
    wrong = strong.sign_disagreement.astype(bool)
    edge_auc = roc_auc_score(wrong, strong.raw_uncertainty) if wrong.nunique() == 2 else np.nan
    edge_summary = pd.DataFrame([{"n_strong_fit_edges": len(strong),
                                  "n_sign_disagreement": int(wrong.sum()),
                                  "wrong_edge_detection_auroc": edge_auc,
                                  "definition": "Sign disagreement with fit-only phi for |phi|>0.1; not clinical truth"}])
    reader_rows = []
    if args.dataset == "lidc":
        proper = eligible & (n_readers >= 3)
        x = arrays["a"]["finding_mean"]
        ambiguity = (4 * x * (1 - x)).mean(axis=1)
        disagreement = (4 * fractions * (1 - fractions)).mean(axis=1)
        case_idx = np.where(proper)[0]
        groups = np.unique(patients[case_idx])
        by_group = {g: case_idx[patients[case_idx] == g] for g in groups}
        rng = np.random.default_rng(2)
        draws = []
        for _ in range(args.bootstrap):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([by_group[g] for g in sampled])
            value = spearmanr(ambiguity[idx], disagreement[idx]).statistic
            if np.isfinite(value):
                draws.append(value)
        reader_rows.append({"measure": "finding_ambiguity_vs_reader_disagreement",
                            "n_nodules": len(case_idx), "n_patients": len(groups),
                            "spearman": spearmanr(ambiguity[case_idx], disagreement[case_idx]).statistic,
                            "ci_low": np.quantile(draws, 0.025) if draws else np.nan,
                            "ci_high": np.quantile(draws, 0.975) if draws else np.nan,
                            "valid_bootstrap": len(draws), "bootstrap_unit": "patient",
                            "interpretation": "descriptive, unadjusted, not causal"})
    reader_ci = pd.DataFrame(reader_rows, columns=["measure", "n_nodules", "n_patients",
                                                     "spearman", "ci_low", "ci_high",
                                                     "valid_bootstrap", "bootstrap_unit", "interpretation"])
    parameters = pd.DataFrame([{"model": m.upper(), "diagnoses": len(arrays["a"]["diagnoses"]),
                                "fit_calibration_parameters": 2 * len(arrays["a"]["diagnoses"]),
                                "reasoning_weights_fitted_from_diagnosis_labels": 0,
                                "runtime_seconds": np.nan,
                                "runtime_note": "Not logged in immutable training manifests"}
                               for m in ("a", "b1", "b2")])
    target.mkdir(parents=True, exist_ok=False)
    tables = {"ablations.csv": ablations, "edge_detection.csv": edge_summary,
              "reader_clustered_ci.csv": reader_ci, "parameters_compute.csv": parameters}
    for name, table in tables.items():
        table.to_csv(target / name, index=False)
    _json_new(target / "supplementary_manifest.json", {
        "dataset": args.dataset, "encoder": args.encoder, "bootstrap": args.bootstrap,
        "source_evaluation_sha256": sha(parent / "evaluation_v2/evaluation_manifest.json"),
        "outputs": {name: sha(target / name) for name in tables},
        "note": "Prespecified point-IFS ablation and descriptive fit-only edge quality; no test-set tuning."})
    print(f"Supplement saved: {target}")


def verify(args):
    parent, *_ = _base(args)
    target = parent / "supplementary_v1"
    with (target / "supplementary_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["source_evaluation_sha256"] != sha(parent / "evaluation_v2/evaluation_manifest.json"):
        raise ValueError("Supplement source changed")
    for name, digest in manifest["outputs"].items():
        if sha(target / name) != digest:
            raise ValueError(f"Supplement changed: {name}")
    print(json.dumps({"status": "PASS", "dataset": args.dataset, "output": str(target)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("lidc", "derm"))
    parser.add_argument("--encoder", choices=("cnn", "monet", "biomedclip"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "build", "verify"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    args.encoder = args.encoder or ("cnn" if args.dataset == "lidc" else "monet")
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    try:
        if args.dry_run:
            parent, *_ = _base(args)
            print(json.dumps({"dataset": args.dataset, "output": str(parent / "supplementary_v1"),
                              "bootstrap": args.bootstrap}, indent=2))
        elif args.build:
            build(args)
        else:
            verify(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
