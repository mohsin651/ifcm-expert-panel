"""Exploratory, label-free decomposition of IFS hesitation by input source.

Reads frozen A/B1/B2 predictions; never refits a diagnosis model or edits them.
Test labels are used only after the four counterfactual scores are fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ifcm.b1 import ROOT, _json_new, run_dir
from ifcm.b1_inputs import lidc_context
from ifcm.evaluate_comparison import _oracle, _read
from ifcm.intuitionistic_reasoning import IFS, point_ifs, reason_b1, reason_b2
from ifcm.safe_cnn import sha


FOLDER = "source_uncertainty_v1"
SCORES = ("a_low_confidence", "total_hesitation", "baseline_hesitation",
          "image_added_hesitation", "edge_added_hesitation", "source_balance")
TRIAGE = ("a_low_confidence", "total_hesitation",
          "image_added_hesitation", "edge_added_hesitation")


def decompose(mean: np.ndarray, finding: IFS, edge: np.ndarray,
              lower: np.ndarray, upper: np.ndarray) -> dict[str, np.ndarray]:
    """Four IFS states and order-symmetric incremental hesitation shares."""
    if mean.shape != finding.mu.shape:
        raise ValueError("Point and interval finding shapes differ")
    point = point_ifs(mean)
    p00 = reason_b1(point, edge).diagnosis.pi
    p10 = reason_b1(finding, edge).diagnosis.pi
    p01 = reason_b2(point, lower, upper).diagnosis.pi
    p11 = reason_b2(finding, lower, upper).diagnosis.pi
    image = ((p10 - p00) + (p11 - p01)) / 2
    knowledge = ((p01 - p00) + (p11 - p10)) / 2
    if (np.min(image) < -1e-9 or np.min(knowledge) < -1e-9 or
            not np.allclose(p00 + image + knowledge, p11, rtol=0, atol=1e-10)):
        raise ValueError("Hesitation decomposition identity or nesting failed")
    return {"neither": p00, "image_only": p10, "edge_only": p01,
            "both": p11, "image_added": image, "edge_added": knowledge}


def _arrays(parent: Path, stage: str) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for model in ("a", "b1", "b2"):
        path = parent / model / f"{stage}_predictions.npz"
        with np.load(path, allow_pickle=False) as z:
            output[model] = {name: z[name] for name in z.files}
    for model in ("b1", "b2"):
        for name in ("ids", "patients", "concepts", "diagnoses", "finding_mean", "fold"):
            if not np.array_equal(output["a"][name], output[model][name]):
                raise ValueError(f"{stage} A/{model} alignment failed: {name}")
    return output


def _chosen(a: dict[str, np.ndarray], dataset: str) -> np.ndarray:
    decision = a["decision"].astype(int)
    return np.where(decision == 1, 0, 1) if dataset == "lidc" else decision


def _scores(arrays: dict[str, dict[str, np.ndarray]], edge: np.ndarray,
            lower: np.ndarray, upper: np.ndarray, dataset: str) -> tuple[dict[str, np.ndarray],
                                                                        dict[str, np.ndarray]]:
    a, b1, b2 = (arrays[model] for model in ("a", "b1", "b2"))
    finding = IFS(b1["finding_mu"], b1["finding_nu"], b1["finding_pi"])
    parts = decompose(a["finding_mean"], finding, edge, lower, upper)
    if (not np.allclose(parts["image_only"], b1["diagnosis_pi"], rtol=0, atol=1e-12) or
            not np.allclose(parts["both"], b2["diagnosis_pi"], rtol=0, atol=1e-12)):
        raise ValueError("Recomputed B1/B2 hesitation differs from frozen predictions")
    ix = np.arange(len(a["ids"]))
    chosen = _chosen(a, dataset)
    selected = {name: values[ix, chosen] for name, values in parts.items()}
    if dataset == "lidc":
        p = a["p_mal"]
        low_conf = 1 - np.maximum(p, 1 - p)
    else:
        low_conf = 1 - a["calibrated"].max(axis=1)
    scores = {"a_low_confidence": low_conf,
              "total_hesitation": selected["both"],
              "baseline_hesitation": selected["neither"],
              "image_added_hesitation": selected["image_added"],
              "edge_added_hesitation": selected["edge_added"],
              "source_balance": selected["image_added"] - selected["edge_added"]}
    return selected, scores


def _auc(label: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(label, score)) if len(np.unique(label)) == 2 else float("nan")


def _paired_auc(label: np.ndarray, left: np.ndarray, right: np.ndarray,
                groups: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    unique = np.unique(groups)
    grouped = {group: np.where(groups == group)[0] for group in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([grouped[group] for group in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(label[idx])) == 2:
            draws.append(_auc(label[idx], left[idx]) - _auc(label[idx], right[idx]))
    bounds = np.quantile(draws, [0.025, 0.975]) if draws else [float("nan")] * 2
    return {"difference": _auc(label, left) - _auc(label, right),
            "ci_low": float(bounds[0]), "ci_high": float(bounds[1]),
            "valid_bootstrap": len(draws), "requested_bootstrap": n_bootstrap,
            "unit": "patient" if len(unique) < len(label) else "case", "seed": seed}


def analyze(args: argparse.Namespace) -> None:
    parent, base, test, edge, ids, patients, clear, y, findings, _, _ = _read(args)
    target = parent / FOLDER
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    fit = _arrays(parent, "fit")
    test = _arrays(parent, "test")
    with np.load(parent / "b2/edge_bounds.npz", allow_pickle=False) as z:
        lower, upper = z["lower"], z["upper"]
        if not np.array_equal(edge, z["mean"]):
            raise ValueError("B2 edge mean differs from A/B1 edge mean")
    fit_parts, fit_scores = _scores(fit, edge, lower, upper, args.dataset)
    test_parts, test_scores = _scores(test, edge, lower, upper, args.dataset)
    if not np.array_equal(test["a"]["ids"].astype(str), ids):
        raise ValueError("Evaluation/test IDs disagree")
    if args.dataset == "lidc":
        _, split, _ = lidc_context(ROOT, base["cnn_run_id"])
        clear_fit = set(split["clear_fit_ids"])
        fit_mask = np.array([id_ in clear_fit for id_ in fit["a"]["ids"].astype(str)])
        eligible = clear
        correct_class = y
    else:
        fit_mask = np.ones(len(fit["a"]["ids"]), bool)
        eligible = y.sum(axis=1) == 1
        correct_class = y.argmax(axis=1)
    if (args.dataset == "lidc" and (fit_mask.sum() != 740 or eligible.sum() != 462)) or (
            args.dataset == "derm" and (fit_mask.sum() != 616 or eligible.sum() != 355)):
        raise ValueError("Fit or test evaluation denominator changed")
    oracle = _oracle(parent, test, edge, findings, args.dataset)["a"]
    wrong = eligible & (test["a"]["decision"] != correct_class)
    perception = wrong & (oracle == correct_class)
    if args.dataset == "lidc" and (wrong.sum(), perception.sum()) != (88, 32):
        raise ValueError("LIDC operational A-error counts changed")
    if args.dataset == "derm" and (wrong.sum(), perception.sum()) != (93, 37):
        raise ValueError("Derm operational A-error counts changed")
    decision_flips = pd.read_csv(parent / "evaluation_v2/decision_flips.csv")
    if list(decision_flips["id"].astype(str)) != list(ids):
        raise ValueError("Decision-flip rows do not align")
    flip_names = [name for name in decision_flips if name.startswith("rank_")]
    if len(flip_names) != 1:
        raise ValueError("Expected one frozen A flip-minus-knowledge score")
    flip = decision_flips[flip_names[0]].to_numpy(dtype=float)

    score_rows = []
    for name in SCORES:
        value = test_scores[name]
        score_rows.append({"score": name, "target": "A_error_vs_correct",
                           "n": int(eligible.sum()), "positives": int(wrong.sum()),
                           "auroc": _auc(wrong[eligible], value[eligible])})
        score_rows.append({"score": name, "target": "perception_vs_knowledge_on_A_errors",
                           "n": int(wrong.sum()), "positives": int(perception.sum()),
                           "auroc": _auc(perception[wrong], value[wrong])})
    score_rows.append({"score": "existing_A_flip_minus_knowledge", "target": "perception_vs_knowledge_on_A_errors",
                       "n": int(wrong.sum()), "positives": int(perception.sum()),
                       "auroc": _auc(perception[wrong], flip[wrong])})
    contrast = [
        {"target": "A_error_vs_correct", "left": "total_hesitation", "right": "a_low_confidence",
         **_paired_auc(wrong[eligible], test_scores["total_hesitation"][eligible],
                       test_scores["a_low_confidence"][eligible], patients[eligible], args.bootstrap, 17)},
        {"target": "perception_vs_knowledge_on_A_errors", "left": "source_balance",
         "right": "existing_A_flip_minus_knowledge",
         **_paired_auc(perception[wrong], test_scores["source_balance"][wrong], flip[wrong],
                       patients[wrong], args.bootstrap, 19)},
    ]

    triage_rows = []
    for name in TRIAGE:
        threshold = float(np.quantile(fit_scores[name][fit_mask], 0.8))
        referred = eligible & (test_scores[name] > threshold)
        retained = eligible & ~referred
        triage_rows.append({"score": name, "fit_threshold_q": 0.8, "fit_threshold": threshold,
                            "fit_n": int(fit_mask.sum()), "test_n": int(eligible.sum()),
                            "referred_n": int(referred.sum()), "retained_n": int(retained.sum()),
                            "test_coverage": float(retained.sum() / eligible.sum()),
                            "referred_error_n": int((referred & wrong).sum()),
                            "retained_error_n": int((retained & wrong).sum()),
                            "retained_error_rate": float((retained & wrong).sum() / retained.sum())
                            if retained.any() else float("nan"),
                            "error_capture": float((referred & wrong).sum() / wrong.sum())})

    case = pd.DataFrame({"id": ids, "patient": patients, "eligible": eligible,
                         "a_wrong": wrong, "perception_error": perception,
                         "knowledge_error": wrong & ~perception,
                         "a_predicted_diagnosis": test["a"]["diagnoses"][_chosen(test["a"], args.dataset)],
                         **{f"pi_{name}": values for name, values in test_parts.items()},
                         **test_scores, "existing_A_flip_minus_knowledge": flip})
    summary = {"dataset": args.dataset, "run_id": args.run_id, "protocol": base["protocol"],
               "status": "exploratory_posthoc", "test_rows": len(ids), "eligible_rows": int(eligible.sum()),
               "fit_threshold_rows": int(fit_mask.sum()), "a_errors": int(wrong.sum()),
               "perception_errors": int(perception.sum()),
               "knowledge_errors": int((wrong & ~perception).sum()),
               "mean_pi_components_on_eligible": {
                   name: float(test_parts[name][eligible].mean()) for name in
                   ("neither", "image_added", "edge_added", "both")},
               "interpretation": "IFS pi is residual hesitation, not calibrated error probability; "
                                 "source terms are order-symmetric increments; operational error types "
                                 "are counterfactual, not clinician-annotated causes. Test results were "
                                 "already inspected during prior method development."}
    target.mkdir(parents=True, exist_ok=False)
    outputs = {"case_scores.csv": case, "score_metrics.csv": pd.DataFrame(score_rows),
               "paired_auc.csv": pd.DataFrame(contrast), "triage.csv": pd.DataFrame(triage_rows)}
    for name, table in outputs.items():
        table.to_csv(target / name, index=False)
    _json_new(target / "summary.json", summary)
    _json_new(target / "manifest.json", {"dataset": args.dataset, "run_id": args.run_id,
             "code_sha256": sha(ROOT / "ifcm/source_uncertainty.py"),
             "inputs_sha256": {str(path): sha(path) for path in
                               [parent / "a/test_predictions.npz", parent / "a/fit_predictions.npz",
                                parent / "b1/test_predictions.npz", parent / "b1/fit_predictions.npz",
                                parent / "b2/test_predictions.npz", parent / "b2/fit_predictions.npz",
                                parent / "b2/edge_bounds.npz",
                                parent / "evaluation_v2/decision_flips.csv"]},
             "outputs_sha256": {name: sha(target / name) for name in (*outputs, "summary.json")},
             "bootstrap": args.bootstrap, "threshold_fit_quantile": 0.8,
             "status": "exploratory_posthoc"})
    print(json.dumps({"output": str(target), **summary, "paired_auc": contrast,
                      "triage": triage_rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("lidc", "derm"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    args.encoder = "cnn" if args.dataset == "lidc" else "monet"
    try:
        analyze(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
