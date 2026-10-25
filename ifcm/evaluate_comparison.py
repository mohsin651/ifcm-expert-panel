"""Frozen, paired A/B1/B2 evaluation. No fitting or model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from ifcm.b1 import ROOT, _json_new, _json_read, apply_calibration, run_dir
from ifcm.b1_inputs import derm_context, derm_diagnoses, derm_findings, derm_probe_draws, lidc_context
from ifcm.comparison import _load_inputs
from ifcm.intuitionistic_reasoning import point_ifs, reason_b1, reason_b2
from ifcm.safe_cnn import sha
from ifcm.uncertainty import exact_flip_probabilities


MODELS = ("a", "b1", "b2")
EVAL_FOLDER = "evaluation_v2"


def _read(args):
    b1, b2, a, base, edge, concepts, diagnoses = _load_inputs(args)
    parent = b1.parent
    for model in MODELS:
        folder = parent / model
        if not (folder / "prediction_manifest.json").exists():
            raise FileNotFoundError(f"Incomplete model predictions: {folder}")
        proof = _json_read(folder / "prediction_manifest.json")
        expected = proof.get("prediction_sha256", proof.get("sha256"))
        if expected != sha(folder / "test_predictions.npz"):
            raise ValueError(f"Frozen {model} predictions changed")
    arrays = {}
    for model in MODELS:
        with np.load(parent / model / "test_predictions.npz", allow_pickle=False) as z:
            arrays[model] = {k: z[k] for k in z.files}
    for model in ("b1", "b2"):
        for name in ("ids", "patients", "split", "fold", "concepts", "diagnoses", "finding_mean"):
            if not np.array_equal(arrays["a"][name], arrays[model][name]):
                raise ValueError(f"A/{model} alignment failed: {name}")
    if args.dataset == "lidc":
        _, split, ratings = lidc_context(ROOT, base["cnn_run_id"])
        ids = arrays["a"]["ids"].astype(str)
        if list(ids) != split["test_ids"]:
            raise ValueError("Strict LIDC test IDs changed")
        clear = np.array([id_ in set(split["clear_test_ids"]) for id_ in ids])
        if clear.sum() != 462:
            raise ValueError("Strict LIDC diagnosis denominator changed")
        y = (ratings.loc[ids, "mal_mean"].to_numpy() > 3).astype(int)
        findings = (ratings.loc[ids, concepts].to_numpy() >= 0.5).astype(float)
        patients = arrays["a"]["patients"].astype(str)
        n_readers = pd.read_csv(ROOT / "results_lidc/lidc_ratings.csv").groupby("nodule").size().reindex(ids).to_numpy()
        return parent, base, arrays, edge, ids, patients, clear, y, findings, ratings.loc[ids, concepts].to_numpy(), n_readers
    meta, splits, _ = derm_context(ROOT, args.encoder)
    ids = arrays["a"]["ids"].astype(str)
    expected = [f"row:{i:04d}" for i in splits["test"]]
    if list(ids) != expected:
        raise ValueError("Derm test indices changed")
    y = derm_diagnoses(meta)[splits["test"]]
    findings = derm_findings(meta)[splits["test"]]
    patients = ids.copy()  # Dataset has no patient IDs: case-level resampling only.
    return parent, base, arrays, edge, ids, patients, np.ones(len(ids), bool), y, findings, findings, np.full(len(ids), -1)


def _prob(model, values):
    return values["p_mal"] if model and "p_mal" in values else values["calibrated"][:, 0]


def _safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def _oracle(parent, arrays, edge, findings, dataset):
    with np.load(parent / "b2/edge_bounds.npz", allow_pickle=False) as z:
        lower, upper = z["lower"], z["upper"]
    scores = {"a": findings @ edge,
              "b1": reason_b1(point_ifs(findings), edge).score,
              "b2": reason_b2(point_ifs(findings), lower, upper).score}
    decisions = {}
    for model in MODELS:
        cal = _json_read(parent / model / "calibration.json")
        _, decision, _ = apply_calibration(scores[model], cal, dataset)
        decisions[model] = decision
    return decisions


def _bootstrap_pairs(y, scores, groups, eligible, n_boot, seed, endpoint):
    rng = np.random.default_rng(seed)
    names = list(scores)
    masks = np.where(eligible)[0]
    group_values = np.unique(groups[masks])
    grouped = {g: masks[groups[masks] == g] for g in group_values}
    observed = {m: _safe_auc(y[masks], scores[m][masks]) for m in names}
    diffs = {pair: [] for pair in (("a", "b1"), ("a", "b2"), ("b1", "b2"))}
    invalid = 0
    for _ in range(n_boot):
        sampled_groups = rng.choice(group_values, size=len(group_values), replace=True)
        idx = np.concatenate([grouped[g] for g in sampled_groups])
        if len(np.unique(y[idx])) < 2:
            invalid += 1
            continue
        auc = {m: roc_auc_score(y[idx], scores[m][idx]) for m in names}
        for pair in diffs:
            diffs[pair].append(auc[pair[0]] - auc[pair[1]])
    output = []
    for (left, right), draws in diffs.items():
        q = np.quantile(draws, [0.025, 0.975]) if draws else [np.nan, np.nan]
        output.append({"endpoint": endpoint, "left": left, "right": right,
                       "difference": observed[left] - observed[right],
                       "ci_low": q[0], "ci_high": q[1], "valid_bootstrap": len(draws),
                       "invalid_bootstrap": invalid, "bootstrap_unit": "patient" if len(np.unique(groups[masks])) < len(masks) else "case",
                       "n_cases": len(masks), "n_groups": len(group_values),
                       "seed": seed, "n_requested": n_boot})
    return output


def _a_decision_flips(parent, args, base, arrays, edge):
    """Reproduce original A sensitivity definitions; no diagnostic labels used."""
    x = arrays["a"]["finding_mean"]
    baseline = arrays["a"]["decision"]
    calibration = _json_read(parent / "a/calibration.json")

    def decide(finding, weights=edge):
        return apply_calibration(finding @ weights, calibration, args.dataset)[1]

    with np.load(parent / "b2/edge_bounds.npz", allow_pickle=False) as z:
        members = z["member_edges"]
    mean_norm = np.linalg.norm(edge)
    know = np.mean([decide(x, w * mean_norm / max(np.linalg.norm(w), 1e-9)) != baseline
                    for w in members], axis=0)
    if args.dataset == "lidc":
        cnn = ROOT / "fresh_runs/ifs_comparison" / base["cnn_run_id"] / "cnn/predictions/outer_test.npz"
        with np.load(cnn, allow_pickle=False) as z:
            if not np.array_equal(z["ids"], arrays["a"]["ids"]):
                raise ValueError("LIDC seed-draw IDs differ from A")
            draws = z["P_tta0"]
        name = "perc_bern"
        perc = exact_flip_probabilities(x, decide, baseline, batch_size=32)
    else:
        _, splits, image = derm_context(ROOT, args.encoder)
        with np.load(parent / "b1/probes.npz", allow_pickle=False) as z:
            probes = {key: z[key] for key in ("coef", "intercept", "active")}
        draws = derm_probe_draws(image[splits["test"]], probes)
        name = "perc_flip"
        perc = np.mean([decide(draws[:, :, b]) != baseline for b in range(draws.shape[2])], axis=0)
    boot = np.mean([decide(draws[:, :, b]) != baseline for b in range(draws.shape[2])], axis=0)
    rank = lambda value: pd.Series(value).rank().to_numpy() / len(value)
    scores = {name: perc, "perc_boot" if args.dataset == "lidc" else "perc_probe": boot,
              "know_flip": know, f"rank_{name}_minus_know_flip": rank(perc) - rank(know)}
    return scores


def _attribution_pairs(y_type, scores, groups, eligible, n_boot, seed):
    names = list(scores)
    rng = np.random.default_rng(seed)
    indices = np.where(eligible)[0]
    group_values = np.unique(groups[indices])
    grouped = {g: indices[groups[indices] == g] for g in group_values}
    pairs = [(left, right) for i, left in enumerate(names) for right in names[i + 1:]]
    observed = {name: _safe_auc(y_type[indices], value[indices]) for name, value in scores.items()}
    draws = {pair: [] for pair in pairs}
    invalid = 0
    for _ in range(n_boot):
        sampled = rng.choice(group_values, size=len(group_values), replace=True)
        idx = np.concatenate([grouped[g] for g in sampled])
        if len(np.unique(y_type[idx])) != 2:
            invalid += 1
            continue
        auc = {name: roc_auc_score(y_type[idx], value[idx]) for name, value in scores.items()}
        for pair in pairs:
            draws[pair].append(auc[pair[0]] - auc[pair[1]])
    output = []
    for (left, right), sample in draws.items():
        q = np.quantile(sample, [0.025, 0.975]) if sample else [np.nan, np.nan]
        output.append({"reference": "fixed_a_errors", "left": left, "right": right,
                       "difference": observed[left] - observed[right], "ci_low": q[0],
                       "ci_high": q[1], "n_wrong": len(indices),
                       "n_perception": int(y_type[indices].sum()),
                       "valid_bootstrap": len(sample), "invalid_bootstrap": invalid,
                       "bootstrap_unit": "patient" if len(group_values) < len(indices) else "case",
                       "seed": seed, "n_requested": n_boot})
    return output


def dry_run(args):
    parent, base, arrays, edge, ids, patients, eligible, y, findings, reader_frac, n_readers = _read(args)
    output = parent / EVAL_FOLDER
    print(json.dumps({"dataset": args.dataset, "protocol": base["protocol"],
                      "run_id": args.run_id, "source": str(parent), "output": str(output),
                      "rows": len(ids), "diagnosis_rows": int(eligible.sum()) if args.dataset == "lidc" else len(ids),
                      "bootstrap": args.bootstrap, "unit": "patient" if args.dataset == "lidc" else "case",
                      "warning": "Evaluation only; no model refitting or test-set selection."}, indent=2))


def evaluate(args):
    parent, base, arrays, edge, ids, patients, eligible, y, findings, reader_frac, n_readers = _read(args)
    target = parent / EVAL_FOLDER
    if target.exists():
        raise FileExistsError(f"Evaluation already exists: {target}")
    oracle = _oracle(parent, arrays, edge, findings, args.dataset)
    metrics, attribution, curves, diagnostics, readers, edges, finding_auc = [], [], [], [], [], [], []
    binary = y if args.dataset == "lidc" else y[:, 0]
    mask_mel = eligible if args.dataset == "lidc" else np.ones(len(ids), bool)
    in4 = eligible if args.dataset == "lidc" else y.sum(axis=1) == 1
    correct_class = y if args.dataset == "lidc" else y.argmax(axis=1)
    score = {model: _prob(model, arrays[model]) for model in MODELS}
    for model in MODELS:
        z = arrays[model]
        p = score[model]
        pred = z["decision"]
        confidence = np.maximum(p, 1 - p) if args.dataset == "lidc" else z["calibrated"].max(axis=1)
        wrong = in4 & (pred != correct_class)
        perc = wrong & (oracle[model] == correct_class)
        know = wrong & (oracle[model] != correct_class)
        metrics.append({"model": model, "dataset": args.dataset, "protocol": base["protocol"],
                        "endpoint": "malignancy" if args.dataset == "lidc" else "melanoma",
                        "n_auc": int(mask_mel.sum()), "n_accuracy": int(in4.sum()),
                        "n_patients": len(np.unique(patients[in4])) if args.dataset == "lidc" else np.nan,
                        "auroc": _safe_auc(binary[mask_mel], p[mask_mel]),
                        "accuracy": accuracy_score(correct_class[in4], pred[in4]),
                        "brier": brier_score_loss(binary[mask_mel], p[mask_mel]),
                        "n_errors": int(wrong.sum()), "n_perception_errors": int(perc.sum()),
                        "n_knowledge_errors": int(know.sum())})
        for lo in np.linspace(0, 0.9, 10):
            selected = mask_mel & (p >= lo) & ((p < lo + 0.1) if lo < 0.9 else (p <= 1))
            curves.append({"model": model, "bin_low": lo, "bin_high": lo + 0.1,
                           "n": int(selected.sum()), "mean_prediction": float(p[selected].mean()) if selected.any() else np.nan,
                           "observed_fraction": float(binary[selected].mean()) if selected.any() else np.nan})
        chosen = pred if args.dataset == "derm" else np.where(pred == 1, 0, 1)
        if model != "a":
            pi = z["diagnosis_pi"][np.arange(len(ids)), chosen]
            diagnostics.append({"model": model, "n": len(pi), "mean_pi": pi.mean(),
                                "median_pi": np.median(pi), "min_pi": pi.min(), "max_pi": pi.max(),
                                "pi_error_auroc": _safe_auc((pred[in4] != correct_class[in4]).astype(int), pi[in4])})
            if not np.allclose(z["diagnosis_mu"] + z["diagnosis_nu"] + z["diagnosis_pi"], 1, atol=1e-10):
                raise ValueError(f"{model} IFS invalid")
        for name, values in (("one_minus_confidence", 1 - confidence),
                             ("diagnosis_pi", z["diagnosis_pi"][np.arange(len(ids)), chosen] if model != "a" else None)):
            if values is None:
                continue
            if wrong.sum() > 0:
                y_type = perc[wrong].astype(int)
                attribution.append({"model": model, "reference": "model_specific",
                                    "score": name, "n_wrong": int(wrong.sum()),
                                    "n_perception": int(perc.sum()), "n_knowledge": int(know.sum()),
                                    "auroc": _safe_auc(y_type, values[wrong])})
    a_wrong = in4 & (arrays["a"]["decision"] != correct_class)
    a_perc = a_wrong & (oracle["a"] == correct_class)
    for model in MODELS:
        z = arrays[model]
        p = score[model]
        confidence = np.maximum(p, 1 - p) if args.dataset == "lidc" else z["calibrated"].max(axis=1)
        chosen = z["decision"] if args.dataset == "derm" else np.where(z["decision"] == 1, 0, 1)
        candidates = [("one_minus_confidence", 1 - confidence)]
        if model != "a":
            candidates.append(("diagnosis_pi", z["diagnosis_pi"][np.arange(len(ids)), chosen]))
        for name, value in candidates:
            attribution.append({"model": model, "reference": "fixed_a_errors",
                                "score": name, "n_wrong": int(a_wrong.sum()),
                                "n_perception": int(a_perc.sum()), "n_knowledge": int((a_wrong & ~a_perc).sum()),
                                "auroc": _safe_auc(a_perc[a_wrong].astype(int), value[a_wrong])})
    paired = _bootstrap_pairs(binary, score, patients, mask_mel, args.bootstrap, 2,
                              "malignancy" if args.dataset == "lidc" else "melanoma")
    if args.dataset == "lidc":
        proper = eligible & (n_readers >= 3)
        ambiguity = (4 * arrays["a"]["finding_mean"] * (1 - arrays["a"]["finding_mean"])).mean(axis=1)
        disagreement = (4 * reader_frac * (1 - reader_frac)).mean(axis=1)
        readers.append({"measure": "finding_ambiguity_vs_reader_disagreement", "n": int(proper.sum()),
                        "n_patients": len(np.unique(patients[proper])),
                        "spearman": spearmanr(ambiguity[proper], disagreement[proper]).statistic,
                        "interpretation": "descriptive; not causal; no residualization"})
    for c, concept in enumerate(arrays["a"]["concepts"]):
        if len(np.unique(findings[eligible, c])) == 2:
            finding_auc.append({"concept": str(concept), "n": int(eligible.sum()),
                                "auroc": roc_auc_score(findings[eligible, c],
                                                       arrays["a"]["finding_mean"][eligible, c])})
    flip_scores = _a_decision_flips(parent, args, base, arrays, edge)
    flip_rows = [{"id": str(id_), "patient": str(patient), **{name: float(value[i]) for name, value in flip_scores.items()}}
                 for i, (id_, patient) in enumerate(zip(ids, patients))]
    for name, value in flip_scores.items():
        if name == "know_flip":
            continue
        for reference, wrong_mask, perc_mask in (("model_specific", a_wrong, a_perc),
                                                  ("fixed_a_errors", a_wrong, a_perc)):
            attribution.append({"model": "a", "reference": reference,
                                "score": name, "n_wrong": int(wrong_mask.sum()),
                                "n_perception": int(perc_mask.sum()),
                                "n_knowledge": int((wrong_mask & ~perc_mask).sum()),
                                "auroc": _safe_auc(perc_mask[wrong_mask].astype(int), value[wrong_mask])})
    paired_attribution = _attribution_pairs(a_perc.astype(int), {
        "a_flip_minus_know": flip_scores[[name for name in flip_scores if name.startswith("rank_")][0]],
        "a_one_minus_confidence": 1 - (np.maximum(score["a"], 1 - score["a"]) if args.dataset == "lidc" else arrays["a"]["calibrated"].max(axis=1)),
        "b1_pi": arrays["b1"]["diagnosis_pi"][np.arange(len(ids)),
                 arrays["b1"]["decision"] if args.dataset == "derm" else np.where(arrays["b1"]["decision"] == 1, 0, 1)],
        "b2_pi": arrays["b2"]["diagnosis_pi"][np.arange(len(ids)),
                 arrays["b2"]["decision"] if args.dataset == "derm" else np.where(arrays["b2"]["decision"] == 1, 0, 1)],
    }, patients, a_wrong, args.bootstrap, 2)
    # Edge-level sign recovery uses FIT labels/findings only and is separate from patient attribution.
    with np.load(parent / "b2/edge_bounds.npz", allow_pickle=False) as z:
        uncertainty = z["raw_half_width"]
    if args.dataset == "lidc":
        _, split, ratings = lidc_context(ROOT, base["cnn_run_id"])
        idx = split["clear_fit_ids"]
        x_fit = (ratings.loc[idx, list(arrays["a"]["concepts"])].to_numpy() >= 0.5).astype(float)
        y_fit = (ratings.loc[idx, "mal_mean"].to_numpy() > 3).astype(float)[:, None]
        edge_diagnoses = [0]
    else:
        meta, split, _ = derm_context(ROOT, args.encoder)
        indices = np.concatenate([split["train"], split["valid"]])
        x_fit = derm_findings(meta)[indices]
        y_fit = derm_diagnoses(meta)[indices]
        edge_diagnoses = range(edge.shape[1])
    for d in edge_diagnoses:
        for c in range(edge.shape[0]):
            phi = np.corrcoef(x_fit[:, c], y_fit[:, d])[0, 1] if x_fit[:, c].std() and y_fit[:, d].std() else np.nan
            edges.append({"concept": str(arrays["a"]["concepts"][c]),
                          "diagnosis": str(arrays["a"]["diagnoses"][d]),
                          "mean_edge": edge[c, d], "raw_uncertainty": uncertainty[c, d],
                          "fit_phi": phi, "strong_fit_association": bool(np.isfinite(phi) and abs(phi) > 0.1),
                          "sign_disagreement": bool(np.isfinite(phi) and abs(phi) > 0.1 and np.sign(phi) != np.sign(edge[c, d]))})
    target.mkdir(parents=True, exist_ok=False)
    tables = {"metrics.csv": metrics, "paired_comparisons.csv": paired,
              "attribution.csv": attribution, "calibration.csv": curves,
              "reader_correlations.csv": readers, "ifs_diagnostics.csv": diagnostics,
              "edge_quality.csv": edges, "finding_auroc.csv": finding_auc,
              "decision_flips.csv": flip_rows, "attribution_paired.csv": paired_attribution}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(target / name, index=False)
    _json_new(target / "evaluation_manifest.json", {
        "dataset": args.dataset, "encoder": args.encoder, "protocol": base["protocol"],
        "n_all_test": len(ids), "n_diagnosis": int(mask_mel.sum()),
        "n_four_class": int(in4.sum()) if args.dataset == "derm" else None,
        "bootstrap": args.bootstrap, "seed": 2,
        "prediction_hashes": {m: sha(parent / m / "test_predictions.npz") for m in MODELS},
        "output_hashes": {name: sha(target / name) for name in tables},
        "limitations": ["LIDC is strict for CNN fitting, but method design had prior exposure to historical pilot results.",
                        "Derm case-level bootstrap assumes independent rows; patient IDs are unavailable.",
                        "Attribution categories are oracle counterfactuals, not annotated causal explanations."]})
    print(f"Evaluation saved: {target}; diagnosis rows={int(mask_mel.sum())}; bootstrap={args.bootstrap}")


def summary(args):
    target = run_dir(args.run_id).parent / EVAL_FOLDER
    manifest = _json_read(target / "evaluation_manifest.json")
    for name, expected in manifest["output_hashes"].items():
        if sha(target / name) != expected:
            raise ValueError(f"Evaluation file changed: {name}")
    print(pd.read_csv(target / "metrics.csv").to_string(index=False))
    print(pd.read_csv(target / "paired_comparisons.csv").to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("lidc", "derm"))
    parser.add_argument("--encoder", choices=("cnn", "monet", "biomedclip"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "evaluate", "summary"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    args.encoder = args.encoder or ("cnn" if args.dataset == "lidc" else "monet")
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    try:
        if args.dry_run:
            dry_run(args)
        elif args.evaluate:
            evaluate(args)
        else:
            summary(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
