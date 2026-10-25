"""Read-only, exact-order upstream adapters for Model B1.

Never import analysis/pilot.py or analysis/lidc.py: both execute experiments on
import. This module copies only their relevant definitions and checks parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob

import numpy as np
import pandas as pd

from ifcm.safe_cnn import CONCEPTS as LIDC_CONCEPTS, inputs as cnn_inputs, read


DERM_CONCEPTS = (
    "atypical_pn", "typical_pn", "irregular_streaks", "regular_streaks",
    "irregular_pigm", "regular_pigm", "regression", "irregular_dg",
    "regular_dg", "bwv", "arborizing", "atypical_vessels", "comma",
    "hairpin", "vessels_regr",
)
DERM_DIAGNOSES = ("MEL", "NEV", "BCC", "SK")
LIDC_DIAGNOSES = ("MAL", "BEN")


@dataclass(frozen=True)
class Panel:
    mean: np.ndarray
    concepts: tuple[str, ...]
    diagnoses: tuple[str, ...]
    members: tuple[str, ...]
    styles: tuple[str, ...]
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class Rows:
    ids: np.ndarray
    patients: np.ndarray
    draws: np.ndarray
    fold: np.ndarray
    concepts: tuple[str, ...]


def _unique_ids(name: str, ids: np.ndarray) -> None:
    if len(ids) != len(set(map(str, ids))):
        raise ValueError(f"Duplicate {name} IDs")


def _style_paths(folder: Path, style: str) -> tuple[Path, ...]:
    pattern = "llm_votes*.csv" if style == "orig" else f"llm_{style}_*.csv"
    paths = tuple(Path(p) for p in sorted(glob.glob(str(folder / pattern))))
    if not paths:
        raise FileNotFoundError(f"No {style} LLM votes under {folder}")
    return paths


def _style_weights(
    paths: tuple[Path, ...], style: str, models: tuple[str, ...],
    concepts: tuple[str, ...], diagnoses: tuple[str, ...],
) -> pd.Series:
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    required = {"model", "concept", "template", "order"}
    required |= ({"p_unsure", *(f"p_{diag}" for diag in diagnoses)}
                 if style == "choice" else {"diag", "p_up", "p_down"})
    if not required <= set(frame):
        raise ValueError(f"Missing {style} vote columns: {sorted(required - set(frame))}")
    frame = frame[frame.model.isin(models)]
    expected = pd.MultiIndex.from_product(
        [models, concepts] if style == "choice" else [models, concepts, diagnoses],
        names=["model", "concept"] if style == "choice" else ["model", "concept", "diag"],
    )
    keys = ["model", "concept"] if style == "choice" else ["model", "concept", "diag"]
    counts = frame.groupby(keys, sort=True, dropna=False).size()
    if not counts.index.equals(expected.sort_values()) or not (counts == 9).all():
        raise ValueError(f"{style} panel has missing/extra links or not 9 prompts per link")
    prompt_keys = keys + ["template", "order"]
    if frame.duplicated(prompt_keys).any() or set(frame.template) != {0, 1, 2} or set(frame.order) != {0, 1, 2}:
        raise ValueError(f"{style} prompt template/order coverage is invalid")
    if style == "choice":
        grouped = frame.groupby(keys, sort=True)[[f"p_{d}" for d in diagnoses] + ["p_unsure"]].mean()
        records = []
        for diag in diagnoses:
            q = grouped[f"p_{diag}"] / (1 - grouped.p_unsure + 1e-6)
            if len(diagnoses) == 2:
                weight = 2 * q - 1
            else:
                chance = 1 / len(diagnoses)
                weight = np.where(q >= chance, (q - chance) / (1 - chance),
                                  (q - chance) / chance)
            part = grouped.reset_index()[["model", "concept"]].copy()
            part["diag"] = diag
            part["w"] = np.asarray(weight)
            records.append(part)
        weighted = pd.concat(records, ignore_index=True)
    else:
        grouped = frame.groupby(keys, sort=True)[["p_up", "p_down"]].mean()
        weighted = (grouped.p_up - grouped.p_down).rename("w").reset_index()
    if (not np.isfinite(weighted.w).all() or
            np.any(weighted.w < -1 - 1e-6) or np.any(weighted.w > 1 + 1e-6)):
        raise ValueError(f"{style} edge weights are invalid")
    weighted["model"] = weighted.model + "|" + style
    return weighted.set_index(["model", "concept", "diag"]).w


def load_panel(root: Path, dataset: str) -> Panel:
    if dataset == "lidc":
        folder = root / "results_lidc"
        concepts, diagnoses, styles = LIDC_CONCEPTS, LIDC_DIAGNOSES, ("comp", "choice")
    elif dataset == "derm":
        folder = root / "results"
        concepts, diagnoses, styles = DERM_CONCEPTS, DERM_DIAGNOSES, ("orig", "comp", "choice")
    else:
        raise ValueError("dataset must be lidc or derm")
    choice_paths = _style_paths(folder, "choice")
    choice_models = tuple(sorted(set(pd.concat(
        (pd.read_csv(p, usecols=["model"]) for p in choice_paths)).model)))
    if len(choice_models) != 10:
        raise ValueError(f"Expected 10 final panel LLMs, found {len(choice_models)}")
    paths = tuple(path for style in styles for path in _style_paths(folder, style))
    pieces = []
    for style in styles:
        pieces.append(_style_weights(_style_paths(folder, style), style,
                                     choice_models, concepts, diagnoses))
    votes = pd.concat(pieces)
    members = tuple(sorted(f"{model}|{style}" for model in choice_models for style in styles))
    expected = pd.MultiIndex.from_product([members, concepts, diagnoses],
                                           names=["model", "concept", "diag"])
    if not votes.index.is_unique or set(votes.index) != set(expected):
        raise ValueError("Panel member/concept/diagnosis coverage mismatch")
    member_weights = np.stack([
        votes.loc[pd.MultiIndex.from_product([[member], concepts, diagnoses])]
        .to_numpy().reshape(len(concepts), len(diagnoses))
        for member in members
    ])
    return Panel(member_weights.mean(axis=0), concepts, diagnoses, members, styles, paths)


def lidc_context(root: Path, cnn_run_id: str) -> tuple[Path, dict, pd.DataFrame]:
    from ifcm.safe_cnn import run_dir

    folder = run_dir(cnn_run_id)
    if folder != root / "fresh_runs/ifs_comparison" / cnn_run_id / "cnn":
        raise ValueError("CNN run is outside this repository")
    source, split, ratings = cnn_inputs(folder)
    boundary = read(folder / "boundary_verification.json")
    if boundary.get("status") != "PASS" or boundary.get("protocol") != source["protocol"]:
        raise ValueError("Strict CNN patient-boundary verification is missing")
    if (boundary.get("fit_clear_nodules") != 740 or
            boundary.get("test_clear_nodules") != 462 or
            boundary.get("test_patients") != 195):
        raise ValueError("Strict CNN boundary counts changed")
    return folder, split, ratings


def lidc_rows(folder: Path, split: dict, ratings: pd.DataFrame, stage: str) -> Rows:
    if stage not in {"fit", "test"}:
        raise ValueError("stage must be fit or test")
    path = folder / "predictions" / ("inner_oof.npz" if stage == "fit" else "outer_test.npz")
    ids_expected = split["fit_ids"] if stage == "fit" else split["test_ids"]
    with np.load(path, allow_pickle=False) as z:
        ids = z["ids"].astype(str)
        patients = z["patients"].astype(str)
        concepts = tuple(map(str, z["concepts"]))
        seeds = z["seeds"]
        fold = z["fold"]
        draws = z["P_tta0"].astype(np.float64)
    _unique_ids("LIDC nodule", ids)
    if (list(ids) != ids_expected or concepts != LIDC_CONCEPTS or
            list(patients) != list(ratings.loc[ids].patient) or
            list(seeds) != list(range(5)) or
            draws.shape != (len(ids), len(concepts), 5) or
            not np.isfinite(draws).all() or np.any((draws < 0) | (draws > 1))):
        raise ValueError("LIDC CNN rows are misaligned or invalid")
    expected_fold = np.array([split["inner_fold_by_patient"].get(p, -1) for p in patients])
    if not np.array_equal(fold, expected_fold):
        raise ValueError("LIDC CNN patient folds are inconsistent")
    return Rows(ids, patients, draws, fold, concepts)


def derm_context(root: Path, encoder: str) -> tuple[pd.DataFrame, dict, np.ndarray]:
    if encoder not in {"monet", "biomedclip"}:
        raise ValueError("Derm encoder must be monet or biomedclip")
    meta = pd.read_csv(root / "meta/meta.csv")
    splits = {name: pd.read_csv(root / f"meta/{name}_indexes.csv")["indexes"].to_numpy(dtype=int)
              for name in ("train", "valid", "test")}
    if (len(meta) != 1011 or [len(splits[k]) for k in ("train", "valid", "test")] != [413, 203, 395] or
            len(set(np.concatenate(list(splits.values())))) != len(meta) or
            set(np.concatenate(list(splits.values()))) != set(range(len(meta)))):
        raise ValueError("Derm official split or metadata alignment changed")
    with np.load(root / f"results/emb_{encoder}.npz", allow_pickle=False) as z:
        image = z["img_clean_0"].astype(np.float64)
    if image.shape[0] != len(meta) or image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Derm embedding rows are invalid")
    return meta, splits, image


def derm_findings(meta: pd.DataFrame) -> np.ndarray:
    pg = meta.pigmentation
    values = pd.DataFrame({
        "atypical_pn": meta.pigment_network == "atypical",
        "typical_pn": meta.pigment_network == "typical",
        "irregular_streaks": meta.streaks == "irregular",
        "regular_streaks": meta.streaks == "regular",
        "irregular_pigm": pg.isin(["diffuse irregular", "localized irregular"]),
        "regular_pigm": pg.isin(["diffuse regular", "localized regular"]),
        "regression": meta.regression_structures != "absent",
        "irregular_dg": meta.dots_and_globules == "irregular",
        "regular_dg": meta.dots_and_globules == "regular",
        "bwv": meta.blue_whitish_veil == "present",
        "arborizing": meta.vascular_structures == "arborizing",
        "atypical_vessels": meta.vascular_structures.isin(["dotted", "linear irregular"]),
        "comma": meta.vascular_structures == "comma",
        "hairpin": meta.vascular_structures == "hairpin",
        "vessels_regr": meta.vascular_structures == "within regression",
    })
    if tuple(values) != DERM_CONCEPTS:
        raise ValueError("Derm concept order mismatch")
    return values.to_numpy(dtype=np.float64)


def derm_diagnoses(meta: pd.DataFrame) -> np.ndarray:
    dx = meta.diagnosis
    y = pd.DataFrame({
        "MEL": dx.str.contains("melanoma"),
        "NEV": dx.str.contains("nevus"),
        "BCC": dx == "basal cell carcinoma",
        "SK": dx == "seborrheic keratosis",
    }).astype(int)
    if tuple(y) != DERM_DIAGNOSES:
        raise ValueError("Derm diagnosis order mismatch")
    return y.to_numpy(dtype=int)


def fit_derm_probes(image: np.ndarray, concepts: np.ndarray, fit_indices: np.ndarray) -> dict:
    """Exact historical 30-bootstrap finding probes; only fit indices supply labels."""
    from sklearn.linear_model import LogisticRegression

    if concepts.shape != (len(image), len(DERM_CONCEPTS)):
        raise ValueError("Derm concept labels do not align with embeddings")
    rng = np.random.default_rng(0)
    boot_indices = [rng.choice(fit_indices, len(fit_indices)) for _ in range(30)]
    coef = np.zeros((len(DERM_CONCEPTS), 30, image.shape[1]), np.float64)
    intercept = np.zeros((len(DERM_CONCEPTS), 30), np.float64)
    active = np.zeros((len(DERM_CONCEPTS), 30), bool)
    for c in range(len(DERM_CONCEPTS)):
        for b, indices in enumerate(boot_indices):
            target = concepts[indices, c]
            if target.min() == target.max():
                continue
            model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            model.fit(image[indices], target)
            coef[c, b] = model.coef_[0]
            intercept[c, b] = model.intercept_[0]
            active[c, b] = True
    return {"coef": coef, "intercept": intercept, "active": active}


def derm_probe_draws(image: np.ndarray, probes: dict) -> np.ndarray:
    from scipy.special import expit

    coef, intercept, active = probes["coef"], probes["intercept"], probes["active"]
    if (coef.ndim != 3 or coef.shape[0:2] != (len(DERM_CONCEPTS), 30) or
            image.ndim != 2 or coef.shape[2] != image.shape[1] or
            intercept.shape != active.shape or intercept.shape != coef.shape[:2]):
        raise ValueError("Derm probe tensor shapes are invalid")
    scores = np.einsum("nd,cbd->ncb", image, coef) + intercept[None, :, :]
    draws = expit(scores)
    return draws * active[None, :, :]
