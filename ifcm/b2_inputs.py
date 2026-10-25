"""Recover aligned member-level signed weights and unsure mass from saved votes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ifcm.b1_inputs import Panel, _style_paths, _style_weights, load_panel
from ifcm.intuitionistic_reasoning import EdgeBounds, panel_edge_bounds


def member_bounds(root: Path, dataset: str) -> tuple[Panel, EdgeBounds, np.ndarray, np.ndarray]:
    panel = load_panel(root, dataset)
    folder = root / ("results_lidc" if dataset == "lidc" else "results")
    models = tuple(sorted({member.rsplit("|", 1)[0] for member in panel.members}))
    edges = np.empty((len(panel.members), len(panel.concepts), len(panel.diagnoses)))
    unsure = np.empty_like(edges)
    expected = pd.MultiIndex.from_product([panel.concepts, panel.diagnoses],
                                           names=["concept", "diag"])
    for i, member in enumerate(panel.members):
        model, style = member.rsplit("|", 1)
        paths = _style_paths(folder, style)
        weights = _style_weights(paths, style, models, panel.concepts, panel.diagnoses)
        edges[i] = weights.loc[pd.MultiIndex.from_product(
            [[member], panel.concepts, panel.diagnoses],
            names=["model", "concept", "diag"])].to_numpy().reshape(edges.shape[1:])
        votes = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
        votes = votes.loc[votes.model == model]
        if "p_unsure" not in votes or not np.isfinite(votes.p_unsure).all():
            raise ValueError("Missing or invalid member unsure probabilities")
        if style == "choice":
            q = votes.groupby("concept").p_unsure.mean().reindex(panel.concepts)
            unsure[i] = np.repeat(q.to_numpy()[:, None], len(panel.diagnoses), axis=1)
        else:
            q = votes.groupby(["concept", "diag"]).p_unsure.mean().reindex(expected)
            unsure[i] = q.to_numpy().reshape(edges.shape[1:])
    if (not np.isfinite(unsure).all() or np.any((unsure < 0) | (unsure > 1)) or
            not np.allclose(edges.mean(axis=0), panel.mean, atol=1e-12, rtol=0)):
        raise ValueError("Member uncertainty or mean edges do not match B1")
    return panel, panel_edge_bounds(edges, unsure), edges, unsure
