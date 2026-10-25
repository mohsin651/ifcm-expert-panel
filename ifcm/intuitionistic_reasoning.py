"""Pure, signed intuitionistic-fuzzy reasoning primitives for Issue 2.

No data loading, calibration fitting, GPU work, or experiment execution occurs
here. The signed conflict projection is a proposed research operator, not an
established clinical model; see docs/intuitionistic_reasoning_spec.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_TOL = 1e-10


def _array(name: str, value: np.ndarray, ndim: int, lower: float, upper: float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-D array")
    if not np.isfinite(array).all() or np.any(array < lower) or np.any(array > upper):
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return array


@dataclass(frozen=True)
class IFS:
    """Atanassov triple; arrays have identical shape and sum to one."""

    mu: np.ndarray
    nu: np.ndarray
    pi: np.ndarray

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=np.float64)
        nu = np.asarray(self.nu, dtype=np.float64)
        pi = np.asarray(self.pi, dtype=np.float64)
        if (mu.shape != nu.shape or mu.shape != pi.shape or
                not mu.shape or any(size == 0 for size in mu.shape)):
            raise ValueError("IFS components must have identical nonempty shapes")
        if (not np.isfinite(mu).all() or not np.isfinite(nu).all() or
                not np.isfinite(pi).all() or
                np.any(mu < -_TOL) or np.any(nu < -_TOL) or np.any(pi < -_TOL) or
                np.any(mu > 1 + _TOL) or np.any(nu > 1 + _TOL) or
                np.any(pi > 1 + _TOL) or
                not np.allclose(mu + nu + pi, 1, rtol=0, atol=_TOL)):
            raise ValueError("IFS requires finite components in [0,1] summing to one")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "pi", pi)

    @property
    def lower(self) -> np.ndarray:
        return self.mu

    @property
    def upper(self) -> np.ndarray:
        return 1 - self.nu


@dataclass(frozen=True)
class EdgeBounds:
    """Signed panel edge intervals and their un-clipped uncertainty summary."""

    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    raw_half_width: np.ndarray
    left_width: np.ndarray
    right_width: np.ndarray


@dataclass(frozen=True)
class ReasoningResult:
    """Diagnosis IFS, its scalar readout, and intermediate evidence ranges."""

    diagnosis: IFS
    score: np.ndarray
    positive_lower: np.ndarray
    positive_upper: np.ndarray
    negative_lower: np.ndarray
    negative_upper: np.ndarray


def from_interval(lower: np.ndarray, upper: np.ndarray) -> IFS:
    """Map a membership interval [L,U] to (L,1-U,U-L)."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError("Interval endpoints must have the same shape")
    _array("lower", lo, lo.ndim, 0, 1)
    _array("upper", hi, hi.ndim, 0, 1)
    if np.any(lo > hi):
        raise ValueError("Interval lower endpoint exceeds upper endpoint")
    return IFS(lo, 1 - hi, hi - lo)


def point_ifs(value: np.ndarray) -> IFS:
    """Zero-hesitancy finding IFS: (x,1-x,0)."""
    x = np.asarray(value, dtype=np.float64)
    return from_interval(x, x)


def finding_ifs_from_draws(
    draws: np.ndarray, mean: np.ndarray | None = None,
    quantiles: tuple[float, float] = (0.1, 0.9),
) -> IFS:
    """Prespecified empirical seed envelope, not a confidence interval."""
    values = _array("draws", draws, 3, 0, 1)
    if quantiles != (0.1, 0.9):
        raise ValueError("Issue 2 fixes the empirical quantiles at (0.1, 0.9)")
    ensemble_mean = values.mean(axis=2)
    if mean is not None:
        supplied = _array("mean", mean, 2, 0, 1)
        if supplied.shape != ensemble_mean.shape or not np.allclose(
                supplied, ensemble_mean, rtol=0, atol=1e-6):
            raise ValueError("mean must equal the seed mean for aligned findings")
    lo, hi = np.quantile(values, quantiles, axis=2)
    return from_interval(np.minimum(lo, ensemble_mean),
                         np.maximum(hi, ensemble_mean))


def panel_edge_bounds(member_edges: np.ndarray, member_unsure: np.ndarray) -> EdgeBounds:
    """Clip mean ± (mean unsure + sample SD) into the signed edge domain."""
    edges = _array("member_edges", member_edges, 3, -1, 1)
    unsure = _array("member_unsure", member_unsure, 3, 0, 1)
    if edges.shape != unsure.shape or edges.shape[0] < 2:
        raise ValueError("Panel requires >=2 aligned edge/unsure members")
    mean = edges.mean(axis=0)
    raw = unsure.mean(axis=0) + edges.std(axis=0, ddof=1)
    lo = np.clip(mean - raw, -1, 1)
    hi = np.clip(mean + raw, -1, 1)
    return EdgeBounds(mean, lo, hi, raw, mean - lo, hi - mean)


def _probabilistic_or(evidence: np.ndarray) -> np.ndarray:
    # Stable evaluation of 1 - product(1 - evidence), including exact evidence=1.
    with np.errstate(divide="ignore"):
        return -np.expm1(np.log1p(-evidence).sum(axis=1))


def reason_b2(findings: IFS, edge_lower: np.ndarray, edge_upper: np.ndarray) -> ReasoningResult:
    """Propagate interval-valued findings and signed edge intervals to diagnoses.

    Findings have shape (N,C), edges (C,D), and outputs (N,D). A missing
    edge is an error, not a neutral zero edge. No labels are accepted.
    """
    if not isinstance(findings, IFS) or findings.mu.ndim != 2:
        raise ValueError("findings must be a 2-D IFS with shape (N,C)")
    xlo = _array("finding lower", findings.lower, 2, 0, 1)
    xhi = _array("finding upper", findings.upper, 2, 0, 1)
    wlo = _array("edge_lower", edge_lower, 2, -1, 1)
    whi = _array("edge_upper", edge_upper, 2, -1, 1)
    if wlo.shape != whi.shape or xlo.shape[1] != wlo.shape[0]:
        raise ValueError("Finding and edge dimensions do not align")
    if np.any(wlo > whi):
        raise ValueError("Edge lower endpoint exceeds upper endpoint")
    positive_lo = xlo[:, :, None] * np.maximum(wlo, 0)[None, :, :]
    positive_hi = xhi[:, :, None] * np.maximum(whi, 0)[None, :, :]
    negative_lo = xlo[:, :, None] * np.maximum(-whi, 0)[None, :, :]
    negative_hi = xhi[:, :, None] * np.maximum(-wlo, 0)[None, :, :]
    p_lo, p_hi = _probabilistic_or(positive_lo), _probabilistic_or(positive_hi)
    n_lo, n_hi = _probabilistic_or(negative_lo), _probabilistic_or(negative_hi)
    mu = p_lo * (1 - n_hi)
    nu = n_lo * (1 - p_hi)
    diagnosis = IFS(mu, nu, 1 - mu - nu)
    return ReasoningResult(diagnosis, mu - nu, p_lo, p_hi, n_lo, n_hi)


def reason_b1(findings: IFS, edge_mean: np.ndarray) -> ReasoningResult:
    """The same signed operator with point-valued LLM knowledge edges."""
    return reason_b2(findings, edge_mean, edge_mean)
