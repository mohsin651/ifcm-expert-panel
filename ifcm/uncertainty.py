"""Exact decision-flip probabilities for independent Bernoulli findings."""

from collections.abc import Callable

import numpy as np


def exact_flip_probabilities(
    findings: np.ndarray,
    predict: Callable[[np.ndarray], np.ndarray],
    baseline: np.ndarray | None = None,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Probability that sampling binary findings changes a classifier's decision.

    Enumerates all 2**n finding configurations once, then integrates their
    independent Bernoulli probabilities per case. This is the limit of the
    project's Bernoulli redraw estimator, without Monte Carlo seed dependence.
    The method is suitable for the 14 LIDC findings, not arbitrarily many.
    """
    x = np.asarray(findings, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] > 18:
        raise ValueError("findings must be 2D with at most 18 columns")
    if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("finding probabilities must be finite and in [0, 1]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    patterns = ((np.arange(1 << x.shape[1])[:, None] >> np.arange(x.shape[1])) & 1).astype(np.float64)
    pattern_decision = np.asarray(predict(patterns))
    if pattern_decision.shape != (len(patterns),):
        raise ValueError("predict must return one decision per pattern")
    reference = np.asarray(predict(x) if baseline is None else baseline)
    if reference.shape != (len(x),):
        raise ValueError("baseline must contain one decision per case")

    result = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        block = x[start:stop]
        # log(0) is avoided only inside the matrix multiply. At exact 0/1
        # endpoints, the excluded patterns are masked after exponentiation.
        safe = np.clip(block, 1e-300, 1 - 1e-15)
        log_mass = patterns @ (np.log(safe) - np.log1p(-safe)).T
        log_mass += np.log1p(-safe).sum(axis=1)[None, :]
        mass = np.exp(log_mass)
        if np.any((block == 0) | (block == 1)):
            for j, probabilities in enumerate(block):
                impossible = ((probabilities == 0) & (patterns == 1)).any(axis=1)
                impossible |= ((probabilities == 1) & (patterns == 0)).any(axis=1)
                mass[impossible, j] = 0
            mass /= mass.sum(axis=0, keepdims=True)
        result[start:stop] = np.sum(mass * (pattern_decision[:, None] != reference[start:stop]), axis=0)
    return result
