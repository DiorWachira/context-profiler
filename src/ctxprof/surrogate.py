"""Sparse linear surrogate: fit mask -> score with Lasso; the weights are the attributions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score


@dataclass
class SurrogateFit:
    weights: np.ndarray
    intercept: float
    alpha: float
    r2_heldout: float | None
    ci_lo: np.ndarray
    ci_hi: np.ndarray


def fit_surrogate(masks: np.ndarray, scores: np.ndarray, *, holdout: float = 0.25, n_boot: int = 50,
                  cv: int = 4, seed: int = 0) -> SurrogateFit:
    """Fit a Lasso with alpha chosen by cross-validation.

    Fit quality is reported as R² on held-out masks (fit on the rest), so a poor linear fit is
    visible rather than hidden. Final weights are refit on all masks with the chosen alpha;
    95% intervals come from a bootstrap over masks at that alpha.
    """
    X = masks.astype(float)
    y = np.asarray(scores, dtype=float)
    n, k = X.shape
    zeros = np.zeros(k)
    if n < 4 or np.allclose(y, y[0]):
        return SurrogateFit(zeros, float(y.mean()) if n else 0.0, 0.0, None, zeros.copy(), zeros.copy())

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(2, int(round(n * holdout))) if n >= 8 else 0
    test, train = order[:n_test], order[n_test:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        folds = max(2, min(cv, len(train)))
        cv_model = LassoCV(cv=folds, random_state=seed, max_iter=50_000, alphas=60).fit(X[train], y[train])
        alpha = float(cv_model.alpha_)
        r2 = None
        if n_test and not np.allclose(y[test], y[test][0]):
            r2 = float(r2_score(y[test], cv_model.predict(X[test])))

        final = Lasso(alpha=alpha, max_iter=50_000).fit(X, y)
        boots = np.empty((n_boot, k))
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            boots[b] = Lasso(alpha=alpha, max_iter=20_000).fit(X[idx], y[idx]).coef_
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0) if n_boot else (zeros, zeros)
    return SurrogateFit(final.coef_.copy(), float(final.intercept_), alpha, r2, lo, hi)


def normalise(weights: np.ndarray, min_effect: float) -> np.ndarray:
    """Influence in [-1, 1]: weight relative to the strongest positive weight of the trace.

    ``min_effect`` floors the denominator, so a trace where nothing matters does not inflate
    noise into influence (which would hide DEAD segments).
    """
    top = max(float(weights.max(initial=0.0)), min_effect)
    return np.clip(weights / top, -1.0, 1.0)
