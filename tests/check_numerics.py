"""Independent small numerical checks beyond the public signature tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import implementations as impl
from pipeline import Features, best_threshold, metrics


def main():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(50, 4))
    y = rng.integers(0, 2, size=50)
    w0 = rng.normal(size=4)
    saved = w0.copy()
    w, loss = impl.reg_logistic_regression(y, x, 0.03, w0, 1, 0.1)

    def objective(v):
        return impl._logloss(y, x, v) + 0.03 * np.sum(v * v)

    eps = 1e-5
    finite_difference = np.array(
        [
            (objective(w0 + eps * e) - objective(w0 - eps * e)) / (2 * eps)
            for e in np.eye(4)
        ]
    )
    np.testing.assert_allclose(w, w0 - 0.1 * finite_difference, atol=1e-8)
    np.testing.assert_array_equal(w0, saved)
    np.testing.assert_allclose(loss, impl._logloss(y, x, w))
    rank_deficient = np.column_stack([x[:, 0], x[:, 0]])
    fitted, loss = impl.least_squares(y, rank_deficient)
    np.testing.assert_allclose(
        rank_deficient.T @ (rank_deficient @ fitted - y), 0, atol=1e-10
    )
    assert np.isfinite(
        impl._logloss(np.array([0, 1]), np.eye(2), np.array([1000.0, -1000.0]))
    )
    small = np.array([[1.0, 0.0], [2.0, np.nan], [1.0, 3.0], [2.0, 2.0]])
    features = Features().fit(small, ["a", "b"])
    assert np.isfinite(features.transform(np.array([[99.0, np.nan]]))).all()
    score = np.array([0.4, 0.4, 0.8, 0.1])
    target = np.array([0, 1, 1, 0])
    threshold = best_threshold(target, score)
    assert metrics(target, score, threshold)["f1"] == max(
        metrics(target, score, t)["f1"] for t in np.unique(score)
    )
    print(
        "Numerical gradient, final loss, input immutability, singular solve, extreme logits, missing/unseen values and tied threshold checks passed."
    )


if __name__ == "__main__":
    main()
