"""Train-only feature processing and NumPy logistic optimization."""

import numpy as np
from implementations import _sigmoid


class Features:
    """Median imputation, clipping, missing flags, optional category expansion."""

    def __init__(self, categorical=True):
        self.categorical = categorical

    def fit(self, x, names):
        self.specs = []
        # Identifiers, collection metadata and survey weights are not predictors.
        excluded = {
            "_STATE",
            "FMONTH",
            "IDATE",
            "IMONTH",
            "IDAY",
            "IYEAR",
            "DISPCODE",
            "SEQNO",
            "_PSU",
            "_STSTR",
            "_STRWT",
            "_RAWRAKE",
            "_WT2RAKE",
            "_CLLCPWT",
            "_DUALCOR",
            "_LLCPWT",
        }
        for j, name in enumerate(names):
            v = x[:, j]
            valid = v[np.isfinite(v)]
            if name in excluded or len(valid) < 0.2 * len(v):
                continue
            unique = np.unique(valid)
            if len(unique) < 2:
                continue
            if self.categorical and len(unique) <= 20:
                self.specs.append((j, "cat", unique))
            else:
                lo, hi = np.quantile(valid, [0.01, 0.99])
                med = np.median(valid)
                filled = np.clip(np.where(np.isfinite(v), v, med), lo, hi)
                self.specs.append(
                    (j, "num", (med, lo, hi, filled.mean(), max(filled.std(), 1e-6)))
                )
        return self

    def transform(self, x):
        cols = [np.ones(len(x), dtype=np.float32)]
        for j, kind, values in self.specs:
            v = x[:, j]
            if kind == "cat":
                cols.extend((v == level).astype(np.float32) for level in values[1:])
            else:
                med, lo, hi, mean, std = values
                v = (np.clip(np.where(np.isfinite(v), v, med), lo, hi) - mean) / std
                cols.append(v.astype(np.float32))
            cols.append((~np.isfinite(x[:, j])).astype(np.float32))
        return np.column_stack(cols)


def train_logistic(x, y, lambda_, epochs=16, seed=433):
    """Minibatch Adam; unweighted BCE, L2 excludes the intercept."""
    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1], dtype=np.float32)
    w[0] = np.log(y.mean() / (1 - y.mean()))
    m, v = np.zeros_like(w), np.zeros_like(w)
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 4096):
            ids = order[start : start + 4096]
            xb = x[ids]
            g = xb.T @ (_sigmoid(xb @ w) - y[ids]) / len(ids)
            penalty = 2 * lambda_ * w
            penalty[0] = 0
            g += penalty
            step += 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            w -= (
                (0.01 / (1 + epoch / 10))
                * (m / (1 - 0.9**step))
                / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
            )
    return w


def metrics(y, score, threshold):
    """Accuracy, precision, recall, F1 and confusion counts."""
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return dict(
        accuracy=(tp + tn) / len(y),
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        f1=2 * tp / max(2 * tp + fp + fn, 1),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def best_threshold(y, score):
    """Exact F1 sweep over distinct prediction scores on tuning data only."""
    order = np.argsort(-score, kind="stable")
    s, target = score[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.cumsum(target)[ends]
    f1 = 2 * tp / (ends + 1 + target.sum())
    return float(s[ends[np.argmax(f1)]])
