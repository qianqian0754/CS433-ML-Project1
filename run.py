"""Reproduce training, validation and the final competition CSV."""

import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from helpers import load_csv_data, create_csv_submission
from implementations import ridge_regression, _sigmoid
from pipeline import Features, train_logistic, metrics, best_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "data_cache.npz"
    names = (args.data / "x_train.csv").open().readline().strip().split(",")[1:]
    if cache.exists():
        a = np.load(cache)
        x, xt, y, ids = (a[k] for k in ("x", "xt", "y", "ids"))
    else:
        print("Loading CSV files", flush=True)
        x, xt, labels, _, ids = load_csv_data(str(args.data))
        x, xt = x.astype(np.float32), xt.astype(np.float32)
        y = (labels == 1).astype(np.float32)
        np.savez(cache, x=x, xt=xt, y=y, ids=ids)
    if args.predict_only:
        with (args.output / "model.pkl").open("rb") as f:
            features, w, threshold = pickle.load(f)
    else:
        rng = np.random.default_rng(433)
        splits = [[], [], []]
        for label in (0, 1):
            idx = rng.permutation(np.flatnonzero(y == label))
            parts = np.split(idx, [int(0.6 * len(idx)), int(0.8 * len(idx))])
            for dest, part in zip(splits, parts):
                dest.extend(part.tolist())
        train, tune, test = [rng.permutation(s) for s in splits]
        results = []
        configs = [
            (False, "ridge", 1e-3),
            (False, "logistic", 1e-4),
            (True, "logistic", 1e-4),
            (True, "logistic", 1e-3),
        ]
        scores = []
        for categorical, model, lam in configs:
            features = Features(categorical).fit(x[train], names)
            a, b = features.transform(x[train]), features.transform(x[tune])
            print(
                f"Training {model}, categorical={categorical}, lambda={lam}, features={a.shape[1]}",
                flush=True,
            )
            if model == "ridge":
                w, _ = ridge_regression(y[train], a, lam)
                score = b @ w
            else:
                w = train_logistic(a, y[train], lam)
                score = _sigmoid(b @ w)
            threshold = best_threshold(y[tune], score)
            results.append(
                dict(
                    model=model,
                    categorical=categorical,
                    lambda_=lam,
                    dimensions=a.shape[1],
                    threshold=threshold,
                    tuning=metrics(y[tune], score, threshold),
                    tuning_default=metrics(y[tune], score, 0.5),
                )
            )
            scores.append((features, w))
            print(results[-1], flush=True)
        selected = int(np.argmax([r["tuning"]["f1"] for r in results]))
        # Freeze selection and threshold before opening the untouched test partition.
        threshold = results[selected]["threshold"]
        for r, (features, w) in zip(results, scores):
            z = features.transform(x[test]) @ w
            score = _sigmoid(z) if r["model"] == "logistic" else z
            r["holdout"] = metrics(y[test], score, r["threshold"])
            r["holdout_default"] = metrics(y[test], score, 0.5)
        summary = dict(
            seed=433,
            rows=len(y),
            test_rows=len(xt),
            raw_features=x.shape[1],
            positive_rate=float(y.mean()),
            missing_rate=float(np.isnan(x).mean()),
            split_sizes=[len(s) for s in splits],
            selected=selected,
            experiments=results,
            majority_holdout=metrics(y[test], np.zeros(len(test)), 0.5),
        )
        (args.output / "metrics.json").write_text(json.dumps(summary, indent=2))
        np.savez(
            args.output / "split_indices.npz", train=train, tune=tune, holdout=test
        )
        r = results[selected]
        features = Features(r["categorical"]).fit(x, names)
        a = features.transform(x)
        print("Refitting selected configuration on all labeled rows", flush=True)
        if r["model"] == "logistic":
            w = train_logistic(a, y, r["lambda_"])
        else:
            w, _ = ridge_regression(y, a, r["lambda_"])
        with (args.output / "model.pkl").open("wb") as f:
            pickle.dump((features, w, threshold), f)
    z = features.transform(xt) @ w
    # Determine score family from the stored experiment selection.
    summary = json.loads((args.output / "metrics.json").read_text())
    model = summary["experiments"][summary["selected"]]["model"]
    score = _sigmoid(z) if model == "logistic" else z
    prediction = np.where(score >= threshold, 1, -1)
    create_csv_submission(ids, prediction, str(args.output / "submission.csv"))
    print(
        f"Saved {len(ids)} predictions to {args.output / 'submission.csv'}", flush=True
    )


if __name__ == "__main__":
    main()
