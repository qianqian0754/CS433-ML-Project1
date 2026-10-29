# CS-433 Project 1: heart disease classification

Here is the group project of Zeming Qu, Yunhan Zhang and Qianqian Li.

## Files

- `requirements.txt`: installation list
- `implementations.py`: six required estimators, with exact assignment signatures.
- `pipeline.py`: train-only preprocessing and minibatch Adam logistic regression.
- `run.py`: complete deterministic training and prediction entry point.
- `helpers.py`: supplied course helpers (formatting only).
- `results/submission.csv`: 109,379 predictions, original test IDs, labels -1/+1.
- `results/model.pkl`: fitted preprocessing, weights and decision threshold.
- `results/metrics.json`: all measured tuning and holdout results.
- `report/main.tex`, `report/main.pdf`: two-page report and editable source.

## Reproduce

Runtime dependencies: Python 3.10+ and NumPy only. ReportLab is used only to draw the report figure, separately from training and inference. Tests additionally need pytest, GitPython and Black as required by the supplied grader; these are not model dependencies.

Extract `dataset_to_release.zip` to a directory containing `x_train.csv`, `y_train.csv`, `x_test.csv`. Run from this project directory:

```sh
python run.py --data /path/to/dataset --output results
```

On this machine the supplied ZIP is in `C:/Users/Zemin/Downloads/ML_course-main/ML_course-main/projects/project1/data/`. The extracted data used for this run is in the chat workspace `work/data/dataset`. Data is excluded from the deliverable to avoid duplicating the supplied dataset.

To reproduce predictions with the supplied fitted weights:

```sh
python run.py --data /path/to/dataset --output results --predict-only
```

Only load the included trusted pickle; Python pickle is executable serialization. A local `data_cache.npz` is created to accelerate subsequent runs. Delete it when changing input CSV files. The cache is excluded from the delivered archive. Retraining fixes RNG seed 433; numerical library versions can cause tiny floating-point differences. Recorded run uses Python 3.12 and NumPy; exact versions are in `results/verification.json`.

## Method and evaluation

Stratified 60/20/20 split: 196,881 training rows, 65,627 tuning rows, 65,627 untouched holdout rows. All preprocessing is fitted on training rows. Feature selection and model/threshold choice use only training and tuning sets. Holdout results are computed after selection, with no subsequent tuning. This is a single split evaluation, not cross-validation.

Drop explicit survey metadata/identifiers and columns with <20% observed values or fewer than two distinct observed values. Numeric columns are median-imputed, clipped to training 1st/99th percentiles, standardized, and given a missing indicator. For the category variant, columns with 2-20 observed distinct values receive one-hot coding with the first category dropped and a missing indicator. Unseen categories map to the baseline representation. Encoded refusal/unknown values remain explicit levels when categorical, otherwise are clipped as numeric values; no undocumented medical recoding is assumed.

Compare ridge (lambda=0.001), numeric logistic (lambda=0.0001), categorical logistic (lambda=0.0001 and 0.001). Ridge minimizes half MSE + lambda times squared norm. Logistic uses unweighted mean binary cross-entropy + lambda times squared norm, excluding the intercept in the additional Adam optimizer. Adam: 16 epochs, batch 4096, beta1=0.9, beta2=0.999, epsilon=1e-8, learning rate 0.01/(1+epoch/10). Thresholds maximize tuning F1 by exact sweep over distinct scores.

Selected categorical logistic lambda=0.0001, threshold=0.1888708323, 717 training design columns. Tuning F1=0.427665; untouched holdout F1=0.419950, accuracy=0.865848, precision=0.339657, recall=0.549957. Holdout confusion counts: TP=3187, FP=6196, FN=2608, TN=53636. Refit preprocessing and the selected model on all labeled rows, retaining the tuning threshold, to generate the final CSV. Holdout metrics evaluate the training-partition fit, not the final all-data model. Test labels and leaderboard scores are unavailable.

## Required estimator conventions

All six methods return final weights and their corresponding unpenalized loss. MSE is half the mean squared residual; logistic loss is mean binary cross-entropy for 0/1 labels. Logistic L2 gradient is 2*lambda*w. Ridge normal equations use 2*N*lambda*I. SGD uses one uniformly drawn observation per iteration. Least squares solves normal equations, with pseudoinverse fallback for singular matrices; it never calls `numpy.linalg.lstsq`. Initial weights are copied.

## Scope and submission

The dataset is a cross-sectional survey with self-reported diagnosed disease; this experiment measures retrospective label classification, not future clinical risk. Some retained health or rehabilitation responses may be consequences of diagnosis. A prospective application needs a feature and temporal leakage audit. Numeric category inference is heuristic, and only one split and four model configurations were evaluated.
