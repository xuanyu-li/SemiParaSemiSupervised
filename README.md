# Semi-parametric Semi-supervised Longitudinal Estimation

This repository contains simulation code for estimating a partially linear treatment effect with supervised, semi-supervised, and nonparametric nuisance estimators on synthetic data.

## Repository layout

- `example.py`: minimal script showing one end-to-end trial.
- `NN_estimator_rewrite.py`: main estimation pipeline, including cross-fitting and inference.
- `data_gen.py`: synthetic data generators used in the experiments.
- `DPLAux.py`: sparse neural-network architectures.
- `NNAux.py`: PyTorch Lightning training modules and prediction helpers.
- `traditional_nonparametric.py`: kernel-based nonparametric baseline.
- `mimic-analysis/`: data-preparation scripts and notes for the MIMIC-based workflow.

## Requirements

The code uses Python 3 and the following libraries:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `torch`
- `pytorch-lightning`
- `PyWavelets`

Install them with your preferred environment manager. For example:

```bash
pip install numpy pandas scipy scikit-learn statsmodels torch pytorch-lightning PyWavelets
```

## Run a single trial

The simplest entry point is:

```bash
python3 example.py
```

This script:

1. fixes random seeds for reproducibility,
2. generates one supervised dataset and one unlabeled dataset,
3. runs `Single_trial_cross_fitting_inference(...)`, and
4. prints a table with the estimated coefficient, estimated variance, and standard error for each method.

## Notes on the example

- `n` is the number of labeled samples.
- `n_unlabeled` controls the unlabeled sample sizes used by the semi-supervised estimator.
- `typenum` selects the synthetic data-generating mechanism (`1`, `2`, or `3`).
- `nodes`, `epochs`, `batch_size`, and `lr` control the neural-network training routine.
- The default settings in `example.py` match the previous simulation script, except the script runs only one trial instead of a Monte Carlo loop.

## Main output

`example.py` prints one row per estimator. The default labels are:

- `deep_partially_linear`
- `linear_regression`
- `dml_supervised`
- `dml_semi_supervised_n2000`
- `dml_semi_supervised_n5000`
- `traditional_nonparametric`
- `dml_oracle_m`

For each row:

- `theta_hat` is the point estimate,
- `variance` is the estimated asymptotic variance divided by sample size as returned by the code, and
- `std_error` is the square root of that variance estimate.
