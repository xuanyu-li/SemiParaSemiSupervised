import random

import numpy as np
import pandas as pd
import torch

from NN_estimator_rewrite import Single_trial_cross_fitting_inference

import warnings

# 1) strongest: ignore ALL warnings coming from that Lightning module
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pytorch_lightning\.trainer\.connectors\.data_connector",
)

# (optional) also cover the new package namespace just in case
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"lightning\.pytorch\.trainer\.connectors\.data_connector",
)

def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # Problem setup
    d = 5
    typenum = 2
    n = 500
    n_unlabeled = [2000]
    alpha = 0.8
    shift = False
    k_folds = 5

    # Training hyperparameters
    batch_size = 100
    nodes = [3, 128]
    lr = 0.001
    epochs = 2000
    sparse_ratio = 0.7
    verbose = False

    results = np.asarray(
        Single_trial_cross_fitting_inference(
            n=n,
            N_unlabeled=n_unlabeled,
            d=d,
            typenum=typenum,
            batch_size=batch_size,
            nodes=nodes,
            lr=lr,
            epochs=epochs,
            verbose=verbose,
            sparseRatio=sparse_ratio,
            K=k_folds,
            shift=shift,
            alpha=alpha,
        ),
        dtype=float,
    )

    index = [
        "deep_partially_linear",
        "linear_regression",
        "dml_supervised",
        "dml_semi_supervised_n2000",
        "traditional_nonparametric",
        "dml_oracle_m",
    ]
    summary = pd.DataFrame(results, columns=["theta_hat", "variance"], index=index)
    summary["std_error"] = np.sqrt(np.clip(summary["variance"], 0.0, None))

    print("Single-trial results")
    print(summary.round(6))


if __name__ == "__main__":
    main()
