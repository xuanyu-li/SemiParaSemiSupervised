from __future__ import annotations

from itertools import product
import numpy as np
import scipy
import torch
import pandas as pd
import shutil
import uuid
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from DPLAux import DPLNetSparse, mNet
import torch.nn as nn
from data_gen import *
from sklearn.model_selection import KFold
import statsmodels.api as sm
from traditional_nonparametric import *
from NNAux import *
import statsmodels.api as sm   

import logging

# Lightning logs (2.6.0)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# Some submodules still log under this name internally
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Silence repetitive Lightning/PyTorch runtime warnings during repeated training.
warnings.filterwarnings(
    "ignore",
    message=r".*does not have many workers which may be a bottleneck.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*You are using `torch\.load` with `weights_only=False`.*",
)


def _single_process_trainer_kwargs():
    return {
        "accelerator": "auto",
        "devices": 1,
        "num_nodes": 1,
    }


# training Deep Partially Linear Regression (DPLR)
def g_deepfit(
    df_train,
    x_train,
    y_train,
    dim_lin,
    dim_nonpar,
    sparseRatio,
    val_data,
    nodes,
    batch_size,
    lr,
    epochs,
    verbose,
    weight_decay=0.0,
    checkpoint_dir="checkpoints_g_deepfit",
):
    # Create a LinearRegression model
    model = LinearRegression()

    # Train the model
    model.fit(df_train.drop(columns=['Y']), df_train['Y'])
    paramsLin = model.coef_[-1]

    coef_init_weight = torch.tensor(paramsLin, dtype=y_train.dtype)
    net = DPLNetSparse(dim_lin, dim_nonpar, coef_init_weight, nodes, sparseRatio)
    pl_module = DPLLightning(net, lr, weight_decay=weight_decay)
    train_ds = TupleTensorDataset(x_train[0], x_train[1], y_train)
    val_ds = TupleTensorDataset(val_data[0][0], val_data[0][1], val_data[1])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=5 * batch_size, shuffle=False)

    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_checkpoint_dir = checkpoint_root / f"run_{uuid.uuid4().hex[:8]}"
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(run_checkpoint_dir),
        filename="g-deepfit-best-{epoch:02d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=15),
        checkpoint_cb,
    ]
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        callbacks=callbacks,
        enable_progress_bar=bool(verbose),
        deterministic=True,
        **_single_process_trainer_kwargs(),
    )
    trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_ckpt_path = checkpoint_cb.best_model_path
    if not best_ckpt_path:
        raise RuntimeError("No checkpoint was saved in g_deepfit.")

    best_net = DPLNetSparse(dim_lin, dim_nonpar, coef_init_weight, nodes, sparseRatio)
    best_module = DPLLightning.load_from_checkpoint(
        best_ckpt_path, net=best_net, lr=lr, weight_decay=weight_decay, map_location="cpu"
    )
    deepCoefEst = (
        best_module.net.linLinear.weight.detach().to("cpu").numpy()[0][0]
    )
    shutil.rmtree(run_checkpoint_dir, ignore_errors=True)

    return deepCoefEst, best_module.net, paramsLin


def _fit_m_network(
    x_train,
    y_train,
    val_data,
    sparseRatio,
    nodes,
    batch_size,
    lr,
    epochs,
    verbose,
    weight_decay=0.0,
    patience=15,
    min_delta=0.0,
    checkpoint_dir="checkpoints_mnet_fit",
):
    dim_nonpar = x_train.size()[1]
    net = mNet(dim_nonpar, nodes, sparseRatio=sparseRatio)
    pl_module = MNetLightning(net, lr, weight_decay=weight_decay)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(val_data[0], val_data[1])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=5 * batch_size, shuffle=False)

    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_checkpoint_dir = checkpoint_root / f"run_{uuid.uuid4().hex[:8]}"
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(run_checkpoint_dir),
        filename="m-net-best-{epoch:02d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=patience, min_delta=min_delta
        ),
        checkpoint_cb
    ]

    trainer = pl.Trainer(
        max_epochs=epochs,          # still acts as a safety cap
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        callbacks=callbacks,
        enable_progress_bar=bool(verbose),
        deterministic=True,
        log_every_n_steps=50,
        **_single_process_trainer_kwargs(),
    )

    trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt_path = checkpoint_cb.best_model_path
    if not best_ckpt_path:
        raise RuntimeError("No checkpoint was saved in neural_network_fit.")

    best_net = mNet(dim_nonpar, nodes, sparseRatio=sparseRatio)
    best_module = MNetLightning.load_from_checkpoint(
        best_ckpt_path, net=best_net, lr=lr, weight_decay=weight_decay, map_location="cpu"
    )
    best_val_loss = float(checkpoint_cb.best_model_score.item())
    shutil.rmtree(run_checkpoint_dir, ignore_errors=True)
    return best_module.net, best_val_loss


def neural_network_fit(
    x_train,
    y_train,
    val_data,
    sparseRatio,
    nodes,
    batch_size,
    lr,
    epochs,
    verbose,
    weight_decay=0.0,
    patience=15,
    min_delta=0.0,
    checkpoint_dir="checkpoints_mnet_fit",
):
    model, _ = _fit_m_network(
        x_train=x_train,
        y_train=y_train,
        val_data=val_data,
        sparseRatio=sparseRatio,
        nodes=nodes,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        verbose=verbose,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        checkpoint_dir=checkpoint_dir,
    )
    return model


def neural_network_fit_with_val_loss(
    x_train,
    y_train,
    val_data,
    sparseRatio,
    nodes,
    batch_size,
    lr,
    epochs,
    verbose,
    weight_decay=0.0,
    patience=15,
    min_delta=0.0,
    checkpoint_dir="checkpoints_mnet_fit",
):
    return _fit_m_network(
        x_train=x_train,
        y_train=y_train,
        val_data=val_data,
        sparseRatio=sparseRatio,
        nodes=nodes,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        verbose=verbose,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        checkpoint_dir=checkpoint_dir,
    )

def prepare_tensors(df: pd.DataFrame):
    """
    Convert a dataframe into (x_lin, x_nonpar), y tensors for model input.
    If 'Y' is not present in df, return None for y.
    """
    has_y = 'Y' in df.columns
    # Separate features
    if has_y:
        x_nonpar = torch.tensor(df.drop(columns=['Z', 'Y']).to_numpy(), dtype=torch.float32)
    else:
        x_nonpar = torch.tensor(df.drop(columns=['Z']).to_numpy(), dtype=torch.float32)
    x_lin = torch.tensor(df['Z'].values, dtype=torch.float32).view(len(df), 1)
    # Handle Y if present
    y = None
    if has_y:
        y = torch.tensor(df['Y'].values, dtype=torch.float32).view(len(df), 1)
    return (x_lin, x_nonpar), y

def supervised_neural_network(df, kfsplit, sparseRatio, nodes, batch_size, lr, weight_decay=0.0, epochs=2000, verbose=False):
    ghat = np.zeros(len(df))
    mhat = np.zeros(len(df))
    for train_index, test_index in kfsplit:
        # g estimation
        df_kc, df_k = df.iloc[train_index], df.iloc[test_index]
        df_kc_train, df_kc_val = train_test_split(df_kc, test_size=0.2)
        x_train, y_train = prepare_tensors(df_kc_train)
        x_val, y_val = prepare_tensors(df_kc_val)
        val_data = (x_val, y_val)
        
        x_test, y_test = prepare_tensors(df_k)
        test_data = (x_test, y_test)
        dim_lin = 1
        dim_nonpar = x_train[1].size()[1]

        deepCoefEst, model, paramsLin = g_deepfit(df_kc, x_train, y_train, dim_lin, dim_nonpar,
                                                 sparseRatio, val_data, nodes, batch_size, lr, epochs, verbose, weight_decay=weight_decay)
        preds = predict_net_dpl(model, x_test[0], x_test[1], batch_size).to("cpu") - x_test[0] * deepCoefEst
        ghat[test_index] = preds.detach().reshape(-1)
        # m estimation
        z_train = x_train[0].view(x_train[0].size()[0],1)
        z_val = x_val[0].view(x_val[0].size()[0],1)
        val_data_Z = (x_val[1], z_val)
        model_m = neural_network_fit(x_train[1], z_train, val_data_Z, sparseRatio, nodes, batch_size, lr, epochs, verbose, weight_decay=weight_decay) 
        preds_m = predict_net_m(model_m, x_test[1], batch_size).to("cpu").numpy()
        mhat[test_index] = preds_m.reshape(-1)

    return ghat, mhat

def semi_supervised_neural_network_unlabeled_only(df, kfsplit, n_unlabel, df_unlabel_all, sparseRatio, nodes, batch_size, lr, epochs, verbose):
    mhat = np.zeros((len(df), len(n_unlabel) if isinstance(n_unlabel, list) else 1))
    # normalize n_unlabel to an iterable
    n_unlabel_list = [n_unlabel] if isinstance(n_unlabel, int) else n_unlabel

    for train_idx, test_idx in kfsplit:
        df_test = df.iloc[test_idx]
        x_test, _ = prepare_tensors(df_test)

        for n_u in n_unlabel_list:
            nodes[0] = 5 if n_u > 2000 else 4  # adjust input layer size based on n_unlabel
            df_ss = df_unlabel_all[:n_u]
            df_ss_tr, df_ss_va = train_test_split(df_ss, test_size=0.2)

            x_tr, _ = prepare_tensors(df_ss_tr)
            x_va, _ = prepare_tensors(df_ss_va)

            model_ss_m = neural_network_fit(x_tr[1], x_tr[0].view(x_tr[0].size()[0],1), (x_va[1], x_va[0].view(x_va[0].size()[0],1)), sparseRatio, nodes, batch_size, lr, epochs, verbose)

            preds_ss = predict_net_m(model_ss_m, x_test[1], batch_size).to("cpu").numpy().reshape(-1)
            mhat[test_idx, n_unlabel_list.index(n_u)] = preds_ss

    return mhat


def select_semi_supervised_unlabeled_only_hyperparameters(
    df_unlabel_all,
    n_unlabel_grid,
    layer_grid,
    width_grid,
    sparseRatio,
    batch_size,
    lr,
    epochs,
    verbose=False,
    weight_decay=0.0,
    patience=15,
    min_delta=0.0,
    val_size=0.2,
    random_state=42,
    checkpoint_dir="checkpoints_mnet_fit",
):
    """
    Fit one m-network for each candidate combination of:
    - unlabeled sample size (`n_unlabel_grid`)
    - number of hidden layers (`layer_grid`)
    - hidden width (`width_grid`)

    The score for each candidate is the best validation loss achieved under
    early stopping. The candidate with the smallest validation loss is returned.
    """
    n_unlabel_list = [n_unlabel_grid] if np.isscalar(n_unlabel_grid) else list(n_unlabel_grid)
    layer_list = [layer_grid] if np.isscalar(layer_grid) else list(layer_grid)
    width_list = [width_grid] if np.isscalar(width_grid) else list(width_grid)

    if len(n_unlabel_list) == 0 or len(layer_list) == 0 or len(width_list) == 0:
        raise ValueError("All hyperparameter grids must contain at least one value.")
    if len(df_unlabel_all) == 0:
        raise ValueError("df_unlabel_all must contain at least one unlabeled observation.")

    tuning_records = []
    best_model = None
    best_config = None
    best_val_loss = np.inf

    for n_u, num_layers, width in product(n_unlabel_list, layer_list, width_list):
        n_u = int(n_u)
        num_layers = int(num_layers)
        width = int(width)

        if n_u <= 1:
            raise ValueError("Each n_unlabel candidate must be at least 2.")
        if n_u > len(df_unlabel_all):
            raise ValueError(
                f"n_unlabel candidate {n_u} exceeds the available unlabeled sample size {len(df_unlabel_all)}."
            )
        if num_layers <= 0 or width <= 0:
            raise ValueError("Layer and width candidates must be strictly positive.")

        df_ss = df_unlabel_all.iloc[:n_u].reset_index(drop=True)
        df_ss_tr, df_ss_va = train_test_split(
            df_ss,
            test_size=val_size,
            random_state=random_state,
            shuffle=True,
        )

        x_tr, _ = prepare_tensors(df_ss_tr)
        x_va, _ = prepare_tensors(df_ss_va)
        val_data = (x_va[1], x_va[0].view(x_va[0].size()[0], 1))
        model_ss_m, candidate_val_loss = neural_network_fit_with_val_loss(
            x_train=x_tr[1],
            y_train=x_tr[0].view(x_tr[0].size()[0], 1),
            val_data=val_data,
            sparseRatio=sparseRatio,
            nodes=[num_layers, width],
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            verbose=verbose,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            checkpoint_dir=checkpoint_dir,
        )

        record = {
            "n_unlabel": n_u,
            "num_hidden_layers": num_layers,
            "nodes_per_layer": width,
            "best_val_loss": float(candidate_val_loss),
        }
        tuning_records.append(record)

        if candidate_val_loss < best_val_loss:
            best_val_loss = float(candidate_val_loss)
            best_model = model_ss_m
            best_config = record.copy()

    if best_model is None or best_config is None:
        raise RuntimeError("Hyperparameter search did not produce a valid model.")

    return best_model, best_config, pd.DataFrame(tuning_records)


def semi_supervised_neural_network_unlabeled_only_tuned(
    df,
    kfsplit,
    n_unlabel_grid,
    df_unlabel_all,
    sparseRatio,
    layer_grid,
    width_grid,
    batch_size,
    lr,
    epochs,
    verbose=False,
    weight_decay=0.0,
    patience=15,
    min_delta=0.0,
    val_size=0.2,
    random_state=42,
    checkpoint_dir="checkpoints_mnet_fit",
    return_tuning_results=False,
):
    """
    Cross-fitted semi-supervised m-hat using unlabeled data only, with
    hyperparameter selection over unlabeled sample size, hidden layers, and width.

    For each fold:
    1. train every candidate on an unlabeled train split,
    2. record the best early-stopped validation loss,
    3. keep the candidate with the smallest validation loss,
    4. use the selected model to predict m(W) on the held-out labeled fold.
    """
    mhat = np.zeros(len(df))
    tuning_results = []
    best_configs = []

    for fold_idx, (_, test_idx) in enumerate(kfsplit):
        df_test = df.iloc[test_idx]
        x_test, _ = prepare_tensors(df_test)

        fold_seed = None if random_state is None else int(random_state) + fold_idx
        best_model, best_config, fold_tuning_df = select_semi_supervised_unlabeled_only_hyperparameters(
            df_unlabel_all=df_unlabel_all,
            n_unlabel_grid=n_unlabel_grid,
            layer_grid=layer_grid,
            width_grid=width_grid,
            sparseRatio=sparseRatio,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            verbose=verbose,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            val_size=val_size,
            random_state=fold_seed,
            checkpoint_dir=checkpoint_dir,
        )

        preds_ss = predict_net_m(best_model, x_test[1], batch_size).to("cpu").numpy().reshape(-1)
        mhat[test_idx] = preds_ss

        fold_tuning_df = fold_tuning_df.copy()
        fold_tuning_df["fold"] = fold_idx
        tuning_results.append(fold_tuning_df)

        best_config = best_config.copy()
        best_config["fold"] = fold_idx
        best_configs.append(best_config)

    if not return_tuning_results:
        return mhat

    tuning_results_df = pd.concat(tuning_results, ignore_index=True)
    best_configs_df = pd.DataFrame(best_configs)
    return mhat, tuning_results_df, best_configs_df

def DML1_estimator(df, kfsplit, ghat, mhat):
    y = df["Y"].to_numpy()
    z = df["Z"].to_numpy()
    g = np.asarray(ghat).reshape(-1)                 # (n,)
    M = np.asarray(mhat)
    if M.ndim == 1:
        M = M[:, None]                              # (n,1)
    elif M.ndim != 2:
        raise ValueError("mhat must be 1D or 2D.")

    rY = y - g                                      # (n,)
    rZ = z[:, None] - M                             # (n,k)

    K = len(kfsplit)
    theta = sum(
        (rY[te, None] * rZ[te]).mean(axis=0) / (rZ[te] * z[te, None]).mean(axis=0)
        for _, te in kfsplit
    ) / K

    eps = rY[:, None] - z[:, None] * theta          # (n,k)
    denom = (rZ * z[:, None]).mean(axis=0)          # (k,)
    var = (eps**2).mean(axis=0) / (denom**2) / len(df)

    return theta.squeeze(), var.squeeze()

def DML2_estimator(df, ghat, mhat):
    y = df["Y"].to_numpy()
    z = df["Z"].to_numpy()

    g = np.asarray(ghat).reshape(-1)          # (n,)
    M = np.asarray(mhat)
    if M.ndim == 1:
        M = M[:, None]                       # (n,1)
    elif M.ndim != 2:
        raise ValueError("mhat must be 1D or 2D.")

    rY = y - g                               # (n,)
    rZ = z[:, None] - M                      # (n,k)

    theta = (rY[:, None] * rZ).mean(axis=0) / (rZ * rZ).mean(axis=0)

    eps = rY[:, None] - z[:, None] * theta   # (n,k)
    denom = (rZ * rZ).mean(axis=0)           # (k,)
    var = (eps * eps).mean(axis=0) / (denom * denom) / len(df)

    return theta.squeeze(), var.squeeze()

def DML_partialout(df, Yhat, Zhat):
    y = df["Y"].to_numpy()
    z = df["Z"].to_numpy()

    yhat = np.asarray(Yhat).reshape(-1)          # (n,)
    Zhat = np.asarray(Zhat)
    if Zhat.ndim == 1:
        Zhat = Zhat[:, None]                     # (n,1)
    elif Zhat.ndim != 2:
        raise ValueError("Zhat must be 1D or 2D (n or n×k).")

    rY = y - yhat                                # (n,)
    rZ = z[:, None] - Zhat                       # (n,k)

    denom = (rZ * rZ).mean(axis=0)               # (k,)
    theta = (rY[:, None] * rZ).mean(axis=0) / denom

    # epsilon = (Y - Yhat) - theta*(Z - Zhat)  == rY - theta*rZ
    eps = rY[:, None] - rZ * theta               # (n,k)

    var = (eps * eps).mean(axis=0) / (denom * denom) / len(df)

    return theta.squeeze(), var.squeeze()

def Single_trial_cross_fitting_inference(n, N_unlabeled, d, typenum, batch_size, nodes, lr, weight_decay=0.0, epochs=2000, verbose=False,
                                         sparseRatio=0.7, K=5, shift=False, alpha=0.8):
    # Generate data
    df, m_oracle = WXYgeneration(n, d, typenum, alpha, return_oracle=True)
    df_unlabel_all = WXgeneration(N_unlabeled[-1], d, typenum, shift, alpha)

    # g_deepfit on all supervised data
    theta_g_deepfit, _, _ = g_deepfit(df, *prepare_tensors(df), dim_lin=1, dim_nonpar=d, sparseRatio=sparseRatio, 
                    val_data=prepare_tensors(df), nodes=nodes, batch_size=batch_size, lr=lr, epochs=epochs, verbose=verbose, weight_decay=weight_decay)
    # linear regression
    model = LinearRegression()
    # Train the model
    X_sm = sm.add_constant(df.drop(columns=['Y']))
    res = sm.OLS(df['Y'], X_sm).fit(cov_type="HC1")  # or omit cov_type for classical SE
    theta_linear = res.params.iloc[-1]
    se_linear = res.bse.iloc[-1]

    
    # K-Fold Cross-Fitting
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    kfsplit = list(kf.split(df))

    # Supervised NN estimator
    ghat_supervised, mhat_supervised = supervised_neural_network(df, kfsplit, sparseRatio, nodes, batch_size, lr, weight_decay, epochs, verbose)

    # Semi-supervised NN estimator
    mhat_semi_supervised_unlabeled_only = semi_supervised_neural_network_unlabeled_only(df, kfsplit, N_unlabeled, df_unlabel_all,
                                                          sparseRatio, nodes, batch_size*2, lr, epochs, verbose)
     
    # Traditional nonparametric estimator using local linear
    theta_np, var_np = kernelreg_partial_linear_theta(df["Y"].to_numpy(), df["Z"].to_numpy(), df.drop(columns=['Y', 'Z']).to_numpy(), reg_type="ll")
    
    # DML Estimators
    theta_DML2_supervised, var_DML2_supervised = DML2_estimator(df, ghat_supervised, mhat_supervised)

    # unlabeled only semi-supervised DML
    theta_DML2_semi_supervised_unlabeled_only, var_DML2_semi_supervised_unlabeled_only = DML2_estimator(df, ghat_supervised, mhat_semi_supervised_unlabeled_only)

    # Oracle DML estimator (true m(W), estimated g(W))
    theta_DML_oracle, var_oracle = DML2_estimator(df, ghat_supervised, m_oracle)

    # Collect results
    pairs = [
    (theta_g_deepfit, var_DML2_supervised),
    (theta_linear, se_linear**2),  
    (theta_DML2_supervised, var_DML2_supervised),
    (theta_DML2_semi_supervised_unlabeled_only, var_DML2_semi_supervised_unlabeled_only),
    (theta_np, var_np),
    (theta_DML_oracle, var_oracle),
]
    M = np.vstack([np.column_stack(np.atleast_1d(t, v))
               for t, v in pairs])
    return M
