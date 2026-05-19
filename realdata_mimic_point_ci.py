import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from sklearn.model_selection import KFold

from NN_estimator_rewrite import (
    DML2_estimator,
    g_deepfit,
    prepare_tensors,
    semi_supervised_neural_network_unlabeled_only,
    supervised_neural_network,
)
from traditional_nonparametric import kernelreg_partial_linear_theta


DEFAULT_W_COLS = [
    "gender",
    "age",
    "weight",
    "tidal_volume",
    "meanbp",
    "hr",
    "rr",
    "tempc",
]


def resolve_input_path(path: str) -> Path:
    input_path = Path(path)
    if input_path.exists():
        return input_path

    script_dir_path = Path(__file__).resolve().parent / path
    if script_dir_path.exists():
        return script_dir_path

    return input_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_labeled(raw: pd.DataFrame, w_cols: list[str]) -> pd.DataFrame:
    cols = w_cols + ["treatment", "outcome"]
    missing = sorted(set(cols) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing labeled columns: {missing}")

    df = raw[cols].copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"treatment": "Z", "outcome": "Y"})
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


def clean_unlabeled(raw: pd.DataFrame, w_cols: list[str]) -> pd.DataFrame:
    cols = w_cols + ["treatment"]
    missing = sorted(set(cols) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing unlabeled columns: {missing}")

    df = raw[cols].copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"treatment": "Z"})
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


def summarize(name: str, theta: float, var: float) -> dict[str, float | str]:
    theta = float(np.asarray(theta).squeeze())
    var = float(np.asarray(var).squeeze())
    se = np.sqrt(var) if np.isfinite(var) and var >= 0 else np.nan
    return {
        "estimator": name,
        "theta_hat": theta,
        "ci95_lower": theta - 1.96 * se if np.isfinite(se) else np.nan,
        "ci95_upper": theta + 1.96 * se if np.isfinite(se) else np.nan,
    }


def run_analysis(
    df_lab: pd.DataFrame,
    df_unlab: pd.DataFrame,
    k_folds: int,
    sparse_ratio: float,
    nodes: list[int],
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    verbose: bool,
    seed: int,
) -> pd.DataFrame:
    if len(df_lab) < k_folds:
        raise ValueError(f"Need at least {k_folds} labeled rows, got {len(df_lab)}.")
    if len(df_unlab) == 0:
        raise ValueError("No usable unlabeled rows after cleaning.")

    set_seed(seed)

    # 1) Linear regression: Y ~ Z + W, HC1 robust confidence interval.
    x_sm = sm.add_constant(df_lab.drop(columns=["Y"]))
    ols_res = sm.OLS(df_lab["Y"], x_sm).fit(cov_type="HC1")
    theta_linear = float(ols_res.params["Z"])
    var_linear = float(ols_res.bse["Z"] ** 2)

    # 2) Local linear partial-linear estimator using statsmodels KernelReg.
    w = df_lab.drop(columns=["Y", "Z"]).to_numpy()
    theta_local, var_local = kernelreg_partial_linear_theta(
        y=df_lab["Y"].to_numpy(),
        t=df_lab["Z"].to_numpy(),
        w=w,
        reg_type="ll",
    )

    # 3) Deep partial linear estimator on all labeled rows.
    batch_size = min(batch_size, len(df_lab))
    theta_deep, _, _ = g_deepfit(
        df_lab,
        *prepare_tensors(df_lab),
        dim_lin=1,
        dim_nonpar=df_lab.drop(columns=["Y", "Z"]).shape[1],
        sparseRatio=sparse_ratio,
        val_data=prepare_tensors(df_lab),
        nodes=list(nodes),
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        verbose=verbose,
        weight_decay=weight_decay,
    )

    # 4) Supervised DML nuisance models.
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    kfsplit = list(kf.split(df_lab))
    ghat_sup, mhat_sup = supervised_neural_network(
        df_lab,
        kfsplit,
        sparse_ratio,
        list(nodes),
        batch_size,
        lr,
        weight_decay,
        epochs,
        verbose,
    )
    theta_dml_sup, var_dml_sup = DML2_estimator(df_lab, ghat_sup, mhat_sup)

    # 5) Semi-supervised DML: estimate m(W)=E[Z|W] using all unlabeled windows.
    mhat_ss = semi_supervised_neural_network_unlabeled_only(
        df=df_lab,
        kfsplit=kfsplit,
        n_unlabel=len(df_unlab),
        df_unlabel_all=df_unlab,
        sparseRatio=sparse_ratio,
        nodes=list(nodes),
        batch_size=batch_size * 2,
        lr=lr,
        epochs=epochs,
        verbose=verbose,
    )
    theta_dml_ss, var_dml_ss = DML2_estimator(df_lab, ghat_sup, mhat_ss)

    rows = [
        summarize("Linear regression", theta_linear, var_linear),
        summarize("Local linear estimation", theta_local, var_local),
        # This follows test_new2.ipynb, which uses the supervised DML variance
        # for the deep partial linear confidence interval.
        summarize("Deep partial linear", theta_deep, var_dml_sup),
        summarize("Supervised DML", theta_dml_sup, var_dml_sup),
        summarize("Semi-supervised DML", theta_dml_ss, var_dml_ss),
    ]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real-data MIMIC analysis using supervised_patients as labeled data "
            "and unlabeled_windows as unlabeled data."
        )
    )
    parser.add_argument("--labeled-input", default="supervised_patients.csv")
    parser.add_argument("--unlabeled-input", default="unlabeled_windows.csv")
    parser.add_argument("--covariates", nargs="+", default=DEFAULT_W_COLS)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sparse-ratio", type=float, default=0.7)
    parser.add_argument("--nodes-first", type=int, default=3)
    parser.add_argument("--nodes-second", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path for the point estimates and confidence intervals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = [args.nodes_first, args.nodes_second]

    raw_lab = pd.read_csv(resolve_input_path(args.labeled_input))
    raw_unlab = pd.read_csv(resolve_input_path(args.unlabeled_input))
    df_lab = clean_labeled(raw_lab, args.covariates)
    df_unlab = clean_unlabeled(raw_unlab, args.covariates)

    result = run_analysis(
        df_lab=df_lab,
        df_unlab=df_unlab,
        k_folds=args.kfolds,
        sparse_ratio=args.sparse_ratio,
        nodes=nodes,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        verbose=args.verbose,
        seed=args.seed,
    )

    print(result.to_string(index=False))

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
