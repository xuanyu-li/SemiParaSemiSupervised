import numpy as np
from numpy.linalg import lstsq
import pandas as pd
from sklearn.model_selection import KFold

def epanechnikov_kernel(u):
    """Univariate Epanechnikov kernel."""
    u = np.asarray(u)
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1
    out[mask] = 0.75 * (1 - u[mask]**2)
    return out

def product_epanechnikov(U):
    """
    U: array of shape (n, d) where each column is (Z_j - z0_j)/h_j.
    Returns product kernel weights of shape (n,).
    """
    K = np.ones(U.shape[0])
    for j in range(U.shape[1]):
        K *= epanechnikov_kernel(U[:, j])
    return K

def local_linear_multivariate(Z, Y, Z0, h):
    """
    Multivariate local linear regression using product Epanechnikov kernel.

    Parameters
    ----------
    Z : array (n, d)
    Y : array (n,)
    Z0: target points, array (m, d)
    h : bandwidths: scalar or array of length d

    Returns
    -------
    mhat : array (m,)
    """
    Z = np.asarray(Z)
    Y = np.asarray(Y).reshape(-1)
    Z0 = np.atleast_2d(Z0)
    n, d = Z.shape

    # allow scalar bandwidth
    if np.isscalar(h):
        h = np.repeat(h, d)
    h = np.asarray(h)
    # Guard against zero/negative bandwidths.
    if np.any(h <= 0):
        raise ValueError("Bandwidth must be strictly positive in every dimension.")

    mhat = np.empty(Z0.shape[0])
    k_nn = min(max(d + 1, 5), n)

    for k, z0 in enumerate(Z0):
        U = (Z - z0) / h       # (n, d)
        K = product_epanechnikov(U)

        if np.all(K == 0):
            # Compact-support kernels can assign zero weight to all points in higher
            # dimensions or with small bandwidth. Fall back to a small nearest-neighbor
            # weighted fit to avoid NaN predictions.
            dist2 = np.sum(((Z - z0) / h) ** 2, axis=1)
            nn_idx = np.argpartition(dist2, k_nn - 1)[:k_nn]
            K = np.zeros(n)
            K[nn_idx] = np.exp(-0.5 * dist2[nn_idx]) + 1e-12

        # Design matrix for local linear: [1, Z - z0]
        # Use weighted least squares directly for numerical stability
        # (avoids singular normal-equation crashes).
        Xloc = np.column_stack([np.ones(n), Z - z0])
        sqrtK = np.sqrt(K)
        Xw = Xloc * sqrtK[:, None]
        Yw = Y * sqrtK
        beta, _, _, _ = lstsq(Xw, Yw, rcond=None)

        mhat[k] = beta[0]   # a = m(z0)
        if not np.isfinite(mhat[k]):
            # Last-resort finite fallback.
            wsum = K.sum()
            mhat[k] = (K @ Y) / wsum if wsum > 0 else np.mean(Y)

    return mhat


def select_bandwidth_cv(df_train, target_col, candidates, cv_folds=5):
    x_all = df_train.drop(columns=["Y", "Z"]).to_numpy()
    y_all = df_train[target_col].to_numpy()
    kf_inner = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    best_bw = None
    best_score = np.inf
    for bw in candidates:
        fold_scores = []
        for tr_idx, va_idx in kf_inner.split(x_all):
            x_tr, y_tr = x_all[tr_idx], y_all[tr_idx]
            x_va, y_va = x_all[va_idx], y_all[va_idx]
            y_pred = local_linear_multivariate(x_tr, y_tr, x_va, bw)

            valid = np.isfinite(y_pred)
            if valid.sum() == 0:
                fold_scores.append(np.inf)
                continue
            fold_scores.append(np.mean((y_pred[valid] - y_va[valid]) ** 2))

        score = np.mean(fold_scores)
        if score < best_score:
            best_score = score
            best_bw = bw

    return best_bw if best_bw is not None else candidates[-1]


def traditional_nonparametric_estimator(df, kfsplit, bandwidth=None):
    Yhat = np.zeros(len(df))
    Zhat = np.zeros(len(df))
    default_candidates = [0.01, 0.05, 0.1, 0.2, 0.5]
    candidates = default_candidates if bandwidth is None else list(np.atleast_1d(bandwidth))

    for train_index, test_index in kfsplit:
        df_kc, df_k = df.iloc[train_index], df.iloc[test_index]
        bw_y = select_bandwidth_cv(df_kc, "Y", candidates, cv_folds=5)
        bw_z = select_bandwidth_cv(df_kc, "Z", candidates, cv_folds=5)
        Yhat[test_index] = local_linear_multivariate(
            df_kc.drop(columns=["Y", "Z"]),
            df_kc["Y"],
            df_k.drop(columns=["Y", "Z"]),
            bw_y,
        )
        Zhat[test_index] = local_linear_multivariate(
            df_kc.drop(columns=["Y", "Z"]),
            df_kc["Z"],
            df_k.drop(columns=["Y", "Z"]),
            bw_z,
        )
    return Yhat, Zhat
    
    
