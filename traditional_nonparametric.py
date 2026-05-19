import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg


def kernelreg_partial_linear_theta(y, t, w, reg_type="ll"):
    y = np.asarray(y).reshape(-1)
    t = np.asarray(t).reshape(-1)
    w = np.asarray(w)
    if w.ndim == 1:
        w = w[:, None]

    var_type = "c" * w.shape[1]

    # 1) mY(w) = E[Y|W]
    kr_y = KernelReg(endog=y, exog=w, var_type=var_type, reg_type=reg_type)
    m_y, _ = kr_y.fit(w)

    # 2) mT(w) = E[T|W] (single treatment)
    kr_t = KernelReg(endog=t, exog=w, var_type=var_type, reg_type=reg_type)
    m_t, _ = kr_t.fit(w)

    # 3) residual regression
    y_tilde = y - m_y
    t_tilde = t - m_t
    fit = sm.OLS(y_tilde, t_tilde[:, None]).fit(cov_type="HC1")

    theta = float(fit.params[0])
    var = float(fit.bse[0] ** 2)
    return theta, var

