import numpy as np
import pandas as pd
import scipy.stats as stats
import pywt
from math import pi
from math import log
# Function for g1 and m1
def g1(w,d):
    return np.sin(2 * pi * w[0]) + 3 * np.cos(2 * pi * w[1]) + ( w[2]) ** 4 + log(0.5 + w[3]) +  np.sqrt(2 * w[4]) - 5
def m1(w,d):
    return (w[0] + log(0.5 + w[1]) + w[2] * w[3] + w[4] - 2)
# Function for g2 and m2
def g2(w,d):
    g1_result = np.array([w[0] + np.sqrt(w[1]), (w[1] ** 2) * (w[2] ** 2), np.sin(2 ** pi * w[3]) * np.sin(2 ** pi * w[4])])

    # g2(a1, a2, a3) = (a2 cos(a1), sin(a1 + a2), a3^2)
    g2_result = np.array([g1_result[1] * np.cos(pi * g1_result[0]),
                          np.sin(g1_result[0] + g1_result[1]),
                          g1_result[2] / g1_result[1]])

    # g3(b1, b2, b3) = b1 + b2 + b3
    return  g2_result[0] * g2_result[2] + 3 * (g2_result[1] + 1) ** 2 - 6
def m2(w,d):
    m1_result = np.sin(2 * pi * w)

    # m2(a1, ..., a5) = (sin(a1 + a2), cos(a3 + a4), a5^2)
    m2_result = np.array([np.sin(m1_result[0] + m1_result[1]),
                          np.cos(m1_result[2] + m1_result[3]),
                          m1_result[4] ** 2])

    # m3(b1, b2, b3) = sin(b1) + sin(b2) + sin(b3)
    return (m2_result[0])**2 +  m2_result[1] * np.sin(m2_result[2])

# Function for g3 and m3
def g3(W, alpha):
    #gammas = [0, 3, 6, 9, 10, 16]
    gammas = [0, 3, 6, 9, 10, 16]
    R = 1
    level = 20
    wv = pywt.Wavelet('db6')
    (_, psi, x) = wv.wavefun(level=level)
    wv = {
        "x": x,
        "psi": psi
    }
    wv_y = wv["psi"][1:]
    wv_y = wv_y.reshape(11, 2 ** level)
    wv_y = np.sum(wv_y, axis=0)

    wv_trans_W = np.apply_along_axis(lambda col: wv_trans(col, wv_y, R, gammas, alpha, level), axis=0, arr=W)
    y = np.mean(wv_trans_W, axis=1)
    return y

def m3(W, alpha):
    #gammas = [0, 3, 6, 9, 10, 16]
    gammas = [0, 3, 6, 9, 10, 16]
    R = 1
    level = 20
    wv = pywt.Wavelet('db6')
    (_, psi, x) = wv.wavefun(level=level)
    wv = {
        "x": x,
        "psi": psi
    }
    wv_y = wv["psi"][1:]
    wv_y = wv_y.reshape(11, 2 ** level)
    wv_y = np.sum(wv_y, axis=0)
    meanw = np.mean(W, axis=1)
    y = wv_trans(meanw, wv_y, R, gammas, alpha, level)
    return y
def WXYgeneration(n, d, typenum, alpha, return_oracle=False):
    # correlation_param = 0.5
    # corr_matrix = correlation_param * np.ones((d, d)) + (1 - correlation_param) * np.eye(d)
    # # Generate Gaussian copula
    # rng = np.random.default_rng()
    # W = rng.multivariate_normal(mean=np.zeros(d), cov=corr_matrix, size=n)
    # W = stats.norm.cdf(W)  # Convert to uniform [0, 1]
    W = np.random.uniform(0, 1, size=(n, d))
    # Generate error terms
    epsilon = np.random.normal(0, 1, n)
    #v = stats.t.rvs(df=2, loc=0, scale=1, size=n)
    v = np.random.normal(0, 1, n)

    if typenum == 1:
        m_oracle = np.array([m1(w,d) for w in W])
        Z = m_oracle + v
        Y = np.array([g1(w,d) for w in W]) + epsilon
    elif typenum == 2:
        m_oracle = np.array([m2(w,d) for w in W])
        Z = m_oracle + v
        Y = np.array([g2(w,d) for w in W]) + epsilon
    elif typenum == 3:
        m_oracle = m3(W, alpha)
        Z = m_oracle + v
        Y = g3(W, alpha) + epsilon
    else:
        raise ValueError("Invalid type. Please choose type 1, 2, or 3.")

    # Create DataFrame
    df = pd.DataFrame(W, columns=[f"W{j}" for j in range(1, d + 1)])
    df['Z'] = Z
    df['Y'] = Y

    if return_oracle:
        return df, m_oracle
    return df
def WXgeneration(n,d,typenum,shift, alpha):
    #W = np.random.uniform(0,1,size = (n,d))
    #
    # correlation_param = 0.5
    # if shift == True:
    #     correlation_param = 0.4
    # corr_matrix = correlation_param * np.ones((d, d)) + (1 - correlation_param) * np.eye(d)
    # # Generate Gaussian copula
    # rng = np.random.default_rng()
    # W = rng.multivariate_normal(mean=np.zeros(d), cov=corr_matrix, size=n)
    # W = stats.norm.cdf(W)  # Convert to uniform [0, 1]
    W = np.random.uniform(0, 1, size=(n, d))
    # Generate error terms
    # v = stats.t.rvs(df=2, loc=0, scale=1, size=n)
    v = np.random.normal(0, 1, n)
    if typenum == 1:
        Z = np.array([m1(w,d) for w in W]) + v
    elif typenum == 2:
        Z = np.array([m2(w,d) for w in W]) + v
    elif typenum == 3:
        Z = m3(W, alpha) + v
    else:
        raise ValueError("Invalid type. Please choose type 1, 2, or 3.")

    # Create DataFrame
    df = pd.DataFrame(W, columns=[f"W{j}" for j in range(1, d + 1)])
    df['Z'] = Z

    return df

def wv_trans(x, wv_y, R, gammas, alpha, level):
    b = np.zeros(len(x))
    for gamma in gammas:
        tempx = (2**gamma) * x
        fractional_part, _ = np.modf(tempx)
        ## adjust the negative number
        fractional_part = fractional_part + 1
        fractional_part, _ = np.modf(fractional_part)
        ## calculate the Daubechies wavelet value
        fractional_part = fractional_part * (2 ** level)
        integer_part = np.round(fractional_part)
        integer_part[integer_part == 0] = 2 ** level
        # derive the index of each element in x corresbonding to the wavelet value
        index_x_psi = integer_part.astype(int) - 1
        # calculate the derived value of the function
        b = b + wv_y[index_x_psi] * R * 2**(-gamma * (alpha + 0.25)) * pow(2, gamma / 2.0)
    return b


