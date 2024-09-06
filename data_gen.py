import numpy as np
import pandas as pd
import scipy.stats as stats

# Function for g1 and m1
def g1(w,d):
    return 2/5 * np.sum(w)

def m1(w,d):
    return 1/5 * np.sum(w)

# Function for g2 and m2
def g2(w,d):
    terms = [(w[5*i-5])**2 - (w[5*i-4] - 1)**2 + abs(w[5*i-3] - 0.5) + 0.6*np.sin(np.pi*w[5*i-2]) + np.log(w[5*i-1] + 0.3) + np.sqrt(w[5*i-2] + 0.2)
             for i in range(1, int(d/5)+1)]
    return np.sum(terms)

def m2(w,d):
    terms = [(w[5*i-5] - 1)**2 - (w[5*i-4])**2 + abs(w[5*i-3] - 0.7) + 0.8*np.sin(np.pi*w[5*i-2]) + np.log(w[5*i-1] + 0.4) + np.sqrt(w[5*i-2] + 0.3)
             for i in range(1, int(d/5)+1)]
    return np.sum(terms)

# Function for g3 and m3
def g3(w,d):
    mean1 = np.mean([w[5*i-5] for i in range(1, int(d/5)+1)])
    mean2 = np.mean([w[5*i-4] for i in range(1, int(d/5)+1)])
    sum3 = np.sum([w[5*i-4] * w[5*i-3] for i in range(1, int(d/5)+1)])
    mean4 = np.mean([w[5*i-3] * w[5*i-2] for i in range(1, int(d/5)+1)])
    pi5 = np.product([2*w[5*i-1] for i in range(1,int(d/5)+1) ])
    return 8 * (mean1**2 +  abs(mean2-0.6) + 0.4 * np.sin(np.pi * sum3) + np.log(mean4 + 0.1) * np.sqrt(pi5+0.4))

def m3(w,d):
    mean1 = np.mean([w[5*i-5] for i in range(1, int(d/5)+1)])
    mean2 = np.mean([w[5*i-4] for i in range(1, int(d/5)+1)])
    sum3 = np.sum([w[5*i-4] * w[5*i-3] for i in range(1, int(d/5)+1)])
    mean4 = np.mean([w[5*i-3] * w[5*i-2] for i in range(1, int(d/5)+1)])
    pi5 = np.product([2*w[5*i-1] for i in range(1,int(d/5)+1) ])
    return   (mean1**2 +  abs(mean2-0.5) + 0.7 * np.cos(np.pi * sum3) + np.log(mean4 + 0.3) * np.sqrt(pi5+0.2))


def WXYgeneration(n, d, typenum):
    correlation_param = 0.5
    corr_matrix = correlation_param * np.ones((d, d)) + (1 - correlation_param) * np.eye(d)
    # Generate Gaussian copula
    rng = np.random.default_rng()
    W = rng.multivariate_normal(mean=np.zeros(d), cov=corr_matrix, size=n)
    W = stats.norm.cdf(W)  # Convert to uniform [0, 1]
    # Generate error terms
    epsilon = stats.t.rvs(df=3, loc=0, scale=1, size=n)
    #v = stats.t.rvs(df=2, loc=0, scale=1, size=n)
    v = np.random.normal(0, 1, n)

    if typenum == 1:
        Z = np.array([m1(w,d) for w in W]) + v
        Y = Z + np.array([g1(w,d) for w in W]) + epsilon
    elif typenum == 2:
        Z = np.array([m2(w,d) for w in W]) + v
        Y = Z + np.array([g2(w,d) for w in W]) + epsilon
    elif typenum == 3:
        Z = np.array([m3(w,d) for w in W]) + v
        Y = Z + np.array([g3(w,d) for w in W]) + epsilon
    else:
        raise ValueError("Invalid type. Please choose type 1, 2, or 3.")

    # Create DataFrame
    df = pd.DataFrame(W, columns=[f"W{j}" for j in range(1, d + 1)])
    df['Z'] = Z
    df['Y'] = Y

    return df

def WXgeneration(n,d,typenum):
    #W = np.random.uniform(0,1,size = (n,d))

    correlation_param = 0.5
    corr_matrix = correlation_param * np.ones((d, d)) + (1 - correlation_param) * np.eye(d)
    # Generate Gaussian copula
    rng = np.random.default_rng()
    W = rng.multivariate_normal(mean=np.zeros(d), cov=corr_matrix, size=n)
    W = stats.norm.cdf(W)  # Convert to uniform [0, 1]
    # Generate error terms
    # v = stats.t.rvs(df=2, loc=0, scale=1, size=n)
    v = np.random.normal(0, 1, n)
    if typenum == 1:
        Z = np.array([m1(w,d) for w in W]) + v
    elif typenum == 2:
        Z = np.array([m2(w,d) for w in W]) + v
    elif typenum == 3:
        Z = np.array([m3(w,d) for w in W]) + v
    else:
        raise ValueError("Invalid type. Please choose type 1, 2, or 3.")

    # Create DataFrame
    df = pd.DataFrame(W, columns=[f"W{j}" for j in range(1, d + 1)])
    df['Z'] = Z

    return df