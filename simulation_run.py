import pandas as pd
import numpy as np
from g_deepPLR import *

def run_simulation(n,n_unlabeled, d, num_reps=200):
    results = []
    for _ in range(num_reps):
        outcomes = Single_trial(n,n_unlabeled,d,typenum,batch_size,nodes,lr,epochs,verbose,sparseRatio)
        results.append(outcomes)

    # Convert results to a NumPy array for easier statistical computation
    results_array = np.array(results)
    means = np.mean(results_array, axis=0)
    std_dev = np.std(results_array, axis=0, ddof=1)

    return means, std_dev

d_values = [5, 10, 15,20]
n_unlabeled_values = [1000, 4000, 7000, 10000]
num_reps = 500
typenum = 3
n = 1000
# hyperparameters
batch_size = 128
nodes = [5,128] # number of network layers and nodes per layer
lr = 0.001
epochs = 500
verbose = False
sparseRatio = 0.5


# Initialize DataFrame structure
outcomes = ['paramsLin', 'deepCoefEst', 'DMLCoefEst', 'SSDMLCoefEst']
columns = [f"d={d}" for d in d_values]
indexsupervised = ['paramsLin', 'deepCoefEst', 'DMLCoefEst']
index_ss = [f"n_unlabeled={n} - SSDMLCoefEst" for n in n_unlabeled_values]
index = indexsupervised + index_ss
results_df = pd.DataFrame(index=index, columns=columns)

# Perform simulations and fill DataFrame
for d in d_values:
    means, std_dev = run_simulation(n, n_unlabeled_values, d, num_reps)
    means = means - 1
    for i,ind in enumerate(index):
            formatted = [f"{m:.4f}({se:.4f})" for m, se in zip(means, std_dev)]
            results_df.loc[f"{ind}",f"d={d}"] = formatted[i]

 
# Print results for verification
print(results_df)
print(results_df.style.to_latex())