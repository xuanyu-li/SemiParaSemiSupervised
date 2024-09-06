import pandas as pd
import numpy as np
from g_deepPLR import *
import torch
from torchtuples import Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dqAux import dqNetSparse,checkLoss,checkErrorMean,getSESingle
import torch.nn as nn
from data_gen import *
from sklearn.model_selection import KFold

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

d_values = [5, 10, 15]
n_unlabeled_values = [1000, 4000, 7000]
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

K = 2

def Single_trial_cross_fitting(n,N_unlabeled,d,typenum,batch_size,nodes,lr,epochs,verbose,sparseRatio,K):
    df = WXYgeneration(n, d, typenum)
    df_train, df_val = train_test_split(df, test_size=0.2)
    df_test = WXYgeneration(5000, d, typenum)
    kf = KFold(n_splits=K, shuffle=True)
    SSDMLCoefEsts_list = []

    for df_Kc, df_K in kf.split(df):
        df_K = df.iloc[df_K]
        df_Kc = df.iloc[df_Kc]

        df_train, df_val = train_test_split(df_Kc, test_size=0.2)

        x_train_nonpar = torch.tensor(df_train.drop(columns=['Z', 'Y']).to_numpy().astype('float32'))
        x_val_nonpar = torch.tensor(df_val.drop(columns=['Z', 'Y']).to_numpy().astype('float32'))
        x_test_nonpar = torch.tensor(df_test.drop(columns=['Z', 'Y']).to_numpy().astype('float32'))

        x_train_lin = torch.tensor(df_train['Z'].values).view(len(df_train), 1).type(torch.float32)
        x_val_lin = torch.tensor(df_val['Z'].values).view(len(df_val), 1).type(torch.float32)
        x_test_lin = torch.tensor(df_test['Z'].values).view(len(df_test), 1).type(torch.float32)

        y_train = torch.tensor(df_train['Y'].values).view(len(df_train), 1).type(torch.float32)
        y_test = torch.tensor(df_test['Y'].values).view(len(df_test), 1).type(torch.float32)
        y_val = torch.tensor(df_val['Y'].values).view(len(df_val), 1).type(torch.float32)

        x_train = (x_train_lin, x_train_nonpar)
        x_test = (x_test_lin, x_test_nonpar)
        x_val = (x_val_lin, x_val_nonpar)
        val_data = (x_val, y_val)
        dim_lin, dim_nonpar = 1, x_val_nonpar.shape[1]

        ### deepPLR
        deepCoefEst, model,paramsLin = g_deepfit(df_train, x_train, y_train, dim_lin, dim_nonpar, sparseRatio, val_data, nodes,
                                       batch_size, lr, epochs, verbose)


        y_K = torch.tensor(df_K['Y'].values).view(len(df_K), 1).type(torch.float32)
        x_merge_nonpar_K = torch.tensor(df_K.drop(columns=['Z', 'Y']).to_numpy().astype('float32'))
        x_merge_lin_K = torch.tensor(df_K['Z'].values).view(len(df_K), 1).type(torch.float32)
        x_merge_K = (x_merge_lin_K, x_merge_nonpar_K)
        preds = model.predict(x_merge_K).to('cpu') - x_merge_lin_K * deepCoefEst
        y_delta_K = (y_K - preds).numpy()


        ### SS-DeepPLR
        SSDMLCoefEsts = []
        for n_unlabeled in N_unlabeled:
            n_unlabel = n_unlabeled
            df_unlabel = WXgeneration(n_unlabel, d, typenum)
            df_ss = pd.concat([df, df_unlabel], axis=0)
            df_ss_train, df_ss_val = train_test_split(df_ss, test_size=0.2)

            x_ss_train_nonpar = torch.tensor(df_ss_train.drop(columns=['Z', 'Y']).to_numpy().astype('float32'))
            x_ss_val_nonpar = torch.tensor(df_ss_val.drop(columns=['Z', 'Y']).to_numpy().astype('float32'))
            x_ss_train_lin = torch.tensor(df_ss_train['Z'].values).view(len(df_ss_train), 1).type(torch.float32)
            x_ss_val_lin = torch.tensor(df_ss_val['Z'].values).view(len(df_ss_val), 1).type(torch.float32)

            x_ss_train = (x_ss_train_lin, x_ss_train_nonpar)
            x_ss_val = (x_ss_val_lin, x_ss_val_nonpar)

            z_ss_delta, z_ss_merge, model_ss_m = m_deepfit(x_ss_train, x_ss_val, nodes, int(batch_size*n_unlabel/n), lr, epochs, verbose)


            z_labeled_K = torch.tensor(df_K['Z'].values).view(len(df_K), 1).type(torch.float32)
            x_labeled_K = torch.tensor(df_K.drop(columns=['Z', 'Y']).to_numpy().astype('float32'))

            z_labeled_delta_K = (z_labeled_K - model_ss_m.predict(x_labeled_K).to('cpu')).numpy()
            z_labeled_K = z_labeled_K.numpy()
            SSDMLCoefEst = np.mean(z_labeled_delta_K * y_delta_K) / np.mean(z_labeled_delta_K * z_labeled_K)
            SSDMLCoefEsts.append(SSDMLCoefEst)
        SSDMLCoefEsts_list.append(SSDMLCoefEsts)
    final_SSDMLCoefEsts = np.mean(SSDMLCoefEsts_list, axis=0)
    return final_SSDMLCoefEsts

def run_cross_fitting_simulation(n,n_unlabeled,K, d, num_reps=200):
    results = []
    for _ in range(num_reps):
        outcomes = Single_trial_cross_fitting(n,n_unlabeled,d,typenum,batch_size,nodes,lr,epochs,verbose,sparseRatio,K)
        results.append(outcomes)

    # Convert results to a NumPy array for easier statistical computation
    results_array = np.array(results)
    means = np.mean(results_array, axis=0)
    std_dev = np.std(results_array, axis=0, ddof=1)

    return means, std_dev


# Initialize DataFrame structure

columns = [f"d={d}" for d in d_values]
index_ss = [f"n_unlabeled={n} - SSDMLCoefEst" for n in n_unlabeled_values]
results_df = pd.DataFrame(index=index_ss, columns=columns)

# Perform simulations and fill DataFrame
for d in d_values:
    means, std_dev = run_cross_fitting_simulation(n, n_unlabeled_values,K, d, num_reps)
    means = means - 1
    for i, ind in enumerate(index_ss):
        formatted = [f"{m:.4f}({se:.4f})" for m, se in zip(means, std_dev)]
        results_df.loc[f"{ind}", f"d={d}"] = formatted[i]

# Print results for verification
print(results_df)
print(results_df.style.to_latex())