import math

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import torch
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from torchtuples import Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dqAux import dqNetSparse,checkLoss,checkErrorMean,getSESingle
import torch.nn as nn
from data_gen import *

# training Partially Linear Quantile Regression (DPLQR)
def g_deepfit(df_train,x_train,y_train,dim_lin,dim_nonpar,sparseRatio,val_data,nodes,batch_size,lr,epochs,verbose):
    # Create a LinearRegression model
    model = LinearRegression()

    # Train the model
    model.fit(df_train.drop(columns=['Y']), df_train['Y'])
    paramsLin = model.coef_[-1]

    coef_init_weight = torch.tensor(paramsLin, dtype=y_train.dtype)
    loss = nn.MSELoss()
    model = Model(dqNetSparse(dim_lin, dim_nonpar, coef_init_weight, nodes, sparseRatio), loss)
    model.optimizer.set_lr(lr)
    callbacks = [tt.callbacks.EarlyStopping()]
    model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val_data,
              val_batch_size=5 * batch_size)
    weights = list(model.net.parameters())
    deepCoefEst = weights[0].detach().to('cpu').numpy()[0][0]

    return deepCoefEst, model,paramsLin


def m_deepfit(x_train,x_val,nodes,batch_size,lr,epochs,verbose):
    loss = nn.MSELoss()
    z_train = (x_train[0]).view(x_train[0].size()[0],1)
    z_val = (x_val[0]).view(x_val[0].size()[0],1)
    x_n_train = x_train[1]
    x_n_val = x_val[1]
    dim_nonpar = x_train[1].size()[1]
    model_m = Model(mNet(dim_nonpar,nodes),loss)
    model_m.optimizer.set_lr(lr)
    callbacks = [tt.callbacks.EarlyStopping()]
    val_data = (x_n_val,z_val)
    model_m.fit(x_n_train, z_train, batch_size, epochs, callbacks, verbose,
                    val_data=val_data, val_batch_size=5*batch_size)
    x_merge = torch.cat((x_n_train,x_n_val),dim=0)
    preds = model_m.predict(x_merge)
    z_merge = torch.cat((z_train,z_val),dim=0)
    z_delta = (z_merge-preds.to('cpu')).numpy()
    z_merge = z_merge.numpy()

    return z_delta,z_merge,model_m



class mNet(nn.Module):
    def __init__(self, input_dim, node):
        super(mNet, self).__init__()

        depth, width = node
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(width, 1))  # Assuming a single output node for regression

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def predict(self,in_nonpar):
        return self.forward(in_nonpar)

 

def Single_trial(n,N_unlabeled,d,typenum,batch_size,nodes,lr,epochs,verbose,sparseRatio):
    df = WXYgeneration(n, d, typenum)
    df_train, df_val = train_test_split(df, test_size=0.2)
    df_test = WXYgeneration(5000, d, typenum)

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

    ### deepPLR and DMPLR
    deepCoefEst, model,paramsLin = g_deepfit(df_train, x_train, y_train, dim_lin, dim_nonpar, sparseRatio, val_data, nodes,
                                   batch_size, lr, epochs, verbose)
    z_delta, z_merge, model_m = m_deepfit(x_train, x_val, nodes, batch_size, lr, epochs, verbose)
    y_merge = torch.cat((y_train, y_val), dim=0)
    x_merge_nonpar = torch.tensor(df.drop(columns=['Z', 'Y']).to_numpy().astype('float32'))
    x_merge_lin = torch.tensor(df['Z'].values).view(len(df), 1).type(torch.float32)
    x_merge = (x_merge_lin, x_merge_nonpar)
    preds = model.predict(x_merge).to('cpu') - x_merge_lin * deepCoefEst
    y_delta = (y_merge - preds).numpy()
    DMLCoefEst = np.mean(z_delta * y_delta) / np.mean(z_delta * z_merge)

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

        x_labeled_train = x_train[1]
        x_labeled_val = x_val[1]
        x_labeled = torch.cat((x_labeled_train, x_labeled_val), dim=0)
        z_labeled_train = (x_train[0]).view(x_train[0].size()[0], 1)
        z_labeled_val = (x_val[0]).view(x_val[0].size()[0], 1)
        z_labeled = torch.cat((z_labeled_train, z_labeled_val), dim=0)
        z_labeled_delta = (z_labeled - model_ss_m.predict(x_labeled).to('cpu')).numpy()
        z_labeled = z_labeled.numpy()
        SSDMLCoefEst = np.mean(z_labeled_delta * y_delta) / np.mean(z_labeled_delta * z_labeled)
        SSDMLCoefEsts.append(SSDMLCoefEst)

    combined_list = [paramsLin, deepCoefEst, DMLCoefEst] + SSDMLCoefEsts
    return combined_list