import numpy as np
import scipy
from scipy.linalg import circulant
from scipy.sparse import  kron, identity, csr_matrix
from scipy.stats import qmc
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import torch
from two_d_data_set import *
from two_d_model import deeponet, NN0, NN1,NN11, NN00, deeponet3, deeponet4
from test_deeponet import domain

# from draft import create_data, expand_function
from geometry import Rect
import time

from utils import count_trainable_params, extract_path_from_dir, save_uniqe, grf, solve_subdomain2
from constants import Constants
names=['1']
def generate_data(names,  save_path, number_samples,seed, seed1, seed2):
    X=[]
    Y=[]
    n=61
    x0=np.linspace(0,1,n)

    side=0
    # left rect side=1
    if side:
        x=x0[:int(0.5*(n-1))]
        s=domain(x,1)
        x_int=x[1:]
    else:
        x=x0[int(0.5*(n-1))-1:]
        s=domain(x,0)
        x_int=x[:-1]
    
    for name in enumerate(names):
        f=grf(x[1:], number_samples,seed=seed )
        g1=grf([1], number_samples,seed=seed1 )
        g2=grf([1], number_samples,seed=seed2 )
        g=(g1+Constants.l*g2)
        # g=(g1+1J*g2)


        for i in range(number_samples):
            A,G=s.solver(f[i],g[i]*0)
            # NN11(x_int,f[i],g[i]*0)
            u=scipy.sparse.linalg.spsolve(A, G).real
            for j in range(len(x_int)):
                X1=[
                    torch.tensor(x_int[j], dtype=torch.float32),
                    torch.tensor(f[i], dtype=torch.float32),
                    torch.tensor(g[i], dtype=torch.cfloat)
                    ]
                Y1=torch.tensor(u[j], dtype=torch.float32)
                save_uniqe([X1,Y1],save_path)
                X.append(X1)
                Y.append(Y1)
               
    return X,Y        

# 

if __name__=='__main__':
    # pass
# if False:
    X,Y=generate_data(names, Constants.train_path, number_samples=500, seed=0,seed1=1,seed2=2)

    X_test, Y_test=generate_data(names,Constants.test_path,number_samples=1,seed=4,seed1=3,seed2=4)


# fig,ax=plt.subplots()
# for x in X:
#     ax.plot(x[1],'r')
# for x in X_test:
#     ax.plot(x[1],'b')


else:    
    train_data=extract_path_from_dir(Constants.train_path)
    test_data=extract_path_from_dir(Constants.test_path)
    start=time.time()
    s_train=[torch.load(f) for f in train_data]
    print(f"loading torch file take {time.time()-start}")
    s_test=[torch.load(f) for f in test_data]


    X_train=[s[0] for s in s_train]
    Y_train=[s[1] for s in s_train]
    X_test=[s[0] for s in s_test]
    Y_test=[s[1] for s in s_test]







# if __name__=='__main__':
    start=time.time()
    train_dataset = SonarDataset(X_train, Y_train)
    print(f"third loop {time.time()-start}")
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    start=time.time()
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    print(f"4th loop {time.time()-start}")

    train_dataloader = create_loader(train_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=False)
    val_dataloader=create_loader(val_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=False)

test_dataset = SonarDataset(X_test, Y_test)
test_dataloader=create_loader(test_dataset, batch_size=4, shuffle=False, drop_last=True)

inp, out=next(iter(test_dataset))

# model=geo_deeponet( 2, inp[1].shape[0], inp[2].shape[0],inp[4].shape[0])
# 9,18 or 10,18
# model=deeponet( 1,31, 100)
model=deeponet3( 1,31, 100) 
# model=deeponet4( 1,29, 100)

inp, out=next(iter(test_dataloader))
model(inp)
print(f" num of model parameters: {count_trainable_params(model)}")
# model([X[0].to(Constants.device),X[1].to(Constants.device)])

