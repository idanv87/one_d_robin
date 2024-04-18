import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import multiprocessing
import timeit
import datetime
import os
import time
import torch
from scipy import interpolate
from packages.my_packages import interpolation_2D, Restriction_matrix, Gauss_zeidel, gmres, Dx_backward, Dx_forward
# from hints import deeponet
from utils import norms, calc_Robin, solve_subdomain, solve_subdomain2, grf
import random
from random import gauss
import scipy
from scipy.sparse import csr_matrix, kron, identity
from scipy.linalg import circulant
from scipy.sparse import block_diag
from scipy.sparse import vstack

# from jax.scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve, cg

from two_d_model import NN0, NN1, NN11, NN00,NN, deeponet3, deeponet4
import timeit


from constants import Constants
from scipy import signal
import matplotlib.pyplot as plt

def load_models():
    experment_path=Constants.path+'runs/'
    model0=deeponet3(1,31,100)
    best_model=torch.load(experment_path+'2024.04.16.13.14.53best_model.pth')
    model0.load_state_dict(best_model['model_state_dict']) 
    model00=deeponet3(1,31,100)
    best_model=torch.load(experment_path+'2024.04.17.17.26.09best_model.pth')
    model00.load_state_dict(best_model['model_state_dict'])  
    model1=deeponet4(1,29,100)
    best_model=torch.load(experment_path+'2024.04.16.16.24.49best_model.pth')
    model1.load_state_dict(best_model['model_state_dict'])  
    model11=deeponet4(1,29,100)
    best_model=torch.load(experment_path+'2024.04.16.18.26.48best_model.pth')
    model11.load_state_dict(best_model['model_state_dict'])
    return model0,model1,model00,model11


class domain:
    def __init__(self,x,side):
        self.x=x 
        self.dx=x[1]-x[0]
        self.side=side
        self.D=self.calc_D_x()
        
    def  calc_D_x(self):   
        Nx = len(self.x[:-1])
        kernel = np.zeros((Nx, 1))
        kernel[-1] = 1.
        kernel[0] = -2.
        kernel[1] = 1.
        D2 = circulant(kernel).astype(complex)
        D2[0, -1] = 0.
        D2[-1, 0] = 0.
        if self.side:
            D2[-1,-1]=-2-2*self.dx*Constants.l
            D2[-1,-2]=2
        else:    
            D2[0,0]=-2-2*self.dx*Constants.l
            D2[0,1]=2    
        return csr_matrix(D2/self.dx/self.dx   )
    
    def solver(self,f,g,res=0):
        bc=np.zeros(len(f)).astype(complex)
        if self.side:
            bc[-1]=-2*g/self.dx
        else:
            bc[0]=-2*g/self.dx    
        # return scipy.sparse.linalg.spsolve(self.D+Constants.k*scipy.sparse.identity(self.D.shape[0]),f+bc)
        if res:
            return self.D+Constants.k*scipy.sparse.identity(self.D.shape[0]), f,bc
        else:    
            return self.D+Constants.k*scipy.sparse.identity(self.D.shape[0]), f+bc
    
   
    
        

             

def laplacian_matrix(x):
        dx=x[1]-x[0]
        Nx = len(x[1:-1])
        kernel = np.zeros((Nx, 1))
        kernel[-1] = 1.
        kernel[0] = -2.
        kernel[1] = 1.
        D2 = circulant(kernel)
        D2[0, -1] = 0.
        D2[-1, 0] = 0.
        return csr_matrix(D2/dx/dx)+Constants.k*scipy.sparse.identity(D2.shape[0])

def dd_block(n,u_global,s0,s1,f):
    u1=u_global[1:int((n-1)/2)]
    u0=u_global[int((n-1)/2)-1:-1]
    
    g=Dx_forward(u0.real,s0.dx)+Constants.l*u0[0]
    A1,G1=s1.solver(f[1:int((n-1)/2)],g)
    res1=-A1@u1+G1
    corr1=scipy.sparse.linalg.spsolve(A1,res1).real
    u1=u1+corr1
    
    g=(-Dx_backward(u1.real,s1.dx)+Constants.l*u1[-1])
    A0,G0=s0.solver(f[int((n-1)/2)-1:-1],g)
    res0=-A0@u0+G0
    corr0=scipy.sparse.linalg.spsolve(A0,res0).real
    u0=u0+corr0
    return u0,u1,np.concatenate([[0],u1[:-1],u0,[0]])

n=61
x=np.linspace(0,1,n)


# f=(np.sin(math.pi*2*x))
f=grf(x, 500,seed=0)[0]
solution=scipy.sparse.linalg.spsolve(laplacian_matrix(x),f[1:-1])
u_global=np.random.rand(len(x))*0

# u0=u[1:-1]
# for k in range(10):
#     u0=Gauss_zeidel(laplacian_matrix(x).todense(),f[1:-1],u0)[0]
#     print(np.linalg.norm(u0))
    
    
x1=x[:int((n-1)/2)]
x0=x[int((n-1)/2)-1:]
u1=u_global[1:int((n-1)/2)]
u0=u_global[int((n-1)/2)-1:-1]


s1=domain(x1,1)
s0=domain(x0,0)
f1=f[1:int((n-1)/2)]
f2=f[int((n-1)/2)-1:-1]

J=2
model0,model1,model00,model11=load_models()
global_err=[]

for l in range(1000):
    if (l%J) == 0:
        
        temp=Gauss_zeidel(laplacian_matrix(x).todense(),f[1:-1],u_global[1:-1])[0].real
        u_global=np.concatenate([[0],temp,[0]])
        res=-laplacian_matrix(x)@u_global[1:-1]+f[1:-1]

        
    else:    
        u1=u_global[1:int((n-1)/2)]
        u0=u_global[int((n-1)/2)-1:-1]

        res=-laplacian_matrix(x)@u_global[1:-1]+f[1:-1]
        f1=res[:int((n-1)/2)-1]
        f0=res[int((n-1)/2)-2:]
        
        for k in range(20):

            g=Dx_forward(u0.real,s0.dx)+Constants.l*u0[0]
            factorg=0.14/(abs(g.real)+1e-12)
            factorf=0.14/(np.std(f1)+1e-12)
            A1,G1=s1.solver(f1,g)
            u1=NN(x1[1:],f1*0,g*factorg,model1)/factorg+NN(x1[1:],f1*factorf,g*0, model11)/factorf
            # u1=scipy.sparse.linalg.spsolve(A1, G1).real
            
            g=(-Dx_backward(u1.real,s1.dx)+Constants.l*u1[-1])
            factorg=0.14/(abs(g.real)+1e-12)
            factorf=0.14/(np.std(f0)+1e-12)
            A0,G0=s0.solver(f0,g)
            u0=NN(x0[:-1],f0*0, g*factorg,model0)/factorg+NN(x0[:-1],f0*factorf,g*0, model00)/factorf
            # u0=scipy.sparse.linalg.spsolve(A0,G0).real
            # u0=NN0(x0[:-1],f[int((n-1)/2)-1:-1]*factor0,g*factor0)/factor0
               
        corr=np.concatenate([[0],u1,u0[1:],[0]])
        u_global=u_global+corr
    global_err.append(np.linalg.norm(res)/np.linalg.norm(f[1:-1]))
    print(np.linalg.norm(res))    
    # print(np.linalg.norm(u1-0*np.sin(math.pi*x1[1:])))

torch.save({'err':global_err, 'x':x[1:-1],'u':u_global[1:-1],'solution':solution}, Constants.outputs_path+str('kJM=110_2_50')+'tab1.pt')    
# plt.show()
print((x[1]-x[0])**2)    



    #     u0,u1, u_global=dd_block(n,u_global,s0,s1,f)
    # else:
    #     u_global=np.concatenate([[0],Gauss_zeidel(laplacian_matrix(x).todense(),f[1:-1],u_global[1:-1])[0],[0]])
    # print(np.linalg.norm(u_global))











     












