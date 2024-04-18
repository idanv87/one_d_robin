import os
import sys
import math
from matplotlib.ticker import ScalarFormatter

import time

from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import sys
from scipy.interpolate import Rbf


from constants import Constants
from utils import  grf

from two_d_data_set import *
from draft import create_data, expand_function
from packages.my_packages import Gauss_zeidel, interpolation_2D
from geometry import *
from two_d_model import  Deeponet1, Deeponet0



# model=geo_deeponet( 2, 77,2, 99)

def original_domain():
    n=21
    x=np.linspace(0,1,n)
    y=np.linspace(0,1,n)
    X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij') 
    x1=x[:int(0.5*(n-1))+1]
    y1=y
    x2=x[int(0.5*(n-1)):]
    y2=y
    X1, Y1 = np.meshgrid(x1[1:], y1[1:-1], indexing='ij') 
    X2, Y2 = np.meshgrid(x2[:-1], y2[1:-1], indexing='ij')
    return X1,Y1,X2,Y2

def deeponet( G, X, Y, side):
    
    x_domain=X.flatten()
    y_domain=Y.flatten()
    int_points=np.vstack([x_domain,y_domain]).T
 
    experment_path=Constants.path+'runs/'
    X1,Y1,X2,Y2=original_domain()
    R1,R2=Restriction_matrix(X,Y,X1,Y1,X2,Y2)
    # runs/k=100F.shape
    if side:
        model_r=Deeponet1(2,[10,19])
        model_c=Deeponet1(2,[10,19])
        F=R1@G
        best_model=torch.load(experment_path+'2024.04.11.17.26.32best_model.pth')
        model_r.load_state_dict(best_model['model_state_dict']) 
        best_model=torch.load(experment_path+'2024.04.11.18.31.19best_model.pth')
        model_c.load_state_dict(best_model['model_state_dict']) 

    else: 
        model_r=Deeponet0(2,[10,19])
        model_c=Deeponet0(2,[10,19])
        F=R2@G  
        best_model=torch.load(experment_path+'2024.04.16.08.42.12best_model.pth')
        model_r.load_state_dict(best_model['model_state_dict']) 
        
        best_model=torch.load(experment_path+'2024.04.13.11.33.27best_model.pth')
        model_c.load_state_dict(best_model['model_state_dict']) 
       
    # k=50
        # p1=math.sqrt(np.var(res1.real))+1e-10
        # q1=math.sqrt(3.7095762168759046)
        # factor1=q1/p1
        # p2=math.sqrt(np.var(res1.imag))+1e-10
        # q2=math.sqrt(181.8135025897232)
        # factor2=q2/p2
    # if side:
    #     best_model=torch.load(experment_path+'2024.04.04.13.51.23best_model.pth')
    #     model_r.load_state_dict(best_model['model_state_dict']) 
    #     best_model=torch.load(experment_path+'2024.04.02.19.34.27best_model.pth')
    #     model_c.load_state_dict(best_model['model_state_dict']) 

    # else:   
    #     best_model=torch.load(experment_path+'2024.04.04.13.16.10best_model.pth')
    #     model_r.load_state_dict(best_model['model_state_dict']) 
        
    #     best_model=torch.load(experment_path+'2024.04.04.14.22.42best_model.pth')
    #     model_c.load_state_dict(best_model['model_state_dict']) 

    

    # x=np.linspace(0,0.5,10)
    # y=np.linspace(0,1,20)
    # domain=Rect(x,y)
    # xi=(domain.X).flatten()
    # yi=(domain.Y).flatten()
    
    # F=np.array(f_real(xi, yi))+1J*np.array(f_imag(xi, yi))

    start_time = time.time()
    with torch.no_grad():
       
        y1=torch.tensor(int_points,dtype=torch.float32).reshape(int_points.shape)
        f=torch.tensor(F.reshape(1,F.shape[0]),dtype=torch.cfloat).repeat(y1.shape[0],1)
        pred2=model_r([y1, f])+1J*model_c([y1, f])

    return pred2.numpy()
    # for j in range(len(x_domain)):
       
    #         X_test_i.append([
    #                     torch.tensor([x_domain[j],y_domain[j]], dtype=torch.float32), 
    #                      torch.tensor(F, dtype=torch.cfloat),
    #                      ])
    #         Y_test_i.append(torch.tensor(0, dtype=torch.float32))

    
    # test_dataset = SonarDataset(X_test_i, Y_test_i)
    # test_dataloader=create_loader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # coords=[]
    # prediction=[]
    # with torch.no_grad():    
    #     for input,output in test_dataloader:
    #         coords.append(input[0])
    #         prediction.append(model(input))

    # coords=np.squeeze(torch.cat(coords,axis=0).numpy())
    # prediction=torch.cat(prediction,axis=0).numpy()

    # return prediction

# def network(model, func, J, J_in, hint_init):
#     A = (-L - Constants.k* scipy.sparse.identity(L.shape[0]))
#     ev,V=scipy.sparse.linalg.eigs(-L,k=15,return_eigenvectors=True,which="SR")

#     b=func(x_domain, y_domain)
#     solution=scipy.sparse.linalg.spsolve(A, b)
#     predicted=deeponet(model, func)
#     # print(np.linalg.norm(solution-predicted)/np.linalg.norm(solution))


#     if hint_init:
#         x=deeponet(model, func)
#     else:
#         x=x=deeponet(model, func)*0

#     res_err=[]
#     err=[]
#     k_it=0

#     for i in range(1000):
#         x_0 = x
#         k_it += 1
#         theta=1
       
#         if (((k_it % J) in J_in) and (k_it > J_in[-1])):
            
#             factor = np.max(abs(sample[0]))/np.max(abs(A@x_0-b))
#             # factor=np.max(abs(grf(F, 1)))/np.max(abs(A@x_0-b))
#             x_temp = x_0*factor + \
#             deeponet(model, interpolation_2D(x_domain,y_domain,(b-A@x_0)*factor )) 
            
#             x=x_temp/factor
            
#             # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

#         else:    
#             x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]



       
#         print(np.linalg.norm(A@x-b)/np.linalg.norm(b))
#         res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))
#         err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))
#         if (res_err[-1] < 1e-13) and (err[-1] < 1e-13):
#             return err, res_err
#         else:
#             pass


   
#     return err, res_err

# def run_hints(func, J, J_in, hint_init):
#     return network(model, func, J, J_in, hint_init)


# def plot_solution( path, eps_name):
#     e_deeponet, r_deeponet= torch.load(path)
    
#     fig3, ax3 = plt.subplots()   # should be J+1
#     fig3.suptitle(F'relative error, \mu={mu}, \sigma={sigma} ')

#     ax3.plot(e_deeponet, 'g')
#     # ax3.plot(r_deeponet,'r',label='res.err')
#     # ax3.legend()
#     ax3.set_xlabel('iteration')
#     ax3.set_ylabel('error')
#     ax3.text(0.9, 0.1, f'final_err={e_deeponet[-1]:.2e}', transform=ax3.transAxes, fontsize=6,
#              ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
#     fig3.savefig(eps_name+'errors.eps', format='eps', bbox_inches='tight')
#     plt.show(block=True)
#     return 1


# torch.save(run_hints(func, J=J, J_in=[0], hint_init=True), Constants.outputs_path+'J='+str(J)+'k='+str(Constants.k)+'errors.pt')
# # plot_solution(Constants.outputs_path+'J='+str(J)+'k='+str(Constants.k)+'errors.pt', 'J='+str(J)+'k='+str(Constants.k)+'errors.pt')















