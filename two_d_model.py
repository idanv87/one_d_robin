import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from constants import Constants

import numpy as np
import os
import sys






class fc(torch.nn.Module):
    def __init__(self, input_shape, output_shape, num_layers, activation_last):
        super().__init__()
        self.activation_last=activation_last
        self.input_shape = input_shape
        self.output_shape = output_shape
        n = 100
        # self.activation = torch.nn.ReLU()
        self.activation = torch.nn.Tanh()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=self.input_shape, out_features=n, bias=True)])
        output_shape = n

        for j in range(num_layers):
            layer = torch.nn.Linear(
                in_features=output_shape, out_features=n, bias=True)
            # initializer(layer.weight)
            output_shape = n
            self.layers.append(layer)

        self.layers.append(torch.nn.Linear(
            in_features=output_shape, out_features=self.output_shape, bias=True))

    def forward(self, y):
        s=y
        for layer in self.layers:
            s = layer(self.activation(s))
        if self .activation_last:
            return self.activation(s)
        else:
            return s


class deeponet(nn.Module):
    # good parameters: n_layers in deeponet=4,n_layers in geo_deeponet=10, infcn=100, ,n=5*p, p=100

    def __init__(self, dim, num_hot_spots, p):
        super().__init__()
        n_layers = 4
        self.n = p
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.branch1 = fc(num_hot_spots*2, self.n, n_layers,activation_last=False)
        self.trunk1 = fc(dim, p,  n_layers, activation_last=True)
        self.c_layer = fc( self.n, p, n_layers, activation_last=False)
        self.c2_layer =fc( 2*num_hot_spots+1, 1, 3, False) 


    def forward(self, X):
        y,f_real, f_imag=X
        f=torch.cat((f_real,f_imag),dim=1)
        branch = self.c_layer(self.branch1(f))
        trunk = self.trunk1(torch.unsqueeze(y,1))
        alpha = torch.squeeze(self.c2_layer(torch.cat((f,torch.unsqueeze(y,1)),dim=1)))
        return torch.sum(branch*trunk, dim=-1, keepdim=False)+alpha
        






    
    
class deeponet3(nn.Module):
    # good parameters: n_layers in deeponet=4,n_layers in geo_deeponet=10, infcn=100, ,n=5*p, p=100

    def __init__(self, dim, num_hot_spots, p):
        super().__init__()
        n_layers = 4
        self.n = p

        self.branch1 = fc(num_hot_spots, p, n_layers,activation_last=False)
        self.branch2 = fc(2, p, n_layers,activation_last=False)
        self.trunk1 = fc(dim, p,  n_layers, activation_last=True)
        self.trunk2 = fc(dim, p,  n_layers, activation_last=True)
        self.bias1 =fc( num_hot_spots+1, 1, 3, False) 
        self.bias2 =fc( 2+1, 1, 3, False) 


    def forward(self, X):
        y,f,g1=X
        g=torch.cat((g1.real.reshape((g1.shape[0],1)),g1.imag.reshape((g1.shape[0],1))),dim=1)
        
        branch1 = self.branch1(f)
        branch2 = self.branch2(g)
        trunk1 = self.trunk1(torch.unsqueeze(y,1))
        trunk2 = self.trunk2(torch.unsqueeze(y,1))
        bias1 = torch.squeeze(self.bias1(torch.cat((f,torch.unsqueeze(y,1)),dim=1)))
        bias2= torch.squeeze(self.bias2(torch.cat((g,torch.unsqueeze(y,1)),dim=1)))
        return torch.sum(branch1*trunk1, dim=-1, keepdim=False)+bias1+torch.sum(branch2*trunk2, dim=-1, keepdim=False)+bias2
    

class deeponet4(nn.Module):
    # good parameters: n_layers in deeponet=4,n_layers in geo_deeponet=10, infcn=100, ,n=5*p, p=100

    def __init__(self, dim, num_hot_spots, p):
        super().__init__()
        n_layers = 4
        self.n = p

        self.branch1 = fc(num_hot_spots, p, n_layers,activation_last=False)
        self.branch2 = fc(2, p, n_layers,activation_last=False)
        self.trunk1 = fc(dim, p,  n_layers, activation_last=True)
        self.trunk2 = fc(dim, p,  n_layers, activation_last=True)
        self.bias1 =fc( num_hot_spots+1, 1, 3, False) 
        self.bias2 =fc( 2+1, 1, 3, False) 


    def forward(self, X):
        y,f,g1=X
        g=torch.cat((g1.real.reshape((g1.shape[0],1)),g1.imag.reshape((g1.shape[0],1))),dim=1)
        
        branch1 = self.branch1(f)
        branch2 = self.branch2(g)
        trunk1 = self.trunk1(torch.unsqueeze(y,1))
        trunk2 = self.trunk2(torch.unsqueeze(y,1))
        bias1 = torch.squeeze(self.bias1(torch.cat((f,torch.unsqueeze(y,1)),dim=1)))
        bias2= torch.squeeze(self.bias2(torch.cat((g,torch.unsqueeze(y,1)),dim=1)))
        return torch.sum(branch1*trunk1, dim=-1, keepdim=False)+bias1+torch.sum(branch2*trunk2, dim=-1, keepdim=False)+bias2


def return_model0(n=0):
    experment_path=Constants.path+'runs/'
    model=deeponet3(1,31,100)
    best_model=torch.load(experment_path+'2024.04.16.13.14.53best_model.pth')
    model.load_state_dict(best_model['model_state_dict'])     
    return model



def NN0(x,f,g):
    with torch.no_grad():
       
        y1=torch.tensor(x,dtype=torch.float32).reshape(x.shape)
        f=torch.tensor(f.reshape(1,f.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        g=torch.tensor(g.reshape(1,1),dtype=torch.cfloat).repeat(y1.shape[0],1)
        pred=return_model0()([y1,f,g])
    return pred.numpy()
    
def return_model1(n=1):
    experment_path=Constants.path+'runs/'
    model=deeponet4(1,29,100)
    best_model=torch.load(experment_path+'2024.04.16.16.24.49best_model.pth')
    model.load_state_dict(best_model['model_state_dict'])     
    return model

def NN1(x,f,g):
    with torch.no_grad():
       
        y1=torch.tensor(x,dtype=torch.float32).reshape(x.shape)
        f=torch.tensor(f.reshape(1,f.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        g=torch.tensor(g.reshape(1,1),dtype=torch.cfloat).repeat(y1.shape[0],1)
        pred=return_model1()([y1,f,g])
    return pred.numpy()
  
def return_model11(n=1):
    experment_path=Constants.path+'runs/'
    model=deeponet4(1,29,100)
    best_model=torch.load(experment_path+'2024.04.16.18.26.48best_model.pth')
    model.load_state_dict(best_model['model_state_dict'])     
    return model

def NN11(x,f,g):
    with torch.no_grad():
       
        y1=torch.tensor(x,dtype=torch.float32).reshape(x.shape)
        f=torch.tensor(f.reshape(1,f.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        g=torch.tensor(g.reshape(1,1),dtype=torch.cfloat).repeat(y1.shape[0],1)
        pred=return_model11()([y1,f,g])
    return pred.numpy()      


def return_model00(n=0):
    experment_path=Constants.path+'runs/'
    model=deeponet3(1,31,100)
    best_model=torch.load(experment_path+'2024.04.17.17.26.09best_model.pth')
    model.load_state_dict(best_model['model_state_dict'])     
    return model



def NN00(x,f,g,model):
    with torch.no_grad():
       
        y1=torch.tensor(x,dtype=torch.float32).reshape(x.shape)
        f=torch.tensor(f.reshape(1,f.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        g=torch.tensor(g.reshape(1,1),dtype=torch.cfloat).repeat(y1.shape[0],1)
        pred=return_model00()([y1,f,g])
    return pred.numpy()

def NN(x,f,g,model):
    with torch.no_grad():
       
        y1=torch.tensor(x,dtype=torch.float32).reshape(x.shape)
        f=torch.tensor(f.reshape(1,f.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        g=torch.tensor(g.reshape(1,1),dtype=torch.cfloat).repeat(y1.shape[0],1)
        pred=model([y1,f,g])
    return pred.numpy()