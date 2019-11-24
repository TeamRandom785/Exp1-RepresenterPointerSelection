''' This script is used to load the dataset 
    Import this file in the following way - import datasetloader
    Classes - 
    1.  Softmax - 
            
            Description - 
            This function is similar to Pytorch's Softmax but calculates Softmax value for Loss function
            and also for the L2 regularizer 
            
            Usage - 
            var = datasetloader.Softmax(Weights)
            var(loss function value,regularizer value)
    Functions -
    1.  Softmax_np

            Description - 
            Softmax for numpy variables

            Usage - 
            datasetloader.softmax_np(x)

    2.  load_data

            Description - 
            This function is used to load the train features, train output and the trained model. 

            Arguments - 
            Dataset name - 'AwA' or 'Cifar'

            Return value - 
            training features, training output labels and pretrained model

            Usage - 
            datasetloader.load_data(dataset_name)
'''

import time
import sys
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
import math
import pickle
import os
dtype = torch.cuda.FloatTensor
device = torch.cuda.is_available()
device = "cuda" if device else "cpu"


number_linear_layers = 2

#Defining Softmax function which is similar to Pytorch's Softmax but this Softmax function can calculate the loss
#w.r.t loss function and also the L2 regularizer. This is used to train the pretrained model's last layer to converge
#completely at its Global Minimum. Also, this Softmax function used LogSumTrick to handle numerical underflow 
class softmax(nn.Module):
    def __init__(self,W):
        super(softmax,self).__init__()
        self.W = [Variable(torch.from_numpy(w).type(dtype), requires_grad=True) for w in W]
     
    #Here x is the loss function and y is the L2 regularizer
    def forward(self, x, y):
        Y = x
        for i in range(len(self.W)):
            Y = torch.matmul(Y, self.W[i])
            Y = torch.sigmoid(Y)

        Y = torch.matmul(Y, self.W[-1])
        
        Y_max,_ = torch.max(Y, dim=1, keepdim = True)
        Y = Y - Y_max
        
        tmp_var_a = torch.log(torch.sum(torch.exp(Y),dim = 1))
        tmp_var_b = torch.sum(Y*y, dim = 1)
        
        sigma = torch.sum(tmp_var_a - tmp_var_b)
        reg_W = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(reg_W,reg_W))
        return (sigma,L2)

#Softmax function for numpy variables
def softmax_np(x):
    e_x = np.exp(x - np.max(x,axis = 1,keepdims = True))
    return e_x / e_x.sum(axis = 1,keepdims = True)

def load_data(dataset):
    if dataset == "Cifar":
        
        input_file = open("Cifar/weight_323436.pkl", "rb")
        [W_32,W_34,W_36,intermediate_output_32,intermediate_output_34,intermediate_output_36] = pickle.load(input_file, encoding = 'latin1')
        print((softmax_np(np.matmul(np.concatenate([intermediate_output_34,np.ones((intermediate_output_34.shape[0],1))],axis = 1),W_36))-intermediate_output_36)[:5,:])
        print(intermediate_output_36[:5,:])
        print('done loading')
        model = softmax([W_36])
        model.to(device)
        return (np.concatenate([intermediate_output_34,np.ones((intermediate_output_34.shape[0],1))],axis = 1), intermediate_output_36, model)
    
    elif dataset == "AwA":
       
        input_file = open("AwA/weight_bias.pickle", "rb")
        [weight,bias] = pickle.load(input_file, encoding = 'latin1')
        train_feature = np.squeeze(np.load('AwA/train_feature_awa.npy'))
        train_output = np.squeeze(np.load('Awa/train_output_awa.npy'))
        weight = np.transpose(np.concatenate([weight,np.expand_dims(bias,1)],axis = 1))
        train_feature = np.concatenate([train_feature,np.ones((train_feature.shape[0],1))],axis = 1)
        train_output = softmax_np(train_output)
        model = softmax([weight])
        model.to(device)
        return (train_feature,train_output,model)




def backtracking_line_search(model, x, y, cur_loss, tau, N, lmbda):
    t = 10.0
    W_init = [model.W[i].data for i in range(number_linear_layers)]
    grad = [model.W[i].grad for i in range(number_linear_layers)]
    grad_all = torch.cat(grad, dim=0)

    # Uniformly decreasing the step among all the linear layers
    while (t >= e-10): 
        for i in range(number_linear_layers):
            model.W[i] = Variable(W_init[i] - t[i]*grad[i], requires_grad = True)
        (Phi, L2) = model(x, y)
        loss = (Phi/N + lmbda*L2).item()
        if (cur_loss - loss >= t * torch.norm(grad_all)**2 / 2): break
        t *= tau


# model.W is a list of several linear layers
def train(model, X, Y, n_epochs, lmbda):
    x = Variable(torch.FloatTensor(X))
    y = Variable(torch.FloatTensor(Y))
    x, y = x.to(devise), y.to(devise)

    N = len(Y)
    min_loss = 1000000.0
    optimizer = optim.SGD(model.W, lr=1.0)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        (Phi, L2) = model(x, y)
        loss = Phi/N + lmbda*L2
        loss.backward()

        # Saving the last(!) linear layer with the smallest loss
        grad_loss = torch.mean(torch.abs(model.W[-1].grad)).item()
        if (grad_loss < min_loss):
            if (epoch == 0): init_loss = grad_loss
            min_loss = grad_loss
            min_W_last = model.W[-1].data
            if (min_loss < init_loss/200):
                print("Stopped at epoch: {}".format(epoch))
                break

        backtracking_line_search(model, x, y, loss.item(), 0.5, N, lmbda)

    preactivation_1 = torch.matmul(x, model.W[0])
    return min_W_last, preactivation_1


