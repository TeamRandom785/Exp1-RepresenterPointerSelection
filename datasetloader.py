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


#Defining Softmax function which is similar to Pytorch's Softmax but this Softmax function can calculate the loss
#w.r.t loss function and also the L2 regularizer. This is used to train the pretrained model's last layer to converge
#completely at its Global Minimum. Also, this Softmax function used LogSumTrick to handle numerical underflow 
class softmax(nn.Module):
    def __init__(self,W):
        super(softmax,self).__init__()
        self.W = Variable(torch.from_numpy(W).type(dtype),requires_grad=True)
     
    #Here x is the loss function and y is the L2 regularizer
    def forward(self, x, y):
        
        #Calculating the matrix mult between x and W 
        Y = torch.matmul(x, self.W)
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
        model = softmax(W_36)
        if device == True:
            model.cuda()
        return (np.concatenate([intermediate_output_34,np.ones((intermediate_output_34.shape[0],1))],axis = 1), intermediate_output_36, model)
    
    elif dataset == "AwA":
       
        input_file = open("AwA/weight_bias.pickle", "rb")
        [weight,bias] = pickle.load(input_file, encoding = 'latin1')
        train_feature = np.squeeze(np.load('AwA/train_feature_awa.npy'))
        train_output = np.squeeze(np.load('Awa/train_output_awa.npy'))
        weight = np.transpose(np.concatenate([weight,np.expand_dims(bias,1)],axis = 1))
        train_feature = np.concatenate([train_feature,np.ones((train_feature.shape[0],1))],axis = 1)
        train_output = softmax_np(train_output)
        model = softmax(weight)
        if device == True:
            model.cuda()
        return (train_feature,train_output,model)



