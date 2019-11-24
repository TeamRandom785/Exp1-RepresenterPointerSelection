from datasetloader_new import *
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
dtype = torch.FloatTensor
device = torch.cuda.is_available()
device = "cuda" if device else "cpu"

def calcAlpha(x, y, N, w_last, preact,lmbd):
    temp = torch.matmul(x,Variable(w_last))
    softmax_value = softmax_torch(temp,N)

    # derivate of softmax cross entropy 
    alpha_1 = softmax_value - y
    alpha_1 = torch.div(alpha_1,(-2.0*lmbd*N))

    # Calculation of alpha 2
    alpha_2 = np.matmul(alpha_1.data.numpy(), w_last.data.numpy().T)
    
    # print(alpha_1.shape, w_last.shape,alpha_2.shape)
    return alpha_1.data.numpy(),alpha_2