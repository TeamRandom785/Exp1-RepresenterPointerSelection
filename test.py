from datasetloader_new import *
import numpy as np 
from calculateAlpha import *
# print(datasetloader.load_data("AwA"))

x,y,model = load_data("AwA")
y = y.astype(float)
x, y, N, min_W_last, preactivation_1, lmbda = train(model,x,y,3,0.003)
# print(min_W_last)
# print(preactivation_1)
alpha_1,  alpha_2 = calcAlpha(x, y, N, min_W_last, preactivation_1, lmbda)
# print(alpha2)
