from datasetloader_new import *
import numpy as np 

# print(datasetloader.load_data("AwA"))

x,y,model = load_data("AwA")
y = y.astype(float)
weight_matrix,pre_act = train(model,x,y,3,0.003)
print(weight_matrix)
print(pre_act)
