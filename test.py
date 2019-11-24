from datasetloader_new import *

# print(datasetloader.load_data("AwA"))

x,y,model = load_data("AwA")

weight_matrix,pre_act = train(x,y,model,3000,0.003)
print(weight_matrix)
print(pre_act)