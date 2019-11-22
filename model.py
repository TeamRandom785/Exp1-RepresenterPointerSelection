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
