from initial import *

class softmax(nn.Module):
    def __init__(self, W):
        super(softmax, self).__init__()
        self.W = [Variable(torch.from_numpy(w).type(dtype), requires_grad=True) for w in W]

    def forward(self, x, y):
        D = x
        for i in range(len(self.W)):
            D = torch.matmul(D,self.W[i])
            D = torch.sigmoid(D)

        D = torch.matmul(D,self.W[-1])

        # calculate loss for the loss function and L2 regularizer
        D_max,_ = torch.max(D,dim = 1, keepdim = True)
        D = D-D_max
        A = torch.log(torch.sum(torch.exp(D),dim = 1))
        B = torch.sum(D*y,dim=1)
        Phi = torch.sum(A-B)
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))
        return (Phi,L2)


