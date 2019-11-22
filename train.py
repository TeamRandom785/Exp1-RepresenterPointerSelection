from initial import *

def backtracking_line_search(model, x, y, cur_loss, tau=0.5, N, lmbda):
	t = 10.0
	W_init = [model.W[i].data for i in range(number_linear_layers)]
	grad = [model.W[i].grad for i in range(number_linear_layers)]
	grad_all = torch.cat([model.W[1].grad, model.W[2].grad], dim=0)

	# Uniformly deacreasing the step among all the linear layers
	while (t >= e-10): 
		for i in range(number_linear_layers):
			model.W[i] = Variable(W_init[i] - t[i]*grad[i], requires_grad = True)
		(Phi, L2) = model(x, y)
		loss = (Phi/N + lmbda*L2).item()
		if (cur_loss - loss >= t * torch.norm(grad_all)**2 / 2): break
		t *= tau

def train(model, X, Y, n_epochs, lmbda):
	x = Variable(torch.FloatTensor(X))
	y = Variable(torch.FloatTensor(Y))
	x, y = x.to(devise), y.to(devise)

	N = len(Y)
	min_loss = 1000000.0
	optimizer = optim.SGD([model.W], lr=1.0)
	for epoch in range(n_epochs):
		optimizer.zero_grad()
		(Phi, L2) = model(x, y)
		loss = Phi/N + lmbda*L2
		loss.backward()

		# Saving the last(!) linear layer with the smallest loss
		grad_loss = torch.mean(torch.abs(model.W[0].grad)).item()
		if (grad_loss < min_loss):
			if (epoch == 0): init_loss = grad_loss
			min_loss = grad_loss
			min_W0 = model.W[0].data
			if (min_loss < init_loss/200):
				print("Stopped at epoch: {}".format(epoch))
				break

		backtracking_line_search(model, x, y, loss.item(), 0.5, N, lmbda)

	return min_W0
		


