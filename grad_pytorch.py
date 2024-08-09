import torch

W = torch.tensor([
		[0.5, -1, 1],
		[0.1, 0, 2]
	], requires_grad=True)

def sigmoid(x):
	return 1/(1+torch.exp(-x))

x = torch.tensor([1, 0.5, -1], requires_grad=True)

y = torch.tensor([0,1], dtype=torch.float)

def softmax(x):
	return torch.exp(x)/torch.sum(torch.exp(x))

def forward(x):
	z1 = torch.matmul(W, x)
	z2 = sigmoid(z1)
	y_hat = softmax(z2)
	return z1, z2, y_hat

def softmax_derivative(x):
	derivative = torch.zeros((len(x), len(x)))
	for i in range(len(x)):
		for j in range(len(y)):
			if i==j:
				derivative[i,j] = x[i]*(1-x[i])
			else:
				derivative[i,j] = -x[i]*x[j]

	return derivative



def loss(y, y_hat):
	return -torch.log(torch.matmul(y, y_hat))

z1, z2, y_hat = forward(x)
z1.retain_grad()
z2.retain_grad()
y_hat.retain_grad()

ce_loss = loss(y, y_hat)

ce_loss.backward()




print(W.grad)

