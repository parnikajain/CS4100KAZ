import numpy as np



W = np.array([
		[0.5, -1, 1],
		[0.1, 0, 2]
	])

def sigmoid(x):
	return 1/(1+np.exp(-x))

x = np.array([1, 0.5, -1])

y = np.array([0,1])

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x))

def forward(x):
	z1 = np.dot(W, x)
	z2 = sigmoid(z1)
	y_hat = softmax(z2)
	return z1, z2, y_hat

def softmax_derivative(x):
	y = softmax(x)
	derivative = np.zeros((len(x), len(x)))
	for i in range(len(x)):
		for j in range(len(x)):
			if i==j:
				derivative[i,j] = y[i]*(1-y[i])
			else:
				derivative[i,j] = -y[i]*y[j]

	return derivative


def loss(y, y_hat):
	return -np.log(np.dot(y, y_hat))


z1, z2, y_hat = forward(x)


print( np.outer(np.matmul((-y*(1/y_hat)), softmax_derivative(z2)) * (z2*(1-z2)), x) )

