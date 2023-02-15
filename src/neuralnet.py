import numpy as np

class NeuralNetwork():
    def __init__(self, network_shape, lr=0.01, loss_fn="quadratic"):
        self.weights = [self.weights_initialization("u")(network_shape[i], network_shape[i+1]) for i in range(len(network_shape)-1)]
        self.biases = [self.weights_initialization("u")(1, network_shape[i+1]) for i in range(len(network_shape)-1)]
        
        self.loss, self.loss_d = self.get_loss(loss_fn)
        self.network_shape = network_shape
        self.reset_gradients()
        self.LR = lr

    def reset_gradients(self):
        self.act_prime = [np.zeros((self.network_shape[i+1])) for i in range(len(self.network_shape)-1)]
        self.acts = [np.zeros(self.network_shape[i]) for i in range(len(self.network_shape))]
        self.costs_d = np.zeros(self.network_shape[-1])
    
    def get_loss(self, loss):
        if loss == "BCE":
            return self.cross_entropy_loss, self.cross_entropy_loss_d
        elif loss == "MSE":
            return self.quadratic_loss, self.quadratic_loss_d
        else:
            return self.quadratic_loss, self.quadratic_loss_d
    
    def quadratic_loss(self, a,y): # MSE
        return 0.5*(y-a)**2

    def quadratic_loss_d(self, a,y):
        return a-y

    def cross_entropy_loss(self, a, y): 
        return -np.sum(y*np.log2(a+1e-8), axis=0)

    def cross_entropy_loss_d(self, a, y): # Gives same derivative as MSE and also cancels out last layer's activation derivative as well
        grad = -(y/(a+1e-8))+((1-y)/(1-a+1e-8))
        # print("\n", y, "\n", [np.round(i, 2) for i in a], "\n", grad, "\n")
        # print("a-y: ", a-y)
        return grad


    def weights_initialization(self, char):
        if char == "u":
            return np.random.randn
        else:
            return np.random.randn
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-np.float64(x)))

    def sigmoid_prime(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def combine_grads(self, grads, new_grad):
        if len(new_grad.shape) == 2:
            new_grad = np.mean(new_grad, axis=0)
        return ((self.iterations-1)/self.iterations)*grads + (1/self.iterations)*new_grad 

    def forward(self, x, y):
        self.acts[0] = np.mean(x, axis=0)
        for i in range(len(self.weights)): 
            # Z = X*W + b
            x = np.dot(x,self.weights[i])+self.biases[i]
            # dA/dZ
            temp = self.sigmoid_prime(np.squeeze(x.copy()))
            if len(temp.shape) > 1:
                temp = np.mean(temp, axis=0)
            self.act_prime[i] = temp
            # A = act(Z)
            x = self.sigmoid(np.squeeze(x))
            # dZ(n+1)/A(n-1)
            temp = x.copy()
            if len(temp.shape) > 1:
                temp = np.mean(temp, axis=0)
            self.acts[i+1] = temp
            # X = A
        temp = self.loss_d(x,y)
        if(len(temp.shape) > 1):
            temp = np.mean(temp, axis=0)
        self.costs_d = temp
        return x, self.loss(x,y)

    # Gradients for: Hidden Layers Count 2
    # Gradient for W3
    # dC/dA3 * dA3/dZ3 * dZ3/dW3
    # Gradient for W2
    # dC/dA3 * dA3/dZ3 * dZ3/dA2 * dA2/dZ2 * dZ2/dW2
    # Gradient for W1
    # dC/dA3 * dA3/dZ3 * dZ3/dA2 * dA2/dZ2 * dZ2/dA1 * dA1/dZ1 * dZ1/dW1

    # Where: 
    # dC/dA = Loss derivative (for quadratic loss [y-a])

    # dA/dZ = Activation function derivative (sigmoid prime)
    # dZn/dWn = X_(n-1)
    # These two can be calculated during the forward pass, as the activation function and its derivative are known and as well as input

    def backward(self):
        chain_grad = self.costs_d.reshape(1,-1)
        # print("dC/dz: ", chain_grad*self.act_prime[-1])
        for i in range(len(self.weights)-1, -1, -1):
            chain_grad *= self.act_prime[i]
            old_weights = self.weights[i].copy()
            self.weights[i] -= self.LR*np.dot(self.acts[i].reshape(-1,1),chain_grad)
            chain_grad = np.dot(chain_grad, np.transpose(old_weights, (1,0)))

        self.reset_gradients()