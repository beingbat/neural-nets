import numpy as np

from common import *


class NeuralNetwork:
    def __init__(
        self, network_shape, lr=0.01, loss_fn="quadratic", weight_init="uniform"
    ):
        self.weights = [
            weights_initialization(weight_init)(
                shape=(network_shape[i], network_shape[i + 1])
            )
            for i in range(len(network_shape) - 1)
        ]
        self.biases = [
            weights_initialization("uniform")(shape=(1, network_shape[i + 1]))
            for i in range(len(network_shape) - 1)
        ]

        self.network_shape = network_shape
        self.LR = lr

        self.reset_intermediate_gradients()
        self.reset_weight_gradients()
        self.loss, self.loss_d = loss(loss_fn)

        self.__unreduced_final_act_prime = None
        self.__unreduced_costd = None

        self.activation_setup()
        self.regularization_setup()
        self.optimizer_setup()

    # OPTIMIZERS

    def optimizer_setup(self, optimizer_name="sgd", args=()):
        self.optimizer = optimizer_name
        match optimizer_name:
            case "sgd_momentum":
                self.momentum = args[0]
            case "adam":
                self.beta1 = args[0]
                self.beta2 = args[1]
                self.original_beta1 = args[0]
                self.original_beta2 = args[1]
                self.adam_mean_weights = [np.zeros(i.shape) for i in self.weights]
                self.adam_mean_bias = [np.zeros(i.shape) for i in self.biases]
                self.adam_variance_weights = [np.zeros(i.shape) for i in self.weights]
                self.adam_variance_bias = [np.zeros(i.shape) for i in self.biases]
            # case _: # sgd

    def decay_betas(self):
        if self.optimizer == "adam":
            self.beta1 *= self.original_beta1
            self.beta2 *= self.original_beta2

    # WEIGHTS AND GRADIENTS

    # This method is used for keeping intermediate gradients (not the weight gradients)
    def reset_intermediate_gradients(self):
        self.act_prime = [
            np.zeros((self.network_shape[i + 1]))
            for i in range(len(self.network_shape) - 1)
        ]
        self.acts = [
            np.zeros(self.network_shape[i]) for i in range(len(self.network_shape))
        ]
        self.costs_d = np.zeros(self.network_shape[-1])

    # This method initializes weights graidents
    def reset_weight_gradients(self):
        self.weights_grad = [np.zeros(i.shape) for i in self.weights]
        self.bias_grad = [np.zeros(i.shape) for i in self.biases]

    # ACTIVATIONS

    def activation_setup(self, activations=("sigmoid",)):
        if len(activations) == 2:
            self.initialize_activations(activations[0], activations[1])
        else:
            self.initialize_activations(activations[0], activations[0])
        self.current_act = self.intermediate_act
        self.current_act_prime = self.intermediate_act_prime

    def initialize_activations(self, intermediate_activations, final_activation):
        match intermediate_activations:
            case "tanh":
                self.intermediate_act = tanh
                self.intermediate_act_prime = tanh_prime
            case "relu":
                self.intermediate_act = relu
                self.intermediate_act_prime = relu_prime
            case _:
                self.intermediate_act = sigmoid
                self.intermediate_act_prime = sigmoid_prime

        match final_activation:
            case "tanh":
                self.final_act = tanh
                self.final_act_prime = tanh_prime
            case "relu":
                self.final_act = relu
                self.final_act_prime = relu_prime
            case "softmax":
                self.final_act = softmax
                self.final_act_prime = softmax_prime
            case _:
                self.final_act = sigmoid
                self.final_act_prime = sigmoid_prime

    def cost_derivative_for_softmax(self):
        costd = self.__unreduced_costd
        if len(costd.shape) == 1:
            costd = costd.reshape(1, -1)
        prime = self.__unreduced_final_act_prime

        for i in range(len(costd)):
            for j in range(len(costd[i])):
                prime[i, :, j] *= costd[i, j]
        return np.mean(
            np.sum(prime, axis=-1).reshape(prime.shape[0], -1), axis=0
        ).reshape(1, -1)

    # REGULARIZATIONS

    def regularization_setup(self, regL=None, reglambda=None, dropout=0.0):
        self.dropout_threshold = dropout
        self.regularization_type = regL
        self.regularization_lambda = reglambda

    def initialize_dropout_mask(self):
        self.dropout_mask = [
            (np.random.sample(i.shape) >= self.dropout_threshold).astype(int)
            for i in self.weights
        ]
        self.dropout_mask[-1] = np.ones(self.dropout_mask[-1].shape)

    def regularization_loss(self):
        if self.regularization_type == "l1":
            return self.regularization_lambda * np.sum(
                np.array([np.sum(i) for i in self.weights])
            )
        elif self.regularization_type == "l2":
            return self.regularization_lambda * np.sum(
                np.array([np.sum(i**2) for i in self.weights])
            )
        return 0

    def regularization_loss_d(self, weights):
        if self.regularization_type == "l1":
            return np.sign(weights) * self.regularization_lambda
        elif self.regularization_type == "l2":
            return weights * self.regularization_lambda / 2
        return 0

    def eval(self):
        self.dropout_threshold = 0

    # Called from train() when epoch gets completed
    def epoch_complete(self):
        self.decay_betas()

    # NETWORK PASSES

    def forward(self, x, y):
        if self.dropout_threshold != 0:
            self.initialize_dropout_mask()
        self.current_act = self.intermediate_act
        self.current_act_prime = self.intermediate_act_prime
        self.acts[0] = np.mean(x, axis=0)
        for i in range(len(self.weights)):
            if i + 1 == len(self.weights):
                self.current_act = self.final_act
                self.current_act_prime = self.final_act_prime

            # Z = X*W + b
            if self.dropout_threshold != 0:
                x = np.dot(x, self.weights[i] * self.dropout_mask[i]) + self.biases[i]
            else:
                x = np.dot(x, self.weights[i]) + self.biases[i]
            # dA/dZ
            temp = self.current_act_prime(np.squeeze(x.copy()))
            if i + 1 == len(self.weights):
                self.__unreduced_final_act_prime = temp.copy()
                if len(self.__unreduced_final_act_prime.shape) == 2:
                    self.__unreduced_final_act_prime = np.expand_dims(
                        self.__unreduced_final_act_prime, axis=0
                    )
            if len(temp.shape) > 1:
                temp = np.mean(temp, axis=0)
            self.act_prime[i] = temp
            # A = act(Z)
            x = self.current_act(np.squeeze(x))
            # dZ(n+1)/A(n-1)
            temp = x.copy()
            if len(temp.shape) > 1:
                temp = np.mean(temp, axis=0)
            self.acts[i + 1] = temp
            # X = A
        temp = self.loss_d(x, y)
        self.__unreduced_costd = temp.copy()
        if len(temp.shape) > 1:
            temp = np.mean(temp, axis=0)
        self.costs_d = temp
        return x, self.loss(x, y) + self.regularization_loss()

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
        chain_grad = self.costs_d.reshape(1, -1)
        # print("dC/dz: ", chain_grad*self.act_prime[-1])
        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1 and self.final_act.__eq__(softmax):
                chain_grad = self.cost_derivative_for_softmax()
            else:
                chain_grad *= self.act_prime[i]
            old_weights = self.weights[i].copy()

            drop_mask = 1
            # Check if dropout is enabled
            if self.dropout_threshold != 0:
                drop_mask = self.dropout_mask[i]

            if self.optimizer == "sgd_momentum":
                self.weights_grad[i] = self.LR * (
                    (1 - self.momentum)
                    * np.dot(self.acts[i].reshape(-1, 1), chain_grad)
                    + self.momentum * self.weights_grad[i]
                )
                self.bias_grad[i] = self.LR * (
                    (1 - self.momentum) * chain_grad + self.momentum * self.bias_grad[i]
                )
            elif self.optimizer == "adam":
                self.weights_grad[i] = np.dot(self.acts[i].reshape(-1, 1), chain_grad)
                self.bias_grad[i] = chain_grad

                self.adam_mean_bias[i] = (
                    self.beta1 * self.adam_mean_bias[i]
                    + (1 - self.beta1) * self.bias_grad[i]
                )
                self.adam_variance_bias[i] = self.beta2 * self.adam_variance_bias[i] + (
                    1 - self.beta2
                ) * (self.bias_grad[i] ** 2)
                self.adam_mean_weights[i] = (
                    self.beta1 * self.adam_mean_weights[i]
                    + (1 - self.beta1) * self.weights_grad[i]
                )
                self.adam_variance_weights[i] = self.beta2 * self.adam_variance_weights[
                    i
                ] + (1 - self.beta2) * (self.weights_grad[i] ** 2)
                alpha_adjusted = self.LR * np.sqrt(1 - self.beta2) / (1 - self.beta1)
                self.weights_grad[i] = (
                    alpha_adjusted
                    * self.adam_mean_weights[i]
                    / (np.sqrt(self.adam_variance_weights[i]) + 1e-6)
                )
                self.bias_grad[i] = (
                    alpha_adjusted
                    * self.adam_mean_bias[i]
                    / (np.sqrt(self.adam_variance_bias[i]) + 1e-6)
                )
            elif self.optimizer == "sgd":
                self.weights_grad[i] = self.LR * np.dot(
                    self.acts[i].reshape(-1, 1), chain_grad
                )
                self.bias_grad[i] = self.LR * chain_grad
            else:
                print(f"Invalid optimizer: {self.optimizer} selected.\n")
                return

            self.biases[i] -= self.bias_grad[i]
            self.weights[i] -= (
                self.weights_grad[i] + self.LR * self.regularization_loss_d(old_weights)
            ) * drop_mask
            chain_grad = np.dot(chain_grad, np.transpose(old_weights, (1, 0)))

        self.reset_intermediate_gradients()
