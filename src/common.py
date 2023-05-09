import numpy as np

# ------------------------------------------ WEIGHTS INITIALIZATIONS --------------------------------------- #


def weights_initialization(char):
    if char == "he":
        # Guassian Distribution with std sqrt(2/input_size)
        return he_weights
    elif char == "xavier":
        # unifrom between -1/sqrt(input_size) | 1/sqrt(input_size)
        return xavier_weights
    else:
        # uniform between -1 and 1
        return uniform_weights


def he_weights(shape):
    return np.random.randn(shape[0], shape[1]) * np.sqrt(2 / shape[0])


def xavier_weights(shape):
    upper = 1.0 / np.sqrt(shape[0])
    lower = -upper
    return lower + np.random.rand(shape[0], shape[1]) * (upper - lower)


def uniform_weights(shape):
    upper = 1
    lower = -upper

    if len(shape) == 3:
        return lower + np.random.rand(shape[0], shape[1], shape[2]) * (upper - lower)
    elif len(shape) == 4:
        return lower + np.random.rand(shape[0], shape[1], shape[2], shape[3]) * (
            upper - lower
        )
    return lower + np.random.rand(shape[0], shape[1]) * (upper - lower)


# ----------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ LOSSES --------------------------------------- #


def loss(loss):
    if loss == "bce":
        return binary_cross_entropy_loss, binary_cross_entropy_loss_d
    elif loss == "ce":
        return cross_entropy_loss, cross_entropy_loss_d
    else:  # mse
        return quadratic_loss, quadratic_loss_d


# MSE
def quadratic_loss(a, y):
    return 0.5 * (y - a) ** 2


def quadratic_loss_d(a, y):
    return a - y


# BCE
def binary_cross_entropy_loss(a, y):
    return -np.sum(y * np.log2(a + 1e-8) + (1 - y) * np.log2(1 - a + 1e-8), axis=0)


# Gives same derivative as MSE and also cancels out last layer's activation derivative as well, speeds up learning but may lead to exploding gradients
# It is binary because it deals each output neuron seperately, which is suitable when outputs are not mutually exclusive. In MNIST problem, the right way to treat outputs is together because they are mutually exclusive hence softmax (general sigmoid) activation should be used in final layer
def binary_cross_entropy_loss_d(a, y):
    grad = -(y / (a + 1e-2)) + ((1 - y) / (1 - a + 1e-2))
    # print("\n", y, "\n", [np.round(i, 2) for i in a], "\n", grad, "\n")
    # print("a-y: ", a-y)
    return grad


# CE Loss
def cross_entropy_loss(a, y):
    return -np.sum(y * np.log2(a + 1e-8), axis=0)


def cross_entropy_loss_d(a, y):
    return -(y / (a + 1e-2))


# ----------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ ACTIVATION METHODS --------------------------------------- #


def sigmoid(x):
    return 1 / (1 + np.exp(-np.float64(x)))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (2 / (1 + np.exp(-2 * np.float64(x)))) - 1


def tanh_prime(x):
    return 1 - tanh(x) ** 2


def relu(x):
    return x * (x >= 0).astype(int)


def relu_prime(x):
    return (x >= 0).astype(int)


def softmax(x):
    exps = np.exp(x)
    if len(x.shape) == 1:
        return np.array([i / np.sum(exps) for i in exps])
    elif len(x.shape) == 2:
        xs = []
        expss = np.exp(x)
        for exps in expss:
            xs.append([i / np.sum(exps) for i in exps])
        return np.array(xs)

    raise Exception


def softmax_prime(x):
    if len(x.shape) == 1:
        return np.array(softmax_prime_helper(x))
    elif len(x.shape) == 2:
        arr = []
        for i in x:
            arr.append(softmax_prime_helper(i))
        return np.array(arr)

    raise Exception


def softmax_prime_helper(x):
    arr = []
    sfts = softmax(x)
    for j in range(len(x)):
        grad = []
        for i in range(len(x)):
            if i == j:
                grad.append(sfts[i] * (1 - sfts[i]))
            else:
                grad.append(-sfts[i] * sfts[j])
        arr.append(grad)
    return arr


# ----------------------------------------------------------------------------------------------------------- #
