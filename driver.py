import sys
sys.path.append("./src/")
from src.book.mnist_loader import load_data_wrapper
from neuralnet import NeuralNetwork
from train import train, test_evaluation

train_data, valid_data, test_data = load_data_wrapper()

# loss_fn = mse, bce, ce (ce and softmax should be used together)
# weight_init = he, xavier, uniform
network = NeuralNetwork(network_shape=(784, 128, 10), 
                  lr=0.1, 
                  loss_fn="ce", 
                  weight_init="he")


# final_layer_act = sigmoid, relu, tanh, softmax
# intermediate_act = sigmoid, relu, tanh
# activations = (intermediate_acts, final_acts) | activations = (one_act_for_all, )
network.activation_setup(activations=("relu", "softmax"))
# regularization = l1, l2 | reglambda = (0.0-1.0) 
# droput = (0.0-1.0) (percent of nodes to dropout in each layer)
network.regularization_setup(regL="None", reglambda=1e-5, dropout=0.0) 
# optimizer_name=name, args=(args for that optimizer)
# sgd_momentum args = (momentum = 0.0 - 1.0)
# adam args = (*beta1* (for first moment [mean]) = 0.0-1.0,  *beta2* (for second moment [variance]) = 0.0-1.0)
network.optimizer_setup(optimizer_name="sgd_momentum", args=(0.2,))

train(network, 
      train_data, 
      epochs=2, 
      batch_size=4)

print("Train Accuracy: ", test_evaluation(network, train_data), "%")
print("Validation Accuracy: ", test_evaluation(network, valid_data), "%")
print("Test Accuracy: ", test_evaluation(network, test_data), "%")