import numpy as np
from tqdm import tqdm
import sys
sys.path.append("./src/")
from src.book.mnist_loader import load_data_wrapper
from neuralnet import NeuralNetwork
from train import train, test_evaluation

train_data, valid_data, test_data = load_data_wrapper()

# loss_fn = mse, bce, ce (ce and softmax should be used together)
# final_layer_act = sigmoid, relu, tanh, softmax
# intermediate_act = sigmoid, relu, tanh
# weight_init = he, xavier, uniform
# regularization = l1, l2
# reg_lambda = lambda for L1/L2 regularization
# dropout = 0.0 - 1.0 (percent of nodes to dropout in each layer)
# momentum = 0.0 - 1.0 (sgd momentum)
# optimizer = sgd, sgd_momentum
network = NeuralNetwork(network_shape=(784, 128, 10), 
                  lr=0.1, 
                  loss_fn="ce", 
                  intermediate_act="relu", 
                  final_layer_act="softmax", 
                  weight_init="he", 
                  regularization=None, 
                  regularization_lambda=1e-5, 
                  dropout=0,
                  momentum=0.2, 
                  optimizer="sgd_momentum")

train(network, 
      train_data, 
      epochs=2, 
      batch_size=4)

print("Train Accuracy: ", test_evaluation(network, train_data), "%")
print("Validation Accuracy: ", test_evaluation(network, valid_data), "%")
print("Test Accuracy: ", test_evaluation(network, test_data), "%")