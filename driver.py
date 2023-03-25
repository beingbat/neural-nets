import numpy as np
from tqdm import tqdm
import sys
sys.path.append("./src/")
from src.book.mnist_loader import load_data_wrapper
from neuralnet import NeuralNetwork
from train import train, test_evaluation

train_data, valid_data, test_data = load_data_wrapper()
# loss_fn = MSE, BCE, CE
# final_layer_act = sigmoid, relu, tanh, softmax
# intermediate_act = sigmoid, relu, tanh
# weight_init = he, xavier, uniform
# regularization = L1, L2
# reg_lambda = lambda of L regularization
# dropout = dropout threshold
n = NeuralNetwork((784, 128, 10), lr=0.1, loss_fn="CE", intermediate_act="relu", final_layer_act="softmax", weight_init="he", regularization=None, regularization_lambda=1e-5, dropout=0)
train(n, train_data, epochs=2, batch_size=4)
print("Train Accuracy: ", test_evaluation(n, train_data), "%")
print("Validation Accuracy: ", test_evaluation(n, valid_data), "%")
print("Test Accuracy: ", test_evaluation(n, test_data), "%")