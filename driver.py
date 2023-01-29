import numpy as np
from tqdm import tqdm
import sys
sys.path.append("./src/")
from src.book.mnist_loader import load_data_wrapper
from neuralnet import NeuralNetwork
from train import train, test_evaluation

train_data, valid_data, test_data = load_data_wrapper()
n = NeuralNetwork((784, 128, 10), lr=0.1)
train(n, train_data, epochs=4, batch_size=4)
print("Accuracy: ", test_evaluation(n, train_data), "%")
print("Accuracy: ", test_evaluation(n, test_data), "%")