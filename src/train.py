import numpy as np
from tqdm import tqdm
from neuralnet import NeuralNetwork

def test_evaluation(net, data):
    matches = []
    d_x, d_y = preprocess_data(data)
    for x, y in zip(d_x, d_y):
        net.eval()
        pred, _ = net.forward(x, y)
        matches.append(np.argmax(pred) == np.argmax(y))
    matches = np.array(matches).astype(int)
    return np.sum(matches) / len(data)

def preprocess_data(data):
    data_x = []
    data_y = []
    for i in data:
        data_x.append(i[0])
        data_y.append(i[1])
    data_x = np.squeeze(data_x)
    data_y = np.squeeze(data_y)
    if len(data_x.shape) == 1:
        data_x = np.expand_dims(data_x, axis=0)
    if len(data_y.shape) == 1:
        data_y = np.expand_dims(data_y, axis=0)
    return data_x, data_y

def train(net : NeuralNetwork, train_data, epochs=1, batch_size=1):
    epoch_iterable = tqdm(range(epochs))
    for i in epoch_iterable:
        epoch_iterable.set_description("Epoch: " + str(i))
        for j in range(round(len(train_data) / batch_size)):
            data = train_data[j * batch_size : (j + 1) * batch_size]
            x, y = preprocess_data(data)
            pred, loss = net.forward(x, y)
            net.backward()
            epoch_iterable.set_postfix(
                {
                    "Batch: ": f'{j:06d}',
                    "Loss: ": f'{np.mean(loss):06f}',
                    "P/A: ": str(np.argmax(pred, axis=-1)) + "/" + str(np.argmax(y, axis=-1)),
                }
            )
        net.decay_betas()