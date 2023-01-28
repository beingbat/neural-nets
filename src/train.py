import numpy as np
from tqdm import tqdm
from neuralnet import NeuralNetwork


def test_evaluation(net, data):
    matches = []
    for x, y in data:
        x = np.squeeze(x)
        y = np.squeeze(y)
        pred, _ = net.forward(x, y)
        matches.append(np.argmax(pred) == np.argmax(y))
    matches = np.array(matches).astype(int)
    return np.sum(matches) / len(data)


def train(net, train_data, epochs=1, batch_size=1):
    epoch_iterable = tqdm(range(epochs))
    for i in epoch_iterable:
        epoch_iterable.set_description("Epoch: " + str(i))
        for j in range(round(len(train_data) / batch_size)):
            data = train_data[j * batch_size : (j + 1) * batch_size]
            for k in data:
                x, y = k
                x = np.squeeze(x)
                y = np.squeeze(y)
                pred, loss = net.forward(x, y)
            net.backward()
            epoch_iterable.set_postfix(
                {
                    "Batch: ": f'{j:06d}',
                    "Loss: ": f'{np.mean(loss):06f}',
                    "P/A: ": str(np.argmax(pred)) + "/" + str(np.argmax(y)),
                }
            )
    
