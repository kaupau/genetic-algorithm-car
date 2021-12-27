import numpy as np

def softmax(z):
    s = np.exp(z.T) / np.sum(np.exp(z.T), axis=0).reshape(-1, 1)
    return s

layer_sizes = [4, 9, 15, 15, 4]
num_weights = np.prod(layer_sizes)

W1_size = (9, 4)
W2_size = (15, 9)
W3_size = (15, 15)
W4_size = (4, 15)

def reshape_weights(weights):
    W1_size = (9, 4)
    W2_size = (15, 9)
    W3_size = (15, 15)
    W4_size = (4, 15)

    W1 = weights[:np.prod(list(W1_size))]
    W1 = np.array(W1)
    W2 = weights[np.prod(list(W1_size)):np.prod(list(W2_size))+np.prod(list(W1_size))]
    W2 = np.array(W2)
    W3 = weights[np.prod(list(W2_size))+np.prod(list(W1_size)):np.prod(list(W3_size))+np.prod(list(W2_size))+np.prod(list(W1_size))]
    W3 = np.array(W3)
    W4 = weights[np.prod(list(W3_size)) + np.prod(list(W2_size)) + np.prod(list(W1_size)):np.prod(list(W4_size)) + np.prod(list(W3_size)) + np.prod(list(W2_size)) + np.prod(list(W1_size))]
    W4 = np.array(W4)

    return W1.reshape(W1_size), W2.reshape(W2_size), W3.reshape(W3_size), W4.reshape(W4_size)

def forward_propagate(weights, inputs):
    W1, W2, W3, W4 = reshape_weights(weights)

    Z1 = np.matmul(W1, np.array(inputs).T)
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2, A1)
    A2 = np.tanh(Z2)
    Z3 = np.matmul(W3, A2)
    A3 = np.tanh(Z3)
    Z4 = np.matmul(W4, A3)
    A4 = softmax(Z4)

    return A4.tolist()[0]

if __name__ == "__main__":
    weights = np.random.uniform(low=-1, high=1, size=(np.prod(num_weights),))
    input_combos = [
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ]
    for i in input_combos:
        network = forward_propagate(weights, i)
        print(network)