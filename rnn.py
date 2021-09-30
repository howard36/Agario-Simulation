import numpy as np
import math
import gene

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

class RNN:
    def __init__(self, input_sz, state_sz, gene):
        self.input_sz = input_sz + 1 + state_sz
        self.state_sz = state_sz
        self.state = np.zeros(state_sz)
        self.gene = gene

        self.w = np.reshape(gene.nums, (state_sz, self.input_sz))

    def feedforward(self, inp, show=False):
        inp.append(1)

        x = np.asarray(inp)
        x = np.concatenate((x, self.state))
        assert(x.shape[0] == self.input_sz)
        x = np.matmul(self.w, x)
        x = tanh(x)
        self.state = x
        return x[:2]
