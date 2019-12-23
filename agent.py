import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Agent:
    def __init__(self, input_size, hidden_size, params):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2
        self.param_size = (self.input_size + 1)*self.hidden_size \
                        + (self.hidden_size+1)*self.output_size

        assert(params.shape == (self.param_size,))
        self.params = params
        self.w1 = self.params[: (self.input_size + 1)*self.hidden_size]
        self.w1 = np.reshape(self.w1, (self.hidden_size, self.input_size + 1))
        self.w2 = self.params[(self.input_size + 1)*self.hidden_size :]
        self.w2 = np.reshape(self.w2, (self.output_size, self.hidden_size + 1))

    def move(self, inp, show=False):
        inp.append(1)
        if not len(inp) == self.input_size+1:
            print('len(inp) = %d' % len(inp))
            print('self.input_size + 1 = %d' % (self.input_size+1))
        assert(len(inp) == self.input_size + 1)

        x = np.asarray(inp)
        if show:
            print('input = ', end='')
            print(x)
        x = np.matmul(self.w1, x)
        x = sigmoid(x)
        x = np.append(x, [1])
        if show:
            print('hidden = ', end='')
            print(x)
        x = np.matmul(self.w2, x)
        return x

