import numpy as np
import math
import gene

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

class Net:
    def __init__(self, input_sz, hidden_sz, output_sz, gene):
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.gene = gene

        l1 = (input_sz+1)*hidden_sz
        l2 = (hidden_sz+1)*output_sz

        assert(gene.size == 2*(l1 + l2))

        abs1 = gene.nums[:l1]/math.sqrt(input_sz + 1)
        sgn1 = gene.nums[l1:2*l1]
        sgn1 = np.ones(sgn1.shape) - 2*sgn1
        self.w1 = np.reshape(abs1*sgn1, (hidden_sz, input_sz + 1))

        abs2 = gene.nums[2*l1:2*l1+l2]/math.sqrt(hidden_sz + 1)
        sgn2 = gene.nums[2*l1+l2:]
        sgn2 = np.ones(sgn2.shape) - 2*sgn2
        self.w2 = np.reshape(abs2*sgn2, (output_sz, hidden_sz + 1))

    def feedforward(self, inp, show=False):
        assert(len(inp) == self.input_sz)
        inp.append(1)

        x = np.asarray(inp)
        if show:
            print('input = ', end='')
            print(x)
        x = np.matmul(self.w1, x)
        x = tanh(x)
        x = np.append(x, [1])
        if show:
            print('hidden = ', end='')
            print(x)
        x = np.matmul(self.w2, x)
        x = tanh(x)
        return x

