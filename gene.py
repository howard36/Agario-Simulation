import numpy as np
import random

class Gene:
    def __init__(self, size, nums=None, fitness=0):
        self.size = size;
        if nums is None:
            self.nums = np.zeros(size)
            '''
            for i in range(self.size):
                if random.random() < 0.5:
                    self.nums[i] = 1
            '''
        else:
            self.nums = nums
        self.fitness = fitness

    def mutate(self, prob, sigma):
        for i in range(self.size):
            if random.random() < prob:
                #print(self.nums)
                #print(type(self.nums))
                self.nums[i] += np.random.normal(0, sigma)
                #self.nums[i] = 1 - self.nums[i]
        self.fitness = 0

    def copy(self):
        return Gene(self.size, self.nums.copy(), self.fitness)

    def maxweight(self):
        return self.nums.max()

    def norm(self):
        return np.sum(np.multiply(self.nums, self.nums))

def crossover(a, b):
    assert(a.size == b.size)
    idx = random.randrange(1, a.size) # [1, a.size)
    return Gene(a.size, np.concatenate((a.nums[:idx], b.nums[idx:])))

