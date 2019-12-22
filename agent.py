import numpy

class Agent:
    def __init__(self, input_agents, input_food):
        self.input_size = 4 + 3*input_agents + 2*input_food
        self.hidden_size = 50
        self.output_size = 2
        self.param_size = (self.input_size + 1)*self.hidden_size 
                        + (self.hidden_size + 1)*self.output_size
        self.params = 

    def move(self, pos, mass, radius, close_agents, close_food):
        inp = [1] # bias
        inp += pos
        inp.append(mass)
        inp.append(radius)
        for i in range(len(close_agents)):
            inp += close_agents[i]
        for i in range(len(close_food)):
            inp += close_food[i]
        assert(len(inp) == self.input_size)

        return [0.01,0.01]

