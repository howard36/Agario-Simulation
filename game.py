import random
import copy
import agent
import math
import numpy as np
from graphics import *

random.seed()
num_agents = 100
num_food = 100
num_close_agents = 5
num_close_food = 10
starting_mass = 10
turns = 1000
speed = 1/10

sigma = 10
input_size = 4 + 3*num_close_agents + 2*num_close_food
hidden_size = 50
param_size = (input_size+1)*hidden_size + (hidden_size+1)*2

def dist(v):
    return math.sqrt(v[0]**2 + v[1]**2)

def dist2(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def printp(p):
    print('(%d, %d)' % (int(100*p[0]), int(100*p[1])), end='')

def closest(pos_list, p, n):
    cpy = copy.deepcopy(pos_list)
    for i in range(len(cpy)):
        cpy[i][0] -= p[0]
        cpy[i][1] -= p[1]
    tmp = [(dist(cpy[i]), i) for i in range(len(cpy))]
    tmp = sorted(tmp)
    return [pair[1] for pair in tmp[:n]]

def printpl(pos_list):
    print('[', end='')
    for i in range(len(pos_list)):
        printp(pos_list[i])
        if i < len(pos_list)-1:
            print(', ', end='')
    print(']')

def get_radius(mass):
    return min(math.sqrt(mass)/200, 0.1)

def draw(win):
    for i in range(num_food):
        if food_circ[i]:
            food_circ[i].undraw()
        food_circ[i] = Circle(Point(food[i][0], food[i][1]), 1/200)
        food_circ[i].setFill('blue')
        food_circ[i].draw(win)
    tmp = [(mass[i], i) for i in range(num_agents)]
    tmp = sorted(tmp)
    tmp = [pair[1] for pair in tmp]
    for i in tmp:
        if pos_circ[i]:
            pos_circ[i].undraw()
        pos_circ[i] = Circle(Point(pos[i][0], pos[i][1]), radius[i])
        pos_circ[i].setFill('green')
        pos_circ[i].draw(win)
    win.flush()

def new_food(i):
    food[i] = [random.random(), random.random()]

def new_agent(i):
    pos[i] = [random.random(), random.random()]
    params = sigma * np.asarray([np.random.normal() for _ in range(param_size)]) / param_size
    '''
    cov = sigma * np.eye(param_size) / param_size
    print(cov.shape)
    params = np.random.multivariate_normal(np.zeros(param_size), cov)
    '''
    agents[i] = agent.Agent(input_size, hidden_size, params)
    mass[i] = starting_mass
    radius[i] = get_radius(mass[i])

def eat(i, j):
    #print('%d ate %d' % (i, j))
    mass[i] += mass[j]
    radius[i] = get_radius(mass[i])
    new_agent(j)
    check_eat(i)

def check_eat(i):
    for j in range(num_food):
        if dist2(food[j], pos[i]) < radius[i]:
            mass[i] += 1
            radius[i] = get_radius(mass[i])
            new_food(j)
    for j in range(num_agents):
        if j == i:
            continue
        if mass[i] > mass[j] and dist2(pos[i], pos[j]) < radius[i]:
            eat(i, j)
        elif mass[j] > mass[i] and dist2(pos[i], pos[j]) < radius[j]:
            eat(j, i)

agents = [None]*num_agents
pos = [None]*num_agents
food = [None]*num_food
mass = [starting_mass]*num_agents
radius = [get_radius(starting_mass)]*num_agents
pos_circ = [None]*num_agents
food_circ = [None]*num_food

for i in range(num_agents):
    new_agent(i)
for i in range(num_food):
    new_food(i)

win = GraphWin(width=600, height=600, autoflush=False)
win.setCoords(0, 0, 1, 1)
win.setBackground('#eee')
draw(win)
print(mass, end=' ')
printpl(pos)
win.getKey()

for _ in range(turns):
    for i in range(num_agents):
        agent_idx = closest(pos, pos[i], num_close_agents+1)
        if agent_idx[0] != i:
            print('Error: agent_idx is wrong')
        agent_idx.pop(0)
        agent_input = []
        for j in agent_idx:
            agent_input.append([pos[j][0]-pos[i][0], pos[j][1]-pos[i][1], mass[i]]) 
        food_idx = closest(food, pos[i], num_close_food)
        food_input = [food[j] for j in food_idx]

        move = agents[i].move(pos[i], mass[i], radius[i], agent_input, food_input) * speed

        pos[i][0] = min(max(pos[i][0] + move[0], 0), 1)
        pos[i][1] = min(max(pos[i][1] + move[1], 0), 1)

        check_eat(i)
    draw(win)

