import random
import copy
import agent
import math
import numpy as np
from graphics import *

random.seed()
num_agents = 50
num_food = 200
num_close_agents = 3
num_close_food = 0
num_survive = 10
starting_mass = 5
mass_decay = 0.99
max_speed = 1/100

gen_time = 200
num_generations = 1000000


sigma = 20
input_size = 3 + 3*num_close_agents + 2*num_close_food
hidden_size = 10
param_size = (input_size+1)*hidden_size + (hidden_size+1)*2

print('Parameter Size: %d' % param_size)

def dist(v):
    return math.sqrt(v[0]**2 + v[1]**2)

def L2(v):
    return np.sum(np.dot(v, v))

def minus(a, b):
    return [a[0]-b[0], a[1]-b[1]]

def dist2(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def printp(p):
    print('(%d, %d)' % (int(100*p[0]), int(100*p[1])), end='')

def printpl(pos_list):
    print('[', end='')
    for i in range(len(pos_list)):
        printp(pos_list[i])
        if i < len(pos_list)-1:
            print(', ', end='')
    print(']')




def closest_agents(i):
    tmp = []
    for j in range(num_agents):
        if j == i or not alive[j]:
            continue
        tmp.append((dist2(pos[j], pos[i]), j))
    tmp = sorted(tmp)
    return [pair[1] for pair in tmp[:num_close_agents]]

def closest_food(i):
    tmp = [(dist2(food[j], pos[i]), j) for j in range(num_food)]
    tmp = sorted(tmp)
    return [pair[1] for pair in tmp[:num_close_food]]

def set_mass(i, m):
    mass[i] = m
    radius[i] = min(math.sqrt(mass[i])/100, 0.1)

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
        if alive[i]:
            pos_circ[i] = Circle(Point(pos[i][0], pos[i][1]), radius[i])
            pos_circ[i].setFill('green')
            pos_circ[i].draw(win)
    win.flush()

def new_food(i):
    food[i] = [random.random(), random.random()]

def new_agent(i):
    #if i == 0:
    #    print("New 0")
    
    #print(best)
    params = best[random.randint(0, len(best)-1)].copy()
    step = np.asarray([np.random.normal() for _ in range(param_size)]) / param_size
    params += sigma * step
    agents[i] = agent.Agent(input_size, hidden_size, params)

    pos[i] = [random.random(), random.random()]
    set_mass(i, starting_mass)
    alive[i] = True
    #print(L2(step), L2(params))

def eat(i, j, t):
    #print('%d ate %d at time %d' % (i, j, t))
    assert(alive[i])
    assert(alive[j])
    set_mass(i, mass[i] + mass[j])
    die(j, t) 
    check_eat(i, t)

def die(i, t):
    assert(alive[i])
    alive[i] = False
    #print(fitness)
    fitness.append((t + mass[i], i))

def check_eaten(i, t):
    for j in range(num_agents):
        if j == i or not alive[j]:
            continue
        if mass[j] > mass[i] and dist2(pos[i], pos[j]) < radius[j]:
            eat(j, i, t)
            return True
    return False

def check_eat(i, t):
    assert(alive[i])
    for j in range(num_food):
        if dist2(food[j], pos[i]) < radius[i]:
            set_mass(i, mass[i]+1)
            new_food(j)
    for j in range(num_agents):
        if j == i or not alive[j]:
            continue
        if mass[i] > mass[j] and dist2(pos[i], pos[j]) < radius[i]:
            eat(i, j, t)

agents = [None]*num_agents
pos = [None]*num_agents
mass = [None]*num_agents
radius = [None]*num_agents
alive = [True]*num_agents

food = [None]*num_food
pos_circ = [None]*num_agents
food_circ = [None]*num_food

best = []
fitness = []

win = GraphWin(width=600, height=600, autoflush=False)
win.setCoords(0, 0, 1, 1)
win.setBackground('#eee')

def run_generation():
    global fitness, best

    for i in range(num_agents):
        new_agent(i)
    for i in range(num_food):
        new_food(i)
    fitness = []
    maxlog = -100
    minlog = 100
    maxlogd = -100
    minlogd = 100

    for t in range(gen_time):
        for i in range(num_agents):
            if not alive[i]:
                continue

            inp = [2*pos[i][0]-1, 2*pos[i][1]-1, np.log10(mass[i]/starting_mass)]
            maxlog = max(maxlog, np.log10(mass[i]/starting_mass))
            minlog = min(minlog, np.log10(mass[i]/starting_mass))
            agent_idx = closest_agents(i)
            for j in agent_idx:
                inp += minus(pos[j], pos[i])
                inp.append(np.log10(mass[j]/mass[i]))
                maxlogd = max(maxlogd, np.log10(mass[j]/mass[i]))
                minlogd = min(minlogd, np.log10(mass[j]/mass[i]))
            if len(agent_idx) < num_close_agents:
                inp += [0]*(3*(num_close_agents - len(agent_idx)))

            food_idx = closest_food(i)
            for j in food_idx:
                inp += minus(food[j], pos[i])

            move = agents[i].move(inp)

            #norm = dist(move)
            #if i == 0:
            #    print("Agent 0's move: (%.3f, %.3f)" % (move[0], move[1]))
            #print(norm)
            #if norm > 1:
            #    move /= norm
            #    set_mass(i, mass[i]/norm)
            move *= max_speed

            pos[i][0] = min(max(pos[i][0] + move[0], 0), 1)
            pos[i][1] = min(max(pos[i][1] + move[1], 0), 1)

            if check_eaten(i, t):
                continue

            if pos[i][0] == 0 or pos[i][0] == 1 or pos[i][1] == 0 or pos[i][1] == 1:
                set_mass(i, mass[i] / 2)
            if mass[i] < 1:
                die(i, t)
            else: # still alive
                check_eat(i, t)

        draw(win)

    for i in range(num_agents):
        if alive[i]:
            die(i, 2*gen_time)
    
    assert(len(fitness) == num_agents)
    fitness = sorted(fitness)
    fitness.reverse()

    best = []
    for i in range(num_survive):
        best.append(agents[fitness[i][1]].params.copy())

    avg_fitness = 0
    for i in range(num_agents):
        avg_fitness += fitness[i][0]
    avg_fitness /= num_agents

    avg_top = 0
    for i in range(num_survive):
        avg_top += fitness[i][0]
    avg_top /= num_survive
    print('Average fitness = %.3f, top %d average = %.3f' % (avg_fitness, num_survive, avg_top))
    print('maxl = %.2f, minl = %.2f, maxld = %.2f, minld = %.2f' % (maxlog, minlog, maxlogd, minlogd))
    

best.append(np.zeros(param_size))
for i in range(num_generations):
    print('Starting Generation %d' % i)
    run_generation()
