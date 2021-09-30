from gene import *
import net
import game
import random
import numpy as np

def choose(n):
    r = random.randrange(n*(n+1)/2)
    for i in range(1, n+1):
        r -= i
        if r < 0:
            return n-i


pop_sz = 100
num_generations = 1000000000
num_survive = 10
mutation_prob = 0.05
mutation_sigma = 0.1
burst_mutation_sigma = 0.1
burst_generations = 10

sim_time = 100
gene_sz = game.gene_sz

print('Gene Size = %d' % gene_sz)

genes = [None]*pop_sz
for i in range(pop_sz):
    genes[i] = Gene(gene_sz)
    genes[i].mutate(1, mutation_sigma)

best_avg = 0
burst_timer = 0
for i in range(num_generations):
    print('Starting Generation %d' % i)
    game.evaluate(genes, sim_time)
    genes.sort(key=lambda x: x.fitness, reverse=True)
    
    avg_fitness = 0
    for i in range(pop_sz):
        avg_fitness += genes[i].fitness
    avg_fitness /= pop_sz

    avg_top = 0
    for i in range(num_survive):
        avg_top += genes[i].fitness
    avg_top /= num_survive
    print('Average fitness = %.1f, best_avg = %.1f' % (avg_fitness, best_avg))

    print('Best: fitness = %.1f, max = %.2f, L2 = %.2f' % (genes[0].fitness, genes[0].maxweight(), genes[0].norm()))

    new_genes = [None]*pop_sz
    for i in range(num_survive):
        new_genes[i] = genes[i].copy()
    for i in range(num_survive, pop_sz):
        idx1 = idx2 = 0
        while idx1 == idx2:
            idx1 = choose(pop_sz)
            idx2 = choose(pop_sz)
        new_genes[i] = crossover(genes[idx1], genes[idx2])

    for i in range(pop_sz):
        new_genes[i].mutate(mutation_prob, mutation_sigma)

    if avg_fitness > best_avg:
        best_avg = avg_fitness
        burst_timer = 0
    else:
        burst_timer += 1
        if burst_timer > burst_generations:
            print("burst mutation")
            for i in range(pop_sz):
                genes[i].mutate(1, burst_mutation_sigma)
            burst_timer = 0
    genes = new_genes


