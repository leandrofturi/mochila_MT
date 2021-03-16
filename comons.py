import time
import math
import numpy as np


def generate_random_clusters(itens, k):
    # k random initial clusters
    perm = np.random.permutation([i['id'] for i in itens])
    pos_crop = np.append(np.random.choice(range(1, len(itens)-1), k-1, replace=False), len(itens))
    pos_atual = 0
    initial_clusters = []

    for pos in sorted(pos_crop):
        initial_clusters.append([itens[i-1] for i in perm[pos_atual:pos]])
        pos_atual = pos
    return initial_clusters


def evaluate_cluster(cluster):
    # euclidean norm
    itens = [s['coord'] for s in cluster]
    mu = np.average(itens, axis=0) # centroid
    norm2 = [np.linalg.norm(i-mu) for i in itens]
    return {'sum_dist': sum(norm2), 'dist': norm2, 'mu': mu.tolist(), 'itens': cluster}


def evaluate_clusters(clusters):
    states = [evaluate_cluster(cluster) for cluster in clusters]
    return states


def objective_function(states):
    # SSE metric
    sse = sum([state['sum_dist'] for state in states])
    return sse


def random_state(states):
    n = len(states)-1
    if n <= 1:
        return states[0]

    index = np.random.randint(0, n)
    return states[index]
