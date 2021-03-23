from comons import *


def initial_population(n, itens, k):
    pop = [evaluate_clusters(generate_random_clusters(itens, k)) for _ in range(n)]
    return pop


def selection(population, n, n_tournament):
    # by tournament
    N = len(population)
    if n >= N:
        return population
    if n_tournament > N:
        n_tournament = N
    
    selecteds = np.random.choice(range(0, len(population)), size=n_tournament, replace=False)
    selecteds.sort()
    values = [objective_function(population[i]) for i in selecteds]
    pos_ordered = np.argpartition(values, range(n))[-n:]
    return [population[p] for p in pos_ordered]


def crossover(dad, mom):
    # son product of an egg fertilized by a sperm
    sperm = random_state(dad)['itens']
    # take all groups that not contain the sperm, and remove it too
    egg = [e for e in [[n for n in m['itens'] if n not in sperm] for m in mom] if e]
    son = egg + [sperm]
    
    k = len(dad)
    while len(son) > k: # concatenate smaller groups
        smallers = np.argpartition([len(s) for s in son], range(2))[:2]
        son[smallers[0]] += son[smallers[1]]
        del son[smallers[1]]
        
    while len(son) < k: # separate larger groups
        larger = np.argmax([len(s) for s in son])
        half = len(son[larger])//2
        son += [son[larger][:half]] + [son[larger][half:]]
        del son[larger]
    
    return evaluate_clusters(son)


def new_individual(itens, k):
    clusters = generate_random_clusters(itens, k)
    return evaluate_clusters(clusters)


def mutation(itens, k):
    # random
    return new_individual(itens, k)


def convergent(population):
    clusters_0 = [sorted([i['id'] for i in p['itens']]) for p in population[0]]
    clusters = [[sorted([i['id'] for i in clusters['itens']]) for clusters in states] for states in population]
    return all(c == clusters_0 for c in clusters)


def evaluate_population(population):
    return sum([objective_function(p) for p in population], [])


def offspring(population, n):
    best_index = np.argpartition([sum([q['sum_dist'] for q in p]) for p in population], range(n))[:n]
    return [population[i] for i in best_index]


def genetic(itens, k, pop_size, iter_max, cross_ratio, mut_ratio, max_time):
    n_tournament = 3
    half_pop = pop_size//2
    pop = initial_population(pop_size, itens, k)
    best_solution = pop
    best_value = evaluate_population(pop)
    iter = 0    
    end = 0
    start = time.process_time()
    
    while True:
        new_pop = pop.copy()
        for _ in range(half_pop): # everyone can cross
            if np.random.uniform(0, 1, 1) <= cross_ratio:
                parents = selection(pop, 2, n_tournament)
                new_pop.append(crossover(parents[0], parents[1]))
            if np.random.uniform(0, 1, 1) <= mut_ratio:
                new_pop.append(mutation(itens, k))

        pop = offspring(new_pop, pop_size)
        val_pop = evaluate_population(pop)
    
        if (val_pop < best_value):
            best_solution = pop
            best_value = val_pop
        
        iter += 1
        end = time.process_time()
        if iter >= iter_max:
            #print("--- Break by iteration!")
            break
        if end-start > max_time:
            #print("--- Break by time!")
            break
        if convergent(pop):
            #print("--- Break by convergence!")
            break

    best_individual = offspring(pop, 1)[0]
    #print("")
    #print("Objective function = %.4f" % objective_function(best_individual))
    #print("Elapsed time = %.4f" % (end-start))
    #[print(', '.join(sorted([str(t['id']) for t in s['itens']]))) for s in best_individual]
    return [s['itens'] for s in best_individual]
    #return objective_function(best_individual), end-start

