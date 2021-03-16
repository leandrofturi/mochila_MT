from comons import *


def roulette_construction(states):
    evaluation = [state['sum_dist'] for state in states]
    roulette = []
    ratio_sum = 0.0
    
    if any([e == 0.0 for e in evaluation]):
        # if any of the groups are unitary, the ratio is the same
        ratio = 1.0/len(states)
        for state in states:
            ratio_sum += ratio
            roulette.append({'ratio': ratio_sum, 'state': state})
    else:
        total_sum = sum(evaluation)
        for state in states:
            ratio_sum += state['sum_dist']/total_sum
            roulette.append({'ratio': ratio_sum, 'state': state})
    
    return roulette


def roulette_chooser(roulette):
    r = np.random.uniform(0, 1, 1)
    for state in roulette:
        if r <= state['ratio']:
            return state['state']


def roulette_run(rounds, states):
    if rounds >= len(states):
        return states

    roulette = roulette_construction(states)

    selecteds = []
    selected = roulette_chooser(roulette)
    selecteds.append(selected)
    for _ in range(rounds-1):
        while selected in selecteds:
            selected = roulette_chooser(roulette)
        selecteds.append(selected)

    return selecteds


def faraways_itens(state, n):
    if n >= len(state['itens']):
        return state['itens']

    pos_faraways = np.argpartition(state['dist'], range(n))[-n:]
    faraways = [state['itens'][p] for p in pos_faraways]
    return faraways


def closest_cluster(item, states):
    # closet to centroid
    other_states = [s for s in states if not item in s['itens']]
    dist_mu = [np.linalg.norm(item['coord'] - np.array(s['mu'])) for s in other_states]
    return other_states[np.argmin(dist_mu)]


def exchange_neighbor(states, size_neighborhood):
    # choose two bad clusters by roulette, take an faraway item from one and put it in the other
    if len(states) <= 1:
        return states

    in_state = roulette_run(1, states)[0]
    if len(in_state['itens']) <= 1:
        return states # try next

    faraways = faraways_itens(in_state, size_neighborhood)
    faraway = random_state(faraways) # randomly
    out_state = closest_cluster(faraway, states)
    in_cluster = [s for s in in_state['itens'] if s != faraway] # remove
    out_cluster = out_state['itens'] + [faraway] # put

    # evaluate new clusters
    new_clusters = [s['itens'] for s in states if s != in_state and s != out_state]
    new_clusters.append(in_cluster)
    new_clusters.append(out_cluster)
    return evaluate_clusters(new_clusters)


def change_probability(value, best_value, t):
    p = math.exp(-(value - best_value)/t)
    r = np.random.uniform(0,1)
    if r < p:
        return True
    else:
        return False


def simulated_annealing(itens, k, t, alfa, size_neighborhood, iter_max, max_time):
    initial_solution = generate_random_clusters(itens, k)
    solution = evaluate_clusters(initial_solution)
    value = objective_function(solution)
    best_solution = solution
    best_value = value
    start = time.process_time()
    
    while True:
        for _ in range(iter_max):
            new_solution = exchange_neighbor(solution, size_neighborhood)
            new_value = objective_function(new_solution)

            if new_value < value:
                solution = new_solution
                value = new_value
                if value < best_value:
                    best_solution = solution
                    best_value = value
            else:
                if change_probability(new_value, value, t):
                    solution = new_solution
                    value = new_value
        t = t*alfa
        end = time.process_time()

        if t < 1:
            #print("--- Break by temperature!")
            break
        if end-start > max_time:
            #print("--- Break by time!")
            break

    best_solution = [s['itens'] for s in best_solution]
    #if best_solution == initial_solution:
        #print("--- No changes in initial solution!")
    
    #print("")
    #print("Objective function = %.4f" % best_value)
    #print("Elapsed time = %.4f" % (end-start))
    #[print(', '.join(sorted([str(t['id']) for t in s]))) for s in best_solution]
    return best_value, end-start