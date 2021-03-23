from comons import *


def available_closest_itens(cluster, available_itens, n):
    # n itens more closet to centroid of cluster
    n = min(n, len(available_itens))
    evaluations = [evaluate_cluster(cluster + [i]) for i in available_itens]
    closest = np.argpartition([e['sum_dist'] for e in evaluations], range(n))[:n]
    return closest


def hill_climbing(itens, k, num_best):
    available_itens = list(range(len(itens)))
    np.random.shuffle(available_itens)
    # choose the firsts itens of clusters randomly
    clusters = [[itens[available_itens.pop()]] for _ in range(k)]

    current_cluster = 0
    while available_itens:
        closest = available_closest_itens(clusters[current_cluster], [itens[p] for p in available_itens], num_best)
        choiced = available_itens[closest[np.random.randint(len(closest))]]
        clusters[current_cluster].append(itens[choiced])
        available_itens.remove(choiced)
        current_cluster = (current_cluster + 1) % k

    return clusters


def grasp(itens, k, num_best, iter_max, max_time):
    best_solution = hill_climbing(itens, k, num_best)
    best_value = objective_function(evaluate_clusters(best_solution))
    iter = 0
    start = time.process_time()

    while True:
        solution = hill_climbing(itens, k, num_best)
        value = objective_function(evaluate_clusters(solution))
        if value < best_value:
            best_solution = solution
            best_value = value

        iter += 1
        end = time.process_time()
        if iter >= iter_max:
            #print("--- Break by iteration!")
            break
        if end-start > max_time:
            #print("--- Break by time!")
            break
    
    #print("")
    #print("Objective function = %.4f" % best_value)
    #print("Elapsed time = %.4f" % (end-start))
    #[print(', '.join(sorted([str(t['id']) for t in s]))) for s in best_solution]
    return best_solution
    #return best_value, end-start
