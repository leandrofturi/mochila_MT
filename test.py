import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from SA import simulated_annealing
from GRASP import grasp
from AG import genetic
from comons import evaluate_clusters, objective_function
matplotlib.use('TkAgg')


# load #############################################################################################################
max_time = 1

iris = load_iris()['data']
iris = [{'id': x, 'coord': y} for x, y in zip(range(len(iris)), iris)]
k_I = [2, 4, 8, 11, 15, 17, 23, 28, 32, 50]

wine = load_wine()['data']
wine = [{'id': x, 'coord': y} for x, y in zip(range(len(wine)), wine)]
k_W = [3, 5, 13, 15, 20, 23, 25, 30, 41, 45]

ionosphere = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data')
ionosphere = np.asarray(ionosphere.iloc[:,:34])
ionosphere = [{'id': x, 'coord': y} for x, y in zip(range(len(ionosphere)), ionosphere)]
k_H =  [2, 3, 5, 10, 15, 20, 25, 30, 40, 50]


# utils ############################################################################################################
# kmeans
def kmeans(dataset, k):
    tmp = [i['coord'] for i in dataset]
    start = time.process_time()
    kmeans = KMeans(n_clusters=k).fit(tmp)
    end = time.process_time()
    
    clusters = [[dataset[i] for i in range(len(dataset)) if kmeans.labels_[i] == l] for l in np.unique(kmeans.labels_)]
    states = evaluate_clusters(clusters)
    return objective_function(states), end-start

def kmeans_twice(dataset, k):
    result = [kmeans(dataset, k) for _ in range(20)]
    return np.mean(result, axis=0)


# Simulated Annealing ##############################################################################################
# iris
I = {}
for k in k_I:
    I[k] = {}
    result = [simulated_annealing(iris, k, 100, 0.85, 7, 350, max_time) for _ in range(2)]
    I[k]['SA'] = np.mean(result, axis=0)
    result = [grasp(iris, k, 200, 5, max_time) for _ in range(2)]
    I[k]['GRASP'] = (np.mean(result, axis=0))
    result = [genetic(iris, k, 50, 100, 0.75, 0.2, max_time) for _ in range(2)]
    I[k]['AG'] = (np.mean(result, axis=0))
    I[k]['kmeans'] = kmeans_twice(iris, k)

    tmp = [v[0] for k, v in I[k].items()]
    df = pd.DataFrame({k: v.tolist() + [z, r] for (k, v), z, r in zip(I[k].items(), stats.zscore(tmp), stats.rankdata(tmp))})
    df.index = ["mean", "time", "zscore", "rank"]
    I[k]['df'] = df

# wine
W = {}
for k in k_W:
    W[k] = {}
    result = [simulated_annealing(wine, k, 100, 0.85, 8, 350, max_time) for _ in range(2)]
    W[k]['SA'] = np.mean(result, axis=0)
    result = [grasp(wine, k, 200, 5, max_time) for _ in range(2)]
    W[k]['GRASP'] = (np.mean(result, axis=0))
    result = [genetic(wine, k, 50, 100, 0.75, 0.2, max_time) for _ in range(2)]
    W[k]['AG'] = (np.mean(result, axis=0))
    W[k]['kmeans'] = kmeans_twice(wine, k)

    tmp = [v[0] for k, v in W[k].items()]
    df = pd.DataFrame({k: v.tolist() + [z, r] for (k, v), z, r in zip(W[k].items(), stats.zscore(tmp), stats.rankdata(tmp))})
    df.index = ["mean", "time", "zscore", "rank"]
    W[k]['df'] = df

# ionosphere
H = {}
for k in k_H:
    H[k] = {}
    result = [simulated_annealing(ionosphere, k, 100, 0.85, 17, 350, max_time) for _ in range(2)]
    H[k]['SA'] = np.mean(result, axis=0)
    result = [grasp(ionosphere, k, 200, 5, max_time) for _ in range(2)]
    H[k]['GRASP'] = (np.mean(result, axis=0))
    result = [genetic(ionosphere, k, 50, 100, 0.75, 0.2, max_time) for _ in range(2)]
    H[k]['AG'] = (np.mean(result, axis=0))
    H[k]['kmeans'] = kmeans_twice(ionosphere, k)

    tmp = [v[0] for k, v in H[k].items()]
    df = pd.DataFrame({k: v.tolist() + [z, r] for (k, v), z, r in zip(H[k].items(), stats.zscore(tmp), stats.rankdata(tmp))})
    df.index = ["mean", "time", "zscore", "rank"]
    H[k]['df'] = df



iris = load_iris()
X = iris.data[:, :2]
y = iris.target
KMEANS = KMeans(n_clusters=3).fit(X).labels_

iris = [{'id': x, 'coord': y} for x, y in zip(range(len(X)), X)]
max_time = 2
SA = make_target(simulated_annealing(iris, 3, 100, 0.85, 7, 350, max_time))
GRASP = make_target(grasp(iris, 3, 200, 5, max_time))
AG = make_target(genetic(iris, 3, 50, 100, 0.75, 0.2, max_time))


# Plot the identified clusters and compare with the answers
fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow', edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=KMEANS, cmap='gist_rainbow', edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)
plt.savefig("KMEANS.png")