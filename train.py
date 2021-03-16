import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.datasets import load_iris, load_wine
from SA import simulated_annealing
from GRASP import grasp
from AG import genetic
matplotlib.use('TkAgg')


# load #############################################################################################################
iris = load_iris()['data']
iris = [{'id': x, 'coord': y} for x, y in zip(range(len(iris)), iris)]
k_I = [3, 7, 10, 13, 22]

wine = load_wine()['data']
wine = [{'id': x, 'coord': y} for x, y in zip(range(len(iris)), wine)]
k_W = [2, 6, 9, 11, 33]



# utils ############################################################################################################
max_time = 1

config = {}
best_mean = {}
best_rank = {}
top_5_mean = {}
rank_all = {}
rank_all_mean = {}
def final(I, W, Z_score_I, Z_score_W, rank_I, rank_W, metodo):
    df_mean = {}
    df_time = {}
    df_rank = {}
    df_Z_score = {}
    for key, value in I.items():
        df_mean['iris_'+str(key)] = value[:,0]
        df_time['iris_'+str(key)] = value[:,1]
    for key, value in rank_I.items():
        df_rank['iris_'+str(key)] = value
    for key, value in Z_score_I.items():
        df_Z_score['iris_'+str(key)] = value
    for key, value in W.items():
        df_mean['wine_'+str(key)] = value[:,0]
        df_time['wine_'+str(key)] = value[:,1]
    for key, value in rank_W.items():
        df_rank['wine_'+str(key)] = value
    for key, value in Z_score_W.items():
        df_Z_score['wine_'+str(key)] = value
    df_mean = pd.DataFrame(df_mean)
    df_time = pd.DataFrame(df_time)
    df_rank = pd.DataFrame(df_rank)
    df_Z_score = pd.DataFrame(df_Z_score)

    tmp = df_mean.iloc[:, :len(I)]
    tmp.columns = I.keys()
    sn.boxplot(data=tmp)
    frame = plt.gca()
    plt.ylabel(ylabel="Média - iris")
    plt.xlabel(xlabel="k")
    plt.savefig(metodo+'_iris_boxplot.png')

    tmp = df_mean.iloc[:,len(I):]
    tmp.columns = W.keys()
    sn.boxplot(data=tmp)
    frame = plt.gca()
    plt.ylabel(ylabel="Média - wine")
    plt.xlabel(xlabel="k")
    plt.savefig(metodo+'_wine_boxplot.png')

    # Obter média, desvio padrão e ranqueamento médio da configuração
    configs['mean'] = df_mean.mean(axis=1)
    configs['std'] = df_mean.std(axis=1)
    configs['rank'] = df_rank.mean(axis=1)

    # Obter melhor configuração por média e por ranqueamento médio do método
    best_mean = configs.iloc[np.argmin(configs['mean'])]
    best_rank = configs.iloc[np.argmin(configs['rank'])]

    # Obter as 5 melhores resultados de médias padronizadas e os 
    # tempos correspondentes das configurações de cada método
    idx = np.argpartition(df_Z_score.values.flatten(), range(5))[:5]
    nrow = len(df_Z_score.index)
    top_5_mean = [[configs[['t','alfa','iter_max']].iloc[int(c/nrow)].values, df_Z_score.values.flatten()[c], df_time.values.flatten()[c]] for c in idx]

    # Obter ranqueamento obtido por cada configuração de
    # método em cada problema e seu ranqueamento médio
    rank_all = stats.rankdata(df_mean.values.flatten())
    rank_all_mean = rank.mean()

    return config, best_mean, best_rank, top_5_mean, rank_all, rank_all_mean



# Simulated Annealing ##############################################################################################
a = [[500, 100, 50], [0.95, 0.85, 0.7], [350, 500]]
configs = [list(x) for x in np.array(np.meshgrid(*a)).T.reshape(-1,len(a))]
configs = pd.DataFrame(configs, columns=['t','alfa','iter_max'])

# iris
size_neighborhood = int(0.05*len(iris))
I = {}
Z_score_I = {}
rank_I = {}
for k in k_I:
    I[k] = []
    for index, row in configs.iterrows():
        result = [simulated_annealing(iris, k, row['t'], row['alfa'], size_neighborhood, int(row['iter_max']), max_time) for _ in range(10)]
        I[k].append(np.mean(result, axis=0))
    I[k] = np.array(I[k])
    Z_score_I[k] = stats.zscore(I[k][:,0])
    rank_I[k] = stats.rankdata(I[k][:,0])

# wine
size_neighborhood = int(0.05*len(wine))
W = {}
Z_score_W = {}
rank_W = {}
for k in k_W:
    W[k] = []
    for index, row in configs.iterrows():
        result = [simulated_annealing(wine, k, row['t'], row['alfa'], size_neighborhood, int(row['iter_max']), max_time) for _ in range(10)]
        W[k].append(np.mean(result, axis=0))
    W[k] = np.array(W[k])
    Z_score_W[k] = stats.zscore(W[k][:,0])
    rank_W[k] = stats.rankdata(W[k][:,0])

config['SA'], best_mean['SA'], best_rank['SA'], top_5_mean['SA'], rank_all['SA'], rank_all_mean['SA'] = final(I, W, Z_score_I, Z_score_W, rank_I, rank_W, 'SA')



# GRASP ############################################################################################################
a = [[20, 50, 100, 200, 350, 500], [5, 10, 15]]
configs = [list(x) for x in np.array(np.meshgrid(*a)).T.reshape(-1,len(a))]
configs = pd.DataFrame(configs, columns=['iter_max','numBest'])

# iris
I = {}
Z_score_I = {}
rank_I = {}
for k in k_I:
    I[k] = []
    for index, row in configs.iterrows():
        result = [grasp(iris, k, int(row['numBest']), int(row['iter_max']), max_time) for _ in range(10)]
        I[k].append(np.mean(result, axis=0))
    I[k] = np.array(I[k])
    Z_score_I[k] = stats.zscore(I[k][:,0])
    rank_I[k] = stats.rankdata(I[k][:,0])

# wine
W = {}
Z_score_W = {}
rank_W = {}
for k in k_W:
    W[k] = []
    for index, row in configs.iterrows():
        result = [grasp(wine, k, int(row['numBest']), int(row['iter_max']), max_time) for _ in range(10)]
        W[k].append(np.mean(result, axis=0))
    W[k] = np.array(W[k])
    Z_score_W[k] = stats.zscore(W[k][:,0])
    rank_W[k] = stats.rankdata(W[k][:,0])

config['GRASP'], best_mean['GRASP'], best_rank['GRASP'], top_5_mean['GRASP'], rank_all['GRASP'], rank_all_mean['GRASP'] = final(I, W, Z_score_I, Z_score_W, rank_I, rank_W, 'GRASP')



# AG ###############################################################################################################
a = [[10, 30, 50],[0.75, 0.85, 0.95],[0.10, 0.20]]
configs = [list(x) for x in np.array(np.meshgrid(*a)).T.reshape(-1,len(a))]
configs = pd.DataFrame(configs, columns=['pop_size','cross_ratio','mut_ratio'])

# iris
I = {}
Z_score_I = {}
rank_I = {}
for k in k_I:
    I[k] = []
    for index, row in configs.iterrows():
        result = [genetic(iris, k, int(row['pop_size']), int(row['iter_max']), row['cross_ratio'], row['mut_ratio'], max_time) for _ in range(10)]
        I[k].append(np.mean(result, axis=0))
    I[k] = np.array(I[k])
    Z_score_I[k] = stats.zscore(I[k][:,0])
    rank_I[k] = stats.rankdata(I[k][:,0])

# wine
W = {}
Z_score_W = {}
rank_W = {}
for k in k_W:
    W[k] = []
    for index, row in configs.iterrows():
        result = [genetic(wine, k, int(row['pop_size']), int(row['iter_max']), row['cross_ratio'], row['mut_ratio'], max_time) for _ in range(10)]
        W[k].append(np.mean(result, axis=0))
    W[k] = np.array(W[k])
    Z_score_W[k] = stats.zscore(W[k][:,0])
    rank_W[k] = stats.rankdata(W[k][:,0])

config['AG'], best_mean['AG'], best_rank['AG'], top_5_mean['AG'], rank_all['AG'], rank_all_mean['AG'] = final(I, W, Z_score_I, Z_score_W, rank_I, rank_W, 'AG')
