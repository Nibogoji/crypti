import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trader.last_version.filter_modeling_v2 import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold


def pca_clustering( data5, n_syn_features=3, limit_training=True, reference_points=10, time_out=False):

    syn_features_target_idx = [i for i in range(n_syn_features)]
    syn_features_target_idx.append(-1)

    training_data5 = data5


    train_Z5, train_features_5, train_target5 = make_categorical_target(training_data5, plot=False)
    for c in range(train_features_5.shape[1]):
        if np.array_equal(train_features_5.iloc[:, c].values.reshape(1, -1), train_target5.reshape(1, -1),
                          equal_nan=False):
            print('Sanity check failed : Target in features')
            break


    # train_features = train_features5

    categorical_train_target5 = train_Z5.reshape(-1, 1)

    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features_5)

    X_train_reduced, var_vals, pc_receipt, pcs_space, res, pcs_importance = custom_pca(scaled_train_features, n_pcs=n_syn_features)
    # What new pcs are made of
    pc_receipt = pd.DataFrame(pc_receipt, columns=['Pc {}'.format(i) for i in range(X_train_reduced.shape[1])])
    # Most n important features to build pcs during the whole experiment to get more or less 12 most importants
    most_important_features = []
    most_important_indexes = []
    importance = []
    for _ in range(int(8 / pcs_space.shape[0])):
        imp = [i for i in pc_receipt.max(axis=0)]
        importance += list(imp * pcs_importance)
        most_important_indexes += [i for i in pc_receipt.idxmax(axis=0)]
        most_important_features += [train_features_5.columns[i] for i in pc_receipt.idxmax(axis=0)]
        pc_receipt = pc_receipt.drop(most_important_indexes[-pcs_space.shape[0]:], axis=0)

    importance = np.asarray(importance).reshape(1, -1)
    importance = importance / np.sum(importance)

    # current_pcs_receipt = pd.DataFrame(data=importance, columns=most_important_features)

    avg_price =training_data5.Price.mean()

    return pcs_space, categorical_train_target5, avg_price


ohlc = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/LTC.csv",index_col=0)
ohlc.index = pd.to_datetime(ohlc.index)
ohlc= ohlc[~ohlc.index.duplicated(keep='first')]

ohlc = ohlc.loc[:,['open', 'high', 'low', 'close']]
ohlc.columns = ['Open', 'High', 'Low', 'Close']

tech_ind = pd.read_csv('Code/dati_provvisori/indicatori.csv', index_col=0)
tech_ind.index = pd.to_datetime(tech_ind.index)
tech_ind= tech_ind[~tech_ind.index.duplicated(keep='first')]
tech_ind.drop(['open', 'high', 'low', 'close','volume'], axis=1, inplace = True)

data_5 = pd.read_csv('Code/dati_provvisori/df_simultaion_v1.csv', index_col=0)
data_5.index = pd.to_datetime(data_5.index)
data_5 = pd.concat([ tech_ind,data_5], axis=1)

data_5['slow_returns'] = (data_5.Price.shift(-20)-data_5.Price)/data_5.Price

data_fast = data_5.drop(['target_shift_10', 'returns', 'target', 'cat_target','slow_returns'],axis = 1).copy()
data_slow = data_5.drop(['target_shift_10', 'returns', 'target', 'cat_target','synt_returns'],axis = 1).copy()

data_slow.rename(columns = {'slow_returns':'returns'}, inplace = True)
data_fast.rename(columns = {'synt_returns':'returns'}, inplace = True)

aap_data_5 = data_fast.copy()
aap_data_10 = data_slow.copy()

ohlc.index = pd.to_datetime(ohlc.index)
aap_data_5.index = pd.to_datetime(aap_data_5.index)
aap_data_10.index = pd.to_datetime(aap_data_10.index)

aap_data_5.dropna(inplace = True)
aap_data_10.dropna(inplace = True)

common_indexes = [i for i in aap_data_10.index if i in aap_data_5.index and i in ohlc.index]

aap_data_10 = aap_data_10.loc[common_indexes,:]
aap_data_5 = aap_data_5.loc[common_indexes,:]
ohlc = ohlc.loc[common_indexes,:]

# data_test = aap_data_5.iloc[-480:]
# aap_data_5 = aap_data_5.iloc[:-480]

returns = aap_data_5['returns']
data = aap_data_5

daily_price = data.resample('D').mean()
daily_price = daily_price['Price']

# perform cross-validation procedure
win_len = 480
splits = [i for i in range(0,data.shape[0] - win_len, win_len)]
windows = []
data = data.iloc[data.shape[0]%480:] # Per comprendere tutti i dati nelle windows

for i in splits:

    windows.append(data.iloc[i:i+win_len, :])


pca_spaces = []
avg_prices = []

for w in windows:

    pca_space, binary_target, avg_price = pca_clustering(w)
    pca_spaces.append(pca_space)
    avg_prices.append(avg_price)

labels = []

for i in range(len(avg_prices)-1):

    if avg_prices[i+1]> avg_prices[i]*1.05:
        labels.append(2)
    elif  avg_prices[i]*0.95<avg_prices[i+1]< avg_prices[i]*1.05:
        labels.append(1)
    else:
        labels.append(0)
labels.append(1)

#'prendere trend detection per fare labeling degli split e plottare per vedere se hanno una disposizione sensata'

global_reference_space = np.zeros(pca_spaces[0].shape)
n_pcs = 3

from scipy.spatial import distance
dim1 = np.zeros((1,pca_spaces[0].shape[1]))
dim2 = np.zeros((1,pca_spaces[0].shape[1]))
dim3 = np.zeros((1,pca_spaces[0].shape[1]))

diffs1 = []
diffs2 = []
diffs3 = []

for i,s in enumerate(pca_spaces):

    new_dim1 = s[0,:]
    new_dim2 = s[1,:]
    new_dim3 = s[2,:]

    diff1 = distance.sqeuclidean(dim1/(i+1), new_dim1)
    diff2 = distance.sqeuclidean(dim2/(i+1), new_dim2)
    diff3 = distance.sqeuclidean(dim3/(i+1), new_dim3)

    diffs1.append(diff1)
    diffs2.append(diff2)
    diffs3.append(diff3)

    dim1 += new_dim1
    dim2 += new_dim2
    dim3 += new_dim3


dim1 = dim1/len(pca_spaces)
dim2 = dim2/len(pca_spaces)
dim3 = dim3/len(pca_spaces)

global_reference_space[0,:] = dim1
global_reference_space[1,:] = dim2
global_reference_space[2,:] = dim3

plt.figure()
plt.plot(diffs1, label = '1st pc')
plt.plot(diffs2, label = '2nd pc')
plt.plot(diffs3, label = '3rd pc')
plt.plot(daily_price.values/10)
plt.legend()

plt.figure()
plt.scatter(global_reference_space[0,:],global_reference_space[1,:])
plt.title('Global Reference Representations')

# pca_spaces = pca_spaces[:15]

binary_colors = ['red',  'gray','green']
colors = plt.cm.rainbow(np.linspace(0, 1, len(pca_spaces)))
plt.figure()
plt.scatter(global_reference_space[0,:],global_reference_space[1,:])
for i,s in enumerate(pca_spaces):
    plt.scatter(s[0,:], s[1,:],facecolors='none', edgecolors= colors[i])

plt.figure()
plt.scatter(global_reference_space[0,:],global_reference_space[1,:])
for i,s in enumerate(pca_spaces):
    plt.scatter(s[0,:], s[1,:],facecolors='none', edgecolors= binary_colors[labels[i]])
plt.title('Window returns')


def matrix_distance(A, B, squared=False):
    """
    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared

d_from_reference = []
d_avg = []

for s in pca_spaces:

    d_from_reference.append(matrix_distance(global_reference_space, s))
    d_avg.append(matrix_distance(global_reference_space, s).mean(axis = 1))

plt.figure()
for i in d_avg:
    plt.scatter(i[0], i[1])


plt.figure()
for i,s in enumerate(d_from_reference):
    plt.scatter(s[0,:], s[1,:],facecolors='none', edgecolors=colors[i])


plt.figure()
for i,s in enumerate(d_from_reference):
    plt.scatter(s[0,:], s[1,:], facecolors='none', edgecolors= binary_colors[labels[i]])

""" Ci sono dei chiari cluster, ma non ho idea quale sia il target che meglio li possa descrivere,,, e se si trovasse un modo per scegliere il target dati i cluster?
Una relazione tra i dati e il futuro e del tipo : t = p(t+a) + b * p(t) tale che i cluster siano piu puliti possibile'

"""
a = np.asarray(d_from_reference).reshape(-1,3)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0).fit(a)
k_labels = kmeans.labels_


plt.figure()
for i,s in enumerate(a):
    plt.scatter(s[0], s[1], facecolors='none', edgecolors= binary_colors[k_labels[i]]) # Unsupervised clustered

clustered_pca = np.concatenate([a,k_labels.reshape(-1,1)], axis=1)
flat_clustered_pca = clustered_pca.reshape(len(d_from_reference),3,4)

clusters = []
for w in range(flat_clustered_pca.shape[0]):
    if flat_clustered_pca[w,:,3].mean() <= 1:
        clusters.append([w,1])
    elif 1< flat_clustered_pca[w,:,3].mean() <=2:
        clusters.append([w,2])
    else:
        clusters.append([w, 3])
data_clusters = {1:dict(),
                 2:dict(),
                 3:dict()}
for w in clusters:
    if w[1] == 1:
        data_clusters[1][w[0]] = windows[w[0]]
    elif w[1] == 2:
        data_clusters[2][w[0]] = windows[w[0]]
    else:
        data_clusters[3][w[0]] = windows[w[0]]


# vediamo se i 2 clusters hanno effetti sul prezzo

plt.figure()
index_1 = []
for d in data_clusters[1].values():

    index_1.append(d.index)
    plt.plot(d.Price, color = 'red')

index_2 = []
for d in data_clusters[2].values():

    index_2.append(d.index)
    plt.plot(d.Price, color = 'blue')
index_3 = []
for d in data_clusters[3].values():

    index_3.append(d.index)
    plt.plot(d.Price, color = 'green')
