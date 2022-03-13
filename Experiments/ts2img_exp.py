import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Experiments.img_interpolation import interpolate_pixels
import pickle

"""
Una volta creata la griglia un pattern sara descrivibile con un array flat che indica le posizioni dei settori popolati e il loro valore come valore. 
Come 1 hot encode lungo grid_sizexgrid_size.
Poi non sara necessario sapere dove esattamente si forma una costruzione, perche importa solo il momento in cui viene vista che dovrebbe
essere sempre l'ultimo settore sull asse X
"""


data = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/LTC.csv",index_col=0)
data.set_index(pd.to_datetime(data.index), inplace= True)

data.close = data.close.interpolate('quadratic')


data = data.reset_index().pivot_table(columns=["index"]).T

plt.plot(data.close)

# RESAMPLE LIBRARY : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html

freqs = ['6H','H','15T']

patterns = {}
snapshots = {}
snapshots.fromkeys(freqs)

for f in freqs:

    print(f)
    snapshots[f] = {}

    freqs_data = data.resample(f).asfreq()
    freqs_data = freqs_data.interpolate('quadratic')
    ts = freqs_data.close.to_numpy()
    # ts = (ts - min(ts)) / (max(ts) - min(ts))  ## ts tra 0 e 1

    print(np.argwhere(np.isnan(ts)))

    grid_size = 128
    counter = grid_size
    next = 2
    target_size = 4

    while (freqs_data.shape[0] - target_size - grid_size) % next != 0:

        freqs_data = freqs_data.iloc[:-1]

    n_windows = int(1 + (freqs_data.shape[0] - target_size - grid_size)/next)
    window =  np.zeros((n_windows, grid_size ** 2))


    targets = np.zeros((n_windows, target_size))
    row = 0

    while counter <= window.shape[0]-target_size:

        w = ts[counter-grid_size:counter] # window
        target = ts[counter: counter+target_size]

        target = target / w[-1]
        w = w/w[-1]

        mat0 = np.zeros((grid_size, grid_size))
        boundaries = np.linspace(min(w) - np.std(w), max(w) + np.std(w), grid_size)

        for i, j in enumerate(w):
            for m in reversed(range(len(boundaries) - 1)):

                if boundaries[m] < j <= boundaries[m + 1]:
                    mat0[len(boundaries) - 1 - m, i] = j

        flat_grid = mat0.flatten().reshape(1,-1)
        window[row,:] = flat_grid # ogni flat come riga di una matrice che racchiude tutto
        targets[row,:] = target

        if target[-1]<0.995:

            cat_target = 2

        elif target[-1] > 1.005:

            cat_target = 1
        else:
            cat_target = 0

        snapshots[f][freqs_data.index[counter-grid_size].strftime("%Y-%m-%d %H:%M:%S")] = (interpolate_pixels(mat0), target, cat_target)

        row += 1
        counter += next

    patterns[f] = window

##### SAVE DICTIONARIES

# patterns_file = open("Experiments/flat_patterns.pkl", "wb")
# pickle.dump(patterns, patterns_file)
# patterns_file.close()
snapshots_file = open("Experiments/snapshots.pkl", "wb")
pickle.dump(snapshots, snapshots_file)
snapshots_file.close()

##### PLOT SNAPSHOT

freq_snapshot = '15T'
time_snapshot = '2020-01-01 12:30:00'

fig = plt.figure(frameon=False)
fig.set_size_inches(3,3)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(snapshots[freq_snapshot][time_snapshot][0], cmap = 'gray')
# fig.savefig('Experiments/ts_img.png')
"""

Preparare filtri per convolution1.

"""

# Symmetric triangols : '2021-01-25T22:30:00' : '2021-01-26T01:300:00'
#                     : '2020-07-07' : '2020-07-24'
#                     : '2021-02-18T07:45:00' : '2021-02-18T12:00:00'

# sym_tr_data = data['2021-02-18T07:45:00' : '2021-02-18T12:00:00']
# # sym_tr_data = sym_tr_data.iloc[:32]
# sym_tr_fr = sym_tr_data.resample('2T').asfreq()
# plt.plot(sym_tr_data.close)


######################################################## 1D CNN on windows
