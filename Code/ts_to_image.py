import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Data/ADA.csv')

ts = data.close.to_numpy()


grid = 58 # int(max(w)/np.std(w))

plt.figure()
w = ts[-grid:]
plt.plot(w)

plt.figure()
df_w = pd.DataFrame(w)
plt.plot(np.log(df_w)- np.log(df_w.shift(1)))



mat0 = np.zeros((grid, grid))
boundaries = np.linspace(min(w)-np.std(w), max(w)+np.std(w), grid)

for i,j in enumerate(w):
    for m in reversed(range(len(boundaries)-1)):

        if j > boundaries[m] and j <= boundaries[m+1]:

            mat0[len(boundaries)-1 - m,i] += j

mat0_df = pd.DataFrame(mat0)

recurrence = np.zeros((mat0.shape[0],1))

for i in range(mat0.shape[0]):
    for j in range(mat0.shape[1]):

        if mat0[i,j] != 0:

            recurrence[i] += 1

supp_and_roof = []
high_and_low = 2
cicli_in_periodo = 1

low = boundaries[-1]
high = boundaries[0]

band_selected = []

for i, j in enumerate(recurrence):


    if j >= high_and_low:

        if boundaries[i] < low:

            low = boundaries[i]
            support = (low, boundaries[i + 1])
                # supp_and_roof.append((boundaries[i], boundaries[i+1]))
            # except:
            #     # supp_and_roof.append((boundaries[i-1], boundaries[i]))
            #     low = (boundaries[i-1], boundaries[i])

        elif boundaries[i] > high:


            high = boundaries[i]
            roof = (boundaries[i-1], boundaries[i])
                # supp_and_roof.append((boundaries[i], boundaries[i+1]))
            # except:
            #     # supp_and_roof.append((boundaries[i-1], boundaries[i]))
            #     high = (boundaries[i], boundaries[i+1])



    # if (i > 0 and i < len(boundaries)) and mat0[i-1, ] > boundaries[i] and boundaries[i+1] > boundaries[i]:
    #     supp_and_roof.append((boundaries[i-1], boundaries[i + 1]))





        # try:
        #     supp_and_roof.append((boundaries[i], boundaries[i+1]))
        # except:
        #     supp_and_roof.append((boundaries[i-1], boundaries[i]))



supp_and_roof.append(support)
supp_and_roof.append(roof)

w = ts[-grid:]
fig, ax = plt.subplots()

ax.plot(w)
for k in range(len(supp_and_roof)):
    ax.fill_between(range(len(w)),supp_and_roof[k][0], supp_and_roof[k][1], alpha=0.2)



