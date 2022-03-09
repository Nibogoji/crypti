import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mplf

"""
https://github.com/matplotlib/mplfinance/blob/master/examples/using_lines.ipynb

"""

data = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/cryptos_25_02_2021.csv", index_col=0)

data.iloc[:,3].plot()

df = data.interpolate()
df.iloc[:,3].plot()


closes = []
for c in df.columns:
    if 'close' in c:

        closes.append(c)

close_df = df[closes]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_close = scaler.fit_transform(close_df.values)
scaled_close = pd.DataFrame(data=scaled_close,columns=closes,index=close_df.index)

scaled_close.plot()