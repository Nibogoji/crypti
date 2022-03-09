import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/LTC.csv",index_col=0)
data.drop(['trades','Market Cap'],axis = 1,inplace=True)
data.index = pd.to_datetime(data.index)

data= data[~data.index.duplicated(keep='first')]

df = data.copy()

def SMA(df, periods=50):
    """
    Calculating the Simple Moving Average for the past n days

    **Values must be descending**
    """
    lst = []

    for i in range(len(df)):
        if i < periods:

            # Appending NaNs for instances unable to look back on
            lst.append(np.nan)

        else:
            # Calculating the SMA
            lst.append(round(np.mean(df[i:periods + i]), 2))

    return lst


def Stoch(closes, lows, highs, periods=14, d_periods=3):
    """
    Calculating the Stochastic Oscillator for the past n days

    **Values must be descending**
    """
    k_lst = []

    d_lst = []

    for i in range(len(closes)):
        if i < periods:

            # Appending NaNs for instances unable to look back on
            k_lst.append(np.nan)

            d_lst.append(np.nan)

        else:

            # Calculating the Stochastic Oscillator

            # Calculating the %K line
            highest = max(highs[i:periods + i])
            lowest = min(lows[i:periods + i])

            k = ((closes[i] - lowest) / (highest - lowest)) * 100

            k_lst.append(round(k, 2))

            # Calculating the %D line
            if len(k_lst) < d_periods:
                d_lst.append(np.nan)
            else:
                d_lst.append(round(np.mean(k_lst[-d_periods - 1:-1])))

    return k_lst, d_lst


def RSI(df, periods=14):
    """
    Calculates the Relative Strength Index

    **Values must be descending**
    """

    df = df.diff()

    lst = []

    for i in range(len(df)):
        if i < periods:

            # Appending NaNs for instances unable to look back on
            lst.append(np.nan)

        else:

            # Calculating the Relative Strength Index
            avg_gain = (sum([x for x in df[i:periods + i] if x >= 0]) / periods)
            avg_loss = (sum([abs(x) for x in df[i:periods + i] if x <= 0]) / periods)

            rs = avg_gain / avg_loss

            rsi = 100 - (100 / (1 + rs))

            lst.append(round(rsi, 2))

    return lst

import ta

kama = ta.momentum.kama(df.close, pow1 = 4, pow2 = 12 )
# TA's RSI
df['kama_rsi'] = ta.momentum.rsi(kama)
# TA's Stochastic Oscillator
df['kama_stoch'] = ta.momentum.stoch(df.high, df.low, df.close)
df['kama_stoch_d'] = ta.momentum.stoch_signal(df.high, df.low, df.close)

from ta.volatility import BollingerBands
from ta.trend import PSARIndicator
from ta.trend import macd
from ta.momentum import rsi
from ta.trend import SMAIndicator

def AddIndicators(df):

    # Add Simple Moving Average (SMA) indicators
    df["sma7"] = SMAIndicator(close=df["close"], window=7, fillna=True).sma_indicator()
    df["sma25"] = SMAIndicator(close=df["close"], window=25, fillna=True).sma_indicator()
    df["sma99"] = SMAIndicator(close=df["close"], window=99, fillna=True).sma_indicator()

    # Add Bollinger Bands indicator
    indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Parabolic Stop and Reverse (Parabolic SAR) indicator
    indicator_psar = PSARIndicator(high=df["high"], low=df["low"], close=df["close"], step=0.02, max_step=2,
                                   fillna=True)
    df['psar'] = indicator_psar.psar()

    # Add Convergence Divergence indicator

    df["MACD"] = macd(close=df["close"], window_slow=26, window_fast=12, fillna=True)

    # Add regular RSI

    df["RSI"] = rsi(close=df["close"], window=14, fillna=True)

    return df


df = AddIndicators(df)

# plt.figure()
# plt.plot(df["RSI"][-5000:], label = 'kama')
# plt.plot(df["kama_rsi"][-5000:])
# plt.plot(df["close"][-5000:])
# plt.legend()

# df = df.reset_index(drop= True)
# fig, axs = plt.subplots(4, 1)
#
# axs[0].plot(df.close)
#
# axs[1].plot(df['ta_stoch_k'])
# axs[1].plot(np.ones((df.shape[0], 1))*80)
# axs[1].plot(np.ones((df.shape[0], 1))*14)
#
# axs[2].plot(df['ta_stoch_d'])
# axs[2].plot(np.ones((df.shape[0], 1))*80)
# axs[2].plot(np.ones((df.shape[0], 1))*11)
#
# axs[3].plot(df['ta_rsi'])
# axs[3].plot(np.ones((df.shape[0], 1))*85)
# axs[3].plot(np.ones((df.shape[0], 1))*10)

"""
RSI BASATO SU KAMA ( PREZZO PULITO DA NOISE ) SEMBRA AVERE VALORE : COMPRA QUANDO VA DA OVERSOLD A NEUTRALE E VENDI QUANDO VA DA OVERBOUGHT A NEUTRALE

DA PROVARE, TROVARE LA AMPIEZZA MEDIA TRA SWINGS E ACCETTARE UN MASSIMO/MINIMO SOLO SE ARRIVA DOPO CHE L'AMPIEZZA VIENE REALIZZATA
"""

####################### ICA su kama RSI https://towardsdatascience.com/separating-mixed-signals-with-independent-component-analysis-38205188f2f4

df.to_csv('Code/dati_provvisori/indicatori.csv')