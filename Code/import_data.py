from binance.client import Client
import pandas as pd
import numpy as np
import requests                    # for "get" request to API
import json                        # parse json into a list               # working with data frames
import datetime as dt              # working with dates
import matplotlib.pyplot as plt
from cryptocmd import CmcScraper

def get_binance_bars(symbol, interval, startTime, endTime):

    url = "https://api.binance.com/api/v3/klines"

    startTime = str(int(startTime.timestamp() * 1000))
    endTime = str(int(endTime.timestamp() * 1000))
    limit = '1000'

    req_params = {"symbol": symbol, 'interval': interval, 'startTime': startTime, 'endTime': endTime, 'limit': limit}

    df = pd.DataFrame(json.loads(requests.get(url, params=req_params).text))

    if (len(df.index) == 0):
        return None

    df = df.loc[:, [0,1,2,3,4,5,8]]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume','trades']

    df.open = df.open.astype("float")
    df.high = df.high.astype("float")
    df.low = df.low.astype("float")
    df.close = df.close.astype("float")
    df.volume = df.volume.astype("float")
    df.trades = df.trades.astype("float")


    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.datetime]
    df.drop('datetime',axis = 1, inplace = True)

    return df

def get_history_df(symble, interval,from_date = dt.datetime(2017, 1, 1), to_date =  dt.datetime(2021, 1, 1)):

    df_list = []
    last_datetime = from_date
    while True:
        print(last_datetime)
        new_df = get_binance_bars(symble, interval, last_datetime, to_date)
        if new_df is None:
            break
        df_list.append(new_df)
        last_datetime = new_df.index[-1]

    df = pd.concat(df_list).drop_duplicates()

    return df

def get_market_cap(symbol):
    # initialise scraper without time interval
    scraper = CmcScraper(symbol)
    df = scraper.get_dataframe()
    df = df.set_index(pd.to_datetime(df['Date']))
    market_cap = df[['Market Cap']]

    return market_cap


def print_crypto_csv(selected_crypto, tick,fiat = 'USDT'):

    now = dt.datetime.now()
    final_df = pd.DataFrame(index=pd.date_range(selected_crypto[0][1],now,freq=tick+'in'))

    for i in selected_crypto:

        crypto = i[0]+fiat

        df = get_history_df(crypto, tick, from_date = i[1],to_date = now)
        market_cap = get_market_cap(i[0])
        df = df.join(market_cap)

        df = df.reset_index().pivot_table(columns=["index"]).T ###############   Aggiunto per eliminare indici duplicati

        df.to_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data\\data_2021/{}.csv".format(i[0]))

        final_df = final_df.join(df,rsuffix=i[0])

    return final_df


if __name__ == '__main__':

    import os

    rerun = True # Se true ricrea i dati

    if rerun or os.path.exists("C:\\Users\\stesc\\Desktop\\crypti\\Datadata_2021/cryptos_27_01_2022.csv") == False:

        selected_crypto = ['BTC','ETH', 'LTC']

        from_date = [dt.datetime(2021, 2, 25, 12), dt.datetime(2021, 2, 25, 12), dt.datetime(2021, 2, 25, 12)]

        crypto_input = [i for i in zip(selected_crypto, from_date)]

        cryptos = print_crypto_csv(crypto_input,'3m',fiat = 'USDT')

        # cryptos.to_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data\\data_2021/cryptos_27_01_2022.csv") ### Usare per unico CSV con tutte le crypto

    else:
        cryptos = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/cryptos_25_02_2021.csv", index_col=0)