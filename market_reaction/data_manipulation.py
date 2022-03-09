import numpy as np
import pandas as pd



def generate_diagonal_market(data):

    indexes = []

    for k in data.keys():

        df = data[k]

        times = df.index
        indexes.append(times)

    flat_indexes = [t for w in indexes for t in w]
    diagonal_df = pd.DataFrame( index = flat_indexes)
    ohlc = {}
    for k in data.keys():

        print(k)

        df = data[k]
        df_price = pd.DataFrame(index=pd.to_datetime(df.index), data=df['price'].values, columns=[k])
        ohlc[k] = df_price.resample('1D').ohlc()
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        ohlc[k].columns = ohlc_columns
        diagonal_df = diagonal_df.join(df_price)


    return diagonal_df, ohlc

def get_diagonal_history(product):

    products_2018 = [i  for i in product.keys() if i.split('-')[0]=='18']
    products_2019 = [i  for i in product.keys() if i.split('-')[0]=='19']
    products_2020 = [i  for i in product.keys() if i.split('-')[0]=='20']

    weekly_products = [products_2018,products_2019,products_2020]
    weekly_products_str = ['products_2018','products_2019','products_2020']

    w_diag_dict = {}
    ohlc_dict = {}

    for p,n in zip(weekly_products, weekly_products_str):

        w_diag_dict[n], ohlc_dict[n]= generate_diagonal_market({key: value for key, value in product.items() if key in p})

    return w_diag_dict, ohlc_dict