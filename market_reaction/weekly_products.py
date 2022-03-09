import pandas as pd
import pickle
import pytz
import datetime
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from Data.codes.plots import *
from Data.codes.data_manipulation import *
os.chdir('C:\\Users\\S49399\\.PyCharmCE2019.2\\projects\\merket_reaction')

####################################### Tick data ###############################
l1_ex= pd.read_csv('./Data/power_data_for_stefano/data/L1 Germany Baseload EEX Wk15-18.csv')
l2_ex = pd.read_csv('./Data/power_data_for_stefano/data/L2 Germany Baseload EEX Wk01-19.csv')


def get_weekly_product():

    trades_w_records = {}
    orders_book = {}
    l1_book = {}

    folders = glob('./Data/power_data_for_stefano/*')[:2]

    for fol in folders:
        os.chdir('C:\\Users\\S49399\\.PyCharmCE2019.2\\projects\\merket_reaction')
        os.chdir(fol)
        files = glob('./*.csv')

        for f in files:

            if 'TRADES' in f and 'Wk' in f and 'WkEnd' not in f:

                print(f)
                df = pd.read_csv(f, index_col=0)

                if 'EEX ' in f:
                    f = f.replace('EEX ', '')

                symbol = f.split('\\')[1]

                week = symbol.split(' ')[3].split('.')[0]
                week = week.split('k')[1].split('-')
                week = week[1] + '-' + week[0]

                df.set_index('deal_date', inplace=True)

                df = df.dropna(axis=0, how='all')
                trades_w_records[week] = df

            elif 'L2' in f and 'Wk' in f and 'WkEnd' not in f:

                print(f)
                df = pd.read_csv(f, index_col=0)

                if 'EEX ' in f:
                    f = f.replace('EEX ', '')

                symbol = f.split('\\')[1]

                week = symbol.split(' ')[3].split('.')[0]
                week = week.split('k')[1].split('-')
                week = week[1] + '-' + week[0]

                df.set_index('snapshot_timestamp', inplace=True, drop=True)

                df = df.dropna(axis=0, how='all')
                orders_book[week] = df

            elif 'L1' in f and 'Wk' in f and 'WkEnd' not in f:

                print(f)
                df = pd.read_csv(f)

                if 'EEX ' in f:
                    f = f.replace('EEX ', '')

                symbol = f.split('\\')[1]

                week = symbol.split(' ')[3].split('.')[0]
                week = week.split('k')[1].split('-')
                week = week[1] + '-' + week[0]

                df.set_index('snapshot_time', inplace=True)

                df = df.dropna(axis=0, how='all')

                l1_book[week] = df

    return l1_book,orders_book,trades_w_records

l1_book,orders_book,trades_w_records = get_weekly_product()

keys = list(orders_book.keys())
ordered_keys = sorted(keys)

trades_w_sort = {}
book_w_sort = {}
l1_book_w_sorted = {}

for k in ordered_keys:

    trades_w_sort[k] = trades_w_records[k]
    book_w_sort[k] = orders_book[k]
    l1_book_w_sorted[k] = l1_book[k]

w_diag_dict, ohlc_dict = get_diagonal_history(book_w_sort)  # Gives dictionary with actions per each delivery and ohlc data

#################################  plot OHLC for weekly products  ##########################################

ohlc_plot_weekly(ohlc_dict, year = 2020, week = 42, mav = (1,3))

