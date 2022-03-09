import pandas as pd
from Data.codes.get_data import get_data_mdap
from Data.codes.get_data import get_curves_wattsight
from Data.codes.get_data import get_data_wattsight
import pickle
import wapi
import pytz
import datetime
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from Data.codes.plots import CustomLocator, plt2grid, ohlc_plotly, ohlc_mpf

os.chdir('C:\\Users\\S49399\\.PyCharmCE2019.2\\projects\\merket_reaction')

####################################### Tick data ###############################
l1_ex= pd.read_csv('./Data/power_data_for_stefano/data/L1 Germany Baseload EEX Wk01-19.csv')
l2_ex = pd.read_csv('./Data/power_data_for_stefano/data/L2 Germany Baseload EEX Wk01-19.csv')

trades_wke_records = {}
orders_book = {}

os.chdir('./Data/power_data_for_stefano/data')
files = glob('./*.csv')

for f in files:

  if 'TRADES' in f and 'WkEnd' in f:

      print(f)
      df = pd.read_csv(f)
      symbol = f.split('\\')[1]

      week = symbol.split(' ')[5].split('.')
      week = week[2] + '-' + week[1] + '-' + week[0]


      trades_wke_records[week] = df

  elif 'L2' in f and 'WkEnd' in f:

      print(f)
      df = pd.read_csv(f)
      symbol = f.split('\\')[1]

      week = symbol.split(' ')[5].split('.')
      week = week[2] + '-' + week[1] + '-' + week[0]

      orders_book[week] = df


keys = list(orders_book.keys())
ordered_keys = sorted(keys)

trades_wke_sort = {}
book_wke_sort = {}

for k in ordered_keys:

    trades_wke_sort[k] = trades_wke_records[k]
    book_wke_sort[k] = orders_book[k]