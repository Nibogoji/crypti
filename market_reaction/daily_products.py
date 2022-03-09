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
l1_ex= pd.read_csv('./Data/power_data_for_stefano/data/L1 Germany Baseload EEX Fri 01.02.19.csv')
l2_ex = pd.read_csv('./Data/power_data_for_stefano/data/L2 Germany Baseload EEX Fri 01.02.19.csv')


trades_d_records = {}
orders_book = {}

os.chdir('./Data/power_data_for_stefano/data')
files = glob('./*.csv')

for f in files:

  if 'TRADES' in f and 'Wk' not in f and 'WkEnd' not in f:

      print(f)
      df = pd.read_csv(f)
      symbol = f.split('\\')[1]

      day = symbol.split(' ')[5].split('.csv')[0]
      week = symbol.split(' ')[4]

      date = datetime.datetime.strptime(day, '%d.%m.%y')

      trades_d_records[date] = df

  elif 'L2' in f and 'Wk' not in f and 'WkEnd'not in f:

      print(f)
      df = pd.read_csv(f)
      symbol = f.split('\\')[1]

      day = symbol.split(' ')[5].split('.csv')[0]
      week = symbol.split(' ')[4]

      date = datetime.datetime.strptime(day, '%d.%m.%y')

      orders_book[date] = df




keys = list(trades_d_records.keys())
ordered_keys = sorted(keys)
trades_d_dict = {}

prices =[]
volumes = []
actions = []
deal_dates = []

for k in ordered_keys:

    trades_d_dict[k] = trades_d_records[k]

    deal_dates.append(trades_d_records[k]['deal_date'])
    prices.append(trades_d_records[k]['price'])
    volumes.append(trades_d_records[k]['volume'])
    actions.append(trades_d_records[k]['aggressor_action'])

deal_dates_series = [t for ds in deal_dates for t in ds]
prices_series = [t for ds in prices for t in ds]
volumes_series = [t for ds in volumes for t in ds]
actions_series = [t for ds in actions for t in ds]

deal_dates_series = np.array(deal_dates_series)
prices_series = np.array(prices_series)
volumes_series = np.array(volumes_series)
actions_series = np.array(actions_series)


prices_buy = np.empty(len(prices_series))
prices_sell = np.empty(len(prices_series))

volume_buy = np.empty(len(volumes_series))
volume_sell = np.empty(len(volumes_series))

prices_buy[:] = np.nan
prices_sell[:] = np.nan

volume_buy[:] = np.nan
volume_sell[:] = np.nan

prices_buy[actions_series == 'Buy'] = prices_series[actions_series == 'Buy']
prices_sell[actions_series == 'Sell'] = prices_series[actions_series == 'Sell']

volume_buy[actions_series == 'Buy'] = volumes_series[actions_series == 'Buy']
volume_sell[actions_series == 'Sell'] = volumes_series[actions_series == 'Sell']

###########################################   days to delivery  ######################################

orders_d_dict = {}
first_trade_to_delivery = []
times_to_delivery_tot = []

months_trades = [[] for i in range(1,13)]
weekdays_trades = [[] for i in range(1,8)]

for k in ordered_keys:

    orders_d_dict[k] = orders_book[k]
    first_trade = datetime.datetime.strptime(orders_d_dict[k]['snapshot_timestamp'].iloc[0], '%Y-%m-%d %H:%M:%S.%f')
    ttd = k-first_trade  # First trade to delivery
    first_trade_to_delivery.append(ttd.total_seconds() / 3600)

    time_delta = list(map(lambda x: (k-datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f')).total_seconds()/3600,
                                          orders_d_dict[k]['snapshot_timestamp'].values))

    times_to_delivery_tot.append(time_delta)

    for m in range(1,13):
        if first_trade.month == m:

            months_trades[m-1].append(time_delta)

    for i in orders_d_dict[k]['snapshot_timestamp'].values:
        for d in range(1,8):
            if datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S.%f').weekday() == d:
                weekdays_trades[d-1].append((k-datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S.%f')).total_seconds()/3600)


times_to_delivery_m_series = []

for i in range(len(months_trades)):

    times_to_delivery_m_series.append([t for ds in months_trades[i] for t in ds])

times_to_delivery_series = [t for ds in times_to_delivery_tot for t in ds]




################### complete dataframe ###########################


first_trade_to_delivery = pd.DataFrame(data=first_trade_to_delivery,
                                       index=[datetime.datetime.strftime(i, '%Y-%m-%d %H:%M:%S.%f') for i in ordered_keys],
                                       columns=['time to delivery'])

data_price_vol = {'price buy':prices_buy,
                  'price sell': prices_sell,
                  'volume buy': volume_buy,
                  'volume sell': volume_sell}

price_vol = pd.DataFrame(index=deal_dates_series,data=data_price_vol)

dates_list = list(deal_dates_series)+[i for i in list(first_trade_to_delivery.index)]
dates_list.sort()
complete_df = pd.DataFrame(index=dates_list)
complete_df = complete_df.join(price_vol)
complete_df = complete_df.join(first_trade_to_delivery)

complete_df.to_csv('C:\\Users\\S49399\\.PyCharmCE2019.2\\projects\\merket_reaction\\Data\\power_data_for_stefano/spot_trading.csv')
price_vol.to_csv('C:\\Users\\S49399\\.PyCharmCE2019.2\\projects\\merket_reaction\\Data\\power_data_for_stefano/price_volumes.csv')

complete_df = pd.read_csv('C:\\Users\\S49399\\.PyCharmCE2019.2\\projects\\merket_reaction\\Data\\power_data_for_stefano/spot_trading.csv',index_col=[0])
price_vol = pd.read_csv('C:\\Users\\S49399\\.PyCharmCE2019.2\\projects\\merket_reaction\\Data\\power_data_for_stefano/price_volumes.csv',index_col=[0])

price_d_only = []
 # prezzi di prodotti giornalieri misti ordinati per trade timestamp
a =  price_vol[['price buy']].values
b = price_vol[['price sell']].values
c = zip(a,b)
for i,j in c:
    z = np.nan
    if i != np.nan:
        z = i[0]
    elif j != np.nan:
        z = j[0]
    else:
        pass

    price_d_only.append(z)

price_d_only = pd.DataFrame(index=pd.to_datetime(price_vol.index), data=price_d_only,columns=['Price daily product'])

ohlc = price_d_only.resample('1D').ohlc()

ohlc_columns = ['Open','High','Low','Close']

ohlc.columns = ohlc_columns
# each trade is composed of: [time, price, quantity]



#################### PLOTS ####################

complete_df = complete_df.iloc[:500]

data_dict_plot = {'ax1': {'0': {'data': complete_df['volume buy'],
                              'label': 'volume buy',
                              'color': 'orange'},
                       '1': {'data': complete_df['volume sell'],
                              'label': 'volume sell',
                              'color': 'blue'}},
                'ax2':{'0': {'data': complete_df['price buy'],
                              'label': 'price buy',
                              'color': 'orange'},
                       '1': {'data': complete_df['price sell'],
                              'label': 'price sell',
                              'color': 'blue'}},
                'ax3':{'0': {'data': complete_df['time to delivery'],
                              'label': 'time to delivery',
                              'color': 'purple'}}
}

plotOrbar = {'ax1': {'0': 'plot',
                       '1': 'plot'},
                'ax2': {'0': 'plot',
                       '1': 'plot'},
                'ax3': {'0': 'bar'}
                }
subplot_slots, subplot_start, share_ax = [1,2,1], [0,1,3], [False, False, 'ax3']


plt2grid(data_dict_plot, subplot_slots, subplot_start, share_ax, plotOrbar, fig_n = 1000, show = True)


      ################  DISTRIBUTION PLOTS ####################

# times_to_delivery_series = pd.DataFrame(times_to_delivery_series)
# times_to_delivery_series.hist(bins=500)

# for i in range(len(times_to_delivery_m_series)):
#
#     month_distribution = pd.DataFrame(times_to_delivery_m_series[i])
#
#     plt.figure(i+1)
#     month_distribution.hist(bins=500)
#
#
# for i in range(6):
#
#     weekday_distribution = pd.DataFrame(weekdays_trades[i])
#     plt.figure(i)
#     weekday_distribution.hist(bins= 500)

