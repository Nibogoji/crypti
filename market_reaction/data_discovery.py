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


#################################################   DATA WATTSIGHT  ####################################

os.chdir('C:\\Users\\S49399\\.PyCharmCE2019.2\\projects\\merket_reaction')

search_inputs = {'commodity':['POW'],
                'category':['WND','SPV','TT,CON','CON','CON,WINDCHILL','RES,HYDRO,WTR','RRE','TT,HYDRO'],
                'units':['MWh/h','°C','%','GWh'],
                'source':['EC12'],
                'area' :['DE'],
                'data_type':['F']
                 }
credentials = (os.environ["client_id"], os.environ["client_secret"])

curves_h_issue = get_curves_wattsight(search_inputs,credentials)

print([c.name for c in curves_h_issue])

actual_price_input = {'commodity':['POW'],
                'category':['PRI,SPOT'],
                'units':['€/MWh'],
                'source':[],
                'area' :['DE'],
                'data_type':['A'],
                 }

curves_price = get_curves_wattsight(actual_price_input,credentials)
print([c.name for c in curves_price])

# spot_actual_price_curve = list([c for c in curves_price][1])

########################################### Set dates of the timeseries or issue dates for instances

issue_date = pytz.timezone('CET').localize(datetime.datetime(2018, 1, 1))
date_from = issue_date + datetime.timedelta(minutes=15)
date_to = pytz.timezone('CET').localize(datetime.datetime(2020, 11, 1)) + datetime.timedelta(hours=24)

final_issue_date = pytz.timezone('CET').localize(datetime.datetime(2020, 11, 1))
dates_issued = pd.date_range(issue_date, final_issue_date,freq = '1d').to_list()

############################################ get data from wattsight
data = None
#data = get_data_wattsight(curves_h_issue, dates_issued, date_from = None, date_to = None)

spot_price = get_data_wattsight(curves_price, dates_issued, date_from = date_from, date_to = date_to)

if not os.path.exists('./Data/Wattsight data'):

    os.makedirs('./Data/Wattsight data')

if data != None:

    with open('./Data/Wattsight data/Wattsight weather data.b', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:

    with open('./Data/Wattsight data/Wattsight weather data.b', 'rb') as fp:
        data_from_pickle = pickle.load(fp)
####################################################################
data = data_from_pickle
t = list(data[curves_h_issue[0].name].keys())[0]
# diagonal_df = pd.DataFrame(index=pd.date_range(issue_date, final_issue_date,freq = '15min').to_list())
#
# for k in list(data[curves_h_issue[0].name].keys()):
#
#     df_data= pd.DataFrame(data = data[curves_h_issue[0].name][k].values, index= data[curves_h_issue[0].name][k].index ,columns = [k])
#
# diagonal_df = diagonal_df.join(df_data)
trading = pd.read_csv('./Data/power_data_for_stefano/spot_trading.csv', index_col=[0])
actual_price = spot_price['pri de spot €/mwh cet min15 a']

actual_price = pd.DataFrame(data=actual_price.values,
                           index=[datetime.datetime.strftime(i, '%Y-%m-%d %H:%M:%S.%f') for i in actual_price.index],
                           columns=['actual spot price'])

common_index = list(trading.index)+list(actual_price.index)
common_index.sort()

spot_market = pd.DataFrame(index=common_index)
spot_market = spot_market.join(trading)
spot_market = spot_market.join(actual_price)

spot_market = spot_market.loc[trading.index[0]:]

filled_price = spot_market['actual spot price'].fillna(method='ffill')
spot_market['actual spot price'] = filled_price

########### PLOT ACTUAL ON TRADING ################

spot_market = spot_market.iloc[:2000]

from matplotlib import ticker
import  matplotlib.pyplot as plt

class CustomLocator(ticker.MaxNLocator):
    def ticks_values(self, vmin, vmax):
        if len(np.arange(vmin,vmax,1)) > 20:
            tcks = list(np.linspace(vmin,vmax,num=20))
        else:
            tcks = list(np.linspace(vmin,vmax,num=len(np.arange(vmin,vmax,1))))

        return tcks

ML = CustomLocator()

fig = plt.figure(1000)

ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1)
ax1 = plt.subplot2grid((3,1), (0,0),  rowspan=2, colspan=1, sharex=ax2)
# ax3 = plt.subplot2grid((5,1), (0,0),  rowspan=2, colspan=1, sharex=ax2)

ax1.plot(list(spot_market.index.astype(str)),spot_market['price buy'], label = 'buy', color = 'orange')
ax1.plot(list(spot_market.index.astype(str)),spot_market['price sell'], label = 'Sell',color = 'blue')
ax1.plot(list(spot_market.index.astype(str)),spot_market['actual spot price'], label = 'Actual Price',color = 'purple')

ax2.bar(list(spot_market.index.astype(str)),spot_market['volume buy'], label = 'Volume buy',color = 'orange')
ax2.bar(list(spot_market.index.astype(str)),spot_market['volume sell'], label = 'Volume sell',color = 'blue')

# ax3.plot(list(spot_market.index.astype(str)),diagonal_df, label = 'Wind ENS',color = 'gray')

ax1.xaxis.set_visible(False)
# ax3.xaxis.set_visible(False)

ax2.xaxis.set_major_locator(ML)

fig.autofmt_xdate()

ax1.legend()
ax2.legend()

fig.show()

pickle.dump(fig, open('./Data/plots/Ticks.fig.pickle', 'wb'))

# figx = pickle.load(open('Ticks.fig.pickle', 'rb'))
#
# figx.show()