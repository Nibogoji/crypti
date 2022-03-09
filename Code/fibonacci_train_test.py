import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
COSTRUISCO DATASET DA MAPPARE AI LIVELLI DI FIBONACCI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/LTC.csv",index_col=0)
df.drop(['trades','Market Cap'],axis = 1,inplace=True)
df.rename(columns = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'},inplace=True)
df.index = pd.to_datetime(df.index)
df= df[~df.index.duplicated(keep='first')]

steps_in_day = 480
lookback_window_size = 50
# train_df = df.iloc[:-steps_in_day*7 - lookback_window_size,:]
# test_df = df.iloc[-steps_in_day*7 - lookback_window_size:,:]

# calcoli a caso

def energy(price, volume,  speed_window = 80, plot = True):


    # price speed euros/minute
    price_dynamics = pd.DataFrame(index=price.index)
    price_dynamics['Inertia'] = price.ewm(span=speed_window).std()**2/price.ewm(span=speed_window).mean()
    price_dynamics['Price speed'] = (price.rolling(window=speed_window).mean().diff(5)/price.rolling(window=speed_window).std()) / 15
    price_dynamics['Price'] = price
    price_dynamics.index = price.index

    # how much time derivative is positive / energia

    price_dynamics['g'] = price_dynamics['Price speed'].diff(5) / 15 # accelerazione media rolling ( in caduta )
    price_dynamics['Volume'] = volume

    price_dynamics['Mass'] = volume.rolling(window=speed_window).mean()/price_dynamics['Price'].rolling(window=speed_window).mean()  # Alti volumi su prezzo basso provoca piu energia quindi voglio che la massa aumenti

    price_dynamics['K'] = 1 / 2 * price_dynamics['Mass'] * price_dynamics['Price speed'] ** 2 * np.sign(price_dynamics['Price speed'])


    price_dynamics.dropna(inplace = True)
    price_energy = pd.DataFrame([price_dynamics['Price'], price_dynamics['K'], price_dynamics['Mass'],price_dynamics['g'],price_dynamics['Price speed'],price_dynamics['Inertia']]).transpose()



    if plot:
        scaler = MinMaxScaler((0, 100))
        price_energy_scaled = scaler.fit_transform(price_energy)
        price_energy_scaled = pd.DataFrame(price_energy_scaled, index=price_energy.index,
                                    columns=['Price', 'K', 'Mass', 'G', 'v', 'Inertia'])
        price_energy_scaled[['Price', 'K', 'Inertia']].plot() # Da plot sembra che quando picchi di inertia e K si avvicinano ce un buon buy


    return price_energy

price_energy = energy(df.Close,df.Volume,speed_window = 80, plot = True)

############################## occhio allo scaling da qui in poi

k_std = price_energy.K.ewm(span=100).std()
k_mean = price_energy.K.ewm(span=100).mean()

inertia_std = price_energy.Inertia.ewm(span=500).std()
inertia_mean = price_energy.Inertia.ewm(span=500).std()

upper_limit_k = k_mean + 2.2 * k_std
lower_limit_k = k_mean - 2.2 * k_std

upper_limit_inertia = 1.5*inertia_mean + 0.8*inertia_std

price_energy[['Inertia']].plot()
plt.plot(upper_limit_inertia, linestyle ='-', color ='purple') # Inertia always positive

# sell_index = [upper_limit_k.index[i] for i in range(len(upper_limit_k)) if (price_energy.K[i] > upper_limit_k[i] and price_energy.Inertia[i] > upper_limit_inertia[i])]
# buy_index = [lower_limit_k.index[i] for i in range(len(lower_limit_k)) if (price_energy.K[i] < lower_limit_k[i] and price_energy.Inertia[i] < lower_limit_inertia[i])]

sell_index = [upper_limit_k.index[i] for i in range(len(upper_limit_k)) if (price_energy.K[i] > upper_limit_k[i] and price_energy.Inertia[i] > upper_limit_inertia[i])] # applicare soglie
buy_index = [lower_limit_k.index[i] for i in range(len(lower_limit_k)) if (price_energy.K[i] < lower_limit_k[i] and price_energy.Inertia[i] > upper_limit_inertia[i])] # applicare soglie

price_energy[['Price']].plot()
# plt.plot(upper_limit_k, linestyle ='-', color ='purple')
# plt.plot(lower_limit_k, linestyle ='-', color ='purple')
plt.scatter(sell_index,price_energy.loc[sell_index, 'Price'], marker = 'v', color = 'red')
plt.scatter(buy_index,price_energy.loc[buy_index, 'Price'], marker = '^', color = 'green')


price_energy['upper K limit 2.2 vola'] = upper_limit_k
price_energy['lower K limit 1.6 vola'] = lower_limit_k
price_energy['upper inertia limit 2.2 vola'] = upper_limit_inertia

price_energy.dropna(inplace=True)
price_energy.to_csv('Code/dati_provvisori/price_energy.csv')


############################## distanza da fibonacci retracement

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""

Macro trend up or down : da rivedere usando le altre cose

"""

custom_sin_wave = pd.read_csv('Code/dati_provvisori/custom_sin.csv', index_col=0)
test_net_trend_adj = pd.read_csv('Code/dati_provvisori/trend_k.csv', index_col=0)
prezzo_test = pd.read_csv('Code/dati_provvisori/prezzo_test.csv', index_col=0)
indicators = pd.read_csv('Code/dati_provvisori/indicatori.csv', index_col=0)

price_energy = pd.read_csv('Code/dati_provvisori/price_energy.csv', index_col=0)
price_energy.index = pd.to_datetime(price_energy.index)

custom_sin_wave.index = pd.to_datetime(custom_sin_wave.index)
test_net_trend_adj.index = pd.to_datetime(test_net_trend_adj.index)
prezzo_test.index = pd.to_datetime(prezzo_test.index)

custom_sin_wave.columns = ['custom sin']
test_net_trend_adj.columns = ['trend k']
prezzo_test.columns = ['prezzo']

first_index = prezzo_test.index[0]
last_index = prezzo_test.index[-1]

custom_sin_wave = custom_sin_wave.loc[first_index:]
test_net_trend_adj = test_net_trend_adj.loc[first_index:]


test_buy = []

attivo = False
for i in range(len(test_net_trend_adj)):

    if test_net_trend_adj.values[i][0] < -0.3:
        attivo = True
    if attivo and -0.16<=test_net_trend_adj.values[i][0]<=-0.15 and custom_sin_wave.values[i][0] > -0.1:
        test_buy.append(test_net_trend_adj.index[i])
        attivo = False

test_sell = [] # sembra essere quando la derivata si inverte nella banda [0.125, 0.175] nel terzo blocco del ciclo
min_sin = False
pausa = False
for i in range(len(test_net_trend_adj)):

    if custom_sin_wave.values[i][0] < -0.1 and not min_sin and not pausa:
        min_sin = True

    if not pausa and min_sin and 0.12 < test_net_trend_adj.values[i][0] < 0.16:

        test_sell.append(test_net_trend_adj.index[i])
        min_sin = False
        pausa = True

    if pausa and custom_sin_wave.values[i][0] > 0 :
        pausa = False


plt.figure()
plt.scatter(test_buy, prezzo_test.loc[test_buy], marker='^', color = 'green')
plt.scatter(test_sell, prezzo_test.loc[test_sell], marker='v', color = 'red')
plt.plot(prezzo_test, label = 'mse')
plt.plot(test_net_trend_adj, linestyle = '--')
plt.plot(custom_sin_wave)
plt.title('Cicli')


up_or_down = pd.DataFrame(index=prezzo_test.index, data = [np.nan for i in range(prezzo_test.shape[0])])
up_or_down.loc[test_buy] = 'up'
up_or_down.loc[test_sell] = 'down'
up_or_down.ffill(inplace=True)
up_or_down.fillna('up', inplace=True)

"""
Provo a comparare i livelli fibonacci a test net trend adj per vedere se la discesa a determinati livelli si collega ad energia k
"""

df = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/LTC.csv",index_col=0)
df = df.reset_index()
df.rename(columns = {'index':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'},inplace=True)
df = df.sort_values('Date')
df.fillna(method = 'ffill',axis = 0, inplace = True)

df.index = df['Date']
df.index = pd.to_datetime(df.index)

df = df.loc[first_index:last_index]
df = df[~df.index.duplicated(keep='first')]

up_or_down['High'] = df.High
up_or_down['Low'] = df.Low

up_or_down['holding'] = df['Market Cap']/df.Volume
up_or_down['trade_size'] = df.Volume/df.trades
up_or_down['trades'] = df.trades

up_or_down.columns = ['trend', 'High', 'Low', 'holding','trade_size', 'trades']
up_or_down['F'] = price_energy.Mass*price_energy.g
up_or_down['Inertia'] = price_energy.Inertia

up_or_down.dropna(inplace= True)

up_or_down[['High', 'trade_size']].plot()

def distance_to_levels(high, low, F, Inertia, trade_size, trades, plot = True, trend = 'up'):

    """
    Series di high e low con datetime indexes

    """

    highest_swing = -1
    lowest_swing = -1
    for i in range(1,high.shape[0]-1):

      if high[i] > high[i-1] and high[i] > high[i+1] and (highest_swing == -1 or high[i] > high[highest_swing]):
        highest_swing = i

      if low[i] < low[i-1] and low[i] < low[i+1] and (lowest_swing == -1 or low[i] < low[lowest_swing]):
        lowest_swing = i

    ratios_up = [0, 0.236, 0.382, 0.5 , 0.618, 0.786, 1]
    ratios_down = [0, 0.236, 0.382, 0.5 , 0.618, 0.786, 1]
    colors = ["black","r","g","b","cyan","magenta","yellow"]
    levels = []
    max_level = high[highest_swing]
    min_level = low[lowest_swing]

    if trend is 'up':
        for ratio in ratios_up:
            # if highest_swing > lowest_swing: # Uptrend
            fibo_level = max_level - (max_level-min_level)*ratio
            correction = F*max_level/min_level * trade_size*3/np.sqrt(trades)
            print(correction)
            print(max_level/min_level)
            levels.append(fibo_level + correction)
    else: # Downtrend
        for ratio in ratios_down:
            fibo_level = min_level + (max_level - min_level) * ratio
            correction = F*max_level/min_level * trade_size*3/np.sqrt(trades)
            levels.append(fibo_level + correction)

    if plot:

        plt.rcParams['figure.figsize'] = [12, 7]
        plt.rc('font', size=14)
        plt.plot((high+low)/2)
        start_date = high.index[min(highest_swing, lowest_swing)]
        end_date = df.index[-100]
        for i in range(len(levels)):
            plt.hlines(levels[i], start_date, end_date, label="{:.1f}%".format(ratios[i] * 100), colors=colors[i],
                       linestyles="dashed")
        plt.legend()
        plt.show()

    return levels

prev_trend = 'up'
prev_trend_start = 0
prev_macro_trend_start = 0
current_counter = 1
fibonacci_levels = []
macro_fibonacci_levels = []

trend_lenght = 7

for i in range(up_or_down.shape[0]):

    if i == 0:
        prev_trend_start = up_or_down.index[i]
        prev_macro_trend_start = up_or_down.index[i]


    trend = up_or_down.iloc[i,0]

    if current_counter % (480*trend_lenght) == 0:

        levels = distance_to_levels(up_or_down.loc[prev_trend_start:up_or_down.index[i],'High'],
                                    up_or_down.loc[prev_trend_start:up_or_down.index[i],'Low'],
                                    up_or_down.loc[up_or_down.index[i], 'F'],
                                    up_or_down.loc[up_or_down.index[i], 'Inertia'],
                                    up_or_down.loc[up_or_down.index[i], 'trade_size'],
                                    up_or_down.loc[up_or_down.index[i], 'trades'],
                                    plot=False, trend = trend)
        fibonacci_levels.append([levels, prev_trend_start,up_or_down.index[i]])
        prev_trend_start = up_or_down.index[i]

    if trend != prev_trend:

        levels = distance_to_levels(up_or_down.loc[prev_macro_trend_start:up_or_down.index[i], 'High'],
                                    up_or_down.loc[prev_macro_trend_start:up_or_down.index[i], 'Low'],
                                    up_or_down.loc[up_or_down.index[i], 'trade_size'],
                                    up_or_down.loc[up_or_down.index[i], 'trades'],
                                    up_or_down.loc[up_or_down.index[i], 'F'],
                                    up_or_down.loc[up_or_down.index[i], 'Inertia'],
                                    plot=False,
                                    trend=trend)
        macro_fibonacci_levels.append([levels, prev_macro_trend_start, up_or_down.index[i]])
        prev_macro_trend_start = up_or_down.index[i]

    prev_trend = trend
    current_counter +=1

mid_price = (up_or_down.High + up_or_down.Low)/2

from sklearn.preprocessing import MinMaxScaler

scaler1= MinMaxScaler((0,100))
scaled_sin = scaler1.fit_transform(custom_sin_wave.values)
scaled_sin = pd.Series(index=custom_sin_wave.index, data=scaled_sin.reshape(-1,))

scaler2 = MinMaxScaler((0,100))
scaled_trend = scaler2.fit_transform(test_net_trend_adj.values)
scaled_trend = pd.Series(index=test_net_trend_adj.index, data=scaled_trend.reshape(-1,))

ratios = [i for i in range(1,8)]
colors = ["black","r","g","b","cyan","magenta","yellow"]

plt.figure()
plt.scatter(test_buy, mid_price.loc[test_buy], marker='^', color = 'green')
plt.scatter(test_sell, mid_price.loc[test_sell], marker='v', color = 'red')
plt.plot(mid_price, label = 'mse')
plt.plot(scaled_trend, linestyle = '--')
plt.plot(scaled_sin)
for i in range(len(fibonacci_levels)):
    if i == len(fibonacci_levels) - 1:
        for j in range(len(fibonacci_levels[i][0])):

            plt.hlines(fibonacci_levels[i][0][j], fibonacci_levels[i][2], mid_price.index[-1],
                       label="{:.1f}%".format(ratios[j] * 100), colors=colors[j],
                       linestyles="dashed")
    else:
        for j in range(len(fibonacci_levels[i][0])):
            plt.hlines(fibonacci_levels[i][0][j], fibonacci_levels[i][2], fibonacci_levels[i+1][2], label="{:.1f}%".format(ratios[j] * 100), colors=colors[j],
                   linestyles="dashed")

for i in range(len(macro_fibonacci_levels)):
    if i == len(macro_fibonacci_levels) - 1:
        for j in range(len(macro_fibonacci_levels[i][0])):

            plt.hlines(macro_fibonacci_levels[i][0][j], macro_fibonacci_levels[i][2], mid_price.index[-1],
                       label="{:.1f}%".format(ratios[j] * 100), colors='brown',
                       linestyles="dashed")
    else:
        for j in range(len(macro_fibonacci_levels[i][0])):

            plt.hlines(macro_fibonacci_levels[i][0][j], macro_fibonacci_levels[i][2], macro_fibonacci_levels[i+1][2], label="{:.1f}%".format(ratios[j] * 100), colors='brown',
                   linestyles="dashed")
plt.title('Cicli')

# Devo poter mappare la probabilita di arrivare ad un determinato livello in base ad altre stats

############################################################   MAPPING TO FIBONACCI LEVELS ############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

price_energy = pd.read_csv('Code/dati_provvisori/price_energy.csv', index_col=0)
price_energy.index = pd.to_datetime(price_energy.index)

custom_sin_wave = pd.read_csv('Code/dati_provvisori/custom_sin.csv', index_col=0)
custom_sin_wave.columns = ['Custom sin wave']
custom_sin_wave.index = pd.to_datetime(custom_sin_wave.index)

test_net_trend_adj = pd.read_csv('Code/dati_provvisori/trend_k.csv', index_col=0)
test_net_trend_adj.columns = ['Net Trend']
test_net_trend_adj.index = pd.to_datetime(test_net_trend_adj.index)

mse_avg = pd.read_csv('Code/dati_provvisori/mse_avg.csv', index_col=0)
mse_avg.columns = ['Mse_avg']
mse_avg.index = pd.to_datetime(mse_avg.index)

price_energy['Custom sin wave'] = custom_sin_wave['Custom sin wave']
price_energy['Net Trend'] = test_net_trend_adj['Net Trend']
price_energy['Mse_avg'] = mse_avg['Mse_avg']

df = price_energy.dropna()

######   cerco il target : drop di prezzo entro delle categorie  . ( O come conto alla rovescia da un drop e piu target
# in base alla profondita del drop oppure categorie di drop in un tempo futuro fisso ( classi sbilanciate ) ) ##########

### Prima provare a prevedere quale e il livello ( 1 o 2 )  piu probabile per poi aggiungere solo quello nel dataset finale---> livello probabile in 7 giorni es.