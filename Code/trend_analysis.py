import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mplf
from numpy import linalg as la
import yfinance
from collections import deque

"""
https://github.com/matplotlib/mplfinance/blob/master/examples/using_lines.ipynb

"""

data = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/cryptos_25_02_2021.csv", index_col=0)
data.set_index(pd.to_datetime(data.index))

df = data.interpolate()

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

# 1 giorno = 480 steps da 3 min... prevedere 1 giorno data ultima settimana: 3360/2880
step = 5
# df = close_df[['close']].iloc[-12000:]
df = close_df[['close']].iloc[3360*step:3360*step + 3360]
df = df.reset_index()

df_test = df
df = df.iloc[:2880]


# Find local minima
from scipy.signal import argrelextrema

n = 50

df['min'] = df.iloc[argrelextrema(df.close.values, np.less_equal, order=n)[0]]['close']
df['max'] = df.iloc[argrelextrema(df.close.values, np.greater_equal, order=n)[0]]['close']

plt.figure()
df['close'].plot()
plt.scatter(df['min'].index,df['min'],color = 'r', marker='*')
plt.scatter(df['max'].index,df['max'],color = 'green', marker='+')

dfMax = df[df['max'].notnull()]
dfMin = df[df['min'].notnull()]

to_cluster = np.asarray([list(dfMin.index),dfMin['min']]).T

def abline(slope, intercept,xmin,xmax,label):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.linspace(xmin,xmax,xmax-xmin)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--',label = label)

def trend_support_get(to_cluster, price, full_price, min_max = 'min',
                      cluster_quantile = 0.1, std_w = 2.5, angle_w = 0.92,
                      std_w2 = 0.5,min_for_support = 3, w_d = 35, under_over_limit = 0.05):

    # Compute clusters for later definition of horizontal supports
    from sklearn.cluster import MeanShift, estimate_bandwidth

    bandwidth = estimate_bandwidth(to_cluster,quantile=cluster_quantile)
    clustering = MeanShift(bandwidth=bandwidth, cluster_all = False).fit(to_cluster)
    labels = clustering.labels_
    centers = clustering.cluster_centers_

    #Compute standard deviation from stationary data
    price_change = df.close.diff(1)
    std = np.std(price_change)

    # #### Generare fasci di rette per ogni minimo locale

    from itertools import combinations

    combos = [i for i in to_cluster]
    combos = [i for i in combinations(combos,2)]

    rette_centrali = []
    rette_up = []
    rette_down = []

    # std_w = 3 # Aumentare per accettare piu rette

    for c in combos:

        r_central_coeffs =  [(c[0][1]-c[1][1])/(c[0][0]-c[1][0]),(c[0][0]*c[1][1]-c[1][0]*c[0][1])/(c[0][0]-c[1][0])]
        down_up_interc = [r_central_coeffs[1] - std*std_w, r_central_coeffs[1] + std*std_w]
        r_down_coeff = [r_central_coeffs[0],down_up_interc[0]]
        r_up_coeff = [r_central_coeffs[0], down_up_interc[1]]

        rette_centrali.append(r_central_coeffs)
        rette_down.append(r_down_coeff)
        rette_up.append(r_up_coeff)

    ### Se almeno 3 centroidi appartengono al fascio, e non ci sono gia rette simili salva la retta

    rette_significative = {}
    supporti = []
    unique_interecepts = [0]
    unique_angles = [0]

    # diminuire per accettare piu rette
    # angle_w = 0.92
    # std_w2 = 0.5

    for r in range(len(rette_centrali)):

        angle = rette_centrali[r][0]
        intercept = rette_centrali[r][1]

        if (all( i + i * angle_w > angle for i in np.asarray(unique_angles))
                or all( i - i * angle_w < angle for i in np.asarray(unique_angles))) and \
                (all(i+std*std_w2 > intercept for i in np.asarray(unique_interecepts))
            or all(i-std*std_w2 < intercept for i in np.asarray(unique_interecepts))):

            # unique_interecepts.append(intercept)
            unique_angles.append(angle)
            continue

        counter = 0
        supports = []
        support_alive = False

        rette_significative[r] = {}

        for c in to_cluster:

            position_to_up = np.sign(c[1]-(rette_up[r][0]*c[0]+rette_up[r][1]))
            position_to_down = np.sign(c[1]-(rette_down[r][0]*c[0]+rette_down[r][1]))

            in_or_out = position_to_down*position_to_up # se negativo sta dentro il fascio

            if in_or_out<=0:

                counter+=1
                supports.append(c)

                if counter >= min_for_support: # Numero di minimi necessari per qualificare un supporto
                    support_alive = True

                if support_alive:

                    rette_significative[r][counter] = supports
            else:

                support_alive = False

        if not rette_significative[r]:

            rette_significative.pop(r)
        else:

            k = list(rette_significative[r].keys())
            sup = sorted(rette_significative[r][k[-1]], key=lambda x: x[0])
            supp = [sup[0][0], sup[-1][0]]

            segment = price['close'].loc[int(supp[0]):int(supp[1])]

            xmin = int(supp[0])
            xmax = int(supp[1])
            x_vals = np.linspace(xmin, xmax, xmax - xmin +1)
            y_vals = rette_centrali[r][1] + rette_centrali[r][0] * x_vals

            r_over = np.sum(y_vals > segment)

            if min_max == 'min':

                if r_over > under_over_limit * len(y_vals): # Escludere rette sotto la curva prezzo per piu del 5%

                    rette_significative.pop(r)

            if min_max == 'max':

                if r_over < under_over_limit * len(y_vals):  # Esscludere rette sopra la curva prezzo per piu del 5%

                    rette_significative.pop(r)




    from scipy.spatial.distance import directed_hausdorff

    accepted_sustains = [[np.asarray([0.0,0.0]),np.asarray([0.0,0.0])]]
    rette_buone = []

    # w_d = 35 # Aumentare per accettare meno rette

    for r in rette_significative:

        k = list(rette_significative[r].keys())
        sup = sorted(rette_significative[r][k[-1]], key=lambda x: x[0])
        supp = [sup[0],sup[-1]]

        if all(directed_hausdorff(supp,i)[0] > std*w_d for i in np.asarray(accepted_sustains,dtype=object)): # distanza di hausdorff

            if all((sup[0] != i[0]).all() for i in np.asarray(accepted_sustains,dtype=object)): # non iniziano nello stesso punto

                if all((sup[-1] != i[1]).all() for i in np.asarray(accepted_sustains, dtype=object)): # non finiscono nello stesso punto

                    accepted_sustains.append(supp)
                    rette_buone.append(r)

    plt.figure()
    full_price['close'].plot()
    plt.scatter(price['min'].index,price['min'],color = 'r', marker='*')
    plt.scatter(price['max'].index,price['max'],color = 'green', marker='+')
    plt.scatter(centers[:,0],centers[:,1],color = 'orange',marker = 's')

    for r in rette_buone:
        print('plotting :',r)
        k = list(rette_significative[r].keys())
        sup = sorted(rette_significative[r][k[-1]], key=lambda x: x[0])


        # abline(rette_down[r][0],rette_down[r][1],int(sup[0][0])-10,int(sup[-1][0])+10,'red')
        # abline(rette_up[r][0],rette_up[r][1],int(sup[0][0])-10,int(sup[-1][0])+10,'green')
        abline(rette_centrali[r][0],rette_centrali[r][1],int(sup[0][0])-1,int(full_price['close'].index[-1]),label='Trend '.format(r))

    centers = sorted(centers, key=lambda x: x[0])

    for i,c in enumerate(centers):

        abline(0, c[1], int(c[0]), int(full_price['close'].index[-1]), label='Sostegno {}'.format(i))


    plt.legend()

    print('Unique Intercepts ',len(unique_interecepts)-1)
    print('Unique Angles ',len(unique_angles)-1)
    print('Rette sostegni ',len(rette_significative))


trend_support_get(to_cluster, df, df_test)



######################################################### ENERGY #######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

df = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/LTC.csv",index_col=0)

df.drop(['trades','Market Cap'],axis = 1,inplace=True)
df.rename(columns = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'},inplace=True)
df.index = pd.to_datetime(df.index)
df= df[~df.index.duplicated(keep='first')]

steps_in_day = 480
lookback_window_size = 50
train_df = df.iloc[:-steps_in_day*7 - lookback_window_size,:]
test_df = df.iloc[-steps_in_day*7 - lookback_window_size:]

# calcoli a caso

test_df = test_df.reset_index()
rolling_vola = []

for i in range(test_df.shape[0]-99):
    vola = np.std(test_df.loc[i:i+99, 'Close'])
    rolling_vola.append(vola)

plt.figure()
plt.plot(rolling_vola)

plt.figure()
plt.hist(test_df['Close'].diff(1), bins = 100, label = 'Daily price change')
plt.hist(rolling_vola, bins  = 100, label = '100 Rolling vola')
plt.legend()

# fill_roll_vola = [rolling_vola[0] for i in range(100)]
# rolling_vola_tot = fill_roll_vola + rolling_vola
rolling_vola_tot = rolling_vola

rolling_vola_tot = np.log(np.asarray(rolling_vola_tot).reshape(-1,))
price_vola = pd.DataFrame([rolling_vola_tot, test_df['Close'][99:]]).transpose()


price_vola = pd.DataFrame(price_vola.values, columns = ['Vola','Price'])

# price speed euros/minute

price_vola['Price speed'] = price_vola['Price'].rolling(window=160).mean().diff(5)/15

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 100))
price_speed = scaler.fit_transform(price_vola)
price_speed = pd.DataFrame(price_speed, columns=price_vola.columns)
price_speed.plot()

# how much time derivative is positive / energia

K = []
U = []
v = test_df['Close'].rolling(window=80).mean().diff(5)/15 # velocita euro al minuto
g = v.diff(5)/15 # accelerazione media rolling ( in caduta )
for i in range(test_df.shape[0]-89):

    m = np.mean(test_df.loc[i:i+89, 'Close'])/np.mean(test_df.loc[i:i+89, 'Volume'])#Volume weighted price
    k = 1/2*m*v[i+89]**2*np.sign(v[i+89])
    K.append(k)

    # u = m*g[i + 89]*h
    # U.append(u)

K = np.asarray(K)

price = test_df.loc[89:,'Close'].reset_index(drop=True)
price_energy = pd.DataFrame([price,K]).transpose()

scaler = MinMaxScaler()
price_energy = scaler.fit_transform(price_energy)
# price_energy = pd.DataFrame(price_energy)
plt.figure()
plt.plot(price_energy[:,0],label = 'Price')
plt.plot(price_energy[:,1],label = 'Ek')
plt.legend()


# Segnale di energia cinetica con segno sempra molto buono per decidere quando vendere. Limite chiaro e stabile
# Forse energia potenziale puo dire quando comprare? Servono fibonacci resistances per stimare di quanto puo cadere, e per le fibo servono
# i trend

# TREND DETECTION E DURATA PROBABILE


df = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/LTC.csv",index_col=0)
df.drop(['trades','Market Cap'],axis = 1,inplace=True)

prezzo = df.close
prezzo= prezzo[~prezzo.index.duplicated(keep='first')]

prezzo = prezzo.iloc[60000:]

# prezzo = prezzo.reset_index(drop= True)
ms_prezzo = prezzo.rolling(window=240).std()
ms_prezzo.index = pd.to_datetime(ms_prezzo.index)
ms_prezzo.dropna(inplace = True)

ma_prezzo = prezzo.rolling(window=240).mean()# 12 ore MA
ma_prezzo.index = pd.to_datetime(ma_prezzo.index)

# plt.figure()
# plt.plot(prezzo)
# plt.plot(ma_prezzo)

ma_prezzo.dropna(inplace = True)
# ma_prezzo = ma_prezzo.reset_index(drop=True)
# calcolo derivate

x0 = 0
x1 = 0
x2 = 0

y1 = 0
y2 = 0
y3 = 0

derivate = []
neg_d = deque(maxlen=100)
pos_d = deque(maxlen=100)

durata_trend_neg = 0
durata_trend_pos = 0

for i in range(ma_prezzo.shape[0]-1):

    x0 = i
    x1 = i+1

    y0 = ma_prezzo.iloc[i]
    y1 = ma_prezzo.iloc[i+1]

    d1 = (y1-y0)/(x1-x0)

    derivate.append((d1, ma_prezzo.index[i+1]))

mean_pos_time = []
mean_neg_time = []

timestap_end_pos_trend = []
timestap_end_neg_trend = []
crossing_up_down = []
crossing_down_up = []


for i in range(len(derivate)-1): # da aggiungere un indicazione su timestamps per riuscire a ricondurre gli incroci al prezzo

    segno_d1 = np.sign(derivate[i][0])
    segno_d2 = np.sign(derivate[i+1][0])

    negative = False
    positive = False

    if segno_d2 < 0:
        negative = True
    else:
        positive = True


    if negative == True:

        if segno_d1 > 0:

            pos_d.append(durata_trend_pos)
            timestap_end_pos_trend.append(derivate[i+1][1])

        durata_trend_pos = 0
        durata_trend_neg += 3 # in minuti

    elif positive == True:

        if segno_d1 < 0:
            neg_d.append(durata_trend_neg)
            timestap_end_neg_trend.append(derivate[i + 1][1])
        durata_trend_neg = 0
        durata_trend_pos += 3

    if pos_d:
        mean_pos_time.append((np.mean(pos_d),derivate[i + 1][1]))

        try:

            if mean_pos_time[-2]<= mean_neg_time[-2] and mean_pos_time[-1] > mean_neg_time[-1]:

                crossing_down_up.append(derivate[i + 1][1])

        except:
            pass

    if neg_d:
        mean_neg_time.append((np.mean(neg_d),derivate[i + 1][1]))
        try:
            if mean_pos_time[-2]>= mean_neg_time[-2] and mean_pos_time[-1] < mean_neg_time[-1]:
                crossing_up_down.append(derivate[i + 1][1])
        except:
            pass

# if len(mean_pos_time) == len(mean_neg_time):
#     trend_alarms = []
#
#     for i in range(len(mean_pos_time)-1):
#
#         if mean_neg_time[i+1] >= mean_pos_time[i+1] and  mean_neg_time[i] < mean_pos_time[i]:
#
#             trend_alarms.append(np.nan)

print(len(mean_pos_time))
print(len(mean_neg_time))
print(len(derivate))

plt.figure() # Indica come la durata media di trend positivi e negativi cambi in caso di bear / bull market

plt.plot([i[1] for i in mean_neg_time],[i[0] for i in mean_neg_time], label = 'neg')
plt.plot([i[1] for i in mean_pos_time[8:]],[i[0] for i in mean_pos_time[8:]], label = 'pos')
plt.plot(ma_prezzo, linestyle = '-')
plt.scatter(crossing_up_down, ma_prezzo[crossing_up_down], label = 'Downtrend begins', marker="v", color = 'red')
plt.scatter(crossing_down_up, ma_prezzo[crossing_down_up], label = 'Uptrend begins', marker="^", color = 'green')
plt.legend()

import matplotlib.dates as mdates

# fig, ax = plt.subplots()
# plt.setp( ax.xaxis.get_majorticklabels(), rotation=45)
# plt.plot(ma_prezzo.iloc[:-1].index, ma_prezzo.iloc[:-1].values)
# plt.scatter(crossing_up_down, ma_prezzo[crossing_up_down], label = 'Downtrend begins', marker="v", color = 'red')
# plt.scatter(crossing_down_up, ma_prezzo[crossing_down_up], label = 'Uptrend begins', marker="^", color = 'green')
# plt.legend()
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H')) # non male a percepire un uptrend in arrivo, ma vede i down troppo tardi

# PROVO A CALCOLARE LA DISTANZA TRA MEAN_NEG_TREND A MEAN_POS_TREND
print(len(mean_neg_time))
print(len(mean_pos_time))


net_trend = np.asarray([i[0] for i in mean_pos_time[1:]]) - np.asarray([i[0] for i in mean_neg_time])
# net_trend = pd.Series(net_trend, [t[1] for t in mean_neg_time])
net_trend = net_trend[20:]

# Standardize between 0-100
from sklearn.preprocessing import MinMaxScaler

net_trend_scaler = MinMaxScaler((0,100))
net_trend = net_trend_scaler.fit_transform(net_trend.reshape(-1,1))
net_trend = pd.Series(net_trend.reshape(-1,), [t[1] for t in mean_neg_time[20:]])

plt.figure() # Indica come la durata media di trend positivi e negativi cambi in caso di bear / bull market

plt.plot(net_trend)
plt.plot(ma_prezzo, linestyle = '-')
plt.scatter(crossing_up_down, ma_prezzo[crossing_up_down], label = 'Downtrend begins', marker="v", color = 'red')
plt.scatter(crossing_down_up, ma_prezzo[crossing_down_up], label = 'Uptrend begins', marker="^", color = 'green')
plt.legend()


class Soglia: # Quando viene colpito il trigger, l'oggetto soglia viene creato per i seguenti 1000 steps

    def __init__(self, data , tipo = 'diagonale', trigger_step =  0, side = 'sup', trigger_up = 65, trigger_down = 38, lunghezza_soglia = 50000, fitting_size = 1000):

        self.tipo = tipo
        self.data = data
        self.side = side
        self.fitting_size = fitting_size
        self.lunghezza_soglia = lunghezza_soglia
        self.soglia = np.zeros(lunghezza_soglia)
        self.trigger_up = trigger_up
        self.trigger_down = trigger_down
        self.crossed = False
        self.coeff_ang = 0
        self.trigger_step = trigger_step

    def fit(self): # Volendo sarebbe da ottimizzare fit size e coeff_ang

        if self.side == 'sup':

            if self.tipo == 'diagonale':

                self.coeff_ang = (self.data[-1] - self.data[- self.fitting_size]) / self.fitting_size

            elif self.tipo == 'orizzontale':

                self.coeff_ang = 0

            for i in range(self.lunghezza_soglia):

                self.soglia[i] = self.trigger_up + i * self.coeff_ang
                # print('-----------------'+str(self.soglia[i]))

        elif self.side == 'inf':

            if self.tipo == 'diagonale':

                self.coeff_ang = (self.data[-1] - self.data[- self.fitting_size]) / self.fitting_size

            elif self.tipo == 'orizzontale':

                self.coeff_ang = 0

            for i in range(self.lunghezza_soglia):

                self.soglia[i] = self.trigger_down + i * self.coeff_ang

    def check_for_crossing(self, current_price, current_step):

        current_step = current_step - self.trigger_step
        print(current_step)
        print(self.trigger_step)
        if current_step > self.lunghezza_soglia:

            print('Soglia troppo corta')

        if self.side == 'sup':

            if current_price < self.trigger_up:
                self.crossed = True
            else:
                self.crossed = False
        elif self.side == 'inf':
            if current_price > self.trigger_down:
                self.crossed = True
            else:
                self.crossed = False

        return self.crossed

    def mostra_soglia(self):

        return self.soglia


#
# plt.figure()
# plt.plot(net_trend[38000:100000])

net_trend_10 = net_trend/10

# Sarebbe utile una trasformazione che avvicini tutti i minimi ew massimi allo stesso livello ( log quando tende a x = inf)
# Cerco il range per il log in modo che x e x+20 tendano a valori simili

# x  = [i for i in range(10)]
# plt.figure()
# plt.plot(np.log(x))
#
# plt.plot(np.tanh([i-5 for i in x])+1, label = 'tanh')
import math
#
def fft_smoothing(df, armonics, plot = True):

    fft_df = df.values
    close_fft = np.fft.fft(fft_df.tolist())
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())

    fft_feature = pd.DataFrame()

    for num_ in armonics:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        inv_fft = np.fft.ifft(fft_list_m10)
        fft_feature[str(num_) + ' Armonics'] = np.real(inv_fft)

    if plot:
        plt.figure()
        plt.plot(df.values,label = 'Original')
        for a in armonics:
            plt.plot(fft_feature[str(a) +' Armonics'].values,label = 'Smoothed {} Armonics'.format(a))
        plt.legend()

    return fft_feature


def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig
# plt.plot([sigmoid(i-5) for i in x] , label = 'sigmoid')
# plt.legend()

net_trend_tanh = np.array([sigmoid(i-5) for i in net_trend_10])
# net_trend_tanh  = net_trend_tanh[80000:90000]

smooth_net_trend_tanh = fft_smoothing( pd.Series(net_trend_tanh, index=net_trend_10.index),[20,50,80])
plt.figure()
plt.plot(smooth_net_trend_tanh['80 Armonics'])
plt.plot()

test_net_trend = pd.Series(smooth_net_trend_tanh['80 Armonics'].values, index=net_trend_10.index)


def crossing_points(data,fitting_size = 1500, up_trigger = 0.8, up_normal = 0.7, down_trigger = 0.2, down_normal = 0.4, side = 'sup', plot = True):

    timesteps = data.index[fitting_size:]
    data = data.values
    default_up_trigger = up_trigger

    if plot:
        plt.figure()
        plt.plot(data[fitting_size:])

    trigger_point_up = []
    crossing_points_up = []
    trigger_point_down = []
    crossing_points_down = []
    triggered = False
    already_crossed = False
    crossed = False
    back_to_neutral = True

    for i, v in enumerate(data[fitting_size:].reshape(-1,)):
        up_trigger = default_up_trigger
        timestep = timesteps[i]
        volatility = ms_prezzo[timestep]

        i += fitting_size
        if side == 'sup':

            if volatility < 0.04 *ma_prezzo[timestep]:
                up_trigger = up_trigger
            else:
                up_trigger = up_trigger
                # up_trigger = up_trigger - 0.25
            print('Up trigger :',up_trigger)


            if v >= up_trigger and triggered == False and back_to_neutral is True:

                print('Retraining ',i)

                soglia = Soglia(data[i-fitting_size:i], trigger_step = i,lunghezza_soglia=10000, fitting_size=fitting_size, side=side,trigger_up = up_trigger, trigger_down=down_trigger)
                soglia.fit()
                retta = soglia.mostra_soglia()
                trigger_point_up.append(i)

                triggered = True
                back_to_neutral = False
                print(soglia.coeff_ang)
        elif side == 'inf':

            if volatility < 0.04 *ma_prezzo[timestep]:
                down_trigger = down_trigger + 0.00
            else:
                down_trigger = down_trigger - 0.00
            print(i)
            print(v)
            if v <= down_trigger and triggered == False and back_to_neutral is True:
                print('Retraining {}'.format(side), i)

                soglia = Soglia(data[i - fitting_size:i], trigger_step=i, lunghezza_soglia=10000,
                                fitting_size=fitting_size, side=side, trigger_up=up_trigger, trigger_down=down_trigger)
                soglia.fit()
                retta = soglia.mostra_soglia()
                trigger_point_down.append(i)

                triggered = True
                back_to_neutral = False
                print(soglia.coeff_ang)

        if side == 'sup':

            if triggered and already_crossed is False and soglia.coeff_ang >= 0 and back_to_neutral is False: # usare soglia.coeff_ang[0] se da problemi
                print('Checking for crossing'+side)
                crossed = soglia.check_for_crossing(v, i)
                print(crossed)
            else:
                triggered = False
                back_to_neutral = True

        elif side == 'inf':

            if triggered and already_crossed is False and soglia.coeff_ang <= 0 and back_to_neutral is False: # usare soglia.coeff_ang[0] se da problemi
                print('Checking for crossing'+ side)
                crossed = soglia.check_for_crossing(v, i)
                print(crossed)
            else:
                triggered = False
                back_to_neutral = True

        if crossed and already_crossed is False and  back_to_neutral is False :
            # print(already_crossed)
            if side == 'sup':
                crossing_points_up.append((i,v))
                diff = trigger_point_up[-1] - fitting_size
            elif side == 'inf':
                crossing_points_down.append((i, v))
                diff = trigger_point_down[-1] - fitting_size

            if side == 'sup':
                if plot:

                    plt.plot([i for i in range(trigger_point_up[-1] - fitting_size, len(retta) + diff)], retta)
                    plt.scatter(crossing_points_up[-1][0] - fitting_size, crossing_points_up[-1][1])
                    plt.scatter(trigger_point_up[-1] - fitting_size, up_trigger, marker='+')

            elif side == 'inf':

                if plot:
                    plt.plot([i for i in range(trigger_point_down[-1] - fitting_size, len(retta) + diff)], retta)
                    plt.scatter(crossing_points_down[-1][0] - fitting_size, crossing_points_down[-1][1])
                    plt.scatter(trigger_point_down[-1] - fitting_size, down_trigger, marker='+')

            already_crossed = True
            print('Crossed ', i)

        if side == 'sup':

            if already_crossed and  v < up_normal:
                print('back to normal')
                triggered = False
                already_crossed = False
                back_to_neutral = True

        elif side == 'inf':

            if already_crossed and  v > down_normal:
                print('back to normal')
                triggered = False
                already_crossed = False
                back_to_neutral = True

    return trigger_point_up ,crossing_points_up , trigger_point_down , crossing_points_down

trigger_point_up ,crossing_points_up , _,_ = crossing_points(test_net_trend, up_trigger = 0.9, up_normal = 0.6, side='sup')
_ ,_ , trigger_point_down , crossing_points_down = crossing_points(test_net_trend, down_trigger = 0.2, down_normal = 0.4, side='inf')

crossing_up_times = net_trend_10.index[[i[0] for i in crossing_points_up]]
crossing_down_times = net_trend_10.index[[i[0] for i in crossing_points_down]]

plt.figure()
plt.plot(ms_prezzo.loc[net_trend_10.index[1500:]]/ma_prezzo.loc[net_trend_10.index[1500:]])
plt.title('Unscaled Volatility %')

from sklearn.preprocessing import MinMaxScaler

scaler2 = MinMaxScaler()
ma_prezzo_scaled = scaler2.fit_transform(ma_prezzo.values.reshape(-1,1),(0,100))
ma_prezzo_scaled = pd.DataFrame(ma_prezzo_scaled,index=ma_prezzo.index)
scaler3 = MinMaxScaler()
ms_prezzo_scaled = scaler3.fit_transform(ms_prezzo.values.reshape(-1,1),(0,100))
ms_prezzo_scaled = pd.DataFrame(ms_prezzo_scaled,index=ms_prezzo.index)

plt.figure()
plt.plot(ma_prezzo_scaled.loc[net_trend_10.index[1500:]])
# plt.plot(ms_prezzo_scaled.loc[net_trend_10.index[1500:]])
plt.scatter(crossing_down_times, ma_prezzo_scaled.loc[crossing_down_times], color = 'green', marker='^')
plt.scatter(crossing_up_times, ma_prezzo_scaled.loc[crossing_up_times], color = 'red', marker='v')
plt.title('Trend detected')


# CERCO UNA SINUSOIDE OTTIMALE DA SOTTRARRE AL SEGNALE IN MODO DA LIVELLARE MINIMI E MASSIMI poi volendo escludere piu possibile gli altri valori posso usare filtro
# moltiplicando per modulo sin(x)


# Ampiezze del segnale piu comuni

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.fftpack import fft, fftfreq

SAMPLE_RATE = 15
DURATION = len(test_net_trend)/20 # dividere per 20 da segnale in ore dato che ogni punto sono 3m

# In pratica sto prendendo 6 punti l'ora come frequenza base. Il plot dira che

N = int(SAMPLE_RATE * DURATION)

spectrum = fft(test_net_trend.values)
freqs = fftfreq(len(test_net_trend.values), 1 / SAMPLE_RATE)
amp = abs(spectrum)/len(test_net_trend.values)
relevant_amp = amp[amp>0.01]
relevant_freq = freqs[np.where(amp > 0.03)]
relevant_freq = relevant_freq[relevant_freq>0]

sorted_amp = sorted(list(set(amp)))

plt.figure()
plt.plot(freqs, amp)

main_freq = abs(freqs[np.argmax(amp==sorted_amp[-2])])
main_freqs = [abs(freqs[np.argmax(amp==sorted_amp[-i])]) for i in reversed(range(2,5))]

time = [i for i in range(len(test_net_trend))]
time = np.asarray(time)

reference_sin_wave = relevant_amp[0]/2*np.sin(main_freqs[0]*time)+1/2
reference_sin_wave = pd.Series(reference_sin_wave, index=test_net_trend.index)

reference_sin_wave_2 = relevant_amp[0]/2*np.sin(main_freqs[1]*time)+1/2
reference_sin_wave_2 = pd.Series(reference_sin_wave_2, index=test_net_trend.index)

reference_sin_wave_3 = relevant_amp[0]/2*np.sin(main_freqs[2]*time)+1/2
reference_sin_wave_3 = pd.Series(reference_sin_wave_3, index=test_net_trend.index)

# custom_sin_wave = 0.8*1/2*np.sin(main_freqs[0]*(time-20000)+np.sin(main_freqs[2]*(time-15000))/2)
custom_sin_wave = 0.8*1/2*np.sin(0.00015*(time-20000)+np.sin(0.00015*(time-15000))/2)
custom_sin_wave = pd.Series(custom_sin_wave, index=test_net_trend.index)

plt.figure()
plt.plot(reference_sin_wave, label = '0.001', linestyle='--')
plt.plot(reference_sin_wave_2, label = '0.0005',  linestyle='--')
# plt.plot(reference_sin_wave_3, label = '0.0003',  linestyle='--')
plt.plot(test_net_trend)
plt.plot(ma_prezzo_scaled.loc[net_trend_10.index[1500:]])
plt.plot(custom_sin_wave)
# plt.legend()

from sklearn.metrics import mean_squared_error
mse_avg = np.zeros(len(test_net_trend))
ma_trend = test_net_trend.ewm(span = 12000).mean()

plt.figure()
plt.plot(ma_trend)
plt.plot(test_net_trend)

# for i in range(len(test_net_trend)):
#     mse_avg[i] = (0.5*(test_net_trend[i]-reference_sin_wave[i])**4 + 0.5*(test_net_trend[i]-reference_sin_wave_3[i])**4)/2
#     if test_net_trend[i]-ma_trend[i]>0.1:
#         mse_avg[i] *=-1.5
#     elif test_net_trend[i]-ma_trend[i]<-0.1:
#         mse_avg[i] *=1.5
#
#     else:
#         mse_avg[i] *= 0.5

for i in range(len(test_net_trend)):
    mse_avg[i] = (0.6*(test_net_trend[i]-reference_sin_wave[i])**4 + 0.4*(test_net_trend[i]-reference_sin_wave_2[i])**4)/2
    if test_net_trend[i]>0.55:
        mse_avg[i] *=-1.5
    elif test_net_trend[i]<0.45:
        mse_avg[i] *=1.5
    else:
        mse_avg[i] *=0.8


mse_avg = pd.Series(mse_avg, index=test_net_trend.index)
ema_mse_avg_fast = mse_avg.ewm(span=100).mean()
ema_mse_avg_slow = mse_avg.ewm(span=4000).mean()

mse_avg_std = mse_avg.ewm(span = 1000).std()



mse_buy = [test_net_trend.index[i] for i in range(1,len(mse_avg)) if 0.002 < mse_avg[i] <= 0.003]
# mse_buy = [test_net_trend.index[i] for i in range(1,len(mse_avg)) if 0 < mse_avg[i] <= 0.001 and mse_avg[i-1]>0.001]

# mse_sell = [test_net_trend.index[i] for i in range(len(mse_avg)) if -0.02< mse_avg[i] < -0.01]
mse_sell = [test_net_trend.index[i] for i in range(len(mse_avg)) if -0.048< mse_avg[i] < -0.045]


# mse_buy = [mse_avg.index[i] for i in range(len(mse_avg)) if 0.002 < mse_avg[i] <= 0.003]
# mse_sell = [mse_avg.index[i] for i in range(len(mse_avg)) if -0.02< mse_avg[i] < -0.01]

plt.figure()
plt.plot(mse_avg, label = 'mse')
# plt.plot(ema_mse_avg_fast, label = 'ema fast')
# plt.plot(ema_mse_avg_slow, label = 'ema slow')
plt.plot(ma_prezzo_scaled.loc[net_trend_10.index[1500:]])
plt.scatter(mse_buy, ma_prezzo_scaled.loc[mse_buy], marker='^', color = 'green')
plt.scatter(mse_sell, ma_prezzo_scaled.loc[mse_sell], marker='v', color = 'red')
# plt.plot(mse_avg+mse_avg_std , linestyle = '-.')
# plt.plot(mse_avg-mse_avg_std , linestyle = '-.' )
plt.legend()

# METODO BASATO SUL FATTO CHE TOGLIENDO LA MEDIA MOBILE AL NET TREND OTTENGO UN SEGNALE PERIODICO BUONO PER I LUNGHI PERIODI

test_net_trend_centered = test_net_trend-ma_trend
vola_adj_trend = test_net_trend_centered +(1/(1000*(ms_prezzo.loc[net_trend_10.index[1500:]]/ma_prezzo.loc[net_trend_10.index[1500:]])))*test_net_trend_centered
test_net_trend_adj  = vola_adj_trend

test_buy = []


attivo = False
for i in range(len(test_net_trend_adj)):

    if test_net_trend_adj[i] < -0.3:
        attivo = True
    if attivo and -0.16<=test_net_trend_adj[i]<=-0.15 and custom_sin_wave[i] > -0.1:
        test_buy.append(test_net_trend_adj.index[i])
        attivo = False

test_sell = [] # sembra essere quando la derivata si inverte nella banda [0.125, 0.175] nel terzo blocco del ciclo
min_sin = False
pausa = False
for i in range(len(test_net_trend_adj)):

    if custom_sin_wave[i] < -0.1 and not min_sin and not pausa:
        min_sin = True

    if not pausa and min_sin and 0.12 < test_net_trend_adj[i] < 0.16:

        test_sell.append(test_net_trend_adj.index[i])
        min_sin = False
        pausa = True

    if pausa and custom_sin_wave[i] > 0 :
        pausa = False


plt.figure()
plt.scatter(test_buy, ma_prezzo_scaled.loc[test_buy], marker='^', color = 'green')
plt.scatter(test_sell, ma_prezzo_scaled.loc[test_sell], marker='v', color = 'red')
plt.plot(ma_prezzo_scaled.loc[net_trend_10.index[1500:]], label = 'mse')
plt.plot(test_net_trend_adj, linestyle = '--')
plt.plot(custom_sin_wave)
plt.plot(ms_prezzo.loc[net_trend_10.index[1500:]]/ma_prezzo.loc[net_trend_10.index[1500:]])

prezzo_test = ma_prezzo_scaled.loc[net_trend_10.index[1500:]]

custom_sin_wave.to_csv('Code/dati_provvisori/custom_sin.csv')
test_net_trend_adj.to_csv('Code/dati_provvisori/trend_k.csv')
prezzo_test.to_csv('Code/dati_provvisori/prezzo_test.csv')
mse_avg.to_csv('Code/dati_provvisori/mse_avg.csv') # errore tra le sinusoidi fondamentali e test net trend ( up-trends - down trends )

# # Ora ho trigger per decidere macro uptrend e downtrend, posso settare le fibonacci areas per poi calcolare energia potenziali di caduta del prezzo