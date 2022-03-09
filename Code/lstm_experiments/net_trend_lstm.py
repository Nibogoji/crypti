import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Flatten, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, Sequential

"""
Per prevedere trading trand netto futuro ---> 10 steps = 30 minuti
Da usare come future input per predirre prezzo encoder decoder
"""


mse_avg = pd.read_csv('Code/dati_provvisori/mse_avg.csv', index_col=0) # errore tra le sinusoidi fondamentali e test net trend ( up-trends - down trends )
mse_avg.index = pd.to_datetime(mse_avg.index)

price_energy = pd.read_csv('Code/dati_provvisori/price_energy.csv', index_col=0)
price_energy.index = pd.to_datetime(price_energy.index)
price_energy = price_energy.iloc[:,:-3]

trades_trend = test_net_trend_adj = pd.read_csv('Code/dati_provvisori/trend_k.csv', index_col=0)
trades_trend.index = pd.to_datetime(trades_trend.index)

price_energy['trades_trend'] = trades_trend
price_energy['anomaly_trend'] = mse_avg
price_energy['target_shift_10'] = price_energy.Price.shift(-10)
price_energy['returns'] = (price_energy.target_shift_10-price_energy.Price)/price_energy.Price
price_energy['target'] = price_energy['trades_trend']

price_energy.dropna(inplace = True, axis = 0)



# plt.figure()
# plt.plot(price_energy['trades_trend'])



categorical_target = np.zeros(price_energy.shape[0])
synt_returns = np.zeros(price_energy.shape[0])
for i in range(price_energy.shape[0]):

    if price_energy.returns[i] >= 0.005:
        categorical_target[i] = 1
        synt_returns[i] = price_energy.returns[i]
    elif price_energy.returns[i] <= -0.005:
        categorical_target[i] = 2
        synt_returns[i] = price_energy.returns[i]
    else:
        categorical_target[i] = 0
        synt_returns[i] = float(0)

price_energy['cat_target'] = categorical_target.reshape(-1,)
price_energy['synt_returns'] = synt_returns.reshape(-1,)

# price_energy.to_csv('Code/dati_provvisori/df_simultaion_v1.csv')

plt.figure()
plt.plot(price_energy.trades_trend)

target = price_energy['target']
data = price_energy.iloc[:,:-5]
# data.drop('trades_trend', inplace = True, axis = 1)

print(data.shape)
size = data.shape[0]
train_size = int(size*0.6)
validation_size = int(size*0.8)


train = data.iloc[:train_size,:]
validation = data.iloc[train_size:validation_size,:]
test =  data.iloc[validation_size:,:]

scaler = StandardScaler()

scaler.fit(data.iloc[:validation_size,:])
scaled_train = scaler.transform(train)
scaled_validation = scaler.transform(validation)
scaled_test = scaler.transform(test)

scaled_data = np.concatenate([scaled_train, scaled_validation, scaled_test], axis=0)
scaled_data = pd.DataFrame(index=price_energy.index, data=scaled_data)

look_back  = 4*20

# Now we have data, future data aligned as actuals in the future, target at t+1
# Neural net is supposed to understand zero values are not important

T = look_back #training values per each sample window
D = len(scaled_data.columns)
X = []
Y = []
prediction_len = 10
t = 0

while t <= len(scaled_data) - T - prediction_len:

    x = scaled_data.iloc[t:t+T,:]
    X.append(x)
    y = target.iloc[t+T:t+T+prediction_len]
    Y.append(y)
    t += prediction_len

X = np.array(X).reshape(-1, T, D) # tensor-like shape N*T*D
Y = np.array(Y).reshape(-1,prediction_len, 1)
N = len(X) #samples


n_total_features = len(scaled_data.columns)


batch_size = 16

# First branch of the net is an lstm which finds an embedding for the past

i = Input(shape=(T,D))
x = LSTM(32, return_sequences=True)(i)
x = LSTM(32)(x)
x = Dense(prediction_len)(x)
model = Model(i,x)
model.compile(
    loss = 'mse',
    optimizer = Adam(learning_rate=0.001),
)

r = model.fit(
    X[:int(N * 0.6), :, :], Y[:int(N * 0.6),:,:],
    epochs = 50,
    validation_data = (X[int(N * 0.6):int(N * 0.8), :, :],Y[int(N * 0.6):int(N * 0.8),:,:])
)

plt.figure()
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.title('Train - Val Loss')
plt.show()

val_predictions = model.predict(X[int(N * 0.6):int(N * 0.8), :, :])
val_predictions = val_predictions.flatten()
val_target = Y[int(N * 0.6):int(N * 0.8),:,:].flatten()

test_predictions = model.predict(X[int(N * 0.8):, :, :]).flatten()
test_target = Y[int(N * 0.8):,:,:].flatten()


test_mae = mean_absolute_error(test_target, test_predictions)
test_mse = mean_squared_error(test_target, test_predictions)


plt.figure()
plt.plot(val_target, label = 'val target')
plt.plot(val_predictions, label = 'val forecast')
plt.legend()
plt.show()

plt.figure()
plt.plot(test_target, label = 'test target')
plt.plot(test_predictions, label = 'test forecast')
plt.legend()
plt.show()

print('test mae ', test_mae)
print('test mse ', test_mse)

predictions = np.concatenate([val_predictions,test_predictions], axis=0)
predictions = pd.Series(index=target.index[-len(predictions):], data= predictions)

plt.figure()
plt.plot(price_energy.trades_trend, label = 'True')
plt.plot(predictions, label = 'Predicted')
plt.legend()


