import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re
os.chdir("C:\\Users\\S49399\\PycharmProjects\\AAG")
from trader.last_version.filter_modeling_v2 import *
import itertools


def smoothing(data, armonic = 100, plot = False):

    """
    Choose armonic decomposition order between 10,50,100,200
    """

    fft_target = data
    close_fft = np.fft.fft(fft_target.to_list())
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['abs'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = fft_df['fft'].to_numpy()

    fft_feature = pd.DataFrame()

    for num_ in [10, 50, 100, 200]:
        fft_list_arm = np.copy(fft_list)
        fft_list_arm[num_:-num_] = 0
        inv_fft = np.fft.ifft(fft_list_arm)
        fft_feature[str(num_) + ' Armonics'] = np.real(inv_fft)
    if plot:

        plt.figure()
        plt.plot(target.values)
        plt.plot(fft_feature['{} Armonics'.format(armonic)].values, label = 'Smoothed')
        plt.legend()
        plt.title('Signal FFT decomposition with {} Armonics'.format(armonic))

    return fft_feature['{} Armonics'.format(armonic)]


def run_experiment(data, lookback = 80, encoder_dim = 8,
                   smooth_target = True,
                   scale_target = False,
                   smooth_features = True,
                   scale_features = True,
                   reduce_features = True,
                   past_pc = 10,
                   future_pc = 2,
                   plot = False):

    future_features = []
    for f in data.columns:

        if 'TEMP' in f and 'Kelvin' not in f:

            continue

        if 'FC' in f or 'FORECAST' in f:

            future_features.append(f)

    horizons = []

    for i,f in enumerate(future_features):
        h = int(re.search(r'\d+', f).group())
        horizons.append((h,i))

    shifted_forecast = data[future_features]
    future_data = shifted_forecast.copy()

    for (s,i) in horizons:
        future_data.iloc[:,i] = shifted_forecast.iloc[:,i].shift(s-1)

    target = data['TARGET_TTF_M1']
    data.drop(future_features,axis = 1,inplace=True)
    data.drop('TARGET_TTF_M1', axis = 1, inplace= True)

    if smooth_target:
        target = smoothing(target)

    if scale_target:

        target_scaler = MinMaxScaler()
        target = target_scaler.fit_transform(target.to_numpy().reshape(-1,1))
        target = pd.Series(target.reshape(-1,))


    # Now we have data, future data aligned as actuals in the future, target at t+1
    # Neural net is supposed to understand zero values are not important
    future_data = future_data.replace(np.nan,0)

    if scale_features:

        past_scaler = StandardScaler()
        data = past_scaler.fit_transform(data)
        data = pd.DataFrame(data)

        future_scaler = StandardScaler()
        future_data = future_scaler.fit_transform(future_data)
        future_data = pd.DataFrame(future_data)

    # Features reduction
    if reduce_features:
        scaler_past = PCA(n_components=past_pc)
        reduced = scaler_past.fit_transform(data)
        data = pd.DataFrame(reduced, columns=[str(i) for i in range(reduced.shape[1])])

        scaler_future = PCA(n_components=future_pc)
        reduced_future = scaler_future.fit_transform(future_data)
        future_data = pd.DataFrame(reduced_future, columns=[str(i) for i in range(reduced_future.shape[1])])

    # Applying smoothing to ll features

    if smooth_features:

        for c in list(data.columns):
            a = data[c]
            b = smoothing(a)
            data.loc[:,c] = b.values

        for c in list(future_data.columns):
            a = future_data[c]
            b = smoothing(a)
            future_data.loc[:,c] = b.values
    #
    # data.plot()
    # future_data.plot()
    # target.plot()

    complete_data = pd.concat([data,future_data,target], axis=1)

    T = lookback #training values per each sample window
    D_past = len(data.columns)
    X_past = []
    Y = []
    prediction_len = 10
    t = 0

    while t <= len(data) - T - prediction_len:

        x = data.iloc[t:t+T,:]
        X_past.append(x)
        y = target.iloc[t+T:t+T+prediction_len]
        Y.append(y)
        t += prediction_len

    X_past = np.array(X_past).reshape(-1, T, D_past) # tensor-like shape N*T*D
    Y = np.array(Y).reshape(-1,prediction_len)
    N = len(X_past) #samples

    D_future = len(future_data.columns)
    X_future = []
    t = 0

    while t <= len(future_data) - T - prediction_len:

        x = future_data.iloc[t+T:t+T+prediction_len]
        X_future.append(x)
        t += prediction_len

    X_future = np.array(X_future).reshape(-1, prediction_len, D_future) # tensor-like shape N*T*D

    n_total_features = len(complete_data.columns)-1
    n_past_features = len(data.columns)
    n_future_features = n_total_features - n_past_features


    # batch_size = 12
    window_len = T
    forecast_len = prediction_len
    latent_dim = encoder_dim

    # Past data embedding
    past_inputs = tf.keras.Input(
        shape=(window_len, n_past_features), name='present_inputs')
    # Encoding the past
    encoder = tf.keras.layers.LSTM(latent_dim, return_sequences=True)
    encoder_out = encoder(past_inputs)
    encoder2 = tf.keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder2(encoder_out)

    future_inputs = tf.keras.Input(
        shape=(forecast_len, n_future_features), name='future_inputs')
    # Combining future inputs with recurrent branch output
    decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True)
    x = decoder_lstm(future_inputs,
                     initial_state=[state_h, state_c])
    decoder_lstm2 = tf.keras.layers.LSTM(latent_dim, return_sequences=True)
    x = decoder_lstm2(x)

    x = tf.keras.layers.Dense(10, activation='tanh')(x)
    output = tf.keras.layers.Dense(1, activation=None)(x)

    model = tf.keras.models.Model(
        inputs=[past_inputs, future_inputs], outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanAbsoluteError()
    model.compile(loss=loss, optimizer=optimizer, metrics=["mae"])

    X_past_train = X_past[:int(N * 0.7), :, :]
    X_future_train = X_future[:int(N * 0.7), :, :]
    Y_train = Y[:int(N * 0.7),:]

    X_past_val = X_past[int(N * 0.7):int(N * 0.8), :, :]
    X_future_val = X_future[int(N * 0.7):int(N * 0.8), :, :]
    Y_val = Y[int(N * 0.7):int(N * 0.8), :]

    history = model.fit((X_past_train,X_future_train), Y_train, epochs=300,
                        validation_data= ((X_past_val,X_future_val), Y_val))

    # model.evaluate(test_windowed)


    val_predictions = model.predict((X_past_val,X_future_val))
    val_predictions = val_predictions.flatten()
    if scale_target:
        val_predictions = target_scaler.inverse_transform(val_predictions.reshape(-1,1))
    # if smooth_target:
    #     val_predictions = smoothing(pd.Series(val_predictions))
        val_target = target_scaler.inverse_transform(Y_val)
        val_target = val_target.flatten()
    else:
        val_target = Y_val.flatten()

    X_past_test = X_past[int(N * 0.8):, :, :]
    X_future_test = X_future[int(N * 0.8):, :, :]
    Y_test = Y[int(N * 0.8):,:]

    test_predictions = model.predict((X_past_test,X_future_test)).flatten()
    if scale_target:
        test_predictions = target_scaler.inverse_transform(test_predictions.reshape(-1,1))
    # if smooth_target:
    #     test_predictions = smoothing(pd.Series(test_predictions))
        test_target = target_scaler.inverse_transform(Y_test)
        test_target = test_target.flatten()
    else:
        test_target = Y_test.flatten()

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    test_mae = mean_absolute_error(test_target, test_predictions)
    test_mse = mean_squared_error(test_target, test_predictions)

    if plot:

        loss_fig = plt.figure()
        plt.plot(history.history['loss'], label = 'loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')
        plt.legend()
        plt.title('Train - Val Loss')
        plt.show()

        val_fig = plt.figure()
        plt.plot(val_target, label = 'val target')
        plt.plot(val_predictions, label = 'val forecast')
        plt.legend()
        plt.show()

        test_fig = plt.figure()
        plt.plot(test_target, label = 'test target')
        plt.plot(test_predictions, label = 'test forecast')
        plt.legend()
        plt.show()
    else:
        loss_fig, val_fig, test_fig = 0,0,0

    print('test mae ', test_mae)
    print('test mse ', test_mse)


    return loss_fig, val_fig, test_fig, test_mae


mastertable = pd.read_csv('aml_workspace/notebooks/research_topics/stefano/seq2seq/mastertable_AR_merged.csv', index_col='Timestamp')

adj_returns = mastertable['TARGET_TTF_M1_R_pp1_adj']

preselect_feat = [i[0] for i in pd.read_csv('aml_workspace/notebooks/research_topics/stefano/seq2seq/pre_selected_features_ensemble_regression.csv').values]


params = [[80,100,120,140],
          [8,16,32,64],
          [True,False],
          [True,False],
          [True,False],
          [True],
          [True,False]]

combos = list(itertools.product(*params))

results = []
params = []
best_score = 100

# for i,v in enumerate(combos):
#
data = mastertable[preselect_feat]
data = data.dropna(axis=0)

data.loc[:, 'TARGET_TTF_M1'] = adj_returns

data = data.dropna(axis=0)
#
#     print('Experiment n. {} out of {}'.format(i+1, len(combos)))
#
#     loss_fig, val_fig, test_fig, score = run_experiment(data, lookback = v[0], encoder_dim = v[1],
#                                                        smooth_target = v[2],
#                                                        scale_target = v[3],
#                                                        smooth_features = v[4],
#                                                        scale_features = v[5],
#                                                        reduce_features = v[6],
#                                                        past_pc = 10,
#                                                        future_pc = 2)
#     print('Score : ', score)
#     if score < best_score:
#         best_score = score
#
#     print('Best score : ', best_score)
#
#     results.append([score, loss_fig, val_fig, test_fig])
#     params.append(combos[i])


#
loss_fig, val_fig, test_fig, score = run_experiment(data, lookback = 100, encoder_dim = 32,
                                                   smooth_target = True,
                                                   scale_target = True,
                                                   smooth_features = False,
                                                   scale_features = True,
                                                   reduce_features = False,
                                                   past_pc = 100,
                                                   future_pc = 3,
                                                   plot=True)