import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
import cv2
import matplotlib.pyplot as plt

imgs_file = open('Experiments/snapshots.pkl','rb')
imgs = pickle.load(imgs_file)
imgs_file.close()

pixels = 128

def build_training_data(imgs, frequency = '6H', pixels = 128):

    X = np.empty((len(imgs[frequency]),pixels, pixels))
    y = np.empty((len(imgs[frequency])))

    for i,v in enumerate(imgs[frequency].values()):

        X[i,:,:]=v[0]
        y[i] = v[2]
    return X,y

X,y = build_training_data(imgs, frequency='15T', pixels=pixels)

from sklearn.utils import shuffle

X,y = shuffle(X,y, random_state = 0)



train_size = int(len(X)*0.8)

x_train = X[:train_size]/255.0  # scales the data. pixel values range from 0 to 255, so this makes it range 0 to 1
x_test = X[train_size:]/255.0  # scales the data. pixel values range from 0 to 255, so this makes it range 0 to 1

y_train = y[:train_size]
y_test = y[train_size:]

encoding_size = 16

encoder_input = keras.Input(shape=(pixels, pixels, 1), name='img')
x = keras.layers.Flatten()(encoder_input)
encoder_output = keras.layers.Dense(64, activation="selu")(x)
encoder_output = keras.layers.Dense(encoding_size, activation="selu")(encoder_output)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')

decoder_input = keras.layers.Dense(encoding_size, activation="selu")(encoder_output)
decoder_input = keras.layers.Dense(64, activation="selu")(decoder_input)
x = keras.layers.Dense(pixels*pixels, activation="selu")(decoder_input)
decoder_output = keras.layers.Reshape((pixels, pixels, 1))(x)



opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')

encoded_input = Input(shape=(encoding_size,))
deco = autoencoder.layers[-4](encoded_input)
deco = autoencoder.layers[-3](deco)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
decoder = keras.Model(encoded_input, deco, name = 'decoder')


autoencoder.summary()

autoencoder.compile(opt, loss='mse')

epochs=1

for epoch in range(epochs):

    history = autoencoder.fit(x_train,
                              x_train,
                              epochs=500,
                              batch_size=32, validation_split=0.10, validation_data = (x_test, x_test)
                              )
    # autoencoder.save(f"Experiments/AE-{epoch+1}.model")
    # encoder.save(f"Experiments/E-{epoch+1}.model")

encoded_train = encoder.predict(x_train.reshape(-1, pixels, pixels, 1))
encoded_test = encoder.predict(x_test.reshape(-1, pixels, pixels, 1))

cnn_x_train = encoded_train.reshape(-1,128,128,1)
cnn_x_test = encoded_test.reshape(-1,128,128,1)

cnn_x_train = cnn_x_train.astype('float32')
cnn_x_test = cnn_x_test.astype('float32')

from tensorflow.keras.utils import to_categorical
#
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), input_shape=( 4, 4, 1), activation='relu', name='Convolution-1'))
# model.add(MaxPooling2D(name='MaxPooling2D-1'))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', name='Convolution-2'))
# model.add(MaxPooling2D(name='MaxPooling2D-2'))
model.add(Flatten())
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(3, activation='softmax', name='Output'))

model.summary()

EPOCHS = 1
BATCH_SIZE = 32
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(cnn_x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)

loss, accuracy = model.evaluate(cnn_x_test, y_test)
print('loss = {}, accuracy = {}'.format(loss, accuracy))

plt.figure()
plt.title('Epoch-Accuracy Graph')
plt.xlabel = 'Epochs'
plt.ylabel = 'Loss'
plt.plot(range(1, len(history.epoch) + 1), history.history['accuracy'])
plt.plot(range(1, len(history.epoch) + 1), history.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])

predictions = model.predict(cnn_x_test)