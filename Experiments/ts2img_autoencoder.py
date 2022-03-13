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

pixels = 28

def build_training_data(imgs, frequency = '6H', pixels = 128):

    X = np.empty((len(imgs[frequency]),pixels, pixels))
    y = np.empty((len(imgs[frequency])))

    for i,v in enumerate(imgs[frequency].values()):

        X[i,:,:]=v[0]
        y[i] = v[2]
    return X,y

X,y = build_training_data(imgs, frequency='15T', pixels=28)

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
    autoencoder.save(f"Experiments/AE-{epoch+1}.model")
    encoder.save(f"Experiments/E-{epoch+1}.model")

n_example = 2

example = encoder.predict([ x_test[n_example].reshape(-1, pixels, pixels, 1) ])
plt.figure()
plt.imshow(example[0].reshape((int(np.sqrt(encoding_size)),int(np.sqrt(encoding_size)))), cmap="gray")

ae_out = autoencoder.predict([ x_test[n_example].reshape(-1, pixels, pixels, 1) ])
img = ae_out[0]  # predict is done on a vector, and returns a vector, even if its just 1 element, so we still need to grab the 0th

plt.figure()
plt.imshow(ae_out[0], cmap="gray")
plt.title('Decoded')

plt.figure()
plt.imshow(x_test[n_example], cmap="gray")
plt.title('Original')

"""
provare a generare un test set con solo 60/70 di ogni immagine e resto nero per vedere se prevede quale possa essere la continuazione
"""

# autoencoder = keras.models.load_model("Experiments/AE-1.model")
encoder = keras.models.load_model("Experiments/E-1.model")

from gmm_mml import GmmMml

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test.reshape(-1, pixels, pixels, 1))


clf=GmmMml()

clf.fit(encoded_imgs)

decoded_imgs = decoder.predict(clf.sample(32))

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
import seaborn as sns

n = 3  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

for c in range(0, clf.bestmu.shape[0]):
    samples = []
    for i in range(0, 11):
        samples.append(np.random.multivariate_normal(clf.bestmu[c], np.swapaxes(clf.bestcov, 0, 2)[c]))
    samples = np.array(samples)
    decoded_imgs = decoder.predict(samples)


    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

gmm_label=clf.predict(encoded_imgs)
enc_imgs_frame=pd.DataFrame(encoded_imgs)
enc_imgs_frame['gmm_label']=gmm_label
enc_imgs_frame.head()
g = sns.pairplot(enc_imgs_frame.sample(1000, random_state=1337), hue="gmm_label")
plt.show()

from sklearn.metrics import confusion_matrix

con_mat = confusion_matrix(y_test, gmm_label)

#
# import torch
# import torchvision.datasets as datasets
# from pythae.models import VAE, VAEConfig
# from pythae.trainers import BaseTrainingConfig
# from pythae.pipelines.training import TrainingPipeline
# from pythae.models.nn.benchmarks.mnist import Encoder_VAE_MNIST, Decoder_AE_MNIST
#
# # mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
#
# train_dataset = torch.from_numpy(x_train).reshape(-1, 1, 128, 128)
# eval_dataset = torch.from_numpy(x_test).reshape(-1, 1, 128, 128)
#
# config = BaseTrainingConfig( output_dir='my_model',
#     learning_rate=1e-3,
#     batch_size=100,
#     num_epochs=100,
# )
#
#
# model_config = VAEConfig(
#     input_dim=(128, 128,1),
#     latent_dim=64
# )
#
# model = VAE(
#     model_config=model_config,
#     encoder=Encoder_VAE_MNIST(model_config),
#     decoder=Decoder_AE_MNIST(model_config)
# )
#
# pipeline = TrainingPipeline(
#     training_config=config,
#     model=model
# )
#
# pipeline(
#     train_data=train_dataset,
#     eval_data=eval_dataset
# )