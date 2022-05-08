import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

imgs_file = open('Experiments/snapshots.pkl','rb')
imgs = pickle.load(imgs_file)
imgs_file.close()

simm_tr1 = imgs['60H']['2021-02-07 00:00:00'][0]

altro_simile2 = imgs['H']['2020-07-28 18:00:00'][0]
diverso1 = imgs['15T']['2019-10-29 17:30:00'][0]
simm_tr2 = imgs['H']['2020-08-02 08:00:00'][0]

diverso2 = imgs['15T']['2020-07-28 18:00:00'][0]

to_plot = [simm_tr1, simm_tr2, altro_simile2,diverso1 ,diverso2]

from scipy.signal import convolve2d
# a = [convolve2d(simm_tr1/255, i/255, mode='valid')[0][0] for i in to_plot]


for i in to_plot:
    fig = plt.figure(frameon=False)
    fig.set_size_inches(3,3)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(i, cmap = 'gray')

pixels = 24

def build_training_data(imgs, frequency = '6H', pixels = 128):

    X = np.empty((len(imgs[frequency]),pixels, pixels))
    y = np.empty((len(imgs[frequency])))

    for i,v in enumerate(imgs[frequency].values()):

        X[i,:,:]=v[0]
        y[i] = v[2]
    return X,y

X_tot, y_tot = [],[]
for f in ['60H','4H','H','15T']:
    X_f,y_f = build_training_data(imgs, frequency=f, pixels=pixels)
    X_tot.append(X_f)
    y_tot.append(y_f)

X = np.concatenate((X_tot[0], X_tot[1], X_tot[2], X_tot[3]))
y = np.concatenate((y_tot[0], y_tot[1], y_tot[2], y_tot[3]))

from scipy.spatial.distance import cosine

X = X[:10000]


simm_triangles = []
for i,w in enumerate(X):
    conv = convolve2d(simm_tr1/255, w/255, mode='valid')
    cos_sim = 1 - cosine(simm_tr1.reshape(-1,)/255, w.reshape(-1,)/255)
    print(  '{}... conv : {} ... cos : {}'.format(i,conv, cos_sim))
    if  conv> 25 and cos_sim > 0.8:

        simm_triangles.append([i,conv, cos_sim])


def dacaying_interpolation(flat_img_transpose, exp_coeff = 0.1 , plot = False):

    """
    Crea exponential smoothing del colore cosi da creare aree da poter comparare, cosine similarity cosi funziona
    :param flat_img_transpose:
    :param exp_coeff:
    :param plot:
    :return:
    """

    size = np.sqrt(len(flat_img_transpose))
    new_img = np.zeros(len(flat_img_transpose))
    no_value_counter = 0
    last_value = 0
    last_was_line = False

    for i,v in enumerate(flat_img_transpose):

        if v !=0:
            last_value = v
            new_img[i] = v
            last_was_line = True
            no_value_counter = 0
        else:
            if last_was_line == True:
                no_value_counter += 1
                new_img[i] = last_value - exp_coeff*np.exp(exp_coeff*no_value_counter/size)
                last_value = new_img[i]
                last_was_line = False
            else:
                no_value_counter += 1
                new_img[i] = last_value + exp_coeff*np.exp(exp_coeff*no_value_counter/size)
                last_value = new_img[i]
                last_was_line = False

    new_img = new_img.reshape(int(size), int(size)).transpose()

    if plot:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(3, 3)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(new_img, cmap='gray')

    return new_img

# flat_img_transpose = simm_tr1.transpose().flatten()/255
# exp_coeff = 0.1
test_img = simm_tr1.transpose().flatten()/255
test_img2 = simm_tr2.transpose().flatten()/255
new_simm_tr1 = dacaying_interpolation(test_img, exp_coeff =0.1, plot =True)
new_simm_tr2 = dacaying_interpolation(test_img2, exp_coeff =0.1, plot = True)


from skimage.metrics import structural_similarity as ssim

ssim_none = ssim(new_simm_tr1, new_simm_tr1)

#
for i in simm_triangles[-12:]:
    fig = plt.figure(frameon=False)
    fig.set_size_inches(3,3)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(X[i[0]], cmap = 'gray')
