
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Read in image and convert to greyscale array object


def interpolate_pixels(image, plot = False):

    pixels = []
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):

            if image[j,i]!= 0:
                pixels.append((i,j))


    for c in range(len(pixels)-1):
        start_point = pixels[c]
        end_point = pixels[c+1]

        color = 255

        # Line thickness of 9 px
        thickness = 1

        # Using cv2.line() method
        # Draw a diagonal green line with thickness of 9 px
        image = cv2.line(image, start_point, end_point, color, thickness)

    if plot:

        fig = plt.figure(frameon=False)
        fig.set_size_inches(image.shape[0]/100,image.shape[1]/100)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, cmap = 'gray')

    return image

if __name__=='__main__':

    im = Image.open('Experiments/ts_img.png')
    im = np.array(im.convert('L'))
    image = interpolate_pixels(im)