import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    image = cv2.imread('example.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("1", gray)
    cv2.waitKey(0)


def plot_histogram(image):
    plt.hist(image, bins='auto')
    plt.show()


def threshold(image, edge):
    new_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i][j] = 0 if image[i][j] < edge else 255
    return new_image


def linear_contrasting(image, min=None, max=None, foil=0, reverse=False):
    if max is None: max = np.max(image)
    if min is None: min = np.min(image)
    a = 255 / (max - min)
    b = -255 * min / (max - min)
    if reverse:
        a = -a
    return __contrasting(image, min, max, foil, a, b)


def saw_contrasting(image, a, reverse=False):
    if reverse:
        a = -a
    return __contrasting(image, a=a)


def __contrasting(image, min=0, max=255, foil=0, a=1, b=0):
    new_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i][j] = int(round(image[i][j] * a + b) % 255) if (min <= image[i][j] <= max) else foil
    return new_image


main()
