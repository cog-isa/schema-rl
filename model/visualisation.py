import numpy as np
from PIL import Image


def get_colour(entity):
    ball = [255, 255, 255]
    paddle = [0, 200, 200]
    brick = [240, 0, 140]
    wall = [64, 64, 64]
    empty = [0, 0, 0]
    if entity[0] == 1.:
        return ball
    elif entity[1] == 1.:
        return paddle
    elif entity[2] == 1.:
        return wall
    elif entity[3] == 1.:
        return brick
    else:
        return empty


def transform_to_img(X, shape=(117, 94, 3), attr_num=4, log=False):
    img = np.zeros(shape, dtype='uint8')
    for i in range(shape[0] * shape[1]):
        colour = np.array(get_colour(X[i]))
        for j in range(shape[2]):
            img[i // shape[1], i % shape[1], j] = int(colour[j])

    return img  # .astype(dtype='uint8')


def img_average(X):
    T = len(X)
    imgs = np.array([transform_to_img(X[i]) for i in range(T)])

    return (imgs.sum(axis=0) // T).astype(dtype='uint8')


def save_img(X, img_name='images/img.png', log=False):
    img = Image.fromarray(X, 'RGB')
    if log:
        img.show()
    img.save(img_name)
