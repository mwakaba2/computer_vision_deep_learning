import timeit
from random import randint

import numpy as np

RED_MULTIPLIER = 0.2989
GREEN_MULTIPLIER = 0.5870
BLUE_MULTIPLIER = 0.1140
MULTIPLIERS = [RED_MULTIPLIER, GREEN_MULTIPLIER, BLUE_MULTIPLIER]
IMG = None
IMG_SIZE = None


def get_greyscale_single(*, img, img_size):
    '''
    Single threaded
    '''
    img_width, img_height, channels = img_size
    output_greyscale = [0] * (img_width * img_height)

    for i in range(img_height):
        for j in range(img_width):
            start = (i * img_width * 3) + (j * 3)
            idx = (i * img_width) + j
            output_greyscale[idx] = (img[start] * RED_MULTIPLIER +
                                     img[start + 1] * GREEN_MULTIPLIER +
                                     img[start + 2] * BLUE_MULTIPLIER)
    return output_greyscale


def get_greyscale_multi_thread(*, img, img_size):
    '''
    Python prevents multiple threads from
    executing simultaneously in the same program
    due to the Global Interpreter Lock (GIL)
    Numpy works around GIL
    '''
    img_width, img_height, channels = img_size
    img = np.reshape(img, (img_height, img_width, channels))
    return np.dot(img[...,:], MULTIPLIERS)


def main():
    global IMG, IMG_SIZE
    greyscale_functions = [get_greyscale_single,
                           get_greyscale_multi_thread]
    img_width, img_height, channels = (1920, 1080, 3)
    IMG_SIZE = (img_width, img_height, channels)
    IMG = [randint(0, 255) for i in range(img_width * img_height * channels)]

    for func in greyscale_functions:
        func_name = func.__name__
        SETUP_CODE = f"from __main__ import {func_name}, IMG, IMG_SIZE"
        TEST_CODE = f"{func_name}(img=IMG, img_size=IMG_SIZE)"
        times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE,
                              repeat=5, number=1)
        mean_time = np.mean(times)
        print((f"{func_name}\nrepetition: 5 times\n"
               f"average time: {mean_time:.3f} seconds\n"))


if __name__ == '__main__':
    main()